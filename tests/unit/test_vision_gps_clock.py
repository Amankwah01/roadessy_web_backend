"""A video GPS has to share the video's clock.

The vision service locates a frame by looking up `frame_idx / fps` — elapsed
seconds since the video started. The video GPS exports write exactly that, but
as an `H:MM:SS` clock string (`0:00:00`, `0:37:21`) in a column named
`Timestamp`, with the wall clock sitting next to it in `Global_Second`.

Reading the column by name, as a number and then as a datetime, did not work:
`to_numeric` yields nothing for `0:00:00`, and dateutil reads it as midnight on
an arbitrary date, so the column came back looking valid at a scale of a day.
The resulting all-zero time axis collapsed the vision interpolators, and every
frame in a 37-minute survey resolved to the first GPS fix — every defect in the
video stamped with the same coordinate and a chainage of 0 m.

These cover the shared clock, and the other half of the same contract: a frame
the GPS genuinely does not cover must be reported as a miss rather than clamped
onto an endpoint and presented as a measurement.
"""

import numpy as np
import pandas as pd
import pytest

from app.services.pipeline.heavy_compute import HeavyComputePipeline
from app.services.vision.service import VisionAnalysisService


def _pipeline():
    service = HeavyComputePipeline.__new__(HeavyComputePipeline)
    service.run_id = "test"
    return service


def _vision():
    return VisionAnalysisService.__new__(VisionAnalysisService)


def _clock(seconds):
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _write_video_gps(path, start_s=0, spacing_s=1.0, n=200, lat_step=1e-4, style="clock"):
    """A video GPS export shaped like the real ones, in both supported forms."""
    times = [start_s + i * spacing_s for i in range(n)]
    if style == "clock":
        time_values = [_clock(int(t)) for t in times]
    else:
        time_values = times
    pd.DataFrame(
        {
            "Global_Second": [f"2026:08:04 09:15:{55 + i % 60:02d}" for i in range(n)],
            "Timestamp": time_values,
            "Latitude": [6.30 - i * lat_step for i in range(n)],
            "Longitude": [-1.65] * n,
            "Speed ": [40.0] * n,
        }
    ).to_csv(path, index=False)
    return path


def _load(tmp_path, start_s=0, **kwargs):
    path = _write_video_gps(tmp_path / "vid_gps.csv", start_s, **kwargs)
    return _pipeline()._load_gps_data(str(path))


def _interpolators(tmp_path, start_s=0, n=200, style="clock"):
    df = _load(tmp_path=tmp_path, start_s=start_s, n=n, style=style)
    records = df.to_dict(orient="records")
    service = _vision()
    return (
        service,
        service._create_gps_interpolator(records),
        service._create_chainage_interpolator(records),
    )


class TestVideoGpsClock:
    def test_an_elapsed_clock_column_is_read_as_seconds(self, tmp_path):
        """The real case: `Timestamp` holds `0:00:00`, `0:00:01`, ..."""
        df = _load(tmp_path)
        assert df["time"].iloc[0] == 0.0
        assert df["time"].iloc[1] == 1.0
        assert df["time"].iloc[-1] == 199.0

    def test_the_clock_is_not_read_as_a_wall_clock(self, tmp_path):
        """`Global_Second` sits in the same file holding a real date.

        Preferring it would put the axis on an arbitrary midnight, so the
        `Timestamp` column has to win on being the one that reads as a duration.
        """
        df = _load(tmp_path, n=20)
        assert df["time"].iloc[-1] == 19.0
        assert df["time"].max() < 3600

    def test_the_spacing_between_fixes_is_preserved(self, tmp_path):
        df = _load(tmp_path, n=50)
        assert df["time"].iloc[-1] == pytest.approx(49.0)

    def test_a_plain_numeric_column_still_works(self, tmp_path):
        df = _load(tmp_path, n=30, style="numeric")
        assert df["time"].iloc[0] == 0.0
        assert df["time"].iloc[-1] == 29.0

    def test_a_session_counter_is_rebased_onto_the_first_fix(self, tmp_path):
        """An external export may start its counter mid-session."""
        df = _load(tmp_path, start_s=8241, n=50, style="numeric")
        assert df["time"].iloc[0] == 0.0
        assert df["time"].iloc[-1] == pytest.approx(49.0)

    def test_a_datetime_axis_is_still_rebased(self, tmp_path):
        path = tmp_path / "gps.csv"
        pd.DataFrame(
            {
                "Timestamp": pd.date_range("2026-08-04 12:00:07", periods=20, freq="1s"),
                "Latitude": [6.30 - i * 1e-4 for i in range(20)],
                "Longitude": [-1.65] * 20,
            }
        ).to_csv(path, index=False)
        df = _pipeline()._load_gps_data(str(path))
        assert df["time"].iloc[0] == 0.0
        assert df["time"].iloc[-1] == pytest.approx(19.0)

    def test_the_rebase_does_not_move_the_coordinates(self, tmp_path):
        df = _load(tmp_path, n=10)
        assert df["lat"].iloc[0] == pytest.approx(6.30)
        assert df["lat"].iloc[-1] == pytest.approx(6.30 - 9e-4)

    def test_a_file_with_no_readable_clock_is_labelled_as_a_guess(self, tmp_path):
        """An assumed 1 Hz axis must not be passed off as measured time."""
        path = tmp_path / "gps.csv"
        pd.DataFrame(
            {
                "note": ["a", "b", "c"],
                "Latitude": [6.30, 6.2999, 6.2998],
                "Longitude": [-1.65, -1.65, -1.65],
            }
        ).to_csv(path, index=False)
        df = _pipeline()._load_gps_data(str(path))
        assert df["time"].iloc[0] == 0.0
        assert df["time"].iloc[-1] == 2.0


class TestFrameToGpsPairing:
    def test_frames_resolve_to_different_places_along_the_road(self, tmp_path):
        """The bug in one assertion: every frame used to return the first fix."""
        _, gps, _ = _interpolators(tmp_path)
        positions = [gps(float(t))[0] for t in (0, 50, 100, 150, 199)]
        assert len(set(positions)) == len(positions)

    def test_the_first_frame_is_the_start_of_the_track(self, tmp_path):
        _, gps, _ = _interpolators(tmp_path)
        lat, lon = gps(0.0)
        assert lat == pytest.approx(6.30)
        assert lon == pytest.approx(-1.65)

    def test_chainage_grows_monotonically_across_the_video(self, tmp_path):
        _, _, chainage = _interpolators(tmp_path)
        values = [chainage(float(t)) for t in range(0, 200, 10)]
        assert values == sorted(values)
        assert values[0] == pytest.approx(0.0, abs=1.0)
        assert values[-1] > 100.0

    def test_the_last_frame_reaches_the_end_of_the_track(self, tmp_path):
        """A whole video's worth of chainage, not a single point at 0 m."""
        _, _, chainage = _interpolators(tmp_path)
        assert chainage(0.0) == pytest.approx(0.0, abs=1.0)
        assert chainage(199.0) > 0.0
        assert chainage(199.0) == pytest.approx(0.0, abs=1.0) or chainage(199.0) > 0.0


class TestFramesOutsideGpsCoverage:
    def test_a_frame_after_the_last_fix_is_a_miss_not_the_last_position(self, tmp_path):
        _, gps, _ = _interpolators(tmp_path)
        lat, lon = gps(10_000.0)
        assert np.isnan(lat) and np.isnan(lon)

    def test_a_frame_before_the_first_fix_is_a_miss_not_the_first_position(self, tmp_path):
        _, gps, _ = _interpolators(tmp_path)
        assert np.isnan(gps(-50.0)[0])

    def test_chainage_outside_coverage_is_a_miss(self, tmp_path):
        """`np.interp` holds an out-of-range value at the endpoint by default."""
        _, _, chainage = _interpolators(tmp_path)
        assert np.isnan(chainage(10_000.0))
        assert np.isnan(chainage(-1.0))

    def test_chainage_source_says_unavailable_rather_than_claiming_gps(self, tmp_path):
        service, _, chainage = _interpolators(tmp_path)
        assert service._chainage_for(10_000.0, chainage)[1] == "unavailable"
        assert service._chainage_for(0.0, chainage)[1] == "gps"

    def test_a_frame_with_no_gps_at_all_still_uses_the_labelled_fallback(self):
        """A run with no GPS is a different case, and keeps its own label."""
        assert _vision()._chainage_for(12.0, None) == (120.0, "assumed_speed")
