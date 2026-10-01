"""Which recording the video's route is taken from.

There are three candidates and they are not interchangeable. The camera's own
GPS log is clean. The vibration logger's fix goes stale for most of a trip and
then jumps kilometres. The video's burned-in telemetry is the same camera as
the footage, so it cannot disagree with it about where the vehicle was.

The selection is for the *video* only. The inertial logger's position comes
from the GPS inside its own CSV and is never taken from here, because the
accelerometer's samples are stamped against that fix and swapping it would
silently re-time the IRI.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.services.gps.service import UniversalGPSService
from app.services.pipeline.heavy_compute import HeavyComputePipeline


def _real_video(tmp_path, name="clip.mp4"):
    """A real, decodable clip with nothing burned into it."""
    import cv2
    import numpy as np

    path = tmp_path / name
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (64, 48))
    for _ in range(5):
        writer.write(np.full((48, 64, 3), 120, dtype=np.uint8))
    writer.release()
    return str(path)


def _pipeline(**attributes):
    service = HeavyComputePipeline.__new__(HeavyComputePipeline)
    service.run_id = "test"
    for key, value in attributes.items():
        setattr(service, key, value)
    return service


class TestOverlayIsAGpsSource:
    def test_it_is_offered_alongside_the_others(self):
        for source in ("auto", "overlay", "video", "csv", "vib_csv", "gpx"):
            assert source in UniversalGPSService.GPS_SOURCES

    def test_it_is_selected_explicitly(self, tmp_path):
        expected = pd.DataFrame({"time": [0.0, 1.2], "lat": [6.68, 6.69], "lon": [-1.57, -1.58]})
        video = _real_video(tmp_path)
        with patch.object(UniversalGPSService, "load_overlay_gps", return_value=expected) as load:
            result = UniversalGPSService.load_gps_universal(video_path=video, source="overlay")
        load.assert_called_once_with(video)
        assert result.equals(expected)

    def test_it_needs_a_video_to_read(self):
        with pytest.raises(ValueError, match="only be read from a video"):
            UniversalGPSService.load_gps_universal(source="overlay")

    def test_an_unknown_source_is_refused_by_name(self):
        with pytest.raises(ValueError, match="Unsupported GPS source: satellite"):
            UniversalGPSService.load_gps_universal(video_path="v.mkv", source="satellite")

    def test_auto_does_not_silently_pick_the_overlay(self):
        """`auto` is the historical order, and should stay that way.

        The overlay is a better position record than the vibration log, but
        switching to it by default would move every existing survey's route
        without anyone asking.
        """
        with patch.object(UniversalGPSService, "load_overlay_gps") as overlay:
            with patch.object(UniversalGPSService, "load_gps_csv", return_value=pd.DataFrame()):
                UniversalGPSService.load_gps_universal(
                    video_path="v.mkv", external_file="gps.csv", source="auto"
                )
        overlay.assert_not_called()

    def test_a_video_with_no_overlay_says_so_rather_than_lying(self, tmp_path):
        # A perfectly good video that simply has nothing burned into it. It
        # decodes, so this is the case that must not be mistaken for a track.
        with pytest.raises(ValueError, match="could not be read"):
            UniversalGPSService.load_overlay_gps(_real_video(tmp_path))


class TestSourceSelectionReachesTheVideoOnly:
    def test_overlay_is_used_when_chosen(self, tmp_path):
        video = _real_video(tmp_path)
        service = _pipeline(gps_source="overlay", video_file_path=video)
        frame = pd.DataFrame({"time": [0.0, 1.2], "lat": [6.68, 6.69], "lon": [-1.57, -1.58]})
        with patch.object(UniversalGPSService, "load_overlay_gps", return_value=frame) as load:
            result = service._load_gps_data("")
        load.assert_called_once_with(video)
        assert len(result) == 2

    def test_an_unreadable_overlay_falls_back_to_the_file(self, tmp_path):
        """A missing overlay is a fallback, not a dead survey."""
        gps_file = tmp_path / "gps.csv"
        gps_file.write_text("time,lat,lon\n0,6.68,-1.57\n1,6.69,-1.58\n", encoding="utf-8")
        video = _real_video(tmp_path)
        service = _pipeline(gps_source="overlay", video_file_path=video)
        with patch.object(
            UniversalGPSService, "load_overlay_gps",
            side_effect=ValueError("no overlay on the frame"),
        ):
            result = service._load_gps_data(str(gps_file))
        assert not result.empty
        assert "lat" in result.columns

    def test_the_default_is_the_uploaded_file(self, tmp_path):
        gps_file = tmp_path / "gps.csv"
        gps_file.write_text("time,lat,lon\n0,6.68,-1.57\n1,6.69,-1.58\n", encoding="utf-8")
        service = _pipeline()
        with patch.object(UniversalGPSService, "load_overlay_gps") as overlay:
            result = service._load_gps_data(str(gps_file))
        overlay.assert_not_called()
        assert not result.empty

    def test_overlay_without_a_video_is_reported_not_guessed(self):
        service = _pipeline(gps_source="overlay", video_file_path=None)
        result = service._load_gps_data("")
        assert result.empty
        assert list(result.columns) == ["time", "lat", "lon", "Speed", "GPSAcc"]


class TestTheInertialLoggerKeepsItsOwnGps:
    """The video's route choice must not reach the accelerometer.

    The two describe the vehicle at different rates. The accelerometer's
    samples are stamped against the fix in its own file, so taking the video's
    route here would re-time the IRI without any signal that it had.
    """

    def _vibration_csv(self, tmp_path):
        path = tmp_path / "vib.csv"
        path.write_text(
            "Time,Latitude,Longitude,Speed\n"
            "0.0,6.6800,-1.5700,30\n"
            "0.1,6.6801,-1.5701,31\n"
            "0.2,6.6802,-1.5702,32\n",
            encoding="utf-8",
        )
        return path

    def test_the_vibrations_own_lat_lon_is_what_is_used(self, tmp_path):
        vib = self._vibration_csv(tmp_path)
        service = _pipeline(gps_source="overlay", video_file_path="v.mkv")
        _timestamps, _accel, _speeds, lat, lon = service._load_vibration_data(str(vib))

        assert lat is not None and len(lat) > 0
        # The values are the file's, not the overlay's.
        assert pytest.approx(6.68, abs=1e-4) == lat[0]
        assert pytest.approx(-1.57, abs=1e-4) == lon[0]

    def test_the_overlay_is_never_consulted_for_the_vibrations(self, tmp_path):
        vib = self._vibration_csv(tmp_path)
        service = _pipeline(gps_source="overlay", video_file_path="v.mkv")
        with patch.object(UniversalGPSService, "load_overlay_gps") as overlay:
            service._load_vibration_data(str(vib))
        overlay.assert_not_called()
