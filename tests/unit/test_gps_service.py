from pathlib import Path

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from app.services.gps import UniversalGPSService


@pytest.fixture()
def fake_video_path(tmp_path):
    video_path = tmp_path / "test_video.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (640, 360),
    )
    for _ in range(10):
        frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return str(video_path)


class TestUniversalGPSServiceDetectSource:
    def test_detect_gpx(self):
        assert UniversalGPSService.detect_gps_source(external_file="test.gpx") == "gpx"

    def test_detect_csv(self):
        assert UniversalGPSService.detect_gps_source(external_file="test.csv") == "csv"

    def test_detect_srt(self):
        assert UniversalGPSService.detect_gps_source(external_file="test.srt") == "srt"

    def test_detect_json(self):
        assert UniversalGPSService.detect_gps_source(external_file="test.json") == "json"

    def test_detect_unsupported(self):
        with pytest.raises(ValueError):
            UniversalGPSService.detect_gps_source(external_file="test.txt")

    def test_detect_no_source(self):
        with pytest.raises(ValueError):
            UniversalGPSService.detect_gps_source()


class TestUniversalGPSServiceLoadGpx:
    def test_load_gpx_returns_dataframe(self, tmp_path):
        gpx_content = """<?xml version="1.0" encoding="UTF-8"?>
<gpx>
  <trk>
    <trkseg>
      <trkpt lat="1.0" lon="2.0"><time>2024-01-01T00:00:00Z</time></trkpt>
      <trkpt lat="1.001" lon="2.001"><time>2024-01-01T00:00:01Z</time></trkpt>
    </trkseg>
  </trk>
</gpx>"""
        gpx_path = tmp_path / "test.gpx"
        gpx_path.write_text(gpx_content)
        df = UniversalGPSService.load_gpx(str(gpx_path))
        assert len(df) == 2
        assert set(df.columns) == {"time", "lat", "lon"}

    def test_load_gpx_empty_raises(self, tmp_path):
        gpx_path = tmp_path / "empty.gpx"
        gpx_path.write_text("<?xml version='1.0'?><gpx></gpx>")
        with pytest.raises(ValueError):
            UniversalGPSService.load_gpx(str(gpx_path))


class TestUniversalGPSServiceLoadGpsCsv:
    def test_load_vibration_csv_uses_recintvl_over_display_timestamp(self, tmp_path):
        csv_path = tmp_path / "vibration.csv"
        csv_path.write_text(
            "Timestamp,Latitude,Longitude,RecIntvl\n"
            "Tue Aug 04 07:04:44 GMT 2026,6.68,-1.57,1.0\n"
            "Tue Aug 04 07:04:45 GMT 2026,6.681,-1.571,1.0\n"
        )
        df = UniversalGPSService.load_gps_universal(
            external_file=str(csv_path), source="vib_csv"
        )
        assert df["time"].tolist() == [0.0, 1.0]
        assert df["lat"].tolist() == [6.68, 6.681]

    def test_load_gps_csv_with_standard_columns(self, tmp_path):
        csv_path = tmp_path / "gps.csv"
        csv_path.write_text("time,lat,lon\n0,1.0,2.0\n1,1.001,2.001\n")
        df = UniversalGPSService.load_gps_csv(str(csv_path))
        assert len(df) == 2
        assert list(df.columns) == ["time", "lat", "lon"]

    def test_load_gps_csv_with_alternative_names(self, tmp_path):
        csv_path = tmp_path / "gps.csv"
        csv_path.write_text("timestamp,latitude,longitude\n0,1.0,2.0\n1,1.001,2.001\n")
        df = UniversalGPSService.load_gps_csv(str(csv_path))
        assert list(df.columns) == ["time", "lat", "lon"]

    def test_load_gps_csv_empty_raises(self, tmp_path):
        csv_path = tmp_path / "gps.csv"
        csv_path.write_text("a,b\n1,2\n")
        with pytest.raises(ValueError):
            UniversalGPSService.load_gps_csv(str(csv_path))


class TestUniversalGPSServiceLoggerTimeAxis:
    """The time axis of the `vid_gps_*.csv` logger exports.

    Their header reads `Global_Second,Timestamp,...` but the values are the other
    way round: `Global_Second` holds a wall clock and `Timestamp` holds elapsed
    time. The same trip is exported twice in the dataset, once with the counter
    as plain seconds starting at 8241 and once as `0:00:00`, so neither a header
    lookup nor a numeric-only parse covers the family.
    """

    HEADER = "Global_Second,Timestamp,Latitude,Longitude,Speed\n"

    def _write(self, tmp_path, name, time_values):
        path = tmp_path / name
        rows = [
            f"2026:08:04 09:15:{55 + i:02d},{value},6.29876{i},{i * -0.001},0\n"
            for i, value in enumerate(time_values)
        ]
        path.write_text(self.HEADER + "".join(rows))
        return str(path)

    def test_swapped_header_reads_seconds_from_the_other_column(self, tmp_path):
        path = self._write(tmp_path, "swapped.csv", [8241, 8242, 8243])
        df = UniversalGPSService.load_gps_csv(path)
        assert df["time"].tolist() == [0.0, 1.0, 2.0]

    def test_session_offset_is_rebased_onto_the_first_fix(self, tmp_path):
        """A counter starting at 8241 s must not push the track off the video.

        Video time is measured from frame zero, so an unrebased 8241 s start
        leaves the entire track outside the video's timeline and every detection
        collapses onto the first fix — reported as a vehicle that never moved.
        """
        path = self._write(tmp_path, "offset.csv", [8241, 8242, 8243])
        df = UniversalGPSService.load_gps_csv(path)
        assert df["time"].min() == 0.0
        assert df["time"].max() == pytest.approx(2.0)

    def test_elapsed_clock_string_is_read_as_seconds(self, tmp_path):
        path = self._write(tmp_path, "clock.csv", ["0:00:00", "0:00:01", "0:01:30"])
        df = UniversalGPSService.load_gps_csv(path)
        assert df["time"].tolist() == [0.0, 1.0, 90.0]

    def test_wall_clock_column_is_not_mistaken_for_the_time_axis(self, tmp_path):
        """`Global_Second` parses as a datetime, so name order cannot decide this."""
        path = self._write(tmp_path, "both.csv", [8241, 8242, 8243])
        df = UniversalGPSService.load_gps_csv(path)
        # Reading the wall clock instead would still rebase to 0, but its
        # spacing is what gives the survey its shape: 1 s per fix.
        assert df["time"].diff().dropna().round(6).eq(1.0).all()

    def test_wall_clock_column_alone_is_accepted(self, tmp_path):
        path = tmp_path / "wallclock.csv"
        path.write_text(
            "Global_Second,Latitude,Longitude\n"
            "2026:08:04 09:15:55,6.1,-1.6\n"
            "2026:08:04 09:15:56,6.101,-1.601\n"
        )
        df = UniversalGPSService.load_gps_csv(str(path))
        assert df["time"].tolist() == [0.0, 1.0]

    def test_unreadable_columns_fall_back_to_one_hz(self, tmp_path):
        """A file with no clock at all still loads, on an axis it admits inventing."""
        path = tmp_path / "nocolumn.csv"
        path.write_text("Temperature,Latitude,Longitude\n30,6.1,-1.6\n31,6.101,-1.601\n")
        df = UniversalGPSService.load_gps_csv(str(path))
        assert df["time"].tolist() == [0.0, 1.0]

    def test_recintvl_accumulation_still_wins_when_requested(self, tmp_path):
        path = tmp_path / "vib.csv"
        path.write_text(
            "Timestamp,Latitude,Longitude,RecIntvl\n"
            "Tue Aug 04 07:04:44 GMT 2026,6.68,-1.57,1.0\n"
            "Tue Aug 04 07:04:45 GMT 2026,6.681,-1.571,1.0\n"
        )
        df = UniversalGPSService.load_gps_csv(str(path), prefer_recintvl=True)
        assert df["time"].tolist() == [0.0, 1.0]

    def test_gpx_epoch_times_are_rebased_like_every_other_source(self, tmp_path):
        """GPX carries absolute epoch seconds; video time does not."""
        gpx_path = tmp_path / "fix.gpx"
        gpx_path.write_text(
            '<?xml version="1.0"?><gpx><trk><trkseg>'
            '<trkpt lat="1.0" lon="2.0"><time>2026-08-04T09:15:55Z</time></trkpt>'
            '<trkpt lat="1.001" lon="2.001"><time>2026-08-04T09:15:56Z</time></trkpt>'
            "</trkseg></trk></gpx>"
        )
        df = UniversalGPSService.load_gpx(str(gpx_path))
        assert df["time"].tolist() == [0.0, 1.0]


class TestUniversalGPSServiceLoadSrtGps:
    def test_load_srt_gps_returns_dataframe(self, tmp_path):
        srt_content = """1
00:00:00,000 --> 00:00:01,000
Lat: 1.0 Lon: 2.0

2
00:00:01,000 --> 00:00:02,000
Lat: 1.001 Lon: 2.001
"""
        srt_path = tmp_path / "gps.srt"
        srt_path.write_text(srt_content)
        df = UniversalGPSService.load_srt_gps(str(srt_path))
        assert len(df) == 2
        assert list(df.columns) == ["time", "lat", "lon"]

    def test_load_srt_gps_case_insensitive(self, tmp_path):
        srt_content = """1
00:00:00,000 --> 00:00:01,000
latitude: 1.0 longitude: 2.0
"""
        srt_path = tmp_path / "gps.srt"
        srt_path.write_text(srt_content)
        df = UniversalGPSService.load_srt_gps(str(srt_path))
        assert len(df) == 1


class TestUniversalGPSServiceLoadGpsJson:
    def test_load_gps_json_list(self, tmp_path):
        json_content = '[{"time": 0, "lat": 1.0, "lon": 2.0}, {"time": 1, "lat": 1.001, "lon": 2.001}]'
        json_path = tmp_path / "gps.json"
        json_path.write_text(json_content)
        df = UniversalGPSService.load_gps_json(str(json_path))
        assert len(df) == 2
        assert list(df.columns) == ["time", "lat", "lon"]

    def test_load_gps_json_dict_with_gps_key(self, tmp_path):
        json_content = '{"gps": [{"time": 0, "lat": 1.0, "lon": 2.0}]}'
        json_path = tmp_path / "gps.json"
        json_path.write_text(json_content)
        df = UniversalGPSService.load_gps_json(str(json_path))
        assert len(df) == 1

    def test_load_gps_json_dict_with_points_key(self, tmp_path):
        json_content = '{"points": [{"time": 0, "lat": 1.0, "lon": 2.0}]}'
        json_path = tmp_path / "gps.json"
        json_path.write_text(json_content)
        df = UniversalGPSService.load_gps_json(str(json_path))
        assert len(df) == 1


class TestUniversalGPSServiceNormalize:
    def test_normalize_valid_dataframe(self):
        df = pd.DataFrame({"time": [0, 1, 2], "lat": [1.0, 1.001, 1.002], "lon": [2.0, 2.001, 2.002]})
        result = UniversalGPSService._normalize_gps_dataframe(df)
        assert list(result.columns) == ["time", "lat", "lon"]
        assert len(result) == 3

    def test_normalize_drops_nan(self):
        df = pd.DataFrame({"time": [0, 1, 2], "lat": [1.0, np.nan, 1.002], "lon": [2.0, 2.001, 2.002]})
        result = UniversalGPSService._normalize_gps_dataframe(df)
        assert len(result) == 2

    def test_normalize_drops_duplicates(self):
        df = pd.DataFrame({"time": [0, 0, 1], "lat": [1.0, 1.0, 1.001], "lon": [2.0, 2.0, 2.001]})
        result = UniversalGPSService._normalize_gps_dataframe(df)
        assert len(result) == 2

    def test_normalize_empty_raises(self):
        df = pd.DataFrame(columns=["time", "lat", "lon"])
        with pytest.raises(ValueError):
            UniversalGPSService._normalize_gps_dataframe(df)

    def test_normalize_missing_columns_raises(self):
        df = pd.DataFrame({"time": [0], "lat": [1.0]})
        with pytest.raises(ValueError):
            UniversalGPSService._normalize_gps_dataframe(df)


class TestUniversalGPSServiceSyncFramesToGps:
    def test_sync_frames_to_gps_interpolates(self):
        gps = pd.DataFrame({
            "time": [0.0, 1.0, 2.0],
            "lat": [1.0, 1.001, 1.002],
            "lon": [2.0, 2.001, 2.002],
        })
        frames = [
            {"frame": 0, "time": 0.0, "frame_data": np.zeros((10, 10), dtype=np.uint8)},
            {"frame": 1, "time": 0.5, "frame_data": np.zeros((10, 10), dtype=np.uint8)},
            {"frame": 2, "time": 1.5, "frame_data": np.zeros((10, 10), dtype=np.uint8)},
        ]
        result = UniversalGPSService.sync_frames_to_gps(frames, gps)
        assert len(result) == 3
        assert result[1]["lat"] == pytest.approx(1.0005)
        assert result[1]["lon"] == pytest.approx(2.0005)

    def test_sync_frames_to_gps_empty_gps(self):
        gps = pd.DataFrame(columns=["time", "lat", "lon"])
        frames = [{"frame": 0, "time": 0.0, "frame_data": np.zeros((10, 10), dtype=np.uint8)}]
        result = UniversalGPSService.sync_frames_to_gps(frames, gps)
        assert result == []


class TestUniversalGPSServiceExtractFrames:
    def test_extract_frames_returns_list(self, fake_video_path):
        frames = UniversalGPSService.extract_frames(fake_video_path)
        assert isinstance(frames, list)
        assert len(frames) > 0
        assert "frame" in frames[0]
        assert "time" in frames[0]
        assert "frame_data" in frames[0]

    def test_extract_frames_frame_indices(self, fake_video_path):
        frames = UniversalGPSService.extract_frames(fake_video_path)
        indices = [f["frame"] for f in frames]
        assert indices == list(range(len(indices)))


class TestUniversalGPSServiceDataframeToSpatialPoints:
    def test_dataframe_to_spatial_points(self):
        df = pd.DataFrame({"time": [0.0, 1.0], "lat": [1.0, 1.001], "lon": [2.0, 2.001]})
        points = UniversalGPSService.dataframe_to_spatial_points(df)
        assert len(points) == 2
        assert points[0]["lat"] == 1.0
        assert points[0]["lon"] == 2.0
        assert "elevation" in points[0]
        assert "timestamp" in points[0]
