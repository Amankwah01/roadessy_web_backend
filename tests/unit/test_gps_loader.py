"""Tests for the GPS CSV loader used by the YOLO video harness.

The loader earns its own tests because the logger exports are genuinely
misleading: their header says `Global_Second,Timestamp` but the values are the
other way round, so anything that trusts the header reads a wall-clock string
where it expects a number and silently ends up with no GPS at all. A silent
zero here looks exactly like "the vehicle never moved", which is a wrong answer
rather than an obvious failure.

These write the files the way the real exports look rather than tidied-up
versions of them.
"""

import csv

import pytest

from scripts.test_yolo_video import load_gps


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return str(path)


class TestLoadGps:
    def test_reads_a_normal_export(self, tmp_path):
        path = write_csv(
            tmp_path / "gps.csv",
            ["Timestamp", "Latitude", "Longitude", "Speed"],
            [[0, 5.0, -1.0, 30], [1, 5.001, -1.0, 30], [2, 5.002, -1.0, 30]],
        )
        points = load_gps(path)
        assert len(points) == 3
        assert points[0] == {"time": 0.0, "lat": 5.0, "lon": -1.0}

    def test_swapped_columns_still_yield_seconds(self, tmp_path):
        """The real header lies: `Global_Second` holds a datetime, not seconds.

        Picking by header name alone returns nothing here, so every detection
        would silently land at 0.0, 0.0.
        """
        path = write_csv(
            tmp_path / "swapped.csv",
            ["Global_Second", "Timestamp", "Latitude", "Longitude", "Speed "],
            [
                ["2026:08:04 09:15:55", 8241, 6.298761667, -1.652243333, 0],
                ["2026:08:04 09:15:56", 8242, 6.298762000, -1.652244000, 12],
                ["2026:08:04 09:15:57", 8243, 6.298762500, -1.652245000, 25],
            ],
        )
        points = load_gps(path)
        assert len(points) == 3
        # Times are rebased to start at zero so they line up with a video
        # timestamp measured from its first frame.
        assert points[0]["time"] == pytest.approx(0.0)
        assert points[-1]["time"] == pytest.approx(2.0)
        assert points[0]["lat"] == pytest.approx(6.298761667)

    def test_times_are_rebased_to_zero(self, tmp_path):
        path = write_csv(
            tmp_path / "offset.csv",
            ["Timestamp", "Latitude", "Longitude"],
            [[8241, 5.0, -1.0], [8242, 5.001, -1.0]],
        )
        points = load_gps(path)
        assert points[0]["time"] == pytest.approx(0.0)
        assert points[1]["time"] == pytest.approx(1.0)

    def test_elapsed_clock_string_is_read_as_seconds(self, tmp_path):
        """The same trip is also exported with the time as `H:MM:SS`.

        A probe that only accepts plain numbers rejects this file outright and
        reports no GPS, which is indistinguishable from a stationary vehicle.
        """
        path = write_csv(
            tmp_path / "clock.csv",
            ["Global_Second", "Timestamp", "Latitude", "Longitude", "Speed"],
            [
                ["2026:08:04 09:15:55", "0:00:00", 6.298761667, -1.652243333, 0],
                ["2026:08:04 09:15:56", "0:00:01", 6.298762000, -1.652244000, 12],
                ["2026:08:04 09:16:55", "0:01:30", 6.298762500, -1.652245000, 25],
            ],
        )
        points = load_gps(path)
        assert len(points) == 3
        assert points[0]["time"] == pytest.approx(0.0)
        assert points[-1]["time"] == pytest.approx(90.0)

    def test_numeric_and_clock_exports_of_one_trip_agree(self, tmp_path):
        """Two exports of the same drive must produce the same track.

        They differ only in how the time column is spelled, and the video is
        matched to GPS by that column, so disagreeing here means one of the two
        files silently mislocates every detection on it.
        """
        coords = [[6.298761667, -1.652243333], [6.298762000, -1.652244000], [6.298762500, -1.652245000]]
        numeric = write_csv(
            tmp_path / "numeric.csv",
            ["Global_Second", "Timestamp", "Latitude", "Longitude", "Speed"],
            [["2026:08:04 09:15:55", 8241, *coords[0], 0],
             ["2026:08:04 09:15:56", 8242, *coords[1], 12],
             ["2026:08:04 09:15:57", 8243, *coords[2], 25]],
        )
        clock = write_csv(
            tmp_path / "clock.csv",
            ["Global_Second", "Timestamp", "Latitude", "Longitude", "Speed"],
            [["2026:08:04 09:15:55", "0:00:00", *coords[0], 0],
             ["2026:08:04 09:15:56", "0:00:01", *coords[1], 12],
             ["2026:08:04 09:15:57", "0:00:02", *coords[2], 25]],
        )
        assert load_gps(numeric) == load_gps(clock)

    def test_out_of_order_rows_are_sorted(self, tmp_path):
        """A logger that buffered rows must not produce a backwards track."""
        path = write_csv(
            tmp_path / "shuffled.csv",
            ["Timestamp", "Latitude", "Longitude"],
            [[2, 5.002, -1.0], [0, 5.0, -1.0], [1, 5.001, -1.0]],
        )
        points = load_gps(path)
        assert [p["time"] for p in points] == [0.0, 1.0, 2.0]

    def test_bad_rows_are_skipped_not_fatal(self, tmp_path):
        path = write_csv(
            tmp_path / "partial.csv",
            ["Timestamp", "Latitude", "Longitude"],
            [[0, 5.0, -1.0], ["", "", ""], [2, 5.002, -1.0]],
        )
        points = load_gps(path)
        assert len(points) == 2

    def test_lowercase_column_names_are_accepted(self, tmp_path):
        path = write_csv(
            tmp_path / "lower.csv",
            ["time", "lat", "lon"],
            [[0, 5.0, -1.0], [1, 5.001, -1.0]],
        )
        assert len(load_gps(path)) == 2

    def test_file_without_coordinates_returns_nothing(self, tmp_path):
        path = write_csv(tmp_path / "no_coords.csv", ["Timestamp", "Speed"], [[0, 30], [1, 31]])
        assert load_gps(path) == []

    def test_file_without_a_usable_time_column_returns_nothing(self, tmp_path):
        """Nothing to align against, so better no points than invented ones."""
        path = write_csv(
            tmp_path / "no_time.csv",
            ["Latitude", "Longitude"],
            [[5.0, -1.0], [5.001, -1.0]],
        )
        assert load_gps(path) == []

    def test_empty_file_returns_nothing(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("", encoding="utf-8")
        assert load_gps(path) == []

    def test_utf8_bom_header_is_stripped(self, tmp_path):
        """Excel-written CSVs carry a BOM that would become part of the first name."""
        path = tmp_path / "bom.csv"
        path.write_text(
            "Timestamp,Latitude,Longitude\n0,5.0,-1.0\n1,5.001,-1.0\n", encoding="utf-8-sig"
        )
        assert len(load_gps(str(path))) == 2
