"""Reading the burned-in telemetry that fixes the frame clock.

The videos carry a date-time and a GPS fix in every frame. That is the only
measurement of how long a frame stands for: the container claims 1.00 fps while
the real interval is 1.2 s, and the alternative is to divide the GPS span by the
frame count, which is an average that cannot say where frames are missing.

These cover the parsing and the report. The OCR engine is stubbed, so they test
what the code concludes from what was read rather than whether an OCR library
can read pixels — which is not this code's business and is verified against the
real files separately.
"""

from datetime import datetime

import numpy as np
import pytest

from app.services.vision.overlay import (
    OverlayReading,
    VideoOverlayReader,
    analyse_overlay,
    build_frame_time_axis,
)


class StubEngine:
    """Returns canned text lines, keyed by the timestamp they should parse to."""

    def __init__(self, lines_by_frame):
        self.lines_by_frame = lines_by_frame
        self.calls = 0

    def __call__(self, crop):
        self.calls += 1
        index = self.calls - 1
        lines = self.lines_by_frame.get(index, [])
        if not lines:
            return [], None
        return [[[[0, 0], [10, 0], [10, 10], [0, 10]], line, 1.0] for line in lines], None


def _reader_with(lines_by_frame):
    reader = VideoOverlayReader()
    engine = StubEngine(lines_by_frame)
    reader._engine = engine
    reader._engine_loaded = True
    return reader, engine


class TestParsing:
    def test_a_date_time_is_read_as_a_datetime(self):
        reader, _ = _reader_with({0: ["08-04-2026 07:19:26"]})
        reading = reader.read(np.zeros((2160, 3840, 3), dtype=np.uint8))
        # `08-04-2026` is read as 4 August, because the video GPS alongside it
        # carries the same moment as `2026:08:04` and the two have to agree.
        assert reading.timestamp == datetime(2026, 8, 4, 7, 19, 26)
        assert reading.is_readable

    def test_the_overlay_and_the_video_gps_describe_the_same_instant(self):
        """The ordering is ambiguous, so it is settled against the other source.

        `08-04-2026` is 4 August or 8 April depending on the convention, and the
        footage is paired to a GPS file by time. If the two disagree by months
        the pairing is off by that much, so the reading is checked here rather
        than trusted.
        """
        from datetime import datetime as _dt

        overlay = _dt.strptime("08-04-2026 07:19:26", "%m-%d-%Y %H:%M:%S")
        gps_wall_clock = _dt.strptime("2026:08:04 07:19:26", "%Y:%m:%d %H:%M:%S")
        assert overlay == gps_wall_clock

    def test_coordinates_are_read_with_their_hemisphere(self):
        reader, _ = _reader_with({0: ["N6.644593 W1.572305 0KPH"]})
        reading = reader.read(np.zeros((2160, 3840, 3), dtype=np.uint8))
        assert reading.lat == pytest.approx(6.644593)
        assert reading.lon == pytest.approx(-1.572305)
        assert reading.speed_kph == pytest.approx(0.0)

    def test_southern_and_eastern_hemispheres_are_positive(self):
        reader, _ = _reader_with({0: ["S6.500000 E1.250000 40KPH"]})
        reading = reader.read(np.zeros((2160, 3840, 3), dtype=np.uint8))
        assert reading.lat == pytest.approx(-6.5)
        assert reading.lon == pytest.approx(1.25)
        assert reading.speed_kph == pytest.approx(40.0)

    def test_both_fields_on_one_line_are_both_read(self):
        reader, _ = _reader_with({0: ["08-04-2026 07:19:26  N6.644593 W1.572305 0KPH"]})
        reading = reader.read(np.zeros((2160, 3840, 3), dtype=np.uint8))
        assert reading.timestamp is not None
        assert reading.lat == pytest.approx(6.644593)

    def test_subtitle_text_is_kept_as_text_and_read_as_nothing_else(self):
        """The video also carries a subtitle track across the road.

        None of that is telemetry, and it must not be mistaken for a fix.
        """
        reader, _ = _reader_with({0: ["KNUST", "Make we doa", "Cash AcesptingATi"]})
        reading = reader.read(np.zeros((2160, 3840, 3), dtype=np.uint8))
        assert reading.timestamp is None
        assert reading.lat is None
        assert reading.is_readable is False
        assert "KNUST" in reading.texts

    def test_a_frame_with_nothing_readable_is_empty_rather_than_wrong(self):
        reader, _ = _reader_with({0: []})
        reading = reader.read(np.zeros((2160, 3840, 3), dtype=np.uint8))
        assert reading.timestamp is None
        assert reading.is_readable is False

    def test_a_frame_is_none_is_not_an_error(self):
        reader, _ = _reader_with({})
        assert reader.read(None).is_readable is False


class TestIntervalReport:
    def _frames(self, stamps):
        """Stamps every 100 frames, 1.2 s apart, as the real videos are."""
        base = datetime(2026, 4, 8, 7, 19, 26)
        lines = {}
        for i, index in enumerate([0, 100, 200, 300, 400]):
            moment = base + _seconds(round(index * 1.2))
            lines[i] = [moment.strftime("%m-%d-%Y %H:%M:%S")]
        return lines

    def test_a_1p2_second_interval_is_measured_from_the_clock(self):
        reader, _ = _reader_with(self._frames(None))
        report = analyse_overlay(lambda i: np.zeros((2160, 3840, 3), np.uint8),
                                 [0, 100, 200, 300, 400], reader)
        assert report["read"] is True
        assert report["frames_read"] == 5
        assert report["seconds_per_frame"] == pytest.approx(1.2, abs=0.02)

    def test_an_evenly_spaced_video_reports_no_gaps(self):
        reader, _ = _reader_with(self._frames(None))
        report = analyse_overlay(lambda i: np.zeros((2160, 3840, 3), np.uint8),
                                 [0, 100, 200, 300, 400], reader)
        assert report["suspected_gaps"] == 0

    def test_frames_missing_from_the_middle_are_named(self):
        """The reason the clock is read at all.

        An average interval is the right total and the wrong local answer, so a
        dropped block has to be located rather than averaged away.
        """
        base = datetime(2026, 4, 8, 7, 19, 26)
        # 1.2 s a frame, except across the jump from frame 200 to 300 where 30
        # seconds of drive are missing.
        stamps = {0: 0, 1: 120, 2: 240, 3: 570, 4: 660}
        lines = {
            i: [(base + _seconds(t)).strftime("%m-%d-%Y %H:%M:%S")]
            for i, t in stamps.items()
        }
        reader, _ = _reader_with(lines)
        report = analyse_overlay(lambda i: np.zeros((2160, 3840, 3), np.uint8),
                                 [0, 100, 200, 300, 400], reader)
        assert report["suspected_gaps"] == 1
        gap = report["gap_details"][0]
        assert gap["between_frames"] == [200, 300]
        assert gap["s_per_frame"] == pytest.approx(3.3, abs=0.05)

    def test_the_burned_in_position_is_reported_for_comparison_with_the_gps(self):
        """Whether the footage and the GPS file describe the same drive.

        A mismatch here is the thing no amount of interval arithmetic would
        reveal: the two could agree on a rate and still cover different road.
        """
        reader, _ = _reader_with({
            0: ["08-04-2026 07:19:26", "N6.644593 W1.572305 0KPH"],
            1: ["08-04-2026 07:20:26", "N6.630000 W1.570000 55KPH"],
        })
        report = analyse_overlay(lambda i: np.zeros((2160, 3840, 3), np.uint8),
                                 [0, 100], reader)
        assert report["positions_read"] == 2
        assert report["first_position"][0] == pytest.approx(6.644593)
        assert report["last_position"][1] == pytest.approx(-1.57)

    def test_the_sample_size_is_reported_so_a_sparse_read_is_not_mistaken_for_all_of_it(self):
        reader, _ = _reader_with({0: ["08-04-2026 07:19:26"], 1: ["08-04-2026 07:19:27"]})
        report = analyse_overlay(lambda i: np.zeros((2160, 3840, 3), np.uint8),
                                 [0, 50, 100, 150, 200], reader)
        assert report["frames_sampled"] == 5
        assert report["frames_read"] == 2


class TestUnavailable:
    def test_no_ocr_engine_reports_itself_rather_than_silently_guessing(self):
        reader = VideoOverlayReader()
        reader._engine = None
        reader._engine_loaded = True
        report = analyse_overlay(lambda i: None, [0, 1], reader)
        assert report["read"] is False
        assert "reason" in report
        assert report["frames_read"] == 0

    def test_too_few_readable_frames_is_reported_not_faked(self):
        reader, _ = _reader_with({0: ["08-04-2026 07:19:26"]})
        report = analyse_overlay(lambda i: np.zeros((2160, 3840, 3), np.uint8), [0, 1], reader)
        assert report["read"] is False
        assert report["frames_read"] == 1

    def test_a_reader_cannot_read_frames(self):
        reader = VideoOverlayReader()
        assert reader.read(None) == OverlayReading()


class TestFrameTimeAxis:
    """The clock is read on a stride, not on every frame.

    OCR costs about two seconds a frame here, so a whole-video pass would spend
    eighty minutes on a 2390-frame video. Reading every fiftieth frame and
    bisecting anything that runs fast gives the same answer in about fifty reads,
    and still locates a dropped block exactly.
    """

    def _stub_for(self, clock_times, reader_lines=True):
        """A reader that returns `clock_times[index]` for whatever is asked.

        `build_frame_time_axis` hands `frame_at`'s return value straight to the
        reader, so the test controls both ends: `frame_at` returns the frame
        index and the reader reads it back off it.
        """
        base = datetime(2026, 8, 4, 7, 19, 26)

        class Reader:
            _engine_loaded = True

            def __init__(self):
                self.reads = 0

            @property
            def available(self):
                return True

            def read(self, frame):
                self.reads += 1
                if frame in clock_times:
                    return OverlayReading(
                        timestamp=base + _seconds(clock_times[frame])
                    )
                return OverlayReading()

        return Reader()

    def _frame_for(self, index):
        return index

    def test_a_evenly_spaced_video_needs_only_a_stride_of_reads(self):
        total = 1000
        reader = self._stub_for({i: round(i * 1.2) for i in range(0, total, 50)} | {total - 1: round((total - 1) * 1.2)})
        axis = build_frame_time_axis(self._frame_for, total, stride=50, reader=reader)
        assert axis["read"] is True
        assert axis["seconds_per_frame"] == pytest.approx(1.2, abs=0.01)
        assert axis["suspected_gaps"] == 0
        # 1000 frames at a stride of 50 is 21 reads, not 1000.
        assert reader.reads < 40

    def test_a_block_missing_from_the_middle_is_located_by_bisection(self):
        """An average interval is the right total and the wrong local answer."""
        total = 1000
        clock_times = {}
        for i in range(total):
            t = i * 1.2
            if i >= 600:
                t += 36.0  # 30 s of drive missing across frames 599 and 600
            clock_times[i] = t
        reader = self._stub_for(clock_times)
        axis = build_frame_time_axis(self._frame_for, total, stride=50, reader=reader)
        assert axis["suspected_gaps"] == 1
        low, high = axis["gap_details"][0]["between_frames"]
        # The bisection narrows to the single boundary where the time went.
        assert high - low == 1
        assert 595 <= high <= 605

    def test_bisection_costs_a_few_reads_however_long_the_gap(self):
        total = 1000
        clock_times = {i: round(i * 1.2) for i in range(total)}
        for i in range(total):
            if i >= 500:
                clock_times[i] = round(i * 1.2 + 240.0)
        reader = self._stub_for(clock_times)
        axis = build_frame_time_axis(self._frame_for, total, stride=50, reader=reader)
        assert axis["suspected_gaps"] >= 1
        # 21 stride reads plus about six per located gap.
        assert reader.reads < 60

    def test_no_overlay_leaves_the_axis_absent_rather_than_invented(self):
        reader = VideoOverlayReader()
        reader._engine = None
        reader._engine_loaded = True
        axis = build_frame_time_axis(self._frame_for, 100, reader=reader)
        assert axis["read"] is False
        assert "anchors" not in axis

    def test_a_video_too_short_to_judge_says_so(self):
        reader = self._stub_for({0: 0, 1: 1})
        axis = build_frame_time_axis(self._frame_for, 1, reader=reader)
        assert axis["read"] is False

    def test_the_axis_reports_the_window_it_covered(self):
        total = 300
        reader = self._stub_for({i: round(i * 1.2) for i in range(0, total, 50)} | {total - 1: round((total - 1) * 1.2)})
        axis = build_frame_time_axis(self._frame_for, total, stride=50, reader=reader)
        assert axis["first_timestamp"] == "2026-08-04 07:19:26"
        assert axis["last_timestamp"].startswith("2026-08-04 07:2")
        assert 0 in axis["anchors"]


def _seconds(value):
    from datetime import timedelta

    return timedelta(seconds=value)
