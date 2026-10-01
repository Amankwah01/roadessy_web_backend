"""Reading the drive's clock out of the video, and the option to use it.

The frame interval decides which 50 m section every defect lands in. Two
separate failures are covered here. The first is arithmetic: these videos
report 1.00 fps and were actually recorded at one frame per 1.2 s, so the
inferred interval walks a defect out of its section by the end of a trip. The
second is trust: a clock reader that is subtly wrong still returns a valid time
on every frame, so it has to be caught by something other than its own output.
"""

import math
from datetime import datetime, timedelta

import cv2
import numpy as np
import pytest

from app.services.vision.overlay import (
    FrameTimeAxis,
    OverlayReading,
    OverlayTemplateReader,
    build_overlay_clock,
)
from app.services.vision.service import VisionAnalysisService


#: A real 1.2 s capture, as the burned-in clock sees it. The clock only has
#: second resolution, so consecutive frames are one or two seconds apart and
#: never 1.2 — which is the single fact most likely to be got wrong.
def _seconds_axis(total, interval=1.2, start=0, drop_at=None, drop_seconds=0.0):
    return {
        i: start + i * interval + (drop_seconds if drop_at is not None and i >= drop_at else 0.0)
        for i in range(total)
    }


class TestFrameTimeAxis:
    def test_without_measurements_it_is_a_plain_scale(self):
        axis = FrameTimeAxis(1.2)
        assert axis.seconds_for(10) == pytest.approx(12.0)
        assert axis.is_exact is False

    def test_a_measured_frame_answers_from_the_measurement(self):
        clocks = _seconds_axis(100)
        axis = FrameTimeAxis(1.2, clocks)
        assert axis.seconds_for(50) == pytest.approx(clocks[50])
        assert axis.is_exact is True

    def test_frames_outside_what_was_read_extend_the_edge(self):
        clocks = {5: 6.0, 6: 7.2, 7: 8.4}
        axis = FrameTimeAxis(1.2, clocks)
        assert axis.seconds_for(0) == pytest.approx(0.0, abs=0.01)
        assert axis.seconds_for(9) == pytest.approx(10.8, abs=0.01)

    def test_a_whole_second_clock_does_not_look_full_of_gaps(self):
        """Steps of 1 and 2 s are ordinary at 1.2 s per frame.

        Comparing each step to the interval itself would report a gap on every
        second where the clock ticked twice, which on a 2390-frame run is
        several hundred false alarms and would drown the real ones.
        """
        axis = FrameTimeAxis(1.2, _seconds_axis(300))
        assert axis.gaps() == []

    def test_a_real_detonation_is_still_reported(self):
        axis = FrameTimeAxis(1.2, _seconds_axis(300, drop_at=150, drop_seconds=30.0))
        gaps = axis.gaps()
        assert len(gaps) == 1
        assert gaps[0]["at_frame"] == 150
        assert gaps[0]["missing_seconds"] == pytest.approx(30.0, abs=0.01)

    def test_the_report_says_whether_every_frame_was_read(self):
        clocks = _seconds_axis(50)
        partial = {i: t for i, t in clocks.items() if i % 2 == 0}
        assert FrameTimeAxis(1.2, clocks).report()["exact"] is True
        assert FrameTimeAxis(1.2, partial).report()["exact"] is False


class TestBuildOverlayClock:
    """A reader that is wrong in a way that still looks right must be refused."""

    class _Stub:
        """Stands in for the template reader, to isolate the acceptance test."""

        def __init__(self, trained=True):
            self._trained = trained

        def bootstrap(self, frames):
            return self._trained

        def read(self, frame):
            # The reader is handed a datetime here and must hand back the same
            # shape the real one does, or the code under test never sees a time.
            return OverlayReading(timestamp=frame)

    def _frames(self, clock_for):
        return lambda index: clock_for(index)

    def test_a_consistent_reader_is_accepted(self, monkeypatch):
        import app.services.vision.overlay as overlay

        clocks = {i: datetime(2026, 8, 4, 7, 19, 26) + timedelta(seconds=t)
                  for i, t in _seconds_axis(120).items()}
        monkeypatch.setattr(
            overlay, "OverlayTemplateReader", lambda *a, **k: self._Stub()
        )
        axis, report = build_overlay_clock(self._frames(lambda i: clocks[i]), 120)

        assert axis is not None
        assert report["method"] == "template"
        # The rate is the mean step. Taking the median would report 1.0 s and
        # place every defect a sixth of the run early.
        assert axis.seconds_per_frame == pytest.approx(1.2, abs=0.02)
        assert axis.seconds_for(100) == pytest.approx(120.0, abs=0.1)

    def test_a_reader_that_invents_time_is_refused(self, monkeypatch):
        import app.services.vision.overlay as overlay

        monkeypatch.setattr(
            overlay, "OverlayTemplateReader", lambda *a, **k: self._Stub()
        )
        # Every frame gets a perfectly valid time, but the steps are not the two
        # values a whole-second clock can show for a 1.2 s interval. A reader
        # like this is the one that has to be caught, because nothing about its
        # output looks wrong — it is the agreement with itself that fails.
        drifting = {
            i: datetime(2026, 8, 4, 7, 19, 26) + timedelta(seconds=round(i * 1.2) + (i % 7))
            for i in range(120)
        }
        axis, report = build_overlay_clock(self._frames(lambda i: drifting[i]), 120)

        assert axis is None
        assert report.get("method") != "template" or report.get("read") is False
        assert "not self-consistent" in (
            report.get("template_reason") or report.get("reason") or ""
        )

    def test_no_learnable_font_falls_back_to_reading_a_stride(self, monkeypatch):
        import app.services.vision.overlay as overlay

        monkeypatch.setattr(
            overlay, "OverlayTemplateReader", lambda *a, **k: self._Stub(trained=False)
        )
        # The slow path is the OCR reader, so that is what has to answer here.
        monkeypatch.setattr(
            overlay.VideoOverlayReader, "read",
            lambda self, frame: OverlayReading(timestamp=frame),
        )
        axis, report = build_overlay_clock(
            self._frames(lambda i: datetime(2026, 8, 4, 7, 19, 26) + timedelta(seconds=i * 1.2)),
            400,
            ocr_stride=100,
        )
        assert axis is not None
        assert report["method"] == "ocr_stride"
        assert axis.seconds_per_frame == pytest.approx(1.2, abs=0.01)

    def test_a_video_with_nothing_readable_yields_no_axis(self, monkeypatch):
        import app.services.vision.overlay as overlay

        monkeypatch.setattr(
            overlay, "OverlayTemplateReader", lambda *a, **k: self._Stub(trained=False)
        )
        monkeypatch.setattr(overlay.VideoOverlayReader, "available", property(lambda s: False))
        axis, report = build_overlay_clock(self._frames(lambda i: None), 50)
        assert axis is None
        assert report["read"] is False

    def test_a_single_frame_video_is_not_analysed(self):
        axis, report = build_overlay_clock(lambda i: None, 1)
        assert axis is None
        assert "fewer than two frames" in report["reason"]


class TestOverlayClockAsAnOption:
    """The option is off by default, and never overrides an explicit interval."""

    def _service(self, **kwargs):
        return VisionAnalysisService(enable_depth_estimation=False, **kwargs)

    def test_it_is_off_by_default(self):
        assert self._service().use_overlay_clock is False

    def test_off_means_the_overlay_is_never_read(self, tmp_path, monkeypatch):
        path = _tiny_video(tmp_path)
        import app.services.vision.overlay as overlay

        service = self._service()
        # The expensive part is the reading, so that is what is watched: with
        # the option off the clock is never touched at all.
        called = []
        monkeypatch.setattr(
            overlay, "build_overlay_clock", lambda *a, **k: called.append(a) or (None, {})
        )
        result = service.process_video_astm(video_path=path, run_id="r")
        assert called == []
        assert result.frame_sampling.overlay_clock == {}
        assert result.frame_sampling.frame_interval_source in {"nominal", "measured"}

    def test_on_reports_the_overlay_and_names_the_source(self, tmp_path, monkeypatch):
        path = _tiny_video(tmp_path)
        service = self._service(use_overlay_clock=True)
        clocks = {i: i * 1.2 for i in range(10)}
        axis = FrameTimeAxis(1.2, clocks, source="overlay")
        monkeypatch.setattr(
            VisionAnalysisService, "_read_overlay_clock",
            lambda self, v, t: (axis, {"read": True, "method": "template", "frames_measured": 10}),
        )
        result = service.process_video_astm(video_path=path, run_id="r")
        assert result.frame_sampling.frame_interval_source == "overlay"
        assert result.frame_sampling.seconds_per_frame == pytest.approx(1.2)
        assert result.frame_sampling.overlay_clock["method"] == "template"

    def test_an_unreadable_overlay_falls_back_and_says_why(self, tmp_path, monkeypatch):
        path = _tiny_video(tmp_path)
        service = self._service(use_overlay_clock=True)
        monkeypatch.setattr(
            VisionAnalysisService, "_read_overlay_clock",
            lambda self, v, t: (None, {"read": False, "reason": "no OCR engine available"}),
        )
        result = service.process_video_astm(video_path=path, run_id="r")
        assert result.frame_sampling.overlay_clock["read"] is False
        assert result.frame_sampling.frame_interval_source != "overlay"

    def test_a_caller_supplied_interval_outranks_the_overlay(self, tmp_path, monkeypatch):
        path = _tiny_video(tmp_path)
        service = self._service(use_overlay_clock=True)
        axis = FrameTimeAxis(1.2, {i: i * 1.2 for i in range(10)}, source="overlay")
        monkeypatch.setattr(
            VisionAnalysisService, "_read_overlay_clock", lambda self, v, t: (axis, {"read": True})
        )
        result = service.process_video_astm(
            video_path=path, run_id="r", frame_interval_s=0.5
        )
        assert result.frame_sampling.frame_interval_source == "explicit"
        assert result.frame_sampling.seconds_per_frame == pytest.approx(0.5)

    def test_the_clock_moves_the_timestamps_a_frame_is_looked_up_at(self, tmp_path, monkeypatch):
        """A dropped block is the whole reason for reading the clock.

        With frames evenly spaced, a frame after a thirty-second drop is looked
        up thirty seconds too early and lands in the wrong 50 m section. The
        measured axis is the only thing that can put it back.
        """
        path = _tiny_video(tmp_path)
        service = self._service(use_overlay_clock=True)
        # Read every frame, with 30 s of drive missing from frame 5 onward.
        clocks = _seconds_axis(10, drop_at=5, drop_seconds=30.0)
        axis = FrameTimeAxis(1.2, clocks, source="overlay")
        monkeypatch.setattr(
            VisionAnalysisService, "_read_overlay_clock", lambda self, v, t: (axis, {"read": True})
        )
        seen = []
        original = service._create_gps_interpolator

        def spy(points):
            inner = original(points)
            if inner is None:
                return inner

            def wrapped(t):
                seen.append(t)
                return inner(t)

            return wrapped

        monkeypatch.setattr(service, "_create_gps_interpolator", spy)
        service.process_video_astm(
            video_path=path,
            run_id="r",
            gps_points=[
                {"time": float(t), "lat": 6.0 + t * 1e-5, "lon": -1.5}
                for t in range(0, 200, 2)
            ],
        )
        # The last frame is asked for at its measured time, not its index times
        # the average interval.
        assert max(seen) == pytest.approx(clocks[9], abs=0.01)


class TestTemplateReaderRefusesToGuess:
    def test_a_reader_that_learned_nothing_is_unavailable(self):
        reader = OverlayTemplateReader()
        assert reader.bootstrap([]) is False
        assert reader.available is False
        assert reader.read(np.zeros((2160, 3840, 3), dtype=np.uint8)).is_readable is False

    def test_the_legality_of_a_digit_is_checked_per_position(self):
        # A month never begins with a 7 and a minute never begins with a 9, so
        # those characters are never even compared at those positions. It is
        # what keeps a misread from becoming a plausible wrong date.
        assert OverlayTemplateReader._candidates(0) == "01"
        assert OverlayTemplateReader._candidates(2) == "-"
        assert OverlayTemplateReader._candidates(10) == " "
        assert OverlayTemplateReader._candidates(14) == "012345"
        assert OverlayTemplateReader._candidates(99) == " -0123456789:"


def _tiny_video(tmp_path, frames=10):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (64, 48))
    for _ in range(frames):
        writer.write(np.full((48, 64, 3), 120, dtype=np.uint8))
    writer.release()
    return str(path)


class TestUnreadableVideoIsNotACleanSurvey:
    """A recording with no index looks exactly like a recording with no defects.

    A dash recorder writes the MP4 index at the end, so a file cut short by a
    crash or a flat battery has no index, and OpenCV opens it and returns zero
    frames. Reporting that as "no defects found" is the worst available
    failure, so the decode is repaired, and if it cannot be, that is said.
    """

    def test_a_readable_video_opens(self, tmp_path):
        service = VisionAnalysisService(enable_depth_estimation=False)
        cap, remuxed, note = service._open_video(_tiny_video(tmp_path))
        try:
            assert note["opened"] is True
            assert remuxed is None
            assert cap.isOpened()
        finally:
            cap.release()

    def test_a_file_with_no_decodable_frame_is_reported_not_assumed_clean(self, tmp_path):
        broken = tmp_path / "broken.mp4"
        broken.write_bytes(b"\x00" * 4096)  # not a container at all
        service = VisionAnalysisService(enable_depth_estimation=False)
        cap, remuxed, note = service._open_video(str(broken))
        try:
            assert note["opened"] is False
            assert remuxed is None
        finally:
            cap.release()

        result = service.process_video_astm(video_path=str(broken), run_id="r")
        assert result.video_quality.frames_analyzed == 0
        assert result.detections == []
        # The distinguishing fact: the survey produced no evidence, and says so.
        assert result.frame_sampling.overlay_clock["video_decode"]["opened"] is False
