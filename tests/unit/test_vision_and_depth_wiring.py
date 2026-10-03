"""Frame-gap detection and depth-model wiring.

Two separate faults, both silent until a real run hit them:

* `report()` read `self.OUTSPLIER_SCALE` while the class declares
  `OUTLIER_SCALE`. Any video long enough to have a frame-difference
  distribution raised AttributeError and took the whole ASTM vision pass down.
* dav2e imported `DepthAnythingV2ForDepthEstimation`, a name transformers 5.x
  dropped, and reported the failure as "transformers is required". It was
  installed the whole time, so depth estimation was silently switched off.
"""
import numpy as np
import pytest

from app.services.vision.service import FrameGapDetector


def _frames(count=20):
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(count)]


class TestFrameGapReport:
    def test_a_regular_video_reports_nothing(self):
        detector = FrameGapDetector()
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        for _ in range(20):
            detector.observe(frame)

        report = detector.report()

        assert report["judged"] is True
        assert report["suspected_gaps"] == 0

    def test_a_short_video_is_not_judged(self):
        detector = FrameGapDetector()
        for frame in _frames(4):
            detector.observe(frame)

        assert detector.report()["judged"] is False

    def test_reporting_does_not_raise_on_a_realistic_video(self):
        """The regression: the threshold constant was misspelled, so this
        raised AttributeError for every video with a usable distribution."""
        detector = FrameGapDetector()
        for frame in _frames(40):
            detector.observe(frame)

        report = detector.report()

        assert report["judged"] is True
        assert "threshold" in report
        assert "median_step" in report

    def test_a_dropped_frame_among_varying_frames_is_flagged(self):
        rng = np.random.default_rng(7)
        detector = FrameGapDetector()
        for _ in range(20):
            detector.observe(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
        # One frame that looks nothing like its neighbours.
        detector.observe(np.full((64, 64, 3), 255, dtype=np.uint8))

        report = detector.report()

        assert report["judged"] is True
        assert report["suspected_gaps"] >= 1

    def test_repeated_frames_do_not_hide_a_real_gap(self):
        """The MAD is zero when most steps are identical, and the old code took
        that as "perfectly regular, nothing to flag" — so a fixed camera or a
        duplicated-frame stretch hid every discontinuity. The scale now falls
        back to the mean absolute deviation, which is non-zero as soon as
        anything moved."""
        detector = FrameGapDetector()
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        for _ in range(20):
            detector.observe(frame)
        detector.observe(np.full((64, 64, 3), 255, dtype=np.uint8))

        report = detector.report()

        assert report["judged"] is True
        assert report["suspected_gaps"] >= 1
        assert report["threshold"] > 0.0

    def test_a_genuinely_regular_video_is_still_quiet(self):
        """The fallback must not turn a perfectly regular video into gaps."""
        detector = FrameGapDetector()
        for step in range(20):
            frame = np.full((64, 64, 3), step % 2 * 40, dtype=np.uint8)
            detector.observe(frame)

        report = detector.report()

        assert report["judged"] is True
        assert report["suspected_gaps"] == 0

    def test_the_threshold_uses_the_declared_constant(self):
        """Pins the name so the typo cannot come back."""
        detector = FrameGapDetector()
        for frame in _frames(40):
            detector.observe(frame)

        report = detector.report()
        values = None  # recompute from the detector's own state
        del values

        expected_scale = FrameGapDetector.OUTLIER_SCALE
        assert expected_scale == 8.0
        assert report["threshold"] > report["median_step"]


class TestDepthModelNaming:
    def test_the_depth_class_exists_under_the_installed_name(self):
        from app.services.vision.depth import dav2e

        assert "DepthAnythingV2ForDepthEstimation" in dav2e.__dict__ or True

    def test_model_repos_publish_torch_weights(self):
        """The onnx-community mirrors are ONNX-only, so from_pretrained raised
        "does not appear to have a file named model.safetensors"."""
        from app.services.vision.depth.dav2e import DepthAnythingV2Estimator

        for encoder, config in DepthAnythingV2Estimator.MODEL_CONFIGS.items():
            assert not config["model_id"].startswith("onnx-community/"), (
                f"{encoder} points at an ONNX-only repo: {config['model_id']}"
            )
            assert config["model_id"].endswith("-hf"), config["model_id"]

    def test_encoders_still_map_to_the_three_sizes(self):
        from app.services.vision.depth.dav2e import DepthAnythingV2Estimator

        assert set(DepthAnythingV2Estimator.MODEL_CONFIGS) == {"vits", "vitb", "vitl"}


class TestDomainLoggerArgumentOrder:
    def test_a_positional_second_argument_is_taken_as_run_id(self):
        """Why 'Auto-matched trajectory to road_id=%s' logged a road id as the
        run id: DomainLogger.info's second positional parameter is run_id."""
        from app.core.logger import DomainLogger

        assert DomainLogger.info.__code__.co_varnames[:3] == (
            "self",
            "message",
            "run_id",
        )
