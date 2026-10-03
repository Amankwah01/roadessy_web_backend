import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

# Every class this file imports was deleted in ec67283 ("refactor backend
# architecture and modularize services"), which replaced the flat
# `app/services/vision_service.py` module with the `app/services/vision/`
# package. The imports below no longer resolve, so the file could not be
# re-enabled by fixing a path.
#
# Nothing is lost by leaving it skipped. Where these tests' subjects survive,
# they are covered by files written against the current structure:
#
#   GroundPlaneHomographyEngine / Extent
#       -> app/services/vision/ground_plane.py
#          (GroundExtent, PavementHomography, homography_from_payload)
#          covered by tests/unit/test_vision_ground_plane.py
#   Severity / PavedDefectClassifier
#       -> app/services/astm/ severity ratings and PCI classification
#          covered by tests/unit/test_astm_severity_ratings.py,
#          tests/unit/test_pci_sections.py
#   DefectRatingMatrix
#       -> no equivalent; the deduct-value logic moved to
#          app/services/astm/pci.py, covered by test_astm_curves.py
#   VisionTrackingService
#       -> app/services/vision/service.py (VisionAnalysisService,
#          FrameGapDetector) covered by test_vision_tracking.py,
#          test_vision_model_contract.py, test_vision_and_depth_wiring.py
#
# Delete this file once you are satisfied the replacements hold. If a class is
# ever reintroduced, port the assertion rather than restoring the module.
pytest.skip(
    "Subjects deleted in ec67283; coverage moved to the files listed above.",
    allow_module_level=True,
)

from app.schemas.payloads import CameraCalibrationMetadata, CameraIntrinsics
from app.services.vision_service import (
    DefectRatingMatrix,
    Extent,
    GroundPlaneHomographyEngine,
    PavedDefectClassifier,
    Severity,
    VisionTrackingService,
)


@pytest.fixture()
def fake_model_path(tmp_path):
    video_path = tmp_path / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[reportAttributeAccessIssue]
    writer = cv2.VideoWriter(
        str(video_path),
        fourcc,
        10,
        (640, 360),
    )
    for _ in range(30):
        frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return str(video_path)


@pytest.fixture()
def geo_engine():
    meta = CameraCalibrationMetadata(
        camera_height_meters=1.5,
        camera_tilt_degrees=15.0,
        intrinsics=CameraIntrinsics(fx=800.0, fy=800.0, cx=320.0, cy=180.0),
    )
    return GroundPlaneHomographyEngine(meta)


class TestDefectRatingMatrix:
    def test_get_score_all_combinations(self):
        expected = {
            (Severity.SLIGHT, Extent.LOCALIZED): 1,
            (Severity.SLIGHT, Extent.INTERMITTENT): 2,
            (Severity.SLIGHT, Extent.PREVALENT): 3,
            (Severity.MODERATE, Extent.LOCALIZED): 4,
            (Severity.MODERATE, Extent.INTERMITTENT): 5,
            (Severity.MODERATE, Extent.PREVALENT): 6,
            (Severity.EXTREME, Extent.LOCALIZED): 7,
            (Severity.EXTREME, Extent.INTERMITTENT): 8,
            (Severity.EXTREME, Extent.PREVALENT): 9,
        }
        for (sev, ext), score in expected.items():
            assert DefectRatingMatrix.get_score(sev, ext) == score

    def test_rate_defect_valid(self):
        result = DefectRatingMatrix.rate_defect("slight", "localized")
        assert result == {"score": 1, "severity": "slight", "extent": "localized"}

    def test_rate_defect_invalid_severity(self):
        result = DefectRatingMatrix.rate_defect("unknown", "localized")
        assert "error" in result

    def test_rate_defect_invalid_extent(self):
        result = DefectRatingMatrix.rate_defect("slight", "unknown")
        assert "error" in result

    def test_rate_defect_case_insensitive(self):
        result = DefectRatingMatrix.rate_defect("SLIGHT", "LOCALIZED")
        assert result == {"score": 1, "severity": "slight", "extent": "localized"}


class TestPavedDefectClassifier:
    def test_defects_contains_nine_types(self):
        assert len(PavedDefectClassifier.DEFECTS) == 9
        expected = {
            "bleeding",
            "corrugation",
            "crack",
            "erosion_gully",
            "patching",
            "pothole",
            "ravelling",
            "rutting",
            "shoving",
        }
        assert set(PavedDefectClassifier.DEFECTS.keys()) == expected

    def test_classify_defect_valid(self):
        result = PavedDefectClassifier.classify_defect("crack", "moderate", "intermittent")
        assert result["defect_type"] == "crack"
        assert result["severity"] == "moderate"
        assert result["extent"] == "intermittent"
        assert result["score"] == 5

    def test_classify_defect_invalid_type(self):
        result = PavedDefectClassifier.classify_defect("unknown", "slight", "localized")
        assert result["defect_type"] == "unknown"
        assert result["score"] == 1

    def test_classify_defect_invalid_severity(self):
        result = PavedDefectClassifier.classify_defect("crack", "unknown", "localized")
        assert "error" in result


class TestGroundPlaneHomographyEngine:
    def test_get_intrinsics_shape(self, geo_engine):
        K = geo_engine.get_intrinsics()
        assert K.shape == (3, 3)

    def test_transform_point_returns_floats(self, geo_engine):
        x, y = geo_engine.transform_point(320.0, 180.0)
        assert isinstance(x, float)
        assert isinstance(y, float)

    def test_transform_point_invariance(self, geo_engine):
        x1, y1 = geo_engine.transform_point(100.0, 100.0)
        x2, y2 = geo_engine.transform_point(100.0, 100.0)
        assert x1 == pytest.approx(x2)
        assert y1 == pytest.approx(y2)


class TestVisionTrackingServiceInit:
    def test_init_builds_class_mapping(self):
        service = VisionTrackingService()
        assert isinstance(service.class_mapping, dict)
        assert len(service.class_mapping) == 9

    def test_init_missing_model_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            VisionTrackingService(model_path=str(tmp_path / "missing.pt"))

    def test_build_class_mapping_from_dict_names(self):
        service = VisionTrackingService.__new__(VisionTrackingService)
        names = {0: "Bleeding", 1: "Crack", 2: "Pothole"}
        mapping = service._build_class_mapping(names)
        assert mapping == {0: "bleeding", 1: "crack", 2: "pothole"}

    def test_build_class_mapping_from_list_names(self):
        service = VisionTrackingService.__new__(VisionTrackingService)
        names = ["Bleeding", "Crack", "Pothole"]
        mapping = service._build_class_mapping(names)
        assert mapping == {0: "bleeding", 1: "crack", 2: "pothole"}

    def test_build_class_mapping_no_valid_raises(self):
        service = VisionTrackingService.__new__(VisionTrackingService)
        names = {0: "Unknown"}
        with pytest.raises(ValueError):
            service._build_class_mapping(names)

    def test_build_class_mapping_direct_defects_keys(self):
        service = VisionTrackingService.__new__(VisionTrackingService)
        names = {0: "Crack", 1: "Rutting", 2: "Pothole"}
        mapping = service._build_class_mapping(names)
        assert mapping == {0: "crack", 1: "rutting", 2: "pothole"}


class TestVisionTrackingServiceHelpers:
    def test_prepare_gray_returns_none_for_blank(self):
        service = VisionTrackingService.__new__(VisionTrackingService)
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        assert service._prepare_gray(blank) is None

    def test_prepare_gray_returns_array_for_valid(self):
        service = VisionTrackingService.__new__(VisionTrackingService)
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        gray = service._prepare_gray(frame)
        assert gray is not None
        assert gray.shape == (360, 640)

    def test_estimate_bbox_length_and_depth_with_geo(self, geo_engine):
        service = VisionTrackingService.__new__(VisionTrackingService)
        bbox = [100.0, 100.0, 200.0, 200.0]
        result = service._estimate_bbox_length_and_depth(bbox, geo_engine)
        assert result is not None
        assert "length_mm" in result
        assert "depth_mm" in result
        assert "depth_m" in result
        assert result["length_mm"] >= 0
        assert result["depth_mm"] >= 0

    def test_estimate_bbox_length_and_depth_transform_exception(self):
        service = VisionTrackingService.__new__(VisionTrackingService)
        bad_engine = MagicMock()
        bad_engine.transform_point.side_effect = Exception("fail")
        result = service._estimate_bbox_length_and_depth([0, 0, 10, 10], bad_engine)
        assert result is None

    def test_should_store_returns_true_when_empty(self):
        service = VisionTrackingService.__new__(VisionTrackingService)
        assert service._should_store([], 0) is True

    def test_should_store_respects_max(self):
        service = VisionTrackingService.__new__(VisionTrackingService)
        obs = [MagicMock(frame_index=0) for _ in range(6)]
        assert service._should_store(obs, 10) is False

    def test_should_store_respects_stride(self):
        service = VisionTrackingService.__new__(VisionTrackingService)
        obs = [MagicMock(frame_index=0)]
        assert service._should_store(obs, 3) is False
        assert service._should_store(obs, 5) is True

    def test_bbox_pixel_area(self):
        service = VisionTrackingService.__new__(VisionTrackingService)
        assert service._bbox_pixel_area([0.0, 0.0, 10.0, 10.0]) == 100.0
        assert service._bbox_pixel_area([0.0, 0.0, 0.0, 10.0]) == 0.0


class TestVisionTrackingServicePipeline:
    def test_process_video_stream_uses_bbox_corners(self, geo_engine):
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            video_path = tmp.name
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[reportAttributeAccessIssue]
            writer = cv2.VideoWriter(
                video_path,
                fourcc,
                10,
                (640, 360),
            )
            for _ in range(30):
                frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
                writer.write(frame)
            writer.release()

        service = VisionTrackingService()

        mock_results = MagicMock()
        mock_box = MagicMock()
        mock_id = MagicMock()
        mock_id.item.return_value = 1
        mock_box.id = [mock_id]
        mock_cls = MagicMock()
        mock_cls.item.return_value = 0
        mock_box.cls = [mock_cls]
        mock_box.xyxy = np.array([[100.0, 100.0, 200.0, 200.0]])
        mock_results.boxes = [mock_box]
        mock_results.masks = MagicMock()

        with patch.object(service.model, "track", return_value=[mock_results]) as mock_track:
            registry = service.process_video_stream(video_path, geo_engine)
            assert isinstance(registry, dict)
            assert 1 in registry
            assert registry[1]["defect_type"] == "bleeding"

    def test_process_video_stream_uses_final_largest_bbox(self, geo_engine):
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            video_path = tmp.name
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[reportAttributeAccessIssue]
            writer = cv2.VideoWriter(
                video_path,
                fourcc,
                10,
                (640, 360),
            )
            for _ in range(30):
                frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
                writer.write(frame)
            writer.release()

        service = VisionTrackingService()

        mock_results_small = MagicMock()
        mock_box_small = MagicMock()
        mock_id_small = MagicMock()
        mock_id_small.item.return_value = 1
        mock_box_small.id = [mock_id_small]
        mock_cls_small = MagicMock()
        mock_cls_small.item.return_value = 0
        mock_box_small.cls = [mock_cls_small]
        mock_box_small.xyxy = np.array([[100.0, 100.0, 200.0, 200.0]])
        mock_results_small.boxes = [mock_box_small]
        mock_results_small.masks = None

        mock_results_large = MagicMock()
        mock_box_large = MagicMock()
        mock_id_large = MagicMock()
        mock_id_large.item.return_value = 1
        mock_box_large.id = [mock_id_large]
        mock_cls_large = MagicMock()
        mock_cls_large.item.return_value = 0
        mock_box_large.cls = [mock_cls_large]
        mock_box_large.xyxy = np.array([[50.0, 50.0, 300.0, 300.0]])
        mock_results_large.boxes = [mock_box_large]
        mock_results_large.masks = None

        call_count = 0

        def side_effect_fn(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [mock_results_small]
            return [mock_results_large]

        with patch.object(service.model, "track", side_effect=side_effect_fn) as mock_track:
            registry = service.process_video_stream(video_path, geo_engine)
            assert isinstance(registry, dict)
            assert 1 in registry
            assert registry[1]["defect_type"] == "bleeding"
            assert registry[1]["bbox"] == [50.0, 50.0, 300.0, 300.0]
            assert registry[1]["max_width"] > 0
            assert registry[1]["max_area"] > 0
            assert registry[1]["severity"] is not None
