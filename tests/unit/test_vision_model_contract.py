"""The detector's class-index contract, and the single model the system loads.

Everything downstream of the model addresses classes by index, so a checkpoint
whose class *order* differs from `VisionAnalysisService.CLASS_NAMES` produces
confidently wrong output rather than an error. A 9-class checkpoint of that kind
was sitting in `data/models` alongside the real one: same architecture, same
class count, 'Erosion Gully' where the service rates 'Pothole'.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.services.vision.service import DEFAULT_MODEL_PATH, VisionAnalysisService

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The taxonomy the service is built against, as the model reports it.
SERVICE_ORDER = ["Bleeding", "Corrugation", "Crack", "Patching", "Pothole",
                 "Ravelling", "Rutting", "Shoving", "Subgrade Failure"]

#: The other 9-class checkpoint that used to sit beside the real model. It is
#: index-compatible in length and therefore loads without complaint.
EROSION_GULLY_ORDER = ["Bleeding", "Corrugation", "Crack", "Patching", "Erosion Gully",
                       "Ravelling", "Rutting", "Shoving", "Subgrade Failure"]


class FakeModel:
    def __init__(self, names):
        self.names = names


class TestVerifyModelClasses:
    def test_matching_names_are_accepted(self):
        VisionAnalysisService._verify_model_classes(FakeModel(SERVICE_ORDER), "defect.pt")

    def test_matching_dict_names_are_accepted(self):
        names = {index: name for index, name in enumerate(SERVICE_ORDER)}
        VisionAnalysisService._verify_model_classes(FakeModel(names), "defect.pt")

    def test_reordered_taxonomy_is_refused(self):
        with pytest.raises(ValueError, match="index 4"):
            VisionAnalysisService._verify_model_classes(
                FakeModel(EROSION_GULLY_ORDER), "road_defect.pt"
            )

    def test_error_names_the_expected_and_actual_class(self):
        with pytest.raises(ValueError) as excinfo:
            VisionAnalysisService._verify_model_classes(FakeModel(EROSION_GULLY_ORDER), "x.pt")
        message = str(excinfo.value)
        assert "'Pothole'" in message
        assert "'Erosion Gully'" in message

    def test_reordered_indices_are_refused(self):
        shuffled = [SERVICE_ORDER[2], SERVICE_ORDER[0], SERVICE_ORDER[1]] + SERVICE_ORDER[3:]
        with pytest.raises(ValueError):
            VisionAnalysisService._verify_model_classes(FakeModel(shuffled), "x.pt")

    def test_absent_names_warn_rather_than_refuse(self):
        """A model that does not report names cannot be checked, so it is allowed."""
        VisionAnalysisService._verify_model_classes(FakeModel(None), "x.pt")


class TestServiceRejectsMismatchedModel:
    def test_service_ends_up_with_no_model(self):
        with patch("ultralytics.YOLO", return_value=FakeModel(EROSION_GULLY_ORDER)):
            service = VisionAnalysisService(
                model_path=str(REPO_ROOT / "data" / "models" / "defect.pt"),
                enable_depth_estimation=False,
            )
        # Half-loaded would be worse than nothing: the class-order complaint is
        # the only signal that the run's output is mislabelled.
        assert service.model is None

    def test_service_keeps_a_matching_model(self):
        with patch("ultralytics.YOLO", return_value=FakeModel(SERVICE_ORDER)):
            service = VisionAnalysisService(
                model_path=str(REPO_ROOT / "data" / "models" / "defect.pt"),
                enable_depth_estimation=False,
            )
        assert service.model is not None


class TestDetectionThresholds:
    """`conf`/`iou`/`imgsz` are the service's own, not the library's.

    The video loops used to call the model with no threshold arguments, so what
    counted as a defect was decided by a dependency's defaults and was invisible
    in this codebase.
    """

    def test_defaults_are_explicit_class_attributes(self):
        assert VisionAnalysisService.DEFAULT_CONF == 0.25
        assert VisionAnalysisService.DEFAULT_IOU == 0.70
        assert VisionAnalysisService.DEFAULT_IMGSZ == 640

    def test_service_adopts_the_defaults_when_not_overridden(self):
        with patch("ultralytics.YOLO", return_value=FakeModel(SERVICE_ORDER)):
            service = VisionAnalysisService(enable_depth_estimation=False)
        assert (service.conf, service.iou, service.imgsz) == (0.25, 0.70, 640)

    def test_overrides_are_honoured(self):
        with patch("ultralytics.YOLO", return_value=FakeModel(SERVICE_ORDER)):
            service = VisionAnalysisService(enable_depth_estimation=False, conf=0.10, iou=0.45, imgsz=1280)
        assert (service.conf, service.iou, service.imgsz) == (0.10, 0.45, 1280)

    def test_thresholds_reach_the_model_call(self):
        service = VisionAnalysisService(enable_depth_estimation=False, conf=0.11, iou=0.42, imgsz=800)
        service.model = MagicMock(return_value=[])
        service._run_detection(np.zeros((4, 4, 3), dtype=np.uint8))
        _, kwargs = service.model.call_args
        assert kwargs["conf"] == 0.11
        assert kwargs["iou"] == 0.42
        assert kwargs["imgsz"] == 800


class _Scalar:
    """Stands in for a one-element tensor: the loops call `.item()` on it."""

    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class _Vector:
    def __init__(self, values):
        self._values = list(values)

    def tolist(self):
        return list(self._values)


def _fake_results(boxes):
    """A stand-in for the ultralytics result object the loops iterate."""
    results = []
    for xyxy, cls_id, conf in boxes:
        box = MagicMock()
        box.cls = [_Scalar(cls_id)]
        box.conf = [_Scalar(conf)]
        box.xyxy = [_Vector(xyxy)]
        results.append(box)
    container = MagicMock()
    container.boxes = results
    return [container]


@pytest.fixture()
def tiny_video(tmp_path):
    """A 10-frame clip, so the production loop runs end to end in a test."""
    import cv2

    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (64, 48))
    for _ in range(10):
        writer.write(np.full((48, 64, 3), 120, dtype=np.uint8))
    writer.release()
    return str(path)


#: A 1 Hz track climbing north at ~30 m/s, so measured chainage (about 270 m
#: over the clip) cannot be confused with the old `timestamp * 10.0`
#: assumption, which would have said 90 m for the same 9 seconds.
_LAT_STEP_DEG = 30.0 / 111320.0
_MOVING_GPS_POINTS = [
    {
        "time": float(second),
        "lat": 6.298762 - second * _LAT_STEP_DEG,
        "lon": -1.652243,
        "timestamp": str(second),
        "elevation": 0.0,
    }
    for second in range(10)
]


class TestProductionVideoLoop:
    """`process_video_with_depth` is the method the analysis route calls.

    It used to raise `TypeError: DefectDetection.__init__() got an unexpected
    keyword argument 'frame_idx'` on the first detection of every run, which the
    caller saw not as a crash but as a survey that found nothing. So these tests
    drive the real loop with a stubbed model rather than testing a helper.
    """

    def _service(self, boxes):
        model = MagicMock(return_value=_fake_results(boxes))
        service = VisionAnalysisService(enable_depth_estimation=False, conf=0.05)
        service.model = model
        return service

    GPS_POINTS = _MOVING_GPS_POINTS

    def test_a_detection_produces_a_detection_record(self, tiny_video):
        service = self._service([([10, 10, 30, 25], 4, 0.31)])
        result = service.process_video_with_depth(
            video_path=tiny_video,
            gps_points=self.GPS_POINTS,
            target_fps=1.0,
            estimate_dense_depth=False,
        )
        assert len(result.detections) == 10, "one detection per frame is expected"
        first = result.detections[0]
        assert first["frame_idx"] == 0
        assert first["yolo_class_id"] == 4
        assert first["defect_type"] == "Pothole"

    def test_every_frame_records_its_own_index(self, tiny_video):
        service = self._service([([10, 10, 30, 25], 2, 0.4)])
        result = service.process_video_with_depth(
            video_path=tiny_video, gps_points=self.GPS_POINTS, target_fps=1.0, estimate_dense_depth=False
        )
        assert [d["frame_idx"] for d in result.detections] == list(range(10))

    def test_detections_follow_gps_along_the_route(self, tiny_video):
        """A moving vehicle must not report every defect at the same place."""
        service = self._service([([10, 10, 30, 25], 4, 0.4)])
        result = service.process_video_with_depth(
            video_path=tiny_video, gps_points=self.GPS_POINTS, target_fps=1.0, estimate_dense_depth=False
        )
        # `center_m` is the defect's own half-extent in metres, so position is
        # carried by the GeoJSON point instead.
        positions = {tuple(d["geom"]["coordinates"]) for d in result.detections}
        assert len(positions) > 1, f"all detections collapsed onto one fix: {positions}"
        # GeoJSON orders coordinates longitude-first.
        first = result.detections[0]["geom"]["coordinates"]
        assert first == pytest.approx([self.GPS_POINTS[0]["lon"], self.GPS_POINTS[0]["lat"]], abs=1e-6)

    def test_chainage_follows_the_gps_track(self, tiny_video):
        """Chainage is distance along the track, not an assumed 10 m/s."""
        service = self._service([([10, 10, 30, 25], 4, 0.4)])
        result = service.process_video_with_depth(
            video_path=tiny_video, gps_points=self.GPS_POINTS, target_fps=1.0, estimate_dense_depth=False
        )
        chainages = [d["chainage_m"] for d in result.detections]
        assert chainages[0] == 0.0
        assert chainages == sorted(chainages), "chainage must be non-decreasing"
        assert {d["chainage_source"] for d in result.detections} == {"gps"}

        # ~270 m of measured track, against the 90 m the constant-speed
        # assumption would have produced for the same 9 seconds.
        assert chainages[-1] == pytest.approx(270.0, abs=5.0)

    def test_chainage_falls_back_to_constant_speed_and_says_so(self, tiny_video):
        service = self._service([([10, 10, 30, 25], 4, 0.4)])
        result = service.process_video_with_depth(
            video_path=tiny_video, gps_points=None, target_fps=1.0, estimate_dense_depth=False
        )
        assert [d["chainage_m"] for d in result.detections] == [float(frame * 10) for frame in range(10)]
        assert {d["chainage_source"] for d in result.detections} == {"assumed_speed"}

    def test_no_detections_yields_no_records(self, tiny_video):
        service = self._service([])
        result = service.process_video_with_depth(
            video_path=tiny_video, gps_points=self.GPS_POINTS, target_fps=1.0, estimate_dense_depth=False
        )
        assert result.detections == []


class TestSingleModelLocation:
    def test_service_default_points_at_the_shipped_weights(self):
        assert Path(DEFAULT_MODEL_PATH) == REPO_ROOT / "data" / "models" / "defect.pt"

    def test_service_default_file_exists(self):
        assert Path(DEFAULT_MODEL_PATH).is_file()

    @pytest.mark.parametrize(
        "module_path",
        ["scripts/yolo_defect_detector.py", "scripts/test_yolo_video.py"],
    )
    def test_script_defaults_point_at_an_existing_model(self, module_path):
        import ast

        source = (REPO_ROOT / module_path).read_text(encoding="utf-8")
        module = ast.parse(source)
        defaults = [
            node.value
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id in ("DEFAULT_MODEL_PATH", "DEFAULT_MODEL")
                for target in node.targets
            )
        ]
        assert defaults, f"{module_path} has no default model constant"

        # The constant is evaluated rather than pattern-matched so the check
        # covers the arithmetic that produced the wrong path, not just its text.
        import importlib

        loaded = importlib.import_module(module_path[:-3].replace("/", "."))
        constant = loaded.DEFAULT_MODEL_PATH if hasattr(loaded, "DEFAULT_MODEL_PATH") else loaded.DEFAULT_MODEL
        assert Path(constant).is_file(), f"{module_path} default model is missing: {constant}"

    def test_data_models_holds_exactly_one_checkpoint(self):
        """Two weights in one directory is how the wrong model got loaded."""
        checkpoints = sorted((REPO_ROOT / "data" / "models").glob("*.pt"))
        assert [path.name for path in checkpoints] == ["defect.pt"]
