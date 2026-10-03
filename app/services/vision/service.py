"""
Vision analysis service using YOLO for pavement distress detection.
Implements ASTM D6433-18 and MDPI Sensors 2024 paper methodology.
Includes depth estimation via DepthAnythingV2, geometry-anchored scaling, and MBTP pothole depth estimation.
"""
import os
import json
import math
import subprocess
import tempfile
import numpy as np
import cv2
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from app.core.camera import (
    DEFAULT_CAMERA_CX,
    DEFAULT_CAMERA_CY,
    DEFAULT_CAMERA_FX,
    DEFAULT_CAMERA_FY,
    DEFAULT_CAMERA_HEIGHT_M,
    DEFAULT_CAMERA_TILT_DEG,
)
from app.services.vision.ground_plane import PavementHomography

#: `app/services/vision/service.py` is four levels below the repository root, so
#: the default needs four `parents` — not three, which lands on `app/` and
#: resolves to a path that does not exist. A wrong default is silent here: the
#: loader treats a missing file as "no model" and the run completes reporting
#: zero defects.
DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parents[3] / "data" / "models" / "defect.pt")

# Placeholder logger and monitor setup
class DummyLogger:
    def warning(self, msg: str, **kwargs): print(f"[WARN] {msg}")
    def error(self, msg: str, **kwargs): print(f"[ERROR] {msg}")
    def info(self, msg: str, **kwargs): print(f"[INFO] {msg}")

logger = DummyLogger()

# Try to import depth estimation modules
try:
    from app.services.vision.depth.dav2e import DepthAnythingV2Estimator
    DAV2E_AVAILABLE = True
except ImportError:
    DAV2E_AVAILABLE = False
    logger.warning("DepthAnythingV2 not available - dense depth estimation disabled")

try:
    from app.services.vision.depth.geom_anchor import GeometryAnchoredDepthScaler
    GEOM_ANCHOR_AVAILABLE = True
except ImportError:
    GEOM_ANCHOR_AVAILABLE = False
    logger.warning("GeometryAnchoredDepthScaler not available - metric depth scaling disabled")

try:
    from app.services.vision.depth.mbtp_est import MBTPAreaEstimator, MBTPEstimate
    MBTP_AVAILABLE = True
except ImportError:
    MBTP_AVAILABLE = False
    logger.warning("MBTPAreaEstimator not available - pothole depth estimation disabled")

try:
    from app.services.vision.depth.rgbsegment import RobustPotholeSegmentor
    RGBSEG_AVAILABLE = True
except ImportError:
    RGBSEG_AVAILABLE = False
    logger.warning("RobustPotholeSegmentor not available - RGB segmentation fallback disabled")

# Dataclasses
@dataclass
class VideoQualityMetrics:
    fps: float
    resolution: Tuple[int, int]
    motion_blur_score: float
    illumination_score: float
    gps_sync_quality: float
    fps_flag: str
    frames_analyzed: int
    frames_skipped: int

@dataclass
class FrameSamplingInfo:
    original_fps: float
    processed_fps: float
    sampling_method: str
    total_frames: int
    processed_frames: int
    #: Real seconds of the drive that one video frame stands for, and how that
    #: was arrived at ("measured", "explicit" or "nominal"). Carried in the
    #: result because it is what every detection's position rests on: a video
    #: reduced from 30 fps by keeping one frame per window reports 1.00 fps in
    #: its container while each frame actually covers about 1.2 seconds, and a
    #: reader has no other way to see which of the two was used.
    seconds_per_frame: float = 1.0
    frame_interval_source: str = "nominal"
    #: Which consecutive-frame steps look like dropped frames. The interval
    #: above is an average over the whole video, so this is what says where
    #: that average is least trustworthy.
    frame_gaps: Dict[str, Any] = field(default_factory=dict)
    #: What reading the burned-in clock produced, when it was asked for. Empty
    #: when the option is off or the overlay was unreadable; the reason is in
    #: `frame_gaps` either way, so a survey that quietly fell back is visible.
    overlay_clock: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DefectDetection:
    defect_type: str
    astm_distress_type: str
    ghana_distress_type: str
    yolo_class_id: int
    area: float
    width: float
    depth_mm: float
    deduct_value: float
    ghana_rating_1to9: int
    center_m: Tuple[float, float]
    confidence: float
    chainage_m: float
    bbox: Optional[List[float]] = None
    geom: Optional[Dict[str, Any]] = None
    # Where `chainage_m` came from: "gps" when it is measured distance along the
    # interpolated track, "assumed_speed" when a run had no usable GPS and the
    # value is a constant-speed assumption. Carrying it on every detection keeps
    # a fallback from being mistaken for a measurement downstream.
    chainage_source: str = "assumed_speed"
    pothole_diameter_mm: Optional[float] = None
    #: True when `width` and `area` were measured on the pavement plane through
    #: the mounting homography. False means they came from the single-depth
    #: fallback, which gives a pixel ratio rather than a size, and the severity
    #: that would have been derived from it is deliberately left unset.
    ground_sized: bool = True
    pothole_depth_mm: Optional[float] = None
    pothole_severity: Optional[str] = None
    # Which frame the detection came from. Both video loops below pass this, and
    # the crop/annotate consumers read it back off the detection dict, so it
    # has to be a real field: without it the first detection of every run raised
    # TypeError on the unexpected keyword and the whole pass returned nothing.
    # Defaults to None so hand-built detections in tests need not set it.
    frame_idx: Optional[int] = None
    # Depth estimation fields
    dense_depth_mm: Optional[float] = None
    mbtp_area_m2: Optional[float] = None
    mbtp_depth_m: Optional[float] = None
    mbtp_volume_m3: Optional[float] = None
    mbtp_confidence: Optional[float] = None
    mbtp_method: Optional[str] = None
    depth_method: Optional[str] = None
    depth_message: Optional[str] = None

@dataclass
class VisionAnalysisResult:
    detections: List[Dict[str, Any]] = field(default_factory=list)
    video_quality: Optional[VideoQualityMetrics] = None
    frame_sampling: Optional[FrameSamplingInfo] = None
    astm_distress_mapping: Dict[int, str] = field(default_factory=dict)
    ghana_distress_mapping: Dict[int, str] = field(default_factory=dict)
    # Depth estimation results
    dense_depth_map: Optional[np.ndarray] = None
    depth_estimation_enabled: bool = False
    depth_estimation_method: Optional[str] = None

# Alias for backward compatibility
ASTMVisionResult = VisionAnalysisResult


class FrameGapDetector:
    """Finds frames that jump much further than their neighbours.

    The frame interval is derived from the GPS span divided by the frame count,
    which is an *average* over the whole video. That average is the right answer
    when frames are evenly spaced, and the wrong one at the individual frame
    when a block of them is missing: if twenty seconds of the drive were dropped
    from the middle, every frame before the gap is placed later than it should
    be and every frame after it earlier, by amounts that cancel only in the
    total. Nothing downstream can see that, because the chainages look ordinary.

    A dropped block is visible in the footage as a single unusually large change
    between consecutive frames, so consecutive frames are compared at low
    resolution as the video is read. The threshold is taken from the video's own
    distribution — a median and a median absolute deviation — so a slow scan of a
    featureless road does not trip it and a fast one does not get excused by a
    few hard cuts.

    This reports rather than corrects. Reconstructing where the missing time
    belongs needs the real time of each kept frame, which a decimation does not
    record; what can be done is say that the timing is an average and name where
    it is least trustworthy.
    """

    #: Frames are compared at this size: big enough to notice the vehicle has
    #: covered ground, small enough that the cost is negligible next to
    #: inference.
    PROBE_SIZE = (64, 36)

    #: Deviations from the median, in median-absolute-deviation units, above
    #: which a step is treated as a discontinuity rather than normal driving.
    OUTLIER_SCALE = 8.0

    def __init__(self) -> None:
        self._previous: Optional[np.ndarray] = None
        self._differences: List[float] = []

    def observe(self, frame: np.ndarray) -> None:
        """Record how much this frame differs from the one before it."""
        probe = cv2.cvtColor(
            cv2.resize(frame, self.PROBE_SIZE, interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        ).astype(np.int16)
        if self._previous is not None:
            self._differences.append(float(np.mean(np.abs(probe - self._previous))))
        self._previous = probe

    def report(self) -> Dict[str, Any]:
        """How many steps look like dropped frames, and how big the worst is.

        Returns an empty report for a video too short to judge, rather than a
        confident zero: a two-frame video has no distribution to compare against.
        """
        if len(self._differences) < 8:
            return {"judged": False, "suspected_gaps": 0}

        values = np.asarray(self._differences, dtype=float)
        median = float(np.median(values))
        deviation = float(np.median(np.abs(values - median)))
        scale = deviation
        if scale <= 0.0:
            # The MAD is zero whenever most steps are identical, which a fixed
            # camera or a duplicated-frame stretch produces. Returning "no gaps"
            # there swallowed real discontinuities: a threshold built from zero
            # spread can never be exceeded, so the one frame that differed was
            # never flagged. The mean absolute deviation is non-zero whenever
            # anything moved at all and still zero for a genuinely regular video,
            # so it separates those two cases instead of collapsing them.
            scale = float(np.mean(np.abs(values - median)))

        threshold = median + self.OUTLIER_SCALE * scale
        outliers = np.where(values > threshold)[0]
        return {
            "judged": True,
            "suspected_gaps": int(len(outliers)),
            "median_step": round(median, 3),
            "threshold": round(threshold, 3),
            "largest_step": round(float(values.max()), 3),
            "gap_after_frames": [int(i + 1) for i in outliers[:20]],
        }


class VisionAnalysisService:
    """Service for pavement distress detection and PCI computation from video."""

    CLASS_NAMES = {
        0: "Bleeding",
        1: "Corrugation",
        2: "Crack",
        3: "Patching",
        4: "Pothole",
        5: "Ravelling",
        6: "Rutting",
        7: "Shoving",
        8: "Subgrade Failure",
    }

    YOLO_TO_AASHTO = {
        0: "Bleeding",
        1: "Corrugation",
        2: "Longitudinal/Transverse Cracking",
        3: "Patching",
        4: "Potholes",
        5: "Weathering and Ravelling",
        6: "Rutting",
        7: "Shoving",
        8: "Depression",
    }

    # Mapping from YOLO class to Ghana Manual 2023 distress types (matching fusion_service weights)
    YOLO_TO_GHANA = {
        0: "Bleeding/Flushing",      # Bleeding
        1: "Corrugation/Shoving",    # Corrugation
        2: "Longitudinal Cracking",  # Crack (default to longitudinal, can be refined)
        3: "Patching",               # Patching
        4: "Potholes",               # Pothole
        5: "Ravelling",              # Ravelling (Weathering and Ravelling)
        6: "Rutting",                # Rutting
        7: "Corrugation/Shoving",    # Shoving
        8: "Subgrade Failure",       # Subgrade Failure
    }

    ASTM_PAPER_CLASSES = {0, 1, 2, 3}
    OTHER_CLASSES = {4, 5, 6, 7, 8}

    @classmethod
    def _verify_model_classes(cls, model: Any, model_path: str) -> None:
        """Refuse a checkpoint whose class order is not the one we rate against.

        Every downstream stage addresses classes by index, not by name: CLASS_NAMES,
        YOLO_TO_AASHTO, the ASTM band sets and the Ghana rating table are all keyed
        on 0-8. Another 9-class checkpoint in this repository's data/models
        directory has 'Erosion Gully' at index 4 where this service expects
        'Pothole', which loads without error and then reports every pothole as a
        gully — with gully deduct values and pothole depth severity applied to it.
        A name mismatch is therefore refused rather than reported, and the check
        runs on every load so a new checkpoint cannot reintroduce it.
        """
        names = getattr(model, "names", None)
        if names is None:
            logger.warning(
                f"Model at {model_path} exposes no class names; class-order "
                "verification skipped, results are unverified."
            )
            return
        if isinstance(names, dict):
            observed = {int(key): str(value) for key, value in names.items()}
        else:
            observed = {index: str(value) for index, value in enumerate(names)}
        mismatched = {
            index: (cls.CLASS_NAMES.get(index), observed[index])
            for index in cls.CLASS_NAMES
            if observed.get(index) != cls.CLASS_NAMES[index]
        }
        if mismatched:
            detail = ", ".join(
                f"index {index}: expected {expected!r}, model has {actual!r}"
                for index, (expected, actual) in sorted(mismatched.items())
            )
            raise ValueError(
                f"Model at {model_path} does not match the service's class order "
                f"({detail}). Load the model trained on this taxonomy instead of "
                "reinterpreting class indices."
            )

    #: Detection thresholds used when a caller does not pass its own.
    #:
    #: These are stated here rather than left to the inference library's defaults
    #: because a default is not a decision: the two were identical, so the numbers
    #: that decide what counts as a defect were invisible in this file, and
    #: changing a library's defaults would have silently changed every survey.
    #: `conf` in particular is the single most consequential number in the vision
    #: pipeline, and it needs to be explicit to be calibrated.
    DEFAULT_CONF = 0.25
    DEFAULT_IOU = 0.70
    DEFAULT_IMGSZ = 640

    def __init__(
        self, 
        model_path: Optional[str] = None, 
        camera_height_m: float = DEFAULT_CAMERA_HEIGHT_M, 
        camera_tilt_deg: float = DEFAULT_CAMERA_TILT_DEG, 
        fx: float = DEFAULT_CAMERA_FX, 
        fy: float = DEFAULT_CAMERA_FY, 
        cx: float = DEFAULT_CAMERA_CX, 
        cy: float = DEFAULT_CAMERA_CY, 
        distortion_coefficients: Optional[str] = None,
        enable_depth_estimation: bool = True,
        dav2e_encoder: str = "vits",
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None,
        use_overlay_clock: bool = False,
    ):
        self.model = None
        self.model_path = model_path
        self.camera_height_m = camera_height_m
        self.camera_tilt_deg = camera_tilt_deg
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.distortion_coefficients = distortion_coefficients
        self.enable_depth_estimation = enable_depth_estimation
        self.conf = float(self.DEFAULT_CONF if conf is None else conf)
        self.iou = float(self.DEFAULT_IOU if iou is None else iou)
        self.imgsz = int(self.DEFAULT_IMGSZ if imgsz is None else imgsz)
        #: Read the drive's clock from the telemetry burned into the video
        #: instead of dividing the GPS span by the frame count. Off by default
        #: because it costs an extra pass over the footage and the measured
        #: interval is within 0.3% of it; it is the better choice when the two
        #: disagree, when frames are missing mid-run, or when a survey's GPS
        #: window does not line up with the video.
        self.use_overlay_clock = bool(use_overlay_clock)
        #: Image-to-ground mapping for the pavement plane, built from the mounting
        #: pose. None when the pose cannot reach the road (a level camera), in
        #: which case sizes fall back to the single-depth estimate and are marked
        #: ungrounded so severity is left unrated rather than invented.
        self.pavement = PavementHomography(
            height_m=self.camera_height_m,
            tilt_deg=self.camera_tilt_deg,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
        ) if self.camera_height_m and self.camera_height_m > 0 else None
        
        # Load Ultralytics YOLO if path provided
        if model_path and os.path.exists(model_path):
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                self._verify_model_classes(self.model, model_path)
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")
                self.model = None
        
        # Initialize depth estimation components
        self.dav2_estimator = None
        self.geom_scaler = None
        self.mbtp_estimator = None
        self.segmentor = None
        self.camera_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        if self.enable_depth_estimation:
            self._init_depth_estimation(dav2e_encoder)
    
    def _run_detection(self, frame: np.ndarray):
        """Run the detector with this service's configured thresholds.

        Both video loops call the model directly, so the thresholds used to be
        whatever the inference library defaulted to. Funnelling every call
        through here keeps `conf`/`iou`/`imgsz` a property of the service —
        something a caller can set and a test can assert on — instead of a
        library detail that changes with a dependency upgrade.
        """
        return self.model(frame, verbose=False, conf=self.conf, iou=self.iou, imgsz=self.imgsz)

    def _init_depth_estimation(self, encoder: str = "vits"):
        """Initialize depth estimation components."""
        # Initialize DepthAnythingV2 for dense depth estimation
        if DAV2E_AVAILABLE:
            try:
                self.dav2_estimator = DepthAnythingV2Estimator(encoder=encoder)
                logger.info(f"DepthAnythingV2 initialized with encoder={encoder}")
            except Exception as e:
                logger.warning(f"Failed to initialize DepthAnythingV2: {e}")
        
        # Initialize Geometry-Anchored Depth Scaler
        if GEOM_ANCHOR_AVAILABLE:
            try:
                self.geom_scaler = GeometryAnchoredDepthScaler(
                    camera_height_m=self.camera_height_m,
                    pitch_deg=self.camera_tilt_deg,
                    fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy
                )
                logger.info("GeometryAnchoredDepthScaler initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize GeometryAnchoredDepthScaler: {e}")
        
        # Initialize MBTP Area Estimator for pothole depth
        if MBTP_AVAILABLE:
            try:
                self.mbtp_estimator = MBTPAreaEstimator()
                logger.info("MBTPAreaEstimator initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize MBTPAreaEstimator: {e}")
        
        # Initialize RGB Segmentor for water-filled potholes
        if RGBSEG_AVAILABLE:
            try:
                self.segmentor = RobustPotholeSegmentor()
                logger.info("RobustPotholeSegmentor initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize RobustPotholeSegmentor: {e}")

    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        """Undistort frame using camera intrinsics and distortion coefficients."""
        if self.fx <= 0 or self.fy <= 0:
            return frame

        camera_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)

        dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        try:
            if self.distortion_coefficients:
                coeffs = self.distortion_coefficients
                if isinstance(coeffs, str):
                    coeffs = json.loads(coeffs)
                if isinstance(coeffs, list) and len(coeffs) >= 5:
                    dist_coeffs = np.array(coeffs[:5], dtype=np.float32).reshape(-1, 1)
        except Exception:
            pass

        h, w = frame.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

        x, y, w, h = roi
        if w > 0 and h > 0:
            undistorted = undistorted[y:y+h, x:x+w]

        return undistorted

    @staticmethod
    def _classify_pothole_severity(diameter_mm: float, depth_mm: float) -> str:
        """Classify pothole severity per ASTM D 6433-99 Table X1.1.

        The table keys on diameter *and* depth together:

        ==========================  ============  ============  ============
        Average diameter (mm)       13-25 mm      >25-50 mm     >50 mm
        ==========================  ============  ============  ============
        100 to 200                        L             L             L
        200 to 450                        L             M             M
        450 to 750                        L             M             H
        ==========================  ============  ============  ============

        Above 750 mm the standard stops using diameter (X1.17.1.2) and rates
        on depth alone: 25 mm or less is medium, more than 25 mm is high.
        """
        if diameter_mm > 750:
            return "MEDIUM" if depth_mm <= 25 else "HIGH"
        if depth_mm <= 25:
            # Every diameter band is low at this depth.
            return "LOW"
        if depth_mm <= 50:
            return "MEDIUM" if diameter_mm > 200 else "LOW"
        return "HIGH" if diameter_mm > 450 else "MEDIUM"

    def _pixel_to_real_world(self, bbox: List[float], img_shape: Tuple[int, int]) -> Tuple[float, float, float]:
        """Estimate length, width (m), and area (m²) of a detection box on the road.

        The pavement is a plane, so the mounting height and pitch in the request
        payload fix an exact image-to-ground homography, and the box is measured
        through it. This is what makes a crack's width mean anything: the old
        code scaled the box's horizontal pixel extent by a single depth, which
        for a crack running along the road scaled the crack's *length* and so
        reported ordinary cracks at the worst severity on the road.

        Returns ``(length_m, width_m, area_m2)``. When the pose cannot reach the
        road -- a level camera, or a box at the horizon -- the single-depth
        estimate is used instead and the returned width is flagged by
        `ground_sized` being False, because a number that cannot be trusted
        should be identifiable as such by the caller.
        """
        if self.pavement is not None:
            extent = self.pavement.measure_box(bbox)
            if extent is not None:
                return extent.length_m, extent.width_m, extent.area_m2

        x1, y1, x2, y2 = bbox
        w_px = abs(x2 - x1)
        h_px = abs(y2 - y1)
        
        # Ground distance estimation based on camera tilt and height
        tilt_rad = math.radians(self.camera_tilt_deg)
        y_center = (y1 + y2) / 2.0
        
        # Estimate depth Z (distance from camera)
        angle_y = math.atan2((y_center - self.cy), self.fy)
        total_angle = tilt_rad + angle_y
        z_m = self.camera_height_m / math.sin(total_angle) if math.sin(total_angle) > 0 else 5.0
        
        # Compute real-world dimensions
        width_m = (w_px * z_m) / self.fx
        length_m = (h_px * z_m) / self.fy
        area_m2 = width_m * length_m

        return width_m, length_m, area_m2

    def _ground_size(self, bbox: List[float], img_shape: Tuple[int, int]) -> Tuple[float, float, float, bool]:
        """``(width_m, length_m, area_m2, ground_sized)``.

        ``ground_sized`` is False when the single-depth fallback was used, so a
        caller can leave the severity unrated instead of rating a crack off a
        number the geometry cannot support.
        """
        if self.pavement is not None:
            extent = self.pavement.measure_box(bbox)
            if extent is not None:
                return extent.width_m, extent.length_m, extent.area_m2, True
        width_m, length_m, area_m2 = self._pixel_to_real_world(bbox, img_shape)
        return width_m, length_m, area_m2, False

    def _calculate_deduct_value(self, class_id: int, area_m2: float) -> float:
        """Estimate ASTM D6433 deduct value based on distress area density."""
        # Baseline density-to-deduct polynomial approximation
        density = min(area_m2 / 100.0, 1.0) * 100
        base_dv = 15.0 * math.log1p(density)
        return float(np.clip(base_dv, 0.0, 100.0))

    #: Real seconds that one video frame covers, when it cannot be derived.
    #: The fallback is the nominal frame rate, which is what a video recorded in
    #: real time will give.
    DEFAULT_FRAME_INTERVAL_S = 1.0

    #: Outside this range a derived interval is treated as unusable rather than
    #: believed. One frame per 100 ms means a 10x time-lapse and one per 600 s
    #: means frames minutes apart; either way the video and the GPS are not
    #: describing the same pass and no single scale relates them.
    MIN_PLAUSIBLE_FRAME_INTERVAL_S = 0.1
    MAX_PLAUSIBLE_FRAME_INTERVAL_S = 600.0

    #: The GPS span and the video's nominal duration may differ by this much
    #: before the video is treated as covering a different stretch of the drive
    #: than the GPS. Container frame counts and durations are themselves only
    #: approximate, so a small disagreement is not evidence of anything.
    FRAME_INTERVAL_AGREEMENT_TOLERANCE = 0.05

    def _resolve_frame_count(self, video_path: str, cap: Any) -> int:
        """How many frames the video really has.

        The frame interval is derived by dividing the GPS span by the frame
        count, so a wrong count rescales every detection in the survey. The
        container's own count is the cheap answer but is not dependable on these
        files: OpenCV reported 1 for a 1877-frame `.mkv` on one pass and the
        correct value on the next, which is enough to turn a 1.1965 s/frame
        interval into 2241 s/frame.

        Three sources are tried in order of cost, most trustworthy first: a full
        decode, which cannot be wrong but reads the whole file; `ffprobe`, which
        is exact from the container's own index; and OpenCV's reported count,
        which is free but unreliable. The cheap answer is only taken when it is
        plausible on its own terms.
        """
        reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if reported >= 2:
            return reported

        try:
            probe = subprocess.run(
                [
                    "ffprobe", "-v", "quiet", "-select_streams", "v:0",
                    "-count_frames", "-show_entries", "stream=nb_read_frames",
                    "-of", "csv=p=0", video_path,
                ],
                capture_output=True, text=True, timeout=600,
            )
            counted = probe.stdout.strip()
            if counted.isdigit() and int(counted) >= 2:
                return int(counted)
        except (OSError, subprocess.SubprocessError):
            # No ffprobe on the box, or it could not read the file. The decode
            # below still gives an answer.
            pass

        if reported >= 1:
            return reported

        counted = 0
        reader = cv2.VideoCapture(video_path)
        while reader.isOpened():
            ok, _ = reader.read()
            if not ok:
                break
            counted += 1
        reader.release()
        return counted

    def _gps_time_span(self, gps_points: Optional[List[Dict[str, Any]]]) -> float:
        """The real time the GPS track covers, in seconds."""
        times: List[float] = []
        for point in gps_points or []:
            try:
                times.append(float(point["time"]))
            except (ValueError, TypeError, KeyError):
                continue
        if len(times) < 2:
            return 0.0
        return max(times) - min(times)

    def _make_gap_detector(self) -> "FrameGapDetector":
        return FrameGapDetector()

    def _resolve_frame_interval(
        self,
        frame_interval_s: Optional[float],
        total_frames: int,
        gps_points: Optional[List[Dict[str, Any]]],
        nominal_fps: float,
    ) -> Tuple[float, str]:
        """How many seconds of the drive one video frame stands for.

        The frame rate in the container is not the answer, and neither is any
        assumed selection rule. These videos were recorded at 30 fps and reduced
        by keeping one frame per window — the sharpest one, so the frames are not
        a fixed stride through real time — and the container still reports
        1.00 fps. Matching a frame at `frame_idx / fps` therefore walks the GPS
        track at the wrong rate: the error grows with the frame index and reaches
        about 20% by the end of a trip, roughly 2.2 km on the 15.5 km runs. That
        is larger than any other positioning error in the pipeline, and it
        displaces every defect in the survey silently.

        Rather than encode a selection rule that may change, the interval is
        measured from the data: the GPS span divided by the frame count. Across
        the KNUST Obuasi trips that gives 1.1965 s/frame, and the spread is
        ±0.002 between a 358-frame and a 2390-frame trip. A fixed per-frame
        interval looks like that; an arbitrary trim does not.

        The cases handled, in the order they are decided:

        * An explicit value is always honoured.
        * No GPS, no frames, or a span that is zero: the nominal frame rate, so
          a real-time video resolves correctly and the caller is told the answer
          is nominal rather than measured.
        * A nominal duration that already agrees with the GPS span to within
          tolerance: the video and the GPS cover the same stretch at the
          container's own rate, so the nominal interval is used and nothing is
          rescaled. This is the common real-time case.
        * Otherwise the measured interval is used, because that is what the two
          sources jointly imply. A video covering only part of the GPS window
          and a time-compressed video produce the same number here, and the code
          does not claim to tell them apart — it reports the ratio so a reader
          can.
        * A measured interval outside the plausible range is refused rather than
          believed, and the nominal rate is used with the disagreement recorded.

        "Best frame" selection also means frames are not exactly evenly spaced
        within their window, so the true time of a frame carries jitter bounded
        by that window. At a second or so that is a few metres, far below the
        50 m segment size, which is why a single interval is a good enough
        description and the code does not attempt to model it.
        """
        if frame_interval_s is not None and frame_interval_s > 0:
            return float(frame_interval_s), "explicit"

        nominal = 1.0 / nominal_fps if nominal_fps and nominal_fps > 0 else self.DEFAULT_FRAME_INTERVAL_S
        span = self._gps_time_span(gps_points)
        if total_frames < 2 or span <= 0:
            return nominal, "nominal"

        nominal_duration = total_frames * nominal
        if nominal_duration <= 0:
            return nominal, "nominal"

        disagreement = abs(nominal_duration - span) / span
        if disagreement <= self.FRAME_INTERVAL_AGREEMENT_TOLERANCE:
            # The container's rate already reconciles the two sources.
            return nominal, "nominal"

        measured = span / (total_frames - 1)
        if not (self.MIN_PLAUSIBLE_FRAME_INTERVAL_S <= measured <= self.MAX_PLAUSIBLE_FRAME_INTERVAL_S):
            # No single scale relates these. Falling back to the nominal rate is
            # a guess, but the recorded source name is what makes it visible.
            return nominal, "nominal_unreconcilable"

        return measured, "measured"

    def _read_overlay_clock(
        self,
        video_path: str,
        total_frames: int,
    ) -> Tuple[Optional[Any], Dict[str, Any]]:
        """The frame-to-time axis read from the burned-in clock, if asked for.

        Returns the axis and a report. The report is returned even on failure
        because a survey that asked for the overlay and did not get it needs to
        say so rather than report the fallback as though it were the answer.
        """
        if not self.use_overlay_clock:
            return None, {}
        try:
            from .overlay import build_overlay_clock

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None, {"read": False, "reason": "video could not be opened"}
            try:
                def frame_at(index: int) -> Optional[Any]:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, frame = cap.read()
                    return frame if ok else None

                axis, report = build_overlay_clock(frame_at, total_frames)
            finally:
                cap.release()
        except Exception as error:  # pragma: no cover - defensive
            return None, {"read": False, "reason": f"overlay clock failed: {error}"}

        if axis is None:
            return None, report
        report["gaps"] = axis.gaps()[:20]
        return axis, report

    def _open_video(self, video_path: str) -> Tuple[Any, Optional[str], Dict[str, Any]]:
        """Open a video, remuxing first if the container cannot be decoded.

        A phone or dash recorder writes MP4 progressively: the `moov` index sits
        at the end of the file, and if the recording is cut short — a crash, a
        flat battery, an upload that stops — that index never arrives. OpenCV
        then opens the file happily and returns zero frames, which is how a
        survey that has no footage ends up reported as a survey that found
        nothing. An empty result and an unreadable file are different facts and
        must not look the same.

        The repair is to remux rather than re-encode, which only moves the index
        and is fast enough to run on a long recording. If ffmpeg is not
        available the caller still gets the original behaviour, and the note
        says the file was never read.
        """
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            ok, frame = cap.read()
            if ok and frame is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return cap, None, {"opened": True, "frames_reported": reported}
            cap.release()

        remuxed = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
                remuxed = handle.name
            probe = subprocess.run(
                ["ffmpeg", "-nostdin", "-y", "-loglevel", "error",
                 "-i", video_path, "-c", "copy",
                 "-movflags", "+faststart", remuxed],
                capture_output=True, text=True, timeout=3600,
            )
            if probe.returncode == 0 and os.path.exists(remuxed):
                repaired = cv2.VideoCapture(remuxed)
                ok, frame = repaired.read()
                if ok and frame is not None:
                    repaired.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    return repaired, remuxed, {
                        "opened": True,
                        "remuxed": True,
                        "frames_reported": int(repaired.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
                    }
                repaired.release()
        except (OSError, subprocess.SubprocessError):
            # No ffmpeg, or it could not read the file. The note below is what
            # tells the caller the video was never actually seen.
            pass
        if remuxed and os.path.exists(remuxed):
            try:
                os.unlink(remuxed)
            except OSError:
                pass
        return cap, None, {
            "opened": False,
            "reason": "the video opened but no frame could be decoded",
        }

    def _create_gps_interpolator(self, gps_points: Optional[List[Dict[str, Any]]]) -> Optional[Any]:
        """Maps frame timestamps to GPS coordinates via linear interpolation."""
        if not gps_points or len(gps_points) < 2:
            return None

        times, lats, lons = [], [], []
        for gp in gps_points:
            try:
                times.append(float(gp['time']))
                lats.append(float(gp['lat']))
                lons.append(float(gp['lon']))
            except (ValueError, TypeError, KeyError):
                continue

        if len(times) < 2:
            return None

        sorted_indices = np.argsort(times)
        times = np.array(times)[sorted_indices]
        lats = np.array(lats)[sorted_indices]
        lons = np.array(lons)[sorted_indices]

        times, unique_indices = np.unique(times, return_index=True)
        lats = lats[unique_indices]
        lons = lons[unique_indices]

        def interpolator(timestamp: float) -> Tuple[float, float]:
            # A frame outside the GPS coverage is a real miss, and returning an
            # endpoint for it would state a position nobody recorded. Holding
            # the frame outside the fixes is the same miss the caller already
            # knows how to handle, and the non-finite coordinate drops the
            # detection out of the per-section tables rather than placing it at
            # the end of the road.
            if timestamp < times[0] or timestamp > times[-1]:
                return float("nan"), float("nan")
            idx = np.searchsorted(times, timestamp)
            if idx >= len(times):
                return float("nan"), float("nan")
            t0, t1 = times[idx - 1], times[idx]
            alpha = (timestamp - t0) / (t1 - t0) if t1 != t0 else 0
            lat = lats[idx - 1] + alpha * (lats[idx] - lats[idx - 1])
            lon = lons[idx - 1] + alpha * (lons[idx] - lons[idx - 1])
            return float(lat), float(lon)

        return interpolator

    #: Fallback speed (m/s) for chainage when a survey has no usable GPS track.
    #: Only ever used as a stated assumption, never silently: callers are told
    #: which of the two they got via `chainage_source`.
    FALLBACK_SPEED_M_S = 10.0

    #: Ceiling used to sanity-check the distance implied between two fixes
    #: (216 km/h). A step implying more than this is a bad fix or a dropout, not
    #: a road, and is clamped rather than believed or discarded outright.
    MAX_PLAUSIBLE_SPEED_M_S = 60.0

    @staticmethod
    def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        radius_m = 6371008.8
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        d_phi = phi2 - phi1
        d_lambda = math.radians(lon2 - lon1)
        a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
        return 2.0 * radius_m * math.asin(min(1.0, math.sqrt(a)))

    def _create_chainage_interpolator(self, gps_points: Optional[List[Dict[str, Any]]]) -> Optional[Any]:
        """Map frame timestamps to distance travelled along the GPS track.

        Chainage is what locates a defect along the road, and it feeds the PCI
        extent and severity weighting, so deriving it from `timestamp * 10.0`
        made every chainage wrong by however far the vehicle was from 10 m/s —
        silently, and with the same answer for two surveys that drove at
        different speeds. The cumulative distance along the interpolated track
        is the measured quantity; the constant-speed form survives only as a
        labelled fallback for a run with no GPS at all.
        """
        if not gps_points or len(gps_points) < 2:
            return None

        fixes: List[Tuple[float, float, float]] = []
        for point in gps_points:
            try:
                fixes.append((float(point["time"]), float(point["lat"]), float(point["lon"])))
            except (ValueError, TypeError, KeyError):
                continue
        if len(fixes) < 2:
            return None

        fixes.sort(key=lambda fix: fix[0])
        times: List[float] = []
        chainages: List[float] = []
        cumulative = 0.0
        previous: Optional[Tuple[float, float, float]] = None
        for time_s, lat, lon in fixes:
            if previous is not None:
                elapsed = time_s - previous[0]
                step = self._haversine_m(previous[1], previous[2], lat, lon)
                # Clamp against a speed the vehicle could actually have reached
                # in the interval between fixes. A dropout still advanced the
                # vehicle, so the distance is kept up to the plausible maximum
                # rather than dropped, and only a fix implying an impossible
                # speed is cut back.
                if elapsed > 0:
                    step = min(step, self.MAX_PLAUSIBLE_SPEED_M_S * elapsed)
                cumulative += step
            times.append(time_s)
            chainages.append(cumulative)
            previous = (time_s, lat, lon)

        if times[-1] <= times[0]:
            return None

        time_array = np.array(times, dtype=float)
        chainage_array = np.array(chainages, dtype=float)

        def interpolator(timestamp: float) -> float:
            # `np.interp` holds a value outside the fixes at the nearest
            # endpoint, which turns a frame the GPS never covered into a real
            # chainage — the end of the road, or its start. A nan propagates
            # instead, so the caller sees the miss and the detection keeps no
            # position it cannot support.
            if timestamp < time_array[0] or timestamp > time_array[-1]:
                return float("nan")
            return float(np.interp(timestamp, time_array, chainage_array))

        return interpolator

    def _chainage_for(
        self,
        timestamp: float,
        chainage_interp: Optional[Any],
    ) -> Tuple[float, str]:
        """Chainage for a frame, and where that chainage came from."""
        if chainage_interp is not None:
            measured = float(chainage_interp(timestamp))
            # A frame outside the GPS coverage has no measured distance. Falling
            # back to the constant-speed estimate here would be a different
            # assumption from the one the rest of the run rests on, and would
            # hide the gap; `unavailable` says the frame simply has no GPS.
            if math.isfinite(measured):
                return round(measured, 2), "gps"
            return 0.0, "unavailable"
        return round(timestamp * self.FALLBACK_SPEED_M_S, 2), "assumed_speed"

    def _estimate_dense_depth(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Estimate dense depth map using DepthAnythingV2."""
        if self.dav2_estimator is None:
            return None
        try:
            depth_map = self.dav2_estimator.estimate(frame)
            return depth_map
        except Exception as e:
            logger.warning(f"Dense depth estimation failed: {e}")
            return None

    def _scale_depth_to_metric(self, inv_depth: np.ndarray) -> np.ndarray:
        """Convert inverse depth to metric depth using geometry-anchored scaling."""
        if self.geom_scaler is None:
            return inv_depth
        try:
            metric_depth = self.geom_scaler.apply(
                inv_depth, self.camera_matrix,
                self.camera_tilt_deg, 0.0, self.camera_height_m
            )
            return metric_depth
        except Exception as e:
            logger.warning(f"Metric depth scaling failed: {e}")
            return inv_depth

    def _estimate_pothole_depth(
        self, 
        rgb_frame: np.ndarray, 
        depth_map: np.ndarray, 
        bbox: List[float]
    ) -> Dict[str, Any]:
        """Estimate pothole depth using MBTP with RGB segmentation fallback."""
        result = {
            "mbtp_area_m2": None,
            "mbtp_depth_m": None,
            "mbtp_volume_m3": None,
            "mbtp_confidence": None,
            "mbtp_method": None,
            "depth_method": "pinhole_geometry",
            "depth_message": "Estimated via camera geometry model"
        }
        
        if self.mbtp_estimator is None or depth_map is None:
            return result
        
        try:
            mbtp_result: MBTPEstimate = self.mbtp_estimator.estimate(
                rgb_image=rgb_frame,
                depth_map=depth_map,
                bbox=bbox,
                K=self.camera_matrix
            )
            
            result["mbtp_area_m2"] = mbtp_result.area_m2
            result["mbtp_depth_m"] = mbtp_result.depth_m
            result["mbtp_volume_m3"] = mbtp_result.volume_m3
            result["mbtp_confidence"] = mbtp_result.confidence
            result["mbtp_method"] = "mbtp"
            result["depth_method"] = "mbtp_triangulation"
            result["depth_message"] = f"MBTP triangulation (confidence: {mbtp_result.confidence:.2f})"
            
        except Exception as e:
            logger.warning(f"MBTP depth estimation failed: {e}")
        
        return result

    def _get_dense_depth_for_bbox(self, depth_map: np.ndarray, bbox: List[float]) -> Optional[float]:
        """Get median dense depth for a bounding box region."""
        if depth_map is None:
            return None
        try:
            x1, y1, x2, y2 = map(int, bbox)
            h, w = depth_map.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            roi = depth_map[y1:y2, x1:x2]
            valid = roi[np.isfinite(roi) & (roi > 0)]
            
            if valid.size == 0:
                return None
            
            return float(np.median(valid))
        except Exception:
            return None

    def process_video_stream(
        self,
        video_path: str,
        gps_points: Optional[List[Dict[str, Any]]] = None,
        run_id: str = "",
    ) -> VisionAnalysisResult:
        """Backward compatibility wrapper."""
        return self.process_video_astm(
            video_path=video_path,
            gps_points=gps_points,
            run_id=run_id,
            target_fps=15.0,
        )

    def process_video_astm(
        self,
        video_path: str,
        gps_points: Optional[List[Dict[str, Any]]] = None,
        run_id: str = "",
        target_fps: float = 15.0,
        frame_interval_s: Optional[float] = None,
    ) -> VisionAnalysisResult:
        """Process video for pavement distress detection using YOLO and ASTM D6433 standards."""
        if not video_path or not os.path.exists(video_path):
            logger.warning(f"Video file not found: {video_path}", run_id=run_id)
            return VisionAnalysisResult(
                video_quality=VideoQualityMetrics(0, (0, 0), 0, 0, 0, "INSUFFICIENT", 0, 0),
                frame_sampling=FrameSamplingInfo(0, 0, "none", 0, 0)
            )

        cap, remuxed_path, open_note = self._open_video(video_path)
        if not open_note.get("opened"):
            logger.error(
                f"Cannot read any frame from {video_path}: {open_note.get('reason')}. "
                f"A recording whose index was never written looks exactly like one with no "
                f"defects, so this survey has produced no evidence rather than a clean result.",
                run_id=run_id,
            )
            return VisionAnalysisResult(
                video_quality=VideoQualityMetrics(0, (0, 0), 0, 0, 0, "INSUFFICIENT", 0, 0),
                frame_sampling=FrameSamplingInfo(
                    0, 0, "none", 0, 0, overlay_clock={"video_decode": open_note}
                ),
            )
        if open_note.get("remuxed"):
            logger.warning(
                f"{os.path.basename(video_path)} would not decode and had to be remuxed "
                f"before it could be read; the footage was not missing, its index was.",
                run_id=run_id,
            )

        original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = self._resolve_frame_count(video_path, cap)

        # Determine FPS flag per MDPI paper specification
        if original_fps >= 15.0:
            fps_flag = "OK"
        elif original_fps >= 10.0:
            fps_flag = "LOW_FPS"
        else:
            fps_flag = "INSUFFICIENT"

        gps_interp = self._create_gps_interpolator(gps_points)
        chainage_interp = self._create_chainage_interpolator(gps_points)
        frame_interval = max(1, int(original_fps / target_fps)) if target_fps > 0 else 1
        seconds_per_frame, interval_source = self._resolve_frame_interval(
            frame_interval_s, total_frames, gps_points, original_fps
        )
        clock, overlay_report = self._read_overlay_clock(video_path, total_frames)
        if clock is not None:
            # The clock read from the video outranks anything inferred from the
            # GPS span, but not a value the caller supplied explicitly.
            if interval_source != "explicit":
                seconds_per_frame = clock.seconds_per_frame
                interval_source = "overlay"
            logger.info(
                f"{os.path.basename(video_path)}: burned-in clock gives "
                f"{clock.seconds_per_frame:.4f} s/frame from {overlay_report.get('frames_measured')} "
                f"frames read by {overlay_report.get('method')}; "
                f"{len(overlay_report.get('gaps') or [])} stretches of the drive are missing from the video.",
                run_id=run_id,
            )
        elif overlay_report:
            logger.warning(
                f"{os.path.basename(video_path)}: the burned-in clock was asked for and not used "
                f"({overlay_report.get('reason') or 'unreadable'}). Falling back to "
                f"{seconds_per_frame:.4f} s/frame from {interval_source}.",
                run_id=run_id,
            )
        if interval_source == "nominal_unreconcilable":
            logger.warning(
                f"{os.path.basename(video_path)}: the GPS covers {self._gps_time_span(gps_points):.0f} s "
                f"but the video is {total_frames} frames at {original_fps:.2f} fps. No single "
                f"interval relates them, so {seconds_per_frame:.3f} s/frame from the nominal rate "
                f"is being used as an assumption, and detection positions are only as good as it is.",
                run_id=run_id,
            )
        elif interval_source == "nominal" and gps_points:
            logger.info(
                f"{os.path.basename(video_path)}: {seconds_per_frame:.3f} s/frame from the container "
                f"frame rate; the GPS span agrees with the video duration.",
                run_id=run_id,
            )

        detections: List[Dict[str, Any]] = []
        processed_frames = 0
        frames_skipped = 0
        frame_idx = 0
        gap_detector = self._make_gap_detector()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval != 0:
                frames_skipped += 1
                frame_idx += 1
                continue

            processed_frames += 1
            # The measured clock wins where there is one; without it this is the
            # same linear map as before, so the default path is unchanged.
            timestamp = (clock.seconds_for(frame_idx) if clock is not None
                        else frame_idx * seconds_per_frame)
            gap_detector.observe(frame)
            frame_undistorted = self.undistort_frame(frame)

            # Get GPS position
            lat, lon = (0.0, 0.0)
            if gps_interp:
                lat, lon = gps_interp(timestamp)
            # A frame outside the GPS coverage comes back as NaN. Storing NaN as
            # a point geometry would fail at the database, and storing (0, 0)
            # would be a real coordinate off the coast of Africa. Neither is
            # correct, so the detection simply carries no position.
            has_position = math.isfinite(lat) and math.isfinite(lon)

            # Perform detection if model exists
            if self.model:
                results = self._run_detection(frame_undistorted)
                if results:
                    for r in results:
                        boxes = getattr(r, "boxes", None)
                        if boxes:
                            for box in boxes:
                                cls_id = int(box.cls[0].item())
                                conf = float(box.conf[0].item())
                                xyxy = box.xyxy[0].tolist()

                                width_m, length_m, area_m2, ground_sized = self._ground_size(
                                    xyxy, (height, width)
                                )
                                astm_name = self.YOLO_TO_AASHTO.get(cls_id, "Unknown")
                                defect_name = self.CLASS_NAMES.get(cls_id, "Unknown")
                                deduct_val = self._calculate_deduct_value(cls_id, area_m2)

                                pothole_dia = None
                                pothole_dep = None
                                pothole_sev = None
                                ghana_rating = 5  # Default moderate

                                if cls_id == 4:  # Pothole
                                    # A pothole is compact, so its diameter is
                                    # the larger of the two metric extents.
                                    pothole_dia = max(width_m, length_m) * 1000.0
                                    pothole_dep = 30.0  # Estimated depth in mm
                                    pothole_sev = self._classify_pothole_severity(pothole_dia, pothole_dep)
                                    ghana_rating = 7 if pothole_sev == "HIGH" else (4 if pothole_sev == "MEDIUM" else 2)
                                elif cls_id == 2:  # Crack
                                    # Severity keys on the crack's width, which
                                    # is only meaningful when the size came off
                                    # the ground plane. Without a reachable
                                    # pavement the width is a pixel ratio, and
                                    # rating on it would put ordinary cracks at
                                    # the top severity, so the rating is left
                                    # off rather than guessed.
                                    if ground_sized:
                                        crack_width_mm = width_m * 1000.0
                                        ghana_rating = 7 if crack_width_mm > 10 else (4 if crack_width_mm > 3 else 2)
                                elif cls_id in (6, 7):  # Rutting, Shoving
                                    ghana_rating = 7 if area_m2 > 0.5 else (4 if area_m2 > 0.1 else 2)
                                else:
                                    # Other distresses: rate by area
                                    ghana_rating = 7 if area_m2 > 1.0 else (4 if area_m2 > 0.2 else 2)

                                chainage_m, chainage_source = self._chainage_for(timestamp, chainage_interp)
                                detection_item = DefectDetection(
                                    defect_type=defect_name,
                                    astm_distress_type=astm_name,
                                    ghana_distress_type=self.YOLO_TO_GHANA.get(cls_id, "Unknown"),
                                    yolo_class_id=cls_id,
                                    area=round(area_m2, 4),
                                    width=round(width_m, 2),
                                    depth_mm=30.0 if cls_id == 4 else 0.0,
                                    deduct_value=round(deduct_val, 2),
                                    ghana_rating_1to9=ghana_rating,
                                    center_m=(round(width_m / 2, 2), round(length_m / 2, 2)),
                                    confidence=round(conf, 3),
                                    chainage_m=chainage_m,
                                    chainage_source=chainage_source,
                                    ground_sized=ground_sized,
                                    bbox=xyxy,
                                    frame_idx=int(frame_idx),
                                    geom=(
                                        {"type": "Point", "coordinates": [lon, lat]}
                                        if has_position
                                        else None
                                    ),
                                    pothole_diameter_mm=pothole_dia,
                                    pothole_depth_mm=pothole_dep,
                                    pothole_severity=pothole_sev
                                )
                                detections.append(detection_item.__dict__)

            frame_idx += 1

        cap.release()
        if remuxed_path and os.path.exists(remuxed_path):
            # A full-length 4K remux is gigabytes; leaving it behind would fill
            # the disk on a busy machine rather than fail a single survey.
            try:
                os.unlink(remuxed_path)
            except OSError:
                pass

        video_quality = VideoQualityMetrics(
            fps=float(original_fps),
            resolution=(width, height),
            motion_blur_score=85.0,
            illumination_score=90.0,
            gps_sync_quality=1.0 if gps_interp else 0.0,
            fps_flag=fps_flag,
            frames_analyzed=processed_frames,
            frames_skipped=frames_skipped
        )

        gap_report = gap_detector.report()
        if gap_report.get("suspected_gaps"):
            logger.warning(
                f"{os.path.basename(video_path)}: {gap_report['suspected_gaps']} step(s) differ far "
                f"more than the rest (largest {gap_report['largest_step']:.1f} against a median of "
                f"{gap_report['median_step']:.1f}). Frames may be missing, in which case the "
                f"{seconds_per_frame:.3f} s/frame interval is an average and detection positions "
                f"either side of a gap are displaced by the time it covers.",
                run_id=run_id,
            )

        frame_sampling = FrameSamplingInfo(
            original_fps=float(original_fps),
            processed_fps=float(original_fps / frame_interval),
            sampling_method="uniform",
            total_frames=total_frames,
            processed_frames=processed_frames,
            seconds_per_frame=round(seconds_per_frame, 6),
            frame_interval_source=interval_source,
            frame_gaps=gap_report,
            overlay_clock=overlay_report,
        )

        return VisionAnalysisResult(
            detections=detections,
            video_quality=video_quality,
            frame_sampling=frame_sampling,
            astm_distress_mapping=self.YOLO_TO_AASHTO,
            ghana_distress_mapping=self.YOLO_TO_GHANA,
            dense_depth_map=None,  # Not stored per frame to save memory
            depth_estimation_enabled=self.enable_depth_estimation and self.dav2_estimator is not None,
            depth_estimation_method="dav2v2_geom_mbtp" if self.dav2_estimator else None
        )

    def process_video_with_depth(
        self,
        video_path: str,
        gps_points: Optional[List[Dict[str, Any]]] = None,
        run_id: str = "",
        target_fps: float = 15.0,
        estimate_dense_depth: bool = True,
        frame_interval_s: Optional[float] = None,
    ) -> VisionAnalysisResult:
        """Process video with dense depth estimation for all frames (slower but more accurate)."""
        if not video_path or not os.path.exists(video_path):
            logger.warning(f"Video file not found: {video_path}", run_id=run_id)
            return VisionAnalysisResult(
                video_quality=VideoQualityMetrics(0, (0, 0), 0, 0, 0, "INSUFFICIENT", 0, 0),
                frame_sampling=FrameSamplingInfo(0, 0, "none", 0, 0)
            )

        cap, remuxed_path, open_note = self._open_video(video_path)
        if not open_note.get("opened"):
            logger.error(
                f"Cannot read any frame from {video_path}: {open_note.get('reason')}. "
                f"A recording whose index was never written looks exactly like one with no "
                f"defects, so this survey has produced no evidence rather than a clean result.",
                run_id=run_id,
            )
            return VisionAnalysisResult(
                video_quality=VideoQualityMetrics(0, (0, 0), 0, 0, 0, "INSUFFICIENT", 0, 0),
                frame_sampling=FrameSamplingInfo(
                    0, 0, "none", 0, 0, overlay_clock={"video_decode": open_note}
                ),
            )
        if open_note.get("remuxed"):
            logger.warning(
                f"{os.path.basename(video_path)} would not decode and had to be remuxed "
                f"before it could be read; the footage was not missing, its index was.",
                run_id=run_id,
            )

        original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = self._resolve_frame_count(video_path, cap)

        if original_fps >= 15.0:
            fps_flag = "OK"
        elif original_fps >= 10.0:
            fps_flag = "LOW_FPS"
        else:
            fps_flag = "INSUFFICIENT"

        gps_interp = self._create_gps_interpolator(gps_points)
        chainage_interp = self._create_chainage_interpolator(gps_points)
        frame_interval = max(1, int(original_fps / target_fps)) if target_fps > 0 else 1
        seconds_per_frame, interval_source = self._resolve_frame_interval(
            frame_interval_s, total_frames, gps_points, original_fps
        )
        clock, overlay_report = self._read_overlay_clock(video_path, total_frames)
        if clock is not None:
            if interval_source != "explicit":
                seconds_per_frame = clock.seconds_per_frame
                interval_source = "overlay"
            logger.info(
                f"{os.path.basename(video_path)}: burned-in clock gives "
                f"{clock.seconds_per_frame:.4f} s/frame from {overlay_report.get('frames_measured')} "
                f"frames read by {overlay_report.get('method')}.",
                run_id=run_id,
            )
        elif overlay_report:
            logger.warning(
                f"{os.path.basename(video_path)}: the burned-in clock was asked for and not used "
                f"({overlay_report.get('reason') or 'unreadable'}). Falling back to "
                f"{seconds_per_frame:.4f} s/frame from {interval_source}.",
                run_id=run_id,
            )
        if interval_source == "nominal_unreconcilable":
            logger.warning(
                f"{os.path.basename(video_path)}: the GPS covers {self._gps_time_span(gps_points):.0f} s "
                f"but the video is {total_frames} frames at {original_fps:.2f} fps. No single "
                f"interval relates them, so {seconds_per_frame:.3f} s/frame from the nominal rate "
                f"is being used as an assumption, and detection positions are only as good as it is.",
                run_id=run_id,
            )
        elif interval_source == "nominal" and gps_points:
            logger.info(
                f"{os.path.basename(video_path)}: {seconds_per_frame:.3f} s/frame from the container "
                f"frame rate; the GPS span agrees with the video duration.",
                run_id=run_id,
            )

        detections: List[Dict[str, Any]] = []
        processed_frames = 0
        frames_skipped = 0
        frame_idx = 0
        gap_detector = self._make_gap_detector()
        
        # For dense depth, we'll store the last estimated depth map
        last_depth_map = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval != 0:
                frames_skipped += 1
                frame_idx += 1
                continue

            processed_frames += 1
            # The measured clock wins where there is one; without it this is the
            # same linear map as before, so the default path is unchanged.
            timestamp = (clock.seconds_for(frame_idx) if clock is not None
                        else frame_idx * seconds_per_frame)
            gap_detector.observe(frame)
            frame_undistorted = self.undistort_frame(frame)

            # Get GPS position
            lat, lon = (0.0, 0.0)
            if gps_interp:
                lat, lon = gps_interp(timestamp)
            # A frame outside the GPS coverage comes back as NaN. Storing NaN as
            # a point geometry would fail at the database, and storing (0, 0)
            # would be a real coordinate off the coast of Africa. Neither is
            # correct, so the detection simply carries no position.
            has_position = math.isfinite(lat) and math.isfinite(lon)

            # Estimate dense depth for this frame (first frame or periodically)
            if estimate_dense_depth and self.dav2_estimator and (frame_idx == 0 or frame_idx % 30 == 0):
                last_depth_map = self._estimate_dense_depth(frame_undistorted)
                if last_depth_map is not None:
                    last_depth_map = self._scale_depth_to_metric(last_depth_map)

            # Perform detection if model exists
            if self.model:
                results = self._run_detection(frame_undistorted)
                if results:
                    for r in results:
                        boxes = getattr(r, "boxes", None)
                        if boxes:
                            for box in boxes:
                                cls_id = int(box.cls[0].item())
                                conf = float(box.conf[0].item())
                                xyxy = box.xyxy[0].tolist()

                                width_m, length_m, area_m2, ground_sized = self._ground_size(
                                    xyxy, (height, width)
                                )
                                astm_name = self.YOLO_TO_AASHTO.get(cls_id, "Unknown")
                                defect_name = self.CLASS_NAMES.get(cls_id, "Unknown")
                                deduct_val = self._calculate_deduct_value(cls_id, area_m2)

                                pothole_dia = None
                                pothole_dep = None
                                pothole_sev = None
                                ghana_rating = 5
                                dense_depth_mm = None
                                mbtp_info = {}

                                # Get dense depth for this detection if available
                                if last_depth_map is not None:
                                    dense_depth = self._get_dense_depth_for_bbox(last_depth_map, xyxy)
                                    if dense_depth:
                                        dense_depth_mm = dense_depth * 1000.0  # Convert to mm

                                if cls_id == 4:  # Pothole - use MBTP for accurate depth
                                    pothole_dia = max(width_m, length_m) * 1000.0
                                    
                                    # Use MBTP if available and we have depth map
                                    if last_depth_map is not None and self.mbtp_estimator:
                                        mbtp_info = self._estimate_pothole_depth(frame_undistorted, last_depth_map, xyxy)
                                        if mbtp_info.get("mbtp_depth_m"):
                                            pothole_dep = mbtp_info["mbtp_depth_m"] * 1000.0
                                        else:
                                            pothole_dep = 30.0
                                    else:
                                        pothole_dep = 30.0  # Fallback estimate
                                    
                                    pothole_sev = self._classify_pothole_severity(pothole_dia, pothole_dep)
                                    ghana_rating = 7 if pothole_sev == "HIGH" else (4 if pothole_sev == "MEDIUM" else 2)
                                elif cls_id == 2:  # Crack
                                    # Only a ground-sized width is a crack width.
                                    if ground_sized:
                                        crack_width_mm = width_m * 1000.0
                                        ghana_rating = 7 if crack_width_mm > 10 else (4 if crack_width_mm > 3 else 2)
                                elif cls_id in (6, 7):  # Rutting, Shoving
                                    ghana_rating = 7 if area_m2 > 0.5 else (4 if area_m2 > 0.1 else 2)
                                else:
                                    ghana_rating = 7 if area_m2 > 1.0 else (4 if area_m2 > 0.2 else 2)

                                chainage_m, chainage_source = self._chainage_for(timestamp, chainage_interp)
                                detection_item = DefectDetection(
                                    defect_type=defect_name,
                                    astm_distress_type=astm_name,
                                    ghana_distress_type=self.YOLO_TO_GHANA.get(cls_id, "Unknown"),
                                    yolo_class_id=cls_id,
                                    area=round(area_m2, 4),
                                    width=round(width_m, 2),
                                    depth_mm=round(pothole_dep, 1) if pothole_dep else 0.0,
                                    deduct_value=round(deduct_val, 2),
                                    ghana_rating_1to9=ghana_rating,
                                    center_m=(round(width_m / 2, 2), round(length_m / 2, 2)),
                                    confidence=round(conf, 3),
                                    chainage_m=chainage_m,
                                    chainage_source=chainage_source,
                                    ground_sized=ground_sized,
                                    bbox=xyxy,
                                    frame_idx=int(frame_idx),
                                    geom=(
                                        {"type": "Point", "coordinates": [lon, lat]}
                                        if has_position
                                        else None
                                    ),
                                    pothole_diameter_mm=round(pothole_dia, 1) if pothole_dia else None,
                                    pothole_depth_mm=round(pothole_dep, 1) if pothole_dep else None,
                                    pothole_severity=pothole_sev,
                                    dense_depth_mm=round(dense_depth_mm, 1) if dense_depth_mm else None,
                                    mbtp_area_m2=mbtp_info.get("mbtp_area_m2"),
                                    mbtp_depth_m=mbtp_info.get("mbtp_depth_m"),
                                    mbtp_volume_m3=mbtp_info.get("mbtp_volume_m3"),
                                    mbtp_confidence=mbtp_info.get("mbtp_confidence"),
                                    mbtp_method=mbtp_info.get("mbtp_method"),
                                    depth_method=mbtp_info.get("depth_method"),
                                    depth_message=mbtp_info.get("depth_message")
                                )
                                detections.append(detection_item.__dict__)

            frame_idx += 1

        cap.release()
        if remuxed_path and os.path.exists(remuxed_path):
            # A full-length 4K remux is gigabytes; leaving it behind would fill
            # the disk on a busy machine rather than fail a single survey.
            try:
                os.unlink(remuxed_path)
            except OSError:
                pass

        video_quality = VideoQualityMetrics(
            fps=float(original_fps),
            resolution=(width, height),
            motion_blur_score=85.0,
            illumination_score=90.0,
            gps_sync_quality=1.0 if gps_interp else 0.0,
            fps_flag=fps_flag,
            frames_analyzed=processed_frames,
            frames_skipped=frames_skipped
        )

        gap_report = gap_detector.report()
        if gap_report.get("suspected_gaps"):
            logger.warning(
                f"{os.path.basename(video_path)}: {gap_report['suspected_gaps']} step(s) differ far "
                f"more than the rest (largest {gap_report['largest_step']:.1f} against a median of "
                f"{gap_report['median_step']:.1f}). Frames may be missing, in which case the "
                f"{seconds_per_frame:.3f} s/frame interval is an average and detection positions "
                f"either side of a gap are displaced by the time it covers.",
                run_id=run_id,
            )

        frame_sampling = FrameSamplingInfo(
            original_fps=float(original_fps),
            processed_fps=float(original_fps / frame_interval),
            sampling_method="uniform",
            total_frames=total_frames,
            processed_frames=processed_frames,
            seconds_per_frame=round(seconds_per_frame, 6),
            frame_interval_source=interval_source,
            frame_gaps=gap_report,
            overlay_clock=overlay_report,
        )

        return VisionAnalysisResult(
            detections=detections,
            video_quality=video_quality,
            frame_sampling=frame_sampling,
            astm_distress_mapping=self.YOLO_TO_AASHTO,
            ghana_distress_mapping=self.YOLO_TO_GHANA,
            dense_depth_map=last_depth_map,
            depth_estimation_enabled=self.enable_depth_estimation and self.dav2_estimator is not None,
            depth_estimation_method="dav2v2_geom_mbtp" if self.dav2_estimator else None
        )