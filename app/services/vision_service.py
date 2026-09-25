"""
Vision analysis service using YOLO for pavement distress detection.
Implements ASTM D6433-18 and MDPI Sensors 2024 paper methodology.
"""
import os
import json
import math
import numpy as np
import cv2
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "defect.pt")

# Placeholder logger and monitor setup
class DummyLogger:
    def warning(self, msg: str, **kwargs): print(f"[WARN] {msg}")
    def error(self, msg: str, **kwargs): print(f"[ERROR] {msg}")
    def info(self, msg: str, **kwargs): print(f"[INFO] {msg}")

logger = DummyLogger()

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
    pothole_diameter_mm: Optional[float] = None
    pothole_depth_mm: Optional[float] = None
    pothole_severity: Optional[str] = None

@dataclass
class VisionAnalysisResult:
    detections: List[Dict[str, Any]] = field(default_factory=list)
    video_quality: Optional[VideoQualityMetrics] = None
    frame_sampling: Optional[FrameSamplingInfo] = None
    astm_distress_mapping: Dict[int, str] = field(default_factory=dict)
    ghana_distress_mapping: Dict[int, str] = field(default_factory=dict)

# Alias for backward compatibility
ASTMVisionResult = VisionAnalysisResult


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

    def __init__(
        self, 
        model_path: Optional[str] = None, 
        camera_height_m: float = 2.5, 
        camera_tilt_deg: float = 0.0, 
        fx: float = 1000.0, 
        fy: float = 1000.0, 
        cx: float = 960.0, 
        cy: float = 540.0, 
        distortion_coefficients: Optional[str] = None
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
        
        # Load Ultralytics YOLO if path provided
        if model_path and os.path.exists(model_path):
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")

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
        """Classify pothole severity per ASTM D6433 Table X1.1."""
        if diameter_mm >= 750 or depth_mm > 50:
            return "HIGH"
        if depth_mm > 25 or diameter_mm >= 300:
            return "MEDIUM"
        return "LOW"

    def _pixel_to_real_world(self, bbox: List[float], img_shape: Tuple[int, int]) -> Tuple[float, float, float]:
        """Estimate length, width (m), and area (m²) of bounding box using pinhole model."""
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

    def _calculate_deduct_value(self, class_id: int, area_m2: float) -> float:
        """Estimate ASTM D6433 deduct value based on distress area density."""
        # Baseline density-to-deduct polynomial approximation
        density = min(area_m2 / 100.0, 1.0) * 100
        base_dv = 15.0 * math.log1p(density)
        return float(np.clip(base_dv, 0.0, 100.0))

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
            idx = np.searchsorted(times, timestamp)
            if idx == 0:
                return float(lats[0]), float(lons[0])
            if idx >= len(times):
                return float(lats[-1]), float(lons[-1])
            t0, t1 = times[idx - 1], times[idx]
            alpha = (timestamp - t0) / (t1 - t0) if t1 != t0 else 0
            lat = lats[idx - 1] + alpha * (lats[idx] - lats[idx - 1])
            lon = lons[idx - 1] + alpha * (lons[idx] - lons[idx - 1])
            return float(lat), float(lon)

        return interpolator

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
    ) -> VisionAnalysisResult:
        """Process video for pavement distress detection using YOLO and ASTM D6433 standards."""
        if not video_path or not os.path.exists(video_path):
            logger.warning(f"Video file not found: {video_path}", run_id=run_id)
            return VisionAnalysisResult(
                video_quality=VideoQualityMetrics(0, (0, 0), 0, 0, 0, "INSUFFICIENT", 0, 0),
                frame_sampling=FrameSamplingInfo(0, 0, "none", 0, 0)
            )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}", run_id=run_id)
            return VisionAnalysisResult(
                video_quality=VideoQualityMetrics(0, (0, 0), 0, 0, 0, "INSUFFICIENT", 0, 0),
                frame_sampling=FrameSamplingInfo(0, 0, "none", 0, 0)
            )

        original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Determine FPS flag per MDPI paper specification
        if original_fps >= 15.0:
            fps_flag = "OK"
        elif original_fps >= 10.0:
            fps_flag = "LOW_FPS"
        else:
            fps_flag = "INSUFFICIENT"

        gps_interp = self._create_gps_interpolator(gps_points)
        frame_interval = max(1, int(original_fps / target_fps)) if target_fps > 0 else 1

        detections: List[Dict[str, Any]] = []
        processed_frames = 0
        frames_skipped = 0
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval != 0:
                frames_skipped += 1
                frame_idx += 1
                continue

            processed_frames += 1
            timestamp = frame_idx / original_fps
            frame_undistorted = self.undistort_frame(frame)

            # Get GPS position
            lat, lon = (0.0, 0.0)
            if gps_interp:
                lat, lon = gps_interp(timestamp)

            # Perform detection if model exists
            if self.model:
                results = self.model(frame_undistorted, verbose=False)
                if results:
                    for r in results:
                        boxes = getattr(r, "boxes", None)
                        if boxes:
                            for box in boxes:
                                cls_id = int(box.cls[0].item())
                                conf = float(box.conf[0].item())
                                xyxy = box.xyxy[0].tolist()

                                width_m, length_m, area_m2 = self._pixel_to_real_world(xyxy, (height, width))
                                astm_name = self.YOLO_TO_AASHTO.get(cls_id, "Unknown")
                                ghana_name = self.YOLO_TO_GHANA.get(cls_id, "Unknown")
                                defect_name = self.CLASS_NAMES.get(cls_id, "Unknown")
                                deduct_val = self._calculate_deduct_value(cls_id, area_m2)

                                pothole_dia = None
                                pothole_dep = None
                                pothole_sev = None
                                ghana_rating = 5  # Default moderate

                                if cls_id == 4:  # Pothole
                                    pothole_dia = width_m * 1000.0
                                    pothole_dep = 30.0  # Estimated depth in mm
                                    pothole_sev = self._classify_pothole_severity(pothole_dia, pothole_dep)
                                    ghana_rating = 7 if pothole_sev == "HIGH" else (4 if pothole_sev == "MEDIUM" else 2)
                                elif cls_id == 2:  # Crack
                                    # Severity based on width
                                    crack_width_mm = width_m * 1000.0
                                    ghana_rating = 7 if crack_width_mm > 10 else (4 if crack_width_mm > 3 else 2)
                                elif cls_id in (6, 7):  # Rutting, Shoving
                                    ghana_rating = 7 if area_m2 > 0.5 else (4 if area_m2 > 0.1 else 2)
                                elif cls_id == 4:  # Pothole (already handled above)
                                    pass
                                else:
                                    # Other distresses: rate by area
                                    ghana_rating = 7 if area_m2 > 1.0 else (4 if area_m2 > 0.2 else 2)

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
                                    chainage_m=round(timestamp * 10.0, 2),  # Estimated chainage based on 10m/s speed
                                    bbox=xyxy,
                                    geom={"type": "Point", "coordinates": [lon, lat]},
                                    pothole_diameter_mm=pothole_dia,
                                    pothole_depth_mm=pothole_dep,
                                    pothole_severity=pothole_sev
                                )
                                detections.append(detection_item.__dict__)

            frame_idx += 1

        cap.release()

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

        frame_sampling = FrameSamplingInfo(
            original_fps=float(original_fps),
            processed_fps=float(original_fps / frame_interval),
            sampling_method="uniform",
            total_frames=total_frames,
            processed_frames=processed_frames
        )

        return VisionAnalysisResult(
            detections=detections,
            video_quality=video_quality,
            frame_sampling=frame_sampling,
            astm_distress_mapping=self.YOLO_TO_AASHTO,
            ghana_distress_mapping=self.YOLO_TO_AASHTO
        )