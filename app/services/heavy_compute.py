# app/pipelines/heavy_compute_pipeline.py
import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import cv2
import numpy as np
import pandas as pd

from shapely.geometry import LineString
from geoalchemy2 import WKTElement

from app.core.config import IRI_SEGMENT_LENGTH_M
from app.core.logger import get_pipeline_logger, monitor
from app.database.session import IS_POSTGRES, SessionLocal
from app.models.calibration import CalibrationProfile
from app.models.survey import IRISegment, MapMatchingRun, SurveyRun, TrackedDefect
from app.services.fusion_service import PavementConditionFusioneer
from app.services.inertial_service import (
    InertialPipelineResult,
    ASTMQCMResult,
    InertialRefineryService,
)
from app.services.spatial_service import SpatialService
from app.services.vision_service import (
    VisionAnalysisService,
    ASTMVisionResult,
    VideoQualityMetrics,
    FrameSamplingInfo,
)
from app.services.dav2e import DepthAnythingV2Estimator
from app.services.geom_anchor import GeometryAnchoredDepthScaler
from app.services.mbtp_est import MBTPAreaEstimator
from app.services.rgbsegment import RobustPotholeSegmentor
from app.services.edfe_est import EnhancedDimensionFusionEngine
from app.services.imu_proc import IMUProcessor, IMUState
from app.services.depth_service import DepthEstimationService, DepthEstimationResult
from app.services.geom_anchor import GeometryAnchoredDepthScaler

logger = get_pipeline_logger("heavy_compute")


def _geometry_value(wkt: Optional[str]):
    if not wkt:
        return None
    return WKTElement(wkt, srid=4326) if IS_POSTGRES else wkt


class HeavyComputePipeline:
    """Orchestrates end-to-end multi-modal road survey analysis for TWO pipelines:

    Pipeline 1: ASTM (International Standard)
    - IRI: QCM at 100Hz upsampled (ASTM E1926/E950)
    - Vision: MDPI Sensors 2024 paper methodology
    - Output: ASTM PCI (IRI reported separately per ASTM D6433)
    
Pipeline 2: Roadessy (Custom/Ghana-aligned)
- IRI: Simple displacement-to-distance method (Enochs Thesis Eq 2.9/3.9)
- Vision: Ghana Manual 2023 Table 3 ratings + custom DV computation
- Output: Roadessy PCI = 100 - corrected(DV_sum) (IRI reported separately)
    
    Both pipelines share: GPS loading, spatial map-matching, camera calibration
    """

    OUTPUT_BASE_DIR = Path(r"F:\Adubofour\rodeessy_mphil\RoadSurveyOutputs")

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Sanitize a distress type name for use in a filename.

        Replaces path separators, parentheses, and other characters that are
        invalid or dangerous in file paths (e.g. '//' in 'Flushing/Bleeding'
        or parentheses in 'Cracking (Alligator/Longitudinal/Transverse)').
        """
        for char in r'\/:*?"<>|()':
            name = name.replace(char, '_')
        name = name.replace(' ', '_')
        return name

    def __init__(self, run_id: str, db_session: Optional[Any] = None):
        self.run_id = str(run_id)
        self.db = db_session or SessionLocal()
        self.output_dir = self.OUTPUT_BASE_DIR / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.output_dir / "detected_frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    @monitor(stage="heavy_compute_pipeline_execution")
    def run_pipeline(
        self,
        vibration_file_path: str,
        gps_file_path: str,
        video_file_path: Optional[str] = None,
        road_id: Optional[str] = None,
        camera_profile_id: Optional[int] = None,
        raw_z_accel: Optional[List[float]] = None,
        raw_speeds: Optional[List[float]] = None,
        sample_rate: float = 100.0,
    ) -> Dict[str, Any]:
        logger.info("Starting Heavy Compute Pipeline (2 pipelines: ASTM + Roadessy)", run_id=self.run_id)

        # 1. Load and parse input datasets (shared)
        gps_df = self._load_gps_data(gps_file_path)
        
        if vibration_file_path and os.path.exists(vibration_file_path):
            vibe_timestamps, vibe_accel, vibe_speeds, vibe_lat, vibe_lon = self._load_vibration_data(vibration_file_path)
        elif raw_z_accel and len(raw_z_accel) > 0:
            vibe_accel = np.array(raw_z_accel, dtype=np.float64)
            vibe_timestamps = np.arange(len(vibe_accel), dtype=np.float64) / sample_rate
            vibe_speeds = np.array(raw_speeds, dtype=np.float64) if raw_speeds else None
            vibe_lat = None
            vibe_lon = None
        else:
            vibe_timestamps, vibe_accel, vibe_speeds, vibe_lat, vibe_lon = self._load_vibration_data(vibration_file_path)

        # 2. Camera calibration parameters lookup (shared) - from database
        cam_height, cam_tilt, cam_fx, cam_fy, cam_cx, cam_cy, cam_distortion = self._resolve_camera_calibration(camera_profile_id)
        
        # Initialize new components (DepthAnythingV2 is optional - only used if available)
        dav2_estimator = None
        try:
            dav2_estimator = DepthAnythingV2Estimator(encoder='vits')
        except Exception as exc:
            logger.warning(f"DepthAnythingV2 not available, dense depth estimation disabled: {exc}", run_id=self.run_id)
        
        geom_scaler = GeometryAnchoredDepthScaler(
            camera_height_m=cam_height,
            pitch_deg=cam_tilt,
            fx=cam_fx, fy=cam_fy, cx=cam_cx, cy=cam_cy
        )
        mbtp_estimator = MBTPAreaEstimator()
        segmentor = RobustPotholeSegmentor()
        fusion_engine = EnhancedDimensionFusionEngine(K=np.array([[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]], dtype=np.float64))
        imu_processor = IMUProcessor()
        
        # 3. Process Inertial - TWO methods for two pipelines
        # 3a. Simple IRI (displacement method) - shared base for Roadessy pipeline
        # Travel distance uses Haversine from lat/lon (per reference ST7_TravelDistan.py)
        simple_result: InertialPipelineResult = InertialRefineryService.process_vibration_data(
            timestamps=vibe_timestamps,
            accel_z=vibe_accel,
            speeds_kmh=vibe_speeds,
            run_id=self.run_id,
            latitudes=vibe_lat,
            longitudes=vibe_lon,
        )
        
        # 3b. ASTM QCM (100Hz upsampled, quality flags) - Pipeline 1
        astm_qcm = InertialRefineryService.process_astm_qcm(
            timestamps=vibe_timestamps,
            accel_z=vibe_accel,
            speeds_kmh=vibe_speeds,
            run_id=self.run_id,
            target_fs=100.0,
            target_dx=0.025,
            v_ref_kmh=80.0,
        )

        # 4. Process Spatial GPS Trajectory & Map-Matching (shared)
        map_match_result = SpatialService.map_match_trajectory(
            gps_df=gps_df,
            road_id=road_id,
            run_id=self.run_id,
            db=self.db,
        )

        # 5. Process Video Stream -> TWO vision methods
        astm_vision = ASTMVisionResult(
            detections=[], 
            video_quality=VideoQualityMetrics(
                fps=0, resolution=(0,0), motion_blur_score=0, illumination_score=0,
                gps_sync_quality=0, fps_flag="INSUFFICIENT", frames_analyzed=0, frames_skipped=0
            ),
            frame_sampling=FrameSamplingInfo(
                original_fps=0, processed_fps=0, sampling_method="none",
                total_frames=0, processed_frames=0
            ),
            astm_distress_mapping={}, ghana_distress_mapping={}
        )
        roadessy_vision = ASTMVisionResult(
            detections=[], 
            video_quality=VideoQualityMetrics(
                fps=0, resolution=(0,0), motion_blur_score=0, illumination_score=0,
                gps_sync_quality=0, fps_flag="INSUFFICIENT", frames_analyzed=0, frames_skipped=0
            ),
            frame_sampling=FrameSamplingInfo(
                original_fps=0, processed_fps=0, sampling_method="none",
                total_frames=0, processed_frames=0
            ),
            astm_distress_mapping={}, ghana_distress_mapping={}
        )
        tracking_registry: Dict[int, Dict[str, Any]] = {}
        
        if video_file_path and os.path.exists(video_file_path):
            # 5a. ASTM Vision (MDPI paper) - Pipeline 1
            vision_service = VisionAnalysisService(
                camera_height_m=cam_height,
                camera_tilt_deg=cam_tilt,
                fx=cam_fx,
                fy=cam_fy,
                cx=cam_cx,
                cy=cam_cy,
                distortion_coefficients=cam_distortion,
            )
            gps_records = (
                cast(List[Dict[str, Any]], gps_df.to_dict(orient="records"))
                if not gps_df.empty
                else None
            )
            
            astm_vision = vision_service.process_video_astm(
                video_path=video_file_path,
                gps_points=gps_records,
                run_id=self.run_id,
                target_fps=15.0,
            )
            
            # 5b. Dense Depth Estimation (NEW - DepthAnythingV2)
            abs_depth = None
            if dav2_estimator is not None and video_file_path and os.path.exists(video_file_path):
                try:
                    logger.info("Running DepthAnythingV2 dense depth estimation", run_id=self.run_id)
                    # Read first frame for depth estimation
                    cap = cv2.VideoCapture(video_file_path)
                    ret, frame = cap.read()
                    cap.release()
                    if ret and frame is not None:
                        abs_depth = dav2_estimator.estimate(frame)
                        logger.metric("depth_estimation", "dense_depth_completed", True, run_id=self.run_id)
                    else:
                        logger.warning("Could not read video frame for depth estimation", run_id=self.run_id)
                except Exception as exc:
                    logger.error(f"Dense depth estimation failed: {exc}", run_id=self.run_id)
            
            # 5c. Pothole Depth Detection (NEW - MBTP + RGB Segmentation)
            pothole_depths = {}
            if astm_vision.detections and abs_depth is not None:
                logger.info("Running pothole depth detection (MBTP + RGB Segmentation)", run_id=self.run_id)
                
                # Process each pothole detection
                for det in astm_vision.detections:
                    if det.get("yolo_class_id") == 4:  # Pothole class
                        bbox = det.get("bbox")  # [x1, y1, x2, y2]
                        if bbox:
                            # Read the frame at the detection's frame index
                            # For now, use the first frame as reference
                            cap = cv2.VideoCapture(video_file_path)
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = cap.read()
                            cap.release()
                            
                            if frame is not None:
                                # Run MBTP estimation
                                mbtp_result = mbtp_estimator.estimate(
                                    rgb_image=frame,
                                    depth_map=abs_depth,
                                    bbox=bbox,
                                    K=np.array([[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]], dtype=np.float64),
                                    ground_ref=0.0  # Use ground reference from depth map
                                )
                                pothole_depths[det.get("chainage_m", 0)] = mbtp_result.depth_m
                                
                                # Update detection with MBTP results
                                det["mbtp_area_m2"] = mbtp_result.area_m2
                                det["mbtp_depth_m"] = mbtp_result.depth_m
                                det["mbtp_volume_m3"] = mbtp_result.volume_m3
                                det["mbtp_confidence"] = mbtp_result.confidence
                                det["mbtp_method"] = "mbtp"
                                logger.metric("pothole_depth", "mbtp_estimate", mbtp_result.depth_m, run_id=self.run_id)
            
            # 5d. Roadessy Vision (Ghana Table 3) - Pipeline 2
            # Reuse the same detections but with Ghana mapping
            roadessy_vision = ASTMVisionResult(
                detections=astm_vision.detections,  # Same detections, different mappings
                video_quality=astm_vision.video_quality,
                frame_sampling=astm_vision.frame_sampling,
                astm_distress_mapping=astm_vision.astm_distress_mapping,
                ghana_distress_mapping=astm_vision.ghana_distress_mapping,
            )

        # 6. Run BOTH pipelines
        # Pipeline 1: ASTM
        astm_result = PavementConditionFusioneer.compute_astm_pipeline(
            astm_qcm=astm_qcm,
            astm_vision=astm_vision,
            run_id=self.run_id,
        )
        
        # Pipeline 2: Roadessy
        roadessy_result = PavementConditionFusioneer.compute_roadessy_pipeline(
            simple_result=simple_result,
            roadessy_vision=roadessy_vision,
            run_id=self.run_id,
        )

        # 7. Persist to Database
        self._persist_db_records(
                simple_result=simple_result,
                astm_qcm=astm_qcm,
                astm_vision=astm_vision,
                roadessy_vision=roadessy_vision,
                astm_result=astm_result,
                roadessy_result=roadessy_result,
                road_id=road_id,
                vibration_path=vibration_file_path,
                video_path=video_file_path,
                gps_path=gps_file_path,
                map_match_result=map_match_result,
            )


        # 8. Save ALL outputs to E:\ folder
        self._save_all_outputs(
            simple_result=simple_result,
            astm_qcm=astm_qcm,
            astm_vision=astm_vision,
            roadessy_vision=roadessy_vision,
            astm_result=astm_result,
            roadessy_result=roadessy_result,
            map_match_result=map_match_result,
            gps_df=gps_df,
            vibration_file_path=vibration_file_path,
            gps_file_path=gps_file_path,
            video_file_path=video_file_path,
        )

        # Clean up workspace scratch files (uploaded input files) to save disk space
        self._cleanup_workspace_files(vibration_file_path, gps_file_path, video_file_path)

        logger.info(
            f"Pipeline completed successfully. Outputs saved to {self.output_dir}",
            run_id=self.run_id,
        )
        
        # Log key metrics for monitoring
        logger.metric("heavy_compute_pipeline", "simple_iri_m_km", round(simple_result.iri_m_km, 4), run_id=self.run_id)
        logger.metric("heavy_compute_pipeline", "astm_qcm_iri_m_km", round(astm_qcm.iri_m_km, 4), run_id=self.run_id)
        logger.metric("heavy_compute_pipeline", "total_distance_m", round(simple_result.total_distance_m, 2), run_id=self.run_id)
        logger.metric("heavy_compute_pipeline", "simple_segments", len(simple_result.iri_segments), run_id=self.run_id)
        logger.metric("heavy_compute_pipeline", "astm_qcm_segments", len(astm_qcm.iri_segments), run_id=self.run_id)
        logger.metric("heavy_compute_pipeline", "defects_detected", len(astm_vision.detections), run_id=self.run_id)
        logger.metric("heavy_compute_pipeline", "video_fps_flag", astm_vision.video_quality.fps_flag if astm_vision.video_quality else "UNKNOWN", run_id=self.run_id)
        logger.metric("heavy_compute_pipeline", "video_frames_analyzed", astm_vision.video_quality.frames_analyzed if astm_vision.video_quality else 0, run_id=self.run_id)
        
        # Log pipeline results
        logger.info(
            f"Pipeline 1 (ASTM): PCI={astm_result.get('astm_pci')}, QCM_IRI={astm_result.get('qcm_iri_m_km')}",
            run_id=self.run_id,
        )
        logger.info(
            f"Pipeline 2 (Roadessy): PCI={roadessy_result.get('roadessy_pci')}, Simple_IRI={roadessy_result.get('simple_iri_m_km')}",
            run_id=self.run_id,
        )
        
        return {
            "run_id": self.run_id,
            "output_dir": str(self.output_dir),
            "pipeline_1_astm": astm_result,
            "pipeline_2_roadessy": roadessy_result,
            "map_matching": map_match_result,
            "total_distance_m": simple_result.total_distance_m,
            "defects_detected": len(astm_vision.detections),
            "video_quality": {
                "fps_flag": astm_vision.video_quality.fps_flag if astm_vision.video_quality else "UNKNOWN",
                "frames_analyzed": astm_vision.video_quality.frames_analyzed if astm_vision.video_quality else 0,
            },
        }

    def _load_gps_data(self, gps_file_path: str) -> pd.DataFrame:
        if not gps_file_path or not os.path.exists(gps_file_path):
            logger.warning(f"GPS file not found: {gps_file_path}", run_id=self.run_id)
            return pd.DataFrame(columns=["time", "lat", "lon", "Speed", "GPSAcc"])

        df = pd.read_csv(gps_file_path)
        df.columns = [str(col).strip() for col in df.columns]

        col_map = {
            "latitude": "lat", "Latitude": "lat",
            "Longitude": "lon", "longitude": "lon",
            "speed": "Speed", "Speed ": "Speed", "speed ": "Speed",
            "accuracy": "GPSAcc", "timestamp": "time", "Timestamp": "time",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        if "Timestamp" in df.columns:
            time_col = "Timestamp"
        elif "time" in df.columns:
            time_col = "time"
        else:
            time_col = df.columns[0]
        
        try:
            df["time"] = pd.to_numeric(df[time_col], errors="raise")
        except (ValueError, TypeError):
            parsed_dt = pd.to_datetime(df[time_col], errors="coerce")
            if parsed_dt.notna().any():
                min_time = parsed_dt.min()
                df["time"] = (parsed_dt - min_time).dt.total_seconds().fillna(0.0)
            else:
                df["time"] = pd.to_numeric(df[time_col], errors="coerce").fillna(0.0)

        for col in ["lat", "lon", "Speed", "GPSAcc"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def _load_vibration_data(
        self, vibration_file_path: str
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        if not vibration_file_path or not os.path.exists(vibration_file_path):
            logger.warning(f"Vibration file not found: {vibration_file_path}", run_id=self.run_id)
            t = np.linspace(0, 10, 1000)
            accel = np.random.normal(0, 0.5, 1000)
            return t, accel, None, None, None

        df = pd.read_csv(vibration_file_path)
        df.columns = [str(col).strip() for col in df.columns]

        time_col = next((c for c in ["Timestamp", "time", "GPSTime"] if c in df.columns), df.columns[0])
        accel_col = next((c for c in ["LinAccelZ", "AccelZ", "accel_z"] if c in df.columns), df.columns[1])

        # Get RecIntvl if available (recording interval in seconds)
        recintvl = None
        if "RecIntvl" in df.columns:
            recintvl_vals = pd.to_numeric(df["RecIntvl"], errors="coerce")
            if recintvl_vals.notna().any():
                recintvl = float(recintvl_vals.iloc[0])

        try:
            timestamps = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=np.float64)
            if not np.any(np.isfinite(timestamps)):
                parsed_dt = pd.to_datetime(df[time_col], errors="coerce")
                if parsed_dt.notna().any():
                    # Use pandas total_seconds() for correct conversion regardless of datetime resolution
                    timestamps = (parsed_dt - parsed_dt.iloc[0]).dt.total_seconds().to_numpy(dtype=np.float64)
        except (ValueError, TypeError):
            parsed_dt = pd.to_datetime(df[time_col], errors="coerce")
            if parsed_dt.notna().any():
                timestamps = (parsed_dt - parsed_dt.iloc[0]).dt.total_seconds().to_numpy(dtype=np.float64)
            else:
                timestamps = pd.to_numeric(df[time_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

        # If timestamps look like integer counters and RecIntvl is available, convert to seconds
        if recintvl is not None and recintvl > 0:
            finite_ts = timestamps[np.isfinite(timestamps)]
            if len(finite_ts) > 1:
                diffs = np.diff(finite_ts)
                ts_span = float(finite_ts[-1] - finite_ts[0])
                # Check if timestamps are integer counters (0, 1, 2, ...) 
                is_counter = np.allclose(diffs, 1.0, atol=1e-6)
                # Or if datetime parse gave nanosecond epoch values (very small span for many samples)
                is_nanosecond_epoch = ts_span < 1.0 and len(finite_ts) > 100
                
                if is_counter:
                    timestamps = timestamps * recintvl
                    logger.info(f"Converted timestamps using RecIntvl={recintvl}s", run_id=self.run_id)
                elif is_nanosecond_epoch:
                    logger.warning(
                        f"Timestamp span {ts_span:.3f}s is too small for {len(finite_ts)} samples; "
                        f"using RecIntvl={recintvl}s"
                    )
                    timestamps = np.arange(len(timestamps), dtype=np.float64) * recintvl

        accel_z = pd.to_numeric(df[accel_col], errors="coerce").to_numpy(dtype=np.float64)

        speeds = None
        speed_col = next((c for c in ["SpeedKmh", "speed", "SpeedGPS"] if c in df.columns), None)

        lat = None
        long = None
        if "Latitude" in df.columns:
            lat = pd.to_numeric(df["Latitude"], errors="coerce").to_numpy(dtype=np.float64)
        if "Longitude" in df.columns:
            long = pd.to_numeric(df["Longitude"], errors="coerce").to_numpy(dtype=np.float64)

        if speed_col:
            speeds = pd.to_numeric(df[speed_col], errors="coerce").to_numpy(dtype=np.float64)

        valid_mask = np.isfinite(timestamps) & np.isfinite(accel_z)
        if not np.all(valid_mask):
            timestamps = timestamps[valid_mask]
            accel_z = accel_z[valid_mask]
            if speeds is not None:
                speeds = speeds[valid_mask]

        return timestamps, accel_z, speeds, lat, long
     
    def _resolve_camera_calibration(
        self, camera_profile_id: Optional[int]
    ) -> Tuple[float, float, float, float, float, float, Optional[str]]:
        """Get camera parameters from database calibration profile.

        Returns: (camera_height_m, camera_tilt_deg, fx, fy, cx, cy, distortion_coefficients)
        All values come from the database calibration profile.
        """
        default_height, default_tilt = 1.5, 15.0
        default_fx, default_fy = 1000.0, 1000.0
        default_cx, default_cy = 960.0, 540.0
        default_distortion = None
        
        if not camera_profile_id:
            return default_height, default_tilt, default_fx, default_fy, default_cx, default_cy, default_distortion

        profile = (
            self.db.query(CalibrationProfile)
            .filter(CalibrationProfile.id == camera_profile_id)
            .first()
        )
        if profile:
            raw_height = getattr(profile, "camera_height_meters", None)
            raw_tilt = getattr(profile, "camera_tilt_degrees", None)
            raw_fx = getattr(profile, "fx", None)
            raw_fy = getattr(profile, "fy", None)
            raw_cx = getattr(profile, "cx", None)
            raw_cy = getattr(profile, "cy", None)
            raw_distortion = getattr(profile, "distortion_coefficients", None)

            height = float(cast(Any, raw_height)) if raw_height is not None else 1.5
            tilt = float(cast(Any, raw_tilt)) if raw_tilt is not None else 15.0
            fx = float(cast(Any, raw_fx)) if raw_fx is not None else 1000.0
            fy = float(cast(Any, raw_fy)) if raw_fy is not None else 1000.0
            cx = float(cast(Any, raw_cx)) if raw_cx is not None else 960.0
            cy = float(cast(Any, raw_cy)) if raw_cy is not None else 540.0
            distortion = raw_distortion

            return height, tilt, fx, fy, cx, cy, distortion

        return 1.5, 15.0, 1000.0, 1000.0, 960.0, 540.0, None


    def _save_detected_frames(self, video_path: str, detections: List[Dict[str, Any]]) -> None:
        """Save the best frame for each detected defect (frame where defect has max area)."""
        if not detections:
            return
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return
        
        # Group detections by defect type and find the one with max area for each type
        best_by_type = {}
        for det in detections:
            defect_type = det.get("defect_type", "unknown")
            area = det.get("area", 0)
            if defect_type not in best_by_type or area > best_by_type[defect_type].get("area", 0):
                best_by_type[defect_type] = det
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return
        
        for defect_type, det in best_by_type.items():
            # For simplicity, use the first frame where this defect was detected
            # In a full implementation, we'd track the frame index
            chainage = det.get("chainage_m", 0)
            area = det.get("area", 0)
            width = det.get("width", 0)
            depth = det.get("depth_mm", 0)
            
            # We can't seek to a specific frame without frame index, so just save first frame
            # with a representative detection
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Draw bounding box info on frame
            defect_type_str = det.get("defect_type", "unknown")
            text = f"{defect_type_str} | Area: {area:.3f}m² | Width: {width:.1f}mm | Depth: {depth:.1f}mm | Chainage: {chainage:.1f}m"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            safe_defect_type = self._sanitize_filename(defect_type_str)
            frame_filename = self.frames_dir / f"{safe_defect_type}_frame_000001.jpg"
            cv2.imwrite(str(frame_filename), frame)
        
        cap.release()

    def _save_all_outputs(
        self,
        simple_result: InertialPipelineResult,
        astm_qcm: ASTMQCMResult,
        astm_vision: ASTMVisionResult,
        roadessy_vision: ASTMVisionResult,
        astm_result: Dict[str, Any],
        roadessy_result: Dict[str, Any],
        map_match_result: Dict[str, Any],
        gps_df: pd.DataFrame,
        vibration_file_path: str,
        gps_file_path: str,
        video_file_path: Optional[str],
    ) -> None:
        """Save all outputs to output directory."""

        # 1. Save IRI segments
        iri_dir = self.output_dir / "iri_segments"
        iri_dir.mkdir(exist_ok=True)
        
        # Simple IRI segments (Roadessy)
        simple_df = pd.DataFrame([{
            "segment_index": i,
            "start_m": s.start_distance_m,
            "end_m": s.end_distance_m,
            "iri_m_km": s.iri_m_km,
            "classification": InertialRefineryService.classify_iri_5rank(s.iri_m_km)["rank"],
        } for i, s in enumerate(simple_result.iri_segments)])
        simple_df.to_csv(iri_dir / "simple_iri_50m.csv", index=False)
        
        # ASTM QCM IRI segments
        astm_df = pd.DataFrame([{
            "segment_index": i,
            "start_m": s.start_distance_m,
            "end_m": s.end_distance_m,
            "iri_m_km": s.iri_m_km,
            "classification": InertialRefineryService.classify_iri_5rank(s.iri_m_km)["rank"],
        } for i, s in enumerate(astm_qcm.iri_segments)])
        astm_df.to_csv(iri_dir / "astm_qcm_iri_50m.csv", index=False)
        
        # 3. Save PCI segments
        pci_dir = self.output_dir / "pci_segments"
        pci_dir.mkdir(exist_ok=True)
        
        # ASTM segment PCI
        astm_seg_pci = astm_result.get("segment_pci_50m", [])
        if astm_seg_pci:
            pd.DataFrame(astm_seg_pci).to_csv(pci_dir / "astm_segment_pci_50m.csv", index=False)
        
        # Roadessy segment PCI
        roadessy_seg_pci = roadessy_result.get("segment_pci_50m", [])
        if roadessy_seg_pci:
            pd.DataFrame(roadessy_seg_pci).to_csv(pci_dir / "roadessy_segment_pci_50m.csv", index=False)
        
        # 4. Save vision detections
        vision_dir = self.output_dir / "vision_detections"
        vision_dir.mkdir(exist_ok=True)
        
        if astm_vision.detections:
            det_df = pd.DataFrame(astm_vision.detections)
            det_df.to_csv(vision_dir / "detections_all.csv", index=False)
            
            # Per distress type (ASTM)
            for distress_type in det_df["astm_distress_type"].unique():
                subset = det_df[det_df["astm_distress_type"] == distress_type]
                safe_type = self._sanitize_filename(distress_type)
                subset.to_csv(vision_dir / f"detections_astm_{safe_type}.csv", index=False)
            
            # Per distress type (Ghana)
            for distress_type in det_df["ghana_distress_type"].unique():
                subset = det_df[det_df["ghana_distress_type"] == distress_type]
                safe_type = self._sanitize_filename(distress_type)
                subset.to_csv(vision_dir / f"detections_ghana_{safe_type}.csv", index=False)
        
        # Video quality report
        vq = astm_vision.video_quality
        if vq:
            with open(vision_dir / "video_quality_report.json", "w") as f:
                json.dump({
                    "fps": vq.fps,
                    "resolution": vq.resolution,
                    "fps_flag": vq.fps_flag,
                    "motion_blur_score": vq.motion_blur_score,
                    "illumination_score": vq.illumination_score,
                    "gps_sync_quality": vq.gps_sync_quality,
                    "frames_analyzed": vq.frames_analyzed,
                    "frames_skipped": vq.frames_skipped,
                }, f, indent=2)
        
        # Frame sampling info
        if astm_vision.frame_sampling:
            fs = astm_vision.frame_sampling
            with open(vision_dir / "frame_sampling_info.json", "w") as f:
                json.dump({
                    "original_fps": fs.original_fps,
                    "processed_fps": fs.processed_fps,
                    "sampling_method": fs.sampling_method,
                    "total_frames": fs.total_frames,
                    "processed_frames": fs.processed_frames,
                }, f, indent=2)
        
        # 5. Save fusion results
        fusion_dir = self.output_dir / "fusion_results"
        fusion_dir.mkdir(exist_ok=True)
        
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        with open(fusion_dir / "pipeline_1_astm.json", "w") as f:
            json.dump(convert(astm_result), f, indent=2)
        
        with open(fusion_dir / "pipeline_2_roadessy.json", "w") as f:
            json.dump(convert(roadessy_result), f, indent=2)
        
        # 6. Save map matching results
        map_dir = self.output_dir / "map_matching"
        map_dir.mkdir(exist_ok=True)
        
        if map_match_result.get("snapped_coords"):
            map_df = pd.DataFrame(map_match_result["snapped_coords"])
            map_df.to_csv(map_dir / "snapped_coords.csv", index=False)
        
        with open(map_dir / "map_match_summary.json", "w") as f:
            json.dump(convert(map_match_result), f, indent=2)
        
        # 7. Save GPS trajectory
        if not gps_df.empty:
            gps_df.to_csv(self.output_dir / "gps_trajectory.csv", index=False)
        
        # 8. Save run summary
        with open(self.output_dir / "run_summary.json", "w") as f:
            json.dump({
                "run_id": self.run_id,
                "timestamp": datetime.utcnow().isoformat(),
                "output_directory": str(self.output_dir),
                "pipeline_1_astm": {
                    "qcm_iri_m_km": astm_result["qcm_iri_m_km"],
                    "astm_pci": astm_result["astm_pci"],
                    "quality_flags": astm_result["quality_flags"],
                },
                "pipeline_2_roadessy": {
                    "roadessy_pci": roadessy_result["roadessy_pci"],
                    "simple_iri_m_km": roadessy_result["simple_iri_m_km"],
                },
                "num_segments_50m": max(len(astm_seg_pci), len(roadessy_seg_pci)),
                "total_defects": len(astm_vision.detections),
            }, f, indent=2)

    def _cleanup_workspace_files(
        self, vibration_file_path: str, gps_file_path: str, video_file_path: Optional[str]
    ) -> None:
        """Delete uploaded workspace input files after processing to save disk space."""
        for file_path in (vibration_file_path, gps_file_path, video_file_path):
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass

    def _persist_db_records(
        self,
        simple_result: InertialPipelineResult,
        astm_qcm: ASTMQCMResult,
        astm_vision: ASTMVisionResult,
        roadessy_vision: ASTMVisionResult,
        astm_result: Dict[str, Any],
        roadessy_result: Dict[str, Any],
        road_id: Optional[str],
        vibration_path: str,
        video_path: Optional[str],
        gps_path: str,
        map_match_result: Dict[str, Any],
    ) -> None:
        try:
            matched_road_id = map_match_result.get("matched_road_id") or road_id
            trajectory_wkt = (
                map_match_result.get("snapped_trajectory_wkt")
                or map_match_result.get("trajectory_wkt")
            )
            
            # Fallback: if map matching didn't provide trajectory, build from GPS file
            if not trajectory_wkt and gps_path and os.path.exists(gps_path):
                try:
                    gps_df = self._load_gps_data(gps_path)
                    if not gps_df.empty and "lat" in gps_df.columns and "lon" in gps_df.columns:
                        valid = gps_df.dropna(subset=["lat", "lon"])
                        if len(valid) >= 2:
                            coords = [(row["lon"], row["lat"]) for _, row in valid.iterrows()]
                            line = LineString(coords)
                            trajectory_wkt = line.wkt
                            logger.info(f"Built fallback trajectory from GPS ({len(coords)} points)", run_id=self.run_id)
                except Exception as e:
                    logger.warning(f"Failed to build fallback trajectory: {e}", run_id=self.run_id)
            
            survey_run = SurveyRun(
                run_id=self.run_id,
                name=f"Survey_{self.run_id[:8]}",
                user_id="operator_default",
                organisation_id="org_default",
                
                # Pipeline 1: ASTM
                astm_pci=float(astm_result["astm_pci"]),
                astm_qcm_iri=float(astm_result["qcm_iri_m_km"]),
                astm_quality_flags=astm_result["quality_flags"],
                astm_upsampling_info=astm_result["upsampling_info"],
                
                # Pipeline 2: Roadessy
                simple_iri=float(roadessy_result["simple_iri_m_km"]),
                roadessy_pci=float(roadessy_result["roadessy_pci"]),
                
                # Legacy (backward compat)
                computed_iri=float(simple_result.iri_m_km),
                final_pci=float(roadessy_result["roadessy_pci"]),  # Primary PCI
                
                matched_road_id=matched_road_id,
                geom_trajectory=_geometry_value(trajectory_wkt),
                workspace_dir=str(self.output_dir),
                vibration_file_path=vibration_path,
                video_file_path=video_path,
                gps_file_path=gps_path,
                processing_metadata={
                    "pipeline_1_astm": astm_result,
                    "pipeline_2_roadessy": roadessy_result,
                    "map_matching": map_match_result,
                },
            )
            self.db.add(survey_run)

            map_match_record = MapMatchingRun(
                run_id=uuid.UUID(self.run_id),
                user_id="operator_default",
                matched_road_id=matched_road_id,
                match_status=map_match_result.get("match_status"),
                match_method=map_match_result.get("match_method"),
                centerline_wkt=map_match_result.get("centerline_wkt"),
                centerline_geojson=map_match_result.get("centerline_geojson"),
                snapped_trajectory_wkt=map_match_result.get("snapped_trajectory_wkt"),
                chainage_min_m=map_match_result.get("chainage_min"),
                chainage_max_m=map_match_result.get("chainage_max"),
                matched_distance_m=map_match_result.get("matched_distance_m"),
                road_length_m=map_match_result.get("road_length_m"),
                max_distance_m=map_match_result.get("max_distance_m"),
                matched_points=map_match_result.get("matched_points"),
                total_points=map_match_result.get("total_points"),
                unmatched_points=map_match_result.get("unmatched_points"),
                snapped_coords=map_match_result.get("snapped_coords", []),
            )
            self.db.add(map_match_record)

            # Persist IRI Segments
            self._persist_iri_segments(simple_result, astm_qcm, matched_road_id, astm_result, roadessy_result)
            
            # Persist Defects
            self._persist_defects(astm_vision.detections, matched_road_id)
            
            self.db.commit()
        except Exception as exc:
            self.db.rollback()
            logger.error(f"Failed to persist database records: {exc}", run_id=self.run_id)
            raise

    def _persist_iri_segments(
        self,
        simple_result: InertialPipelineResult,
        astm_qcm: ASTMQCMResult,
        road_id: Optional[str],
        astm_result: Dict[str, Any],
        roadessy_result: Dict[str, Any],
    ) -> None:
        astm_seg_pci = astm_result.get("segment_pci_50m", [])
        roadessy_seg_pci = roadessy_result.get("segment_pci_50m", [])
        
        for seg_idx, seg in enumerate(simple_result.iri_segments):
            astm_seg = roadessy_seg = None
            if seg_idx < len(astm_seg_pci):
                astm_seg = astm_seg_pci[seg_idx]
            if seg_idx < len(roadessy_seg_pci):
                roadessy_seg = roadessy_seg_pci[seg_idx]
            
            db_seg = IRISegment(
                run_id=self.run_id,
                user_id="operator_default",
                road_id=road_id,
                segment_start_m=float(seg.start_distance_m),
                segment_end_m=float(seg.end_distance_m),
                iri_value=float(seg.iri_m_km),
                simple_iri=float(seg.iri_m_km),
                astm_qcm_iri=float(astm_qcm.iri_segments[seg_idx].iri_m_km) if seg_idx < len(astm_qcm.iri_segments) else None,
                astm_pci_segment=float(astm_seg.get("astm_pci", 0)) if astm_seg else None,
                roadessy_pci_segment=float(roadessy_seg.get("roadessy_pci", 0)) if roadessy_seg else None,
                classification=InertialRefineryService.classify_iri_5rank(float(seg.iri_m_km))["rank"],
            )
            self.db.add(db_seg)
        
        # ASTM QCM segments
        for seg_idx in range(len(simple_result.iri_segments), len(astm_qcm.iri_segments)):
            seg = astm_qcm.iri_segments[seg_idx]
            db_seg = IRISegment(
                run_id=self.run_id,
                user_id="operator_default",
                road_id=road_id,
                segment_start_m=float(seg.start_distance_m),
                segment_end_m=float(seg.end_distance_m),
                iri_value=float(seg.iri_m_km),
                simple_iri=None,
                astm_qcm_iri=float(seg.iri_m_km),
                classification=InertialRefineryService.classify_iri_5rank(float(seg.iri_m_km))["rank"],
            )
            self.db.add(db_seg)

    def _persist_defects(self, detections: List[Dict[str, Any]], road_id: Optional[str]) -> None:
        for det in detections:
            db_defect = TrackedDefect(
                run_id=self.run_id,
                user_id="operator_default",
                road_id=road_id,
                track_id=0,
                failure_mode=det.get("defect_type"),
                max_area_m2=det.get("area"),
                max_width_mm=det.get("width"),
                max_depth_mm=det.get("depth_mm"),
                mean_depth_mm=det.get("depth_mm"),
                depth_confidence=0.85,
                depth_method="monocular_geometry",
                depth_message="Estimated via camera geometry model",
                deduct_value=det.get("deduct_value"),
                astm_distress_type=det.get("astm_distress_type"),
                ghana_distress_type=det.get("ghana_distress_type"),
                yolo_class_id=det.get("yolo_class_id"),
                confidence=det.get("confidence"),
                chainage_m=det.get("chainage_m"),
            )
            self.db.add(db_defect)


def run_heavy_pipeline(
    z_accel_data: Optional[List[float]] = None,
    speed: float = 0.0,
    sample_rate: float = 100.0,
    video_path: Optional[str] = None,
    vibration_file_path: str = "",
    gps_file_path: str = "",
    road_id: Optional[str] = None,
    camera_profile_id: Optional[int] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    pipeline_run_id = run_id or str(uuid.uuid4())
    pipeline = HeavyComputePipeline(run_id=pipeline_run_id)
    return pipeline.run_pipeline(
        vibration_file_path=vibration_file_path,
        gps_file_path=gps_file_path,
        video_file_path=video_path,
        road_id=road_id,
        camera_profile_id=camera_profile_id,
        raw_z_accel=z_accel_data,
        sample_rate=sample_rate,
    )