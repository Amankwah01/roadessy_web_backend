# app/pipelines/heavy_compute_pipeline.py
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import cv2
import numpy as np
import pandas as pd

from geoalchemy2 import WKTElement
from shapely.geometry import LineString

from app.core.camera import resolve_camera_geometry
from app.core.config import ASTM_QCM_IRI_ENABLED, astm_track_state
from app.core.logger import get_pipeline_logger, monitor
from app.database.session import IS_POSTGRES, SessionLocal
from app.models.calibration import CalibrationProfile
from app.models.survey import IRISegment, MapMatchingRun, SurveyRun, TrackedDefect
from app.services.fusion import PavementConditionFusioneer
from app.services.fusion.alignment import build_inertial_track
from app.services.inertial import (
    InertialPipelineResult,
    ASTMQCMResult,
    InertialRefineryService,
)
from app.services.gps.service import UniversalGPSService
from app.services.spatial import SpatialService
from app.services.vision import (
    VisionAnalysisService,
    ASTMVisionResult,
    VideoQualityMetrics,
    FrameSamplingInfo,
)
from app.services.vision.depth import (
    DepthAnythingV2Estimator,
    GeometryAnchoredDepthScaler,
    MBTPAreaEstimator,
    create_depth_scaler,
)

logger = get_pipeline_logger("heavy_compute")


def _geometry_value(wkt: Optional[str]):
    if not wkt:
        return None
    return WKTElement(wkt, srid=4326) if IS_POSTGRES else wkt


def _coerce_uuid(value: Optional[str]) -> Optional[uuid.UUID]:
    """Return a UUID when the value parses as one, otherwise None."""
    if not value:
        return None
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return None


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

    OUTPUT_BASE_DIR = Path(
        os.getenv("ROADESSY_OUTPUT_DIR", r"F:\Adubofour\rodeessy_mphil\RoadSurveyOutputs")
    )

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
        self.defect_images_dir = Path(
            os.getenv("ROADESSY_DEFECT_IMAGES_DIR", str(self.output_dir / "defect_images"))
        )
        self.defect_images_dir.mkdir(parents=True, exist_ok=True)
        # Defaults so a caller that reaches the GPS loader without going through
        # `run_pipeline` still gets the historical behaviour rather than an
        # attribute error. `gps_source` selects the *video's* route only; the
        # inertial logger always uses the GPS recorded in its own CSV.
        self.gps_source = "auto"
        self.video_file_path: Optional[str] = None

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
        user_id: Optional[str] = None,
        organisation_id: Optional[str] = None,
        run_name: Optional[str] = None,
        project_id: Optional[str] = None,
        road_uuid: Optional[str] = None,
        camera_overrides: Optional[Dict[str, Any]] = None,
        use_overlay_clock: bool = False,
        gps_source: str = "auto",
    ) -> Dict[str, Any]:
        #: Read each frame's time from the clock burned into the video rather than
        #: dividing the GPS span by the frame count. Off by default.
        self.use_overlay_clock = bool(use_overlay_clock)
        #: Which recording the track is taken from. "auto" prefers the uploaded
        #: GPS file and falls back to the vibration log, which is the historical
        #: order; "overlay" reads the position burned into the video itself.
        self.gps_source = gps_source or "auto"
        self.video_file_path = video_file_path
        self.user_id = user_id or "operator_default"
        self.organisation_id = organisation_id or "org_default"
        self.run_name = run_name or f"Survey_{self.run_id[:8]}"
        self.project_id = project_id
        self.road_uuid = road_uuid

        # The banner names the tracks that will actually run. It used to say
        # "2 pipelines: ASTM + Roadessy" unconditionally, which becomes a lie the
        # moment the ASTM track is switched off: a reader looking for the ASTM IRI
        # in this log would take its absence for a fault rather than for the
        # switch. The reason is logged with it, so a muted track is a stated
        # decision on the record.
        astm_state = astm_track_state()
        tracks = "ASTM + Roadessy" if astm_state["running"] else "Roadessy only"
        logger.info(
            f"Starting Heavy Compute Pipeline ({tracks})",
            run_id=self.run_id,
        )
        if not astm_state["running"]:
            logger.warning(
                "ASTM track is disabled in this build and will not be computed: "
                f"no ASTM PCI, no ASTM QCM IRI, no ASTM segments. {astm_state['reason']}",
                run_id=self.run_id,
            )
        if not astm_state["pci_enabled"]:
            logger.warning(
                f"ASTM D6433 PCI is not published: {astm_state['pci_reason']}",
                run_id=self.run_id,
            )

        # 1. Load and parse input datasets (shared)
        gps_df = self._load_gps_data(gps_file_path)

        # The inertial logger's position comes from the GPS recorded inside its
        # own CSV and is never taken from `gps_df`. The two describe the vehicle
        # at different rates and the accelerometer's own fix is the one its
        # samples are actually stamped against, so letting the video's route
        # choice reach in here would silently re-time the IRI.
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

        # 2. Camera calibration parameters lookup (shared).
        # Explicit request metadata wins, then a stored profile, then defaults.
        cam_height, cam_tilt, cam_fx, cam_fy, cam_cx, cam_cy, cam_distortion = self._resolve_camera_calibration(
            camera_profile_id, camera_overrides
        )
        
        # Initialize new components (DepthAnythingV2 is optional - only used if available)
        dav2_estimator = None
        try:
            dav2_estimator = DepthAnythingV2Estimator(encoder='vits')
        except Exception as exc:
            logger.warning(f"DepthAnythingV2 not available, dense depth estimation disabled: {exc}", run_id=self.run_id)
        
        # Only the MBTP estimator is used downstream; the scaler/segmentor/
        # fusion/IMU helpers are constructed inside their owning services.
        mbtp_estimator = MBTPAreaEstimator()

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
        #
        # Skipped outright when the ASTM track is muted: this resamples to 100 Hz
        # and runs the quarter-car simulation, which is not cheap, and nothing
        # downstream reads the result. `ASTM_QCM_IRI_ENABLED` used to guard only
        # a CSV write, so setting it false changed nothing but the output files.
        # An empty-but-valid result rather than None: everything downstream reads
        # `.iri_segments` and `.iri_m_km`, and an explicit empty result says
        # "measured nothing" where None would mean "not supplied" and force a
        # None-check at every reader. The 0.0 IRI it carries never reaches the
        # database — `_persist_db_records` stores NULL when this is muted.
        astm_qcm = None
        if ASTM_QCM_IRI_ENABLED:
            astm_qcm = InertialRefineryService.process_astm_qcm(
                timestamps=vibe_timestamps,
                accel_z=vibe_accel,
                speeds_kmh=vibe_speeds,
                run_id=self.run_id,
                target_fs=100.0,
                target_dx=0.025,
                v_ref_kmh=80.0,
            )
        else:
            astm_qcm = ASTMQCMResult(
                iri_m_km=0.0,
                total_distance_m=0.0,
                iri_segments=[],
                sample_rate_hz=0.0,
                quality_status="muted",
                quality_flags=["astm_qcm_iri_muted"],
                upsampling_info={},
            )
            logger.info(
                "ASTM QCM IRI muted, skipping the quarter-car simulation "
                f"(100 Hz resample and simulation not run; ASTM_QCM_IRI_ENABLED=false)",
                run_id=self.run_id,
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
                use_overlay_clock=self.use_overlay_clock,
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

                # Scale relative depth to metric using geometry-anchored scaling
                K_cam = np.array([[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]], dtype=np.float64)
                geom_scaler = create_depth_scaler(
                    camera_height_m=cam_height,
                    pitch_deg=cam_tilt,
                    roll_deg=0.0,
                    fx=cam_fx, fy=cam_fy, cx=cam_cx, cy=cam_cy,
                )
                metric_depth = geom_scaler.apply(abs_depth, K_cam, cam_tilt, 0.0, cam_height)

                # Open one capture for all pothole frame reads
                cap_depth = cv2.VideoCapture(video_file_path)

                # Process each pothole detection
                for det in astm_vision.detections:
                    if det.get("yolo_class_id") == 4:  # Pothole class
                        bbox = det.get("bbox")  # [x1, y1, x2, y2]
                        if bbox:
                            # Read the frame at the detection's actual frame index
                            frame_idx = int(det.get("frame_idx", 0))
                            cap_depth.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                            ret, frame = cap_depth.read()

                            if frame is not None:
                                # Run MBTP estimation
                                try:
                                    mbtp_result = mbtp_estimator.estimate(
                                        rgb_image=frame,
                                        depth_map=metric_depth,
                                        bbox=bbox,
                                        K=K_cam,
                                        ground_ref=None,
                                    )
                                    pothole_depths[det.get("chainage_m", 0)] = mbtp_result.depth_m

                                    # Update detection with MBTP results
                                    det["mbtp_area_m2"] = mbtp_result.area_m2
                                    det["mbtp_depth_m"] = mbtp_result.depth_m
                                    det["mbtp_volume_m3"] = mbtp_result.volume_m3
                                    det["mbtp_confidence"] = mbtp_result.confidence
                                    det["mbtp_method"] = "mbtp"

                                    # Update depth fields for Roadessy severity classification
                                    if mbtp_result.depth_m > 0:
                                        depth_mm = round(mbtp_result.depth_m * 1000.0, 1)
                                        det["depth_mm"] = depth_mm
                                        det["pothole_depth_mm"] = depth_mm

                                    logger.metric("pothole_depth", "mbtp_estimate", mbtp_result.depth_m, run_id=self.run_id)
                                except Exception as exc:
                                    logger.warning(f"MBTP estimation failed for pothole at chainage {det.get('chainage_m')}: {exc}", run_id=self.run_id)

                cap_depth.release()
            
            # 5d. Roadessy Vision (Ghana Table 3) - Pipeline 2
            # Reuse the same detections but with Ghana mapping
            roadessy_vision = ASTMVisionResult(
                detections=astm_vision.detections,  # Same detections, different mappings
                video_quality=astm_vision.video_quality,
                frame_sampling=astm_vision.frame_sampling,
                astm_distress_mapping=astm_vision.astm_distress_mapping,
                ghana_distress_mapping=astm_vision.ghana_distress_mapping,
            )

        # The road the 50 m sections are cut from, in coordinates as well as
        # distance, so the vision detections can be placed on it. Without this
        # the two 50 m tables are indexed from two different origins.
        inertial_track = build_inertial_track(vibe_lat, vibe_lon)

        # Give each detection its station along the matched road, when the run
        # matched one. A detection's own chainage is distance since the survey
        # started, so it is zero wherever the vehicle was put on the road; the
        # station is the same defect measured from the road's beginning and
        # holds across every survey of it. Applied after the vision results
        # exist and before anything is written out, so the CSVs, the annotated
        # video and the database all carry the same station.
        if astm_vision.detections:
            astm_vision.detections[:] = SpatialService.apply_absolute_stations(
                astm_vision.detections, gps_df, map_match_result
            )
            if roadessy_vision.detections is astm_vision.detections:
                roadessy_vision.detections = astm_vision.detections
            positioned = sum(
                1 for d in astm_vision.detections if d.get("station_m") is not None
            )
            if positioned:
                logger.info(
                    f"{positioned} of {len(astm_vision.detections)} defects placed at a road "
                    f"station on {map_match_result.get('matched_road_id') or 'the matched road'}.",
                    run_id=self.run_id,
                )
            else:
                logger.warning(
                    "No defect could be given a road station. The run's chainage is still "
                    "measured from where the survey started; it is not an absolute position "
                    "on the road and cannot be compared with another survey's until the "
                    "run matches a road.",
                    run_id=self.run_id,
                )

        # 6. Run BOTH pipelines
        # Pipeline 1: ASTM
        astm_result = PavementConditionFusioneer.compute_astm_pipeline(
            astm_qcm=astm_qcm,
            astm_vision=astm_vision,
            run_id=self.run_id,
            inertial_track=inertial_track,
        )
        
        # Pipeline 2: Roadessy
        roadessy_result = PavementConditionFusioneer.compute_roadessy_pipeline(
            simple_result=simple_result,
            roadessy_vision=roadessy_vision,
            run_id=self.run_id,
            inertial_track=inertial_track,
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
                defect_image_map=self._save_detected_frames(video_file_path, astm_vision.detections),
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
        if astm_result.get("astm_pci") is None:
            logger.info(
                "Pipeline 1 (ASTM): PCI muted (not published), "
                f"QCM_IRI={astm_result.get('qcm_iri_m_km')}",
                run_id=self.run_id,
            )
        else:
            logger.info(
                f"Pipeline 1 (ASTM): PCI={astm_result.get('astm_pci')}, "
                f"QCM_IRI={astm_result.get('qcm_iri_m_km')}",
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
                # The interval and where it came from are what every
                # detection's chainage rests on, and whether the burned-in clock
                # was read or the interval was inferred from the GPS span.
                "seconds_per_frame": astm_vision.frame_sampling.seconds_per_frame,
                "frame_interval_source": astm_vision.frame_sampling.frame_interval_source,
                "overlay_clock": astm_vision.frame_sampling.overlay_clock,
            },
        }

    def _load_gps_data(self, gps_file_path: str) -> pd.DataFrame:
        # Read defensively: this loader is called directly by tests that build
        # the object without `__init__`, and the source selection is optional in
        # any case. Defaulting here means "the historical behaviour" always.
        if getattr(self, "gps_source", "auto") == "overlay":
            # The video's own burned-in telemetry, rather than a file. It is the
            # one position record that came from the same camera as the footage,
            # so it cannot disagree with the video about where the vehicle was
            # the way a separate logger can.
            video_path = getattr(self, "video_file_path", None) or ""
            if not video_path or not os.path.exists(video_path):
                logger.error(
                    "The video overlay was chosen as the GPS source but no video is available.",
                    run_id=self.run_id,
                )
                return pd.DataFrame(columns=["time", "lat", "lon", "Speed", "GPSAcc"])
            try:
                df = UniversalGPSService.load_overlay_gps(video_path)
            except (ValueError, OSError) as error:
                # A missing overlay is a fallback, not a failed survey: the
                # camera's own GPS log is the next best source and is preferred
                # over the vibration logger, whose fix goes stale.
                logger.warning(
                    f"Overlay GPS unavailable ({error}). Falling back to the uploaded GPS file.",
                    run_id=self.run_id,
                )
                df = pd.DataFrame()
            if not df.empty:
                return df

        if not gps_file_path or not os.path.exists(gps_file_path):
            logger.warning(f"GPS file not found: {gps_file_path}", run_id=self.run_id)
            return pd.DataFrame(columns=["time", "lat", "lon", "Speed", "GPSAcc"])

        df = pd.read_csv(gps_file_path)
        df.columns = [str(col).strip() for col in df.columns]

        col_map = {
            "latitude": "lat", "Latitude": "lat",
            "Longitude": "lon", "longitude": "lon",
            "speed": "Speed", "Speed ": "Speed", "speed ": "Speed",
            "accuracy": "GPSAcc",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # The time axis is resolved by inspecting column *values*, via the same
        # routine the rest of the GPS loading uses. Deciding by name was wrong for
        # these exports: the video GPS files carry the elapsed video clock in
        # `Timestamp` as `H:MM:SS` strings (`0:00:00`, `0:37:21`) and the wall
        # clock in `Global_Second`, and reading the named column as a number then
        # as a datetime left the whole column at zero — `to_numeric` yields nothing
        # for `0:00:00`, and dateutil reads it as midnight on an arbitrary date.
        # A time column of all zeros then collapses the vision interpolators, and
        # every frame in the video resolved to the first GPS fix at chainage 0 m.
        time_series, time_source = UniversalGPSService._resolve_time_axis(df, prefer_recintvl=False)
        if time_series is None:
            # No column could be read as a clock. A 1 Hz axis is a guess, so it is
            # labelled as one rather than presented as measured time.
            time_series = pd.Series(np.arange(len(df), dtype=np.float64), index=df.index)
            time_source = "assumed_1hz"
        df["time"] = pd.to_numeric(time_series, errors="coerce")

        # The vision service looks a frame up at `frame_idx / fps` — elapsed
        # seconds since the video started — so the axis has to start at zero. The
        # video GPS already does, but an external GPS export carries a session
        # counter, and the interpolators hold anything outside their range at the
        # nearest endpoint rather than reporting a miss.
        if len(df) and df["time"].notna().any():
            df["time"] = df["time"] - float(df["time"].min())

        if time_source not in ("time", "timestamp", "elapsed", "seconds", "second", "assumed_1hz"):
            logger.warning(
                f"GPS time axis read from {time_source!r}; frames are matched against "
                f"elapsed video time, so confirm this file's clock is video-relative.",
                run_id=self.run_id,
            )

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
        speed_col = next((c for c in ["SpeedKmh", "speed", "SpeedGPS", "Speed"] if c in df.columns), None)

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
            # The coordinates have to be cut by the same mask as the series they
            # belong to. Left longer, they describe road the IRI never covered,
            # and the measured track used to place the vision detections would
            # run further than the survey the 50 m sections were cut from.
            if lat is not None:
                lat = lat[valid_mask]
            if long is not None:
                long = long[valid_mask]

        return timestamps, accel_z, speeds, lat, long
     
    def _resolve_camera_calibration(
        self,
        camera_profile_id: Optional[int],
        camera_overrides: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, float, float, float, float, float, Optional[str]]:
        """Resolve camera parameters.

        Precedence: explicit request metadata -> stored calibration profile ->
        the shared fallback in `app.core.camera`. Intrinsics may be nested under
        an "intrinsics" key, matching the shape produced by
        /analysis/calibrate-checkerboard.

        The fallbacks live in `app.core.camera` rather than here, because three
        routes used to carry their own and disagreed; the same video analyzed
        through two of them produced two different ground scales and so two
        different defect sizes, invisibly, because the reported IRI was the same
        either way.
        """
        resolved = resolve_camera_geometry(
            overrides=camera_overrides,
            stored=self.db.query(CalibrationProfile)
            .filter(CalibrationProfile.id == camera_profile_id)
            .first()
            if camera_profile_id
            else None,
        )
        return resolved.as_tuple()

    def _save_detected_frames(
        self,
        video_path: Optional[str],
        detections: List[Dict[str, Any]],
    ) -> Dict[int, str]:
        """Write one representative annotated frame per detected defect.

        Returns a mapping of detection index to the stored image filename so the
        database row can reference a file the API can actually serve.
        """
        image_map: Dict[int, str] = {}
        if not detections or not video_path or not os.path.exists(video_path):
            return image_map

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return image_map

        try:
            for index, det in enumerate(detections):
                bbox = det.get("bbox")
                defect_type = det.get("defect_type", "unknown")
                area = det.get("area", 0)
                width = det.get("width", 0)
                depth = det.get("depth_mm", 0)
                chainage = det.get("chainage_m", 0)

                # Detections without geometry still warrant a representative frame.
                frame = self._extract_frame(cap, det)
                if frame is None:
                    continue

                overlay = frame.copy()
                rect = self._bbox_to_pixel_rect(bbox)
                if rect is not None:
                    cv2.rectangle(overlay, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 3)

                label = (
                    f"{defect_type} | Area: {area:.3f}m2 | Width: {width:.1f}mm"
                    f" | Depth: {depth:.1f}mm | Chainage: {chainage:.1f}m"
                )
                cv2.putText(overlay, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                safe_type = self._sanitize_filename(str(defect_type))
                filename = f"{safe_type}_{index:04d}.jpg"
                if cv2.imwrite(str(self.defect_images_dir / filename), overlay):
                    image_map[index] = filename
        finally:
            cap.release()

        return image_map

    @staticmethod
    def _bbox_to_pixel_rect(bbox: Any) -> Optional[Tuple[int, int, int, int]]:
        """Normalize a detection bbox to an (x1, y1, x2, y2) pixel rectangle.

        Detections carry bbox as a flat xyxy list; nested corner pairs are also
        accepted so hand-built detections still render.
        """
        try:
            if not bbox or len(bbox) not in (2, 4):
                return None
            first = bbox[0]
            if isinstance(first, (list, tuple)):
                if len(bbox) != 2 or len(first) != 2 or len(bbox[1]) != 2:
                    return None
                (x1, y1), (x2, y2) = first, bbox[1]
            else:
                if len(bbox) != 4:
                    return None
                x1, y1, x2, y2 = bbox
            x1, x2 = sorted((float(x1), float(x2)))
            y1, y2 = sorted((float(y1), float(y2)))
            return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        except (TypeError, ValueError, IndexError):
            return None

    def _extract_frame(
        self, cap: "cv2.VideoCapture", det: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """Read the frame a detection was made on, falling back to frame 0."""
        frame_index = det.get("frame_idx")
        if frame_index is None:
            rect = self._bbox_to_pixel_rect(det.get("bbox"))
            if rect is not None:
                frame_index = (rect[1] + rect[3]) // 2
        try:
            frame_index = max(0, int(frame_index))
        except (TypeError, ValueError):
            frame_index = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret and frame is not None:
            return frame
        return None

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
        
        # ASTM QCM IRI segments. The IRI is a real measurement from the
        # vibration data and is independent of the muted PCI, so it still runs
        # unless explicitly switched off.
        if ASTM_QCM_IRI_ENABLED:
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
        
        # ASTM segment PCI. Absent while the PCI is muted, so no file is written
        # rather than a file of blanks.
        astm_seg_pci = astm_result.get("segment_pci_50m", [])
        if astm_seg_pci:
            pd.DataFrame(astm_seg_pci).to_csv(pci_dir / "astm_segment_pci_50m.csv", index=False)
        elif not astm_result.get("astm_pci_enabled", True):
            logger.info(
                "ASTM segment PCI not written: PCI muted", run_id=self.run_id
            )
        
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

        # 8. Re-encode the survey video with detection overlays
        self._save_annotated_video(video_file_path, astm_vision.detections)

        # 9. Render IRI profile plot so /analysis/download/iri-plot can serve it
        self._save_iri_plot(simple_result, astm_qcm)

        # 10. Human-readable analytics summary for /analysis/runs/{id}/analytics
        self._save_analytics_summary(simple_result, astm_qcm, astm_result, roadessy_result)

        # 11. Save run summary
        with open(self.output_dir / "run_summary.json", "w") as f:
            json.dump({
                "run_id": self.run_id,
                "name": self.run_name,
                "user_id": self.user_id,
                "organisation_id": self.organisation_id,
                "timestamp": datetime.utcnow().isoformat(),
                "output_directory": str(self.output_dir),
                "pipeline_1_astm": {
                    "qcm_iri_m_km": astm_result["qcm_iri_m_km"],
                    "astm_pci": astm_result["astm_pci"],
                    "astm_pci_enabled": astm_result.get("astm_pci_enabled", True),
                    "quality_flags": astm_result["quality_flags"],
                },
                "pipeline_2_roadessy": {
                    "roadessy_pci": roadessy_result["roadessy_pci"],
                    "simple_iri_m_km": roadessy_result["simple_iri_m_km"],
                },
                "num_segments_50m": max(len(astm_seg_pci), len(roadessy_seg_pci)),
                "total_defects": len(astm_vision.detections),
            }, f, indent=2)

    def _save_annotated_video(
        self, video_path: Optional[str], detections: List[Dict[str, Any]]
    ) -> None:
        """Re-encode the survey video with bounding boxes and labels burned in."""
        if not video_path or not os.path.exists(video_path):
            return

        output_path = self.output_dir / "video_with_detections.mp4"
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning("Annotated video skipped: source could not be opened", run_id=self.run_id)
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        # Group detections by their source frame so each frame draws only its own boxes.
        by_frame: Dict[int, List[Dict[str, Any]]] = {}
        for det in detections:
            frame_index = det.get("frame_idx")
            if frame_index is None:
                continue
            try:
                by_frame.setdefault(int(frame_index), []).append(det)
            except (TypeError, ValueError):
                continue

        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            cap.release()
            logger.warning("Annotated video skipped: writer could not be opened", run_id=self.run_id)
            return

        try:
            frame_index = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))

                for det in by_frame.get(frame_index, []):
                    rect = self._bbox_to_pixel_rect(det.get("bbox"))
                    if rect is not None:
                        cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
                    label = (
                        f"{det.get('defect_type', 'unknown')} "
                        f"{det.get('area', 0):.3f}m2 D={det.get('depth_mm', 0):.0f}mm"
                    )
                    cv2.putText(
                        frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )

                writer.write(frame)
                frame_index += 1
        finally:
            writer.release()
            cap.release()

    def _save_iri_plot(
        self, simple_result: InertialPipelineResult, astm_qcm: ASTMQCMResult
    ) -> None:
        """Render the IRI profile for the run as a PNG."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 5))
            simple_x = [seg.end_distance_m for seg in simple_result.iri_segments]
            simple_y = [seg.iri_m_km for seg in simple_result.iri_segments]
            ax.plot(simple_x, simple_y, label="Simple IRI", linewidth=1.5)
            astm_x = [seg.end_distance_m for seg in astm_qcm.iri_segments]
            astm_y = [seg.iri_m_km for seg in astm_qcm.iri_segments]
            ax.plot(astm_x, astm_y, label="ASTM QCM IRI", linewidth=1.5, alpha=0.8)
            ax.set_xlabel("Chainage (m)")
            ax.set_ylabel("IRI (m/km)")
            ax.set_title(f"Roadessy IRI profile - {self.run_name}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(self.output_dir / "iri_plot.png", dpi=120)
            plt.close(fig)
        except Exception as exc:
            logger.warning(f"Failed to render IRI plot: {exc}", run_id=self.run_id)

    def _save_analytics_summary(
        self,
        simple_result: InertialPipelineResult,
        astm_qcm: ASTMQCMResult,
        astm_result: Dict[str, Any],
        roadessy_result: Dict[str, Any],
    ) -> None:
        """Write a plain-text analytics report for the run."""
        lines = [
            "ROADESSY SURVEY ANALYTICS SUMMARY",
            "=" * 40,
            f"Run ID        : {self.run_id}",
            f"Run name      : {self.run_name}",
            f"User          : {self.user_id}",
            f"Organisation  : {self.organisation_id}",
            f"Generated     : {datetime.utcnow().isoformat()}",
            "",
            "PIPELINE 1 - ASTM (E1926/E950/D6433)",
            (
                f"  ASTM PCI            : {astm_result.get('astm_pci')}"
                if astm_result.get("astm_pci") is not None
                else "  ASTM PCI            : not published (PCI muted in this build)"
            ),
            f"  QCM IRI (m/km)      : {astm_result.get('qcm_iri_m_km')}",
            f"  Segments (ASTM PCI) : {len(astm_result.get('segment_pci_50m') or [])}",
            "",
            "PIPELINE 2 - ROADESSY (Ghana Manual 2023)",
            f"  Roadessy PCI        : {roadessy_result.get('roadessy_pci')}",
            f"  Simple IRI (m/km)   : {roadessy_result.get('simple_iri_m_km')}",
            "",
            "TRAVEL",
            f"  Total distance (m)  : {simple_result.total_distance_m}",
            f"  Simple segments     : {len(simple_result.iri_segments)}",
            f"  ASTM segments       : {len(astm_qcm.iri_segments)}",
        ]
        try:
            (self.output_dir / "analytics_summary.txt").write_text(
                "\n".join(lines), encoding="utf-8"
            )
        except OSError as exc:
            logger.warning(f"Failed to write analytics summary: {exc}", run_id=self.run_id)

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
        defect_image_map: Optional[Dict[int, str]] = None,
    ) -> None:
        try:
            matched_road_id = map_match_result.get("matched_road_id") or road_id
            trajectory_wkt = (
                map_match_result.get("snapped_trajectory_wkt")
                or map_match_result.get("trajectory_wkt")
            )

            # road_id on SurveyRun is a UUID FK to roads.id; matched_road_id is the
            # free-form centerline id from map matching and is stored separately.
            road_id_uuid = _coerce_uuid(self.road_uuid)
            project_id_uuid = _coerce_uuid(self.project_id)
            
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
                name=self.run_name,
                user_id=self.user_id,
                organisation_id=self.organisation_id,
                project_id=project_id_uuid,
                road_id=road_id_uuid,
                
                # Pipeline 1: ASTM. astm_pci and astm_qcm_iri are both None while
                # their track is muted; the columns are nullable and stay null
                # rather than being filled with a placeholder number. A 0.0 here
                # would read as a measured zero IRI, or a perfect road.
                astm_pci=(
                    float(astm_result["astm_pci"])
                    if astm_result.get("astm_pci") is not None
                    else None
                ),
                astm_qcm_iri=(
                    float(astm_result["qcm_iri_m_km"])
                    if astm_result.get("qcm_iri_m_km") is not None
                    else None
                ),
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
                user_id=self.user_id,
                matched_road_id=matched_road_id,
                match_status=map_match_result.get("match_status"),
                match_method=map_match_result.get("match_method"),
                centerline_wkt=map_match_result.get("centerline_wkt"),
                centerline_geojson=map_match_result.get("centerline_geojson"),
                snapped_trajectory_wkt=map_match_result.get("snapped_trajectory_wkt"),
                chainage_min_m=map_match_result.get("chainage_min_m", map_match_result.get("chainage_min")),
                chainage_max_m=map_match_result.get("chainage_max_m", map_match_result.get("chainage_max")),
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
            self._persist_defects(astm_vision.detections, matched_road_id, defect_image_map)
            
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
                user_id=self.user_id,
                road_id=road_id,
                segment_start_m=float(seg.start_distance_m),
                segment_end_m=float(seg.end_distance_m),
                iri_value=float(seg.iri_m_km),
                simple_iri=float(seg.iri_m_km),
                astm_qcm_iri=float(astm_qcm.iri_segments[seg_idx].iri_m_km) if seg_idx < len(astm_qcm.iri_segments) else None,
                astm_pci_segment=(
                    float(astm_seg.get("astm_pci"))
                    if astm_seg and astm_seg.get("astm_pci") is not None
                    else None
                ),
                roadessy_pci_segment=float(roadessy_seg.get("roadessy_pci", 0)) if roadessy_seg else None,
                classification=InertialRefineryService.classify_iri_5rank(float(seg.iri_m_km))["rank"],
            )
            self.db.add(db_seg)
        
        # ASTM QCM segments
        for seg_idx in range(len(simple_result.iri_segments), len(astm_qcm.iri_segments)):
            seg = astm_qcm.iri_segments[seg_idx]
            db_seg = IRISegment(
                run_id=self.run_id,
                user_id=self.user_id,
                road_id=road_id,
                segment_start_m=float(seg.start_distance_m),
                segment_end_m=float(seg.end_distance_m),
                iri_value=float(seg.iri_m_km),
                simple_iri=None,
                astm_qcm_iri=float(seg.iri_m_km),
                classification=InertialRefineryService.classify_iri_5rank(float(seg.iri_m_km))["rank"],
            )
            self.db.add(db_seg)

    def _persist_defects(
        self,
        detections: List[Dict[str, Any]],
        road_id: Optional[str],
        defect_image_map: Optional[Dict[int, str]] = None,
    ) -> None:
        for track_id, det in enumerate(detections):
            db_defect = TrackedDefect(
                run_id=self.run_id,
                user_id=self.user_id,
                road_id=road_id,
                track_id=track_id,
                failure_mode=det.get("defect_type"),
                max_area_m2=det.get("area"),
                max_width_mm=det.get("width"),
                max_depth_mm=det.get("depth_mm"),
                mean_depth_mm=det.get("depth_mm"),
                depth_confidence=0.85,
                depth_method="monocular_geometry",
                depth_message="Estimated via camera geometry model",
                deduct_value=det.get("deduct_value"),
                image_path=(defect_image_map or {}).get(track_id),
                astm_distress_type=det.get("astm_distress_type"),
                ghana_distress_type=det.get("ghana_distress_type"),
                ghana_rating_1to9=det.get("ghana_rating_1to9"),
                yolo_class_id=det.get("yolo_class_id"),
                confidence=det.get("confidence"),
                chainage_m=det.get("chainage_m"),
                station_m=det.get("station_m"),
                station_road_id=det.get("station_road_id"),
                station_source=det.get("station_source"),
                chainage_offset_m=det.get("chainage_offset_m"),
                # The vision service already resolved a position per detection
                # from the interpolated GPS track. It was computed, carried on
                # every record, and then dropped here, so `geom_location` stayed
                # null for every defect in the database and the only way to place
                # one on a map was to re-run the video.
                geom_location=self._detection_point(det),
            )
            self.db.add(db_defect)

    @staticmethod
    def _detection_point(detection: Dict[str, Any]):
        """GeoJSON point for a detection, or None when it has no usable position.

        Coordinates arrive as `[lon, lat]`, which is GeoJSON order and the
        reverse of the (lat, lon) the rest of this codebase passes around, so the
        order is taken from the record rather than assumed. A detection with
        (0.0, 0.0) is dropped: that is the service's "no GPS" value, and storing
        it would place the defect in the Gulf of Guinea.
        """
        geom = detection.get("geom") or {}
        coordinates = geom.get("coordinates") if isinstance(geom, dict) else None
        if not coordinates or len(coordinates) < 2:
            return None
        try:
            lon = float(coordinates[0])
            lat = float(coordinates[1])
        except (TypeError, ValueError):
            return None
        if lon == 0.0 and lat == 0.0:
            return None
        if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
            return None
        return _geometry_value(f"POINT({lon} {lat})")


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
    user_id: Optional[str] = None,
    organisation_id: Optional[str] = None,
    run_name: Optional[str] = None,
    project_id: Optional[str] = None,
    road_uuid: Optional[str] = None,
    camera_overrides: Optional[Dict[str, Any]] = None,
    use_overlay_clock: bool = False,
    gps_source: str = "auto",
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
        raw_speeds=[speed] * len(z_accel_data) if z_accel_data and speed else None,
        sample_rate=sample_rate,
        user_id=user_id,
        organisation_id=organisation_id,
        run_name=run_name,
        project_id=project_id,
        road_uuid=road_uuid,
        camera_overrides=camera_overrides,
        use_overlay_clock=use_overlay_clock,
        gps_source=gps_source,
    )