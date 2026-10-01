# app/api/routes/analysis.py
import json
import logging
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, List, Optional, Tuple, cast

import cv2
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse, Response
from sqlalchemy import text
from sqlalchemy.orm import Session
from shapely import wkb as shapely_wkb
from shapely import wkt as shapely_wkt

from app.core.config import ASTM_PCI_DISABLED_REASON, ASTM_PCI_ENABLED
from app.core.logger import get_pipeline_logger, iri_logger
from app.core.security import get_principal
from app.database.session import get_db
from app.models.survey import (
    IRISegment,
    MapMatchingRun,
    Project,
    Road,
    SurveyRun,
    TrackedDefect,
)
from app.services.inertial import InertialRefineryService
from app.services.pipeline.heavy_compute import run_heavy_pipeline

router = APIRouter(prefix="/analysis", tags=["Modular Road Analysis Engine"])
logger = logging.getLogger("roadessy.analysis")

# Artifact storage lives beside the pipeline output directory so the API serves
# exactly what the pipeline wrote, instead of a second empty tree.
OUTPUT_BASE_DIR = Path(
    os.getenv("ROADESSY_OUTPUT_DIR", r"F:\Adubofour\rodeessy_mphil\RoadSurveyOutputs")
)
DEFECT_IMAGES_DIR = str(OUTPUT_BASE_DIR / "defect_images")
VIDEOS_DIR = str(OUTPUT_BASE_DIR / "videos")
GPS_DIR = str(OUTPUT_BASE_DIR / "gps")
PLOTS_DIR = str(OUTPUT_BASE_DIR / "plots")
ANALYTICS_DIR = str(OUTPUT_BASE_DIR / "analytics")
for _directory in (DEFECT_IMAGES_DIR, VIDEOS_DIR, GPS_DIR, PLOTS_DIR, ANALYTICS_DIR):
    os.makedirs(_directory, exist_ok=True)


def _run_workspace_candidates(run_id: str) -> List[Path]:
    """Directories a run's artifacts may live in, most specific first."""
    return [
        OUTPUT_BASE_DIR / str(run_id),
        OUTPUT_BASE_DIR / "defect_images",
        OUTPUT_BASE_DIR / "videos",
        OUTPUT_BASE_DIR / "gps",
        OUTPUT_BASE_DIR / "plots",
        OUTPUT_BASE_DIR / "analytics",
    ]


def _resolve_run_artifact(run_id: str, file_name: str) -> Optional[str]:
    """Resolve a stored artifact filename against the known run directories.

    Rejects any path that escapes the artifact root.
    """
    safe_name = os.path.basename(str(file_name))
    if not safe_name or safe_name in {".", ".."}:
        return None

    for directory in _run_workspace_candidates(run_id):
        candidate = directory / safe_name
        try:
            candidate.resolve().relative_to(OUTPUT_BASE_DIR.resolve())
        except (ValueError, OSError):
            continue
        if candidate.is_file():
            return str(candidate)
    return None


def _find_run_artifact(run_id: str, extensions: Tuple[str, ...]) -> Optional[str]:
    """Find a run artifact by trying known filenames then a prefix scan."""
    for extension in extensions:
        resolved = _resolve_run_artifact(run_id, f"{run_id}{extension}")
        if resolved:
            return resolved
    for directory in _run_workspace_candidates(run_id):
        if not directory.is_dir():
            continue
        for entry in sorted(directory.iterdir()):
            if entry.is_file() and entry.name.startswith(str(run_id)) and entry.suffix.lower() in extensions:
                return str(entry)
    return None


def _haversine_distance_m(point_a, point_b):
    lon1, lat1 = float(point_a[0]), float(point_a[1])
    lon2, lat2 = float(point_b[0]), float(point_b[1])
    radius_m = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return radius_m * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _interpolate_trajectory_point(coords, cumulative_distances, distance_m):
    if distance_m <= 0:
        return list(coords[0])
    if distance_m >= cumulative_distances[-1]:
        return list(coords[-1])

    for index in range(1, len(cumulative_distances)):
        if cumulative_distances[index] >= distance_m:
            previous_distance = cumulative_distances[index - 1]
            segment_distance = cumulative_distances[index] - previous_distance
            ratio = 0.0 if segment_distance == 0 else (distance_m - previous_distance) / segment_distance
            return [
                float(previous_value) + ratio * (float(current_value) - float(previous_value))
                for previous_value, current_value in zip(coords[index - 1], coords[index])
            ]

    return list(coords[-1])


def _slice_trajectory(coords, start_m, end_m):
    if len(coords) < 2:
        return None

    start = max(0.0, float(start_m))
    end = max(start, float(end_m))
    cumulative_distances = [0.0]
    for previous, current in zip(coords, coords[1:]):
        cumulative_distances.append(cumulative_distances[-1] + _haversine_distance_m(previous, current))

    total_distance = cumulative_distances[-1]
    if total_distance <= 0:
        return None

    start = min(start, total_distance)
    end = min(end, total_distance)
    sliced_coords = [_interpolate_trajectory_point(coords, cumulative_distances, start)]
    for coordinate, distance_m in zip(coords[1:], cumulative_distances[1:]):
        if distance_m >= start and distance_m <= end:
            sliced_coords.append(list(coordinate))
    sliced_coords.append(_interpolate_trajectory_point(coords, cumulative_distances, end))

    deduplicated_coords = []
    for coordinate in sliced_coords:
        if not deduplicated_coords or coordinate != deduplicated_coords[-1]:
            deduplicated_coords.append(coordinate)
    return deduplicated_coords if len(deduplicated_coords) >= 2 else None


def _decode_metadata(value):
    if isinstance(value, (dict, list)):
        return value
    if not value:
        return {}
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return {}


def _geometry_wkt(value):
    if value is None:
        return None
    # Handle Shapely geometry objects
    if hasattr(value, "wkt"):
        return value.wkt
    # Handle WKB hex strings from PostGIS (geoalchemy2 returns WKB hex)
    if isinstance(value, str) and len(value) > 0:
        try:
            # Try to parse as WKB hex
            geom = shapely_wkb.loads(value, hex=True)
            return shapely_wkt.dumps(geom)
        except Exception:
            pass
    return str(value)


def _resolve_ownership(
    principal: dict, submitted_user_id: str, submitted_org_id: str
) -> tuple:
    """Prefer the verified session identity over client-supplied form fields."""
    verified_user_id = principal.get("user_id")
    verified_org_id = principal.get("org_id")
    resolved_user = verified_user_id or submitted_user_id or "operator_default"
    resolved_org = verified_org_id or submitted_org_id or "org_default"
    return resolved_user, resolved_org


def _parse_calibration_overrides(raw: Optional[str]) -> Optional[dict]:
    """Parse calibration metadata JSON into pipeline camera overrides."""
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"calibration_metadata_json is not valid JSON: {exc}",
        ) from exc
    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="calibration_metadata_json must be a JSON object",
        )
    return parsed


def _as_float(value: Any) -> Optional[float]:
    """Coerce a nullable numeric column to float, preserving None."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _run_total_distance_m(db: Session, run_uuid) -> Optional[float]:
    """Total surveyed distance, taken from the furthest IRI segment end."""
    try:
        segment = (
            db.query(IRISegment)
            .filter(IRISegment.run_id == run_uuid)
            .order_by(IRISegment.segment_end_m.desc())
            .first()
        )
        return float(getattr(segment, "segment_end_m")) if segment else None
    except Exception:
        return None


def _map_matching_payload(record):
    if record is None:
        return None
    return {
        "id": record.id,
        "run_id": str(record.run_id),
        "matched_road_id": record.matched_road_id,
        "match_status": record.match_status,
        "match_method": record.match_method,
        "centerline_wkt": record.centerline_wkt,
        "centerline_geojson": record.centerline_geojson,
        "snapped_trajectory_wkt": record.snapped_trajectory_wkt,
        "chainage_min_m": record.chainage_min_m,
        "chainage_max_m": record.chainage_max_m,
        "matched_distance_m": record.matched_distance_m,
        "road_length_m": record.road_length_m,
        "max_distance_m": record.max_distance_m,
        "matched_points": record.matched_points,
        "total_points": record.total_points,
        "unmatched_points": record.unmatched_points,
        "snapped_coords": record.snapped_coords,
        "created_at": record.created_at.isoformat() if record.created_at else None,
    }


@router.post("/calibrate-checkerboard", response_class=JSONResponse)
async def calibrate_checkerboard(
    images: list[UploadFile] = File(..., description="Checkerboard images for calibration"),
    camera_height_meters: float = Form(1.6, description="Camera mounting height above the pavement in meters"),
    camera_tilt_degrees: float = Form(15.0, description="Camera pitch angle relative to horizontal"),
    inner_corners_x: int = Form(7, ge=2, description="Number of checkerboard inner corners across"),
    inner_corners_y: int = Form(5, ge=2, description="Number of checkerboard inner corners down"),
    square_size_meters: float = Form(0.025, gt=0, description="Physical side length of one checkerboard square in meters"),
    already_undistorted: bool = Form(False, description="Set true if the source images/video have already been undistorted"),
):
    """Estimate camera intrinsics from a set of checkerboard images and return metadata for the pipeline."""
    board_size = (inner_corners_x, inner_corners_y)
    objp = np.zeros((inner_corners_x * inner_corners_y, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:inner_corners_x, 0:inner_corners_y].T.reshape(-1, 2) * square_size_meters

    object_points = []
    image_points = []
    image_size = None
    skipped_images = []

    calibration_logger = get_pipeline_logger("calibrate_checkerboard")
    calibration_logger.info("calibration started", uploaded_images=len(images), board_size=board_size, square_size_meters=square_size_meters, already_undistorted=already_undistorted)

    for upload in images:
        contents = await upload.read()
        if not contents:
            skipped_images.append({"filename": upload.filename, "reason": "empty file"})
            continue

        arr = np.frombuffer(contents, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            skipped_images.append({"filename": upload.filename, "reason": "not a readable image"})
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_image_size = gray.shape[::-1]
        if image_size is None:
            image_size = current_image_size
        elif current_image_size != image_size:
            skipped_images.append({"filename": upload.filename, "reason": "image dimensions differ from the first usable image"})
            continue

        found, corners = cv2.findChessboardCorners(gray, board_size, None)
        if not found:
            skipped_images.append({"filename": upload.filename, "reason": "checkerboard corners were not detected"})
            continue

        corners_refined = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        object_points.append(objp)
        image_points.append(corners_refined)

    if len(image_points) < 3 or image_size is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least three usable checkerboard images with matching dimensions are required")

    flags = 0
    if already_undistorted:
        flags = (
            cv2.CALIB_ZERO_TANGENT_DIST
            + cv2.CALIB_FIX_K1
            + cv2.CALIB_FIX_K2
            + cv2.CALIB_FIX_K3
            + cv2.CALIB_FIX_K4
            + cv2.CALIB_FIX_K5
            + cv2.CALIB_FIX_K6
        )

    ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(object_points, image_points, image_size, None, None, flags=flags)  # type: ignore[call-overload]

    if not ret:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Checkerboard calibration failed")

    if already_undistorted:
        dist_coeffs = np.zeros_like(dist_coeffs)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "calibration_complete",
            "uploaded_images": len(images),
            "used_images": len(image_points),
            "skipped_images": skipped_images,
            "camera_metadata": {
                "camera_height_meters": float(camera_height_meters),
                "camera_tilt_degrees": float(camera_tilt_degrees),
                "checkerboard_inner_corners": {"x": inner_corners_x, "y": inner_corners_y},
                "square_size_meters": float(square_size_meters),
                "intrinsics": {
                    "fx": float(camera_matrix[0, 0]),
                    "fy": float(camera_matrix[1, 1]),
                    "cx": float(camera_matrix[0, 2]),
                    "cy": float(camera_matrix[1, 2]),
                    "distortion_coefficients": [float(coefficient) for coefficient in dist_coeffs.ravel()],
                },
                "already_undistorted": already_undistorted,
            },
            "calibration_summary": {
                "reprojection_error": round(float(ret), 3),
                "instructions": "Use these values in the analysis workflow for geometry-based defect measurements and DV assignment.",
                "already_undistorted": already_undistorted,
            },
        },
    )


# -------------------------------------------------------------------------
# 1. ISOLATED INERTIAL ROUTE (Test IMU Data / Compute IRI Separately)
# -------------------------------------------------------------------------
@router.post("/test-inertial-compare", response_class=JSONResponse)
async def analyze_inertial_telemetry_compare(vibrational_data: UploadFile = File(..., description="High-frequency accelerometer log (.csv)")):
    """
    Runs BOTH IRI methods (simple displacement-to-distance and ASTM quarter-car)
    on the same CSV and returns both results side by side for comparison.
    """
    iri_logger.stage_start("test-inertial-compare", vibrational_data=vibrational_data.filename)
    scratch_id = str(uuid.uuid4())
    temp_dir = f"scratch_inertial_compare_{scratch_id}"
    os.makedirs(temp_dir, exist_ok=True)
    vib_path = os.path.join(temp_dir, "vibrations.csv")

    try:
        with open(vib_path, "wb") as buffer:
            shutil.copyfileobj(vibrational_data.file, buffer)

        # Load CSV for both methods
        df = pd.read_csv(vib_path)
        df.columns = [str(col).strip() for col in df.columns]
        time_col = next((c for c in ["Timestamp", "time", "GPSTime"] if c in df.columns), df.columns[0])
        accel_col = next((c for c in ["LinAccelZ", "AccelZ", "accel_z"] if c in df.columns), df.columns[1])
        speed_col = next((c for c in ["SpeedKmh", "speed", "SpeedGPS"] if c in df.columns), None)

        t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=np.float64)
        accel = pd.to_numeric(df[accel_col], errors="coerce").to_numpy(dtype=np.float64)
        speeds = pd.to_numeric(df[speed_col], errors="coerce").to_numpy(dtype=np.float64) if speed_col else None

        # Extract lat/lon for Haversine distance calculation (Simple IRI only)
        lat = None
        lon = None
        lat_col = next((c for c in ["Latitude", "latitude", "lat"] if c in df.columns), None)
        lon_col = next((c for c in ["Longitude", "longitude", "lon", "long"] if c in df.columns), None)
        if lat_col:
            lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy(dtype=np.float64)
        if lon_col:
            lon = pd.to_numeric(df[lon_col], errors="coerce").to_numpy(dtype=np.float64)

        # Method 1: Simple displacement-to-distance (uses Haversine from lat/lon)
        simple_result = InertialRefineryService.process_vibration_data(t, accel, speeds, latitudes=lat, longitudes=lon)
        simple_iri = simple_result.iri_m_km

        # Method 2: ASTM Quarter-car simulation (Pipeline 1) - uses speed×time distance
        try:
            qc_result = InertialRefineryService.process_astm_qcm(
                timestamps=t, accel_z=accel, speeds_kmh=speeds,
            )
            qc_iri = qc_result.iri_m_km
            qc_status = qc_result.quality_status
            qc_flags = list(qc_result.quality_flags)
            qc_distance = qc_result.total_distance_m
            qc_segments = [
                {
                    "start_distance_m": seg.start_distance_m,
                    "end_distance_m": seg.end_distance_m,
                    "iri_m_km": seg.iri_m_km,
                    "sample_count": seg.sample_count,
                }
                for seg in qc_result.iri_segments
            ]
        except Exception as e:
            qc_iri = None
            qc_status = "error"
            qc_flags = [f"quarter_car_failed: {str(e)}"]
            qc_distance = 0.0
            qc_segments = []

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "Both IRI methods processed successfully",
                "filename": vibrational_data.filename,
                "simple": {
                    "method": "displacement_to_distance",
                    "iri_m_km": round(simple_iri, 4),
                    "total_distance_m": round(simple_result.total_distance_m, 2),
                    "segments": [
                        {
                            "start_distance_m": seg.start_distance_m,
                            "end_distance_m": seg.end_distance_m,
                            "iri_m_km": seg.iri_m_km,
                            "sample_count": seg.sample_count,
                        }
                        for seg in simple_result.iri_segments
                    ],
                },
                "quarter_car": {
                    "method": "quarter_car_simulation",
                    "iri_m_km": round(qc_iri, 4) if qc_iri is not None else None,
                    "total_distance_m": round(qc_distance, 2),
                    "quality_status": qc_status,
                    "quality_flags": qc_flags,
                    "segments": qc_segments,
                },
                "comparison": {
                    "iri_diff_m_km": round(abs(simple_iri - qc_iri), 4) if qc_iri is not None else None,
                    "iri_ratio": round(simple_iri / qc_iri, 4) if qc_iri and qc_iri > 0 else None,
                },
            },
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inertial comparison fault: {str(e)}")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@router.post("/test-inertial", response_class=JSONResponse)
async def analyze_inertial_telemetry(vibrational_data: UploadFile = File(..., description="High-frequency accelerometer log (.csv)")):
    """
    Isolates the IMU processing pipeline. Use this endpoint to verify raw acceleration
    filtering, frequency calculation, and pure International Roughness Index (IRI) extraction.
    """
    iri_logger.stage_start("test-inertial", vibrational_data=vibrational_data.filename)
    scratch_id = str(uuid.uuid4())
    temp_dir = f"scratch_inertial_{scratch_id}"
    os.makedirs(temp_dir, exist_ok=True)
    vib_path = os.path.join(temp_dir, "vibrations.csv")

    try:
        with open(vib_path, "wb") as buffer:
            shutil.copyfileobj(vibrational_data.file, buffer)

        # Load CSV and run simple displacement IRI
        df = pd.read_csv(vib_path)
        df.columns = [str(col).strip() for col in df.columns]
        time_col = next((c for c in ["Timestamp", "time", "GPSTime"] if c in df.columns), df.columns[0])
        accel_col = next((c for c in ["LinAccelZ", "AccelZ", "accel_z"] if c in df.columns), df.columns[1])
        speed_col = next((c for c in ["SpeedKmh", "speed", "SpeedGPS"] if c in df.columns), None)

        t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=np.float64)
        accel = pd.to_numeric(df[accel_col], errors="coerce").to_numpy(dtype=np.float64)
        speeds = pd.to_numeric(df[speed_col], errors="coerce").to_numpy(dtype=np.float64) if speed_col else None

        inertial_result = InertialRefineryService.process_vibration_data(t, accel, speeds)
        computed_iri = inertial_result.iri_m_km

        iri_logger.metric("test-inertial", "computed_iri", round(computed_iri, 4), distance_m=round(inertial_result.total_distance_m, 2), samples=len(inertial_result.iri_segments))
        iri_logger.stage_end("test-inertial", duration_ms=0)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "Inertial pipeline processed successfully in isolation",
                "metrics": {
                    "total_samples_analyzed": int(len(t)),
                    "total_distance_m": round(inertial_result.total_distance_m, 2),
                    "computed_iri_m_km": round(computed_iri, 4),
                    "segments_50m": len(inertial_result.iri_segments),
                },
            },
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inertial fault: {str(e)}")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@router.get("/runs/geojson", response_class=JSONResponse)
async def list_all_runs_geojson(limit: int = 50, db: Session = Depends(get_db)):
    """
    Return survey run trajectories as a GeoJSON FeatureCollection.
    Each feature is a LineString representing the GPS trajectory of a survey run.
    """
    try:
        sql = text("""
            SELECT
                sr.run_id,
                sr.name,
                sr.computed_iri,
                sr.final_pci,
                sr.astm_pci,
                sr.roadessy_pci,
                sr.simple_iri,
                sr.astm_qcm_iri,
                sr.matched_road_id,
                sr.created_at,
                ST_AsGeoJSON(sr.geom_trajectory) as geom_geojson
            FROM survey_runs sr
            WHERE sr.geom_trajectory IS NOT NULL
            ORDER BY sr.created_at DESC
            LIMIT :limit
        """)
        rows = db.execute(sql, {"limit": limit}).fetchall()

        features = []
        for row in rows:
            if not row.geom_geojson:
                continue

            geom = json.loads(row.geom_geojson)

            features.append({
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "run_id": str(row.run_id),
                    "name": row.name,
                    "computed_iri": float(row.computed_iri) if row.computed_iri is not None else None,
                    "final_pci": float(row.final_pci) if row.final_pci is not None else None,
                    "astm_pci": float(row.astm_pci) if row.astm_pci is not None else None,
                    "roadessy_pci": float(row.roadessy_pci) if row.roadessy_pci is not None else None,
                    "simple_iri": float(row.simple_iri) if row.simple_iri is not None else None,
                    "astm_qcm_iri": float(row.astm_qcm_iri) if row.astm_qcm_iri is not None else None,
                    "matched_road_id": row.matched_road_id,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
            })

        return JSONResponse(status_code=status.HTTP_200_OK, content={
            "type": "FeatureCollection",
            "features": features,
        })
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to fetch runs: {str(e)}")


@router.get("/iri/segments/geojson", response_class=JSONResponse)
async def list_all_iri_segments_geojson(
    run_id: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    Return IRI segments with classifications as a GeoJSON FeatureCollection.
    Each feature is a LineString segment with PCI/IRI classification properties.
    """
    try:
        sql = text("""
            SELECT
                iseg.id,
                iseg.run_id,
                iseg.road_id,
                iseg.segment_start_m,
                iseg.segment_end_m,
                iseg.iri_value,
                iseg.simple_iri,
                iseg.astm_qcm_iri,
                iseg.astm_pci_segment,
                iseg.roadessy_pci_segment,
                iseg.classification,
                sr.name as run_name,
                ST_AsGeoJSON(sr.geom_trajectory) as run_geom_geojson
            FROM iri_segments iseg
            JOIN survey_runs sr ON iseg.run_id = sr.run_id
            WHERE sr.geom_trajectory IS NOT NULL
            ORDER BY iseg.run_id, iseg.segment_start_m
        """)
        rows = db.execute(sql).fetchall()

        if run_id:
            try:
                requested_run_id = uuid.UUID(run_id)
            except ValueError:
                raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid run_id UUID")
            rows = [row for row in rows if row.run_id == requested_run_id]

        features = []
        for row in rows:
            if not row.run_geom_geojson:
                continue

            run_geom = json.loads(row.run_geom_geojson)
            if run_geom.get("type") != "LineString":
                continue

            coords = run_geom.get("coordinates", [])
            sliced_coords = _slice_trajectory(coords, row.segment_start_m, row.segment_end_m)
            if sliced_coords is None:
                continue

            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": sliced_coords},
                "properties": {
                    "segment_id": row.id,
                    "run_id": str(row.run_id),
                    "run_name": row.run_name,
                    "road_id": row.road_id,
                    "segment_start_m": float(row.segment_start_m),
                    "segment_end_m": float(row.segment_end_m),
                    "start_m": float(row.segment_start_m),
                    "end_m": float(row.segment_end_m),
                    "iri_value": float(row.iri_value) if row.iri_value is not None else None,
                    "iri": float(row.iri_value) if row.iri_value is not None else None,
                    "simple_iri": float(row.simple_iri) if row.simple_iri is not None else None,
                    "astm_qcm_iri": float(row.astm_qcm_iri) if row.astm_qcm_iri is not None else None,
                    "astm_pci_segment": float(row.astm_pci_segment) if row.astm_pci_segment is not None else None,
                    "roadessy_pci_segment": float(row.roadessy_pci_segment) if row.roadessy_pci_segment is not None else None,
                    "classification": row.classification,
                }
            })

        return JSONResponse(status_code=status.HTTP_200_OK, content={
            "type": "FeatureCollection",
            "features": features,
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to fetch IRI segments: {str(e)}")


@router.get("/runs/latest", response_class=JSONResponse)
async def get_latest_run(db: Session = Depends(get_db)):
    """Return the most recent survey run as a flat object (no wrapping list)."""
    try:
        run = db.query(SurveyRun).order_by(SurveyRun.created_at.desc()).first()
        if run is None:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "run_id": None,
                    "name": None,
                    "computed_iri": None,
                    "final_pci": None,
                    "astm_pci": None,
                    "astm_qcm_iri": None,
                    "roadessy_pci": None,
                    "simple_iri": None,
                },
            )

        # Return every track metric the frontend schema declares, not just two.
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "run_id": str(run.run_id),
                "name": run.name,
                "computed_iri": _as_float(run.computed_iri),
                "final_pci": _as_float(run.final_pci),
                "astm_pci": _as_float(getattr(run, "astm_pci", None)),
                "astm_pci_enabled": ASTM_PCI_ENABLED,
                "astm_qcm_iri": _as_float(getattr(run, "astm_qcm_iri", None)),
                "roadessy_pci": _as_float(getattr(run, "roadessy_pci", None)),
                "simple_iri": _as_float(getattr(run, "simple_iri", None)),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to fetch latest run: {str(e)}")


@router.get("/runs/{run_id}", response_class=JSONResponse)
async def get_run_summary(run_id: str, db: Session = Depends(get_db)):
    try:
        rid = uuid.UUID(run_id)
    except Exception:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid run_id UUID")

    run = db.query(SurveyRun).filter(SurveyRun.run_id == rid).one_or_none()
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    defects_count = db.query(TrackedDefect).filter(TrackedDefect.run_id == rid).count()
    
    # Get IRI segments with all track data
    iri_segments = db.query(IRISegment).filter(IRISegment.run_id == rid).order_by(IRISegment.segment_start_m).all()
    
    # Compute segment geometries from run trajectory
    trajectory_coords = None
    if run.geom_trajectory:
        try:
            geom = shapely_wkb.loads(bytes(run.geom_trajectory.data))
            if geom.geom_type == "LineString":
                trajectory_coords = list(geom.coords)
        except Exception:
            pass

    def _slice_trajectory_local(coords, start_m, end_m):
        """Slice a trajectory between start_m and end_m meters."""
        if not coords or len(coords) < 2:
            return None

        start = max(0.0, float(start_m))
        end = max(start, float(end_m))
        cumulative_distances = [0.0]
        for previous, current in zip(coords, coords[1:]):
            cumulative_distances.append(cumulative_distances[-1] + _haversine_distance_m(previous, current))

        total_distance = cumulative_distances[-1]
        if total_distance <= 0:
            return None

        start = min(start, total_distance)
        end = min(end, total_distance)
        sliced_coords = [_interpolate_trajectory_point(coords, cumulative_distances, start)]
        for coordinate, distance_m in zip(coords[1:], cumulative_distances[1:]):
            if distance_m >= start and distance_m <= end:
                sliced_coords.append(list(coordinate))
        sliced_coords.append(_interpolate_trajectory_point(coords, cumulative_distances, end))

        return sliced_coords if len(sliced_coords) >= 2 else None

    def _interpolate_trajectory_point_local(coords, cumulative_distances, distance_m):
        if distance_m <= 0:
            return list(coords[0])
        if distance_m >= cumulative_distances[-1]:
            return list(coords[-1])

        for index in range(1, len(cumulative_distances)):
            if cumulative_distances[index] >= distance_m:
                previous_distance = cumulative_distances[index - 1]
                segment_distance = cumulative_distances[index] - previous_distance
                ratio = 0.0 if segment_distance == 0 else (distance_m - previous_distance) / segment_distance
                return [
                    float(previous_value) + ratio * (float(current_value) - float(previous_value))
                    for previous_value, current_value in zip(coords[index - 1], coords[index])
                ]

        return list(coords[-1])

    # Build all_roughness_segments with actual geometries
    all_roughness_segments = {"type": "FeatureCollection", "features": []}
    if trajectory_coords:
        segment_features = []
        for seg in iri_segments:
            sliced = _slice_trajectory_local(trajectory_coords, seg.segment_start_m, seg.segment_end_m)
            if sliced:
                segment_features.append({
                    "id": str(seg.id),
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": sliced
                    },
                    "properties": {
                        "run_id": str(seg.run_id),
                        "run_title": run.name,
                        "start_chainage_m": seg.segment_start_m,
                        "end_chainage_m": seg.segment_end_m,
                        "roughness_value": seg.iri_value or seg.simple_iri or seg.astm_qcm_iri or 0,
                        "avg_speed_mps": 0,
                        "quality_flags": {
                            "road_condition": seg.classification,
                        }
                    }
                })
        all_roughness_segments["features"] = segment_features
    
    processing_metadata = _decode_metadata(getattr(run, "processing_metadata", None))
    map_matching = _map_matching_payload(
        db.query(MapMatchingRun).filter(MapMatchingRun.run_id == rid).first()
    )

    created_at_val = getattr(run, "created_at", None)
    created_at_iso = created_at_val.isoformat() if created_at_val is not None else None

    # Get potholes from tracked defects with failure_mode == "pothole"
    pothole_defects = db.query(TrackedDefect).filter(
        TrackedDefect.run_id == rid,
        TrackedDefect.failure_mode == "pothole"
    ).all()

    potholes_features = []
    for defect in pothole_defects:
        if defect.geom_location:
            try:
                geom = shapely_wkb.loads(bytes(defect.geom_location.data))
                potholes_features.append({
                    "id": str(defect.id),
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(geom.x), float(geom.y)]
                    },
                    "properties": {
                        "timestamp": defect.created_at.isoformat() if defect.created_at else "",
                        "severity": defect.ghana_rating_1to9 or 1,
                        "confidence": defect.confidence or 0,
                        "meta": {},
                        "created_at": defect.created_at.isoformat() if defect.created_at else "",
                        "run": str(rid),
                        "org": run.organisation_id,
                        "image_asset": defect.image_path,
                        "image_url": defect.image_path,
                        "clip_asset": None,
                    }
                })
            except Exception:
                pass

    potholes_geojson = {
        "type": "FeatureCollection",
        "features": potholes_features
    }

    # Get assets for this run
    assets = []
    # Note: Asset model would need to be queried here if it exists
    # For now return empty array

    # Get user/org info for created_by and team
    created_by = {
        "id": run.user_id,
        "username": run.user_id,  # Placeholder
        "email": run.user_id,
        "full_name": run.user_id
    }

    project = None
    if run.project_id:
        project = db.query(Project).filter(Project.id == run.project_id).first()
    team_info = None
    if project:
        team_info = {
            "id": str(project.team_id) if project.team_id else str(project.id),
            "org": {
                "id": project.org_id,
                "name": project.org_name,
                "is_active": True,
                "created_at": project.created_at.isoformat() if project.created_at else "",
                "is_personal": project.visibility == "PERSONAL",
                "owner": project.created_by
            },
            "name": project.name,
            "description": project.description,
            "is_default_personal": project.visibility == "PERSONAL",
            "is_active": True,
            "created_at": project.created_at.isoformat() if project.created_at else ""
        }

    updated_at_val = getattr(run, "updated_at", None)
    updated_at_iso = updated_at_val.isoformat() if updated_at_val is not None else created_at_iso

    road_row = db.query(Road).filter(Road.id == run.road_id).first() if run.road_id else None
    road_name = getattr(road_row, "name", None)

    total_distance_m = _run_total_distance_m(db, rid)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "run_id": str(run.run_id),
            "name": run.name,
            "user_id": run.user_id,
            "organisation_id": run.organisation_id,
            "project_id": str(run.project_id) if run.project_id else None,
            "road_id": str(run.road_id) if run.road_id else None,
            
            # Track 1: ASTM. astm_pci is null while the PCI is muted, so the
            # flag and the reason travel beside it rather than a bare null.
            "astm_pci": getattr(run, "astm_pci", None),
            "astm_pci_enabled": ASTM_PCI_ENABLED,
            "astm_pci_disabled_reason": None if ASTM_PCI_ENABLED else ASTM_PCI_DISABLED_REASON,
            "astm_qcm_iri": getattr(run, "astm_qcm_iri", None),
            "astm_quality_flags": getattr(run, "astm_quality_flags", None),
            "astm_upsampling_info": getattr(run, "astm_upsampling_info", None),
            
            # Track 2: Roadessy
            "simple_iri": getattr(run, "simple_iri", None),
            
            # Track 3: Ghana
            "roadessy_pci": getattr(run, "roadessy_pci", None),
            
            # Legacy
            "computed_iri": run.computed_iri,
            "final_pci": run.final_pci,
            "matched_road_id": run.matched_road_id,
            "geom_trajectory": _geometry_wkt(run.geom_trajectory),
            "map_matching": map_matching,
            "created_at": created_at_iso,
            "updated_at": updated_at_iso,
            "processing_metadata": processing_metadata,
            "defects_count": defects_count,
            
            # Frontend expected fields
            "title": run.name,
            "status": "COMPLETED",
            "scope": project.visibility if project and getattr(project, "visibility", None) else "PERSONAL",
            "distance_m": total_distance_m,
            "avg_roughness": getattr(run, "computed_iri", None),
            "pothole_counts": len(potholes_features),
            "external_project_id": str(run.project_id) if run.project_id else None,
            "external_road_id": str(run.road_id) if run.road_id else None,
            "project_name": project.name if project else None,
            "road_name": road_name,
            "roughness_geojson_url": f"/analysis/iri/segments/geojson?run_id={rid}",
            "all_roughness_segments": all_roughness_segments,
            "potholes": potholes_geojson,
            "all_assets": assets,
            "created_by": created_by,
            "team": team_info,
            "can_write": True,
            "started_at": None,
            "ended_at": None,
            
            # IRI Segments with all tracks
            "iri_segments": [
                {
                    "id": seg.id,
                    "run_id": str(seg.run_id),
                    "road_id": seg.road_id,
                    "segment_start_m": seg.segment_start_m,
                    "segment_end_m": seg.segment_end_m,
                    "iri_value": seg.iri_value,
                    "simple_iri": getattr(seg, "simple_iri", None),
                    "astm_qcm_iri": getattr(seg, "astm_qcm_iri", None),
                    "astm_pci_segment": getattr(seg, "astm_pci_segment", None),
                    "roadessy_pci_segment": getattr(seg, "roadessy_pci_segment", None),
                    "classification": seg.classification,
                    "geom_location": None,  # Would need geometry
                }
                for seg in iri_segments
            ],
            
            # Tracked defects
            "tracked_defects": [
                {
                    "id": defect.id,
                    "run_id": str(defect.run_id),
                    "track_id": defect.track_id,
                    "failure_mode": defect.failure_mode,
                    "max_area_m2": defect.max_area_m2,
                    "max_width_mm": defect.max_width_mm,
                    "max_depth_mm": defect.max_depth_mm,
                    "mean_depth_mm": defect.mean_depth_mm,
                    "depth_confidence": defect.depth_confidence,
                    "depth_method": defect.depth_method,
                    "depth_message": defect.depth_message,
                    "deduct_value": defect.deduct_value,
                    "image_path": defect.image_path,
                    "astm_distress_type": defect.astm_distress_type,
                    "ghana_distress_type": defect.ghana_distress_type,
                    "ghana_rating_1to9": defect.ghana_rating_1to9,
                    "yolo_class_id": defect.yolo_class_id,
                    "confidence": defect.confidence,
                    "chainage_m": defect.chainage_m,
                    # Distance since the run started is not comparable between
                    # surveys; the station along the matched road is, and it is
                    # null when the run did not match a road.
                    "station_m": defect.station_m,
                    "station_road_id": defect.station_road_id,
                    "station_source": defect.station_source,
                    "chainage_offset_m": defect.chainage_offset_m,
                    "geom_location": None,
                }
                for defect in db.query(TrackedDefect).filter(TrackedDefect.run_id == rid).all()
            ],
        },
    )


@router.get("/runs/{run_id}/map-matching", response_class=JSONResponse)
async def get_run_map_matching(run_id: str, db: Session = Depends(get_db)):
    try:
        rid = uuid.UUID(run_id)
    except Exception:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid run_id UUID")

    record = db.query(MapMatchingRun).filter(MapMatchingRun.run_id == rid).order_by(MapMatchingRun.id.desc()).first()
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Map-matching result not found")

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=_map_matching_payload(record),
    )


@router.get("/runs", response_class=JSONResponse)
async def list_runs(limit: int = 50, db: Session = Depends(get_db)):
    """Return a list of recent SurveyRun summaries for dashboard tables."""
    try:
        rows = db.query(SurveyRun).order_by(SurveyRun.created_at.desc()).limit(limit).all()
        run_ids = [run.run_id for run in rows]
        map_matching_by_run = {}
        if run_ids:
            map_records = db.query(MapMatchingRun).filter(MapMatchingRun.run_id.in_(run_ids)).all()
            map_matching_by_run = {record.run_id: _map_matching_payload(record) for record in map_records}

        runs = []
        for run in rows:
            processing_metadata = _decode_metadata(getattr(run, "processing_metadata", None))
            created_at_val = getattr(run, "created_at", None)
            created_at_iso = created_at_val.isoformat() if created_at_val is not None else None

            matched_road_id = getattr(run, "matched_road_id", None)
            simple_iri = getattr(run, "simple_iri", None)
            astm_qcm_iri = getattr(run, "astm_qcm_iri", None)
            astm_pci = getattr(run, "astm_pci", None)
            roadessy_pci = getattr(run, "roadessy_pci", None)
            computed_iri = run.computed_iri
            final_pci = run.final_pci
            name = run.name
            defects_count = db.query(TrackedDefect).filter(TrackedDefect.run_id == run.run_id).count()

            runs.append({
                "run_id": str(run.run_id),
                "name": name,
                "computed_iri": computed_iri,
                "final_pci": final_pci,
                "roadessy_pci": roadessy_pci,
                "astm_pci": astm_pci,
                "simple_iri": simple_iri,
                "astm_qcm_iri": astm_qcm_iri,
                "matched_road_id": matched_road_id,
                "total_distance_m": _run_total_distance_m(db, run.run_id),
                "defects_count": defects_count,
                "map_matching": map_matching_by_run.get(run.run_id),
                "created_at": created_at_iso,
                "processing_metadata": processing_metadata,
            })

        return JSONResponse(status_code=status.HTTP_200_OK, content={"runs": runs})
    except Exception as e:
        logger.exception("Failed to list runs")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list runs: {str(e)}",
        )


@router.get("/roads", response_class=JSONResponse)
async def list_roads_with_survey_data(db: Session = Depends(get_db)):
    """
    Return all road centerlines with their latest survey metrics as GeoJSON.
    Each feature includes PCI, IRI, defect counts, and other survey properties.
    """
    try:
        sql = text("""
            WITH latest_surveys AS (
                SELECT DISTINCT ON (sr.matched_road_id)
                    sr.run_id,
                    sr.matched_road_id,
                    sr.computed_iri,
                    sr.final_pci,
                    sr.astm_pci,
                    sr.roadessy_pci,
                    sr.simple_iri,
                    sr.astm_qcm_iri,
                    sr.created_at
                FROM survey_runs sr
                WHERE sr.matched_road_id IS NOT NULL
                ORDER BY sr.matched_road_id, sr.created_at DESC
            )
            SELECT
                g.gid::VARCHAR AS road_id,
                g.name AS road_name,
                g.fclass,
                g.ref,
                ST_AsGeoJSON(g.geom) as geom_geojson,
                ST_AsText(g.geom) as geom_wkt,
                ST_AsText(ST_Buffer(g.geom::geography, 7.0)::geometry) as buffer_wkt,
                ls.computed_iri,
                ls.final_pci,
                ls.astm_pci,
                ls.roadessy_pci,
                ls.simple_iri,
                ls.astm_qcm_iri,
                ls.run_id,
                ls.created_at as survey_date,
                COUNT(td.id) as total_defects,
            COUNT(CASE WHEN td.failure_mode = 'Pothole' THEN 1 END) as pothole_count,
            COUNT(CASE WHEN td.failure_mode = 'Crack' THEN 1 END) as crack_count,
            COUNT(CASE WHEN td.failure_mode = 'Rutting' THEN 1 END) as rutting_count,
            COUNT(CASE WHEN td.failure_mode = 'Patching' THEN 1 END) as patching_count,
            COUNT(CASE WHEN td.failure_mode = 'Ravelling' THEN 1 END) as ravelling_count,
            COUNT(CASE WHEN td.failure_mode = 'Bleeding' THEN 1 END) as bleeding_count,
            COUNT(CASE WHEN td.failure_mode = 'Subgrade Failure' THEN 1 END) as subgrade_failure_count
            FROM gh_roads g
            LEFT JOIN latest_surveys ls ON g.gid::VARCHAR = ls.matched_road_id
            LEFT JOIN tracked_defects td ON ls.run_id = td.run_id
            GROUP BY g.gid, g.name, g.fclass, g.ref, g.geom,
                     ls.computed_iri, ls.final_pci, ls.astm_pci, ls.roadessy_pci, ls.simple_iri, ls.astm_qcm_iri, ls.run_id, ls.created_at
            ORDER BY ls.created_at DESC NULLS LAST
        """)
        rows = db.execute(sql).fetchall()

        features = []
        for row in rows:
            if not row.geom_geojson:
                continue

            geom = json.loads(row.geom_geojson)

            features.append({
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "road_id": row.road_id,
                    "road_name": row.road_name,
                    "buffer_wkt": row.buffer_wkt if row.buffer_wkt else None,
                    "computed_iri": float(row.computed_iri) if row.computed_iri is not None else None,
                    "final_pci": float(row.final_pci) if row.final_pci is not None else None,
                    "astm_pci": float(row.astm_pci) if row.astm_pci is not None else None,
                    "roadessy_pci": float(row.roadessy_pci) if row.roadessy_pci is not None else None,
                    "simple_iri": float(row.simple_iri) if row.simple_iri is not None else None,
                    "astm_qcm_iri": float(row.astm_qcm_iri) if row.astm_qcm_iri is not None else None,
                    "run_id": str(row.run_id) if row.run_id else None,
                    "survey_date": row.survey_date.isoformat() if row.survey_date else None,
                    "total_defects": int(row.total_defects) if row.total_defects else 0,
                    "pothole_count": int(row.pothole_count) if row.pothole_count else 0,
                    "crack_count": int(row.crack_count) if row.crack_count else 0,
                    "rutting_count": int(row.rutting_count) if row.rutting_count else 0,
                    "patching_count": int(row.patching_count) if row.patching_count else 0,
                    "ravelling_count": int(row.ravelling_count) if row.ravelling_count else 0,
                    "bleeding_count": int(row.bleeding_count) if row.bleeding_count else 0,
                    "subgrade_failure_count": int(row.subgrade_failure_count) if row.subgrade_failure_count else 0,
                }
            })

        return JSONResponse(status_code=status.HTTP_200_OK, content={
            "type": "FeatureCollection",
            "features": features,
        })
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to fetch roads: {str(e)}")


@router.get("/roads/gh", response_class=JSONResponse)
async def list_gh_roads_with_survey_data(db: Session = Depends(get_db)):
    """
    Return Ghana road network centerlines with their latest survey metrics as GeoJSON.
    Queries gh_roads table and joins with latest survey_runs.
    """
    try:
        sql = text("""
            WITH latest_surveys AS (
                SELECT DISTINCT ON (sr.matched_road_id)
                    sr.run_id,
                    sr.matched_road_id,
                    sr.computed_iri,
                    sr.final_pci,
                    sr.astm_pci,
                    sr.roadessy_pci,
                    sr.simple_iri,
                    sr.astm_qcm_iri,
                    sr.created_at
                FROM survey_runs sr
                WHERE sr.matched_road_id IS NOT NULL
                ORDER BY sr.matched_road_id, sr.created_at DESC
            )
            SELECT
                g.gid::VARCHAR AS road_id,
                g.name AS road_name,
                g.fclass,
                g.ref,
                ST_AsGeoJSON(g.geom) as geom_geojson,
                ls.computed_iri,
                ls.final_pci,
                ls.astm_pci,
                ls.roadessy_pci,
                ls.simple_iri,
                ls.astm_qcm_iri,
                ls.run_id,
                ls.created_at as survey_date,
                COUNT(td.id) as total_defects,
                COUNT(CASE WHEN td.failure_mode = 'Pothole' THEN 1 END) as pothole_count,
                COUNT(CASE WHEN td.failure_mode = 'Crack' THEN 1 END) as crack_count,
                COUNT(CASE WHEN td.failure_mode = 'Rutting' THEN 1 END) as rutting_count,
                COUNT(CASE WHEN td.failure_mode = 'Patching' THEN 1 END) as patching_count,
                COUNT(CASE WHEN td.failure_mode = 'Ravelling' THEN 1 END) as ravelling_count,
                COUNT(CASE WHEN td.failure_mode = 'Bleeding' THEN 1 END) as bleeding_count,
                COUNT(CASE WHEN td.failure_mode = 'Subgrade Failure' THEN 1 END) as subgrade_failure_count
            FROM gh_roads g
            LEFT JOIN latest_surveys ls ON g.gid::VARCHAR = ls.matched_road_id
            LEFT JOIN tracked_defects td ON ls.run_id = td.run_id
            GROUP BY g.gid, g.name, g.fclass, g.ref, g.geom,
                     ls.computed_iri, ls.final_pci, ls.astm_pci, ls.roadessy_pci, ls.simple_iri, ls.astm_qcm_iri, ls.run_id, ls.created_at
            ORDER BY ls.created_at DESC NULLS LAST
        """)
        rows = db.execute(sql).fetchall()

        features = []
        for row in rows:
            if not row.geom_geojson:
                continue

            geom = json.loads(row.geom_geojson)

            features.append({
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "road_id": row.road_id,
                    "road_name": row.road_name,
                    "fclass": row.fclass,
                    "ref": row.ref,
                    "computed_iri": float(row.computed_iri) if row.computed_iri is not None else None,
                    "final_pci": float(row.final_pci) if row.final_pci is not None else None,
                    "astm_pci": float(row.astm_pci) if row.astm_pci is not None else None,
                    "roadessy_pci": float(row.roadessy_pci) if row.roadessy_pci is not None else None,
                    "simple_iri": float(row.simple_iri) if row.simple_iri is not None else None,
                    "astm_qcm_iri": float(row.astm_qcm_iri) if row.astm_qcm_iri is not None else None,
                    "run_id": str(row.run_id) if row.run_id else None,
                    "survey_date": row.survey_date.isoformat() if row.survey_date else None,
                    "total_defects": int(row.total_defects) if row.total_defects else 0,
                    "pothole_count": int(row.pothole_count) if row.pothole_count else 0,
                    "crack_count": int(row.crack_count) if row.crack_count else 0,
                    "rutting_count": int(row.rutting_count) if row.rutting_count else 0,
                    "patching_count": int(row.patching_count) if row.patching_count else 0,
                    "ravelling_count": int(row.ravelling_count) if row.ravelling_count else 0,
                    "bleeding_count": int(row.bleeding_count) if row.bleeding_count else 0,
                    "subgrade_failure_count": int(row.subgrade_failure_count) if row.subgrade_failure_count else 0,
                }
            })

        return JSONResponse(status_code=status.HTTP_200_OK, content={
            "type": "FeatureCollection",
            "features": features,
        })
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to fetch gh_roads: {str(e)}")


@router.get("/runs/{run_id}/defects", response_class=JSONResponse)
async def get_run_defects(run_id: str, db: Session = Depends(get_db)):
    try:
        rid = uuid.UUID(run_id)
    except Exception:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid run_id UUID")

    try:
        sql = text(
            "SELECT id, track_id, failure_mode, max_area_m2, max_width_mm, max_depth_mm, mean_depth_mm, depth_confidence, depth_method, depth_message, deduct_value, image_path, ST_AsText(geom_location) as geom_wkt, created_at FROM tracked_defects WHERE run_id = :rid ORDER BY id"
        )
        rows = db.execute(sql, {"rid": str(rid)}).fetchall()
        defects = [
            {
                "run_id": str(rid),
                "id": r.id,
                "track_id": r.track_id,
                "failure_mode": r.failure_mode,
                "max_area_m2": float(r.max_area_m2) if r.max_area_m2 is not None else None,
                "max_width_mm": float(r.max_width_mm) if r.max_width_mm is not None else None,
                "max_depth_mm": float(r.max_depth_mm) if r.max_depth_mm is not None else None,
                "mean_depth_mm": float(r.mean_depth_mm) if r.mean_depth_mm is not None else None,
                "depth_confidence": float(r.depth_confidence) if r.depth_confidence is not None else None,
                "depth_method": r.depth_method,
                "depth_message": r.depth_message,
                "deduct_value": float(r.deduct_value) if r.deduct_value is not None else None,
                "image_path": r.image_path,
                "geom_wkt": r.geom_wkt,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]

        return JSONResponse(status_code=status.HTTP_200_OK, content={"defects": defects})
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to fetch defects: {str(e)}")


@router.get("/runs/{run_id}/defects/{track_id}/image", response_class=FileResponse)
async def get_defect_image(run_id: str, track_id: int, db: Session = Depends(get_db)):
    try:
        rid = uuid.UUID(run_id)
    except Exception:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid run_id UUID")

    defect = db.query(TrackedDefect).filter(TrackedDefect.run_id == rid, TrackedDefect.track_id == track_id).first()
    image_filename = getattr(defect, "image_path", None) if defect is not None else None
    if not defect or not isinstance(image_filename, str) or not image_filename:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Defect image not found")

    # image_path is stored relative to the run workspace written by the pipeline.
    image_path = _resolve_run_artifact(str(rid), image_filename)
    if image_path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image file not found")

    return FileResponse(image_path, media_type="image/jpeg")


@router.get("/runs/{run_id}/iri", response_class=JSONResponse)
async def get_run_iri_segments(run_id: str, db: Session = Depends(get_db)):
    try:
        rid = uuid.UUID(run_id)
    except Exception:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid run_id UUID")

    segments = db.query(IRISegment).filter(IRISegment.run_id == rid).order_by(IRISegment.segment_start_m).all()
    if not segments:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No IRI segments found for this run")

    data = [
        {
            "start_m": float(cast(float, getattr(seg, "segment_start_m"))),
            "end_m": float(cast(float, getattr(seg, "segment_end_m"))),
            "iri": float(cast(float, getattr(seg, "iri_value"))),
            "simple_iri": getattr(seg, "simple_iri", None),
            "astm_qcm_iri": getattr(seg, "astm_qcm_iri", None),
            "astm_pci_segment": getattr(seg, "astm_pci_segment", None),
            "roadessy_pci_segment": getattr(seg, "roadessy_pci_segment", None),
            "classification": cast(Optional[str], getattr(seg, "classification", None)),
        }
        for seg in segments
    ]

    return JSONResponse(status_code=status.HTTP_200_OK, content={"run_id": run_id, "segments": data})


# -------------------------------------------------------------------------
# 2. MAIN SURVEY INGESTION ROUTE
# -------------------------------------------------------------------------
@router.post("/process", response_class=JSONResponse)
async def process_full_survey(
    user_id: str = Form(..., description="ID of the user running the survey"),
    organisation_id: str = Form(..., description="Organization identifier"),
    name: str = Form(..., description="Descriptive name for the survey run"),
    matched_road_id: Optional[str] = Form(None, description="Optional road centerline ID to match against"),
    road_id: Optional[str] = Form(None, description="UUID of the roads.id row this survey belongs to"),
    project_id: Optional[str] = Form(None, description="UUID of the projects.id row this survey belongs to"),
    video: UploadFile = File(..., description="Forward-facing dashcam video stream (.mp4)"),
    gps: Optional[UploadFile] = File(None, description="GPS telemetry log (.csv or .gpx)"),
    gps_source: Optional[str] = Form(
        "auto",
        description=(
            "Which recording the route is taken from: 'auto' (uploaded GPS file, "
            "else the vibration log), 'overlay' to read the position burned into "
            "the video, or 'video' for the container's own metadata."
        ),
    ),
    vibration: UploadFile = File(..., description="High-frequency accelerometer log (.csv)"),
    calibration_metadata_json: Optional[str] = Form(None, description="Stringified camera calibration metadata JSON"),
    calibration_profile_id: Optional[int] = Form(None, description="Stored calibration_profiles.id to use for this run"),
    use_overlay_clock: Optional[bool] = Form(
        False,
        description=(
            "Take each frame's time from the date-time burned into the video "
            "instead of dividing the GPS span by the frame count. Slower, but it "
            "is the only option that can see frames missing from the middle of "
            "a run. Falls back automatically when the overlay is unreadable."
        ),
    ),
    db: Session = Depends(get_db),
    principal: dict = Depends(get_principal),
):
    """
    Executes the complete Roadessy survey pipeline with TWO pipelines:
    Pipeline 1: ASTM (QCM IRI @ 100Hz + MDPI Vision + ASTM PCI)
    Pipeline 2: Roadessy (Simple IRI + Ghana Manual Table 3 Vision -> Roadessy PCI)

    Ownership is taken from the verified Clerk session when one is present; the
    form fields are only trusted for anonymous development callers.
    """
    resolved_user_id, resolved_org_id = _resolve_ownership(principal, user_id, organisation_id)

    run_id = uuid.uuid4()
    scratch_dir = os.path.join(str(OUTPUT_BASE_DIR), "workspaces", str(run_id))
    os.makedirs(scratch_dir, exist_ok=True)

    vid_path = os.path.join(scratch_dir, "video.mp4")
    gps_path = os.path.join(scratch_dir, "gps.csv")
    vib_path = os.path.join(scratch_dir, "vibrations.csv")

    try:
        # Save uploaded files locally to workspace directory
        if video and video.file:
            with open(vid_path, "wb") as buffer:
                shutil.copyfileobj(video.file, buffer)

        if gps and gps.file:
            with open(gps_path, "wb") as buffer:
                shutil.copyfileobj(gps.file, buffer)

        if vibration and vibration.file:
            with open(vib_path, "wb") as buffer:
                shutil.copyfileobj(vibration.file, buffer)

        camera_overrides = _parse_calibration_overrides(calibration_metadata_json)

        # Run the heavy compute pipeline (2 pipelines)
        result = run_heavy_pipeline(
            vibration_file_path=vib_path,
            gps_file_path=gps_path,
            video_path=vid_path,
            road_id=matched_road_id,
            camera_profile_id=calibration_profile_id,
            run_id=str(run_id),
            user_id=resolved_user_id,
            organisation_id=resolved_org_id,
            run_name=name.strip() or None,
            project_id=project_id,
            road_uuid=road_id,
            camera_overrides=camera_overrides,
            use_overlay_clock=bool(use_overlay_clock),
            gps_source=(gps_source or "auto"),
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "run_id": str(run_id),
                "output_dir": result.get("output_dir"),
                "pipeline_1_astm": result.get("pipeline_1_astm"),
                "pipeline_2_roadessy": result.get("pipeline_2_roadessy"),
                "map_matching": result.get("map_matching"),
                "total_distance_m": result.get("total_distance_m"),
                "defects_detected": result.get("defects_detected"),
                "video_quality": result.get("video_quality"),
            },
        )

    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Survey processing analysis failed: {str(exc)}",
        )
    finally:
        if os.path.exists(scratch_dir):
            shutil.rmtree(scratch_dir, ignore_errors=True)


# -------------------------------------------------------------------------
# Download & Analytics Endpoints
# -------------------------------------------------------------------------



@router.get("/runs/{run_id}/analytics", response_class=JSONResponse)
async def get_run_analytics(run_id: str, db: Session = Depends(get_db)):
    """List analytics files available for a given run."""
    files: List[str] = []
    seen = set()
    for directory in _run_workspace_candidates(run_id):
        if not directory.is_dir():
            continue
        for entry in sorted(directory.iterdir()):
            if entry.is_file() and entry.name not in seen:
                seen.add(entry.name)
                files.append(entry.name)

    if not files:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No analytics found for this run")

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"run_id": run_id, "files": sorted(files)},
    )


_ANALYTICS_MEDIA_TYPES = {
    ".txt": "text/plain",
    ".json": "application/json",
    ".csv": "text/csv",
}


@router.get("/runs/{run_id}/analytics/{file_name:path}", response_class=FileResponse)
async def download_run_analytics_file(run_id: str, file_name: str):
    """Serve a specific analytics file for a run."""
    file_path = _resolve_run_artifact(run_id, file_name)
    if file_path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analytics file not found")
    media_type = _ANALYTICS_MEDIA_TYPES.get(os.path.splitext(file_path)[1].lower(), "application/octet-stream")
    return FileResponse(file_path, media_type=media_type, filename=os.path.basename(file_path))


@router.get("/download/video-with-detections/{run_id}", response_class=FileResponse)
async def download_video_with_detections(run_id: str):
    """Serve the processed video with detections overlay for a run."""
    video_path = _find_run_artifact(run_id, (".mp4", ".mov", ".mkv", ".avi"))
    if video_path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Processed video not found")
    return FileResponse(video_path, media_type="video/mp4", filename=f"{run_id}_detections.mp4")


@router.get("/download/gps-shp/{run_id}", response_class=Response)
async def download_gps_shapefile(run_id: str):
    """Serve GPS data as CSV for a run."""
    gps_path = _resolve_run_artifact(run_id, "gps_trajectory.csv") or _find_run_artifact(
        run_id, (".csv", ".gpx")
    )
    if gps_path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="GPS data not found")

    content = open(gps_path, "r", encoding="utf-8", errors="replace").read()
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={run_id}_gps.csv"},
    )


@router.get("/download/iri-plot/{run_id}", response_class=FileResponse)
async def download_iri_plot(run_id: str):
    """Serve the IRI plot image for a run."""
    plot_path = _resolve_run_artifact(run_id, "iri_plot.png") or _find_run_artifact(
        run_id, (".png", ".jpg", ".jpeg", ".svg")
    )
    if plot_path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="IRI plot not found")
    return FileResponse(plot_path, media_type="image/png", filename=f"{run_id}_iri_plot.png")