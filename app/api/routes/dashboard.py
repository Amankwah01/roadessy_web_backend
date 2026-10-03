# app/api/routes/dashboard.py
import math
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database.session import get_db
from app.models.survey import SurveyRun, TrackedDefect, IRISegment, MapMatchingRun, Project, Road
from shapely import wkb as shapely_wkb
from shapely import wkt as shapely_wkt

router = APIRouter(prefix="/api/me/dashboard", tags=["Dashboard"])


def _haversine_distance_m(point_a, point_b):
    """Calculate haversine distance between two points in meters."""
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
    """Interpolate a point at a given distance along a trajectory."""
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
    """Slice a trajectory between start_m and end_m meters."""
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

    return sliced_coords if len(sliced_coords) >= 2 else None


def _geometry_wkt(geom):
    """Convert geometry to WKT string."""
    if geom is None:
        return None
    try:
        if hasattr(geom, 'data'):
            return shapely_wkt.dumps(shapely_wkb.loads(bytes(geom.data)))
        return str(geom)
    except Exception:
        return None


@router.get("/summary", response_class=JSONResponse)
async def get_dashboard_summary(
    org_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """Get dashboard summary statistics."""
    try:
        # Base query for runs
        run_query = db.query(SurveyRun)
        if org_id:
            run_query = run_query.filter(SurveyRun.organisation_id == org_id)
        
        # For now, use a simpler approach
        processing_count = db.query(func.count(SurveyRun.run_id)).filter(
            SurveyRun.organisation_id == org_id if org_id else True
        ).scalar() or 0
        
        # Total distance from runs
        total_distance = db.query(func.coalesce(func.sum(SurveyRun.simple_iri), 0)).filter(
            SurveyRun.organisation_id == org_id if org_id else True
        ).scalar() or 0
        
        # Average IRI
        avg_iri = db.query(func.coalesce(func.avg(SurveyRun.simple_iri), 0)).filter(
            SurveyRun.organisation_id == org_id if org_id else True
        ).scalar() or 0
        
        # Total potholes
        pothole_count = db.query(func.count(TrackedDefect.id)).join(
            SurveyRun, SurveyRun.run_id == TrackedDefect.run_id
        ).filter(
            SurveyRun.organisation_id == org_id if org_id else True,
            TrackedDefect.failure_mode == "Pothole"
        ).scalar() or 0
        
        # Recent runs
        recent_runs_query = run_query.order_by(SurveyRun.created_at.desc()).limit(10)
        recent_runs = recent_runs_query.all()
        
        runs_list = []
        for run in recent_runs:
            # Get project/road info
            project_name = None
            road_name = None
            project_id = None
            road_id = None
            
            if run.project_id:
                project = db.query(Project).filter(Project.id == run.project_id).first()
                if project:
                    project_name = project.name
                    project_id = str(project.id)
            
            if run.road_id:
                road = db.query(Road).filter(Road.id == run.road_id).first()
                if road:
                    road_name = road.name
                    road_id = str(road.id)
            
            runs_list.append({
                "id": str(run.run_id),
                "title": run.name,
                "status": "COMPLETED" if run.final_pci is not None else "PROCESSING",
                "scope": "PERSONAL",
                "created_at": run.created_at.isoformat() if run.created_at else None,
                "started_at": None,
                "ended_at": None,
                "distance_m": float(total_distance) if total_distance else 0,
                "project_id": project_id,
                "project_name": project_name,
                "road_id": road_id,
                "road_name": road_name,
            })
        
        # Recent sensor assets (placeholder - no asset table yet)
        recent_sensor_assets = []
        
        return JSONResponse(status_code=status.HTTP_200_OK, content={
            "processing_count": processing_count,
            "pothole_count": pothole_count,
            "total_distance_m": float(total_distance) if total_distance else 0,
            "average_iri": float(avg_iri) if avg_iri else 0,
            "recent_runs": runs_list,
            "recent_sensor_assets": recent_sensor_assets,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to fetch dashboard summary: {str(e)}")


@router.get("/queue", response_class=JSONResponse)
async def get_processing_queue(
    org_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """Runs that are uploaded but not finished, so the dashboard can show a queue.

    Derived from what the runs table already records rather than from a separate
    job table: a run is in flight while it has a workspace and has published
    neither index, and it leaves the queue when `processing_metadata` records a
    finish or either index appears. A second source of truth about what is
    running would be a second thing to keep in step with the pipeline.

    There is no completion percentage to report. The pipeline writes its
    progress as log lines and artifacts, not as a counter, so `progress` is null
    and the client shows an indeterminate state rather than a number invented
    from how long the run has been going.
    """
    query = db.query(SurveyRun).filter(
        SurveyRun.final_pci.is_(None),
        SurveyRun.roadessy_pci.is_(None),
        SurveyRun.astm_pci.is_(None),
        SurveyRun.removed_at.is_(None),
    )
    if org_id:
        query = query.filter(SurveyRun.organisation_id == org_id)

    runs = query.order_by(SurveyRun.created_at.desc()).limit(limit).all()

    items = []
    for run in runs:
        metadata = run.processing_metadata if isinstance(run.processing_metadata, dict) else {}
        stage = metadata.get("stage") or metadata.get("phase")
        finished_at = next(
            (
                metadata.get(key)
                for key in ("finished_at", "completed_at", "updated_at")
                if isinstance(metadata.get(key), str)
            ),
            None,
        )
        items.append(
            {
                "id": str(run.run_id),
                "run_id": str(run.run_id),
                "name": run.name or str(run.run_id),
                "status": "DONE" if finished_at else "PROCESSING",
                "stage": stage if isinstance(stage, str) else None,
                "progress": None,
                "created_at": run.created_at.isoformat() if run.created_at else None,
                "video_uploaded": bool(run.video_file_path),
                "vibration_uploaded": bool(run.vibration_file_path),
                "gps_uploaded": bool(run.gps_file_path),
            }
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"items": items, "count": len(items)},
    )


@router.get("/runs/all", response_class=JSONResponse)
async def list_all_runs(
    org_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """List all runs for the dashboard."""
    try:
        query = db.query(SurveyRun)
        if org_id:
            query = query.filter(SurveyRun.organisation_id == org_id)
        
        runs = query.order_by(SurveyRun.created_at.desc()).limit(limit).all()
        
        results = []
        for run in runs:
            defects_count = db.query(func.count(TrackedDefect.id)).filter(TrackedDefect.run_id == run.run_id).scalar() or 0
            potholes_count = db.query(func.count(TrackedDefect.id)).filter(
                TrackedDefect.run_id == run.run_id,
                TrackedDefect.failure_mode == "Pothole"
            ).scalar() or 0
            
            # Get project/road info
            project_name = None
            road_name = None
            project_id = None
            road_id = None
            
            if run.project_id:
                project = db.query(Project).filter(Project.id == run.project_id).first()
                if project:
                    project_name = project.name
                    project_id = str(project.id)
            
            if run.road_id:
                road = db.query(Road).filter(Road.id == run.road_id).first()
                if road:
                    road_name = road.name
                    road_id = str(road.id)
            
            # Get map matching
            map_match = db.query(MapMatchingRun).filter(MapMatchingRun.run_id == run.run_id).first()
            map_matching = None
            if map_match:
                map_matching = {
                    "id": map_match.id,
                    "run_id": str(map_match.run_id),
                    "matched_road_id": map_match.matched_road_id,
                    "match_status": map_match.match_status,
                    "match_method": map_match.match_method,
                    "centerline_wkt": map_match.centerline_wkt,
                    "centerline_geojson": map_match.centerline_geojson,
                    "snapped_trajectory_wkt": map_match.snapped_trajectory_wkt,
                    "chainage_min_m": map_match.chainage_min_m,
                    "chainage_max_m": map_match.chainage_max_m,
                    "matched_distance_m": map_match.matched_distance_m,
                    "road_length_m": map_match.road_length_m,
                    "max_distance_m": map_match.max_distance_m,
                    "matched_points": map_match.matched_points,
                    "total_points": map_match.total_points,
                    "unmatched_points": map_match.unmatched_points,
                    "snapped_coords": map_match.snapped_coords,
                    "created_at": map_match.created_at.isoformat() if map_match.created_at else None,
                }
            
            results.append({
                "run_id": str(run.run_id),
                "name": run.name,
                "computed_iri": run.computed_iri,
                "final_pci": run.final_pci,
                "roadessy_pci": run.roadessy_pci,
                "astm_pci": run.astm_pci,
                "simple_iri": run.simple_iri,
                "astm_qcm_iri": run.astm_qcm_iri,
                "matched_road_id": run.matched_road_id,
                "total_distance_m": run.distance_m if hasattr(run, 'distance_m') and run.distance_m else 0,
                "distance_m": run.distance_m if hasattr(run, 'distance_m') and run.distance_m else 0,
                "defects_count": defects_count,
                "potholes": potholes_count,
                "map_matching": map_matching,
                "project_name": project_name,
                "road_name": road_name,
                "project_id": project_id,
                "road_id": road_id,
                "created_at": run.created_at.isoformat() if run.created_at else None,
                "processing_metadata": run.processing_metadata,
                # Compute status based on processing_metadata
                "status": "DONE" if run.processing_metadata else "CREATED",
            })
        
        return JSONResponse(status_code=status.HTTP_200_OK, content={"runs": results})
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to list runs: {str(e)}")


@router.get("/all-map-data/", response_class=JSONResponse)
async def get_all_map_data(
    org_id: Optional[str] = Query(None),
    limit: int = Query(1000, ge=1, le=5000),
    bbox: Optional[str] = Query(None, description="Bounding box as minLon,minLat,maxLon,maxLat"),
    db: Session = Depends(get_db),
):
    """Get all map data (roughness segments and potholes) for the map view."""
    try:
        # Build base query for runs in org
        run_query = db.query(SurveyRun)
        if org_id:
            run_query = run_query.filter(SurveyRun.organisation_id == org_id)
        
        runs = run_query.all()
        run_ids = [r.run_id for r in runs]
        
        if not run_ids:
            return JSONResponse(status_code=status.HTTP_200_OK, content={
                "segments": {"type": "FeatureCollection", "features": []},
                "potholes": {"type": "FeatureCollection", "features": []},
                "truncated": {"segments": False, "potholes": False},
                "limit": limit,
            })

        # Pre-load run trajectories for segment geometry computation
        run_trajectories = {}
        for run in runs:
            if run.geom_trajectory:
                try:
                    geom = shapely_wkb.loads(bytes(run.geom_trajectory.data))
                    if geom.geom_type == "LineString":
                        run_trajectories[str(run.run_id)] = list(geom.coords)
                except Exception:
                    pass

        # Get roughness segments (IRISegments)
        iri_segments = db.query(IRISegment).filter(IRISegment.run_id.in_(run_ids)).limit(limit).all()
        
        segment_features = []
        for seg in iri_segments:
            # Get run info
            run = next((r for r in runs if r.run_id == seg.run_id), None)
            if not run:
                continue
                
            # Compute segment geometry by slicing the run trajectory
            coords = [[0, 0], [0, 0]]  # Fallback placeholder
            trajectory_coords = run_trajectories.get(str(seg.run_id))
            if trajectory_coords:
                sliced = _slice_trajectory(trajectory_coords, seg.segment_start_m, seg.segment_end_m)
                if sliced:
                    coords = sliced
            
            segment_features.append({
                "id": str(seg.id),
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords
                },
                "properties": {
                    "run_id": str(seg.run_id),
                    "run_title": run.name,
                    "start_chainage_m": seg.segment_start_m,
                    "end_chainage_m": seg.segment_end_m,
                    "roughness_value": seg.iri_value or seg.simple_iri or seg.astm_qcm_iri or 0,
                    "avg_speed_mps": 0,  # Not stored yet
                    "quality_flags": {
                        "road_condition": seg.classification,
                    }
                }
            })

        # Get potholes from tracked defects
        pothole_defects = db.query(TrackedDefect).filter(
            TrackedDefect.run_id.in_(run_ids),
            TrackedDefect.failure_mode == "pothole"
        ).limit(limit).all()

        pothole_features = []
        for defect in pothole_defects:
            run = next((r for r in runs if r.run_id == defect.run_id), None)
            if not run:
                continue
                
            coords = [0, 0]
            if defect.geom_location:
                try:
                    geom = shapely_wkb.loads(bytes(defect.geom_location.data))
                    coords = [float(geom.x), float(geom.y)]
                except Exception:
                    pass
            
            pothole_features.append({
                "id": str(defect.id),
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": coords
                },
                "properties": {
                    "run_id": str(defect.run_id),
                    "run_title": run.name,
                    "timestamp": defect.created_at.isoformat() if defect.created_at else None,
                    "severity": defect.ghana_rating_1to9 or 1,
                    "confidence": defect.confidence or 0,
                    "image_url": defect.image_path,
                }
            })

        truncated_segments = len(segment_features) >= limit
        truncated_potholes = len(pothole_features) >= limit

        return JSONResponse(status_code=status.HTTP_200_OK, content={
            "segments": {"type": "FeatureCollection", "features": segment_features},
            "potholes": {"type": "FeatureCollection", "features": pothole_features},
            "truncated": {"segments": truncated_segments, "potholes": truncated_potholes},
            "limit": limit,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to fetch map data: {str(e)}")