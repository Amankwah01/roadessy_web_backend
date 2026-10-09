# app/api/routes/dashboard.py
import math
import uuid
import logging
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, text

from app.database.session import get_db
from app.models.survey import SurveyRun, TrackedDefect, IRISegment, MapMatchingRun, Project, Road, PCISegment
from shapely import wkb as shapely_wkb
from shapely import wkt as shapely_wkt

router = APIRouter(prefix="/api/me/dashboard", tags=["Dashboard"])
logger = logging.getLogger(__name__)


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
            geometry_data = geom.data
            if isinstance(geometry_data, str):
                return geometry_data
            return shapely_wkt.dumps(shapely_wkb.loads(bytes(geometry_data)))
        return str(geom)
    except Exception:
        return None


def _geometry_distance_m(geom) -> Optional[float]:
    """Calculate geodesic length for a stored LineString or MultiLineString."""
    geometry_wkt = _geometry_wkt(geom)
    if not geometry_wkt:
        return None
    try:
        geometry = shapely_wkt.loads(geometry_wkt)
        lines = [geometry] if geometry.geom_type == "LineString" else list(geometry.geoms)
        lengths = []
        for line in lines:
            coordinates = list(line.coords)
            lengths.extend(
                _haversine_distance_m(start, end)
                for start, end in zip(coordinates, coordinates[1:])
            )
        return sum(lengths) if lengths else None
    except Exception:
        logger.warning("Could not calculate distance from stored run geometry", exc_info=True)
        return None


def _run_distance_m(db: Session, run: SurveyRun) -> Optional[float]:
    """Resolve one run's distance from chainage, matching, or GPS geometry."""
    segment_end = (
        db.query(func.max(IRISegment.segment_end_m))
        .filter(IRISegment.run_id == run.run_id)
        .scalar()
    )
    if segment_end is not None and float(segment_end) > 0:
        return float(segment_end)

    matching = db.query(MapMatchingRun).filter(MapMatchingRun.run_id == run.run_id).first()
    if matching:
        matched_distance = next(
            (
                distance
                for distance in (matching.matched_distance_m, matching.road_length_m)
                if distance is not None and float(distance) > 0
            ),
            None,
        )
        if matched_distance is not None and float(matched_distance) > 0:
            return float(matched_distance)

    return _geometry_distance_m(run.geom_trajectory)


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

        # Average IRI
        avg_iri = db.query(func.coalesce(func.avg(SurveyRun.simple_iri), 0)).filter(
            SurveyRun.organisation_id == org_id if org_id else True
        ).scalar() or 0
        
        # Average PCI
        avg_pci = db.query(func.coalesce(func.avg(SurveyRun.roadessy_pci), 0)).filter(
            SurveyRun.organisation_id == org_id if org_id else True
        ).scalar() or 0
        
        # Average ASTM PCI
        avg_astm_pci = db.query(func.coalesce(func.avg(SurveyRun.astm_pci), 0)).filter(
            SurveyRun.organisation_id == org_id if org_id else True
        ).scalar() or 0
        
        # Segment chainages are cumulative, so each run's distance is its
        # furthest station, not the sum of every segment (which can overlap).
        distance_rows = (
            db.query(IRISegment.run_id, func.max(IRISegment.segment_end_m))
            .join(SurveyRun, SurveyRun.run_id == IRISegment.run_id)
            .filter(SurveyRun.organisation_id == org_id if org_id else True)
            .group_by(IRISegment.run_id)
            .all()
        )
        distance_by_run = {
            run_id: float(distance_m)
            for run_id, distance_m in distance_rows
            if distance_m is not None and float(distance_m) > 0
        }

        matching_query = db.query(MapMatchingRun).join(
            SurveyRun, SurveyRun.run_id == MapMatchingRun.run_id
        )
        if org_id:
            matching_query = matching_query.filter(SurveyRun.organisation_id == org_id)
        for match in matching_query.all():
            current = distance_by_run.get(match.run_id, 0)
            if current > 0:
                continue
            matched_distance = next(
                (
                    distance
                    for distance in (match.matched_distance_m, match.road_length_m)
                    if distance is not None and float(distance) > 0
                ),
                None,
            )
            if matched_distance is not None and float(matched_distance) > 0:
                distance_by_run[match.run_id] = float(matched_distance)

        # Geometry is only needed as a fallback when neither IRI chainage nor
        # map matching has published a usable length.
        all_run_ids = {
            row[0]
            for row in db.query(SurveyRun.run_id).filter(
                SurveyRun.organisation_id == org_id if org_id else True
            ).all()
        }
        missing_geometry_ids = [
            run_id for run_id in all_run_ids if distance_by_run.get(run_id, 0) <= 0
        ]
        if missing_geometry_ids:
            geometry_query = db.query(SurveyRun.run_id, SurveyRun.geom_trajectory).filter(
                SurveyRun.run_id.in_(missing_geometry_ids)
            )
            for run_id, geometry in geometry_query.all():
                geometry_distance = _geometry_distance_m(geometry)
                if geometry_distance is not None and geometry_distance > 0:
                    distance_by_run[run_id] = geometry_distance

        known_distance_total = sum(distance_by_run.values())
        has_complete_distance_data = len(distance_by_run) == len(all_run_ids)
        total_distance = (
            known_distance_total
            if has_complete_distance_data
            else None
        )

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
                "distance_m": distance_by_run.get(run.run_id),
                "project_id": project_id,
                "project_name": project_name,
                "road_id": road_id,
                "road_name": road_name,
                "simple_iri": run.simple_iri,
                "astm_qcm_iri": run.astm_qcm_iri,
                "roadessy_pci": run.roadessy_pci,
                "astm_pci": run.astm_pci,
                "final_pci": run.final_pci,
            })
        
        # Recent sensor assets (placeholder - no asset table yet)
        recent_sensor_assets = []
        
        return JSONResponse(status_code=status.HTTP_200_OK, content={
            "processing_count": processing_count,
            "pothole_count": pothole_count,
            "total_distance_m": float(total_distance) if total_distance is not None else None,
            "average_iri": float(avg_iri) if avg_iri else 0,
            "average_pci": float(avg_pci) if avg_pci else 0,
            "average_astm_pci": float(avg_astm_pci) if avg_astm_pci else 0,
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

            distance_m = _run_distance_m(db, run)
            
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
                "total_distance_m": distance_m,
                "distance_m": distance_m,
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
        # Prefer matched road centerline if available, otherwise use GPS trajectory
        run_trajectories = {}
        run_centerlines = {}  # Store full centerline for slicing
        for run in runs:
            # First try to get matched road centerline
            map_match = db.query(MapMatchingRun).filter(MapMatchingRun.run_id == run.run_id).first()
            if map_match and map_match.centerline_geojson:
                centerline_geojson = map_match.centerline_geojson
                if centerline_geojson and centerline_geojson.get("type") == "LineString":
                    coords = centerline_geojson.get("coordinates", [])
                    if coords:
                        run_centerlines[str(run.run_id)] = coords
                        # Also use for trajectory (the matched road)
                        run_trajectories[str(run.run_id)] = coords
                        continue
            
            # Fallback to GPS trajectory
            if run.geom_trajectory:
                try:
                    geom = shapely_wkb.loads(bytes(run.geom_trajectory.data))
                    if geom.geom_type == "LineString":
                        run_trajectories[str(run.run_id)] = list(geom.coords)
                except Exception:
                    pass

        # Get roughness segments (IRISegments)
        iri_segments = db.query(IRISegment).filter(IRISegment.run_id.in_(run_ids)).limit(limit).all()

        # Get PCI segments for the same runs
        pci_segments = db.query(PCISegment).filter(PCISegment.run_id.in_(run_ids)).limit(limit).all()
        # Index PCI segments by (run_id, segment_start_m, segment_end_m) for lookup
        pci_by_run_and_chainage = {}
        for pci in pci_segments:
            key = (str(pci.run_id), round(pci.segment_start_m, 1), round(pci.segment_end_m, 1))
            pci_by_run_and_chainage[key] = pci

        segment_features = []
        for seg in iri_segments:
            # Get run info
            run = next((r for r in runs if r.run_id == seg.run_id), None)
            if not run:
                continue
                
            # Look up PCI segment
            pci_key = (str(seg.run_id), round(seg.segment_start_m, 1), round(seg.segment_end_m, 1))
            pci_seg = pci_by_run_and_chainage.get(pci_key)

            # Compute segment geometry by slicing the run trajectory
            trajectory_coords = run_trajectories.get(str(seg.run_id))
            if not trajectory_coords:
                continue  # Skip segments without valid trajectory
            
            sliced = _slice_trajectory(trajectory_coords, seg.segment_start_m, seg.segment_end_m)
            if not sliced:
                continue  # Skip if slicing failed
            
            # Validate coordinates are not [0,0] (invalid fallback)
            if all(c[0] == 0 and c[1] == 0 for c in sliced):
                continue
            
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
                    "roughness_value": seg.simple_iri,
                    "simple_iri": seg.simple_iri,
                    "roadessy_pci_segment": pci_seg.roadessy_pci if pci_seg else None,
                    "avg_speed_mps": None,
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
                
            if not defect.geom_location:
                continue  # Skip defects without geometry
            
            try:
                geom = shapely_wkb.loads(bytes(defect.geom_location.data))
                coords = [float(geom.x), float(geom.y)]
            except Exception:
                continue  # Skip if geometry parsing fails
            
            # Skip invalid [0,0] coordinates
            if coords[0] == 0 and coords[1] == 0:
                continue
            
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


@router.get("/road/{run_id}", response_class=JSONResponse)
async def get_road_detail(
    run_id: str,
    db: Session = Depends(get_db),
):
    """Get detailed road information for a run, used by the map popup.
    
    Returns data matching the RoadInfo type from the frontend:
    - PCI/IRI metrics
    - Defect counts by type
    - IRI segments
    - Map matching results
    - Road geometry (trajectory)
    """
    try:
        run_uuid = uuid.UUID(run_id)
    except Exception:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid run_id UUID")

    run = db.query(SurveyRun).filter(SurveyRun.run_id == run_uuid).first()
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

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

    # Get IRI segments
    segments = db.query(IRISegment).filter(IRISegment.run_id == run_uuid).order_by(IRISegment.segment_start_m).all()
    pci_segments = db.query(PCISegment).filter(PCISegment.run_id == run_uuid).all()
    pci_by_chainage = {
        (round(segment.segment_start_m, 1), round(segment.segment_end_m, 1)): segment
        for segment in pci_segments
    }
    
    segment_data = []
    for seg in segments:
        pci_segment = pci_by_chainage.get(
            (round(seg.segment_start_m, 1), round(seg.segment_end_m, 1))
        )
        segment_data.append({
            "start_m": float(seg.segment_start_m),
            "end_m": float(seg.segment_end_m),
            "iri": float(seg.iri_value),
            "simple_iri": float(seg.simple_iri) if seg.simple_iri is not None else None,
            "astm_qcm_iri": float(seg.astm_qcm_iri) if seg.astm_qcm_iri is not None else None,
            "astm_pci_segment": float(pci_segment.astm_pci) if pci_segment and pci_segment.astm_pci is not None else None,
            "roadessy_pci_segment": float(pci_segment.roadessy_pci) if pci_segment and pci_segment.roadessy_pci is not None else None,
            "classification": seg.classification,
        })

    # Get defects and count by type
    defects = db.query(TrackedDefect).filter(TrackedDefect.run_id == run_uuid).all()
    
    defect_counts = {
        "total_defects": len(defects),
        "pothole_count": 0,
        "crack_count": 0,
        "rutting_count": 0,
        "patching_count": 0,
        "ravelling_count": 0,
        "bleeding_count": 0,
        "subgrade_failure_count": 0,
    }
    
    for defect in defects:
        mode = (defect.failure_mode or "").lower()
        if mode == "pothole":
            defect_counts["pothole_count"] += 1
        elif mode == "crack":
            defect_counts["crack_count"] += 1
        elif mode == "rutting":
            defect_counts["rutting_count"] += 1
        elif mode == "patching":
            defect_counts["patching_count"] += 1
        elif mode == "ravelling":
            defect_counts["ravelling_count"] += 1
        elif mode == "bleeding":
            defect_counts["bleeding_count"] += 1
        elif mode == "subgrade_failure":
            defect_counts["subgrade_failure_count"] += 1

    # Get map matching
    map_match = db.query(MapMatchingRun).filter(MapMatchingRun.run_id == run_uuid).first()
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

    # Get road geometry (trajectory) - prefer matched road centerline if available
    road_geometry = None
    if map_match and map_match.centerline_geojson:
        # Use the matched road centerline (snapped to road network)
        road_geometry = map_match.centerline_geojson
    elif run.geom_trajectory:
        # Fallback to original GPS trajectory
        try:
            if hasattr(run.geom_trajectory, 'data'):
                geom = shapely_wkb.loads(bytes(run.geom_trajectory.data))
            else:
                geom = shapely_wkt.loads(str(run.geom_trajectory))
            if geom.geom_type == "LineString":
                road_geometry = {
                    "type": "LineString",
                    "coordinates": [[float(c[0]), float(c[1])] for c in geom.coords]
                }
        except Exception:
            pass

    return JSONResponse(status_code=status.HTTP_200_OK, content={
        "road_id": road_id or str(run_uuid),
        "road_name": road_name or run.name or "Unknown Road",
        "simple_iri": run.simple_iri,
        "astm_qcm_iri": run.astm_qcm_iri,
        "astm_pci": run.astm_pci,
        "roadessy_pci": run.roadessy_pci,
        "total_distance_m": _run_distance_m(db, run),
        **defect_counts,
        "run_id": str(run_uuid),
        "road_geometry": road_geometry,
        "map_matching": map_matching,
        "iri_segments": segment_data,
    })