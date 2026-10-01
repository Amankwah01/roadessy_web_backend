# app/api/routes/runs.py
"""Run-facing endpoints for the dashboard.

The `/analysis/*` router is the pipeline's own surface: it answers questions
about processing. This router answers questions a person asks about one run —
what is on it, remove it, give me the report — which is a different concern and
a different set of fields.

`GET /api/runs/{run_id}` is the one that is genuinely awkward. It is the only
run endpoint that is neither a list nor an artifact download, and it is the one
that assembles a run from four tables, so the assembly lives here rather than
being spread across the pipeline's routes.
"""

import html
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import ASTM_PCI_DISABLED_REASON, ASTM_PCI_ENABLED
from app.database.session import get_db
from app.models.survey import IRISegment, SurveyRun, TrackedDefect
from app.services.astm import ASTMD6433PCI

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Runs"])
me_router = APIRouter(prefix="/api/me/runs", tags=["Runs"])


def _parse_uuid(value: str, label: str):
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Invalid {label} UUID"
        )


def _wkt_to_point(wkt: Optional[str]):
    """First coordinate of a WKT geometry as [lon, lat], or None."""
    if not wkt:
        return None
    try:
        from shapely import wkt as shapely_wkt

        geom = shapely_wkt.loads(wkt)
        centroid = geom.centroid
        return [float(centroid.x), float(centroid.y)]
    except Exception:
        # A malformed or unparseable geometry should cost the map one point,
        # not the whole run detail page.
        logger.warning("Could not parse run geometry as WKT", exc_info=True)
        return None


def _severity_percent(deduct_value: Optional[float]) -> float:
    """Map a PCI deduct value onto the 0-100 scale the map renders.

    Deduct is points lost, so it inverts: a full deduct is the worst case and
    should be 100.
    """
    if deduct_value is None:
        return 0.0
    try:
        return max(0.0, min(100.0, float(deduct_value) * 10.0))
    except (TypeError, ValueError):
        return 0.0


def _roughness_feature_collection(db: Session, run_uuid, run_title: str) -> Dict[str, Any]:
    """IRI segments as a GeoJSON FeatureCollection, one feature per segment.

    Segments have no geometry of their own, so each is a short line across the
    run's own trajectory: the segment's position is its chainage along the
    route, and a zero-length placeholder would be invisible on the map.
    """
    segments = (
        db.query(IRISegment)
        .filter(IRISegment.run_id == run_uuid)
        .order_by(IRISegment.segment_start_m)
        .all()
    )
    if not segments:
        return {"type": "FeatureCollection", "features": []}

    run = db.query(SurveyRun).filter(SurveyRun.run_id == run_uuid).first()
    line = []
    if run is not None and run.geom_trajectory is not None:
        try:
            from shapely import wkt as shapely_wkt

            line = list(shapely_wkt.loads(run.geom_trajectory).coords)
        except Exception:
            logger.warning("Could not read run trajectory for roughness geometry", exc_info=True)
            line = []

    total = float(segments[-1].segment_end_m) or 1.0

    def _point_at(distance_m: float):
        if not line:
            # No trajectory to place segments along. A degenerate pair of
            # points keeps the feature valid GeoJSON; the map skips it.
            return [0.0, 0.0]
        fraction = max(0.0, min(1.0, float(distance_m) / total))
        index = min(len(line) - 1, int(round(fraction * (len(line) - 1))))
        return [float(line[index][0]), float(line[index][1])]

    features = []
    for segment in segments:
        start = float(segment.segment_start_m)
        end = float(segment.segment_end_m)
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [_point_at(start), _point_at(end)],
                },
                "properties": {
                    "run_id": str(run_uuid),
                    "run_title": run_title,
                    "segment_start_m": start,
                    "segment_end_m": end,
                    "iri": float(segment.iri_value),
                    "simple_iri": segment.simple_iri,
                    "astm_qcm_iri": segment.astm_qcm_iri,
                    "astm_pci_segment": segment.astm_pci_segment,
                    "roadessy_pci_segment": segment.roadessy_pci_segment,
                    "classification": segment.classification,
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _pothole_feature_collection(db: Session, run_uuid, run_title: str) -> Dict[str, Any]:
    """Tracked defects as a GeoJSON FeatureCollection."""
    rows = db.execute(
        text(
            "SELECT id, track_id, failure_mode, deduct_value, max_area_m2, "
            "max_depth_mm, mean_depth_mm, image_path, "
            "ST_AsText(geom_location) AS geom_wkt, created_at "
            "FROM tracked_defects WHERE run_id = :rid ORDER BY id"
        ),
        {"rid": str(run_uuid)},
    ).fetchall()

    features = []
    for row in rows:
        position = _wkt_to_point(row.geom_wkt)
        if position is None:
            continue
        features.append(
            {
                "type": "Feature",
                "id": row.id,
                "geometry": {"type": "Point", "coordinates": position},
                "properties": {
                    "id": row.id,
                    "run_id": str(run_uuid),
                    "run_title": run_title,
                    "track_id": row.track_id,
                    "failure_mode": row.failure_mode,
                    "severity": _severity_percent(row.deduct_value),
                    "deduct_value": row.deduct_value,
                    "max_area_m2": row.max_area_m2,
                    "max_depth_mm": row.max_depth_mm,
                    "mean_depth_mm": row.mean_depth_mm,
                    "timestamp": row.created_at.isoformat() if row.created_at else None,
                    "confidence": 1.0,
                    "image_url": (
                        f"/analysis/runs/{run_uuid}/defects/{row.track_id}/image"
                        if row.image_path and row.track_id is not None
                        else None
                    ),
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _public_base_url(request: Optional[Request]) -> str:
    """Origin this API is reachable at, for URLs handed to a server-side caller.

    Prefers the address the request actually arrived on, so a dev server on
    127.0.0.1 and a deployed host both publish an address that works, rather
    than a hardcoded one that is right in only one of them.
    """
    if request is not None:
        return str(request.base_url).rstrip("/")
    configured = os.environ.get("ROADESSY_PUBLIC_BASE_URL", "").rstrip("/")
    if configured:
        return configured
    return "http://127.0.0.1:8000"


@router.get("/api/runs/{run_id}", response_class=JSONResponse)
async def get_run_detail(
    run_id: str,
    request: Request,
    inline_segments: bool = Query(
        True,
        description=(
            "Embed the roughness segments in the response. false answers with "
            "roughness_geojson_url instead, for a detail page that renders "
            "before the (much larger) segment collection arrives."
        ),
    ),
    db: Session = Depends(get_db),
):
    """One run, assembled for the detail page."""
    run_uuid = _parse_uuid(run_id, "run_id")

    run = db.query(SurveyRun).filter(SurveyRun.run_id == run_uuid).first()
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    project_name = run.project.name if run.project is not None else None
    road_name = run.road.name if run.road is not None else None
    title = run.name or str(run_uuid)

    project_id = str(run.project_id) if run.project_id else None
    road_id = str(run.road_id) if run.road_id else None

    defect_count = (
        db.query(TrackedDefect).filter(TrackedDefect.run_id == run_uuid).count()
    )
    distance_m = None
    furthest = (
        db.query(IRISegment)
        .filter(IRISegment.run_id == run_uuid)
        .order_by(IRISegment.segment_end_m.desc())
        .first()
    )
    if furthest is not None:
        distance_m = float(furthest.segment_end_m)

    created_at = run.created_at
    updated_at = run.created_at
    if run.processing_metadata:
        try:
            metadata = run.processing_metadata
            if isinstance(metadata, dict):
                for key in ("finished_at", "completed_at", "updated_at"):
                    value = metadata.get(key)
                    if isinstance(value, str):
                        updated_at = datetime.fromisoformat(value)
                        break
        except Exception:
            logger.warning("Could not read processing_metadata timestamps", exc_info=True)

    is_personal = run.organisation_id is None or run.user_id == run.organisation_id
    status_value = "DONE" if run.final_pci is not None or run.roadessy_pci is not None else "CREATED"

    payload: Dict[str, Any] = {
        "id": str(run_uuid),
        "run_id": str(run_uuid),
        "scope": "PERSONAL" if is_personal else "TEAM",
        "title": title,
        "name": title,
        "status": status_value,
        "started_at": created_at.isoformat() if created_at else None,
        "ended_at": updated_at.isoformat() if updated_at else None,
        "distance_m": distance_m,
        "avg_roughness": run.computed_iri if run.computed_iri is not None else run.simple_iri,
        "pothole_counts": defect_count,
        "external_project_id": project_id,
        "external_road_id": road_id,
        "project_id": project_id,
        "road_id": road_id,
        "project_name": project_name,
        "road_name": road_name,
        "created_at": created_at.isoformat() if created_at else datetime.now(timezone.utc).isoformat(),
        "updated_at": updated_at.isoformat() if updated_at else datetime.now(timezone.utc).isoformat(),
        "astm_pci": run.astm_pci,
        "astm_pci_enabled": ASTM_PCI_ENABLED,
        "astm_pci_disabled_reason": None if ASTM_PCI_ENABLED else ASTM_PCI_DISABLED_REASON,
        "astm_qcm_iri": run.astm_qcm_iri,
        "simple_iri": run.simple_iri,
        "roadessy_pci": run.roadessy_pci,
        "astm_quality_flags": run.astm_quality_flags,
        "astm_upsampling_info": run.astm_upsampling_info,
        "potholes": _pothole_feature_collection(db, run_uuid, title),
        "roughness_geojson_url": None,
    }

    if inline_segments:
        payload["all_roughness_segments"] = _roughness_feature_collection(db, run_uuid, title)
    else:
        # The caller fetches this separately and from the server, where a
        # relative path has nothing to resolve against, so it is published as an
        # absolute URL. It is a plain GET with no Authorization header, which
        # only resolves while the API accepts anonymous callers; once it
        # requires a token this has to move behind the authenticated helper.
        payload["all_roughness_segments"] = None
        payload["roughness_geojson_url"] = (
            f"{_public_base_url(request)}/api/runs/{run_uuid}/roughness.geojson"
        )

    return JSONResponse(status_code=status.HTTP_200_OK, content=payload)


@router.get("/api/runs/{run_id}/roughness.geojson", response_class=JSONResponse)
async def get_run_roughness_geojson(run_id: str, db: Session = Depends(get_db)):
    """A run's IRI segments as GeoJSON, addressed directly so the detail page
    can fetch them without the run record."""
    run_uuid = _parse_uuid(run_id, "run_id")
    run = db.query(SurveyRun).filter(SurveyRun.run_id == run_uuid).first()
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=_roughness_feature_collection(db, run_uuid, run.name or str(run_uuid)),
    )


class ReportRequest(BaseModel):
    """Nothing to configure yet; kept so the route can take a body later
    without the call sites changing."""


@me_router.post("/{run_id}/report/", response_class=JSONResponse)
async def generate_run_report(
    run_id: str,
    payload: Optional[ReportRequest] = None,
    db: Session = Depends(get_db),
):
    """Build this run's report and return where to find it.

    Regenerated on every call rather than served from a cache: a run's segments
    and defects are written after the upload completes, so a report built
    moments after the run arrived would describe an incomplete run.
    """
    run_uuid = _parse_uuid(run_id, "run_id")
    run = db.query(SurveyRun).filter(SurveyRun.run_id == run_uuid).first()
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    from app.api.routes.analysis import _run_workspace_candidates

    workspace = _run_workspace_candidates(str(run_uuid))
    if not workspace or not workspace[0].exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="This run has no stored output to report on yet.",
        )

    run_dir = workspace[0]
    report_name = f"run_report_{run_uuid}.html"
    report_path = run_dir / report_name

    defect_count = db.query(TrackedDefect).filter(TrackedDefect.run_id == run_uuid).count()
    segments = db.query(IRISegment).filter(IRISegment.run_id == run_uuid).all()
    distance_m = max((float(s.segment_end_m) for s in segments), default=0.0)

    def _row(label: str, value: Any) -> str:
        return f"<tr><th style='text-align:left;padding:4px 12px'>{label}</th><td style='padding:4px 12px'>{value if value is not None else '-'}</td></tr>"

    rows = "".join(
        [
            _row("Run", run.name),
            _row("Run ID", str(run_uuid)),
            _row("Created", run.created_at.isoformat() if run.created_at else None),
            _row("Distance (m)", round(distance_m, 2)),
            _row(
                "PCI (D6433-derived, provisional)",
                round(run.astm_pci, 2)
                if run.astm_pci is not None
                else "not published (PCI muted)",
            ),
            _row("Roadessy PCI", round(run.roadessy_pci, 2) if run.roadessy_pci is not None else None),
            _row("Simple IRI", round(run.simple_iri, 3) if run.simple_iri is not None else None),
            _row("QCM IRI (m/km)", round(run.astm_qcm_iri, 3) if run.astm_qcm_iri is not None else None),
            _row("Defects", defect_count),
        ]
    )
    limitations = ASTMD6433PCI.LIMITATIONS
    reasons = "".join(
        "<li style='margin:0.4rem 0'>{reason}</li>".format(
            reason=html.escape(reason))
        for reason in limitations["reasons"]
    )
    if run.astm_pci is None:
        note = (
            "<p><strong>No D6433-derived PCI is reported for this run.</strong> "
            f"{html.escape(ASTM_PCI_DISABLED_REASON)}</p>"
        )
    else:
        note = (
            "<p>This is a provisional index for relative condition screening and "
            "deterioration tracking. It is not "
            f"{html.escape(limitations['not'])}, and it should not be cited as one."
            "</p>"
            f"<p>Method: {html.escape(limitations['method'])}. "
            f"Curves: {html.escape(limitations['curves'])}.</p>"
            "<p>Three things limit what this number can be used for:</p>"
            f"<ol>{reasons}</ol>"
            f"<p>Band labels are {html.escape(limitations['ratings_are'])}.</p>"
        )
    document = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Run report {run_uuid}</title></head>"
        "<body style=\"font-family:system-ui,sans-serif;margin:2rem\">"
        f"<h1>Survey report</h1><table style='border-collapse:collapse'>{rows}</table>"
        "<div style='max-width:52rem;margin-top:2.5rem;font-size:0.85rem;color:#444'>"
        "<strong>About the D6433-derived PCI</strong>"
        f"{note}"
        "</div>"
        "</body></html>"
    )

    try:
        report_path.write_text(document, encoding="utf-8")
    except OSError as error:
        logger.exception("Could not write run report for %s", run_uuid)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to write run report: {error}",
        )

    # Storage objects are left alone where deleting them is irreversible, so the
    # response describes a regenerated file rather than a guaranteed download.
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "url": f"/api/me/runs/{run_uuid}/report/download/",
            "file_name": report_name,
            "size_bytes": report_path.stat().st_size,
        },
    )


@me_router.get("/{run_id}/report/download/")
async def download_run_report(run_id: str):
    """Serve the most recently generated report for a run."""
    from app.api.routes.analysis import _resolve_run_artifact

    run_uuid = _parse_uuid(run_id, "run_id")
    report_name = f"run_report_{run_uuid}.html"
    report_path = _resolve_run_artifact(str(run_uuid), report_name)
    if report_path is None or not os.path.exists(report_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No report has been generated for this run yet.",
        )
    return FileResponse(report_path, media_type="text/html", filename=report_name)


@me_router.delete("/{run_id}/", response_class=JSONResponse)
async def delete_run(run_id: str, db: Session = Depends(get_db)):
    """Remove a run from its owner's view.

    Soft on the data: the row stays, marked removed, so the admin console can
    still account for the upload and a run can be un-removed. Stored artifacts
    are left alone, because deleting them is irreversible where this is not.
    """
    run_uuid = _parse_uuid(run_id, "run_id")
    run = db.query(SurveyRun).filter(SurveyRun.run_id == run_uuid).first()
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    if getattr(run, "removed_at", None) is not None:
        return JSONResponse(
            status_code=status.HTTP_200_OK, content={"status": "removed", "run_id": str(run_uuid)}
        )

    run.removed_at = datetime.datetime.utcnow()
    db.commit()

    return JSONResponse(
        status_code=status.HTTP_200_OK, content={"status": "removed", "run_id": str(run_uuid)}
    )
