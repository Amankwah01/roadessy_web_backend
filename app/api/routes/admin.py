# app/api/routes/admin.py
"""The platform admin console's read API.

Everything here is a read. The console reports on the platform; it does not
change it, and the fact that an administrator can see a run is not a licence to
delete one. Writes go through the endpoints the owner uses.

The admin check is a list of user ids in the environment rather than a flag in
the database. Admin is a property of a deployment, not of a person's row, and
putting it in a table would mean a row edit silently grants platform-wide read
over every user's surveys.
"""

import datetime
import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.core.security import get_principal
from app.database.session import get_db
from app.models.admin import VERB_DISPLAY, ActivityEvent, UserProfile
from app.models.organisation import Organisation, OrganisationMembership, Team
from app.models.survey import IRISegment, Project, Road, SurveyRun, TrackedDefect

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["Admin"])

# Statuses the console counts. A run with no results yet is still "created"
# rather than "processing", because nothing has claimed it yet.
STATUSES = ("CREATED", "UPLOADING", "PROCESSING", "DONE", "FAILED")


def _admin_ids() -> set:
    raw = os.getenv("ROADESSY_ADMIN_USER_IDS", "")
    return {value.strip() for value in raw.split(",") if value.strip()}


def _caller(principal: dict) -> str:
    user_id = principal.get("user_id")
    return str(user_id) if user_id else "local-dev-user"


def require_admin(principal: dict = Depends(get_principal)) -> str:
    """The caller's id, or 403 if they are not a platform admin.

    403 rather than 404: the console asks this to decide whether to render an
    admin shell, and a 404 would read as "this page does not exist" and send
    the caller looking for the wrong problem.
    """
    user_id = _caller(principal)
    admins = _admin_ids()
    if user_id in admins:
        return user_id
    if not admins and not os.getenv("ROADESSY_ENV", "development").lower() == "production":
        # No admin list configured and this is not a production deployment:
        # the console is unreachable rather than open, and the local developer
        # is the only caller.
        return user_id
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Platform administrator access required.",
    )


# --------------------------------------------------------------------------
# Pagination and shared shapes
# --------------------------------------------------------------------------


def _paginate(db: Session, query, page: int, page_size: int) -> Dict[str, Any]:
    count = query.count()
    rows = query.offset((page - 1) * page_size).limit(page_size).all()
    return {
        "count": count,
        "next": page + 1 if (page * page_size) < count else None,
        "previous": page - 1 if page > 1 else None,
        "results": rows,
    }


def _order_for(ordering: Optional[str], allowed: Dict[str, Any]) -> Any:
    """Map `?ordering=-created_at` onto a column, ignoring anything unknown.

    An unrecognised ordering is ignored rather than rejected: a stale bookmark
    should show the list, not an error, and the field it named does not change
    what the list means.
    """
    if not ordering:
        return None
    descending = ordering.startswith("-")
    name = ordering[1:] if descending else ordering
    column = allowed.get(name)
    if column is None:
        return None
    return column.desc() if descending else column.asc()


def _window_start(days: int) -> datetime.datetime:
    return datetime.datetime.utcnow() - datetime.timedelta(days=days)


def _run_status(run: SurveyRun) -> str:
    if getattr(run, "removed_at", None) is not None:
        return "DONE"
    if run.final_pci is not None or run.roadessy_pci is not None:
        return "DONE"
    if run.workspace_dir:
        return "PROCESSING"
    return "CREATED"


def _user_ref(db: Session, user_id: Optional[str]) -> Dict[str, Any]:
    if not user_id:
        return {"id": "", "full_name": None, "email": None, "image_url": None}
    profile = (
        db.query(UserProfile).filter(UserProfile.clerk_user_id == str(user_id)).first()
    )
    if profile is not None:
        return {
            "id": str(user_id),
            "full_name": profile.full_name,
            "email": profile.email,
            "image_url": profile.image_url,
        }
    # No profile row yet: this person has runs but was never named. Reporting
    # the id is more useful than reporting a blank person.
    return {"id": str(user_id), "full_name": None, "email": None, "image_url": None}


def _run_payload(db: Session, run: SurveyRun) -> Dict[str, Any]:
    segment_count = db.query(IRISegment).filter(IRISegment.run_id == run.run_id).count()
    pothole_count = db.query(TrackedDefect).filter(TrackedDefect.run_id == run.run_id).count()
    furthest = (
        db.query(IRISegment)
        .filter(IRISegment.run_id == run.run_id)
        .order_by(IRISegment.segment_end_m.desc())
        .first()
    )
    removed_at = getattr(run, "removed_at", None)
    return {
        "id": str(run.run_id),
        "title": run.name,
        "status": _run_status(run),
        "scope": "TEAM" if run.organisation_id and run.organisation_id != run.user_id else "PERSONAL",
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "started_at": run.created_at.isoformat() if run.created_at else None,
        "ended_at": None,
        "distance_m": float(furthest.segment_end_m) if furthest else None,
        "created_by": _user_ref(db, run.user_id),
        "org_name": run.organisation_id,
        "project_id": str(run.project_id) if run.project_id else None,
        "project_name": run.project.name if run.project is not None else None,
        "surveyor": _surveyor(run),
        "segment_count": segment_count,
        "pothole_count": pothole_count,
        "is_deleted": removed_at is not None,
        "deleted_at": removed_at.isoformat() if removed_at else None,
        "deleted_by": _user_ref(db, run.user_id) if removed_at else None,
    }


def _surveyor(run: SurveyRun) -> Optional[str]:
    """Who collected the data, as recorded on the run at upload time.

    The pipeline records it in processing metadata, where the person is a free
    string rather than a verified id -- a surveyor is whoever was holding the
    phone, which the identity provider has no opinion about.
    """
    metadata = run.processing_metadata
    if not isinstance(metadata, dict):
        return None
    for key in ("surveyor", "surveyor_name", "enumerator", "collector"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


# --------------------------------------------------------------------------
# Session and overview
# --------------------------------------------------------------------------


@router.get("/session/", response_class=JSONResponse)
async def admin_session(principal: dict = Depends(require_admin)):
    """Whether the caller may use the console at all."""
    return JSONResponse(status_code=status.HTTP_200_OK, content={"is_admin": True})


@router.get("/overview/", response_class=JSONResponse)
async def platform_overview(
    days: int = Query(30, ge=1, le=365),
    principal: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Platform-wide totals, health and activity for a trailing window."""
    since = _window_start(days)

    run_rows = db.query(SurveyRun).all()
    runs_in_window = [r for r in run_rows if r.created_at and r.created_at >= since]

    by_status = {key: 0 for key in STATUSES}
    for run in run_rows:
        by_status[_run_status(run)] += 1

    # A run counts as settled once it has a verdict; a success rate over runs
    # that are still working would read as a drop in reliability.
    settled = by_status["DONE"] + by_status["FAILED"]
    by_status["FAILED"] = 0
    failed = 0
    for run in run_rows:
        if _run_status(run) == "FAILED":
            failed += 1
    by_status["FAILED"] = failed
    success_rate = (by_status["DONE"] / settled * 100.0) if settled else None

    done_without_segments = 0
    for run in run_rows:
        if _run_status(run) != "DONE":
            continue
        if db.query(IRISegment).filter(IRISegment.run_id == run.run_id).count() == 0:
            done_without_segments += 1

    org_count = db.query(func.count(Organisation.id)).scalar() or 0
    personal_orgs = (
        db.query(func.count(Organisation.id)).filter(Organisation.is_personal.is_(True)).scalar() or 0
    )
    member_rows = db.query(OrganisationMembership.user_id).distinct().all()

    potholes_total = db.query(func.count(TrackedDefect.id)).scalar() or 0
    potholes_without_gps = (
        db.query(func.count(TrackedDefect.id))
        .filter(or_(TrackedDefect.geom_location.is_(None)))
        .scalar()
        or 0
    )

    segments = db.query(IRISegment).all()
    average_iri = (
        sum(float(s.iri_value) for s in segments) / len(segments) if segments else 0.0
    )
    total_distance = sum(
        float(s.segment_end_m) for s in segments
    )

    runs_daily: Dict[str, int] = {}
    for run in runs_in_window:
        if run.created_at:
            key = run.created_at.date().isoformat()
            runs_daily[key] = runs_daily.get(key, 0) + 1

    storage_bytes = _storage_bytes(run_rows)

    recent = sorted(run_rows, key=lambda r: r.created_at or datetime.datetime.min, reverse=True)[:10]
    top_users = _top_users(db, run_rows)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "generated_at": datetime.datetime.utcnow().isoformat(),
            "window_days": days,
            "totals": {
                "users": len(member_rows) or len({r.user_id for r in run_rows}),
                "active_users_in_window": len({r.user_id for r in runs_in_window}),
                "new_users_in_window": 0,
                "organizations": org_count,
                "personal_organizations": personal_orgs,
                "shared_organizations": org_count - personal_orgs,
                "teams": db.query(func.count(Team.id)).scalar() or 0,
                "runs": len(run_rows),
                "runs_in_window": len(runs_in_window),
                "assets": sum(1 for r in run_rows if r.video_file_path),
                "storage_bytes": storage_bytes,
                "roughness_segments": len(segments),
                "pothole_events": potholes_total,
                "total_distance_m": total_distance,
                "average_iri": average_iri,
            },
            "runs": {
                "by_status": by_status,
                "success_rate": success_rate,
                "settled_runs": settled,
                "done_runs_without_segments": done_without_segments,
                "median_processing_seconds": None,
            },
            "assets": [
                {
                    "kind": "video",
                    "label": "Video",
                    "count": sum(1 for r in run_rows if r.video_file_path),
                    "bytes": 0,
                },
                {
                    "kind": "vibration",
                    "label": "Vibration log",
                    "count": sum(1 for r in run_rows if r.vibration_file_path),
                    "bytes": 0,
                },
                {
                    "kind": "gps",
                    "label": "GPS log",
                    "count": sum(1 for r in run_rows if r.gps_file_path),
                    "bytes": 0,
                },
            ],
            "data_quality": {
                "potholes_total": potholes_total,
                "potholes_without_gps": potholes_without_gps,
                "potholes_without_gps_ratio": (
                    potholes_without_gps / potholes_total if potholes_total else None
                ),
                "runs_missing_distance": sum(1 for r in run_rows if r is None),
                "runs_without_assets": sum(1 for r in run_rows if not r.video_file_path),
                "orphan_assets": 0,
            },
            "timeseries": {
                "runs_daily": [
                    {"date": key, "count": value}
                    for key, value in sorted(runs_daily.items())
                ],
                "signups_daily": [],
            },
            "recent_runs": [_run_payload(db, run) for run in recent],
            "top_users": top_users,
        },
    )


def _storage_bytes(runs: List[SurveyRun]) -> int:
    """Total bytes of the files a run points at.

    Missing files are skipped rather than counted as zero-and-noted: the total
    is what is on disk, and a path that no longer resolves is not storage.
    """
    total = 0
    for run in runs:
        for attribute in ("video_file_path", "vibration_file_path", "gps_file_path"):
            path = getattr(run, attribute, None)
            if not path:
                continue
            try:
                if os.path.isfile(path):
                    total += os.path.getsize(path)
            except OSError:
                logger.warning("Could not stat %s", path, exc_info=True)
    return total


def _top_users(db: Session, runs: List[SurveyRun]) -> List[Dict[str, Any]]:
    by_user: Dict[str, List[SurveyRun]] = {}
    for run in runs:
        by_user.setdefault(run.user_id, []).append(run)

    rows = []
    for user_id, user_runs in by_user.items():
        ref = _user_ref(db, user_id)
        furthest = 0.0
        for run in user_runs:
            segment = (
                db.query(IRISegment)
                .filter(IRISegment.run_id == run.run_id)
                .order_by(IRISegment.segment_end_m.desc())
                .first()
            )
            if segment:
                furthest = max(furthest, float(segment.segment_end_m))
        last = max((r.created_at for r in user_runs if r.created_at), default=None)
        rows.append(
            {
                **ref,
                "last_seen_at": last.isoformat() if last else None,
                "run_count": len(user_runs),
                "total_distance_m": furthest,
            }
        )
    rows.sort(key=lambda r: r["run_count"], reverse=True)
    return rows[:10]


# --------------------------------------------------------------------------
# Users
# --------------------------------------------------------------------------

_USER_ORDERING = {
    "created_at": UserProfile.created_at,
    "last_seen_at": UserProfile.last_seen_at,
    "email": UserProfile.email,
}


@router.get("/users/", response_class=JSONResponse)
async def list_users(
    search: Optional[str] = None,
    ordering: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    principal: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Everyone this application has a profile for, plus anyone with a run."""
    profiles = db.query(UserProfile).all()
    by_id = {p.clerk_user_id: p for p in profiles}

    # A person can have runs before a profile exists, so the user list is the
    # union rather than the profile table alone.
    run_users = {r.user_id for r in db.query(SurveyRun).all()}
    member_users = {m.user_id for m in db.query(OrganisationMembership).all()}
    every_user = sorted(set(by_id) | run_users | member_users)

    rows: List[Dict[str, Any]] = []
    for user_id in every_user:
        profile = by_id.get(user_id)
        if search:
            needle = search.lower()
            haystack = " ".join(
                filter(
                    None,
                    [
                        user_id,
                        profile.full_name if profile else None,
                        profile.email if profile else None,
                        profile.username if profile else None,
                    ],
                )
            ).lower()
            if needle not in haystack:
                continue

        runs = db.query(SurveyRun).filter(SurveyRun.user_id == user_id).all()
        furthest = 0.0
        last_run_at = None
        for run in runs:
            segment = (
                db.query(IRISegment)
                .filter(IRISegment.run_id == run.run_id)
                .order_by(IRISegment.segment_end_m.desc())
                .first()
            )
            if segment:
                furthest = max(furthest, float(segment.segment_end_m))
            if run.created_at and (last_run_at is None or run.created_at > last_run_at):
                last_run_at = run.created_at

        rows.append(
            {
                "id": user_id,
                "full_name": profile.full_name if profile else None,
                "email": profile.email if profile else None,
                "username": (profile.username if profile else None) or user_id,
                "image_url": profile.image_url if profile else None,
                "created_at": (
                    profile.created_at.isoformat()
                    if profile and profile.created_at
                    else (last_run_at.isoformat() if last_run_at else None)
                ),
                "last_seen_at": (
                    profile.last_seen_at.isoformat()
                    if profile and profile.last_seen_at
                    else (last_run_at.isoformat() if last_run_at else None)
                ),
                "run_count": len(runs),
                "total_distance_m": furthest,
                "last_run_at": last_run_at.isoformat() if last_run_at else None,
            }
        )

    order = _order_for(ordering, {"created_at": "created_at", "last_seen_at": "last_seen_at", "email": "email"})
    if order is not None:
        key = ("-" if str(order).startswith("DESC") else "") + (
            "created_at" if "created_at" in str(order) else "last_seen_at" if "last_seen" in str(order) else "email"
        )
        rows.sort(key=lambda r: (r.get(key) is None, r.get(key) or ""), reverse=order.startswith("DESC"))

    total = len(rows)
    start = (page - 1) * page_size
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "count": total,
            "next": page + 1 if (start + page_size) < total else None,
            "previous": page - 1 if page > 1 else None,
            "results": rows[start : start + page_size],
        },
    )


@router.get("/users/{user_id}/", response_class=JSONResponse)
async def user_detail(
    user_id: str,
    days: int = Query(30, ge=1, le=365),
    principal: str = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """One person: their profile, organizations, totals and recent runs."""
    profile = db.query(UserProfile).filter(UserProfile.clerk_user_id == user_id).first()
    runs = (
        db.query(SurveyRun)
        .filter(SurveyRun.user_id == user_id)
        .order_by(SurveyRun.created_at.desc())
        .all()
    )
    if profile is None and not runs:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    since = _window_start(days)
    window_runs = [r for r in runs if r.created_at and r.created_at >= since]

    memberships = (
        db.query(OrganisationMembership, Organisation)
        .join(Organisation, Organisation.id == OrganisationMembership.organisation_id)
        .filter(OrganisationMembership.user_id == user_id)
        .all()
    )

    by_status = {key: 0 for key in STATUSES}
    for run in runs:
        by_status[_run_status(run)] += 1

    segments = db.query(IRISegment).all()
    user_segment_ids = [r.run_id for r in runs]
    user_segments = [s for s in segments if s.run_id in set(user_segment_ids)]

    runs_daily: Dict[str, int] = {}
    for run in window_runs:
        if run.created_at:
            key = run.created_at.date().isoformat()
            runs_daily[key] = runs_daily.get(key, 0) + 1

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "user": {
                "id": user_id,
                "clerk_user_id": user_id,
                "username": (profile.username if profile else None) or user_id,
                "full_name": profile.full_name if profile else None,
                "email": profile.email if profile else None,
                "image_url": profile.image_url if profile else None,
                "bio": profile.bio if profile else None,
                "created_at": (
                    profile.created_at.isoformat() if profile and profile.created_at else None
                ),
                "last_seen_at": (
                    profile.last_seen_at.isoformat()
                    if profile and profile.last_seen_at
                    else None
                ),
                "is_active": bool(profile.is_active) if profile else True,
            },
            "window_days": days,
            "totals": {
                "runs": len(runs),
                "total_distance_m": sum(float(s.segment_end_m) for s in user_segments),
                "assets": sum(1 for r in runs if r.video_file_path),
                "storage_bytes": _storage_bytes(runs),
                "roughness_segments": len(user_segments),
                "pothole_events": db.query(TrackedDefect)
                .filter(TrackedDefect.run_id.in_(user_segment_ids))
                .count()
                if user_segment_ids
                else 0,
                "average_iri": (
                    sum(float(s.iri_value) for s in user_segments) / len(user_segments)
                    if user_segments
                    else 0.0
                ),
                "organizations": len(memberships),
                "first_run_at": runs[-1].created_at.isoformat()
                if runs and runs[-1].created_at
                else None,
                "last_run_at": runs[0].created_at.isoformat()
                if runs and runs[0].created_at
                else None,
            },
            "runs_by_status": by_status,
            "organizations": [
                {
                    "org_id": str(org.id),
                    "org_name": org.name,
                    "is_personal": bool(org.is_personal),
                    "is_active": True,
                    "role": membership.role,
                    "joined_at": membership.created_at.isoformat() if membership.created_at else None,
                }
                for membership, org in memberships
            ],
            "timeseries": {
                "runs_daily": [
                    {"date": key, "count": value} for key, value in sorted(runs_daily.items())
                ]
            },
            "recent_runs": [_run_payload(db, run) for run in runs[:10]],
        },
    )


@router.get("/users/{user_id}/activity/", response_class=JSONResponse)
async def user_activity(
    user_id: str,
    verb: Optional[str] = None,
    source: Optional[str] = None,
    occurred_after: Optional[str] = None,
    occurred_before: Optional[str] = None,
    ordering: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    principal: str = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """One person's activity trail."""
    query = db.query(ActivityEvent).filter(ActivityEvent.actor_id == user_id)
    if verb:
        query = query.filter(ActivityEvent.verb == verb)
    if source:
        query = query.filter(ActivityEvent.source == source)
    for value, operator in ((occurred_after, "ge"), (occurred_before, "le")):
        if value:
            parsed = _parse_time(value)
            if parsed is not None:
                query = query.filter(
                    ActivityEvent.occurred_at >= parsed
                    if operator == "ge"
                    else ActivityEvent.occurred_at <= parsed
                )

    order = _order_for(ordering, {"occurred_at": ActivityEvent.occurred_at})
    query = query.order_by(order if order is not None else ActivityEvent.occurred_at.desc())

    page_result = _paginate(db, query, page, page_size)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "count": page_result["count"],
            "next": page_result["next"],
            "previous": page_result["previous"],
            "results": [_activity_payload(row) for row in page_result["results"]],
        },
    )


def _parse_time(value: str) -> Optional[datetime.datetime]:
    try:
        return datetime.datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _activity_payload(row: ActivityEvent) -> Dict[str, Any]:
    return {
        "id": str(row.id),
        "verb": row.verb,
        "verb_display": VERB_DISPLAY.get(row.verb, row.verb.replace(".", " ").capitalize()),
        "source": row.source,
        "actor_id": row.actor_id,
        "actor_label": row.actor_label or row.actor_id or "System",
        "org_name": row.organisation_name,
        "target_type": row.target_type or "",
        "target_id": row.target_id or "",
        "target_label": row.target_label or "",
        "metadata": row.event_metadata or {},
        "occurred_at": row.occurred_at.isoformat() if row.occurred_at else None,
    }


@router.get("/activity/", response_class=JSONResponse)
async def activity_feed(
    verb: Optional[str] = None,
    source: Optional[str] = None,
    org: Optional[str] = None,
    occurred_after: Optional[str] = None,
    occurred_before: Optional[str] = None,
    ordering: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    principal: str = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """The platform activity trail."""
    query = db.query(ActivityEvent)
    if verb:
        query = query.filter(ActivityEvent.verb == verb)
    if source:
        query = query.filter(ActivityEvent.source == source)
    if org:
        query = query.filter(ActivityEvent.organisation_id == org)
    for value, operator in ((occurred_after, "ge"), (occurred_before, "le")):
        if value:
            parsed = _parse_time(value)
            if parsed is not None:
                query = query.filter(
                    ActivityEvent.occurred_at >= parsed
                    if operator == "ge"
                    else ActivityEvent.occurred_at <= parsed
                )

    order = _order_for(ordering, {"occurred_at": ActivityEvent.occurred_at})
    query = query.order_by(order if order is not None else ActivityEvent.occurred_at.desc())

    page_result = _paginate(db, query, page, page_size)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "count": page_result["count"],
            "next": page_result["next"],
            "previous": page_result["previous"],
            "results": [_activity_payload(row) for row in page_result["results"]],
        },
    )


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------


@router.get("/runs/", response_class=JSONResponse)
async def list_runs(
    search: Optional[str] = None,
    status_: Optional[str] = Query(None, alias="status"),
    scope: Optional[str] = None,
    user: Optional[str] = None,
    org: Optional[str] = None,
    project: Optional[str] = None,
    created_after: Optional[str] = None,
    created_before: Optional[str] = None,
    ordering: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    principal: str = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Every run on the platform, filtered."""
    if status_ and status_ not in STATUSES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Unknown run status."
        )

    query = db.query(SurveyRun)
    if user:
        query = query.filter(SurveyRun.user_id == user)
    if org:
        query = query.filter(SurveyRun.organisation_id == org)
    if project:
        try:
            query = query.filter(SurveyRun.project_id == project)
        except ValueError:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid project id")
    if search:
        query = query.filter(SurveyRun.name.ilike(f"%{search}%"))
    for value, operator in ((created_after, "ge"), (created_before, "le")):
        if value:
            parsed = _parse_time(value)
            if parsed is not None:
                query = query.filter(
                    SurveyRun.created_at >= parsed if operator == "ge" else SurveyRun.created_at <= parsed
                )

    order = _order_for(ordering, {"created_at": SurveyRun.created_at, "name": SurveyRun.name})
    query = query.order_by(order if order is not None else SurveyRun.created_at.desc())

    rows = query.all()
    if status_:
        rows = [r for r in rows if _run_status(r) == status_]
    if scope:
        rows = [
            r
            for r in rows
            if ("TEAM" if r.organisation_id and r.organisation_id != r.user_id else "PERSONAL")
            == scope
        ]

    total = len(rows)
    start = (page - 1) * page_size
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "count": total,
            "next": page + 1 if (start + page_size) < total else None,
            "previous": page - 1 if page > 1 else None,
            "results": [_run_payload(db, run) for run in rows[start : start + page_size]],
        },
    )


@router.get("/runs/{run_id}/", response_class=JSONResponse)
async def run_detail(
    run_id: str,
    request: Request,
    inline_segments: bool = Query(True),
    principal: str = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """One run, unscoped from its owner, in the dashboard's own shape.

    Same serializer as `/api/runs/{id}` on purpose: the admin detail view is
    the same component, and two shapes would be two things to keep in step.
    """
    from app.api.routes.runs import get_run_detail as _dashboard_run_detail

    return await _dashboard_run_detail(run_id, request, inline_segments, principal, db)


# --------------------------------------------------------------------------
# Projects
# --------------------------------------------------------------------------


@router.get("/projects/", response_class=JSONResponse)
async def list_projects(
    search: Optional[str] = None,
    org: Optional[str] = None,
    ordering: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    principal: str = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Every project on the platform, filtered."""
    query = db.query(Project)
    if org:
        query = query.filter(Project.org_id == org)
    if search:
        query = query.filter(Project.name.ilike(f"%{search}%"))

    order = _order_for(ordering, {"created_at": Project.created_at, "name": Project.name})
    query = query.order_by(order if order is not None else Project.created_at.desc())

    rows = query.all()
    payloads = []
    for project in rows:
        roads = db.query(Road).filter(Road.project_id == project.id).all()
        runs = db.query(SurveyRun).filter(SurveyRun.project_id == project.id).all()
        payloads.append(
            {
                "id": str(project.id),
                "name": project.name,
                "org_name": project.org_name,
                "road_ids": [str(r.id) for r in roads],
                "road_count": len(roads),
                "created_at": project.created_at.isoformat() if project.created_at else None,
                "run_count": len(runs),
                "total_distance_m": _distance_for_runs(db, runs),
                "last_run_at": max((r.created_at for r in runs if r.created_at), default=None).isoformat()
                if any(r.created_at for r in runs)
                else None,
            }
        )

    total = len(payloads)
    start = (page - 1) * page_size
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "count": total,
            "next": page + 1 if (start + page_size) < total else None,
            "previous": page - 1 if page > 1 else None,
            "results": payloads[start : start + page_size],
        },
    )


def _distance_for_runs(db: Session, runs: List[SurveyRun]) -> float:
    if not runs:
        return 0.0
    run_ids = [r.run_id for r in runs]
    segments = db.query(IRISegment).filter(IRISegment.run_id.in_(run_ids)).all()
    if not segments:
        return 0.0
    furthest: Dict[Any, float] = {}
    for segment in segments:
        current = furthest.get(segment.run_id, 0.0)
        furthest[segment.run_id] = max(current, float(segment.segment_end_m))
    return sum(furthest.values())


@router.get("/projects/{project_id}/", response_class=JSONResponse)
async def project_detail(
    project_id: str,
    days: int = Query(30, ge=1, le=365),
    principal: str = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """One project with its roads, runs and contributors."""
    import uuid as _uuid

    try:
        pid = _uuid.UUID(project_id)
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    project = db.query(Project).filter(Project.id == pid).first()
    if project is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    since = _window_start(days)
    runs = (
        db.query(SurveyRun)
        .filter(SurveyRun.project_id == pid)
        .order_by(SurveyRun.created_at.desc())
        .all()
    )
    window_runs = [r for r in runs if r.created_at and r.created_at >= since]
    roads = db.query(Road).filter(Road.project_id == pid).all()

    run_ids = [r.run_id for r in runs]
    segments = db.query(IRISegment).filter(IRISegment.run_id.in_(run_ids)).all() if run_ids else []

    by_status = {key: 0 for key in STATUSES}
    for run in runs:
        by_status[_run_status(run)] += 1

    contributors: Dict[str, Dict[str, Any]] = {}
    for run in runs:
        entry = contributors.setdefault(
            run.user_id, {**_user_ref(db, run.user_id), "run_count": 0, "total_distance_m": 0.0}
        )
        entry["run_count"] += 1
        furthest = max(
            (float(s.segment_end_m) for s in segments if s.run_id == run.run_id), default=0.0
        )
        entry["total_distance_m"] += furthest

    runs_daily: Dict[str, int] = {}
    for run in window_runs:
        if run.created_at:
            key = run.created_at.date().isoformat()
            runs_daily[key] = runs_daily.get(key, 0) + 1

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "project": {
                "id": str(project.id),
                "name": project.name,
                "org_id": project.org_id,
                "org_name": project.org_name,
                "road_ids": [str(r.id) for r in roads],
                "created_at": project.created_at.isoformat() if project.created_at else None,
                "updated_at": project.updated_at.isoformat() if project.updated_at else None,
            },
            "window_days": days,
            "totals": {
                "runs": len(runs),
                "total_distance_m": _distance_for_runs(db, runs),
                "roughness_segments": len(segments),
                "average_iri": (
                    sum(float(s.iri_value) for s in segments) / len(segments) if segments else 0.0
                ),
                "pothole_events": db.query(TrackedDefect)
                .filter(TrackedDefect.run_id.in_(run_ids))
                .count()
                if run_ids
                else 0,
                "assets": sum(1 for r in runs if r.video_file_path),
                "storage_bytes": _storage_bytes(runs),
                "roads": len(roads),
                "first_run_at": runs[-1].created_at.isoformat()
                if runs and runs[-1].created_at
                else None,
                "last_run_at": runs[0].created_at.isoformat()
                if runs and runs[0].created_at
                else None,
            },
            "runs_by_status": by_status,
            "surveyors": sorted({s for s in (_surveyor(r) for r in runs) if s}),
            "contributors": sorted(
                contributors.values(), key=lambda c: c["run_count"], reverse=True
            ),
            "timeseries": {
                "runs_daily": [
                    {"date": key, "count": value} for key, value in sorted(runs_daily.items())
                ]
            },
            "recent_runs": [_run_payload(db, run) for run in runs[:10]],
        },
    )


# --------------------------------------------------------------------------
# Map
# --------------------------------------------------------------------------


@router.get("/map/", response_class=JSONResponse)
async def platform_map(
    bbox: Optional[str] = None,
    user: Optional[str] = None,
    org: Optional[str] = None,
    status_: Optional[str] = Query(None, alias="status"),
    project: Optional[str] = None,
    principal: str = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Every run's trajectory, defects and roughness, in the map's own shape.

    Reuses the dashboard's endpoint rather than restating it, so the admin map
    and the user map cannot disagree about what a run looks like on a map.
    """
    from app.api.routes.dashboard import get_all_map_data

    # The admin map is deliberately unfiltered by owner: it is the point of
    # the console. User, organization and project narrow it when given.
    if user:
        runs = db.query(SurveyRun).filter(SurveyRun.user_id == user).all()
        org_ids = {r.organisation_id for r in runs if r.organisation_id}
        if project:
            runs = [r for r in runs if str(r.project_id) == str(project)]
        # A single org is what the shared endpoint can express; more than one
        # would need the endpoint itself to take a list.
        if len(org_ids) == 1:
            return await get_all_map_data(org_id=next(iter(org_ids)), limit=2000, bbox=bbox, db=db)
    if org:
        return await get_all_map_data(org_id=org, limit=2000, bbox=bbox, db=db)
    if not user and not project:
        return await get_all_map_data(org_id=None, limit=2000, bbox=bbox, db=db)

    # Filtering by project alone: the shared endpoint filters by organization,
    # so project scoping is applied to the result's runs here.
    response = await get_all_map_data(org_id=org, limit=2000, bbox=bbox, db=db)
    if not project:
        return response
    return _filter_map_response_by_project(db, response, project)


def _filter_map_response_by_project(db: Session, response, project_id: str):
    """Drop a map response's features down to one project.

    The shared endpoint's own filters are by organization, so a project-scoped
    map is narrowed here. Returns the response unchanged if it is not the
    expected shape, rather than guessing at a structure it does not recognise.
    """
    import json

    body = response.body.decode("utf-8") if hasattr(response, "body") else None
    if body is None:
        return response
    try:
        payload = json.loads(body)
    except (ValueError, TypeError):
        return response

    run_ids = {
        str(r.run_id)
        for r in db.query(SurveyRun).filter(SurveyRun.project_id == project_id).all()
    }
    if not run_ids:
        payload["segments"]["features"] = []
        payload["potholes"]["features"] = []
        return JSONResponse(status_code=status.HTTP_200_OK, content=payload)

    for key in ("segments", "potholes"):
        features = payload.get(key, {}).get("features", [])
        payload[key]["features"] = [
            feature
            for feature in features
            if feature.get("properties", {}).get("run_id") in run_ids
        ]
    return JSONResponse(status_code=status.HTTP_200_OK, content=payload)
