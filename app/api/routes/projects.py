# app/api/routes/projects.py
import datetime
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError

from app.database.session import get_db
from app.models.organisation import Organisation
from app.models.survey import SurveyRun, Project, Road, TrackedDefect, MapMatchingRun

router = APIRouter(prefix="/api/me/projects", tags=["Projects & Roads"])

# Roads are also addressed flat, because the dashboard has a road on screen and
# only the road id. Same handlers, different mount point.
roads_router = APIRouter(prefix="/api/me/roads", tags=["Projects & Roads"])


def _geometry_wkt(value):
    if value is None:
        return None
    if hasattr(value, "wkt"):
        return value.wkt
    if isinstance(value, str) and len(value) > 0:
        try:
            from shapely import wkb as shapely_wkb
            from shapely import wkt as shapely_wkt
            geom = shapely_wkb.loads(value, hex=True)
            return shapely_wkt.dumps(geom)
        except Exception:
            pass
    return str(value)


def _resolve_org_name(db: Session, org_id: str) -> str:
    """The organization's display name, from what this database already knows.

    Clerk is the frontend's concern and never called from here, so the name
    normally arrives on the request body. This is the fallback for a caller that
    sent only an id: a sibling project in the same org carries the name, and
    otherwise this database's own organization record does.

    Falls back to the id when nothing knew it. That is still better than an
    empty label, and the caller can see exactly which value it got.
    """
    # 1. A project in the same org already knows the name.
    existing = db.query(Project).filter(Project.org_id == org_id).first()
    if existing is not None and existing.org_name:
        return existing.org_name

    # 2. This database's own organization record.
    if org_id:
        try:
            organisation = (
                db.query(Organisation).filter(Organisation.id == uuid.UUID(org_id)).first()
            )
            if organisation is not None and organisation.name:
                return organisation.name
        except (ValueError, AttributeError, TypeError):
            # Not a UUID, so it cannot be one of ours.
            pass

    # 3. Nothing knew it. The id is still better than an empty label.
    return org_id


@router.get("/", response_class=JSONResponse)
async def list_projects(
    org_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """List projects for the authenticated user, optionally filtered by organization."""
    try:
        query = db.query(Project)
        if org_id:
            query = query.filter(Project.org_id == org_id)
        
        projects = query.order_by(Project.updated_at.desc()).limit(limit).all()
        
        # Get counts for each project
        project_ids = [str(p.id) for p in projects]
        road_counts = {}
        run_counts = {}
        distance_totals = {}
        avg_roughness = {}
        
        if project_ids:
            # Road counts
            road_rows = db.query(Road.project_id, func.count(Road.id)).filter(
                Road.project_id.in_(project_ids)
            ).group_by(Road.project_id).all()
            road_counts = {str(r[0]): r[1] for r in road_rows}
            
            # Run counts and totals
            run_rows = db.query(
                Project.id,
                func.count(SurveyRun.run_id),
                func.coalesce(func.sum(SurveyRun.simple_iri), 0),
                func.coalesce(func.sum(SurveyRun.astm_qcm_iri), 0),
                func.coalesce(func.sum(SurveyRun.computed_iri), 0)
            ).join(
                Road, Road.project_id == Project.id
            ).join(
                SurveyRun, SurveyRun.road_id == Road.id
            ).filter(
                Project.id.in_(project_ids)
            ).group_by(Project.id).all()
            
            for r in run_rows:
                pid = str(r[0])
                run_counts[pid] = r[1]
                # Compute average roughness from available IRI fields
                iri_sum = sum(filter(None, [r[2], r[3], r[4]]))
                iri_count = sum(1 for v in [r[2], r[3], r[4]] if v is not None)
                avg_roughness[pid] = iri_sum / iri_count if iri_count > 0 else 0
                distance_totals[pid] = float(r[2] or 0)  # simple_iri distance
        
        results = []
        for p in projects:
            pid = str(p.id)
            results.append({
                "id": pid,
                "name": p.name,
                "org_id": p.org_id,
                "org_name": p.org_name,
                "visibility": p.visibility,
                "team_id": str(p.team_id) if p.team_id else None,
                "team_name": p.team_name,
                "can_edit": True,  # TODO: implement permission check
                "road_count": road_counts.get(pid, 0),
                "run_count": run_counts.get(pid, 0),
                "total_distance_m": distance_totals.get(pid, 0),
                "avg_roughness": avg_roughness.get(pid, 0),
                "created_at": p.created_at.isoformat() if p.created_at else None,
                "updated_at": p.updated_at.isoformat() if p.updated_at else None,
            })
        
        return JSONResponse(status_code=status.HTTP_200_OK, content={"projects": results})
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to list projects: {str(e)}")


class CreateProjectRequest(BaseModel):
    """Body for creating a project.

    The frontend posts JSON, so this is a body rather than a bag of query
    parameters. `org_name` and the team name come from the client, which reads
    them from Clerk, and are derived server-side only when the caller omits
    them.
    """

    name: str = Field(min_length=1, max_length=200)
    org_id: str
    org_name: Optional[str] = None
    visibility: str = "ORG"
    team_id: Optional[str] = None
    team_name: Optional[str] = None
    description: Optional[str] = None
    created_by: Optional[str] = None


@router.post("/", response_class=JSONResponse)
async def create_project(
    payload: CreateProjectRequest,
    db: Session = Depends(get_db),
):
    """Create a new project."""
    name = payload.name.strip()
    org_id = payload.org_id
    org_name = payload.org_name
    team_id = payload.team_id
    team_name = payload.team_name
    description = payload.description
    created_by = payload.created_by
    visibility = payload.visibility
    if not name:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="A project needs a name."
        )
    # `org_id` is stored as free text, so without this a caller can write an id
    # that no organization route could ever address — an orphan the org API
    # cannot list, and one no error ever surfaces. Every real org id is a UUID
    # from GET /api/me/organisations/, so anything else is refused here rather
    # than becoming a project nobody can find again.
    try:
        uuid.UUID(org_id)
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid org_id",
        )
    try:
        # Map frontend visibility to backend values
        # Frontend sends "ORG" for organization-wide, "TEAM" for team-specific
        # Backend expects "PERSONAL" or "TEAM"
        if visibility == "ORG":
            visibility = "PERSONAL"
        
        # Resolve the organization's name. The client normally sends it, since
        # Clerk is the frontend's source of truth; a sibling project or this
        # database's own org record covers the case where it did not. Falling
        # back to the id happens only when all of those are silent, and is
        # better than refusing to create the project over a label.
        if not org_name:
            org_name = _resolve_org_name(db, org_id)
        
        # Use org_id as created_by fallback
        if not created_by:
            created_by = org_id
        
        # Parse the team id once. A malformed id is the caller's mistake, not a
        # server fault, so it is answered as 422 rather than caught by the
        # blanket handler below and reported as a 500.
        team_uuid = None
        if team_id:
            try:
                team_uuid = uuid.UUID(team_id)
            except (ValueError, AttributeError, TypeError):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Invalid team_id",
                )

        # Get team_name if team_id provided but team_name not given
        if team_uuid is not None and not team_name:
            # Try to find team name from existing projects
            existing_team = db.query(Project).filter(Project.team_id == team_uuid).first()
            if existing_team:
                team_name = existing_team.team_name

        project = Project(
            name=name,
            org_id=org_id,
            org_name=org_name,
            visibility=visibility,
            team_id=team_uuid,
            team_name=team_name,
            description=description,
            created_by=created_by,
        )
        db.add(project)
        db.commit()
        db.refresh(project)
        
        return JSONResponse(status_code=status.HTTP_201_CREATED, content={
            "id": str(project.id),
            "name": project.name,
            "org_id": project.org_id,
            "org_name": project.org_name,
            "visibility": project.visibility,
            "team_id": str(project.team_id) if project.team_id else None,
            "team_name": project.team_name,
            "description": project.description,
            "can_edit": True,
            "road_count": 0,
            "run_count": 0,
            "total_distance_m": 0,
            "avg_roughness": 0,
            "created_at": project.created_at.isoformat() if project.created_at else None,
            "updated_at": project.updated_at.isoformat() if project.updated_at else None,
        })
    except HTTPException:
        # Already a deliberate answer (bad team_id); the blanket handler below
        # would turn it into a 500 and lose the reason.
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create project: {str(e)}")


@router.get("/{project_id}", response_class=JSONResponse)
async def get_project(project_id: str, db: Session = Depends(get_db)):
    """Get a single project by ID."""
    try:
        pid = uuid.UUID(project_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid project_id UUID")
    
    project = db.query(Project).filter(Project.id == pid).first()
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    
    # Get counts
    road_count = db.query(func.count(Road.id)).filter(Road.project_id == pid).scalar() or 0
    run_count = db.query(func.count(SurveyRun.run_id)).join(Road, Road.id == SurveyRun.road_id).filter(Road.project_id == pid).scalar() or 0
    
    return JSONResponse(status_code=status.HTTP_200_OK, content={
        "id": str(project.id),
        "name": project.name,
        "org_id": project.org_id,
        "org_name": project.org_name,
        "visibility": project.visibility,
        "team_id": str(project.team_id) if project.team_id else None,
        "team_name": project.team_name,
        "description": project.description,
        "can_edit": True,
        "road_count": road_count,
        "run_count": run_count,
        "total_distance_m": 0,  # TODO: compute
        "avg_roughness": 0,  # TODO: compute
        "created_at": project.created_at.isoformat() if project.created_at else None,
        "updated_at": project.updated_at.isoformat() if project.updated_at else None,
    })


class UpdateProjectRequest(BaseModel):
    """Partial update for a project. Every field is optional; absent means unchanged.

    The two callers that reach this (rename, and a visibility change) send
    different subsets, so an absent key must not be read as "clear this".
    `team_id: null` is meaningful and clears the team.
    """

    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    description: Optional[str] = None
    visibility: Optional[str] = None
    team_id: Optional[str] = None
    team_name: Optional[str] = None


@router.patch("/{project_id}", response_class=JSONResponse)
async def update_project(
    project_id: str,
    payload: UpdateProjectRequest,
    db: Session = Depends(get_db),
):
    """Update a project."""
    try:
        pid = uuid.UUID(project_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid project_id UUID")
    
    project = db.query(Project).filter(Project.id == pid).first()
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    
    # `exclude_unset` rather than a None check: the caller distinguishes "leave
    # the description alone" from "clear the description", and both arrive as
    # either an absent key or an explicit null.
    changed = payload.model_dump(exclude_unset=True)
    
    if "name" in changed:
        name = (changed["name"] or "").strip()
        if not name:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="A project needs a name."
            )
        project.name = name
    if "description" in changed:
        project.description = changed["description"]
    if changed.get("visibility") is not None:
        # Map frontend "ORG" to backend "PERSONAL"
        visibility = changed["visibility"]
        project.visibility = "PERSONAL" if visibility == "ORG" else visibility
        # Widening or narrowing scope has to move the team with it, or a project
        # left on "ORG" keeps pointing at the team it was just released from.
        if project.visibility == "PERSONAL":
            project.team_id = None
            project.team_name = None
    if "team_id" in changed:
        team_id = changed["team_id"]
        if not team_id:
            project.team_id = None
            project.team_name = None
        else:
            try:
                project.team_id = uuid.UUID(team_id)
            except (ValueError, TypeError):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid team_id"
                )
            if project.visibility != "TEAM":
                project.visibility = "TEAM"
    if changed.get("team_name") is not None:
        project.team_name = changed["team_name"]
    
    project.updated_at = datetime.datetime.utcnow()
    db.commit()
    db.refresh(project)
    
    return JSONResponse(status_code=status.HTTP_200_OK, content={
        "id": str(project.id),
        "name": project.name,
        "org_id": project.org_id,
        "org_name": project.org_name,
        "visibility": project.visibility,
        "team_id": str(project.team_id) if project.team_id else None,
        "team_name": project.team_name,
        "description": project.description,
        "can_edit": True,
        "created_at": project.created_at.isoformat() if project.created_at else None,
        "updated_at": project.updated_at.isoformat() if project.updated_at else None,
    })


@router.delete("/{project_id}", response_class=JSONResponse)
async def delete_project(project_id: str, db: Session = Depends(get_db)):
    """Delete a project."""
    try:
        pid = uuid.UUID(project_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid project_id UUID")
    
    project = db.query(Project).filter(Project.id == pid).first()
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    
    db.delete(project)
    db.commit()
    
    return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "deleted", "id": project_id})


# Roads endpoints
@router.get("/{project_id}/roads", response_class=JSONResponse)
async def list_project_roads(project_id: str, db: Session = Depends(get_db)):
    """List roads in a project."""
    try:
        pid = uuid.UUID(project_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid project_id UUID")
    
    project = db.query(Project).filter(Project.id == pid).first()
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    
    roads = db.query(Road).filter(Road.project_id == pid).order_by(Road.created_at.desc()).all()
    
    # Get run counts and metrics for each road
    road_ids = [str(r.id) for r in roads]
    run_counts = {}
    avg_roughness = {}
    avg_roughness = {}
    defect_counts = {}
    
    if road_ids:
        run_rows = db.query(
            SurveyRun.road_id,
            func.count(SurveyRun.run_id),
            func.coalesce(func.avg(SurveyRun.simple_iri), 0)
        ).filter(
            SurveyRun.road_id.in_(road_ids)
        ).group_by(SurveyRun.road_id).all()
        
        for r in run_rows:
            rid = str(r[0])
            run_counts[rid] = r[1]
            avg_roughness[rid] = float(r[2]) if r[2] else 0
        
        # Get defect counts
        defect_rows = db.query(
            SurveyRun.road_id,
            func.count(TrackedDefect.id)
        ).join(
            TrackedDefect, TrackedDefect.run_id == SurveyRun.run_id
        ).filter(
            SurveyRun.road_id.in_(road_ids)
        ).group_by(SurveyRun.road_id).all()
        
        for d in defect_rows:
            defect_counts[str(d[0])] = d[1]
    
    results = []
    for r in roads:
        rid = str(r.id)
        results.append({
            "id": rid,
            "project": str(r.project_id),
            "project_name": r.project_name,
            "name": r.name,
            "start_node": r.start_node,
            "end_node": r.end_node,
            "surface_type": r.surface_type,
            "description": r.description,
            "run_count": run_counts.get(rid, 0),
            # The project is already loaded above; Road has no back-reference to
            # read it from, and a lazy attribute here would be a query per road.
            "project_visibility": project.visibility,
            "can_edit": True,
            "total_defects": defect_counts.get(rid, 0),
            "avg_roughness": avg_roughness.get(rid, 0),
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
        })
    
    return JSONResponse(status_code=status.HTTP_200_OK, content={"roads": results})


class RoadInput(BaseModel):
    """Body for creating a road. `name` is the only required field; the rest of
    a road's identity is where it starts and ends, which a survey does not
    always know yet."""

    name: str = Field(min_length=1, max_length=200)
    start_node: Optional[str] = None
    end_node: Optional[str] = None
    surface_type: Optional[str] = None
    description: Optional[str] = None
    created_by: Optional[str] = None


@router.post("/{project_id}/roads", response_class=JSONResponse)
async def create_road(
    project_id: str,
    payload: RoadInput,
    db: Session = Depends(get_db),
):
    """Create a new road in a project."""
    try:
        pid = uuid.UUID(project_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid project_id UUID")
    
    project = db.query(Project).filter(Project.id == pid).first()
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="A road needs a name.")
    
    existing = db.query(Road).filter(Road.project_id == pid, Road.name == name).first()
    if existing:
        # The unique constraint would catch this too, but as a 500. A taken name
        # is the person typing a name that exists, not a server fault.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Another road in this project already has that name.",
        )
    
    try:
        road = Road(
            project_id=pid,
            project_name=project.name,
            name=name,
            start_node=payload.start_node,
            end_node=payload.end_node,
            surface_type=payload.surface_type,
            description=payload.description,
            # Roads belong to the project, and the project already records who
            # created it, so a missing created_by is the project's author.
            created_by=payload.created_by or project.created_by or "",
        )
        db.add(road)
        db.commit()
        db.refresh(road)
        
        return JSONResponse(status_code=status.HTTP_201_CREATED, content={
            "id": str(road.id),
            "project": str(road.project_id),
            "project_name": road.project_name,
            "name": road.name,
            "start_node": road.start_node,
            "end_node": road.end_node,
            "surface_type": road.surface_type,
            "description": road.description,
            "run_count": 0,
            "project_visibility": project.visibility,
            "can_edit": True,
            "created_at": road.created_at.isoformat() if road.created_at else None,
            "updated_at": road.updated_at.isoformat() if road.updated_at else None,
        })
    except IntegrityError:
        # Lost a race against the check above; still a taken name, not a fault.
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Another road in this project already has that name.",
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create road: {str(e)}")


class UpdateRoadRequest(BaseModel):
    """Partial update for a road; an absent key means unchanged."""

    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    start_node: Optional[str] = None
    end_node: Optional[str] = None
    surface_type: Optional[str] = None
    description: Optional[str] = None


@router.patch("/{project_id}/roads/{road_id}", response_class=JSONResponse)
async def update_project_road(
    project_id: str,
    road_id: str,
    payload: UpdateRoadRequest,
    db: Session = Depends(get_db),
):
    """Update a road, addressed through the project that owns it."""
    return await _apply_road_update(project_id, road_id, payload, db)


@roads_router.patch("/{road_id}", response_class=JSONResponse)
async def update_road(
    road_id: str,
    payload: UpdateRoadRequest,
    db: Session = Depends(get_db),
):
    """Update a road by id alone.

    The dashboard addresses roads flat (`/api/me/roads/{id}`) because the road
    is already on screen; the project-nested form above is kept for callers that
    have only the project in hand.
    """
    return await _apply_road_update(None, road_id, payload, db)


async def _apply_road_update(
    project_id: Optional[str],
    road_id: str,
    payload: UpdateRoadRequest,
    db: Session,
):
    try:
        rid = uuid.UUID(road_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid road_id UUID")

    query = db.query(Road).filter(Road.id == rid)
    if project_id is not None:
        try:
            query = query.filter(Road.project_id == uuid.UUID(project_id))
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid project_id UUID"
            )
    road = query.first()
    if not road:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Road not found")

    changed = payload.model_dump(exclude_unset=True)

    if "name" in changed:
        name = (changed["name"] or "").strip()
        if not name:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="A road needs a name.")
        clash = (
            db.query(Road)
            .filter(Road.project_id == road.project_id, Road.name == name, Road.id != rid)
            .first()
        )
        if clash:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Another road in this project already has that name.",
            )
        road.name = name

    for field in ("start_node", "end_node", "surface_type", "description"):
        if field in changed:
            setattr(road, field, changed[field])

    road.updated_at = datetime.datetime.utcnow()
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Another road in this project already has that name.",
        )
    db.refresh(road)

    return JSONResponse(status_code=status.HTTP_200_OK, content={
        "id": str(road.id),
        "project": str(road.project_id),
        "project_name": road.project_name,
        "name": road.name,
        "start_node": road.start_node,
        "end_node": road.end_node,
        "surface_type": road.surface_type,
        "description": road.description,
        "can_edit": True,
        "created_at": road.created_at.isoformat() if road.created_at else None,
        "updated_at": road.updated_at.isoformat() if road.updated_at else None,
    })


@router.get("/{project_id}/runs", response_class=JSONResponse)
async def list_project_runs(project_id: str, limit: int = Query(50, ge=1, le=200), db: Session = Depends(get_db)):
    """List runs in a project."""
    try:
        pid = uuid.UUID(project_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid project_id UUID")
    
    project = db.query(Project).filter(Project.id == pid).first()
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    
    runs = db.query(SurveyRun).filter(SurveyRun.project_id == pid).order_by(SurveyRun.created_at.desc()).limit(limit).all()
    
    results = []
    for run in runs:
        # Count defects
        defects_count = db.query(func.count(TrackedDefect.id)).filter(TrackedDefect.run_id == run.run_id).scalar() or 0
        
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
        
        # Get road name
        road_name = None
        if run.road_id:
            road = db.query(Road).filter(Road.id == run.road_id).first()
            if road:
                road_name = road.name
        
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
            "total_distance_m": None,
            "defects_count": defects_count,
            "map_matching": map_matching,
            "project_name": run.project.name if run.project else None,
            "road_name": road_name,
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "processing_metadata": run.processing_metadata,
        })
    
    return JSONResponse(status_code=status.HTTP_200_OK, content={"runs": results})


@router.get("/{project_id}/roads/{road_id}", response_class=JSONResponse)
async def get_road(project_id: str, road_id: str, db: Session = Depends(get_db)):
    """Get a single road by ID."""
    try:
        pid = uuid.UUID(project_id)
        rid = uuid.UUID(road_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid UUID")
    
    road = db.query(Road).filter(Road.id == rid, Road.project_id == pid).first()
    if not road:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Road not found")
    
    return JSONResponse(status_code=status.HTTP_200_OK, content={
        "id": str(road.id),
        "project": str(road.project_id),
        "project_name": road.project_name,
        "name": road.name,
        "start_node": road.start_node,
        "end_node": road.end_node,
        "surface_type": road.surface_type,
        "description": road.description,
        "road_geom": _geometry_wkt(road.road_geom),
        "can_edit": True,
        "created_at": road.created_at.isoformat() if road.created_at else None,
        "updated_at": road.updated_at.isoformat() if road.updated_at else None,
    })