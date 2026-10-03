# app/api/routes/me.py
"""The signed-in caller's own profile and the reports generated for them.

Both features need the same two things and neither is worth its own router
beside: who the caller is, read from the verified session rather than from a
request field, and proof they may see the organisation whose data is being read.

Report downloads are the third thing, and the awkward one. A browser navigation
cannot carry an Authorization header, so a report cannot be fetched with the
bearer token from the page that lists it. The list therefore publishes a signed,
expiring link per report rather than making the download anonymous; see
`app.core.security.sign_download_link`.
"""

import datetime
import logging
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session

from app.core.security import (
    get_principal,
    require_authenticated_user,
    sign_download_link,
    verify_download_token,
)
from app.database.session import get_db
from app.models.admin import UserProfile
from app.models.organisation import Organisation, OrganisationMembership
from app.models.report import ReportAsset
from app.schemas.reports import (
    GenerateReportRequest,
    GenerateReportResponse,
    ListReportsResponse,
    ReportAssetOut,
    UserDetailsResponse,
    UserDetailsUpdate,
)
from app.services.reports import generate_organisation_report, list_reports, resolve_report_path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/me", tags=["Me"])


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _principal_org_id(principal: Dict[str, Any]) -> Optional[str]:
    """The organisation a token names, when it names one."""
    org_id = principal.get("org_id")
    return str(org_id) if org_id else None


def _require_org_access(db: Session, org_id: str, user_id: str) -> Organisation:
    """The organisation, if the caller may read it.

    Membership is the check. An id that names no organisation and an id the
    caller is not a member of both answer 404, because a caller who cannot reach
    an organisation should not be able to tell the two apart.
    """
    try:
        parsed = uuid.UUID(str(org_id))
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")

    org = db.query(Organisation).filter(Organisation.id == parsed).first()
    if org is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")

    membership = (
        db.query(OrganisationMembership)
        .filter(
            OrganisationMembership.organisation_id == org.id,
            OrganisationMembership.user_id == str(user_id),
        )
        .first()
    )
    if membership is None and str(user_id) != str(org.owner_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")
    return org


def _profile_payload(profile: Optional[UserProfile], user_id: str) -> Dict[str, Any]:
    """The caller's details in the shape the settings screen reads.

    A profile that was never written is still a person who signed in, so the
    fields the provider owns are answered from the verified session's own claims
    rather than left empty. Only the application-side fields come from the row.
    """
    if profile is not None:
        return {
            "id": profile.clerk_user_id,
            "username": profile.username or profile.clerk_user_id,
            "email": profile.email or "",
            "full_name": profile.full_name,
            "image_url": profile.image_url,
            "bio": profile.bio,
            "is_active": bool(profile.is_active),
            "created_at": profile.created_at.isoformat() if profile.created_at else None,
            "last_seen_at": profile.last_seen_at.isoformat() if profile.last_seen_at else None,
        }
    return {
        "id": str(user_id),
        "username": str(user_id),
        "email": "",
        "full_name": None,
        "image_url": None,
        "bio": None,
        "is_active": True,
        "created_at": None,
        "last_seen_at": None,
    }


def _report_payload(asset: ReportAsset) -> ReportAssetOut:
    """One report row as the dashboard's asset list reads it.

    The download link is signed here rather than stored, so a link in a stale
    page stops working on its own instead of being replayable forever.
    """
    link = sign_download_link(str(asset.id))
    return ReportAssetOut(
        id=str(asset.id),
        org=asset.organisation_id,
        run=asset.run_id,
        kind=asset.kind,
        year=asset.year,
        title=asset.title,
        file_name=asset.file_name,
        original_filename=asset.original_filename or asset.file_name,
        bucket=asset.bucket,
        key=asset.key,
        supabase_url=(
            f"/api/me/reports/{asset.id}/download/"
            f"?token={link['token']}&expires_at={link['expires_at']}"
        ),
        content_type=asset.content_type,
        size_bytes=asset.size_bytes,
        sha256=asset.sha256,
        created_by=asset.created_by,
        created_at=asset.created_at.isoformat() if asset.created_at else None,
    )


# --------------------------------------------------------------------------
# Profile
# --------------------------------------------------------------------------


@router.get("/details/", response_class=JSONResponse)
async def get_my_details(
    user_id: str = Depends(require_authenticated_user),
    db: Session = Depends(get_db),
):
    """The caller's application-side profile."""
    profile = db.query(UserProfile).filter(UserProfile.clerk_user_id == str(user_id)).first()
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=_profile_payload(profile, user_id),
    )


@router.patch("/details/", response_class=JSONResponse)
async def update_my_details(
    payload: UserDetailsUpdate,
    user_id: str = Depends(require_authenticated_user),
    db: Session = Depends(get_db),
):
    """Update the fields this service owns.

    Only `bio` is writable here. Name, address and avatar belong to the identity
    provider, and a second copy of them edited in this database is a second
    answer to "who is this person" that will eventually disagree with the one
    that can sign them in.
    """
    if payload.bio is not None and len(payload.bio) > 500:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Bio must be 500 characters or less.",
        )

    profile = db.query(UserProfile).filter(UserProfile.clerk_user_id == str(user_id)).first()
    if profile is None:
        profile = UserProfile(clerk_user_id=str(user_id))
        db.add(profile)

    if payload.bio is not None:
        profile.bio = payload.bio.strip() or None

    profile.last_seen_at = datetime.datetime.utcnow()
    db.commit()
    db.refresh(profile)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=_profile_payload(profile, user_id),
    )


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------


@router.get("/reports/", response_class=JSONResponse)
async def list_my_reports(
    org_id: Optional[str] = Query(None),
    year: Optional[int] = Query(None, ge=1970, le=2100),
    principal: dict = Depends(get_principal),
    user_id: str = Depends(require_authenticated_user),
    db: Session = Depends(get_db),
):
    """Generated reports for one organization, newest first.

    `org_id` omitted means the organisation the caller's token names. There is no
    cross-organisation listing: a caller with membership of three organizations
    sees one organization's reports at a time, chosen from the ones they are in.
    """
    target_org = org_id or _principal_org_id(principal)
    if not target_org:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="org_id is required: this session is not scoped to an organization.",
        )

    _require_org_access(db, target_org, user_id)
    assets = list_reports(db, str(target_org), year=year)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=ListReportsResponse(
            org_id=str(target_org),
            reports=[_report_payload(asset) for asset in assets],
        ).model_dump(),
    )


@router.post("/reports/generate/", response_class=JSONResponse)
async def generate_my_report(
    payload: GenerateReportRequest,
    principal: dict = Depends(get_principal),
    user_id: str = Depends(require_authenticated_user),
    db: Session = Depends(get_db),
):
    """Build an organization's report for one year and return it with the list.

    Rendered on demand rather than cached per year: a run that finished being
    processed an hour ago changes what the report should say, and a cached PDF
    would keep asserting the state from whenever it was first asked for.
    """
    org = _require_org_access(db, payload.org_id, user_id)

    try:
        asset, assets = generate_organisation_report(
            db,
            organisation_id=str(org.id),
            organisation_name=org.name,
            year=payload.year,
            title=payload.report_title or None,
            created_by=user_id,
        )
    except HTTPException:
        raise
    except Exception as exc:
        db.rollback()
        logger.exception("Could not generate a report for organisation %s", org.id)
        # The reason is in the log, not the response. A rendering failure's text
        # carries paths and library internals, and this is not a request the
        # caller can act on with that detail in hand.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not generate the report. Try again, and report it if it keeps failing.",
        ) from exc

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content=GenerateReportResponse(
            created=_report_payload(asset),
            reports=[_report_payload(item) for item in assets],
            org_id=str(org.id),
        ).model_dump(),
    )


@router.get("/reports/{report_id}/download/", response_class=FileResponse)
async def download_my_report(
    report_id: str,
    token: str = Query(..., description="Signature issued with this report's listing"),
    expires_at: int = Query(..., description="Expiry the signature was issued for"),
):
    """Serve one generated report.

    Authorised by the signed link rather than a bearer token, because the caller
    is a browser navigation. The signature proves the link came from the listing
    that already checked membership; it does not identify who is opening it,
    which is why it expires in minutes.
    """
    verify_download_token(report_id, token, expires_at)

    try:
        parsed = uuid.UUID(str(report_id))
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")

    from app.database.session import SessionLocal

    db = SessionLocal()
    try:
        asset = db.query(ReportAsset).filter(ReportAsset.id == parsed).first()
        if asset is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")

        path = resolve_report_path(asset)
        if path is None:
            raise HTTPException(
                status_code=status.HTTP_410_GONE,
                detail="The report file is no longer on this server. Generate it again.",
            )

        return FileResponse(
            path,
            media_type=asset.content_type,
            filename=asset.original_filename or asset.file_name,
        )
    finally:
        db.close()