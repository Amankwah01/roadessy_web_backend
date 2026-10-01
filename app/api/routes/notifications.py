# app/api/routes/notifications.py
"""Notifications, the unread summary, and per-user preferences.

Notifications are written by the thing that happened rather than by a client,
so a run that finishes while nobody is looking still produces one. Reading is
strictly per user: a notification addressed to an organization is visible to
its members, but never to anyone outside it.
"""

import datetime
import logging
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.core.security import get_principal
from app.database.session import get_db
from app.models.organisation import (
    Invitation,
    Notification,
    NotificationPreference,
    Organisation,
    OrganisationMembership,
    TeamMember,
)
from app.services.notifications import preference_enabled

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/me/notifications", tags=["Notifications"])


def _caller(principal: dict) -> str:
    user_id = principal.get("user_id")
    if user_id:
        return str(user_id)
    return "local-dev-user"


def _caller_email(principal: dict) -> Optional[str]:
    """The caller's address, when the token carried one.

    An invitation is addressed to an email, so this is how one is matched to
    the person accepting it. Absent in development and for any token without an
    email claim, which then matches only invitations that named no address.
    """
    email = principal.get("email")
    if isinstance(email, str) and email.strip():
        return email.strip().lower()
    return None


def _addressed_to(principal: dict):
    """SQL filter matching invitations that belong to this caller."""
    user_id = _caller(principal)
    email = _caller_email(principal)
    if email:
        return or_(
            Invitation.invited_user_id == user_id,
            Invitation.email == email,
        )
    return Invitation.invited_user_id == user_id


def notify(
    db: Session,
    *,
    user_id: Optional[str],
    organisation_id: Optional[uuid.UUID],
    kind: str,
    title: str,
    body: Optional[str],
    link: Optional[str],
) -> Notification:
    """Record one notification. Does not commit; the caller batches the write.

    Suppressed when the recipient has turned this kind off, so that turning
    something off means it is not stored rather than stored and hidden.
    """
    if user_id and not preference_enabled(db, user_id, kind):
        logger.debug("Skipping %s notification for %s (turned off)", kind, user_id)
        return Notification(user_id=None, kind=kind, title=title)

    notification = Notification(
        user_id=user_id,
        organisation_id=organisation_id,
        kind=kind,
        title=title,
        body=body,
        link=link,
    )
    db.add(notification)
    return notification


def _visible_query(db: Session, user_id: str):
    """Notifications this user may see: their own, plus their organizations'."""
    org_ids = [
        row[0]
        for row in db.query(OrganisationMembership.organisation_id)
        .filter(OrganisationMembership.user_id == user_id)
        .all()
    ]
    clause = [Notification.user_id == user_id]
    if org_ids:
        clause.append(Notification.organisation_id.in_(org_ids))
    return db.query(Notification).filter(or_(*clause))


def _payload(
    db: Session,
    notification: Notification,
    user_id: str,
    org_names: Dict[str, str],
) -> Dict[str, Any]:
    org_name = "Your account"
    if notification.organisation_id is not None:
        org_name = org_names.get(str(notification.organisation_id), "An organization")

    # An organization-scoped notification only carries a usable link while the
    # reader is still inside that organization; otherwise it would land them on
    # a page about somewhere they were just removed from.
    link = notification.link
    if link and notification.organisation_id is not None:
        if str(notification.organisation_id) not in org_names:
            link = None

    return {
        "id": str(notification.id),
        "kind": notification.kind,
        "title": notification.title,
        "body": notification.body or "",
        "link": link,
        "org_id": str(notification.organisation_id) if notification.organisation_id else None,
        "org_name": org_name,
        "actor_label": "Someone",
        "is_read": notification.read_at is not None,
        "created_at": notification.created_at.isoformat() if notification.created_at else None,
        "is_mine": notification.user_id == user_id if notification.user_id else False,
    }


def _org_names(db: Session, user_id: str) -> Dict[str, str]:
    rows = (
        db.query(Organisation.id, Organisation.name)
        .join(OrganisationMembership, OrganisationMembership.organisation_id == Organisation.id)
        .filter(OrganisationMembership.user_id == user_id)
        .all()
    )
    return {str(row[0]): row[1] for row in rows}


@router.get("/", response_class=JSONResponse)
async def list_notifications(
    notification_filter: str = Query("all", alias="filter"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    principal: dict = Depends(get_principal),
    db: Session = Depends(get_db),
):
    """One page of notifications, newest first, with the unread count."""
    user_id = _caller(principal)
    if notification_filter not in ("all", "unread"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Unknown filter."
        )

    query = _visible_query(db, user_id)
    unread_query = query.filter(Notification.read_at.is_(None))

    total = query.count()
    unread_count = unread_query.count()
    if notification_filter == "unread":
        query = unread_query

    rows = (
        query.order_by(Notification.created_at.desc()).offset(offset).limit(limit).all()
    )
    names = _org_names(db, user_id)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "count": total,
            "unread_count": unread_count,
            "results": [_payload(db, row, user_id, names) for row in rows],
        },
    )


@router.get("/summary/", response_class=JSONResponse)
async def notification_summary(
    principal: dict = Depends(get_principal),
    db: Session = Depends(get_db),
):
    """What the header bell shows: an unread count and the newest few."""
    user_id = _caller(principal)
    names = _org_names(db, user_id)

    query = _visible_query(db, user_id)
    unread_count = query.filter(Notification.read_at.is_(None)).count()
    latest = query.order_by(Notification.created_at.desc()).limit(5).all()

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "unread_count": unread_count,
            "latest": [_payload(db, row, user_id, names) for row in latest],
            # Invitations live in the identity provider, not here, and this API
            # has no access to it. Saying so beats reporting a confident zero.
            "invitations": [],
            "invitations_available": False,
        },
    )


@router.get("/invitations/", response_class=JSONResponse)
async def list_invitations(
    principal: dict = Depends(get_principal),
    db: Session = Depends(get_db),
):
    """Organization invitations still waiting for this user to answer."""

    rows = (
        db.query(Invitation, Organisation)
        .join(Organisation, Organisation.id == Invitation.organisation_id)
        .filter(
            Invitation.status == "PENDING",
            Invitation.revoked_at.is_(None),
            _addressed_to(principal),
        )
        .order_by(Invitation.created_at.desc())
        .all()
    )

    results = [
        {
            "id": str(invitation.id),
            "org_id": str(org.id),
            "org_name": org.name,
            "org_image_url": None,
            "role": invitation.role or "MEMBER",
            "email": invitation.email,
            "created_at": invitation.created_at.isoformat() if invitation.created_at else None,
            "expires_at": None,
        }
        for invitation, org in rows
    ]
    return JSONResponse(
        status_code=status.HTTP_200_OK, content={"available": True, "results": results}
    )


@router.post("/{notification_id}/read/", response_class=JSONResponse)
async def mark_read(
    notification_id: str,
    principal: dict = Depends(get_principal),
    db: Session = Depends(get_db),
):
    """Mark one notification read. Someone else's is not found, not refused."""
    user_id = _caller(principal)
    try:
        nid = uuid.UUID(notification_id)
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Notification not found")

    names = _org_names(db, user_id)
    row = _visible_query(db, user_id).filter(Notification.id == nid).first()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Notification not found")

    if row.read_at is None:
        row.read_at = datetime.datetime.utcnow()
        db.commit()
        db.refresh(row)
    return JSONResponse(status_code=status.HTTP_200_OK, content=_payload(db, row, user_id, names))


@router.post("/read-all/", response_class=JSONResponse)
async def mark_all_read(
    principal: dict = Depends(get_principal),
    db: Session = Depends(get_db),
):
    """Mark every visible notification read, and report how many changed."""
    user_id = _caller(principal)
    updated = (
        _visible_query(db, user_id)
        .filter(Notification.read_at.is_(None))
        .update({Notification.read_at: datetime.datetime.utcnow()}, synchronize_session=False)
    )
    db.commit()
    return JSONResponse(status_code=status.HTTP_200_OK, content={"updated": int(updated or 0)})


@router.post("/invitations/{invitation_id}/dismiss/", response_class=JSONResponse)
async def dismiss_invitation(
    invitation_id: str,
    principal: dict = Depends(get_principal),
    db: Session = Depends(get_db),
):
    """Hide an invitation from the caller's list without answering it.

    The identity provider has no decline, so the invitation stays open on that
    side. This only stops it being shown again here, and is recorded as revoked
    so it stops appearing in the owner's pending list too.
    """
    user_id = _caller(principal)
    try:
        iid = uuid.UUID(invitation_id)
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invitation not found")

    invitation = (
        db.query(Invitation)
        .filter(Invitation.id == iid, Invitation.status == "PENDING")
        .first()
    )
    if invitation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invitation not found")

    # A caller may only dismiss an invitation addressed to them.
    if invitation.invited_user_id and invitation.invited_user_id != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invitation not found")
    email = _caller_email(principal)
    if not invitation.invited_user_id and invitation.email and invitation.email != email:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invitation not found")

    invitation.status = "REVOKED"
    invitation.revoked_at = datetime.datetime.utcnow()
    db.commit()
    return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "dismissed", "id": invitation_id})


class AcceptInvitationRequest(BaseModel):
    org_id: str


@router.post("/invitations/accepted/", response_class=JSONResponse)
async def confirm_invitation_accepted(
    payload: AcceptInvitationRequest,
    principal: dict = Depends(get_principal),
    db: Session = Depends(get_db),
):
    """Record that the caller joined an organization in the browser.

    Membership is created by the identity provider, not by this API, so this
    only reconciles the two sides: it marks a matching pending invitation
    accepted and makes sure the organization is in this database.

    Answers the organization's id, or null while it cannot be matched yet. The
    caller shows a switcher either way rather than an error, because a brief
    lag between the provider and this database is normal rather than a failure.
    """
    user_id = _caller(principal)

    invitation = (
        db.query(Invitation)
        .filter(
            Invitation.status == "PENDING",
            Invitation.revoked_at.is_(None),
            _addressed_to(principal),
        )
        .order_by(Invitation.created_at.desc())
        .first()
    )
    if invitation is None:
        return JSONResponse(status_code=status.HTTP_200_OK, content={"org_id": None})

    organisation = db.query(Organisation).filter(Organisation.id == invitation.organisation_id).first()
    if organisation is None:
        return JSONResponse(status_code=status.HTTP_200_OK, content={"org_id": None})

    # Creating the membership idempotently: this endpoint can be called twice
    # for one acceptance, and a duplicate membership would show the person
    # twice in the member list.
    existing = (
        db.query(OrganisationMembership)
        .filter(
            OrganisationMembership.organisation_id == organisation.id,
            OrganisationMembership.user_id == user_id,
        )
        .first()
    )
    if existing is None:
        db.add(
            OrganisationMembership(
                organisation_id=organisation.id,
                user_id=user_id,
                role=invitation.role or "MEMBER",
            )
        )
        if invitation.team_id:
            db.add(
                TeamMember(team_id=invitation.team_id, user_id=user_id)
            )

    invitation.status = "ACCEPTED"
    invitation.accepted_at = datetime.datetime.utcnow()
    db.commit()

    return JSONResponse(
        status_code=status.HTTP_200_OK, content={"org_id": str(organisation.id)}
    )


class UpdatePreferencesRequest(BaseModel):
    run_processed: Optional[bool] = None
    run_failed: Optional[bool] = None
    team_changes: Optional[bool] = None
    project_runs: Optional[bool] = None


@router.get("/preferences/", response_class=JSONResponse)
async def get_preferences(
    principal: dict = Depends(get_principal),
    db: Session = Depends(get_db),
):
    """What this user has turned off. Absent means "follow the default"."""
    user_id = _caller(principal)
    row = db.query(NotificationPreference).filter(NotificationPreference.user_id == user_id).first()
    defaults = {"run_processed": True, "run_failed": True, "team_changes": True, "project_runs": True}
    if row is None:
        return JSONResponse(status_code=status.HTTP_200_OK, content=defaults)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={key: bool(getattr(row, f"{key}_enabled", True)) for key in defaults},
    )


@router.patch("/preferences/", response_class=JSONResponse)
async def update_preferences(
    payload: UpdatePreferencesRequest,
    principal: dict = Depends(get_principal),
    db: Session = Depends(get_db),
):
    """Turn individual notification kinds on or off."""
    user_id = _caller(principal)
    row = db.query(NotificationPreference).filter(NotificationPreference.user_id == user_id).first()
    if row is None:
        row = NotificationPreference(user_id=user_id)
        db.add(row)

    changed = payload.model_dump(exclude_unset=True)
    for key, value in changed.items():
        if value is not None:
            setattr(row, f"{key}_enabled", bool(value))

    row.updated_at = datetime.datetime.utcnow()
    db.commit()
    db.refresh(row)

    defaults = {"run_processed": True, "run_failed": True, "team_changes": True, "project_runs": True}
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={key: bool(getattr(row, f"{key}_enabled", True)) for key in defaults},
    )
