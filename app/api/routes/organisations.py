# app/api/routes/organisations.py
"""Organizations, membership, teams and invitations.

The caller is never asked who they are: every route reads the verified session
and answers from that, so a request cannot act on behalf of someone else by
asking nicely in a field.

Permission flags are computed per caller and returned on every org and team,
rather than leaving each screen to re-derive the role ladder. The ladder has
rules a role string cannot express -- an owner outranks every admin, a lead's
authority stops at their own team, a personal workspace cannot be shared -- and
a second copy of those rules on the client is a second copy that will drift.
"""

import datetime
import logging
import os
import secrets
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import AliasChoices, BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.security import get_principal
from app.database.session import get_db
from app.models.organisation import (
    ADMIN,
    MEMBER,
    OWNER,
    Invitation,
    Organisation,
    OrganisationMembership,
    Team,
    TeamMember,
)
from app.models.survey import Project

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/me/organisations", tags=["Organisations"])
teams_router = APIRouter(prefix="/api/me/teams", tags=["Organisations"])


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------


def _caller(principal: dict) -> str:
    """The acting user's id.

    In production this is always the verified session. In development there may
    be no session at all, and the routes are then exercised as a single
    well-known local user rather than being unusable.
    """
    user_id = principal.get("user_id")
    if user_id:
        return str(user_id)
    return "local-dev-user"


def _membership(db: Session, org_id: uuid.UUID, user_id: str) -> Optional[OrganisationMembership]:
    return (
        db.query(OrganisationMembership)
        .filter(
            OrganisationMembership.organisation_id == org_id,
            OrganisationMembership.user_id == user_id,
        )
        .first()
    )


def _org_or_404(db: Session, org_id: str) -> Organisation:
    try:
        oid = uuid.UUID(str(org_id))
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")
    org = db.query(Organisation).filter(Organisation.id == oid).first()
    if org is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")
    return org


def _require_member(db: Session, org: Organisation, user_id: str) -> OrganisationMembership:
    membership = _membership(db, org.id, user_id)
    if membership is None:
        # A caller outside the org is told the org does not exist rather than
        # that they may not see it: an id they cannot reach should look the
        # same as one that was never real.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")
    return membership


def _org_permissions(role: Optional[str], is_personal: bool) -> Dict[str, bool]:
    is_owner = role == OWNER
    is_admin = role in (OWNER, ADMIN)
    return {
        "can_manage_org": is_owner,
        "can_manage_members": is_admin,
        "can_manage_teams": is_admin,
        # A personal workspace is one person's, so there is no org-wide
        # authoring to hand out *to others* inside it.
        # The owner can still create projects in their own personal workspace.
        "can_author_projects": is_admin and (not is_personal or is_owner),
    }


def _team_permissions(role: Optional[str], is_lead: bool) -> Dict[str, bool]:
    can_administer = role in (OWNER, ADMIN)
    return {
        "can_manage": can_administer or is_lead,
        "can_set_lead": can_administer,
        "can_delete": can_administer,
        # A lead authors inside their own team only; that limit lives here
        # rather than in the client's copy of the ladder.
        "can_author_projects": can_administer or is_lead,
    }


def _org_payload(db: Session, org: Organisation, membership: Optional[OrganisationMembership]) -> Dict[str, Any]:
    role = membership.role if membership else None
    member_count = (
        db.query(func.count(OrganisationMembership.id))
        .filter(OrganisationMembership.organisation_id == org.id)
        .scalar()
        or 0
    )
    return {
        "id": str(org.id),
        "name": org.name,
        "slug": org.slug,
        "is_active": True,
        "created_at": org.created_at.isoformat() if org.created_at else None,
        "is_personal": bool(org.is_personal),
        "owner": org.owner_id,
        "member_count": member_count,
        "role": role,
        "is_owner": role == OWNER,
        "permissions": _org_permissions(role, bool(org.is_personal)),
    }


# --------------------------------------------------------------------------
# Organizations
# --------------------------------------------------------------------------


@router.get("/", response_class=JSONResponse)
async def list_organisations(
    principal: dict = Depends(get_principal),
    db: Session = Depends(get_db),
):
    """Every organization the caller is inside, personal workspace first."""
    user_id = _caller(principal)
    rows = (
        db.query(Organisation, OrganisationMembership)
        .join(
            OrganisationMembership,
            OrganisationMembership.organisation_id == Organisation.id,
        )
        .filter(OrganisationMembership.user_id == user_id)
        .order_by(Organisation.is_personal.desc(), Organisation.created_at.asc())
        .all()
    )
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=[_org_payload(db, org, membership) for org, membership in rows],
    )


async def _get_or_create_personal_workspace(user_id: str, db: Session) -> Organisation:
    """Get the user's personal workspace, creating it if it doesn't exist."""
    # First check if user already has a personal workspace
    existing = (
        db.query(Organisation)
        .join(OrganisationMembership, OrganisationMembership.organisation_id == Organisation.id)
        .filter(
            OrganisationMembership.user_id == user_id,
            Organisation.is_personal.is_(True),
        )
        .first()
    )
    if existing:
        return existing

    # Create personal workspace
    org = Organisation(
        name="Personal Workspace",
        slug=f"personal-{user_id[:8]}",
        is_personal=True,
        owner_id=user_id,
    )
    db.add(org)
    db.flush()

    membership = OrganisationMembership(
        organisation_id=org.id,
        user_id=user_id,
        role=OWNER,
    )
    db.add(membership)
    db.commit()
    db.refresh(org)
    return org


class SyncOrganisationsRequest(BaseModel):
    """Request to sync organizations.

    The organizations come from the caller rather than being fetched here: the
    identity provider is the frontend's concern, and this service does not call
    it.
    """

    external_organizations: List[Dict[str, str]] = Field(
        default=[],
        # The old name is still accepted so a deployed frontend keeps working
        # across the rename.
        validation_alias=AliasChoices("external_organizations", "clerk_organizations"),
    )


@router.post("/sync", response_class=JSONResponse)
async def sync_organisations(
    payload: SyncOrganisationsRequest,
    principal: dict = Depends(get_principal),
    db: Session = Depends(get_db),
):
    """Sync externally-held organizations and ensure the personal workspace exists.

    Called on first login and whenever the caller's organizations change. The
    organizations arrive in the request because the frontend holds the identity
    provider session; this service does not call the provider itself.
    """
    user_id = _caller(principal)
    if not user_id or user_id == "local-dev-user":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    # Ensure personal workspace exists
    personal_org = await _get_or_create_personal_workspace(user_id, db)

    # Sync the externally-held organizations sent by the frontend
    for external_org in payload.external_organizations:
        external_org_id = external_org.get("id")
        external_org_name = external_org.get("name", "Unnamed Organization")
        external_org_slug = external_org.get("slug")

        if not external_org_id or not external_org_slug:
            continue

        # Check if this org already exists (by slug)
        existing = db.query(Organisation).filter(Organisation.slug == external_org_slug).first()
        if existing:
            # Ensure user is a member
            membership = db.query(OrganisationMembership).filter(
                OrganisationMembership.organisation_id == existing.id,
                OrganisationMembership.user_id == user_id,
            ).first()
            if not membership:
                membership = OrganisationMembership(
                    organisation_id=existing.id,
                    user_id=user_id,
                    role=MEMBER,
                )
                db.add(membership)
        else:
            # Create new organization
            org = Organisation(
                name=external_org_name,
                slug=external_org_slug,
                is_personal=False,
                owner_id=user_id,
            )
            db.add(org)
            db.flush()

            membership = OrganisationMembership(
                organisation_id=org.id,
                user_id=user_id,
                role=OWNER,
            )
            db.add(membership)

    db.commit()

    # Return all organizations including the personal one
    rows = (
        db.query(Organisation, OrganisationMembership)
        .join(
            OrganisationMembership,
            OrganisationMembership.organisation_id == Organisation.id,
        )
        .filter(OrganisationMembership.user_id == user_id)
        .order_by(Organisation.is_personal.desc(), Organisation.created_at.asc())
        .all()
    )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"organisations": [_org_payload(db, org, membership) for org, membership in rows]},
    )


class CreateOrganisationRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    slug: Optional[str] = Field(default=None, max_length=200)


@router.post("/", response_class=JSONResponse)
async def create_organisation(
    payload: CreateOrganisationRequest,
    principal: dict = Depends(get_principal),
    db: Session = Depends(get_db),
):
    """Create a shared organization, with the caller as its owner."""
    user_id = _caller(principal)
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="A name is required.")

    org = Organisation(
        name=name,
        slug=payload.slug,
        is_personal=False,
        owner_id=user_id,
    )
    db.add(org)
    db.flush()

    membership = OrganisationMembership(
        organisation_id=org.id,
        user_id=user_id,
        role=OWNER,
    )
    db.add(membership)
    db.commit()
    db.refresh(org)

    return JSONResponse(status_code=status.HTTP_201_CREATED, content=_org_payload(db, org, membership))


class UpdateOrganisationRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    slug: Optional[str] = Field(default=None, max_length=200)


@router.patch("/{org_id}", response_class=JSONResponse)
async def update_organisation(
    org_id: str,
    payload: UpdateOrganisationRequest,
    principal: dict = Depends(get_principal),
    db: Session = Depends(get_db),
):
    """Rename an organization. Owners only."""
    user_id = _caller(principal)
    org = _org_or_404(db, org_id)
    membership = _require_member(db, org, user_id)
    if membership.role != OWNER:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the owner can change this organization.",
        )

    changed = payload.model_dump(exclude_unset=True)
    if "name" in changed and changed["name"]:
        org.name = changed["name"].strip()
    if "slug" in changed:
        org.slug = changed["slug"]

    org.updated_at = datetime.datetime.utcnow()
    db.commit()
    db.refresh(org)
    return JSONResponse(status_code=status.HTTP_200_OK, content=_org_payload(db, org, membership))


@router.delete("/{org_id}", response_class=JSONResponse)
async def delete_organisation(
    org_id: str,
    principal: dict = Depends(get_principal),
    db: Session = Depends(get_db),
):
    """Remove an organization.

    A personal workspace is not removable: it is where everything lives when
    there is no other org, and deleting it would take the person's work with it.
    """
    user_id = _caller(principal)
    org = _org_or_404(db, org_id)
    membership = _require_member(db, org, user_id)
    if membership.role != OWNER:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the owner can remove this organization.",
        )
    if org.is_personal:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A personal workspace cannot be removed.",
        )

    # Projects point at an org by id alone, so the projects are unlinked rather
    # than deleted. Removing an org should not destroy the surveys inside it.
    for project in db.query(Project).filter(Project.org_id == str(org.id)).all():
        project.visibility = "PERSONAL"
        project.team_id = None
        project.team_name = None

    db.delete(org)
    db.commit()
    return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "deleted", "id": str(org_id)})


class TransferOwnershipRequest(BaseModel):
    user_id: str


@router.post("/{org_id}/transfer-ownership/", response_class=JSONResponse)
async def transfer_ownership(
    org_id: str,
    payload: TransferOwnershipRequest,
    principal: dict = Depends(get_principal),
    db: Session = Depends(get_db),
):
    """Hand an organization to another member, who becomes its owner.

    The outgoing owner stays a member as an admin: losing an organization should
    not also mean being locked out of the work in it.
    """
    caller_id = _caller(principal)
    org = _org_or_404(db, org_id)
    caller_membership = _require_member(db, org, caller_id)
    if caller_membership.role != OWNER:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the owner can transfer ownership.",
        )

    target = _membership(db, org.id, payload.user_id)
    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="That person is not a member of this organization.",
        )

    target.role = OWNER
    caller_membership.role = ADMIN
    org.owner_id = payload.user_id
    org.updated_at = datetime.datetime.utcnow()
    db.commit()

    from app.api.routes.notifications import notify

    notify(
        db,
        user_id=payload.user_id,
        organisation_id=org.id,
        kind="org.ownership_received",
        title="You now own an organization",
        body=f"You are now the owner of {org.name}.",
        link="/dashboard/organization",
    )
    db.commit()
    db.refresh(org)
    return JSONResponse(status_code=status.HTTP_200_OK, content=_org_payload(db, org, target))


# --------------------------------------------------------------------------
# Members
# --------------------------------------------------------------------------


def _member_payload(db: Session, membership: OrganisationMembership) -> Dict[str, Any]:
    # Scoped to this organization: a user may be on teams in other orgs, and
    # those are not visible from this member list.
    teams = (
        db.query(Team)
        .join(TeamMember, TeamMember.team_id == Team.id)
        .filter(
            TeamMember.user_id == membership.user_id,
            Team.organisation_id == membership.organisation_id,
        )
        .all()
    )
    return {
        "id": str(membership.id),
        "user_id": membership.user_id,
        "email": membership.user_email,
        "full_name": membership.user_name,
        "image_url": membership.user_image_url,
        "role": membership.role,
        "is_owner": membership.role == OWNER,
        "joined_at": membership.created_at.isoformat() if membership.created_at else None,
        "teams": [
            {"id": str(team.id), "name": team.name, "role": "LEAD" if team.lead_id == membership.user_id else "MEMBER"}
            for team in teams
        ],
    }


@router.get("/{org_id}/members/", response_class=JSONResponse)
async def list_members(
    org_id: str,
    principal: dict = Depends(get_principal),
    db: Session = Depends(get_db),
):
    """Active members, owner first. Any member may read it."""
    user_id = _caller(principal)
    org = _org_or_404(db, org_id)
    _require_member(db, org, user_id)

    rows = (
        db.query(OrganisationMembership)
        .filter(OrganisationMembership.organisation_id == org.id)
        .order_by(OrganisationMembership.created_at.asc())
        .all()
    )
    # Owner first, then by join order. Done in Python because the sort key is
    # a role comparison rather than a column, and the list is small.
    rows.sort(key=lambda m: 0 if m.role == OWNER else 1)
    return JSONResponse(status_code=status.HTTP_200_OK, content=[_member_payload(db, m) for m in rows])


class UpdateMemberRequest(BaseModel):
    role: str


@router.patch("/{org_id}/members/{membership_id}/", response_class=JSONResponse)
async def update_member(
    org_id: str,
    membership_id: str,
    payload: UpdateMemberRequest,
    principal: dict = Depends(get_principal),
    db: Session = Depends(get_db),
):
    """Change a member's role. Admins and owners only.

    An owner cannot be demoted: an organization with no owner has nobody who
    can transfer it or remove it, and that is not a state to reach by accident.
    """
    caller_id = _caller(principal)
    org = _org_or_404(db, org_id)
    caller_membership = _require_member(db, org, caller_id)
    if caller_membership.role not in (OWNER, ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only owners and admins can change roles.",
        )
    if payload.role not in (OWNER, ADMIN, MEMBER):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Unknown role."
        )

    try:
        mid = uuid.UUID(membership_id)
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Member not found")

    target = (
        db.query(OrganisationMembership)
        .filter(OrganisationMembership.id == mid, OrganisationMembership.organisation_id == org.id)
        .first()
    )
    if target is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Member not found")
    if target.role == OWNER and payload.role != OWNER:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Transfer ownership before changing the owner's role.",
        )
    # An admin cannot mint an owner; only an existing owner can.
    if payload.role == OWNER and caller_membership.role != OWNER:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the owner can make someone else an owner.",
        )

    previous = target.role
    target.role = payload.role
    target.updated_at = datetime.datetime.utcnow()
    db.commit()
    db.refresh(target)

    from app.api.routes.notifications import notify

    notify(
        db,
        user_id=target.user_id,
        organisation_id=org.id,
        kind="org.role_changed",
        title="Your role changed",
        body=f"You are now {payload.role.lower()} of {org.name}.",
        link="/dashboard/organization",
    )
    db.commit()
    del previous
    return JSONResponse(status_code=status.HTTP_200_OK, content=_member_payload(db, target))


@router.delete("/{org_id}/members/{membership_id}/", response_class=JSONResponse)
async def remove_member(
    org_id: str,
    membership_id: str,
    principal: dict = Depends(get_principal),
    db: Session = Depends(get_db),
):
    """Remove someone from an organization. Admins and owners only."""
    caller_id = _caller(principal)
    org = _org_or_404(db, org_id)
    caller_membership = _require_member(db, org, caller_id)
    if caller_membership.role not in (OWNER, ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only owners and admins can remove members.",
        )

    try:
        mid = uuid.UUID(membership_id)
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Member not found")

    target = (
        db.query(OrganisationMembership)
        .filter(OrganisationMembership.id == mid, OrganisationMembership.organisation_id == org.id)
        .first()
    )
    if target is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Member not found")
    if target.role == OWNER:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Transfer ownership before removing the owner.",
        )
    if target.user_id == caller_id and caller_membership.role == ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the owner can remove themselves.",
        )

    removed_user = target.user_id
    db.delete(target)
    # Team memberships do not survive leaving the organization that owns them.
    for team in db.query(Team).filter(Team.organisation_id == org.id).all():
        db.query(TeamMember).filter(
            TeamMember.team_id == team.id, TeamMember.user_id == removed_user
        ).delete()
    db.commit()

    from app.api.routes.notifications import notify

    notify(
        db,
        user_id=removed_user,
        organisation_id=org.id,
        kind="org.removed",
        title=f"You were removed from {org.name}",
        body="You no longer have access to this organization.",
        link=None,
    )
    db.commit()
    return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "removed", "id": membership_id})


# --------------------------------------------------------------------------
# Invitations
# --------------------------------------------------------------------------


def _invitation_payload(invitation: Invitation, org: Organisation) -> Dict[str, Any]:
    created = invitation.created_at or datetime.datetime.utcnow()
    return {
        "id": str(invitation.id),
        "email_address": invitation.email,
        "role": invitation.role or MEMBER,
        "status": invitation.status,
        # Epoch milliseconds: the client compares against Clerk's own
        # invitation timestamps, which are epoch millis, and a unit mismatch
        # here reads as an invitation that arrived in 1970.
        "created_at": int(created.timestamp() * 1000),
        "organisation_id": str(org.id),
        "team_id": str(invitation.team_id) if invitation.team_id else None,
    }


@router.get("/{org_id}/invitations/", response_class=JSONResponse)
async def list_invitations(
    org_id: str,
    principal: dict = Depends(get_principal),
    db: Session = Depends(get_db),
):
    """Pending invitations. Owners and admins only."""
    user_id = _caller(principal)
    org = _org_or_404(db, org_id)
    membership = _require_member(db, org, user_id)
    if membership.role not in (OWNER, ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only owners and admins can see invitations.",
        )

    rows = (
        db.query(Invitation)
        .filter(
            Invitation.organisation_id == org.id,
            Invitation.status == "PENDING",
            Invitation.revoked_at.is_(None),
        )
        .order_by(Invitation.created_at.asc())
        .all()
    )
    return JSONResponse(
        status_code=status.HTTP_200_OK, content=[_invitation_payload(row, org) for row in rows]
    )


class CreateInvitationRequest(BaseModel):
    email: str = Field(min_length=3)
    role: str = MEMBER
    team_id: Optional[str] = None
    user_id: Optional[str] = None


@router.post("/{org_id}/invitations/", response_class=JSONResponse)
async def create_invitation(
    org_id: str,
    payload: CreateInvitationRequest,
    principal: dict = Depends(get_principal),
    db: Session = Depends(get_db),
):
    """Invite someone by email. Owners and admins only."""
    user_id = _caller(principal)
    org = _org_or_404(db, org_id)
    membership = _require_member(db, org, user_id)
    if membership.role not in (OWNER, ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only owners and admins can invite people.",
        )
    if payload.role not in (OWNER, ADMIN, MEMBER):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Unknown role.")
    if payload.role == OWNER and membership.role != OWNER:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the owner can invite someone as an owner.",
        )

    team_id = None
    if payload.team_id:
        try:
            team_id = uuid.UUID(payload.team_id)
        except (ValueError, AttributeError, TypeError):
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid team_id")
        if db.query(Team).filter(Team.id == team_id, Team.organisation_id == org.id).first() is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Team not found")

    email = payload.email.strip().lower()
    already = (
        db.query(Invitation)
        .filter(
            Invitation.organisation_id == org.id,
            Invitation.email == email,
            Invitation.status == "PENDING",
            Invitation.revoked_at.is_(None),
        )
        .first()
    )
    if already is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="That person already has a pending invitation."
        )

    invitation = Invitation(
        organisation_id=org.id,
        team_id=team_id,
        email=email,
        role=payload.role,
        token=secrets.token_urlsafe(32),
        invited_by=user_id,
        invited_user_id=payload.user_id,
    )
    db.add(invitation)
    db.commit()
    db.refresh(invitation)

    return JSONResponse(
        status_code=status.HTTP_201_CREATED, content=_invitation_payload(invitation, org)
    )


@router.delete("/{org_id}/invitations/{invitation_id}/", response_class=JSONResponse)
async def revoke_invitation(
    org_id: str,
    invitation_id: str,
    principal: dict = Depends(get_principal),
    db: Session = Depends(get_db),
):
    """Revoke a pending invitation. Owners and admins only."""
    user_id = _caller(principal)
    org = _org_or_404(db, org_id)
    membership = _require_member(db, org, user_id)
    if membership.role not in (OWNER, ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only owners and admins can revoke invitations.",
        )

    try:
        iid = uuid.UUID(invitation_id)
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invitation not found")

    invitation = (
        db.query(Invitation)
        .filter(Invitation.id == iid, Invitation.organisation_id == org.id)
        .first()
    )
    if invitation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Invitation not found")

    invitation.status = "REVOKED"
    invitation.revoked_at = datetime.datetime.utcnow()
    db.commit()
    return JSONResponse(
        status_code=status.HTTP_200_OK, content={"status": "revoked", "id": invitation_id}
    )


# --------------------------------------------------------------------------
# Teams
# --------------------------------------------------------------------------


def _team_member_payload(member: TeamMember) -> Dict[str, Any]:
    return {
        "id": str(member.id),
        "membership_id": str(member.id),
        "user_id": member.user_id,
        "email": member.user_email,
        "full_name": member.user_name,
        "image_url": member.user_image_url,
        "role": "MEMBER",
        "added_at": member.created_at.isoformat() if member.created_at else None,
    }


def _team_payload(db: Session, team: Team, role: Optional[str], caller_id: Optional[str] = None) -> Dict[str, Any]:
    members = db.query(TeamMember).filter(TeamMember.team_id == team.id).all()
    lead_payload = None
    for member in members:
        if team.lead_id and member.user_id == team.lead_id:
            lead = _team_member_payload(member)
            lead["role"] = "LEAD"
            lead_payload = lead
            break
    is_lead = bool(team.lead_id) and caller_id is not None and team.lead_id == caller_id
    return {
        "id": str(team.id),
        "org_id": str(team.organisation_id),
        "name": team.name,
        "description": team.description,
        # The backend has no separate default-team concept; a team's default
        # status follows from it being a personal workspace's only team.
        "is_default_personal": False,
        "created_at": team.created_at.isoformat() if team.created_at else None,
        "lead": lead_payload,
        "member_count": len(members),
        "permissions": _team_permissions(role, is_lead),
    }


@router.get("/{org_id}/teams/", response_class=JSONResponse)
async def list_teams(
    org_id: str,
    principal: dict = Depends(get_principal),
    db: Session = Depends(get_db),
):
    """Active teams. Any member may read it."""
    user_id = _caller(principal)
    org = _org_or_404(db, org_id)
    membership = _require_member(db, org, user_id)

    teams = (
        db.query(Team)
        .filter(Team.organisation_id == org.id)
        .order_by(Team.created_at.asc())
        .all()
    )
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=[_team_payload(db, team, membership.role, user_id) for team in teams],
    )


class CreateTeamRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: Optional[str] = None


@router.post("/{org_id}/teams/", response_class=JSONResponse)
async def create_team(
    org_id: str,
    payload: CreateTeamRequest,
    principal: dict = Depends(get_principal),
    db: Session = Depends(get_db),
):
    """Create a team. Owners and admins only."""
    user_id = _caller(principal)
    org = _org_or_404(db, org_id)
    membership = _require_member(db, org, user_id)
    if membership.role not in (OWNER, ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Only owners and admins can create teams."
        )

    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="A team needs a name.")

    clash = db.query(Team).filter(Team.organisation_id == org.id, Team.name == name).first()
    if clash is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="A team with that name already exists."
        )

    team = Team(
        organisation_id=org.id,
        name=name,
        description=payload.description,
    )
    db.add(team)
    db.commit()
    db.refresh(team)

    return JSONResponse(
        status_code=status.HTTP_201_CREATED, content=_team_payload(db, team, membership.role, user_id)
    )


def _team_or_404(db: Session, team_id: str) -> Team:
    try:
        tid = uuid.UUID(str(team_id))
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Team not found")
    team = db.query(Team).filter(Team.id == tid).first()
    if team is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Team not found")
    return team


def _require_org_admin(db: Session, team: Team, user_id: str) -> OrganisationMembership:
    membership = _membership(db, team.organisation_id, user_id)
    if membership is None or membership.role not in (OWNER, ADMIN):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Team not found"
        )
    return membership


@teams_router.get("/{team_id}/", response_class=JSONResponse)
async def get_team(
    team_id: str,
    principal: dict = Depends(get_principal),
    db: Session = Depends(get_db),
):
    """One team with its members. Any member of the organization may read it."""
    user_id = _caller(principal)
    team = _team_or_404(db, team_id)
    membership = _membership(db, team.organisation_id, user_id)
    if membership is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Team not found")

    payload = _team_payload(db, team, membership.role, user_id)
    members = db.query(TeamMember).filter(TeamMember.team_id == team.id).all()
    if team.lead_id:
        for member in members:
            if member.user_id == team.lead_id:
                lead = _team_member_payload(member)
                lead["role"] = "LEAD"
                payload["lead"] = lead
                break
    payload["members"] = [
        {**_team_member_payload(member), "role": "LEAD" if team.lead_id and member.user_id == team.lead_id else "MEMBER"}
        for member in members
    ]
    return JSONResponse(status_code=status.HTTP_200_OK, content=payload)


class UpdateTeamRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    description: Optional[str] = None


@teams_router.patch("/{team_id}/", response_class=JSONResponse)
async def update_team(
    team_id: str,
    payload: UpdateTeamRequest,
    principal: dict = Depends(get_principal),
    db: Session = Depends(get_db),
):
    """Rename a team or change its description. Owners, admins and the lead."""
    user_id = _caller(principal)
    team = _team_or_404(db, team_id)
    membership = _membership(db, team.organisation_id, user_id)
    if membership is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Team not found")
    is_lead = bool(team.lead_id) and team.lead_id == user_id
    if membership.role not in (OWNER, ADMIN) and not is_lead:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only owners, admins and the team lead can change this team.",
        )

    changed = payload.model_dump(exclude_unset=True)
    if "name" in changed and changed["name"]:
        team.name = changed["name"].strip()
    if "description" in changed:
        team.description = changed["description"]
    team.updated_at = datetime.datetime.utcnow()
    db.commit()
    db.refresh(team)
    return JSONResponse(status_code=status.HTTP_200_OK, content=_team_payload(db, team, membership.role, user_id))


@teams_router.delete("/{team_id}/", response_class=JSONResponse)
async def delete_team(
    team_id: str,
    principal: dict = Depends(get_principal),
    db: Session = Depends(get_db),
):
    """Delete a team, or refuse and name the projects that still use it.

    A project scoped to a team that no longer exists would silently lose its
    scope, so the caller is told which projects are in the way instead.
    """
    user_id = _caller(principal)
    team = _team_or_404(db, team_id)
    membership = _require_org_admin(db, team, user_id)

    blocking = (
        db.query(Project)
        .filter(Project.team_id == team.id)
        .all()
    )
    if blocking:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "detail": "Projects are still scoped to this team.",
                "projects": [{"id": str(p.id), "name": p.name} for p in blocking],
            },
        )

    db.delete(team)
    db.commit()
    del membership
    return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "deleted", "id": team_id})


class AddTeamMemberRequest(BaseModel):
    user_id: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    image_url: Optional[str] = None


@teams_router.post("/{team_id}/members/", response_class=JSONResponse)
async def add_team_member(
    team_id: str,
    payload: AddTeamMemberRequest,
    principal: dict = Depends(get_principal),
    db: Session = Depends(get_db),
):
    """Put an existing organization member into a team."""
    user_id = _caller(principal)
    team = _team_or_404(db, team_id)
    _require_org_admin(db, team, user_id)

    member_membership = _membership(db, team.organisation_id, payload.user_id)
    if member_membership is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="That person is not a member of this organization.",
        )

    existing = (
        db.query(TeamMember)
        .filter(TeamMember.team_id == team.id, TeamMember.user_id == payload.user_id)
        .first()
    )
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="That person is already on this team."
        )

    member = TeamMember(
        team_id=team.id,
        user_id=payload.user_id,
        user_name=payload.full_name or member_membership.user_name,
        user_email=payload.email or member_membership.user_email,
        user_image_url=payload.image_url or member_membership.user_image_url,
    )
    db.add(member)
    db.commit()
    db.refresh(member)

    from app.api.routes.notifications import notify

    notify(
        db,
        user_id=payload.user_id,
        organisation_id=team.organisation_id,
        kind="team.added",
        title=f"You were added to {team.name}",
        body=f"You are now on the {team.name} team.",
        link="/dashboard/organization",
    )
    db.commit()
    return JSONResponse(
        status_code=status.HTTP_201_CREATED, content=_team_member_payload(member)
    )


@teams_router.delete("/{team_id}/members/{team_member_id}/", response_class=JSONResponse)
async def remove_team_member(
    team_id: str,
    team_member_id: str,
    principal: dict = Depends(get_principal),
    db: Session = Depends(get_db),
):
    """Take someone off a team. Owners, admins and the lead."""
    user_id = _caller(principal)
    team = _team_or_404(db, team_id)
    membership = _membership(db, team.organisation_id, user_id)
    if membership is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Team not found")
    is_lead = bool(team.lead_id) and team.lead_id == user_id
    if membership.role not in (OWNER, ADMIN) and not is_lead:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only owners, admins and the team lead can change this team.",
        )

    try:
        mid = uuid.UUID(team_member_id)
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="That person is not on this team")

    member = (
        db.query(TeamMember)
        .filter(TeamMember.id == mid, TeamMember.team_id == team.id)
        .first()
    )
    if member is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="That person is not on this team")

    removed_user = member.user_id
    db.delete(member)
    if team.lead_id == removed_user:
        team.lead_id = None
    db.commit()

    from app.api.routes.notifications import notify

    notify(
        db,
        user_id=removed_user,
        organisation_id=team.organisation_id,
        kind="team.removed",
        title=f"You were removed from {team.name}",
        body=f"You are no longer on the {team.name} team.",
        link="/dashboard/organization",
    )
    db.commit()
    return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "removed", "id": team_member_id})


class SetLeadRequest(BaseModel):
    user_id: str


@teams_router.put("/{team_id}/lead/", response_class=JSONResponse)
async def set_team_lead(
    team_id: str,
    payload: SetLeadRequest,
    principal: dict = Depends(get_principal),
    db: Session = Depends(get_db),
):
    """Make an existing team member the lead. Owners and admins only."""
    user_id = _caller(principal)
    team = _team_or_404(db, team_id)
    _require_org_admin(db, team, user_id)

    member = (
        db.query(TeamMember)
        .filter(TeamMember.team_id == team.id, TeamMember.user_id == payload.user_id)
        .first()
    )
    if member is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="That person is not on this team."
        )

    team.lead_id = payload.user_id
    team.updated_at = datetime.datetime.utcnow()
    db.commit()
    db.refresh(team)

    from app.api.routes.notifications import notify

    notify(
        db,
        user_id=payload.user_id,
        organisation_id=team.organisation_id,
        kind="team.lead_assigned",
        title=f"You are now the lead of {team.name}",
        body=f"You lead the {team.name} team.",
        link="/dashboard/organization",
    )
    db.commit()

    payload_out = _team_payload(db, team, None, user_id)
    return JSONResponse(status_code=status.HTTP_200_OK, content=payload_out)


@teams_router.delete("/{team_id}/lead/", response_class=JSONResponse)
async def remove_team_lead(
    team_id: str,
    principal: dict = Depends(get_principal),
    db: Session = Depends(get_db),
):
    """Remove the team's lead. Owners and admins only.

    A team with no lead is a legal state: the last lead leaving must not be
    blocked, and the team is simply administered by the organization until
    somebody is made lead again.
    """
    user_id = _caller(principal)
    team = _team_or_404(db, team_id)
    _require_org_admin(db, team, user_id)

    previous = team.lead_id
    team.lead_id = None
    team.updated_at = datetime.datetime.utcnow()
    db.commit()
    db.refresh(team)

    if previous:
        from app.api.routes.notifications import notify

        notify(
            db,
            user_id=previous,
            organisation_id=team.organisation_id,
            kind="team.lead_removed",
            title=f"You are no longer the lead of {team.name}",
            body=f"The {team.name} team no longer has a lead.",
            link="/dashboard/organization",
        )
        db.commit()

    return JSONResponse(status_code=status.HTTP_200_OK, content=_team_payload(db, team, None, user_id))
