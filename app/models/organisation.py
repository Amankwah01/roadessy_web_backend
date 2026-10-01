# app/models/organisation.py
"""Multi-tenant models: organizations, teams, membership and invitations.

These back the `/api/me/organisations/*` and `/api/me/notifications/*` surface.
Ownership is modelled by pointing a membership at a user id and giving it a
role, rather than by columns on the organization itself, so that "who owns
this" and "who is in this" are the same question asked two ways.

User ids are opaque strings from the identity provider, never foreign keys:
this database does not own the user table and must not be the place that
decides who a user is.
"""

import datetime
import uuid

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.database.session import Base

# Roles, most privileged first. An owner can always do what an admin can;
# a member can do neither.
OWNER = "OWNER"
ADMIN = "ADMIN"
MEMBER = "MEMBER"
ORG_ROLES = (OWNER, ADMIN, MEMBER)


def _uuid_pk():
    return Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class Organisation(Base):
    __tablename__ = "organisations"

    id = _uuid_pk()
    name = Column(String, nullable=False)
    slug = Column(String, nullable=True, index=True)
    # A personal workspace belongs to exactly one person and is where every
    # project went before organizations existed. It is the same row with a flag
    # rather than a separate table, so the scoping rules stay in one place.
    is_personal = Column(Boolean, nullable=False, default=False)
    owner_id = Column(String, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    memberships = relationship("OrganisationMembership", backref="organisation", cascade="all, delete-orphan")
    teams = relationship("Team", backref="organisation", cascade="all, delete-orphan")


class OrganisationMembership(Base):
    __tablename__ = "organisation_memberships"

    id = _uuid_pk()
    organisation_id = Column(UUID(as_uuid=True), ForeignKey("organisations.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)
    # Denormalised so a member list does not need a join per row, and so a
    # person's name can change without rewriting every membership.
    user_name = Column(String, nullable=True)
    user_email = Column(String, nullable=True)
    user_image_url = Column(String, nullable=True)
    role = Column(String, nullable=False, default=MEMBER)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("organisation_id", "user_id", name="uq_org_membership_user"),
    )


class Team(Base):
    __tablename__ = "teams"

    id = _uuid_pk()
    organisation_id = Column(UUID(as_uuid=True), ForeignKey("organisations.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    # A team with no lead is a legal state, not an error: the last lead left.
    # Reported as no lead rather than rejecting the membership change that
    # caused it, which would make it impossible to fix.
    lead_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    members = relationship("TeamMember", backref="team", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("organisation_id", "name", name="uq_team_name_per_org"),
    )


class TeamMember(Base):
    __tablename__ = "team_members"

    id = _uuid_pk()
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)
    user_name = Column(String, nullable=True)
    user_email = Column(String, nullable=True)
    user_image_url = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("team_id", "user_id", name="uq_team_member_user"),
    )


class Invitation(Base):
    """An invitation to join an organization or a team.

    Keyed on the email and the scope rather than on a user, because the point
    is to invite someone who does not have an account yet.
    """

    __tablename__ = "invitations"

    id = _uuid_pk()
    organisation_id = Column(UUID(as_uuid=True), ForeignKey("organisations.id", ondelete="CASCADE"), nullable=False, index=True)
    # Null means an invitation to the organization itself; set means a team.
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id", ondelete="CASCADE"), nullable=True, index=True)
    email = Column(String, nullable=False, index=True)
    role = Column(String, nullable=True)
    # Set by the backend when the invitation is issued. The link is only ever
    # handed out through the accept route, never in a list response.
    token = Column(String, nullable=False, unique=True, index=True)
    status = Column(String, nullable=False, default="PENDING")
    invited_by = Column(String, nullable=True)
    # A caller may narrow an invitation to one specific person, in which case
    # only that user id may accept it.
    invited_user_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    accepted_at = Column(DateTime, nullable=True)
    revoked_at = Column(DateTime, nullable=True)


class Notification(Base):
    __tablename__ = "notifications"

    id = _uuid_pk()
    # A notification is addressed to one user, or to an organization when the
    # event is about the organization rather than about one person.
    user_id = Column(String, nullable=True, index=True)
    organisation_id = Column(UUID(as_uuid=True), ForeignKey("organisations.id", ondelete="CASCADE"), nullable=True, index=True)
    kind = Column(String, nullable=False, index=True)
    title = Column(String, nullable=False)
    body = Column(Text, nullable=True)
    # Where clicking it goes, as a dashboard-relative path.
    link = Column(String, nullable=True)
    read_at = Column(DateTime, nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)


class NotificationPreference(Base):
    __tablename__ = "notification_preferences"

    id = _uuid_pk()
    user_id = Column(String, nullable=False, unique=True, index=True)
    # Null means "follow the organization default". Resolved at read time so
    # changing the default moves everyone who never chose.
    in_app_enabled = Column(Boolean, nullable=True)
    email_enabled = Column(Boolean, nullable=True)
    email_digest = Column(String, nullable=True)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
