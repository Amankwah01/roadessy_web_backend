# app/models/admin.py
"""User profiles and the activity trail the admin console reads.

There is no user table: identity lives in the provider, and `user_id` here is
the same opaque id the provider issued. A profile is the application's own
record about a person -- their display name, when they were last seen -- and is
created the first time something is learned about them rather than on sign-up,
because the application has no sign-up of its own.

Activity rows are written by the things that happen, not by the admin console.
That is the point of them: the console is a reader of history, and a history
only exists if something recorded it at the time.
"""

import datetime
import uuid

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Index, String, Text, JSON
from sqlalchemy.dialects.postgresql import UUID

from app.database.session import Base

# Verbs the application records. Kept as data rather than as an enum so that
# adding one is a code change in the place that emits it, not a schema
# migration for every historical row.
RUN_CREATED = "run.created"
ASSET_UPLOADED = "asset.uploaded"
RUN_SUCCEEDED = "run.processing.succeeded"
RUN_FAILED = "run.processing.failed"
SENSOR_LOG_REJECTED = "sensor_log.rejected"
REPORT_GENERATED = "report.generated"

VERB_DISPLAY = {
    "session.started": "Signed in",
    "session.ended": "Signed out",
    "session.revoked": "Session revoked",
    "user.signed_up": "Signed up",
    "user.deleted": "Deleted",
    RUN_CREATED: "Started a survey run",
    ASSET_UPLOADED: "Uploaded an asset",
    RUN_SUCCEEDED: "Run processed",
    RUN_FAILED: "Run failed",
    SENSOR_LOG_REJECTED: "Rejected a sensor log",
    REPORT_GENERATED: "Generated a report",
}

SOURCE_CLERK = "CLERK"
SOURCE_APP = "APP"


def _uuid_pk():
    return Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class UserProfile(Base):
    __tablename__ = "user_profiles"

    id = _uuid_pk()
    # The provider's user id. Unique because it identifies the person; the
    # surrogate key exists only so admin URLs do not have to carry provider
    # ids around.
    clerk_user_id = Column(String, nullable=False, unique=True, index=True)
    username = Column(String, nullable=True, index=True)
    full_name = Column(String, nullable=True)
    email = Column(String, nullable=True, index=True)
    image_url = Column(String, nullable=True)
    bio = Column(Text, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    # Last time anything was written for this person. Not a heartbeat: it moves
    # when they use the application.
    last_seen_at = Column(DateTime, nullable=True, index=True)

    # Deliberately no relationship to OrganisationMembership: membership stores
    # the provider's user id, not this table's surrogate key, so there is no
    # foreign key to join on. Membership is queried by user_id directly.


class ActivityEvent(Base):
    __tablename__ = "activity_events"

    id = _uuid_pk()
    verb = Column(String, nullable=False, index=True)
    # CLERK for something the identity provider reported, APP for something
    # this service did. The distinction is what makes a session event
    # attributable when it never touched the application.
    source = Column(String, nullable=False, default=SOURCE_APP, index=True)
    actor_id = Column(String, nullable=True, index=True)
    # Denormalised so the label survives the account being deleted: a trail
    # that goes blank when a person leaves is not a trail.
    actor_label = Column(String, nullable=True)
    organisation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("organisations.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    organisation_name = Column(String, nullable=True)
    target_type = Column(String, nullable=True)
    target_id = Column(String, nullable=True)
    target_label = Column(String, nullable=True)
    # The column is called metadata, which the declarative API reserves as an
    # attribute name, so the attribute is `event_metadata` and the wire name
    # stays `metadata` for the console.
    event_metadata = Column("metadata", JSON, nullable=True)
    occurred_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)

    __table_args__ = (
        Index("ix_activity_actor_time", "actor_id", "occurred_at"),
        Index("ix_activity_verb_time", "verb", "occurred_at"),
    )


def record(
    db,
    *,
    verb: str,
    actor_id=None,
    actor_label: str | None = None,
    source: str = SOURCE_APP,
    organisation_id=None,
    organisation_name: str | None = None,
    target_type: str | None = None,
    target_id=None,
    target_label: str | None = None,
    metadata: dict | None = None,
) -> ActivityEvent:
    """Record one activity event. Does not commit; the caller batches the write.

    A failure to record must not fail the thing being recorded, so callers pass
    this inside their own transaction and treat the trail as best-effort.
    """
    event = ActivityEvent(
        verb=verb,
        source=source,
        actor_id=str(actor_id) if actor_id is not None else None,
        actor_label=actor_label,
        organisation_id=organisation_id,
        organisation_name=organisation_name,
        target_type=target_type,
        target_id=str(target_id) if target_id is not None else None,
        target_label=target_label,
        metadata=metadata or {},
    )
    db.add(event)

    if actor_id is not None:
        touch(db, actor_id)
    return event


def touch(db, user_id) -> None:
    """Note that a person was active, creating their profile if it is the
    first thing this application has ever known about them."""
    try:
        profile = (
            db.query(UserProfile).filter(UserProfile.clerk_user_id == str(user_id)).first()
        )
        if profile is None:
            profile = UserProfile(clerk_user_id=str(user_id))
            db.add(profile)
        profile.last_seen_at = datetime.datetime.utcnow()
    except Exception:
        # A profile is bookkeeping; failing to write one must not fail a
        # request that would otherwise succeed.
        import logging

        logging.getLogger(__name__).exception("Could not update profile for %s", user_id)
