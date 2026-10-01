# app/services/notifications.py
"""Which notification kinds a user has turned off.

Preferences are stored as one nullable boolean per kind rather than as a
preference row per kind, so "not set" stays distinguishable from "set to
false" -- the first follows the default, the second does not.

Kinds map to preference keys, and a kind with no key is always delivered: a
notification nobody can turn off is the safe default for something that reports
a failure.
"""

import logging
from typing import Optional

from sqlalchemy.orm import Session

from app.models.organisation import NotificationPreference

logger = logging.getLogger(__name__)

# Notification kind -> the preference key that gates it.
KIND_PREFERENCE_KEY = {
    "run.processed": "run_processed",
    "run.failed": "run_failed",
    "team.added": "team_changes",
    "team.removed": "team_changes",
    "team.lead_assigned": "team_changes",
    "team.lead_removed": "team_changes",
    "project.run_uploaded": "project_runs",
    "org.role_changed": None,
    "org.removed": None,
    "org.ownership_received": None,
}


def preference_enabled(db: Session, user_id: str, kind: str) -> bool:
    """Whether this user should receive a notification of this kind.

    A database failure here must not silently swallow a notification, so it
    answers yes and logs: the cost of a duplicate notice is lower than the cost
    of a missed one.
    """
    key = KIND_PREFERENCE_KEY.get(kind)
    if key is None:
        return True
    try:
        row = (
            db.query(NotificationPreference)
            .filter(NotificationPreference.user_id == user_id)
            .first()
        )
    except Exception:
        logger.exception("Could not read notification preferences for %s", user_id)
        return True
    if row is None:
        return True
    value: Optional[bool] = getattr(row, f"{key}_enabled", None)
    # None means the user never chose, so the default applies.
    return True if value is None else bool(value)
