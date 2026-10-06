"""Who may see a run, and who may change it.

One definition of both rules, because the run list publishes them and the
delete route enforces them. They were separate before: the list told the
interface to show Delete to everyone (`can_write` was hard-coded `true`) and the
delete route checked nothing at all, so the permission existed only as a field
in a response.

Scope has a second reason to be here rather than in the route. A run listing
that is not scoped to the caller returns every organisation's surveys, and this
is the place that says which rows the caller is entitled to at all.
"""

from typing import Set

from sqlalchemy.orm import Session

from app.models.organisation import Organisation, OrganisationMembership
from app.models.survey import SurveyRun


def caller_org_ids(db: Session, user_id: str) -> Set[str]:
    """Every organisation the caller belongs to, as owner or as a member.

    Both are read because they are not the same set: an owner need not have a
    membership row, and a member is not the owner. Reading only memberships hid
    a personal organisation from the person who owns it.
    """
    owned = {
        row[0]
        for row in db.query(Organisation.id).filter(Organisation.owner_id == user_id).all()
    }
    member_of = {
        row[0]
        for row in db.query(OrganisationMembership.organisation_id)
        .filter(OrganisationMembership.user_id == user_id)
        .all()
    }
    return owned | member_of


def visible_runs_query(db: Session, user_id: str):
    """Runs the caller may see: their own, plus any in an organisation they are in.

    The caller's own runs are included whatever their organisation, because an
    upload made before the client sent one has none, and hiding it from the
    person who made it would be worse than showing it too broadly.

    Removed runs are excluded here rather than by each caller. Deletion is a
    soft `removed_at` stamp, and the listing did not filter it, so a run the user
    had deleted stayed on their screen until they reloaded with different data.
    """
    org_ids = caller_org_ids(db, user_id)
    query = db.query(SurveyRun).filter(SurveyRun.removed_at.is_(None))
    if org_ids:
        return query.filter(
            (SurveyRun.organisation_id.in_(org_ids)) | (SurveyRun.user_id == user_id)
        )
    return query.filter(SurveyRun.user_id == user_id)


def can_write_run(db: Session, run: SurveyRun, user_id: str) -> bool:
    """Whether the caller may delete this run or rebuild its report.

    Narrower than seeing it, which is the point: a member sees their teammates'
    surveys so a project reads as shared work, but only the person who recorded a
    run and the owner of the organisation it belongs to may change or remove it.
    """
    if run.user_id == user_id:
        return True
    if run.organisation_id is None:
        return False
    owned = (
        db.query(Organisation.id)
        .filter(Organisation.id == run.organisation_id, Organisation.owner_id == user_id)
        .first()
    )
    return owned is not None


def run_status(run: SurveyRun) -> str:
    """The states a run can actually be in.

    `processing_metadata` is written once, at the end of a successful pipeline
    run, and nowhere else in the service -- there is no failure marker. So a run
    whose pipeline crashed and a run that has not started yet are the same row as
    far as this table is concerned, and this reports only what it can tell apart.
    The interface offers these two states and no others, because offering
    "Failed" for a state nothing writes is a filter that silently always comes
    back empty.
    """
    return "DONE" if getattr(run, "processing_metadata", None) else "CREATED"