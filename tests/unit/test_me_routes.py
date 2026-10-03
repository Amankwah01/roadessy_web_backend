"""The signed-in caller's own details and the reports generated for them.

What is asserted here is the access question rather than the happy path: who the
caller is read from the verified session rather than a field, membership decides
which organisation's data may be read, and a download link is time-limited rather
than permanent.
"""

import asyncio
import json
import time
import uuid

import pytest
import sqlalchemy as sa
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import MetaData

from app.api.routes import me
from app.api.routes.me import (
    download_my_report,
    generate_my_report,
    get_my_details,
    list_my_reports,
    update_my_details,
)
from app.core import security
from app.models.admin import UserProfile
from app.models.organisation import Organisation, OrganisationMembership
from app.models.report import KIND_ORGANISATION, ReportAsset
from app.schemas.reports import GenerateReportRequest, UserDetailsUpdate
from app.services.reports import reports_root


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

#: Fixed ids rather than generated ones, so a failure names the organisation it
#: happened in and the stored rows are predictable.
ORG_ONE = "11111111-1111-4111-8111-111111111111"
ORG_TWO = "22222222-2222-4222-8222-222222222222"


def _session():
    """A session over just the tables these handlers read and write.

    Two SQLite-specific problems, worked around in the test rather than in the
    models, which are correct for the Postgres they run against in production:

    * `Base.metadata.create_all` builds every model, and the survey tables carry
      PostGIS geometry columns SQLite cannot create.
    * `sqlalchemy.Uuid` renders as a column of declared type `UUID`. SQLite reads
      that as NUMERIC affinity, so a stored uuid comes back out as a float. The
      tables are copied into a private schema with those columns typed as text,
      which keeps the value a string the model can turn back into a uuid.
    """
    engine = create_engine("sqlite:///:memory:")
    schema = MetaData()
    for model in (ReportAsset, Organisation, OrganisationMembership, UserProfile):
        model.__table__.to_metadata(schema)
    for table in schema.tables.values():
        for column in table.columns:
            if isinstance(column.type, sa.Uuid):
                column.type = sa.String(36)
    schema.create_all(engine)
    return sessionmaker(bind=engine)()


def _org(db, org_id=ORG_ONE, name="Ghana Roads Authority", owner="owner-1"):
    org = Organisation(id=uuid.UUID(org_id), name=name, slug=f"org-{org_id[:8]}", owner_id=owner)
    db.add(org)
    db.commit()
    return org


def _member(db, org_id, user_id):
    membership = OrganisationMembership(
        id=uuid.uuid4(),
        organisation_id=uuid.UUID(org_id),
        user_id=str(user_id),
        role="member",
    )
    db.add(membership)
    db.commit()
    return membership


def _asset(db, org_id=ORG_ONE, year=2026, file_name="report.pdf"):
    asset = ReportAsset(
        id=uuid.uuid4(),
        organisation_id=org_id,
        year=year,
        title=f"{year} report",
        kind=KIND_ORGANISATION,
        file_name=file_name,
        original_filename=file_name,
        content_type="application/pdf",
        size_bytes=2048,
        sha256="a" * 64,
        storage_path=str(reports_root() / org_id / str(year) / file_name),
        bucket="reports",
        key=f"{org_id}/{year}/{file_name}",
        created_by="user-1",
    )
    db.add(asset)
    db.commit()
    return asset


def _body(response):
    return json.loads(response.body)


# ---------------------------------------------------------------------------
# Details
# ---------------------------------------------------------------------------


def test_details_answer_for_a_person_who_signed_in_without_a_profile():
    """First sign-in is the common case and writes nothing yet. The caller must
    still get their own id back rather than an empty object."""
    db = _session()

    response = asyncio.run(get_my_details("user-1", db))
    body = _body(response)

    assert response.status_code == 200
    assert body["id"] == "user-1"
    assert body["username"] == "user-1"
    assert body["bio"] is None
    assert body["is_active"] is True


def test_details_read_the_row_when_one_exists():
    db = _session()
    profile = UserProfile(
        clerk_user_id="user-1",
        username="amoah",
        email="amoah@example.org",
        full_name="Ama Amoah",
        bio="Road engineer.",
        is_active=True,
    )
    db.add(profile)
    db.commit()

    body = _body(asyncio.run(get_my_details("user-1", db)))

    assert body["username"] == "amoah"
    assert body["email"] == "amoah@example.org"
    assert body["full_name"] == "Ama Amoah"
    assert body["bio"] == "Road engineer."


def test_details_are_the_callers_own_and_not_another_persons():
    """The id comes from the verified session, so the read cannot be aimed at
    somebody else by a request field."""
    db = _session()
    db.add(UserProfile(clerk_user_id="user-2", username="user-2", bio="Not yours."))
    db.commit()

    body = _body(asyncio.run(get_my_details("user-1", db)))

    assert body["id"] == "user-1"
    assert body["bio"] is None


def test_updating_the_bio_creates_the_profile_if_it_did_not_exist():
    db = _session()

    response = asyncio.run(update_my_details(UserDetailsUpdate(bio="Survey lead."), "user-1", db))
    body = _body(response)

    assert response.status_code == 200
    assert body["bio"] == "Survey lead."
    assert body["id"] == "user-1"
    assert body["last_seen_at"] is not None, "an update records that the person was here"

    stored = db.query(UserProfile).filter(UserProfile.clerk_user_id == "user-1").first()
    assert stored.bio == "Survey lead."


def test_a_blank_bio_clears_it_rather_than_storing_whitespace():
    db = _session()
    db.add(UserProfile(clerk_user_id="user-1", bio="Old text."))
    db.commit()

    body = _body(asyncio.run(update_my_details(UserDetailsUpdate(bio="   "), "user-1", db)))

    assert body["bio"] is None


def test_saving_the_bio_twice_updates_one_row():
    db = _session()

    asyncio.run(update_my_details(UserDetailsUpdate(bio="First"), "user-1", db))
    asyncio.run(update_my_details(UserDetailsUpdate(bio="Second"), "user-1", db))

    rows = db.query(UserProfile).filter(UserProfile.clerk_user_id == "user-1").all()
    assert len(rows) == 1
    assert rows[0].bio == "Second"


def test_an_over_long_bio_is_refused_by_the_schema():
    """A caller sending an over-long bio is turned away by the wire schema, so the
    handler never sees it."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        UserDetailsUpdate(bio="x" * 501)


def test_the_handler_also_refuses_an_over_long_bio_it_is_handed_directly():
    """The route re-checks the length rather than trusting that only the schema
    ever calls it, and refuses before writing a profile."""
    db = _session()

    payload = UserDetailsUpdate.construct(bio="x" * 501)

    with pytest.raises(HTTPException) as raised:
        asyncio.run(update_my_details(payload, "user-1", db))

    assert raised.value.status_code == 422
    assert db.query(UserProfile).filter(UserProfile.clerk_user_id == "user-1").first() is None


# ---------------------------------------------------------------------------
# Listing reports
# ---------------------------------------------------------------------------


def test_listing_returns_the_organisations_reports_with_a_signed_link():
    db = _session()
    _org(db)
    _member(db, ORG_ONE, "user-1")
    asset = _asset(db)

    body = _body(asyncio.run(list_my_reports(ORG_ONE, None, {}, "user-1", db)))

    assert body["org_id"] == ORG_ONE
    assert [row["id"] for row in body["reports"]] == [str(asset.id)]
    row = body["reports"][0]
    assert row["content_type"] == "application/pdf"
    assert row["size_bytes"] == 2048
    assert row["supabase_url"].startswith(f"/api/me/reports/{asset.id}/download/?token=")


def test_the_signed_link_in_the_listing_actually_verifies():
    """The link is the whole authorisation for a browser navigation, so a token
    that the list publishes but the download refuses would be a dead feature."""
    db = _session()
    _org(db)
    _member(db, ORG_ONE, "user-1")
    asset = _asset(db)

    body = _body(asyncio.run(list_my_reports(ORG_ONE, None, {}, "user-1", db)))
    url = body["reports"][0]["supabase_url"]

    query = dict(part.split("=", 1) for part in url.split("?", 1)[1].split("&"))
    # Raises nothing: the published token is this service's own signature.
    security.verify_download_token(str(asset.id), query["token"], int(query["expires_at"]))
    # And it is bound to this report, not a token that opens any report.
    with pytest.raises(HTTPException):
        security.verify_download_token(str(uuid.uuid4()), query["token"], int(query["expires_at"]))


def test_listing_without_an_organisation_is_a_bad_request_not_an_empty_list():
    """An empty list would read as "this organisation has no reports", which is a
    different claim from "this session is not scoped to one"."""
    db = _session()

    with pytest.raises(HTTPException) as raised:
        asyncio.run(list_my_reports(None, None, {}, "user-1", db))

    assert raised.value.status_code == 400


def test_listing_falls_back_to_the_organisations_token_names():
    db = _session()
    _org(db)
    _member(db, ORG_ONE, "user-1")
    asset = _asset(db)

    body = _body(asyncio.run(list_my_reports(None, None, {"org_id": ORG_ONE}, "user-1", db)))

    assert body["org_id"] == ORG_ONE
    assert [row["id"] for row in body["reports"]] == [str(asset.id)]


def test_a_non_member_cannot_list_an_organisations_reports():
    db = _session()
    _org(db, owner="owner-1")
    _member(db, ORG_ONE, "user-2")
    _asset(db)

    with pytest.raises(HTTPException) as raised:
        asyncio.run(list_my_reports(ORG_ONE, None, {}, "user-1", db))

    assert raised.value.status_code == 404


def test_an_organisation_that_does_not_exist_and_one_you_cannot_see_answer_the_same():
    """Otherwise a caller could map which organisation ids are real by reading the
    status code, which is a way of learning about data they may not see."""
    db = _session()
    _org(db, owner="owner-1")
    _member(db, ORG_ONE, "user-1")

    statuses = []
    for org_id in (str(uuid.uuid4()), ORG_ONE):
        try:
            asyncio.run(list_my_reports(org_id, None, {}, "user-2", db))
        except HTTPException as exc:
            statuses.append(exc.status_code)

    assert statuses == [404, 404]


def test_an_organisation_id_that_is_not_an_id_is_a_not_found():
    db = _session()

    with pytest.raises(HTTPException) as raised:
        asyncio.run(list_my_reports("not-a-uuid", None, {}, "user-1", db))

    assert raised.value.status_code == 404


def test_listing_a_year_returns_only_that_year():
    db = _session()
    _org(db)
    _member(db, ORG_ONE, "user-1")
    _asset(db, year=2025, file_name="y2025.pdf")
    _asset(db, year=2026, file_name="y2026.pdf")

    body = _body(asyncio.run(list_my_reports(ORG_ONE, 2025, {}, "user-1", db)))

    assert [row["year"] for row in body["reports"]] == [2025]


def test_an_out_of_range_year_is_rejected_by_the_request_schema():
    """The year is bounded by the wire schema, so a nonsense year is refused
    before it can reach a report."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        GenerateReportRequest(org_id=ORG_ONE, year=3000)
    with pytest.raises(ValidationError):
        GenerateReportRequest(org_id=ORG_ONE, year=1969)


# ---------------------------------------------------------------------------
# Generating
# ---------------------------------------------------------------------------


def test_generating_returns_the_new_report_with_the_listing(monkeypatch):
    """This asserts the route's contract. The report itself is built and written
    in `tests/unit/test_reports_service.py`; here the service is a stand-in so
    the survey tables — which SQLite cannot create — are not needed."""
    monkeypatch.setattr(me, "generate_organisation_report", _stub_generator)
    db = _session()
    org = _org(db)
    _member(db, ORG_ONE, "user-1")

    response = asyncio.run(
        generate_my_report(
            GenerateReportRequest(org_id=str(org.id), year=2026, report_title="Q3 review"),
            {},
            "user-1",
            db,
        )
    )
    body = _body(response)

    assert response.status_code == 201
    assert body["org_id"] == str(org.id)
    assert body["created"]["title"] == "Q3 review"
    assert body["created"]["created_by"] == "user-1"
    assert [row["id"] for row in body["reports"]] == [body["created"]["id"]]
    assert body["created"]["supabase_url"].startswith("/api/me/reports/")


def _stub_generator(db, *, organisation_id, organisation_name, year, title, created_by):
    """Stands in for the report service: records what it was asked for, so the
    route can be asserted on without the survey tables."""
    asset = ReportAsset(
        id=uuid.uuid4(),
        organisation_id=organisation_id,
        year=year,
        title=title or f"{organisation_name} Road Condition Report {year}",
        kind=KIND_ORGANISATION,
        file_name=f"{organisation_id}_{year}.pdf",
        original_filename=f"{organisation_id}_{year}.pdf",
        content_type="application/pdf",
        size_bytes=2048,
        sha256="a" * 64,
        storage_path=str(reports_root() / organisation_id / str(year) / "report.pdf"),
        bucket="reports",
        key=f"{organisation_id}/{year}/report.pdf",
        created_by=created_by,
    )
    db.add(asset)
    db.commit()
    return asset, [asset]


def test_generating_for_an_organisation_you_are_not_in_refuses():
    db = _session()
    other_org_id = str(uuid.uuid4())
    _org(db, org_id=other_org_id, owner="user-2")

    with pytest.raises(HTTPException) as raised:
        asyncio.run(
            generate_my_report(
                GenerateReportRequest(org_id=other_org_id, year=2026, report_title=""),
                {},
                "user-1",
                db,
            )
        )

    assert raised.value.status_code == 404


def test_a_failure_while_rendering_is_a_server_error(monkeypatch):
    """The traceback goes to the log. The caller gets a status it can act on,
    without the server's internals in the message."""

    def _explode(*args, **kwargs):
        raise RuntimeError("matplotlib ran out of memory")

    monkeypatch.setattr(me, "generate_organisation_report", _explode)
    db = _session()
    _org(db)
    _member(db, ORG_ONE, "user-1")

    with pytest.raises(HTTPException) as raised:
        asyncio.run(
            generate_my_report(
                GenerateReportRequest(org_id=ORG_ONE, year=2026, report_title=""),
                {},
                "user-1",
                db,
            )
        )

    assert raised.value.status_code == 500
    assert "matplotlib ran out of memory" not in raised.value.detail, (
        "the reason belongs in the log; a rendering failure's text carries paths "
        "and library internals that the caller cannot act on"
    )
    assert "Try again" in raised.value.detail


def test_the_owner_may_read_their_own_organisation_without_a_membership_row():
    """The owner's own organisation must be reachable even if the membership row
    was never written."""
    db = _session()
    _org(db, org_id=ORG_ONE, owner="owner-1")

    body = _body(asyncio.run(list_my_reports(ORG_ONE, None, {}, "owner-1", db)))

    assert body["org_id"] == ORG_ONE


# ---------------------------------------------------------------------------
# Download links
# ---------------------------------------------------------------------------


def test_a_valid_token_is_the_signature_for_this_report_at_this_expiry():
    link = security.sign_download_link("asset-1")

    security.verify_download_token("asset-1", link["token"], link["expires_at"])


def test_a_token_issued_for_one_report_does_not_open_another():
    link = security.sign_download_link("asset-1")

    with pytest.raises(HTTPException) as raised:
        security.verify_download_token("asset-2", link["token"], link["expires_at"])

    assert raised.value.status_code == 403


def test_a_token_that_has_expired_is_refused():
    """The listing is a page that can sit open. Without an expiry a link pasted
    into a chat would work forever."""
    link = security.sign_download_link("asset-1", ttl_seconds=1)
    time.sleep(1.1)

    with pytest.raises(HTTPException) as raised:
        security.verify_download_token("asset-1", link["token"], link["expires_at"])

    assert raised.value.status_code == 403
    assert "expired" in raised.value.detail.lower()


def test_an_expiry_the_signature_was_not_issued_for_is_refused():
    """Extending the expiry in the URL must not help: the signature covers the
    expiry, so a longer window is a different, unsigned claim."""
    link = security.sign_download_link("asset-1", ttl_seconds=600)

    with pytest.raises(HTTPException) as raised:
        security.verify_download_token("asset-1", link["token"], link["expires_at"] + 86_400)

    assert raised.value.status_code == 403


def test_a_link_signed_with_a_different_secret_is_refused(monkeypatch):
    monkeypatch.setenv("ROADESSY_DOWNLOAD_SECRET", "the-real-secret")
    link = security.sign_download_link("asset-1")

    monkeypatch.setenv("ROADESSY_DOWNLOAD_SECRET", "a-different-secret")
    with pytest.raises(HTTPException) as raised:
        security.verify_download_token("asset-1", link["token"], link["expires_at"])

    assert raised.value.status_code == 403


def test_a_download_link_is_short_lived_by_default():
    """Ten minutes is long enough to click a link and short enough that a link in
    a chat stops working before anyone reuses it."""
    link = security.sign_download_link("asset-1")

    assert security.DOWNLOAD_LINK_TTL_SECONDS == 600
    assert link["expires_at"] - int(time.time()) <= 600
    assert link["expires_at"] > int(time.time())


@pytest.mark.parametrize("expires_at", ["soon", "", None])
def test_an_unreadable_expiry_is_refused(expires_at):
    link = security.sign_download_link("asset-1")

    with pytest.raises(HTTPException) as raised:
        security.verify_download_token("asset-1", link["token"], expires_at)

    assert raised.value.status_code == 403


def test_an_empty_token_is_refused():
    link = security.sign_download_link("asset-1")

    with pytest.raises(HTTPException) as raised:
        security.verify_download_token("asset-1", "", link["expires_at"])

    assert raised.value.status_code == 403


def test_a_report_id_that_is_not_an_id_is_a_not_found():
    """A valid signature over a malformed id reaches the id parsing, which must
    answer 404 rather than raise out of the route."""
    link = security.sign_download_link("not-a-uuid")

    with pytest.raises(HTTPException) as raised:
        asyncio.run(download_my_report("not-a-uuid", link["token"], link["expires_at"]))

    assert raised.value.status_code == 404


def test_a_forged_token_never_reaches_the_file_lookup():
    """The signature is checked before anything else, so a request without one
    cannot use this route to probe which report ids exist."""
    with pytest.raises(HTTPException) as raised:
        asyncio.run(download_my_report(str(uuid.uuid4()), "made-up", int(time.time()) + 600))

    assert raised.value.status_code == 403


def test_a_token_naming_an_organisation_does_not_grant_access_to_it():
    """Membership, not a claim, decides what may be read. A token that names an
    organisation the caller is not in gets nothing."""
    db = _session()
    _org(db, owner="user-2")

    assert me._principal_org_id({"org_id": ORG_ONE}) == ORG_ONE
    assert me._principal_org_id({"org_id": None}) is None
    assert me._principal_org_id({}) is None

    with pytest.raises(HTTPException) as raised:
        me._require_org_access(db, ORG_ONE, "user-1")

    assert raised.value.status_code == 404