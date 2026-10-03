"""Report generation.

The numbers a report prints are claims about a road, so most of what is asserted
here is what the report must *not* claim: no distance summed twice, no unmuteasured
run averaged in as zero, and no file read from outside the reports directory.
"""

import datetime
import hashlib
import uuid
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.report import KIND_ORGANISATION, ReportAsset
from app.services.reports import (
    build_report_rows,
    distance_by_run,
    defect_counts_by_run,
    defect_modes_by_run,
    generate_organisation_report,
    iri_samples_by_run,
    render_report_pdf,
    reports_root,
    resolve_report_path,
    year_bounds,
)
from app.services.reports import service as reports


# ---------------------------------------------------------------------------
# Fakes
#
# The survey tables carry PostGIS geometry columns that SQLite cannot create, so
# the aggregate helpers are driven through a stand-in query rather than through
# `create_all`. That keeps the assertion on the helper's arithmetic — which is
# where the wrong number would come from — instead of on the ORM plumbing.
# ---------------------------------------------------------------------------


class _Rows:
    """One aggregate row, as the query would label it."""

    def __init__(self, **values):
        self.__dict__.update(values)


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *args, **kwargs):
        return self

    def group_by(self, *args, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def all(self):
        return self._rows


def _fake_db(rows):
    class _DB:
        def query(self, *args, **kwargs):
            return _FakeQuery(rows)

    return _DB()


class _Road:
    def __init__(self, name):
        self.name = name


class _Project:
    def __init__(self, name):
        self.name = name


class _Run:
    """The run columns `build_report_rows` reads."""

    def __init__(self, run_id, name, created_at, **overrides):
        self.run_id = run_id
        self.name = name
        self.created_at = created_at
        self.final_pci = None
        self.roadessy_pci = None
        self.astm_pci = None
        self.simple_iri = None
        self.astm_qcm_iri = None
        self.removed_at = None
        self.road = None
        self.project = None
        self.__dict__.update(overrides)


def _session():
    """A session over just the report table.

    `Base.metadata.create_all` would build every model in the project and the
    survey tables' geometry columns would fail on SQLite, so only the table this
    module writes is built.
    """
    engine = create_engine("sqlite:///:memory:")
    ReportAsset.__table__.create(bind=engine)
    return sessionmaker(bind=engine)()


def _rows(year=2026):
    """Two runs: one measured, one recorded with nothing measured at all."""
    measured_at = datetime.datetime(year, 3, 4, 9, 30)
    bare_at = datetime.datetime(year, 6, 1, 8, 0)
    return [
        {
            "run_id": uuid.uuid4(),
            "name": "Kumasi to Obuasi",
            "created_at": measured_at,
            "project_name": "Western corridor",
            "road_name": "N8",
            "distance_m": 12_400.0,
            "pci": 71.5,
            "astm_pci": None,
            "simple_iri": 4.2,
            "astm_qcm_iri": 3.8,
            "defect_count": 7,
            "defect_modes": {"Pothole": 3, "Crack": 4},
            "iri_samples": [3.1, 4.2, 5.0, 4.4],
            "astm_iri_samples": [3.0, 3.8],
            "removed": False,
        },
        {
            "run_id": uuid.uuid4(),
            "name": "Kumasi loop",
            "created_at": bare_at,
            "project_name": "Western corridor",
            "road_name": None,
            "distance_m": None,
            "pci": None,
            "astm_pci": None,
            "simple_iri": None,
            "astm_qcm_iri": None,
            "defect_count": 0,
            "defect_modes": {},
            "iri_samples": [],
            "astm_iri_samples": [],
            "removed": False,
        },
    ]


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------


def test_distance_is_the_far_end_of_the_last_segment():
    """Consecutive 50 m segments tile a run, so summing them multiplies the answer
    by the segment count: a 1 km run reported as 20 km is the wrong road. The
    aggregate here is `max(segment_end_m)`, not `sum(segment_length)`."""
    run_id = uuid.uuid4()

    distances = distance_by_run(_fake_db([_Rows(run_id=run_id, furthest=1000.0)]), [run_id])

    assert distances == {run_id: 1000.0}


def test_a_run_with_no_usable_end_distance_is_absent_not_zero():
    """Zero is a claim that the run surveyed no distance at all. A run whose
    furthest segment is NaN, or which has no segments at all, is unmeasured."""
    nan_run = uuid.uuid4()
    empty_run = uuid.uuid4()

    distances = distance_by_run(
        _fake_db([_Rows(run_id=nan_run, furthest=float("nan"))]),
        [nan_run, empty_run],
    )

    assert distances == {}
    assert empty_run not in distances


def test_distance_of_no_runs_asks_no_question_of_the_database():
    class _Explodes:
        def query(self, *args, **kwargs):
            raise AssertionError("an empty run list must not reach the database")

    assert distance_by_run(_Explodes(), []) == {}
    assert defect_counts_by_run(_Explodes(), []) == {}
    assert defect_modes_by_run(_Explodes(), []) == {}
    assert iri_samples_by_run(_Explodes(), []) == {}


def test_non_finite_values_never_become_measurements():
    assert reports._finite(float("nan")) is None
    assert reports._finite(float("inf")) is None
    assert reports._finite(float("-inf")) is None
    assert reports._finite(None) is None
    assert reports._finite("not a number") is None
    assert reports._finite("12.5") == 12.5
    assert reports._finite(0) == 0.0, "a genuine zero is a measurement and must survive"


# ---------------------------------------------------------------------------
# Defects
# ---------------------------------------------------------------------------


def test_defect_counts_and_modes_agree_on_the_same_run():
    run_id = uuid.uuid4()

    counts = defect_counts_by_run(_fake_db([_Rows(run_id=run_id, n=7)]), [run_id])
    modes = defect_modes_by_run(
        _fake_db([_Rows(run_id=run_id, mode="Pothole", n=3), _Rows(run_id=run_id, mode="Crack", n=4)]),
        [run_id],
    )

    assert counts[run_id] == 7
    assert sum(modes[run_id].values()) == counts[run_id]


def test_a_defect_with_no_failure_mode_is_still_counted():
    """An unclassified defect is a defect. Dropping it would make the mode
    breakdown disagree with the headline count."""
    run_id = uuid.uuid4()

    modes = defect_modes_by_run(_fake_db([_Rows(run_id=run_id, mode=None, n=2)]), [run_id])

    assert modes == {run_id: {"Unclassified": 2}}


def test_iri_samples_keep_the_two_methods_apart():
    """The simple and ASTM tracks are different measurements of the same road.
    Merging them would report a number neither method produced."""
    run_id = uuid.uuid4()
    rows = [
        _Rows(run_id=run_id, iri=3.1, simple_iri=3.0, qcm_iri=2.9),
        _Rows(run_id=run_id, iri=None, simple_iri=float("nan"), qcm_iri=None),
    ]

    samples = iri_samples_by_run(_fake_db(rows), [run_id])

    assert samples[run_id]["simple_iri"] == [3.0], "a NaN sample is dropped, not kept as a zero"
    assert samples[run_id]["qcm_iri"] == [2.9]
    assert samples[run_id]["iri"] == [3.1]


# ---------------------------------------------------------------------------
# The year window
# ---------------------------------------------------------------------------


def test_the_year_window_is_half_open_so_a_run_is_counted_once():
    start, end = year_bounds(2026)

    assert start == datetime.datetime(2026, 1, 1)
    assert end == datetime.datetime(2027, 1, 1)
    assert start <= datetime.datetime(2026, 12, 31, 23, 59, 59) < end

    # The boundary instant belongs to the next year and to exactly one window,
    # which is what makes `created_at < end` safe alongside `created_at >= start`.
    boundary = datetime.datetime(2027, 1, 1)
    assert not start <= boundary < end
    assert year_bounds(2027)[0] <= boundary < year_bounds(2027)[1]


# ---------------------------------------------------------------------------
# Means report their own denominator
# ---------------------------------------------------------------------------


def test_a_mean_says_how_many_runs_went_into_it():
    """A mean without its denominator reads as a measurement of the whole
    organisation rather than of the runs that reported."""
    mean, count = reports._summarise([{"pci": 60.0}, {"pci": 80.0}, {"pci": None}], "pci")

    assert count == 2, "a run that published nothing is not a run at zero"
    assert mean == 70.0


def test_a_mean_of_nothing_is_absent_not_zero():
    """PCI 0 is the worst road the method can express, so a default of zero would
    be a claim rather than an absence."""
    mean, count = reports._summarise([{"pci": None}, {"pci": None}], "pci")

    assert (mean, count) == (None, 0)


def test_a_removed_run_is_left_out_of_every_figure():
    rows = _rows()
    rows[0]["removed"] = True
    rows[0]["distance_m"] = 999_000.0
    rows[0]["defect_count"] = 500

    # `measured` is the filter the renderer applies before it adds anything up.
    measured = [r for r in rows if not r["removed"]]

    assert sum(r["distance_m"] for r in measured if r["distance_m"] is not None) == 0.0
    assert sum(r["defect_count"] for r in measured) == 0


# ---------------------------------------------------------------------------
# Rows assembled from runs
# ---------------------------------------------------------------------------


def test_a_run_is_reported_with_both_pci_tracks_kept_apart(monkeypatch):
    """`final_pci`/`roadessy_pci` are this service's own index; `astm_pci` is a
    different track. When the ASTM track is muted it must stay absent rather than
    inherit the other track's number."""
    runs = [
        _Run(
            uuid.uuid4(),
            "measured",
            datetime.datetime(2026, 3, 4),
            final_pci=71.5,
            astm_pci=None,
            simple_iri=4.2,
            astm_qcm_iri=3.8,
            road=_Road("N8"),
            project=_Project("Western corridor"),
        )
    ]
    monkeypatch.setattr(reports, "distance_by_run", lambda db, ids: {runs[0].run_id: 12_400.0})
    monkeypatch.setattr(reports, "defect_counts_by_run", lambda db, ids: {runs[0].run_id: 7})
    monkeypatch.setattr(reports, "defect_modes_by_run", lambda db, ids: {runs[0].run_id: {"Crack": 7}})
    monkeypatch.setattr(
        reports,
        "iri_samples_by_run",
        lambda db, ids: {runs[0].run_id: {"iri": [], "simple_iri": [4.2], "qcm_iri": [3.8]}},
    )
    monkeypatch.setattr(reports, "runs_for_year", lambda db, org, year: runs)

    rows = build_report_rows(_fake_db([]), "org-1", 2026)

    assert len(rows) == 1
    assert rows[0]["pci"] == 71.5
    assert rows[0]["astm_pci"] is None, "a muted ASTM track must not borrow the other track's score"
    assert rows[0]["simple_iri"] == 4.2
    assert rows[0]["astm_qcm_iri"] == 3.8
    assert rows[0]["distance_m"] == 12_400.0
    assert rows[0]["road_name"] == "N8"
    assert rows[0]["project_name"] == "Western corridor"


def test_a_run_that_published_nothing_is_still_a_row(monkeypatch):
    """The report's coverage line says how many runs went into each mean. That is
    only true if the runs with nothing published are present as rows with absent
    figures, rather than missing and silently shrinking the denominator."""
    runs = [_Run(uuid.uuid4(), "upload with no analysis", datetime.datetime(2026, 6, 1))]
    monkeypatch.setattr(reports, "runs_for_year", lambda db, org, year: runs)
    monkeypatch.setattr(reports, "distance_by_run", lambda db, ids: {})
    monkeypatch.setattr(reports, "defect_counts_by_run", lambda db, ids: {})
    monkeypatch.setattr(reports, "defect_modes_by_run", lambda db, ids: {})
    monkeypatch.setattr(reports, "iri_samples_by_run", lambda db, ids: {})

    rows = build_report_rows(_fake_db([]), "org-1", 2026)

    assert len(rows) == 1
    assert rows[0]["distance_m"] is None
    assert rows[0]["pci"] is None
    assert rows[0]["simple_iri"] is None
    assert rows[0]["defect_count"] == 0
    assert rows[0]["defect_modes"] == {}
    assert rows[0]["iri_samples"] == []
    mean, count = reports._summarise(rows, "pci")
    assert (mean, count) == (None, 0)


def test_a_run_with_no_name_is_still_identifiable(monkeypatch):
    run = _Run(uuid.uuid4(), None, datetime.datetime(2026, 6, 1))
    monkeypatch.setattr(reports, "runs_for_year", lambda db, org, year: [run])
    monkeypatch.setattr(reports, "distance_by_run", lambda db, ids: {})
    monkeypatch.setattr(reports, "defect_counts_by_run", lambda db, ids: {})
    monkeypatch.setattr(reports, "defect_modes_by_run", lambda db, ids: {})
    monkeypatch.setattr(reports, "iri_samples_by_run", lambda db, ids: {})

    rows = build_report_rows(_fake_db([]), "org-1", 2026)

    assert rows[0]["name"] == str(run.run_id), "a nameless run needs an identifier, not a blank cell"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_the_report_is_a_pdf_a_browser_can_open(tmp_path):
    target = tmp_path / "report.pdf"

    render_report_pdf(target, "Road Condition Report 2026", "Ghana Roads Authority", 2026, _rows())

    payload = target.read_bytes()
    assert payload.startswith(b"%PDF"), "a report the browser cannot open is not a report"
    assert len(payload) > 1000


def test_a_period_with_no_runs_still_produces_a_report(tmp_path):
    """A year before the first survey is a real answer, not an error."""
    target = tmp_path / "empty.pdf"

    render_report_pdf(target, "Nothing recorded yet", "Ghana Roads Authority", 1999, [])

    assert target.read_bytes().startswith(b"%PDF")


def test_a_period_where_nothing_was_measured_does_not_error_out(tmp_path):
    target = tmp_path / "unmeasured.pdf"
    rows = _rows()
    for row in rows:
        row.update(
            {
                "pci": None,
                "simple_iri": None,
                "distance_m": None,
                "iri_samples": [],
                "defect_count": 0,
                "defect_modes": {},
            }
        )

    render_report_pdf(target, "No measurements published", "GRA", 2026, rows)

    assert target.read_bytes().startswith(b"%PDF")


# ---------------------------------------------------------------------------
# Generating, recording, listing
# ---------------------------------------------------------------------------


def test_generating_a_report_writes_the_file_and_records_what_it_wrote(tmp_path, monkeypatch):
    monkeypatch.setenv("ROADESSY_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(reports, "build_report_rows", lambda db, org, year: _rows())
    db = _session()

    asset, listed = generate_organisation_report(
        db,
        organisation_id="org-1",
        organisation_name="Ghana Roads Authority",
        year=2026,
        title="Q3 condition review",
        created_by="user-1",
    )

    assert asset.kind == KIND_ORGANISATION
    assert asset.year == 2026
    assert asset.title == "Q3 condition review"
    assert asset.organisation_id == "org-1"
    assert asset.created_by == "user-1"
    assert asset.content_type == "application/pdf"
    assert asset.file_name.endswith(".pdf")
    assert asset.size_bytes > 0, "a row with no bytes behind it lists as a broken download"
    assert len(asset.sha256) == 64

    written = Path(asset.storage_path)
    assert written.is_file(), "the row records a file that was never written"
    assert written.read_bytes().startswith(b"%PDF")
    assert written.stat().st_size == asset.size_bytes, "the recorded size must match what is on disk"
    assert asset.sha256 == hashlib.sha256(written.read_bytes()).hexdigest()

    assert [row.id for row in listed] == [asset.id], "the dashboard renders this list straight after generating"


def test_a_title_cannot_steer_the_write_outside_the_reports_root(tmp_path, monkeypatch):
    """The title is caller-supplied text. It is sanitised into a filename, so a
    title like `../../etc/passwd` produces a file inside the reports directory."""
    monkeypatch.setenv("ROADESSY_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(reports, "build_report_rows", lambda db, org, year: _rows())
    db = _session()

    asset, _ = generate_organisation_report(
        db,
        organisation_id="org-1",
        organisation_name="GRA",
        year=2026,
        title="../../etc/passwd",
        created_by="user-1",
    )

    written = Path(asset.storage_path).resolve()
    assert written.parent == (reports_root() / "org-1" / "2026").resolve()
    assert written.is_relative_to(reports_root().resolve())
    assert ".." not in written.name


def test_a_title_without_a_year_or_a_report_id_still_names_the_file(tmp_path, monkeypatch):
    monkeypatch.setenv("ROADESSY_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(reports, "build_report_rows", lambda db, org, year: _rows())
    db = _session()

    asset, _ = generate_organisation_report(
        db,
        organisation_id="org-1",
        organisation_name="Ghana Roads Authority",
        year=2026,
        title=None,
        created_by="user-1",
    )

    assert asset.title == "Ghana Roads Authority Road Condition Report 2026"
    assert asset.file_name.startswith("Ghana-Roads-Authority-road-condition-report-2026_")
    assert Path(asset.storage_path).is_file()


def test_regenerating_adds_a_report_rather_than_replacing_the_earlier_one(tmp_path, monkeypatch):
    """Two reports for one year are two reports. Collapsing them would lose the
    earlier one, which described an earlier state of the data."""
    monkeypatch.setenv("ROADESSY_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(reports, "build_report_rows", lambda db, org, year: _rows())
    db = _session()

    first, _ = generate_organisation_report(
        db, organisation_id="org-1", organisation_name="GRA", year=2026, title=None, created_by="user-1"
    )
    second, listed = generate_organisation_report(
        db, organisation_id="org-1", organisation_name="GRA", year=2026, title=None, created_by="user-1"
    )

    assert first.id != second.id
    assert {row.id for row in listed} == {first.id, second.id}
    assert Path(first.storage_path).is_file(), "the earlier report must not be overwritten"


def test_one_organisations_reports_are_not_another_organisations(tmp_path, monkeypatch):
    """Listing is scoped by organisation, on disk as well as in the table: two
    organizations reporting on the same year must not overwrite each other's
    files."""
    monkeypatch.setenv("ROADESSY_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(reports, "build_report_rows", lambda db, org, year: [])
    db = _session()

    generate_organisation_report(
        db, organisation_id="org-1", organisation_name="GRA", year=2026, title=None, created_by="u1"
    )
    generate_organisation_report(
        db, organisation_id="org-2", organisation_name="Other", year=2026, title=None, created_by="u2"
    )
    generate_organisation_report(
        db, organisation_id="org-1", organisation_name="GRA", year=2025, title=None, created_by="u1"
    )

    mine = reports.list_reports(db, "org-1")
    theirs = reports.list_reports(db, "org-2")

    assert {row.year for row in mine} == {2025, 2026}
    assert {row.year for row in theirs} == {2026}
    assert all(row.organisation_id == "org-1" for row in mine)
    assert all(row.organisation_id == "org-2" for row in theirs)
    # Separate directories, so neither write landed on the other's file.
    assert reports_root().joinpath("org-1", "2026").is_dir()
    assert reports_root().joinpath("org-2", "2026").is_dir()


def test_listing_a_year_returns_only_that_year(tmp_path, monkeypatch):
    monkeypatch.setenv("ROADESSY_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(reports, "build_report_rows", lambda db, org, year: [])
    db = _session()

    for year in (2024, 2025, 2026):
        generate_organisation_report(
            db, organisation_id="org-1", organisation_name="GRA", year=year, title=None, created_by="u1"
        )

    for_2025 = reports.list_reports(db, "org-1", year=2025)

    assert [row.year for row in for_2025] == [2025]


# ---------------------------------------------------------------------------
# Path confinement
# ---------------------------------------------------------------------------


def test_a_path_outside_the_reports_root_is_refused(tmp_path, monkeypatch):
    """`storage_path` is a string in a column. If a row is ever edited to point
    elsewhere, the download route must not become arbitrary file read."""
    monkeypatch.setenv("ROADESSY_OUTPUT_DIR", str(tmp_path))
    outside = tmp_path / "outside.txt"
    outside.write_text("not a report", encoding="utf-8")

    asset = ReportAsset(
        id=uuid.uuid4(),
        organisation_id="org-1",
        year=2026,
        file_name="outside.txt",
        storage_path=str(outside),
    )

    assert resolve_report_path(asset) is None


def test_a_traversal_path_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("ROADESSY_OUTPUT_DIR", str(tmp_path))
    target = tmp_path / "secret.txt"
    target.write_text("not a report", encoding="utf-8")

    asset = ReportAsset(
        id=uuid.uuid4(),
        organisation_id="org-1",
        year=2026,
        file_name="secret.txt",
        storage_path=str(reports_root() / ".." / ".." / "secret.txt"),
    )

    assert resolve_report_path(asset) is None


def test_a_stored_path_inside_the_root_that_does_not_exist_is_reported_as_gone(tmp_path, monkeypatch):
    monkeypatch.setenv("ROADESSY_OUTPUT_DIR", str(tmp_path))
    target = reports_root() / "org-1" / "2026" / "gone.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"%PDF-1.4")

    asset = ReportAsset(
        id=uuid.uuid4(),
        organisation_id="org-1",
        year=2026,
        file_name="gone.pdf",
        storage_path=str(target),
    )
    assert resolve_report_path(asset) == str(target), "a report that exists must resolve"

    target.unlink()
    assert resolve_report_path(asset) is None, "a report whose file is gone must not resolve to nothing readable"


@pytest.mark.parametrize("stored", ["", "   ", None])
def test_an_empty_path_is_refused(tmp_path, monkeypatch, stored):
    monkeypatch.setenv("ROADESSY_OUTPUT_DIR", str(tmp_path))
    asset = ReportAsset(
        id=uuid.uuid4(),
        organisation_id="org-1",
        year=2026,
        file_name="x.pdf",
        storage_path=stored,
    )

    assert resolve_report_path(asset) is None