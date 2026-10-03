"""Regression tests for the road a run creates inline.

Entering a road name on a survey used to fail. The row was built without
`project_name` or `created_by`, both NOT NULL with no default, and it was only
flushed rather than committed. So the caller saw an IntegrityError reported as a
500, and even a row that had flushed would have been discarded when the request
session closed — and the pipeline, which opens its own session, would never have
seen it to satisfy the survey_runs.road_id foreign key.
"""

import asyncio
import uuid

from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from starlette.datastructures import UploadFile

from app.api.routes import analysis
from app.models.survey import Project, Road

# geoalchemy2 routes its DDL and its bound values through these PostGIS
# functions. SQLite has none of them, so they are stubbed onto the connection.
# `roads.road_geom` is never set by this test and stays null; the stubs only
# have to let the DDL run and the null-valued INSERT bind without erroring.
_POSTGIS_FUNCTIONS = [
    "RecoverGeometryColumn",
    "CreateSpatialIndex",
    "DropSpatialIndex",
    "AddGeometryColumn",
    "UpdateGeometrySRID",
    "GeomFromEWKT",
    "ST_GeomFromEWKB",
    "ST_AsEWKB",
    "ST_AsText",
    "ST_Equals",
    "AsEWKB",
    "AsEWKT",
    "AsText",
    "AsBinary",
    "ST_AsBinary",
]


def _session(tmp_path):

    """A session over just the tables this test touches.

    Building every model would create the other geometry columns too, which is
    why the functions above are stubbed rather than every table being created.
    """
    engine = _engine(tmp_path)
    Project.__table__.create(bind=engine)
    Road.__table__.create(bind=engine)
    return sessionmaker(bind=engine)(), engine


def _engine(tmp_path):
    # A file-backed database with NullPool, so every session gets a genuinely
    # separate connection. That separation is the whole point: it is what makes
    # a second session unable to see a row that was only flushed, and therefore
    # what lets the commit test fail when the commit is missing. An in-memory
    # database with StaticPool shares one connection between sessions and would
    # show the flushed row regardless, so it cannot detect the regression.
    engine = create_engine(f"sqlite:///{tmp_path / 'roads.db'}", poolclass=NullPool)

    @event.listens_for(engine, "connect")

    def _stub_postgis(dbapi_connection, _record):
        for name in _POSTGIS_FUNCTIONS:
            dbapi_connection.create_function(name, -1, lambda *args: None)

    return engine


def _project(session, org_id="org-1"):
    project = Project(
        id=uuid.uuid4(),
        name="Kumasi Bypass",
        org_id=org_id,
        org_name="Roadessy",
        created_by="user-1",
    )
    session.add(project)
    session.commit()
    return project


def _call(session, project, principal=None, **overrides):
    """Call /process with the pipeline stubbed out.

    Only the road-creation step is under test; running the real pipeline would
    need a real video and GPS file.
    """
    stub = {
        "output_dir": "out",
        "pipeline_1_astm": {},
        "pipeline_2_roadessy": {},
        "map_matching": None,
    }
    kwargs = dict(
        user_id="user-1",
        organisation_id="org-1",
        name="Run 1",
        matched_road_id=None,
        road_id=None,
        # A real UUID, not the string a form post would carry. Postgres coerces
        # the string to a UUID itself, but SQLite's Uuid type has no such
        # coercion and would fail on `.hex` before reaching the code under test.
        project_id=project.id,

        road_name="  Junction A to B  ",
        road_start_node=" Junction A ",
        road_end_node=" Junction B ",
        road_surface_type=" Asphalt ",
        road_description=None,
        video=UploadFile(filename="v.mp4", file=open(__file__, "rb")),
        gps=None,
        gps_source="auto",
        vibration=UploadFile(filename="v.csv", file=open(__file__, "rb")),
        calibration_metadata_json=None,
        calibration_profile_id=None,
        use_overlay_clock=False,
    )
    kwargs.update(overrides)

    with patch.object(analysis, "run_heavy_pipeline", return_value=stub) as pipeline:
        response = asyncio.run(
            analysis.process_full_survey(
                db=session,
                principal=principal if principal is not None else {"user_id": "user-1"},
                **kwargs,
            )
        )
    return response, pipeline



def test_inline_road_is_persisted_with_section_details(tmp_path):
    session, _engine_ = _session(tmp_path)
    project = _project(session)

    _call(session, project)

    road = session.query(Road).one()
    assert road.name == "Junction A to B", "name should be trimmed, not stored padded"
    assert road.start_node == "Junction A"
    assert road.end_node == "Junction B"
    assert road.surface_type == "Asphalt"
    assert road.project_id == project.id


def test_inline_road_fills_the_not_null_columns(tmp_path):
    """`project_name` and `created_by` have no default; omitting them is an
    IntegrityError, which was the actual failure people were seeing."""
    session, _engine_ = _session(tmp_path)
    project = _project(session)

    _call(session, project)

    road = session.query(Road).one()
    assert road.project_name == "Kumasi Bypass"
    assert road.created_by == "user-1"


def test_inline_road_is_committed_not_merely_flushed(tmp_path):
    """The pipeline runs in its own session and needs to see this row.

    A flushed-but-uncommitted row lives only in the request's transaction, so it
    disappears when that session closes and is invisible to any other session.
    """
    session, engine = _session(tmp_path)
    project = _project(session)

    _, pipeline = _call(session, project)

    # A brand-new session: anything it can see was committed.
    fresh = sessionmaker(bind=engine)()
    assert fresh.query(Road).count() == 1
    # And the id handed to the pipeline is the committed row's id, so the
    # survey_runs.road_id foreign key resolves from another session.
    assert pipeline.call_args.kwargs["road_uuid"] == str(fresh.query(Road).one().id)


def test_taken_road_name_is_a_conflict_not_a_server_fault(tmp_path):
    session, _engine_ = _session(tmp_path)
    project = _project(session)
    session.add(
        Road(
            project_id=project.id,
            project_name=project.name,
            name="Junction A to B",
            created_by="user-1",
        )
    )
    session.commit()

    with pytest.raises(Exception) as raised:
        _call(session, project)

    assert getattr(raised.value, "status_code", None) == 409
    assert "already exists" in getattr(raised.value, "detail", "")


def test_existing_road_id_is_left_alone(tmp_path):
    """A run pointed at an existing road must not create a second one.

    The road name is still sent alongside the id, which is the realistic case:
    the form carries a name the user typed before picking a road from the list.
    It is the id, not the absence of a name, that has to suppress the creation.
    """
    session, _engine_ = _session(tmp_path)
    project = _project(session)
    existing = Road(
        project_id=project.id,
        project_name=project.name,
        name="Already here",
        created_by="user-1",
    )
    session.add(existing)
    session.commit()

    _call(session, project, road_id=str(existing.id))

    assert session.query(Road).count() == 1

