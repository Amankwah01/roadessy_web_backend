from types import SimpleNamespace

from shapely.geometry import LineString

from app.api.routes.dashboard import (
    _geometry_distance_m,
    _run_distance_m as dashboard_run_distance_m,
)
from app.api.routes.runs import _run_distance_m


class _Query:
    def __init__(self, scalar_result=None, first_result=None):
        self.scalar_result = scalar_result
        self.first_result = first_result

    def filter(self, *_args, **_kwargs):
        return self

    def scalar(self):
        return self.scalar_result

    def first(self):
        return self.first_result


class _Database:
    def __init__(self, query_results):
        self.query_results = iter(query_results)

    def query(self, *_args):
        return _Query(*next(self.query_results))


def test_geometry_distance_is_geodesic_and_recognizes_wkt_strings():
    geometry = LineString([(0, 0), (0, 0.01)])

    distance = _geometry_distance_m(geometry.wkt)

    assert distance is not None
    assert 1_100 < distance < 1_120


def test_run_distance_prefers_segment_chainage():
    db = _Database([(325.0,)])

    assert _run_distance_m(db, "run-id", None) == 325.0


def test_run_distance_uses_map_matching_when_segments_are_missing():
    matching = SimpleNamespace(matched_distance_m=420.0, road_length_m=500.0)
    db = _Database([(None,), (None, matching)])

    assert _run_distance_m(db, "run-id", None) == 420.0


def test_run_distance_uses_valid_road_length_when_matched_distance_is_invalid():
    matching = SimpleNamespace(matched_distance_m=-1.0, road_length_m=500.0)
    db = _Database([(None,), (None, matching)])

    assert _run_distance_m(db, "run-id", None) == 500.0


def test_dashboard_run_distance_uses_valid_road_length_when_match_is_invalid():
    matching = SimpleNamespace(matched_distance_m=-1.0, road_length_m=500.0)
    run = SimpleNamespace(run_id="run-id", geom_trajectory=None)
    db = _Database([(None,), (None, matching)])

    assert dashboard_run_distance_m(db, run) == 500.0


def test_run_distance_uses_trajectory_when_chainage_and_matching_are_missing():
    trajectory = LineString([(0, 0), (0, 0.01)]).wkt
    db = _Database([(None,), (None, None)])

    distance = _run_distance_m(db, "run-id", trajectory)

    assert distance is not None
    assert 1_100 < distance < 1_120


def test_run_distance_remains_unknown_without_any_distance_source():
    db = _Database([(None,), (None, None)])

    assert _run_distance_m(db, "run-id", None) is None
