"""Road selection must rank by shared centreline, not by closest approach.

`ORDER BY ST_Distance(...) LIMIT 1` ranked candidates by their single nearest
point to the trajectory. One survey picked road 66673 that way and then matched
10 of 893 GPS points to it: a road that merely brushed the track at one junction
outranked the road the vehicle actually drove along, and because the old status
rule accepted any non-empty match, the run reported success.

The query needs PostGIS, so it is not exercised here. What is pinned is the
property that was wrong — the ranking — so the query cannot silently go back to
nearest-approach without a test failing.
"""
import re

import pytest

from app.services.spatial.service import SpatialService


def _sql():
    """The SQL _find_nearest_road sends to the database."""
    source = SpatialService._find_nearest_road.__code__
    constants = [c for c in source.co_consts if isinstance(c, str) and "ST_" in c]
    assert constants, "no SQL string found on _find_nearest_road"
    return "\n".join(constants)


class TestRoadSelectionRanking:
    def test_candidates_are_ranked_by_shared_length(self):
        sql = _sql()

        assert re.search(r"ORDER BY\s+overlap_m\s+DESC", sql, re.IGNORECASE), (
            "road selection must rank by overlap length; ordering by distance "
            "reintroduces the nearest-approach bug"
        )

    def test_it_does_not_rank_by_nearest_distance(self):
        sql = _sql()

        assert not re.search(r"ORDER BY[^;]*ST_Distance", sql, re.IGNORECASE), (
            "ST_Distance returns the single closest approach between two "
            "geometries, which is what made a brushing road win"
        )

    def test_overlap_is_computed_and_reported(self):
        """The overlap length is what makes the choice explainable afterwards."""
        sql = _sql()

        assert "ST_Intersection" in sql
        assert "ST_Length" in sql
        assert "overlap_m" in sql

    def test_the_distance_predicate_is_kept_as_a_filter(self):
        """The radius still bounds the search; it is only the ranking that changed."""
        sql = _sql()

        assert "ST_DWithin" in sql
        assert "ST_Intersects" in sql

    def test_buffering_stays_in_metres(self):
        """Buffering a 4326 geometry directly would be in degrees, so the cast
        to geography is load-bearing, not incidental."""
        sql = _sql()

        assert re.search(r"ST_Buffer\(\s*geom::geography", sql, re.IGNORECASE)


class TestSelectionParameters:
    def test_selection_uses_the_selection_radius(self):
        """Not the point-match tolerance: those are different questions."""
        import inspect

        signature = inspect.signature(SpatialService._find_nearest_road)

        assert signature.parameters["max_distance_m"].default == (
            SpatialService.ROAD_SELECTION_RADIUS_M
        )

    @pytest.mark.parametrize(
        "name", ["ROAD_SELECTION_RADIUS_M", "POINT_MATCH_TOLERANCE_M", "MIN_MATCH_RATIO"]
    )
    def test_the_thresholds_are_independently_tunable(self, name):
        assert isinstance(getattr(SpatialService, name), float)