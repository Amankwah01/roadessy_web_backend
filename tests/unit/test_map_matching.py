"""A run must not report "matched" when the drive did not follow the road.

One survey selected road 66673 — the nearest candidate — and then matched 10 of
893 GPS points to its centreline. Because `match_status` was
`"matched" if matched else "unmatched"`, any single matched point made the whole
run a success, and chainage was read off a centreline the drive never touched.
Every defect positioned from that chainage looked plausible and was wrong.

Raising the tolerance would not have helped: it makes more points snap to a road
they are not on. What was missing was the ratio check that says "this is not a
match", and the distance distribution that says why.
"""
import numpy as np
import pandas as pd
import pytest

from app.services.spatial.service import SpatialService


def _track(count, lat0=6.30, lon0=-1.65, dlat=0.0, dlon=0.0):
    """A straight run of GPS points, optionally offset from a point of interest."""
    return pd.DataFrame(
        {
            "lat": [lat0 + dlat + i * 1e-5 for i in range(count)],
            "lon": [lon0 + dlon + i * 1e-5 for i in range(count)],
        }
    )


#: A straight east-west centreline just south of the track above.
CENTERLINE = "LINESTRING (-1.6500 6.2990, -1.6400 6.2990)"


def _on_the_road(count):
    """Points hugging the centreline."""
    return pd.DataFrame(
        {
            "lat": [6.29901] * count,
            "lon": [-1.6500 + i * 1e-5 for i in range(count)],
        }
    )


class TestMatchStatus:
    def test_a_run_that_follows_the_road_is_matched(self):
        result = SpatialService.map_match_trajectory(
            gps_df=_on_the_road(40), centerline_wkt=CENTERLINE
        )

        assert result["match_status"] == SpatialService.MATCH_OK
        assert result["match_ratio"] == 1.0
        assert result["matched_points"] == 40

    def test_a_sparse_match_is_not_reported_as_a_match(self):
        """The regression: a handful of stray points near the line used to be
        enough to claim the whole run matched."""
        # 8 points on the centreline, 32 well outside the tolerance.
        near = pd.DataFrame(
            {
                "lat": [6.29901] * 8,
                "lon": [-1.6500 + i * 1e-5 for i in range(8)],
            }
        )
        far = pd.DataFrame(
            {
                "lat": [6.30500 + i * 1e-5 for i in range(32)],
                "lon": [-1.6500 + i * 1e-5 for i in range(32)],
            }
        )
        gps = pd.concat([near, far], ignore_index=True)

        result = SpatialService.map_match_trajectory(
            gps_df=gps, centerline_wkt=CENTERLINE
        )

        assert result["match_status"] == SpatialService.MATCH_POOR
        assert result["match_status"] != "matched"
        assert result["match_ratio"] < SpatialService.MIN_MATCH_RATIO

    def test_no_match_at_all_is_unmatched(self):
        result = SpatialService.map_match_trajectory(
            gps_df=_track(30, dlat=0.01), centerline_wkt=CENTERLINE
        )

        assert result["match_status"] == SpatialService.MATCH_NONE
        assert result["matched_points"] == 0

    def test_a_sparse_match_reports_why(self):
        """The distance distribution is what separates 'wrong road' from 'poor
        centreline geometry' — different problems with different fixes."""
        near = pd.DataFrame(
            {"lat": [6.29901] * 4, "lon": [-1.6500 + i * 1e-5 for i in range(4)]}
        )
        far = pd.DataFrame(
            {"lat": [6.30500] * 20, "lon": [-1.6500 + i * 1e-5 for i in range(20)]}
        )
        gps = pd.concat([near, far], ignore_index=True)

        result = SpatialService.map_match_trajectory(
            gps_df=gps, centerline_wkt=CENTERLINE
        )

        assert result["median_distance_m"] > SpatialService.POINT_MATCH_TOLERANCE_M
        # Every point is accounted for, matched or not. The far cluster sits
        # ~660 m off, which is what says "wrong road" rather than "GPS drift".
        assert result["max_distance_observed_m"] >= result["median_distance_m"]
        assert result["max_distance_observed_m"] > 500.0
        assert result["mean_distance_m"] > 0.0

    def test_snapped_coords_are_still_returned_for_a_poor_match(self):
        """A poor match must not silently discard the evidence that it was poor."""
        near = pd.DataFrame(
            {"lat": [6.29901] * 3, "lon": [-1.6500 + i * 1e-5 for i in range(3)]}
        )
        far = pd.DataFrame(
            {"lat": [6.30500] * 20, "lon": [-1.6500 + i * 1e-5 for i in range(20)]}
        )
        gps = pd.concat([near, far], ignore_index=True)

        result = SpatialService.map_match_trajectory(
            gps_df=gps, centerline_wkt=CENTERLINE
        )

        assert result["match_status"] == SpatialService.MATCH_POOR
        assert len(result["snapped_coords"]) == 23


class TestThresholdSeparation:
    def test_road_selection_and_point_matching_are_separate_dials(self):
        assert SpatialService.ROAD_SELECTION_RADIUS_M >= SpatialService.POINT_MATCH_TOLERANCE_M
        # They must be independently settable; a shared constant is what made the
        # two decisions impossible to reason about separately.
        assert "ROAD_SELECTION_RADIUS_M" in vars(SpatialService)
        assert "POINT_MATCH_TOLERANCE_M" in vars(SpatialService)
        assert not hasattr(SpatialService, "DEFAULT_MAX_DISTANCE_M")

    def test_an_explicit_tolerance_still_overrides(self):
        result = SpatialService.map_match_trajectory(
            gps_df=_on_the_road(20), centerline_wkt=CENTERLINE, max_distance_m=5.0
        )

        assert result["max_distance_m"] == 5.0

    def test_the_ratio_floor_is_configurable_on_the_class(self):
        assert 0.0 < SpatialService.MIN_MATCH_RATIO <= 1.0