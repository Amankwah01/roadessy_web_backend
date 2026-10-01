"""Tests for placing vision detections on the road the IRI was measured over.

The two 50 m tables are indexed by distance, and the two distances come from two
loggers started by hand. These cover the projection that puts both on one
stretch of road, and — just as importantly — the cases where that projection is
impossible and the two tables must instead be reported as not comparable.
"""

import numpy as np
import pytest

from app.services.fusion.alignment import (
    MAX_PLAUSIBLE_STEP_M,
    ON_TRACK_TOLERANCE_M,
    apply_alignment,
    build_inertial_track,
    cumulative_distance_m,
    place_detections,
)

# A straight north-south road: 0.001 deg of latitude is ~111.32 m, so this track
# runs 0 to ~1113 m over 1000 points.
_LAT_SPAN_DEG = 0.01
_POINTS = 1000


def _track():
    return build_inertial_track(
        np.linspace(6.60, 6.60 + _LAT_SPAN_DEG, _POINTS),
        np.full(_POINTS, -1.60),
    )


def _detection_at_lat(lat, lon=-1.60, chainage_m=0.0):
    return {
        "chainage_m": chainage_m,
        "geom": {"type": "Point", "coordinates": [lon, lat]},
    }


def _lat_at_distance_m(metres):
    return 6.60 + metres / 111320.0


class TestBuildInertialTrack:
    def test_cumulative_distance_matches_coordinate_separation(self):
        track = _track()
        expected = cumulative_distance_m(track.lat, track.lon)
        assert track.total_distance_m == pytest.approx(expected[-1])

    def test_distance_starts_at_zero_and_increases(self):
        track = _track()
        assert track.distance_m[0] == 0.0
        assert np.all(np.diff(track.distance_m) > 0)

    def test_missing_coordinates_yield_no_track(self):
        assert build_inertial_track(None, None) is None
        assert build_inertial_track([6.6], [None]) is None

    def test_a_stationary_logger_is_not_a_zero_length_road(self):
        """Constant coordinates mean no fix, not a stationary vehicle.

        Reporting it as a road of length zero would put every detection off-track
        and silently unalign the two tables.
        """
        assert build_inertial_track([6.6] * 10, [-1.6] * 10) is None

    def test_a_single_point_is_not_enough_to_measure_against(self):
        assert build_inertial_track([6.6], [-1.6]) is None

    def test_non_finite_points_are_dropped(self):
        track = build_inertial_track(
            [6.60, np.nan, 6.60005, np.inf, 6.6001],
            [-1.60, -1.60, -1.60, -1.60, -1.60],
        )
        assert len(track) == 3


class TestPlaceDetections:
    def test_a_detection_lands_at_its_own_position_on_the_road(self):
        """A camera measuring from its own origin is still placed correctly.

        The camera's chainage of 0 m corresponds to 470 m along the measured
        road, so a real 470 m offset is expected here — the point is that the
        *placed* distance reflects where the coordinate physically is, not the
        chainage the vision service happened to compute.
        """
        track = _track()
        placement = place_detections(
            [_detection_at_lat(_lat_at_distance_m(470.0), chainage_m=0.0)], track
        ).placements[0]
        assert placement.distance_m == pytest.approx(470.0, abs=15.0)
        assert placement.offset_m == pytest.approx(470.0, abs=15.0)

    def test_offset_is_the_difference_between_the_two_chains(self):
        track = _track()
        placement = place_detections(
            [_detection_at_lat(_lat_at_distance_m(500.0), chainage_m=30.0)], track
        ).placements[0]
        assert placement.offset_m == pytest.approx(
            placement.distance_m - 30.0, abs=1.0
        )

    def test_a_detection_far_from_the_road_is_reported_not_absorbed(self):
        """A coordinate from another trip must not be snapped onto this road.

        150 m is generous against a few metres of GPS noise and far below the
        ~11 km between junctions, so anything past it is a different road.
        """
        track = _track()
        placement = place_detections(
            [_detection_at_lat(_lat_at_distance_m(100.0), lon=-1.55)], track
        ).placements[0]
        assert placement.distance_m is None
        assert placement.off_track_m > ON_TRACK_TOLERANCE_M

    def test_a_detection_without_coordinates_cannot_be_placed(self):
        track = _track()
        placement = place_detections([{"chainage_m": 10.0}], track).placements[0]
        assert placement.distance_m is None
        assert placement.off_track_m is None

    def test_the_sentinels_for_no_gps_are_not_real_coordinates(self):
        """(0, 0) is Null Island and a null GPS reading, not a position."""
        track = _track()
        detection = _detection_at_lat(0.0, lon=0.0, chainage_m=10.0)
        assert place_detections([detection], track).placements[0].distance_m is None

    def test_out_of_range_coordinates_are_rejected(self):
        track = _track()
        detection = {
            "chainage_m": 10.0,
            "geom": {"type": "Point", "coordinates": [500.0, 500.0]},
        }
        assert place_detections([detection], track).placements[0].distance_m is None

    def test_no_track_leaves_everything_unplaced(self):
        alignment = place_detections([_detection_at_lat(6.6001)], None)
        assert alignment.placed_count == 0
        assert alignment.unplaced_count == 1
        assert alignment.is_aligned is False

    def test_placed_and_unplaced_are_counted_separately(self):
        track = _track()
        alignment = place_detections(
            [
                _detection_at_lat(_lat_at_distance_m(100.0)),
                {"chainage_m": 5.0},
                _detection_at_lat(_lat_at_distance_m(200.0), lon=-1.55),
            ],
            track,
        )
        assert alignment.placed_count == 1
        assert alignment.unplaced_count == 2

    def test_no_detections_is_not_an_error(self):
        alignment = place_detections([], _track())
        assert alignment.placed_count == 0
        assert alignment.is_aligned is False


class TestTrackContinuity:
    def test_a_normal_drive_is_contiguous(self):
        assert _track().is_contiguous is True

    def test_a_stale_fix_that_snaps_back_is_not_a_road(self):
        """The shape every KNUST Obuasi vibration log actually has.

        The logger holds a fix for most of the trip and then jumps across
        country. Its cumulative distance counts that straight line as driven
        road, so projecting detections onto it would place them against road
        nobody drove — confidently, and wrongly.
        """
        # 400 stationary fixes, then a 5 km leap, then the real track resumes.
        lat = np.concatenate([
            np.full(400, 6.600),
            np.full(200, 6.650),  # ~5.56 km north
            np.linspace(6.6501, 6.655, 200),
        ])
        lon = np.concatenate([
            np.full(600, -1.600),
            np.full(200, -1.600),
        ])
        track = build_inertial_track(lat, lon)
        assert track.is_contiguous is False
        assert track.teleport_count == 1
        assert track.worst_step_m > MAX_PLAUSIBLE_STEP_M

    def test_the_jump_is_what_makes_the_distance_wrong(self):
        """Without the teleport the same fixes measure a real 7.9 km road."""
        lat = np.concatenate([
            np.full(400, 6.600),
            np.full(200, 6.650),
            np.linspace(6.6501, 6.655, 200),
        ])
        lon = np.full(800, -1.600)
        contaminated = build_inertial_track(lat, lon)
        assert contaminated.total_distance_m > 5_500.0
        assert contaminated.is_contiguous is False

    def test_a_step_just_under_the_limit_is_accepted(self):
        """The limit must not reject real driving."""
        step_deg = (MAX_PLAUSIBLE_STEP_M * 0.9) / 111320.0
        track = build_inertial_track(
            np.arange(20) * step_deg, np.full(20, -1.600)
        )
        assert track.is_contiguous is True
        assert track.worst_step_m < MAX_PLAUSIBLE_STEP_M


class TestAlignmentSummary:
    def test_an_unplaceable_chain_is_reported_as_not_comparable(self):
        """The point of the summary is that it can say "no" out loud."""
        summary = place_detections([_detection_at_lat(6.6001)], None).summary()
        assert summary["aligned"] is False
        assert "reason" in summary

    def test_a_usable_projection_reports_its_offset(self):
        track = _track()
        summary = place_detections(
            [
                _detection_at_lat(_lat_at_distance_m(100.0), chainage_m=0.0),
                _detection_at_lat(_lat_at_distance_m(400.0), chainage_m=300.0),
            ],
            track,
        ).summary()
        assert summary["aligned"] is True
        assert summary["placed"] == 2
        assert summary["unplaced"] == 0
        assert summary["offset_median_m"] == pytest.approx(100.0, abs=15.0)
        assert summary["track_length_m"] == pytest.approx(track.total_distance_m, abs=1.0)

    def test_the_summary_carries_the_quality_of_the_frame_it_used(self):
        """A frozen logger still yields confident section indices.

        The quality of the track those indices were cut in has to travel with
        them, or a reader has no way to know the frame was a staircase.
        """
        summary = place_detections(
            [_detection_at_lat(_lat_at_distance_m(100.0))], _track()
        ).summary()
        assert set(summary["track_quality"]) == {
            "contiguous", "teleport_count", "worst_step_m", "median_step_m", "frozen_fraction",
        }
        assert summary["track_quality"]["contiguous"] is True

    def test_a_frozen_track_advertises_itself_as_frozen(self):
        lat = np.concatenate([np.full(400, 6.60), np.linspace(6.6001, 6.605, 200)])
        track = build_inertial_track(lat, np.full(600, -1.60))
        assert track.median_step_m == 0.0
        assert track.frozen_fraction > 0.5

    def test_spread_distinguishes_a_shift_from_a_different_road(self):
        """A consistent offset just shifts the table; a spread means no shift fits.

        Two detections whose offsets differ by hundreds of metres cannot both be
        explained by the camera starting late, which is what a reader needs to
        know before trusting the section indices.
        """
        track = _track()
        summary = place_detections(
            [
                _detection_at_lat(_lat_at_distance_m(100.0), chainage_m=0.0),
                _detection_at_lat(_lat_at_distance_m(900.0), chainage_m=0.0),
            ],
            track,
        ).summary()
        assert summary["offset_spread_m"] > 500.0


class TestApplyAlignment:
    def test_chainage_becomes_the_measured_road_distance(self):
        track = _track()
        detections = [_detection_at_lat(_lat_at_distance_m(470.0), chainage_m=0.0)]
        aligned = apply_alignment(detections, place_detections(detections, track))
        assert aligned[0]["chainage_m"] == pytest.approx(470.0, abs=15.0)
        assert aligned[0]["chainage_source"] == "gps_aligned"

    def test_the_original_chainage_is_kept_as_the_vision_frame(self):
        """Discarding it would lose the only record of where the video was."""
        track = _track()
        detections = [_detection_at_lat(_lat_at_distance_m(470.0), chainage_m=0.0)]
        aligned = apply_alignment(detections, place_detections(detections, track))
        assert aligned[0]["vision_chainage_m"] == 0.0
        assert aligned[0]["chainage_offset_m"] == pytest.approx(470.0, abs=15.0)

    def test_an_unplaced_detection_says_why_instead_of_being_guessed_at(self):
        track = _track()
        detections = [
            {"chainage_m": 10.0},
            _detection_at_lat(_lat_at_distance_m(100.0), lon=-1.55, chainage_m=7.0),
        ]
        aligned = apply_alignment(detections, place_detections(detections, track))
        assert aligned[0]["chainage_source"] == "unaligned"
        assert aligned[1]["chainage_source"] == "off_track"
        # Neither invents a distance: each keeps the chainage it came with.
        assert aligned[0]["chainage_m"] == 10.0
        assert aligned[1]["chainage_m"] == 7.0

    def test_the_input_detections_are_not_mutated(self):
        track = _track()
        detections = [_detection_at_lat(_lat_at_distance_m(470.0), chainage_m=0.0)]
        apply_alignment(detections, place_detections(detections, track))
        assert detections[0]["chainage_m"] == 0.0
        assert "vision_chainage_m" not in detections[0]
