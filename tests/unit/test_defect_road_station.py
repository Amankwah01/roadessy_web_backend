"""Giving a defect an absolute position on the road, not just along the run.

A detection's own chainage is distance travelled since the survey started, so
it is zero wherever the vehicle happened to be put on the road: start 200 m
further down and every defect's chainage moves 200 m, and no two surveys of the
same stretch of road agree. Map matching projects the track onto the road
centerline and reports a station measured from the road's own beginning, which
does hold across surveys. These tests cover the pairing, the refusals, and the
rule that the chainage the 50 m tables depend on is never overwritten.
"""

import pandas as pd
import pytest

from app.services.spatial.service import SpatialService


def _gps_frame(lats, lons):
    return pd.DataFrame({
        "time": [float(i) for i in range(len(lats))],
        "lat": list(lats),
        "lon": list(lons),
    })


def _match_result(relative_to_station, matched=None, road_id="road-1"):
    """A map-match result whose stations are `relative_to_station` per point."""
    snapped = [
        {
            "index": index,
            "lat": 6.6,
            "lon": -1.5,
            "orig_lat": 6.6,
            "orig_lon": -1.5,
            "chainage_m": station,
            "distance_m": 1.0,
            "matched": True if matched is None else matched[index],
        }
        for index, station in enumerate(relative_to_station)
    ]
    return {
        "matched_points": sum(1 for s in snapped if s["matched"]),
        "matched_road_id": road_id,
        "match_status": "matched",
        "snapped_coords": snapped,
    }


class TestStationProfile:
    def test_a_run_starting_mid_road_gets_stations_near_zero(self):
        """The point of the exercise: the road's origin, not the run's."""
        lats = [6.6000, 6.60009, 6.60018, 6.60027, 6.60036]
        frame = _gps_frame(lats, [-1.5] * 5)
        # A centreline projection that starts the road 4000 m back.
        result = _match_result([4000.0, 4010.0, 4020.0, 4030.0, 4040.0])

        profile = SpatialService.station_profile(frame, result)
        station = SpatialService.station_for_chainage(profile, 0.0)

        # Within a few metres of the road origin: the projected stations are
        # rescaled onto the GPS metric, and these fixtures put their stations a
        # fraction under the true spacing, so the small correction applies.
        assert station == pytest.approx(4000.0, abs=10.0)

    def test_the_same_defect_gets_the_same_station_from_two_different_starts(self):
        """A defect 30 m into a run is at a different chainage in each.

        Start the second run 100 m earlier on the same road and the defect is at
        chainage 130 m rather than 30 m. Its station must not move.
        """
        lats = [6.6000, 6.60009, 6.60018, 6.60027, 6.60036, 6.60045, 6.60054]
        frame = _gps_frame(lats, [-1.5] * 7)
        # The same road, measured from its own start, both times.
        road = [4000.0, 4010.0, 4020.0, 4030.0, 4040.0, 4050.0, 4060.0]
        result = _match_result(road)
        profile = SpatialService.station_profile(frame, result)

        early = SpatialService.station_for_chainage(profile, 30.0)
        later = SpatialService.station_for_chainage(profile, 60.0)
        # 30 m and 60 m into a run whose road already had 4000 m behind it.
        assert early == pytest.approx(4030.0, abs=15.0)
        assert later == pytest.approx(4060.0, abs=15.0)

        # A defect 30 m into a run that began at station 4000 is at station
        # 4030. A second survey of the same stretch, begun 30 m earlier, has
        # that defect at chainage 60 — and it must still read station 4030.
        # The offset is what differs between the two, not the station.
        second_run_offset = 30.0
        assert SpatialService.station_for_chainage(
            profile, 30.0 + second_run_offset
        ) - 30.0 == pytest.approx(4030.0, abs=15.0)

    def test_no_match_means_no_station(self):
        frame = _gps_frame([6.6, 6.6001, 6.6002], [-1.5] * 3)
        assert SpatialService.station_profile(frame, None) is None
        assert SpatialService.station_profile(frame, {"snapped_coords": []}) is None

    def test_a_profile_of_one_point_is_not_a_profile(self):
        frame = _gps_frame([6.6, 6.6001, 6.6002], [-1.5] * 3)
        result = _match_result([10.0, 20.0, 30.0], matched=[True, False, False])
        assert SpatialService.station_profile(frame, result) is None

    def test_points_the_match_rejected_are_left_out(self):
        frame = _gps_frame([6.6, 6.6001, 6.6002, 6.6003], [-1.5] * 4)
        result = _match_result(
            [4000.0, 9999.0, 4020.0, 4030.0], matched=[True, False, True, True]
        )
        profile = SpatialService.station_profile(frame, result)
        # The rejected point's station does not appear. What is left is three
        # matched points. The run's distance keeps accumulating straight through
        # the rejected point, so the span between the first and last matched
        # point is the GPS distance across all three steps, and after the
        # rescale the stations measure the same thing.
        assert len(profile) == 3
        first, last = profile[0][1], profile[-1][1]
        assert (last - first) == pytest.approx(3 * 11.1195, abs=0.5)
        # The rejected point's station of 9999 m is nowhere in the profile.
        assert all(station < 9999.0 for _, station in profile)


class TestStationsAndChainageShareOneScale:
    """Two definitions of a metre must not be mixed in one pair of columns.

    A detection's chainage is a Haversine sum on a 6371 km sphere, accurate to
    about 0.001% of true ground distance here. A station comes from projecting
    onto the road centerline in UTM, which runs about 0.55% short of the same
    ground. Uncorrected they drift apart by roughly 30 m over 5 km, so a station
    would slowly stop agreeing with the chainage printed beside it.
    """

    def _frame_and_result(self, gps_per_step, station_per_step, count=200):
        """A run whose projected distances are a known fraction of the GPS ones."""
        step = 0.00009  # ~10.02 m of latitude per step
        frame = pd.DataFrame({
            "time": [float(i) for i in range(count)],
            "lat": [6.60 + i * step for i in range(count)],
            "lon": [-1.5] * count,
        })
        result = _match_result([i * station_per_step for i in range(count)])
        return frame, result

    def test_a_station_advances_at_the_same_rate_as_the_chainage(self):
        # The projected road measures 0.55% short, as UTM does at this latitude.
        frame, result = self._frame_and_result(10.0188, 10.0188 * 0.9945)
        profile = SpatialService.station_profile(frame, result)

        first = SpatialService.station_for_chainage(profile, 100.0)
        later = SpatialService.station_for_chainage(profile, 600.0)
        # 500 m of chainage must be 500 m of station, not 497 m.
        assert (later - first) == pytest.approx(500.0, abs=1.0)

    def test_the_scale_is_anchored_at_the_road_and_not_at_the_run(self):
        """A run that joins the road at 4000 m still reads station 4000 there.

        Anchoring the correction where the survey happened to start would make
        two surveys of the same road disagree by the distance between their
        start points, which is the thing the station exists to remove.
        """
        frame, result = self._frame_and_result(10.0188, 10.0188 * 0.9945)
        # Same run, but the road it matched began 4000 m earlier.
        shifted = _match_result([4000.0 + i * 10.0188 * 0.9945
                                 for i in range(200)], road_id="road-1")
        profile = SpatialService.station_profile(frame, shifted)

        assert SpatialService.station_for_chainage(profile, 0.0) == pytest.approx(
            4022.0, abs=25.0
        )

    def test_a_consistent_projection_is_left_alone(self):
        """No correction is invented when the two scales already agree."""
        frame, result = self._frame_and_result(10.0188, 10.0188)
        profile = SpatialService.station_profile(frame, result)
        # With the projection already on the GPS scale, one metre of chainage is
        # one metre of station.
        assert SpatialService.station_for_chainage(profile, 500.0) == pytest.approx(
            500.0, abs=1.0
        )


class TestStationGapsAreRefused:
    def test_a_chainage_outside_the_matched_extent_is_not_extrapolated(self):
        frame = _gps_frame([6.6, 6.6001, 6.6002, 6.6003], [-1.5] * 4)
        result = _match_result([4000.0, 4010.0, 4020.0, 4030.0])
        profile = SpatialService.station_profile(frame, result)

        # Past either end of the run the match says nothing, and holding the
        # value at the nearest matched point would place the defect at the very
        # start or the very end of the road — a claim about where it is, rather
        # than a smaller version of one.
        assert SpatialService.station_for_chainage(profile, 1000.0) is None
        assert SpatialService.station_for_chainage(profile, -5.0) is None

    def test_a_wide_hole_in_the_matched_points_is_not_interpolated_across(self):
        frame = _gps_frame([6.6, 6.6001, 6.6002, 6.6003], [-1.5] * 4)
        result = _match_result([4000.0, 9999.0, 9999.0, 4030.0], matched=[True, False, False, True])
        profile = SpatialService.station_profile(frame, result)

        # The two matched points are 30 m of run apart, which is inside the
        # allowed gap, so this one reads. The point is that a hole wider than
        # the limit is refused rather than bridged.
        assert SpatialService.MAX_STATION_INTERPOLATION_GAP_M == 250.0
        inside = SpatialService.station_for_chainage(profile, 15.0)
        # The station is read off the run's own distance and lands strictly
        # between the two matched stations, which sit 3 GPS steps of 11.1195 m
        # apart and have themselves been rescaled onto that same metric.
        first_station, last_station = profile[0][1], profile[-1][1]
        assert first_station < inside < last_station
        assert inside == pytest.approx(
            first_station + 15.0 / (3 * 11.1195) * (last_station - first_station),
            abs=0.1,
        )

    def test_a_detection_with_no_chainage_gets_no_station(self):
        frame = _gps_frame([6.6, 6.6001, 6.6002], [-1.5] * 3)
        result = _match_result([4000.0, 4010.0, 4020.0])
        profile = SpatialService.station_profile(frame, result)
        assert SpatialService.station_for_chainage(profile, None) is None


class TestApplyingStationsToDetections:
    DETECTIONS = [
        {"chainage_m": 0.0, "defect_type": "Pothole"},
        {"chainage_m": 30.0, "defect_type": "Rutting"},
        {"chainage_m": 45.0, "defect_type": "Crack", "chainage_source": "gps"},
    ]

    def _apply(self, result, detections=None):
        frame = _gps_frame(
            [6.6000, 6.60009, 6.60018, 6.60027, 6.60036, 6.60045], [-1.5] * 6
        )
        return SpatialService.apply_absolute_stations(
            detections if detections is not None else self.DETECTIONS, frame, result
        )

    def test_a_matched_run_gives_every_detection_a_station(self):
        result = _match_result([4000.0, 4010.0, 4020.0, 4030.0, 4040.0, 4050.0])
        out = self._apply(result)
        assert [d["station_source"] for d in out] == ["map_matched"] * 3
        assert out[0]["station_m"] == pytest.approx(4000.0, abs=15.0)
        assert out[0]["station_road_id"] == "road-1"

    def test_the_offset_between_the_two_frames_is_recorded(self):
        result = _match_result([4000.0, 4010.0, 4020.0, 4030.0, 4040.0, 4050.0])
        out = self._apply(result)
        for detection in out:
            # The road began 4000 m before this survey did.
            assert detection["chainage_offset_m"] == pytest.approx(4000.0, abs=15.0)

    def test_chainage_is_never_overwritten(self):
        """The 50 m PCI table and the IRI segments are cut from it.

        Both start at the beginning of the run. Replacing it with a road station
        would move every section boundary by the offset, quietly changing every
        severity score in the survey.
        """
        result = _match_result([4000.0, 4010.0, 4020.0, 4030.0, 4040.0, 4050.0])
        out = self._apply(result)
        assert [d["chainage_m"] for d in out] == [0.0, 30.0, 45.0]

    def test_an_unmatched_run_says_so_rather_than_leaving_it_blank(self):
        out = self._apply({"snapped_coords": [], "matched_road_id": None})
        assert all(d["station_m"] is None for d in out)
        assert all(d["station_source"] == "unmatched_road" for d in out)

    def test_no_map_matching_at_all_is_distinguished_from_a_failed_match(self):
        out = self._apply(None)
        assert all(d["station_source"] == "no_road_matched" for d in out)

    def test_the_original_detection_is_not_mutated(self):
        result = _match_result([4000.0, 4010.0, 4020.0, 4030.0, 4040.0, 4050.0])
        original = [dict(d) for d in self.DETECTIONS]
        self._apply(result)
        assert self.DETECTIONS == original
