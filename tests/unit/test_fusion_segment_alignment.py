"""The two 50 m tables have to describe the same 50 m of road.

These cover the two failures the section tables actually had:

* the ASTM table passed absolute detection chainages against a 0-50 m section,
  so only the first segment ever matched a detection and a survey full of
  potholes scored 100 PCI on 59 of its 60 segments;
* the vision and vibration loggers start and stop at different points, so
  section 2 of one table was a different stretch of road than section 2 of the
  other. Aligning them is `compute_*_pipeline`'s job, and these check it does
  that, and says so when it cannot.
"""

import numpy as np
import pytest

from app.services.fusion import PavementConditionFusioneer
from app.services.fusion.alignment import build_inertial_track
from app.services.inertial import ASTMQCMResult, InertialPipelineResult, IRISegmentResult
from app.services.vision import ASTMVisionResult


def _qcm(segment_count, iri=3.0):
    return ASTMQCMResult(
        iri_m_km=iri,
        total_distance_m=segment_count * 50.0,
        iri_segments=[
            IRISegmentResult(i * 50.0, (i + 1) * 50.0, iri, 10)
            for i in range(segment_count)
        ],
        sample_rate_hz=100.0,
    )


def _simple(segment_count, iri=2.5):
    return InertialPipelineResult(
        iri_m_km=iri,
        total_distance_m=segment_count * 50.0,
        iri_segments=[
            IRISegmentResult(i * 50.0, (i + 1) * 50.0, iri, 10)
            for i in range(segment_count)
        ],
        raw_timestamps=None,
        vertical_displacement=None,
    )


@pytest.fixture
def astm_pci_on(monkeypatch):
    """Opt in to the muted D6433 PCI.

    The ASTM segment table is empty while the PCI is off, so the alignment these
    tests check cannot be observed through it. The mute is the default; this
    turns it on so the alignment logic itself stays covered.
    """
    from app.services.fusion import service as fusion_service

    monkeypatch.setattr(fusion_service, "ASTM_PCI_ENABLED", True)
    return True


def _vision(detections):
    return ASTMVisionResult(detections=detections, video_quality=None)


def _pothole(chainage_m, deduct_value=10.0):
    """A detection shaped like the ones the vision service emits.

    It carries a measured size. A detection with no area has zero density, and a
    zero-density distress earns no deduct value, so a fixture without one would
    report a perfectly clean segment and these tests would pass for the wrong
    reason. `yolo_class_id` 4 is pothole.
    """
    return {
        "chainage_m": chainage_m,
        "defect_type": "pothole",
        "yolo_class_id": 4,
        "area": 0.25,
        "width": 0.5,
        "deduct_value": deduct_value,
        "severity": "moderate",
    }


def _track(length_m=600.0, points=600):
    """A straight measured road running north from (6.60, -1.60)."""
    return build_inertial_track(
        np.linspace(6.60, 6.60 + length_m / 111320.0, points),
        np.full(points, -1.60),
    )


def _pothole_at_lat(lat, chainage_m):
    detection = _pothole(chainage_m)
    detection["geom"] = {"type": "Point", "coordinates": [-1.60, lat]}
    return detection


class TestAstmSegmentPciRebase:
    def test_a_detection_far_from_the_start_still_penalises_its_own_segment(self):
        """The bug: only segment 0 ever matched, so the rest read as perfect.

        A detection 210 m along the road belongs to segment 4 (200-250 m). It
        was being compared against a 0-50 m section, found to be outside the
        survey, and scored as a clean segment — so 59 of 60 segments reported
        PCI 100 on a survey containing potholes.
        """
        segments = PavementConditionFusioneer._compute_astm_segment_pci(
            _qcm(60).iri_segments, [_pothole(210.0)]
        )
        assert len(segments) == 60
        assert segments[4]["astm_pci"] < 100.0

    def test_only_the_owning_segment_is_penalised(self):
        """A single pothole must not lower the whole survey's score."""
        segments = PavementConditionFusioneer._compute_astm_segment_pci(
            _qcm(60).iri_segments, [_pothole(210.0)]
        )
        penalised = [s["segment_index"] for s in segments if s["astm_pci"] < 100.0]
        assert penalised == [4]

    def test_a_200m_section_containing_many_potholes_is_not_reported_as_good(self):
        """The reported symptom: 59 of 60 segments scoring 100."""
        detections = [_pothole(200.0 + i * 25.0) for i in range(40)]
        segments = PavementConditionFusioneer._compute_astm_segment_pci(
            _qcm(60).iri_segments, detections
        )
        damaged = [s for s in segments if s["astm_pci"] < 100.0]
        assert len(damaged) >= 8
        assert damaged, "a road with potholes in 8 sections cannot score clean in all of them"

    def test_detections_on_a_segment_boundary_land_in_one_segment_only(self):
        """A boundary is counted once, never in both neighbours or neither."""
        for boundary in (50.0, 100.0, 200.0):
            segments = PavementConditionFusioneer._compute_astm_segment_pci(
                _qcm(10).iri_segments, [_pothole(boundary)]
            )
            penalised = [s["segment_index"] for s in segments if s["astm_pci"] < 100.0]
            assert penalised == [int(boundary // 50)]

    def test_detections_past_the_survey_are_counted_and_given_no_row(self):
        """Vision measured 529 m further than vibration on the worst KNUST trip.

        The extra detections cannot be placed in a section, but they must be
        visible — a silently dropped defect is indistinguishable from a road
        without one.
        """
        segments = PavementConditionFusioneer._compute_astm_segment_pci(
            _qcm(4).iri_segments, [_pothole(10.0), _pothole(400.0)]
        )
        assert len(segments) == 4
        assert segments[0]["astm_pci"] < 100.0
        assert PavementConditionFusioneer._detections_beyond_survey(
            [_pothole(10.0), _pothole(400.0)], 200.0
        ) == 1

    def test_every_50m_interval_is_present_in_order(self):
        segments = PavementConditionFusioneer._compute_astm_segment_pci(
            _qcm(12).iri_segments, []
        )
        assert [s["segment_index"] for s in segments] == list(range(12))
        assert [s["start_m"] for s in segments] == [i * 50.0 for i in range(12)]
        assert [s["end_m"] for s in segments] == [(i + 1) * 50.0 for i in range(12)]

    def test_each_row_carries_the_iri_measured_over_it(self):
        """A PCI row is only meaningful beside the IRI for the same 50 m."""
        qcm = _qcm(6)
        qcm.iri_segments[3] = IRISegmentResult(150.0, 200.0, 7.5, 10)
        segments = PavementConditionFusioneer._compute_astm_segment_pci(qcm.iri_segments, [])
        assert [s["astm_iri_m_km"] for s in segments] == [3.0, 3.0, 3.0, 7.5, 3.0, 3.0]

    def test_a_later_pothole_costs_more_than_a_first_segment_one(self):
        """Same defect, same damage, different segment: the scores must match.

        A rebase that shifted a section rather than the detections would make
        the same pothole score differently depending on where it sat.
        """
        first = PavementConditionFusioneer._compute_astm_segment_pci(
            _qcm(20).iri_segments, [_pothole(10.0)]
        )
        last = PavementConditionFusioneer._compute_astm_segment_pci(
            _qcm(20).iri_segments, [_pothole(960.0)]
        )
        assert first[0]["astm_pci"] == last[19]["astm_pci"]

    def test_no_iri_segments_produces_no_rows(self):
        assert PavementConditionFusioneer._compute_astm_segment_pci([], [_pothole(10.0)]) == []


class TestPipelineAlignment:
    def test_a_camera_that_started_early_is_corrected_so_the_tables_agree(
        self, astm_pci_on
    ):
        """The camera began 470 m before the accelerometer; both must line up.

        Left uncorrected, the pothole sits at segment 0 of the defect table
        while the first 470 m of measured road has no detection beside it — a
        reader comparing section 0 of each is comparing the start of the road
        with the middle of it.
        """
        track = _track(600.0)
        # 470 m along the measured road is the camera's own 0 m.
        detection = _pothole_at_lat(6.60 + 470.0 / 111320.0, 0.0)
        result = PavementConditionFusioneer.compute_astm_pipeline(
            astm_qcm=_qcm(12),
            astm_vision=_vision([detection]),
            inertial_track=track,
        )
        assert result["alignment"]["aligned"] is True
        damaged = [s for s in result["segment_pci_50m"] if s["astm_pci"] < 100.0]
        assert [s["segment_index"] for s in damaged] == [9]  # 470 m -> 450-500 m
        assert damaged[0]["start_m"] == 450.0

    def test_the_aligned_detection_keeps_its_vision_chainage(self):
        """The camera's own distance is the record of where the video was."""
        track = _track(600.0)
        detection = _pothole_at_lat(6.60 + 470.0 / 111320.0, 0.0)
        aligned = PavementConditionFusioneer._align_detections([detection], track)[0]
        assert aligned[0]["vision_chainage_m"] == 0.0
        assert aligned[0]["chainage_m"] == pytest.approx(470.0, abs=15.0)
        assert aligned[0]["chainage_source"] == "gps_aligned"

    def test_both_pipelines_align_the_same_way(self, astm_pci_on):
        """ASTM and Roadessy index the same sections, so they must not disagree."""
        track = _track(600.0)
        detection = _pothole_at_lat(6.60 + 470.0 / 111320.0, 0.0)
        astm = PavementConditionFusioneer.compute_astm_pipeline(
            astm_qcm=_qcm(12), astm_vision=_vision([detection]), inertial_track=track
        )
        roadessy = PavementConditionFusioneer.compute_roadessy_pipeline(
            simple_result=_simple(12), roadessy_vision=_vision([detection]), inertial_track=track
        )
        assert astm["alignment"]["placed"] == roadessy["alignment"]["placed"]
        assert [s["segment_index"] for s in astm["segment_pci_50m"] if s["astm_pci"] < 100.0] == [
            s["segment_index"] for s in roadessy["segment_pci_50m"] if s["roadessy_pci"] < 100.0
        ]

    def test_a_jump_contaminated_track_is_refused_rather_than_used(self):
        """The vibration GPS is frozen and then teleports on every KNUST trip.

        Projecting onto it would place each detection against a straight line
        across country and report a section index with total confidence. The
        pipeline has to refuse and say why instead.
        """
        lat = np.concatenate([np.full(400, 6.60), np.full(200, 6.65), np.linspace(6.6501, 6.655, 200)])
        track = build_inertial_track(lat, np.full(800, -1.60))
        assert track.is_contiguous is False

        result = PavementConditionFusioneer.compute_astm_pipeline(
            astm_qcm=_qcm(12), astm_vision=_vision([_pothole(210.0)]), inertial_track=track
        )
        assert result["alignment"]["aligned"] is False
        assert "not continuous" in result["alignment"]["reason"]

    def test_without_a_track_the_pipelines_say_the_tables_are_not_comparable(self):
        """No track means no shared origin, and that must be stated not hidden."""
        result = PavementConditionFusioneer.compute_astm_pipeline(
            astm_qcm=_qcm(4), astm_vision=_vision([_pothole(10.0)])
        )
        assert result["alignment"]["aligned"] is False
        assert "reason" in result["alignment"]

    def test_the_roadessy_pipeline_reports_its_alignment_too(self):
        result = PavementConditionFusioneer.compute_roadessy_pipeline(
            simple_result=_simple(4), roadessy_vision=_vision([_pothole(10.0)])
        )
        assert result["alignment"]["aligned"] is False

    def test_a_detection_from_another_road_is_left_out_rather_than_snapped_in(self):
        """A GPS fix from a different trip must not become a section of this road."""
        track = _track(600.0)
        detection = _pothole_at_lat(6.60 + 100.0 / 111320.0, 10.0)
        detection["geom"]["coordinates"] = [-1.55, 6.60]  # ~5.5 km east
        result = PavementConditionFusioneer.compute_astm_pipeline(
            astm_qcm=_qcm(12), astm_vision=_vision([detection]), inertial_track=track
        )
        assert result["alignment"]["placed"] == 0
        assert result["alignment"]["unplaced"] == 1
        assert all(s["astm_pci"] == 100.0 for s in result["segment_pci_50m"])

    def test_a_vision_run_longer_than_the_survey_is_reported_not_truncated_silently(
        self, astm_pci_on
    ):
        """The worst KNUST trip: vibration 5.808 km, vision 6.337 km.

        Both tables keep the survey's 116 segments; the extra 529 m of detections
        are reported rather than given rows carrying a PCI with no IRI beside them.
        """
        track = _track(5808.0, points=1200)
        detections = [
            _pothole_at_lat(6.60 + 100.0 / 111320.0, 0.0),
            _pothole_at_lat(6.60 + 6337.0 / 111320.0, 5000.0),  # past the survey
        ]
        result = PavementConditionFusioneer.compute_astm_pipeline(
            astm_qcm=_qcm(116), astm_vision=_vision(detections), inertial_track=track
        )
        assert len(result["segment_pci_50m"]) == 116
        assert len(result["qcm_iri_segments_50m"]) == 116
        assert result["alignment"]["placed"] == 1
        assert result["alignment"]["unplaced"] == 1
