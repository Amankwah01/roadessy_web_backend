"""Tests for the fusion service.

Scoped to what `PavementConditionFusioneer` actually exposes. An earlier
version of this file tested `evaluate_severity`, `calculate_deduct_value` and
`compute_pci_cascade`; none of those exist on the class, and no production code
called them, so the whole file failed on a phantom API. These cover the real
methods — the PCI classifiers, the vision and Roadessy PCI computations, and
the segment-level variants of both.

Behavioural claims here were read off the implementation and confirmed by
running it, rather than assumed from the old expectations.
"""

import pytest

from app.services.fusion import PavementConditionFusioneer
from app.services.inertial import IRISegmentResult


def _segments(count=1, iri=3.0, length_m=100.0):
    return [
        IRISegmentResult(
            start_distance_m=i * length_m,
            end_distance_m=(i + 1) * length_m,
            iri_m_km=iri,
            sample_count=10,
        )
        for i in range(count)
    ]


@pytest.fixture
def astm_pci_on(monkeypatch):
    """Turn the muted D6433 PCI on for the duration of a test.

    The calculation is disabled by default (see `app/core/config.py`), so any
    test asserting a computed PCI has to opt in explicitly. That these tests
    exist and still pass is the evidence that the code is retained rather than
    deleted: flipping this fixture back off must not require touching them.
    """
    from app.services.fusion import service as fusion_service

    monkeypatch.setattr(fusion_service, "ASTM_PCI_ENABLED", True)
    return True


class _Vision:
    """Stand-in for ASTMVisionResult; only `.detections` is read."""

    def __init__(self, detections):
        self.detections = detections


class TestClassifyPci:
    @pytest.mark.parametrize(
        "pci,expected",
        [
            (100, "Good"),
            (86, "Good"),
            (85, "Satisfactory"),
            (71, "Satisfactory"),
            (56, "Fair"),
            (41, "Poor"),
            (26, "Very Poor"),
            (11, "Serious"),
            (10, "Failed"),
            (0, "Failed"),
        ],
    )
    def test_each_band(self, pci, expected):
        assert PavementConditionFusioneer._classify_pci(pci) == expected

    def test_bands_are_contiguous_with_no_gap(self):
        """Every score between 0 and 100 lands in a real band.

        The bands used to be inclusive ranges, which left a one-unit gap at each
        boundary: 85.999 matched nothing and fell through to "Failed", the worst
        rating, for a road one point from Satisfactory. These sit just below each
        boundary, where a gap would show up.
        """
        for boundary, expected_below in [
            (86.0, "Satisfactory"),
            (71.0, "Fair"),
            (56.0, "Poor"),
            (41.0, "Very Poor"),
            (26.0, "Serious"),
            (11.0, "Failed"),
        ]:
            assert PavementConditionFusioneer._classify_pci(boundary - 0.001) == expected_below

    def test_exact_boundary_takes_the_higher_band(self):
        assert PavementConditionFusioneer._classify_pci(86) == "Good"
        assert PavementConditionFusioneer._classify_pci(71) == "Satisfactory"

    def test_out_of_range_falls_back_to_failed(self):
        """A score outside every band is reported Failed, not raised.

        PCI is a percentage, so a value above 100 or below 0 means an upstream
        calculation went wrong. That should not take down the request.
        """
        assert PavementConditionFusioneer._classify_pci(101) == "Failed"
        assert PavementConditionFusioneer._classify_pci(-5) == "Failed"


class TestComputeVisionPci:
    def test_no_defects_is_a_perfect_road(self):
        pci, breakdown = PavementConditionFusioneer._compute_vision_pci({})
        assert pci == 100.0
        assert breakdown["defect_count"] == 0
        assert breakdown["vision_pci"] == 100.0

    def test_zero_deduct_values_do_not_reduce_pci(self):
        """A tracked defect carrying no deduct value contributes nothing.

        A defect can be tracked and still score zero; counting it as damage
        would penalise a road for something that measured clean.
        """
        pci, _ = PavementConditionFusioneer._compute_vision_pci(
            {1: {"deduct_value": 0.0, "defect_type": "pothole"}}
        )
        assert pci == 100.0

    def test_defect_reduces_pci_and_reports_its_own_value(self):
        pci, breakdown = PavementConditionFusioneer._compute_vision_pci(
            {1: {"deduct_value": 5.0, "defect_type": "pothole"}}
        )
        assert pci == 95.0
        assert breakdown["total_dv"] == 5.0
        assert breakdown["defect_count"] == 1
        assert breakdown["breached_types"] == 1

    def test_correction_factor_falls_as_defect_count_grows(self):
        """Fewer severe defects is penalised harder per defect than many.

        The factor is keyed on how many defects breach the 5.0 threshold, so it
        decreases as that count rises — the standard multi-defect adjustment,
        where spreading the same damage around is not as bad as concentrating it.
        """
        factors = []
        for count in range(1, 6):
            registry = {
                i: {"deduct_value": 5.0, "defect_type": "pothole"} for i in range(count)
            }
            _, breakdown = PavementConditionFusioneer._compute_vision_pci(registry)
            factors.append(breakdown["correction_factor"])
        assert factors == sorted(factors, reverse=True)

    def test_pci_never_goes_below_zero(self):
        """A total deduct value beyond 100 clamps at 0 rather than going negative."""
        registry = {i: {"deduct_value": 90.0, "defect_type": "pothole"} for i in range(5)}
        pci, _ = PavementConditionFusioneer._compute_vision_pci(registry)
        assert pci >= 0.0


class TestComputeRoadessyPci:
    def test_no_detections_is_a_perfect_road(self):
        pci, breakdown = PavementConditionFusioneer._compute_roadessy_pci(
            4.0, _segments(), _Vision([])
        )
        assert pci == 100.0
        assert breakdown["roadessy_dv_sum"] == 0.0
        assert breakdown["roadessy_pci_class"] == "Good"

    def test_iri_is_reported_separately_and_does_not_change_pci(self):
        """PCI comes from defects alone; IRI is carried alongside it.

        Per the Ghana Manual the two are distinct measures, so a worse IRI must
        not quietly lower a defect-based PCI.
        """
        mild, _ = PavementConditionFusioneer._compute_roadessy_pci(
            1.0, _segments(), _Vision([{"ghana_distress_type": "Potholes", "ghana_rating_1to9": 3}])
        )
        severe, _ = PavementConditionFusioneer._compute_roadessy_pci(
            9.9, _segments(), _Vision([{"ghana_distress_type": "Potholes", "ghana_rating_1to9": 3}])
        )
        assert mild == severe

    def test_worse_defect_rating_lowers_pci(self):
        slight, _ = PavementConditionFusioneer._compute_roadessy_pci(
            4.0, _segments(), _Vision([{"ghana_distress_type": "Potholes", "ghana_rating_1to9": 1}])
        )
        severe, _ = PavementConditionFusioneer._compute_roadessy_pci(
            4.0, _segments(), _Vision([{"ghana_distress_type": "Potholes", "ghana_rating_1to9": 9}])
        )
        assert severe < slight

    def test_distress_weight_is_applied_per_type(self):
        """A heavily weighted distress type contributes more than a light one.

        Potholes weight 1.0 and bleeding 0.4, so the same rating should cost
        more PCI as a pothole than as bleeding.
        """
        pothole, _ = PavementConditionFusioneer._compute_roadessy_pci(
            4.0, _segments(), _Vision([{"ghana_distress_type": "Potholes", "ghana_rating_1to9": 5}])
        )
        bleeding, _ = PavementConditionFusioneer._compute_roadessy_pci(
            4.0, _segments(), _Vision([{"ghana_distress_type": "Bleeding/Flushing", "ghana_rating_1to9": 5}])
        )
        assert pothole < bleeding

    def test_breakdown_keeps_per_type_detail(self):
        _, breakdown = PavementConditionFusioneer._compute_roadessy_pci(
            4.0,
            _segments(),
            _Vision(
                [
                    {"ghana_distress_type": "Potholes", "ghana_rating_1to9": 5},
                    {"ghana_distress_type": "Potholes", "ghana_rating_1to9": 3},
                ]
            ),
        )
        potholes = breakdown["roadessy_rating_details"]["Potholes"]
        # Both potholes collapse into one averaged rating for the type.
        assert potholes["avg_rating_1to9"] == 4.0
        assert potholes["weight"] == 1.0
        assert potholes["contribution"] == 4.0

    def test_unknown_distress_type_uses_a_default_weight(self):
        """An unrecognised distress type still scores rather than being dropped."""
        _, breakdown = PavementConditionFusioneer._compute_roadessy_pci(
            4.0,
            _segments(),
            _Vision([{"ghana_distress_type": "Something New", "ghana_rating_1to9": 4}]),
        )
        assert breakdown["roadessy_rating_details"]["Something New"]["weight"] == 0.5

    def test_pci_stays_within_bounds(self):
        detections = [
            {"ghana_distress_type": "Potholes", "ghana_rating_1to9": 9} for _ in range(20)
        ]
        pci, _ = PavementConditionFusioneer._compute_roadessy_pci(4.0, _segments(), _Vision(detections))
        assert 0.0 <= pci <= 100.0


class TestRoadessySegmentPci:
    def test_detections_are_bucketed_into_50m_segments(self):
        segments = PavementConditionFusioneer._compute_roadessy_segment_pci(
            _segments(count=3),
            [
                {"chainage_m": 10, "deduct_value": 5},
                {"chainage_m": 70, "deduct_value": 3},
                {"chainage_m": 120, "deduct_value": 9},
            ],
        )
        assert [s["segment_index"] for s in segments] == [0, 1, 2]
        assert [s["start_m"] for s in segments] == [0, 50, 100]
        assert [s["end_m"] for s in segments] == [50, 100, 150]

    def test_segment_pci_reflects_only_its_own_detections(self):
        segments = PavementConditionFusioneer._compute_roadessy_segment_pci(
            _segments(count=2),
            [
                {"chainage_m": 10, "deduct_value": 0},
                {"chainage_m": 70, "deduct_value": 20},
            ],
        )
        assert segments[0]["roadessy_pci"] == 100.0
        assert segments[1]["roadessy_pci"] < 100.0

    def test_detections_beyond_the_survey_are_reported_not_bucketed(self):
        """A detection past the surveyed length is counted, not given a row.

        The table is the inertial survey's: one row per 50 m it measured, which
        is also the distance each row's IRI was measured over. Growing the table
        to fit a detection would emit a row carrying a PCI with no IRI beside it.
        The detection is instead counted in the warning, so the deficit is
        visible rather than the road looking cleaner than the footage shows.

        Vision and vibration GPS cover the same drive but disagree on its length
        (0.06-9.11% across the KNUST Obuasi trips), so this happens for real.
        """
        segments = PavementConditionFusioneer._compute_roadessy_segment_pci(
            _segments(count=1),
            [{"chainage_m": 220, "deduct_value": 4}],
        )
        assert len(segments) == 1
        assert segments[0]["roadessy_pci"] == 100.0
        assert PavementConditionFusioneer._detections_beyond_survey(
            [{"chainage_m": 220, "deduct_value": 4}], 50.0
        ) == 1

    def test_detections_inside_the_survey_are_bucketed_normally(self):
        segments = PavementConditionFusioneer._compute_roadessy_segment_pci(
            _segments(count=5),
            [{"chainage_m": 10, "deduct_value": 4}, {"chainage_m": 220, "deduct_value": 4}],
        )
        assert len(segments) == 5
        assert segments[0]["roadessy_pci"] < 100.0
        assert segments[4]["roadessy_pci"] < 100.0

    def test_empty_inputs_produce_no_segments(self):
        assert PavementConditionFusioneer._compute_roadessy_segment_pci([], []) == []

    def test_every_segment_is_classified(self):
        for s in PavementConditionFusioneer._compute_roadessy_segment_pci(
            _segments(count=2), [{"chainage_m": 10, "deduct_value": 30}]
        ):
            assert s["roadessy_pci_class"] in {
                "Good", "Satisfactory", "Fair", "Poor", "Very Poor", "Serious", "Failed",
            }


def _video_quality(fps_flag="OK"):
    from app.services.vision import VideoQualityMetrics

    return VideoQualityMetrics(
        fps=30.0,
        resolution=(1920, 1080),
        motion_blur_score=0.1,
        illumination_score=0.8,
        gps_sync_quality=0.95,
        fps_flag=fps_flag,
        frames_analyzed=300,
        frames_skipped=0,
    )


class TestAstmPciIsMutedByDefault:
    """The D6433 PCI is switched off, not deleted.

    These run with no opt-in, so they pin the default behaviour: no PCI, no
    rating, and a stated reason, while the IRI keeps flowing.
    """

    def _qcm(self):
        from app.services.inertial import ASTMQCMResult

        return ASTMQCMResult(
            iri_m_km=4.2,
            total_distance_m=500.0,
            iri_segments=[],
            sample_rate_hz=100.0,
        )

    def _vision(self):
        from app.services.vision import ASTMVisionResult

        return ASTMVisionResult(
            detections=[
                {
                    "chainage_m": 10.0,
                    "defect_type": "pothole",
                    "deduct_value": 10.0,
                    "severity": "moderate",
                }
            ],
            video_quality=_video_quality(),
        )

    def test_no_pci_is_returned(self):
        pci, _ = PavementConditionFusioneer._compute_astm_pci(
            self._qcm(), self._vision()
        )
        assert pci is None

    def test_the_muted_state_is_explicit_not_an_empty_result(self):
        """A bare None reads as 'nothing to report'. The reader is told the
        number is withheld and why, so a missing PCI is not mistaken for a
        missing measurement."""
        _, breakdown = PavementConditionFusioneer._compute_astm_pci(
            self._qcm(), self._vision()
        )
        assert breakdown["enabled"] is False
        assert breakdown["reason"].strip()

    def test_the_iri_is_unaffected(self):
        """The curve defect is in the PCI, not the vibration measurement, so
        the IRI must survive the mute. Muting it too would lose a real number."""
        _, breakdown = PavementConditionFusioneer._compute_astm_pci(
            self._qcm(), self._vision()
        )
        assert breakdown["qcm_iri_m_km"] == 4.2

    def test_the_pipeline_reports_no_rating_and_no_segments(self):
        """The whole ASTM track is muted, not just its PCI.

        So the QCM IRI is withheld as well: the measurement was never taken for
        this build, and reporting the 4.2 m/km the fixture carries would put a
        figure in the report that no run can produce. None reads as "not
        reported", which is what a muted track actually means.
        """
        result = PavementConditionFusioneer.compute_astm_pipeline(
            astm_qcm=self._qcm(),
            astm_vision=self._vision(),
            run_id="mute-test",
        )
        assert result["astm_pci"] is None
        assert result["astm_pci_class"] is None
        assert result["astm_pci_enabled"] is False
        assert result["segment_pci_50m"] == []
        assert result["qcm_iri_m_km"] is None
        assert result["qcm_iri_segments_50m"] == []
        assert result["astm_track_disabled_reason"].strip()

    def test_the_roadessy_pipeline_is_untouched(self):
        """The two pipelines are independent; disabling one must not silence
        the other."""
        import types

        from app.services.vision import ASTMVisionResult

        result = PavementConditionFusioneer.compute_roadessy_pipeline(
            simple_result=types.SimpleNamespace(
                iri_m_km=3.1, iri_segments=[], total_distance_m=500.0
            ),
            roadessy_vision=ASTMVisionResult(
                detections=[], video_quality=_video_quality()
            ),
            run_id="mute-test",
        )
        assert result["roadessy_pci"] is not None


class TestComputeAstmPci:
    def test_returns_pci_with_ir_reported_separately(self, astm_pci_on):
        """ASTM D6433 keeps IRI out of the PCI score.

        The two are combined on the reporting side, not inside the score, so
        this is the structural assertion that they stay separate.
        """
        from app.services.astm import ASTMD6433PCI
        from app.services.inertial import ASTMQCMResult
        from app.services.vision import ASTMVisionResult

        qcm = ASTMQCMResult(
            iri_m_km=4.2,
            total_distance_m=500.0,
            iri_segments=[],
            sample_rate_hz=100.0,
        )
        vision = ASTMVisionResult(
            detections=[
                {
                    "chainage_m": 10.0,
                    "defect_type": "pothole",
                    "deduct_value": 10.0,
                    "severity": "moderate",
                }
            ],
            video_quality=_video_quality(),
        )
        pci, breakdown = PavementConditionFusioneer._compute_astm_pci(qcm, vision)
        assert 0.0 <= pci <= 100.0
        assert breakdown["qcm_iri_m_km"] == 4.2
        assert breakdown["astm_pci"] == pytest.approx(pci, abs=0.01)
        assert ASTMD6433PCI._classify_pci(pci) == breakdown["astm_pci_class"]

    def test_no_defects_is_a_perfect_score(self, astm_pci_on):
        from app.services.inertial import ASTMQCMResult
        from app.services.vision import ASTMVisionResult

        qcm = ASTMQCMResult(iri_m_km=0.0, total_distance_m=500.0, iri_segments=[], sample_rate_hz=100.0)
        vision = ASTMVisionResult(detections=[], video_quality=_video_quality())
        pci, _ = PavementConditionFusioneer._compute_astm_pci(qcm, vision)
        assert pci == 100.0

    def test_absent_video_quality_is_reported_not_crashed(self, astm_pci_on):
        """Video quality is optional, so a missing one must not raise.

        The FPS flag is surfaced to the reader as part of the quality story, so
        its absence is reported as UNKNOWN rather than failing the whole fusion.
        """
        from app.services.inertial import ASTMQCMResult
        from app.services.vision import ASTMVisionResult

        qcm = ASTMQCMResult(iri_m_km=3.0, total_distance_m=500.0, iri_segments=[], sample_rate_hz=100.0)
        vision = ASTMVisionResult(detections=[], video_quality=None)
        _, breakdown = PavementConditionFusioneer._compute_astm_pci(qcm, vision)
        assert breakdown["video_quality_fps_flag"] == "UNKNOWN"
