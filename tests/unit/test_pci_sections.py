"""The 50 m section logic: the spine of both PCI pipelines.

Sections are fixed at 50 m by ASTM D6433, and everything downstream — the
density a deduct value is looked up from, the per-segment CSV, the per-segment
database rows — depends on a detection landing in the right section. That makes
the section boundaries worth pinning: they are arithmetic that produces plausible
wrong numbers when it drifts, rather than errors anyone would notice.

The bug these tests exist for: sections were built from the survey length alone,
while each detection's chainage comes from the vision GPS and the survey length
is integrated from the vibration GPS. When the vision track ran past the last
section, those detections were discarded with no trace, and a road with defects
in its last 350 m was reported as a clean one.
"""
import pytest

from app.services.astm.pci import (
    ASTMD6433PCI,
    Distress,
    DistressType,
    SectionMetrics,
    SeverityLevel,
)
from app.services.fusion.service import PavementConditionFusioneer

#: The 50 m reporting unit from ASTM D6433. Sections, the per-segment CSV and the
#: per-segment database rows are all built on this number.
SECTION_LENGTH_M = ASTMD6433PCI.SECTION_LENGTH_M


def pothole_at(chainage, width=0.5, area=0.25, cls_id=4):
    return {
        "yolo_class_id": cls_id,
        "chainage_m": float(chainage),
        "width": width,
        "area": area,
        "deduct_value": 8.0,
    }


class FakeIRISegment:
    def __init__(self, index):
        self.start_distance_m = index * SECTION_LENGTH_M
        self.end_distance_m = (index + 1) * SECTION_LENGTH_M
        self.iri_m_km = 3.0


class TestDetectionsBeyondSurvey:
    """Counting the detections that have no row, so the deficit is visible.

    The vision GPS and the vibration GPS describe the same drive but disagree on
    its length by 0.06-9.11% across the KNUST Obuasi trips, so a detection near
    the end of the road can fall outside the surveyed length. The tables follow
    the survey; these detections have to be announced rather than quietly
    missing, or the report reads as a road with fewer defects than the video
    showed.
    """

    def test_counts_detections_past_the_end(self):
        detections = [pothole_at(10), pothole_at(260), pothole_at(410)]
        assert PavementConditionFusioneer._detections_beyond_survey(detections, 250.0) == 2

    def test_a_detection_exactly_at_the_end_counts_as_beyond(self):
        assert PavementConditionFusioneer._detections_beyond_survey([pothole_at(250.0)], 250.0) == 1

    def test_a_detection_just_inside_does_not_count(self):
        assert PavementConditionFusioneer._detections_beyond_survey([pothole_at(249.9)], 250.0) == 0

    def test_nothing_beyond_an_empty_survey_end(self):
        assert PavementConditionFusioneer._detections_beyond_survey([pothole_at(10)], 0.0) == 0

    def test_nothing_beyond_with_no_detections(self):
        assert PavementConditionFusioneer._detections_beyond_survey([], 250.0) == 0

    def test_a_detection_without_chainage_does_not_count(self):
        assert PavementConditionFusioneer._detections_beyond_survey([{}], 250.0) == 0


class TestSectionGeometry:
    def test_section_length_is_fifty_metres(self):
        """ASTM D6433 reporting unit. Everything else divides by it."""
        assert SECTION_LENGTH_M == 50.0

    def test_section_count_follows_the_survey_length(self):
        result = ASTMD6433PCI.compute_pci_from_detections([], total_length_m=500.0)
        assert len(result["sections"]) == 10

    def test_a_partial_final_section_is_still_a_section(self):
        result = ASTMD6433PCI.compute_pci_from_detections([], total_length_m=520.0)
        assert len(result["sections"]) == 11
        assert result["sections"][-1]["end_m"] == 520.0

    def test_sections_tile_the_length_without_gaps_or_overlap(self):
        sections = ASTMD6433PCI.compute_pci_from_detections([], total_length_m=1000.0)["sections"]
        for previous, current in zip(sections, sections[1:]):
            assert current["start_m"] == previous["end_m"]

    def test_a_clean_survey_scores_one_hundred(self):
        result = ASTMD6433PCI.compute_pci_from_detections([], total_length_m=2000.0)
        assert result["overall_pci"] == 100.0
        assert result["rating"] == "Good"


class TestSectionAssignment:
    @pytest.mark.parametrize(
        "chainage,expected_section",
        [
            (0.0, 0),     # first metre belongs to the first section
            (49.999, 0),  # still the first section
            (50.0, 1),     # boundaries are inclusive below
            (99.9, 1),
            (100.0, 2),
            (495.0, 9),
        ],
    )
    def test_a_detection_lands_in_its_own_section(self, chainage, expected_section):
        result = ASTMD6433PCI.compute_pci_from_detections(
            [pothole_at(chainage)], total_length_m=500.0
        )
        with_defects = [s["section_index"] for s in result["sections"] if s["breakdown"]]
        assert with_defects == [expected_section]

    def test_two_detections_in_different_sections_score_both(self):
        result = ASTMD6433PCI.compute_pci_from_detections(
            [pothole_at(10), pothole_at(210)], total_length_m=300.0
        )
        with_defects = sorted(s["section_index"] for s in result["sections"] if s["breakdown"])
        assert with_defects == [0, 4]

    def test_detections_past_the_survey_are_counted_not_invented_into_sections(self):
        """The survey's measured length defines the table, not the vision track.

        600 m of vision chainage against a 250 m survey must produce 5 sections,
        not 12: the survey distance is also the distance each row's IRI was
        measured over, so a row past it would carry a PCI with no IRI beside it.
        The detections are reported in `detections_outside_survey` rather than
        dropped, so the shortfall is visible rather than looking like a clean
        road.
        """
        detections = [pothole_at(seg * 50 + 10) for seg in range(12)]
        result = ASTMD6433PCI.compute_pci_from_detections(detections, total_length_m=250.0)
        assert len(result["sections"]) == 5
        assert result["survey_end_m"] == 250.0
        assert result["detections_outside_survey"] == 7

    def test_detections_inside_the_survey_are_not_counted_as_outside(self):
        detections = [pothole_at(10), pothole_at(240)]
        result = ASTMD6433PCI.compute_pci_from_detections(detections, total_length_m=250.0)
        assert result["detections_outside_survey"] == 0

    def test_a_clean_survey_reports_nothing_outside(self):
        result = ASTMD6433PCI.compute_pci_from_detections([], total_length_m=250.0)
        assert result["detections_outside_survey"] == 0

    def test_a_detection_exactly_on_the_survey_end_is_outside(self):
        """The last section is half-open, so its end belongs to nothing."""
        result = ASTMD6433PCI.compute_pci_from_detections(
            [pothole_at(250.0)], total_length_m=250.0
        )
        assert result["detections_outside_survey"] == 1

    def test_a_missing_chainage_does_not_raise(self):
        result = ASTMD6433PCI.compute_pci_from_detections(
            [{"yolo_class_id": 4, "width": 0.5, "area": 0.25}], total_length_m=200.0
        )
        assert len(result["sections"]) == 4


class TestOverallAveraging:
    def test_empty_sections_count_toward_the_mean(self):
        """A long clean run must not inflate the score by adding empty sections.

        ASTM averages the section scores, and an undistressed section is a 100.
        If empty sections were skipped, extending the survey with clean road
        would raise the reported PCI.
        """
        detections = [pothole_at(10) for _ in range(4)]
        short = ASTMD6433PCI.compute_pci_from_detections(detections, total_length_m=200.0)
        long = ASTMD6433PCI.compute_pci_from_detections(detections, total_length_m=1000.0)
        assert short["overall_pci"] < long["overall_pci"]

    def test_overall_is_the_mean_of_section_scores(self):
        result = ASTMD6433PCI.compute_pci_from_detections(
            [pothole_at(10), pothole_at(210)], total_length_m=300.0
        )
        expected = sum(s["pci"] for s in result["sections"]) / len(result["sections"])
        assert result["overall_pci"] == pytest.approx(expected, abs=0.01)

    def test_score_is_clamped_into_range(self):
        heavy = [pothole_at(seg * 50 + 5, width=3.0, area=9.0) for seg in range(20)]
        result = ASTMD6433PCI.compute_pci_from_detections(heavy, total_length_m=1000.0)
        assert 0.0 <= result["overall_pci"] <= 100.0


class TestClassMapping:
    def test_every_model_class_maps_to_a_distress_type(self):
        """A class with no mapping falls back to LONGITUDINAL, mis-scoring it.

        This is the guard against a checkpoint whose class order differs from the
        one the service rates against.
        """
        for class_id in range(9):
            assert class_id in ASTMD6433PCI.YOLO_TO_AASHTO

    def test_pothole_class_maps_to_potholes(self):
        assert ASTMD6433PCI.YOLO_TO_AASHTO[4] == DistressType.POTHOLES


class TestSegmentPciOutputs:
    def test_astm_table_follows_the_iri_segment_count(self):
        """The table is the IRI's: one row per 50 m the survey measured."""
        detections = [pothole_at(seg * 50 + 10) for seg in range(12)]
        segments = PavementConditionFusioneer._compute_astm_segment_pci(
            [FakeIRISegment(i) for i in range(5)], detections
        )
        assert len(segments) == 5
        assert sum(1 for s in segments if s["astm_pci"] < 100.0) == 5

    def test_roadessy_table_follows_the_iri_segment_count(self):
        detections = [pothole_at(seg * 50 + 10) for seg in range(12)]
        segments = PavementConditionFusioneer._compute_roadessy_segment_pci(
            [FakeIRISegment(i) for i in range(5)], detections
        )
        assert len(segments) == 5
        assert sum(1 for s in segments if s["roadessy_pci"] < 100.0) == 5

    def test_both_tables_have_the_same_number_of_rows(self):
        """The two pipelines must not disagree on the shape of the survey.

        The Roadessy path used to widen to the furthest detection while the ASTM
        path did not, so the same survey could produce 5 rows in one pipeline
        and 12 in the other.
        """
        detections = [pothole_at(seg * 50 + 10) for seg in range(12)]
        iri = [FakeIRISegment(i) for i in range(5)]
        astm = PavementConditionFusioneer._compute_astm_segment_pci(iri, detections)
        roadessy = PavementConditionFusioneer._compute_roadessy_segment_pci(iri, detections)
        assert len(astm) == len(roadessy) == 5

    def test_a_late_segment_scores_its_own_detection(self):
        """Every segment must score, not just the first.

        The per-segment section call compares detection chainage against a 0-50 m
        section, so chainages have to be rebased onto the segment first. Without
        that, a detection 2 km along the road looked like it was beyond the
        survey, was placed in no section, and the segment reported a clean 100:
        59 of 60 segments did exactly that while each held a pothole.
        """
        segments = [FakeIRISegment(i) for i in range(60)]
        detections = [pothole_at(i * 50 + 10) for i in range(60)]
        table = PavementConditionFusioneer._compute_astm_segment_pci(segments, detections)
        assert len(table) == 60
        scored = [s["astm_pci"] for s in table]
        assert sum(1 for value in scored if value < 100.0) == 60, (
            "a segment holding a pothole reported a clean 100"
        )
        # Every segment is identical here, so one value stands for all of them.
        assert len(set(scored)) == 1

    def test_both_tables_have_the_same_number_of_rows_without_iri(self):
        detections = [pothole_at(10)]
        assert PavementConditionFusioneer._compute_astm_segment_pci([], detections) == []
        assert PavementConditionFusioneer._compute_roadessy_segment_pci([], detections) == []

    def test_segments_are_indexed_from_zero_and_contiguous(self):
        detections = [pothole_at(10), pothole_at(410)]
        segments = PavementConditionFusioneer._compute_astm_segment_pci(
            [FakeIRISegment(i) for i in range(9)], detections
        )
        assert [s["segment_index"] for s in segments] == list(range(9))
        for previous, current in zip(segments, segments[1:]):
            assert current["start_m"] == previous["end_m"]

    def test_iri_is_paired_with_its_own_segment(self):
        segments = PavementConditionFusioneer._compute_astm_segment_pci(
            [FakeIRISegment(i) for i in range(3)], [pothole_at(60)]
        )
        assert [s["astm_iri_m_km"] for s in segments] == [3.0, 3.0, 3.0]

    def test_no_detections_still_reports_full_coverage(self):
        segments = PavementConditionFusioneer._compute_astm_segment_pci(
            [FakeIRISegment(i) for i in range(4)], []
        )
        assert len(segments) == 4
        assert all(s["astm_pci"] == 100.0 for s in segments)


class TestSeverityAndDensity:
    def test_crack_severity_steps_with_width(self):
        """ASTM D 6433-99 X1.14.2: under 10 mm low, 10-75 mm medium, 75 mm high."""
        assert ASTMD6433PCI.get_severity_from_width(DistressType.LONGITUDINAL, 2.0) == SeverityLevel.LOW
        assert ASTMD6433PCI.get_severity_from_width(DistressType.LONGITUDINAL, 10.0) == SeverityLevel.MEDIUM
        assert ASTMD6433PCI.get_severity_from_width(DistressType.LONGITUDINAL, 75.0) == SeverityLevel.HIGH

    def test_density_rises_with_severe_distress(self):
        def section_with(width_mm):
            return SectionMetrics(
                section_index=0,
                start_m=0.0,
                end_m=50.0,
                area_m2=162.5,
                distresses=[
                    Distress(
                        type=DistressType.POTHOLES,
                        severity=SeverityLevel.LOW,
                        length_m=1.0,
                        area_m2=0.25,
                        width_mm=width_mm,
                    )
                ],
            )

        mild = ASTMD6433PCI.compute_section_pci(section_with(100.0))["pci"]
        severe = ASTMD6433PCI.compute_section_pci(section_with(900.0))["pci"]
        assert severe < mild

    def test_deduct_values_are_non_negative(self):
        """Swept across every curve the code prices, at three densities.

        Driven off `CURVE_KEYS` rather than a table of coefficients that has since
        been deleted, so it covers whatever the dataset actually holds.
        """
        from app.services.astm import curves

        level_for = {"L": SeverityLevel.LOW, "M": SeverityLevel.MEDIUM,
                     "H": SeverityLevel.HIGH}
        for distress_type, key in ASTMD6433PCI.CURVE_KEYS.items():
            for severity in ("L", "M", "H"):
                curve = curves.repository().get(key, severity)
                for density in (curve.low_density, 1.0, curve.high_density):
                    assert curve.interpolate(density) >= 0.0
                    assert ASTMD6433PCI.get_dv(
                        distress_type, level_for[severity], density
                    ) >= 0.0
