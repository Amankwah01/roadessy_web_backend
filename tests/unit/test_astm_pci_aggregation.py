"""How a 50 m section's deduct value is accumulated, and what it costs.

A section's density for a distress type is the total extent of that type divided
by the section's area, and the deduct curve is read once from that density. The
obvious-looking alternative — reading the curve once per detection and adding
the results — is not the same thing at all, because the curves have an
intercept, and the intercept is most of the value where real densities sit.

These tests pin the aggregation, and the severity handling inside it, to the
standard's method rather than to whichever of the two happens to be in the code.
"""

import pytest

from app.services.astm.pci import (
    ASTMD6433PCI,
    Distress,
    DistressType,
    SectionMetrics,
    SeverityLevel,
)

SECTION_AREA_M2 = 50.0 * 3.25


@pytest.fixture
def astm_pci_on(monkeypatch):
    """Opt in to the muted D6433 PCI for tests that go through the pipeline.

    `ASTMD6433PCI` is used directly throughout this file and needs no opt-in;
    only the calls that go via `PavementConditionFusioneer` do, because the mute
    sits at that boundary.
    """
    from app.services.fusion import service as fusion_service

    monkeypatch.setattr(fusion_service, "ASTM_PCI_ENABLED", True)
    return True


def _distress(dtype, *, length_m=0.0, area_m2=0.0, width_mm=0.0, depth_mm=None):
    return Distress(
        type=dtype,
        severity=SeverityLevel.LOW,
        length_m=length_m,
        area_m2=area_m2,
        width_mm=width_mm,
        depth_mm=depth_mm,
        chainage_start_m=0.0,
        chainage_end_m=length_m,
    )


def _section(distresses):
    return ASTMD6433PCI.compute_section_pci(
        SectionMetrics(0, 0.0, 50.0, SECTION_AREA_M2, list(distresses))
    )


def _potholes(count, width_mm=0.0, depth_mm=None):
    # The caller carries a pothole count in `length_m`; each becomes one unit.
    return [_distress(DistressType.POTHOLES, length_m=1.0, width_mm=width_mm,
                      depth_mm=depth_mm)
            for _ in range(count)]


class TestDensityIsOfTheTotalNotOfEachDetection:
    def test_twenty_potholes_cost_what_the_curve_says_they_cost(self):
        """The bug this replaces: the curve's intercept, charged per pothole.

        Per detection, twenty potholes cost a TDV of 10.0 and took the section
        from 98.0 to 89.1. That was a rating change caused by arithmetic, not by
        the road. Read once from the total, twenty potholes at 6.15% density land
        on the X3.19 low curve between the 1% and 10% anchors.
        """
        result = _section(_potholes(20))
        tdv = result["breakdown"]["potholes"]["tdv"]
        density = 20 * 0.5 / SECTION_AREA_M2 * 100.0
        expected = ASTMD6433PCI.get_dv(DistressType.POTHOLES, SeverityLevel.LOW, density)
        assert tdv == pytest.approx(expected, abs=0.01)
        # The value is read from between two real anchors, not from an intercept:
        # 6.15% is above 1% and below 10%, so it must land between 10 and 50.
        assert 10.0 < tdv < 50.0
        # 9.5.1: with no more than one deduct value above 2.0, the total is the
        # maximum corrected deduct value outright.
        assert result["pci"] == pytest.approx(100.0 - tdv, abs=0.05)

    def test_the_pothole_count_actually_moves_the_deduct_value(self):
        """A count is a quantity, so more holes must cost more.

        This asserts the relationship rather than a pasted number, because the
        number is the curve's to set. The placeholder fits were nearly flat here:
        the curve was all intercept at these densities, so twenty potholes cost the
        same as one and a count of holes barely registered. On the digitized
        X3.19 curve the value climbs steadily with the count.
        """
        one = _section(_potholes(1))["breakdown"]["potholes"]["tdv"]
        five = _section(_potholes(5))["breakdown"]["potholes"]["tdv"]
        twenty = _section(_potholes(20))["breakdown"]["potholes"]["tdv"]
        assert one < five < twenty
        # And the total area is what drives it, not how the holes are grouped.
        assert twenty == pytest.approx(
            _section(_potholes(10) + _potholes(10))["breakdown"]["potholes"]["tdv"],
            abs=0.01,
        )

    def test_area_distress_does_accumulate(self):
        """Unlike pothole counts, area really does drive the density.

        Ten square metres of ravelling in a 162.5 m2 section is 6.2% density,
        which is up the curve where the quadratic term matters, so splitting it
        into one detection or ten must not change the answer.
        """
        as_one = _section([_distress(DistressType.RAVELLING, area_m2=10.0)])
        as_ten = _section([_distress(DistressType.RAVELLING, area_m2=1.0)
                           for _ in range(10)])
        assert as_one["pci"] == pytest.approx(as_ten["pci"], abs=0.01)

    def test_splitting_a_linear_crack_does_not_change_the_pci(self):
        cracks = [_distress(DistressType.LONGITUDINAL, length_m=10.0, width_mm=3.0)]
        pieces = [_distress(DistressType.LONGITUDINAL, length_m=2.5, width_mm=3.0)
                  for _ in range(4)]
        assert _section(cracks)["pci"] == pytest.approx(_section(pieces)["pci"], abs=0.01)

    def test_severity_levels_are_charged_separately(self):
        """ASTM takes one deduct value per type *and* severity level.

        A section with two low and one high pothole is not the same as three
        low ones: the high one is read off the high curve, which has a much
        larger intercept, and the two low ones are pooled and read once.

        Table X1.1 needs diameter *and* depth, so both are given: a 100 mm
        pothole 20 mm deep is low, a 600 mm one 60 mm deep is high.
        """
        mixed = _section([
            _distress(DistressType.POTHOLES, length_m=1.0, width_mm=100.0, depth_mm=20.0),
            _distress(DistressType.POTHOLES, length_m=1.0, width_mm=100.0, depth_mm=20.0),
            _distress(DistressType.POTHOLES, length_m=1.0, width_mm=600.0, depth_mm=60.0),
        ])
        all_low = _section(_potholes(3, width_mm=100.0, depth_mm=20.0))
        assert mixed["pci"] < all_low["pci"] - 3.0


    def test_measured_depth_reaches_the_severity_bands(self):
        """Table X1.1 rates potholes on diameter *and* depth together.

        Vision measures the depth and reports it, but it was not reaching the
        severity bands, so every pothole was rated on its diameter alone and
        `_pothole_severity` took the depth-less branch. On the digitized X3.19
        curve that caps a pothole at the low band, and high is worth several times
        low, so a genuinely deep hole was scored as a shallow one.
        """
        def pci_for(depth_mm):
            return ASTMD6433PCI.compute_pci_from_detections(
                [{"chainage_m": 10.0, "yolo_class_id": 4, "area": 0.05,
                  "width": 0.3, "pothole_depth_mm": depth_mm / 1000.0}],
                total_length_m=50.0,
            )["sections"][0]["pci"]

        # A 300 mm hole is low at 20 mm deep and high at 60 mm, on the same
        # diameter, so depth alone has to move the score.
        assert pci_for(60.0) < pci_for(20.0) - 5.0

    def test_an_unmeasured_depth_is_not_treated_as_zero_depth(self):
        """Absent is not shallow, and it is not deep either.

        `_pothole_severity` treats a missing depth as low. That is the safe
        default, but it must be reached by the depth being absent rather than by a
        fabricated zero, and a detection reporting a depth of exactly zero is the
        detector saying it did not measure one.
        """
        assert ASTMD6433PCI._detection_depth_mm({}, DistressType.POTHOLES) is None
        assert ASTMD6433PCI._detection_depth_mm(
            {"pothole_depth_mm": 0.0}, DistressType.POTHOLES) is None
        assert ASTMD6433PCI._detection_depth_mm(
            {"depth_mm": 0.0}, DistressType.RUTTING) is None
        # Vision reports metres for potholes and millimetres for rutting.
        assert ASTMD6433PCI._detection_depth_mm(
            {"pothole_depth_mm": 0.03}, DistressType.POTHOLES) == pytest.approx(30.0)
        assert ASTMD6433PCI._detection_depth_mm(
            {"depth_mm": 18.0}, DistressType.RUTTING) == pytest.approx(18.0)


class TestAmountPerDistressType:
    def test_linear_distress_is_measured_by_length_then_made_an_area(self):
        """X1.14.3 measures cracking in linear metres; 9.2 divides by area.

        The two have to be reconciled before the division, and the reconciliation
        is length times the lane width. Dividing metres by square metres directly
        gives metres inverse, not a percentage, which is how eight ordinary cracks
        in a 50 m section came out at 246% density -- off the end of every
        published curve, so the lookup raised rather than answering.
        """
        d = _distress(DistressType.LONGITUDINAL, length_m=4.0, area_m2=99.0)
        # 4 m of crack across a 3.25 m lane is 13 m2 of pavement, whatever the
        # bounding box area happened to be.
        assert ASTMD6433PCI._distress_amount(d, 3.25) == pytest.approx(13.0)
        assert ASTMD6433PCI._distress_amount(d, 3.0) == pytest.approx(12.0)

    def test_a_length_distress_density_is_a_percentage_not_a_rate(self):
        """The invariant that matters: density stays within 0-100%.

        Cracking a whole 50 m section lengthwise is 100% of the pavement. Anything
        above that means the quantity and the denominator are in different units,
        which is the bug this guards.
        """
        full = _distress(DistressType.LONGITUDINAL, length_m=50.0)
        assert ASTMD6433PCI.compute_density(full, SECTION_AREA_M2, 3.25) == pytest.approx(100.0)
        assert ASTMD6433PCI.compute_density(
            _distress(DistressType.LONGITUDINAL, length_m=10.0),
            SECTION_AREA_M2, 3.25) == pytest.approx(20.0)

    def test_a_density_past_a_hundred_percent_is_capped_and_reported(self):
        """100% is the most of a section that can be distressed, so it is the cap.

        A detection set can still report more -- two boxes over one crack, or a
        length derived from a bounding area reaching past the section end. The
        published curves stop at 100% because there is nothing past it to read, so
        raising past the top would throw away the section's PCI including the
        distresses whose densities were ordinary. Capping floors the PCI and says
        so instead.
        """
        # 400 m of crack in a 50 m section is impossible, and the old arithmetic
        # turned it into a 246% density that ran off the end of Fig. X3.15.
        impossible = [_distress(DistressType.LONGITUDINAL, length_m=50.0)
                      for _ in range(8)]
        result = _section(impossible)
        assert result["density_capped_at_100pct"] == ["longitudinal"]
        assert result["pci"] >= 0.0
        for source in result["deduct_value_sources"]:
            assert source["density_percent"] <= 100.0

    def test_an_ordinary_section_reports_no_cap(self):
        """The cap is for the impossible case only; silence is the signal."""
        ordinary = [_distress(DistressType.LONGITUDINAL, length_m=10.0, width_mm=3.0)]
        assert _section(ordinary)["density_capped_at_100pct"] == []

    def test_area_distress_is_measured_by_area(self):
        d = _distress(DistressType.RAVELLING, length_m=7.0, area_m2=2.0)
        assert ASTMD6433PCI._distress_amount(d) == 2.0

    def test_potholes_are_measured_by_count_at_the_standard_area(self):
        """X1.17.1.2: one pothole counts as 0.5 m2 (5.5 ft2).

        The standard states the factor itself, converting an oversized pothole to
        "the equivalent number of holes" by dividing its area by 0.5 m2. It is
        also what puts a single pothole on the curve: at 0.5 m2 a lone hole in a
        162.5 m2 section is 0.31% density, inside the 0.1% the curve starts at,
        where at the 0.1 m2 this used to carry it fell below the floor and priced
        at zero.
        """
        assert ASTMD6433PCI._distress_amount(
            _distress(DistressType.POTHOLES, length_m=3.0)
        ) == pytest.approx(1.5)

    def test_one_pothole_is_expensive_enough_to_reach_the_curve(self):
        """The regression guard on that factor, in the units that matter.

        A lone pothole has to earn something. If the per-hole area is ever reduced
        again, the density drops under the curve's lower anchor and potholes
        silently stop costing anything, which reads as a better road rather than
        as a broken lookup.
        """
        pothole = _section(_potholes(1))
        density = (0.5 / SECTION_AREA_M2) * 100.0
        assert density > 0.1, "density fell below the X3.19 curve's lowest anchor"
        assert pothole["pci"] < 100.0

    def test_depressions_are_measured_by_area(self):
        """X1.10.2 measures depressions in square metres, not as a count.

        Potholes are the only counted distress here (X1.17.2). Treating
        depressions as a count gave them a density unrelated to the area they
        actually covered.
        """
        d = _distress(DistressType.DEPRESSION, length_m=2.0, area_m2=3.0)
        assert ASTMD6433PCI._distress_amount(d) == pytest.approx(3.0)

    def test_only_potholes_are_measured_by_count(self):
        assert ASTMD6433PCI.MEASUREMENT[DistressType.POTHOLES] == "count"
        assert ASTMD6433PCI.MEASUREMENT[DistressType.LONGITUDINAL] == "length"
        for dtype in (DistressType.ALLIGATOR, DistressType.RUTTING,
                      DistressType.RAVELLING, DistressType.PATCHING,
                      DistressType.SHOVING, DistressType.BLEEDING,
                      DistressType.CORRUGATION, DistressType.DEPRESSION):
            assert ASTMD6433PCI.MEASUREMENT[dtype] == "area"

    def test_a_zero_area_section_has_no_density(self):
        d = _distress(DistressType.RAVELLING, area_m2=5.0)
        assert ASTMD6433PCI.compute_density(d, 0.0) == 0.0


class TestSectionAndOverallScoring:
    def test_a_clean_section_scores_100(self):
        result = _section([])
        assert result["pci"] == 100.0
        assert result["rating"] == "Good"

    def test_pci_is_the_complement_of_the_corrected_deduct(self):
        result = _section(_potholes(5))
        assert result["pci"] == pytest.approx(100.0 - result["total_cdv"], abs=0.01)

    def test_the_overall_score_averages_every_section_equally(self):
        """A clean section still counts.

        Leaving empty sections out would let a road's score depend on how many
        50 m sections the two GPS loggers happened to produce, which is not a
        property of the road.
        """
        detections = [
            {"chainage_m": 10.0, "yolo_class_id": 4, "area": 0.1, "width": 0.1}
        ]
        result = ASTMD6433PCI.compute_pci_from_detections(detections, total_length_m=500.0)
        assert len(result["sections"]) == 10
        worst = min(s["pci"] for s in result["sections"])
        # One dirty section of ten: the overall score is far better than that
        # section alone because the nine clean ones are counted, and it is not
        # 100 because the dirty one is counted too.
        assert result["overall_pci"] > worst
        assert result["overall_pci"] < 100.0
        assert result["overall_pci"] == pytest.approx(
            (sum(s["pci"] for s in result["sections"]) / 10), abs=0.01
        )

    def test_a_defect_is_scored_in_the_section_containing_it(self):
        """The regression this guards: a defect at 120 m is not in section 0-50.

        Comparing every detection against a fixed 0-50 m window put every defect
        on a long survey into the first section, which is both wrong and the
        reason the two 50 m tables disagreed.
        """
        detections = [
            {"chainage_m": 120.0, "yolo_class_id": 4, "area": 0.1, "width": 0.1}
        ]
        result = ASTMD6433PCI.compute_pci_from_detections(detections, total_length_m=500.0)
        dirty = [s["section_index"] for s in result["sections"] if s["pci"] < 100.0]
        assert dirty == [2]

    def test_detections_past_the_survey_are_counted_not_dropped(self):
        detections = [
            {"chainage_m": 10.0, "yolo_class_id": 4, "area": 0.1, "width": 0.1},
            {"chainage_m": 4800.0, "yolo_class_id": 4, "area": 0.1, "width": 0.1},
        ]
        result = ASTMD6433PCI.compute_pci_from_detections(detections, total_length_m=1000.0)
        assert result["detections_outside_survey"] == 1
        assert len(result["sections"]) == 20


class TestSeverityBandsFollowTheStandard:
    """Bands per ASTM D 6433-99, quoted from the sections that set them.

    The detector hands over a width measured on the pavement plane through the
    mounting homography, so for a crack that width is the crack's width and the
    bands mean what the standard says they mean.
    """

    def test_crack_width_bands_are_ten_and_seventy_five_millimetres(self):
        """X1.14.2: low under 10 mm, medium 10 to under 75 mm, high at 75 mm."""
        # X1.14.2.1: "nonfilled crack width is less than 10 mm" is low.
        assert ASTMD6433PCI.get_severity_from_width(
            DistressType.LONGITUDINAL, 9.9
        ) is SeverityLevel.LOW
        # X1.14.2.2: "greater than or equal to 10 mm and less than 75 mm".
        assert ASTMD6433PCI.get_severity_from_width(
            DistressType.LONGITUDINAL, 10.0
        ) is SeverityLevel.MEDIUM
        assert ASTMD6433PCI.get_severity_from_width(
            DistressType.LONGITUDINAL, 74.9
        ) is SeverityLevel.MEDIUM
        # X1.14.2.3: high reaches down to 75 mm, not to 20 mm.
        assert ASTMD6433PCI.get_severity_from_width(
            DistressType.LONGITUDINAL, 75.0
        ) is SeverityLevel.HIGH

    def test_a_twenty_millimetre_crack_is_medium_not_high(self):
        """The old bands called this high, which is where ordinary cracks landed.

        Nothing on a 20 mm crack is wrong; the band it was being put in was.
        """
        assert ASTMD6433PCI.get_severity_from_width(
            DistressType.LONGITUDINAL, 20.0
        ) is SeverityLevel.MEDIUM

    def test_transverse_cracking_uses_the_same_width_bands(self):
        for width, expected in ((5.0, SeverityLevel.LOW),
                                (30.0, SeverityLevel.MEDIUM),
                                (90.0, SeverityLevel.HIGH)):
            assert ASTMD6433PCI.get_severity_from_width(
                DistressType.TRANSVERSE, width
            ) is expected

    def test_pothole_bands_need_diameter_and_depth_together(self):
        """Table X1.1, read across each diameter band.

        ===========  ==========  ===========  ==========
        Diameter mm  13-25 mm    >25-50 mm    >50 mm
        ===========  ==========  ===========  ==========
        100-200      L           L            M
        200-450      L           M            M
        450-750      L           M            H
        ===========  ==========  ===========  ==========

        Only the widest band reaches high, and the narrowest is still medium
        once the hole is deeper than 50 mm.
        """
        # 100-200 mm
        assert ASTMD6433PCI.get_severity_from_width(
            DistressType.POTHOLES, 150.0, 20.0) is SeverityLevel.LOW
        assert ASTMD6433PCI.get_severity_from_width(
            DistressType.POTHOLES, 150.0, 40.0) is SeverityLevel.LOW
        assert ASTMD6433PCI.get_severity_from_width(
            DistressType.POTHOLES, 150.0, 60.0) is SeverityLevel.MEDIUM
        # 200-450 mm
        assert ASTMD6433PCI.get_severity_from_width(
            DistressType.POTHOLES, 400.0, 20.0) is SeverityLevel.LOW
        assert ASTMD6433PCI.get_severity_from_width(
            DistressType.POTHOLES, 400.0, 40.0) is SeverityLevel.MEDIUM
        assert ASTMD6433PCI.get_severity_from_width(
            DistressType.POTHOLES, 400.0, 60.0) is SeverityLevel.MEDIUM
        # 450-750 mm
        assert ASTMD6433PCI.get_severity_from_width(
            DistressType.POTHOLES, 600.0, 20.0) is SeverityLevel.LOW
        assert ASTMD6433PCI.get_severity_from_width(
            DistressType.POTHOLES, 600.0, 40.0) is SeverityLevel.MEDIUM
        assert ASTMD6433PCI.get_severity_from_width(
            DistressType.POTHOLES, 600.0, 60.0) is SeverityLevel.HIGH

    def test_a_pothole_over_750_mm_rates_on_depth_alone(self):
        """X1.17.1.2 drops diameter above 750 mm: shallow is medium, not high."""
        assert ASTMD6433PCI.get_severity_from_width(
            DistressType.POTHOLES, 900.0, 25.0) is SeverityLevel.MEDIUM
        assert ASTMD6433PCI.get_severity_from_width(
            DistressType.POTHOLES, 900.0, 30.0) is SeverityLevel.HIGH

    def test_depressions_are_rated_on_depth_not_width(self):
        """X1.10.1: 13-25 mm low, 25-50 mm medium, over 50 mm high."""
        assert ASTMD6433PCI.get_severity_from_width(
            DistressType.DEPRESSION, 4000.0, 20.0) is SeverityLevel.LOW
        assert ASTMD6433PCI.get_severity_from_width(
            DistressType.DEPRESSION, 1.0, 40.0) is SeverityLevel.MEDIUM
        assert ASTMD6433PCI.get_severity_from_width(
            DistressType.DEPRESSION, 1.0, 60.0) is SeverityLevel.HIGH

    def test_condition_rated_distresses_carry_no_invented_band(self):
        """Alligator and the rest are rated on condition, not on a size.

        The standard decides them by whether pieces are spalled, whether a patch
        needs replacing, and how the vehicle rides (X1.5.1, X1.15.1, X1.19.1).
        A size threshold has no standing there, so a wide patch must not come
        out as a severe one on the strength of its extent.
        """
        for dtype in (DistressType.ALLIGATOR, DistressType.PATCHING,
                      DistressType.RAVELLING, DistressType.CORRUGATION,
                      DistressType.SHOVING, DistressType.BLEEDING):
            assert dtype in ASTMD6433PCI.CONDITION_RATED_TYPES
            assert ASTMD6433PCI.get_severity_from_width(
                dtype, 9000.0) is SeverityLevel.LOW


class TestMultiDistressCorrectionFollowsNinePointFive:
    """ASTM D 6433-99 9.5, against the standard's own worked example in Fig. 6.

    The standard has no table of factors for how many kinds of distress were
    found. It ranks the individual deduct values, keeps the m that Eq 4 allows,
    sums them, and corrects the sum against the curve for q.
    """

    def test_the_standard_worked_example_reproduces_exactly(self):
        # 9.5.3: HDV 25.1 gives m = 7.9.
        assert ASTMD6433PCI.allowable_deducts(25.1) == pytest.approx(7.9, abs=0.01)
        # 9.5.4: the eight values, the eighth scaled by the 0.9 fraction -> 4.8.
        kept = ASTMD6433PCI._truncate_to_m(
            [25.1, 23.4, 17.9, 11.2, 7.9, 7.5, 6.9, 5.3]
        )
        assert kept[-1] == pytest.approx(4.77, abs=0.05)
        # 9.5.5.1: the total the standard prints, 104.7.
        assert sum(kept) == pytest.approx(104.7, abs=0.05)
        # 9.5.5.2: q is the number above 2.0, and the standard says 8.
        assert sum(1 for v in kept if v > 2.0) == 8

    def test_m_falls_as_the_worst_distress_dominates(self):
        """A road whose damage is one severe defect keeps only that one."""
        assert ASTMD6433PCI.allowable_deducts(0.0) == pytest.approx(10.0)
        assert ASTMD6433PCI.allowable_deducts(25.1) == pytest.approx(7.9)
        assert ASTMD6433PCI.allowable_deducts(100.0) == pytest.approx(1.0)

    def test_a_dominant_distress_truncates_the_minor_ones_away(self):
        """9.5.4: with a high HDV, m is small and the tail is dropped.

        HDV 40 gives m = 6.5, so nine values become seven: the six largest
        whole, plus a share of the seventh. The eight small ones do not all get
        charged.
        """
        kept = ASTMD6433PCI._truncate_to_m([40.0] + [4.0] * 8)
        assert len(kept) == 7
        assert kept[0] == 40.0
        # Six whole values and one fractional share, not eight more whole ones.
        assert sum(kept) == pytest.approx(62.0, abs=0.01)
        assert sum(kept) < 40.0 + 8 * 4.0

    def test_the_correction_is_largest_of_the_iterations(self):
        """9.5.5.5: the answer is the largest CDV the passes produce.

        The passes do not rise monotonically, so taking the last one would
        understate the damage. This case peaks at q = 2, not at the final q = 1.
        """
        value = ASTMD6433PCI.corrected_deduct_value([40.0] + [4.0] * 8)
        assert value == pytest.approx(89.37, abs=0.05)

    def test_a_single_severe_defect_still_costs(self):
        assert 0.0 < ASTMD6433PCI.corrected_deduct_value([40.0]) < 100.0

    def test_more_defects_never_raises_the_pci(self):
        """Monotone, which the per-type factor multiplied through could break."""
        previous_pci = 101.0
        for count in (1, 2, 3, 4, 6, 8, 10, 15, 20):
            pci = 100.0 - ASTMD6433PCI.corrected_deduct_value([10.0] * count)
            assert pci <= previous_pci + 1e-9
            previous_pci = pci

    def test_one_value_above_two_takes_the_total_outright(self):
        """9.5.1: no curve is consulted when at most one value exceeds 2.0."""
        assert ASTMD6433PCI.corrected_deduct_value([25.1, 1.0, 0.5]) == pytest.approx(26.6)

    def test_nothing_is_never_a_penalty(self):
        assert ASTMD6433PCI.corrected_deduct_value([]) == 0.0
        assert ASTMD6433PCI.corrected_deduct_value([0.0, 0.0]) == 0.0

    def test_more_distress_never_raises_the_pci(self):
        """Monotonicity, which a per-type factor multiplied through could break."""
        previous = 100.0
        for extra in range(0, 12):
            values = [10.0, 8.0, 6.0][: 1 + extra % 3] + [3.0] * extra
            pci = 100.0 - ASTMD6433PCI.corrected_deduct_value(values)
            assert pci <= previous + 1e-9
            previous = max(previous, pci)

    def test_the_removed_factor_table_is_not_consulted(self):
        """Guard against the invented per-type factor creeping back in.

        Two distress types and the same two deduct values must land on the same
        corrected value, because the standard's procedure does not look at how
        many kinds were found.
        """
        assert not hasattr(ASTMD6433PCI, "CORRECTION_FACTORS")
        same = [12.0, 7.0]
        assert ASTMD6433PCI.corrected_deduct_value(same) == pytest.approx(
            ASTMD6433PCI.corrected_deduct_value(list(same))
        )


class TestProvenanceReachesTheStoredResult:
    """A stored PCI number must not travel without its caveat.

    The whole exercise here has been to stop a plausible number being read as an
    ASTM result when it is not one. That only matters if the caveat reaches where
    the number is stored and served, so this follows the value from the section
    calculation to the run result that a caller persists and an endpoint returns.
    """

    def test_a_run_result_says_its_own_pci_is_provisional(self):
        result = ASTMD6433PCI.compute_pci_from_detections(
            [{"chainage_m": 10.0, "yolo_class_id": 4, "area": 0.05, "width": 0.2}],
            total_length_m=200.0,
        )
        provenance = result["provenance"]
        # The procedure is D6433-24 (9.6.2 does not exist before -20), but the
        # curves were digitized from the -99 figures. Reporting one edition for
        # both would hide the only thing a reader needs to know.
        assert provenance["standard"] == "ASTM D6433 D6433-24"
        assert provenance["curve_data_edition"] == "D6433-99"
        # True today because the curves are a first-pass digitization and the
        # X3.26 correction is still a placeholder. When both are fixed this is the
        # assertion to change, and it is the signal that they are.
        assert provenance["pci_is_provisional"] is True
        assert provenance["curves_provisional"] is True
        # X3.26 is the -99 figure and the dataset is -99 data, so the -99 figure
        # number is the accurate one even though the procedure is -24.
        assert provenance["cdv_curve_figure"] == "X3.26"
        assert provenance["cdv_curve_is_placeholder"] is True
        assert provenance["figures_used"] == ["X3.19"]

    def test_every_run_carries_what_the_number_is_and_is_not(self):
        """A PCI that reaches a reader without its limits reads as compliance.

        Clause 7 samples and 11.1 defines section PCI over randomly surveyed
        units; a video pass is a census, so this is a different statistic from
        the standard's regardless of how good the curves get. That has to travel
        with the value, not sit in a commit message.
        """
        result = ASTMD6433PCI.compute_pci_from_detections(
            [{"chainage_m": 10.0, "yolo_class_id": 4, "area": 0.05, "width": 0.2}],
            total_length_m=200.0,
        )
        limits = result["provenance"]["limitations"]

        assert limits["not"] == "an ASTM D6433 compliance value"
        assert "relative" in limits["use"]
        assert any("census" in reason for reason in limits["reasons"])
        # Ghana reports the network as Good/Fair/Poor; the seven ASTM bands are
        # a different vocabulary and must not be presented as interchangeable.
        assert "Good/Fair/Poor" in limits["ratings_are"]

    def test_a_clean_run_still_carries_the_limitations(self):
        """A perfect score still needs its caveats, most of all."""
        result = ASTMD6433PCI.compute_pci_from_detections([], total_length_m=200.0)
        assert result["provenance"]["limitations"] is not None

    def test_a_clean_run_still_reports_itself_as_provisional(self):
        """A road with no defects has no wrong answer, but the method is still unverified.

        `pci_is_provisional` is a statement about the method, not about this
        survey, so it stays true. `curves_provisional` is the one that says whether
        a deduct value was actually read from unverified data, and a clean run
        reads none — which is the distinction worth keeping, because it is what
        tells you how much of a number came out of a digitized graph.
        """
        result = ASTMD6433PCI.compute_pci_from_detections([], total_length_m=200.0)
        assert result["overall_pci"] == 100.0
        assert result["provenance"]["pci_is_provisional"] is True
        assert result["provenance"]["curves_provisional"] is False
        assert result["provenance"]["figures_used"] == []

    def test_capped_sections_are_named_at_run_level(self):
        result = ASTMD6433PCI.compute_pci_from_detections(
            [{"chainage_m": 10.0, "yolo_class_id": 2, "area": 6.0, "width": 0.02}],
            total_length_m=200.0,
        )
        assert 0 in result["provenance"]["density_capped_sections"]

    def test_the_stored_quality_flags_carry_the_provenance(self, astm_pci_on):
        """The JSON column the pipeline already writes is where it has to land.

        `SurveyRun.astm_quality_flags` was declared and served by the run
        endpoints, but nothing wrote to it. The PCI caveat goes there, so the
        existing response shape carries it without a schema change.

        The PCI is muted by default, so this opts in: the point being tested is
        where the provenance lands once a PCI is actually produced.
        """
        from app.services.fusion.service import PavementConditionFusioneer

        pci, breakdown = PavementConditionFusioneer._compute_astm_pci(
            _fake_qcm(), _fake_vision())
        flags = breakdown["provenance"]
        assert flags["pci_is_provisional"] is True
        assert flags["figures_used"] == ["X3.19"]

    def test_while_muted_the_column_carries_the_mute_reason_instead(self, monkeypatch):
        """The provenance block exists to stop a number being read as an ASTM
        result. With no number produced, what has to reach the column is the
        reason there is none, or the endpoint shows a null that is
        indistinguishable from a run that never computed anything."""
        from app.services.fusion import service as fusion_service
        from app.services.fusion.service import PavementConditionFusioneer

        monkeypatch.setattr(fusion_service, "ASTM_PCI_ENABLED", False)
        _pci, breakdown = PavementConditionFusioneer._compute_astm_pci(
            _fake_qcm(), _fake_vision())
        assert breakdown["enabled"] is False
        assert breakdown["reason"].strip()
        assert "provenance" not in breakdown


def _fake_qcm():
    class Q:
        total_distance_m = 200.0
        iri_m_km = 2.5
        quality_flags = {"iri": "ok"}
        iri_segments = []

        class _Seg:
            start_distance_m = 0.0
            end_distance_m = 50.0
            iri_m_km = 2.5
        iri_segments = [_Seg()]
    return Q()


def _fake_vision():
    class V:
        detections = [{"yolo_class_id": 4, "chainage_m": 10.0,
                       "area": 0.05, "width": 0.2, "pothole_depth_mm": 0.02}]

        class _VQ:
            fps_flag = "OK"
        video_quality = _VQ()
    return V()


class TestProvisionalSeveritiesAreReported:
    def test_condition_rated_distresses_are_counted_as_provisional(self):
        result = _section([_distress(DistressType.PATCHING, area_m2=2.0)])
        assert result["provisional_severity_detections"] == 1

    def test_measured_distresses_are_not_provisional(self):
        result = _section([_distress(DistressType.LONGITUDINAL, length_m=3.0,
                                     width_mm=4.0)])
        assert result["provisional_severity_detections"] == 0
