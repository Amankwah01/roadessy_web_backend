"""Two clauses that were not implemented, and one that was implemented wrongly.

**9.3, per-type-and-severity deduct values.** The standard gives a deduct value
for each distress type *and severity level combination*. The code summed the
severities of one type into a single deduct value first, so a distress present
at two levels looked like one severe distress: it counted as a single deduct
above 2.0 and 9.5.1 then used that total as the maximum CDV outright. Fig. 4
lists the two levels of its alligator cracking as separate entries for exactly
this reason.

**9.6.2, PCI correction for multiple severities.** New in D6433-24. A distress
recorded at more than one severity may be one continuous extent graded
differently along its length, so the standard requires the merged readings to be
compared against the as-found one and the highest PCI to win:

    9.6.2.1  two severities      (x1, x2) against (0, X2)
    9.6.2.2  three severities    (l,m,h) (0,M,h) (l,0,H) (0,m,H) (0,0,H)

NOTE 1 under 12.1 records this being run against the standard's own Fig. 4 and
needing no correction. That is a fact about that data, not an exemption.

The PCI numbers these tests assert are not meaningful yet. Every lookup goes
through placeholder deduct-value curves, so the arithmetic is exercised while
the values stay provisional. What is pinned here is the structure: how many
deduct values exist, how many configurations are compared, and that the highest
PCI is the one reported.
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


def _crack(length_m, width_mm):
    """Longitudinal cracking, which X1.14.2 rates on width, so the width
    chooses the severity and a section can carry several levels at once."""
    return Distress(
        type=DistressType.LONGITUDINAL,
        severity=SeverityLevel.LOW,
        length_m=length_m,
        width_mm=width_mm,
        depth_mm=None,
        chainage_start_m=0.0,
        chainage_end_m=length_m,
    )


def _section(distresses):
    return ASTMD6433PCI.compute_section_pci(
        SectionMetrics(0, 0.0, 50.0, SECTION_AREA_M2, list(distresses))
    )


def _pothole(length_m=1.0, width_mm=0.0, depth_mm=None):
    return Distress(
        type=DistressType.POTHOLES,
        severity=SeverityLevel.LOW,
        length_m=length_m,
        width_mm=width_mm,
        depth_mm=depth_mm,
        chainage_start_m=0.0,
        chainage_end_m=length_m,
    )


class TestDeductValuesArePerTypeAndSeverity:
    def test_one_type_at_two_levels_gives_two_deduct_values(self):
        """The defect: these collapsed into one summed deduct value."""
        result = _section([_crack(20.0, 5.0), _crack(15.0, 30.0)])
        assert len(result["individual_deduct_values"]) == 2

    def test_two_types_at_one_level_also_give_two(self):
        """The contrast: this was already right, which is why it went unseen."""
        result = _section([
            _crack(20.0, 5.0),
            Distress(
                type=DistressType.PATCHING, severity=SeverityLevel.LOW,
                length_m=0.0, area_m2=14.0, width_mm=0.0, depth_mm=None,
                chainage_start_m=0.0, chainage_end_m=0.0,
            ),
        ])
        assert len(result["individual_deduct_values"]) == 2

    def test_three_levels_give_three(self):
        result = _section([_crack(10.0, 5.0), _crack(10.0, 30.0), _crack(8.0, 90.0)])
        assert len(result["individual_deduct_values"]) == 3

    def test_breakdown_still_reports_the_type_total(self):
        """9.3 splits the deduct values; the report still totals by type."""
        result = _section([_crack(20.0, 5.0), _crack(15.0, 30.0)])
        total = sum(result["individual_deduct_values"])
        assert result["breakdown"]["longitudinal"]["tdv"] == pytest.approx(
            total, abs=0.01
        )


class TestMultiSeverityCorrectionClause962:
    def test_single_severity_needs_no_correction(self):
        result = _section([_crack(20.0, 5.0)])
        correction = result["multi_severity_correction"]
        assert correction["variants_evaluated"] == 1
        assert correction["applied"] is False
        assert result["pci"] == correction["baseline_pci"]

    def test_two_severities_evaluate_two_configurations(self):
        """9.6.2.1: (x1, x2) against (0, X2)."""
        result = _section([_crack(20.0, 5.0), _crack(15.0, 30.0)])
        assert result["multi_severity_correction"]["variants_evaluated"] == 2

    def test_three_severities_evaluate_the_five_9_6_2_2_lists(self):
        """Baseline plus the four merges the standard names."""
        result = _section([_crack(10.0, 5.0), _crack(10.0, 30.0), _crack(8.0, 90.0)])
        assert result["multi_severity_correction"]["variants_evaluated"] == 5

    def test_reported_pci_is_never_below_the_as_found_reading(self):
        """"the correct PCI should be based on the combination that provides
        the highest PCI", so a correction can only raise it."""
        for distresses in (
            [_crack(20.0, 5.0), _crack(15.0, 30.0)],
            [_crack(10.0, 5.0), _crack(10.0, 30.0), _crack(8.0, 90.0)],
            [_pothole(1.0, width_mm=20.0, depth_mm=10.0),
             _pothole(1.0, width_mm=900.0, depth_mm=60.0)],
        ):
            correction = _section(distresses)["multi_severity_correction"]
            assert correction["reported_pci"] >= correction["baseline_pci"]

    def test_variants_move_density_rather_than_destroying_it(self):
        """A merge relocates extent between severity levels; it cannot lose it.

        This is what separates the standard's operation from simply adding the
        levels together, and it holds independently of the curve values.
        """
        densities = {
            (DistressType.LONGITUDINAL, SeverityLevel.LOW): 4.0,
            (DistressType.LONGITUDINAL, SeverityLevel.MEDIUM): 3.0,
            (DistressType.LONGITUDINAL, SeverityLevel.HIGH): 1.5,
        }
        baseline = sum(densities.values())
        variants = ASTMD6433PCI._merge_variants(densities)

        assert len(variants) == 5
        for variant in variants:
            assert sum(variant.values()) == pytest.approx(baseline)

    def test_the_fully_merged_variant_puts_everything_in_one_level(self):
        densities = {
            (DistressType.LONGITUDINAL, SeverityLevel.LOW): 4.0,
            (DistressType.LONGITUDINAL, SeverityLevel.MEDIUM): 3.0,
            (DistressType.LONGITUDINAL, SeverityLevel.HIGH): 1.5,
        }
        variants = ASTMD6433PCI._merge_variants(densities)
        merged_all = variants[-1]
        assert merged_all[(DistressType.LONGITUDINAL, SeverityLevel.LOW)] == 0.0
        assert merged_all[(DistressType.LONGITUDINAL, SeverityLevel.MEDIUM)] == 0.0
        assert merged_all[(DistressType.LONGITUDINAL, SeverityLevel.HIGH)] == \
            pytest.approx(8.5)

    def test_a_zero_density_variant_contributes_no_deduct_value(self):
        """(0, 0, H) is three levels, but only one of them is distress.

        A folded level is absent rather than free, so it must not survive
        into the ranked list as a zero-valued deduct.
        """
        densities = {
            (DistressType.LONGITUDINAL, SeverityLevel.LOW): 0.0,
            (DistressType.LONGITUDINAL, SeverityLevel.MEDIUM): 0.0,
            (DistressType.LONGITUDINAL, SeverityLevel.HIGH): 2.0,
        }
        pci, _, _ = ASTMD6433PCI._pci_from_densities(densities)
        assert pci < 100.0, "the surviving high level should still cost something"

        all_clear = {key: 0.0 for key in densities}
        assert ASTMD6433PCI._pci_from_densities(all_clear)[0] == 100.0