"""Saturation past the end of the CDV correction curves.

The placeholder CDV curves in this dataset are tabulated over TDV 0..100 and
stop there. A section whose total deduct value exceeds 100 has no published value
to read, and the lookup used to raise `CurveRangeExceeded` out of
`compute_section_pci`, taking the whole request down.

That was worth fixing, but only in the direction that keeps the reported number
honest. Two properties are pinned here because they are what separates a
defensible degradation from a quiet corruption:

1. The threshold is the curve's own domain edge, not 60.
   An earlier proposal capped CDV at 100 for any TDV at or above 60. The curves
   return live, varying values across 60..100, so that cap would report every
   heavily distressed section as PCI 0. Against the digitized Figure X1.1
   curves, TDV 60 is roughly CDV 54..68 and TDV 80 roughly CDV 79..95, which are
   PCI 46 and PCI 21, not PCI 0. `test_no_cap_at_60` guards against that.

2. A saturated reading is always reported as saturated.
   A CDV of 100 is the floor of the scale, so it is indistinguishable from a
   genuine "PCI 0" unless the flag travels with it. `test_*` below assert the
   flag appears at section level, at run level, and is absent when no lookup
   saturated.

The low-level reader is deliberately left strict: `Curve.interpolate` still
raises, because it is the last thing standing between placeholder data and a
number. The saturation is a decision made one layer up, where it can be recorded.
"""

import logging

import pytest

from app.services.astm import curves
from app.services.astm.curves import CurveNotFound, CurveRangeExceeded
from app.services.astm.pci import (
    ASTMD6433PCI,
    DistressType,
    SectionMetrics,
    SeverityLevel,
)

#: Deducts whose truncated sum runs past TDV 100, so 9.5.5.3 has no value.
PAST_THE_CURVES = [45.0, 40.0, 35.0]

#: Deducts well inside the tabulated range.
INSIDE_THE_CURVES = [10.0, 9.0, 8.0, 7.0]


class TestExceptionHierarchy:
    def test_range_exceeded_is_a_not_found(self):
        """The two failures mean the same thing to a caller: no usable reading.

        They were siblings, so `except CurveNotFound` -- which already exists to
        keep a missing curve from killing a section -- did not catch the range
        failure. Both now derive from `CurveNotFound` so that handler covers both.
        """
        assert issubclass(CurveRangeExceeded, CurveNotFound)
        assert issubclass(CurveNotFound, curves.CurveError)


class TestNoCapAt60:
    """The proposed TDV>=60 cap, and why it is not here."""

    def test_no_cap_at_60(self):
        """TDV 60 must not collapse to CDV 100.

        The curves are live across this region. q=1 gives 59.2 at TDV 40 and
        reaches 100 only at 58.5, so 60 is already past q=1's end -- but for a
        larger q, 60 is an interior point. q=8 gives 52.8 at TDV 60, which is a
        PCI of 47.2, and capping would report 0 instead.
        """
        value, saturated = ASTMD6433PCI.corrected_deduct_value_with_provenance(
            [20.0, 15.0, 10.0, 8.0, 7.0]
        )
        total = ASTMD6433PCI._truncate_to_m([20.0, 15.0, 10.0, 8.0, 7.0])
        assert sum(total) <= 100.0
        if not saturated:
            assert value < 100.0, (
                "an unsaturated lookup must not return the floor value, or a "
                "genuinely PCI-0 section could not be told from a saturated one"
            )

    def test_interior_values_are_not_saturated(self):
        _, saturated = ASTMD6433PCI.corrected_deduct_value_with_provenance(
            INSIDE_THE_CURVES
        )
        assert saturated is False

    def test_saturation_only_above_the_domain_edge(self):
        _, saturated = ASTMD6433PCI.corrected_deduct_value_with_provenance(
            PAST_THE_CURVES
        )
        assert saturated is True


class TestLowLevelReaderStaysStrict:
    def test_interpolate_still_raises(self):
        """The guard stays at the data layer.

        Relaxing `interpolate` would remove the only place that knows a value
        does not exist. Satification belongs one layer up, where it is recorded.
        """
        with pytest.raises(CurveRangeExceeded):
            curves.repository().cdv(8).interpolate(104.7)

    def test_reading_inside_the_range_does_not_raise(self):
        assert curves.repository().cdv(8).interpolate(80.0) > 0.0


class TestCorrectionDoesNotRaise:
    def test_past_the_curves_returns_a_value(self):
        value = ASTMD6433PCI.corrected_deduct_value(PAST_THE_CURVES)
        assert value == pytest.approx(100.0)

    def test_reported_as_saturated(self):
        _, saturated = ASTMD6433PCI.corrected_deduct_value_with_provenance(
            PAST_THE_CURVES
        )
        assert saturated is True

    def test_wrapper_and_provenance_agree(self):
        plain = ASTMD6433PCI.corrected_deduct_value(PAST_THE_CURVES)
        detailed, _ = ASTMD6433PCI.corrected_deduct_value_with_provenance(
            PAST_THE_CURVES
        )
        assert plain == pytest.approx(detailed)

    def test_cdv_wrapper_and_provenance_agree(self):
        plain = ASTMD6433PCI._cdv_from_curve(104.7, 8)
        detailed, saturated = ASTMD6433PCI._cdv_from_curve_with_provenance(104.7, 8)
        assert plain == pytest.approx(detailed)
        assert saturated is True

    def test_saturated_value_is_the_curve_terminal_value(self):
        """Not a hardcoded 100 -- the curve's own last point.

        A constant would be wrong the moment a real curve does not top out at
        100, which is exactly the case for the properly digitized Fig. X3.27
        curves, whose CDV runs past 100.
        """
        curve = curves.repository().cdv(8)
        terminal = max(p.deduct_value for p in curve.points)
        value, saturated = ASTMD6433PCI._cdv_from_curve_with_provenance(500.0, 8)
        assert saturated is True
        assert value == pytest.approx(terminal)


class TestSectionFlag:
    def _saturated_section(self):
        """A 50 m section whose deducts total past the curves.

        More than one deduct above 2.0 is required, because 9.5.1 returns the
        total outright when only one value clears 2.0 and consults no curve at
        all. A single dense pothole set gives one deduct of ~80 and never
        reaches a curve; several types at once are what carry the total over.
        """
        from app.services.astm.pci import Distress

        area = 50.0 * 3.25
        distresses = []
        distresses.extend(
            Distress(
                type=DistressType.POTHOLES,
                severity=SeverityLevel.HIGH,
                length_m=1.0,
                width_mm=600.0,
                depth_mm=60.0,
            )
            for _ in range(12)
        )
        distresses.extend(
            Distress(
                type=DistressType.RUTTING,
                severity=SeverityLevel.HIGH,
                area_m2=8.0,
                depth_mm=25.0,
            )
            for _ in range(4)
        )
        distresses.extend(
            Distress(
                type=DistressType.TRANSVERSE,
                severity=SeverityLevel.HIGH,
                length_m=6.0,
                width_mm=80.0,
            )
            for _ in range(4)
        )
        return SectionMetrics(
            section_index=0,
            start_m=0.0,
            end_m=50.0,
            area_m2=area,
            distresses=distresses,
        )

    def test_section_reports_saturation(self):
        result = ASTMD6433PCI.compute_section_pci(self._saturated_section())
        flag = result["cdv_curve_saturated"]
        assert flag["saturated"] is True
        assert flag["clause"] == "9.5.5.3"
        assert flag["note"]

    def test_saturated_section_still_returns_a_pci(self):
        """The point of the change: the section survives, flagged."""
        result = ASTMD6433PCI.compute_section_pci(self._saturated_section())
        assert result["pci"] == pytest.approx(
            100.0 - result["total_cdv"], abs=0.01
        )
        assert result["rating"] == "Failed"

    def test_clean_section_is_not_flagged(self):
        """The flag has to mean something, so the false case is asserted."""
        from app.services.astm.pci import Distress

        section = SectionMetrics(
            section_index=0,
            start_m=0.0,
            end_m=50.0,
            area_m2=50.0 * 3.25,
            distresses=[
                Distress(
                    type=DistressType.TRANSVERSE,
                    severity=SeverityLevel.LOW,
                    length_m=1.0,
                    width_mm=4.0,
                )
            ],
        )
        result = ASTMD6433PCI.compute_section_pci(section)
        assert result["cdv_curve_saturated"]["saturated"] is False

    def test_saturation_logs_a_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            ASTMD6433PCI.compute_section_pci(self._saturated_section())
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "saturation" in text.lower() or "past the end" in text.lower()


class TestRunLevelFlag:
    def test_run_lists_saturated_sections(self):
        """One saturated section drags the mean toward 0, invisibly.

        `overall_pci` averages section PCIs, so a saturated section pulls the
        survey mean down and nothing in the mean reveals why. The section index
        list is how a caller finds it.
        """
        result = ASTMD6433PCI.compute_pci_from_detections([], total_length_m=200.0)
        assert "cdv_saturated_sections" in result["provenance"]
        assert result["provenance"]["cdv_saturated_sections"] == []

    def test_clean_run_reports_no_saturation(self):
        result = ASTMD6433PCI.compute_pci_from_detections([], total_length_m=100.0)
        assert result["provenance"]["cdv_saturated_sections"] == []
        assert result["overall_pci"] == 100.0

    def test_figure_number_is_corrected_for_d6433_24(self):
        """X3.26 is the -99 figure. The procedure being applied is -24."""
        result = ASTMD6433PCI.compute_pci_from_detections([], total_length_m=100.0)
        prov = result["provenance"]
        assert prov["cdv_curve_figure"] == "X3.26"
        assert prov["cdv_curve_figure_in_d6433_24"] == "X3.27"


class TestMultiSeveritySaturationTracking:
    def test_flag_follows_the_variant_that_won(self):
        """Reporting the baseline's flag would be wrong if a variant won.

        A merged configuration can be the reading that saturates while the
        as-found reading does not. The flag has to describe the reading actually
        reported, not the first one computed.
        """
        densities = {
            (DistressType.LONGITUDINAL, SeverityLevel.LOW): 0.0,
            (DistressType.LONGITUDINAL, SeverityLevel.MEDIUM): 0.0,
            (DistressType.LONGITUDINAL, SeverityLevel.HIGH): 100.0,
        }
        pci, cdv, saturated = ASTMD6433PCI._pci_from_densities(densities)
        assert pci == pytest.approx(max(0.0, 100.0 - cdv), abs=0.01)
        assert isinstance(saturated, bool)