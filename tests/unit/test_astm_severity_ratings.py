"""Severity for the distresses ASTM does not rate on a measurable extent.

X1.4.1 to X1.4.5: "Ride quality must be evaluated in order to establish a
severity level for" bumps and sags, corrugation, railroad crossings, shoving,
and swells. X1.4.6 then defines the levels by what the vehicle does -- L for
vibrations noticeable with no speed reduction, M for significant vibrations
needing some reduction, H for excessive vibrations needing considerable
reduction. X1.4.7 sets the conditions: driving at the posted speed.

None of that is measurable from a detection box, so the severity is an
inspector's and has to arrive as one. This checks that it is used when given,
and that its absence is reported rather than hidden behind the LOW fallback.
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


def _d(dtype, *, area_m2=0.0, length_m=0.0, rated=None):
    return Distress(
        type=dtype,
        severity=SeverityLevel.LOW,
        length_m=length_m,
        area_m2=area_m2,
        width_mm=0.0,
        depth_mm=None,
        chainage_start_m=0.0,
        chainage_end_m=length_m,
        rated_severity=rated,
    )


def _section(distresses):
    return ASTMD6433PCI.compute_section_pci(
        SectionMetrics(0, 0.0, 50.0, SECTION_AREA_M2, list(distresses))
    )


class TestRideQualityTypesAreTheX14List:
    def test_both_modelled_x14_types_are_present(self):
        assert DistressType.CORRUGATION in ASTMD6433PCI.RIDE_QUALITY_TYPES
        assert DistressType.SHOVING in ASTMD6433PCI.RIDE_QUALITY_TYPES

    def test_ride_quality_types_are_a_subset_of_condition_rated(self):
        """They are rated on condition, and X1.4 is how two of them are rated."""
        assert ASTMD6433PCI.RIDE_QUALITY_TYPES <= ASTMD6433PCI.CONDITION_RATED_TYPES

    def test_a_measurable_type_is_not_ride_quality(self):
        assert DistressType.LONGITUDINAL not in ASTMD6433PCI.RIDE_QUALITY_TYPES


class TestInspectorRatingIsUsed:
    @pytest.mark.parametrize("rated", [
        SeverityLevel.LOW, SeverityLevel.MEDIUM, SeverityLevel.HIGH,
    ])
    def test_the_supplied_rating_reaches_the_deduct_value(self, rated):
        """The rating has to change the answer, not just be recorded."""
        densities = ASTMD6433PCI._severity_densities(
            SectionMetrics(0, 0.0, 50.0, SECTION_AREA_M2,
                           [_d(DistressType.SHOVING, area_m2=20.0, rated=rated)])
        )
        assert densities == {(DistressType.SHOVING, rated): pytest.approx(
            20.0 / SECTION_AREA_M2 * 100.0
        )}

    def test_the_same_distress_rates_differently_by_level(self):
        def pci_for(level):
            return _section([
                _d(DistressType.CORRUGATION, area_m2=20.0, rated=level)
            ])["individual_deduct_values"][0]

        low, medium, high = (pci_for(s) for s in
                             (SeverityLevel.LOW, SeverityLevel.MEDIUM,
                              SeverityLevel.HIGH))
        assert low < medium < high, (
            "ride quality must cost more as the ride gets worse"
        )

    def test_a_rated_section_is_not_provisional(self):
        result = _section([
            _d(DistressType.SHOVING, area_m2=20.0, rated=SeverityLevel.MEDIUM)
        ])
        assert result["provisional_severity_detections"] == 0
        assert result["severity_ratings"]["ride_quality_rated"] == ["shoving"]
        assert result["severity_ratings"]["unrated"] == []


class TestMissingRatingIsReported:
    def test_an_unrated_section_is_provisional(self):
        result = _section([_d(DistressType.SHOVING, area_m2=20.0)])
        assert result["provisional_severity_detections"] == 1
        assert result["severity_ratings"]["unrated"] == ["shoving"]

    def test_the_fallback_is_low_and_says_so(self):
        """LOW understates the distress, so the report must not read as rated."""
        unrated = _section([_d(DistressType.SHOVING, area_m2=20.0)])
        rated_low = _section([
            _d(DistressType.SHOVING, area_m2=20.0, rated=SeverityLevel.LOW)
        ])
        # Same number, because the fallback is LOW...
        assert unrated["pci"] == rated_low["pci"]
        # ...but only one of them is provisional.
        assert rated_low["provisional_severity_detections"] == 0
        assert unrated["provisional_severity_detections"] == 1

    def test_measured_types_are_never_provisional_for_severity(self):
        """Cracking is rated on width, so there is nothing to judge."""
        from app.services.astm.pci import Distress as D

        crack = D(
            type=DistressType.LONGITUDINAL, severity=SeverityLevel.LOW,
            length_m=20.0, width_mm=30.0, depth_mm=None,
            chainage_start_m=0.0, chainage_end_m=20.0,
        )
        result = _section([crack])
        assert result["provisional_severity_detections"] == 0
        assert result["severity_ratings"]["unrated"] == []

    def test_resolve_severity_reports_its_basis(self):
        shoving = _d(DistressType.SHOVING, area_m2=5.0, rated=SeverityLevel.HIGH)
        assert ASTMD6433PCI.resolve_severity(shoving) == (
            SeverityLevel.HIGH, "inspector"
        )
        assert ASTMD6433PCI.resolve_severity(_d(DistressType.SHOVING, area_m2=5.0)) == (
            SeverityLevel.LOW, "unrated"
        )

        crack = Distress(
            type=DistressType.LONGITUDINAL, severity=SeverityLevel.LOW,
            length_m=20.0, width_mm=30.0, depth_mm=None,
            chainage_start_m=0.0, chainage_end_m=20.0,
            rated_severity=SeverityLevel.HIGH,   # must be ignored
        )
        assert ASTMD6433PCI.resolve_severity(crack) == (
            SeverityLevel.MEDIUM, "measured"
        )

    def test_the_count_matches_the_distress_count_not_the_type_count(self):
        """Three shoving detections are three outstanding judgements."""
        result = _section([_d(DistressType.SHOVING, area_m2=2.0)
                           for _ in range(3)])
        assert result["provisional_severity_detections"] == 3