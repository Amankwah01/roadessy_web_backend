"""Curve data as reference data: lookup, interpolation, and refusing to guess.

The behaviours pinned here are the ones that make a later digitization safe to
drop in. They matter more than the numbers, because the numbers are placeholders
today and will be replaced by a read of the published graphs.
"""

import csv
import json
import math
import os

import pytest

from app.services.astm import curves
from app.services.astm.curves import (
    Curve,
    CurveNotFound,
    CurvePoint,
    CurveRangeExceeded,
    CurveRepository,
)
from app.services.astm.pci import ASTMD6433PCI, DistressType, SeverityLevel


def _line_curve(points, x_scale="log10", **kwargs):
    return Curve(
        distress="test", severity="M", figure="X9.9",
        points=[CurvePoint(d, v) for d, v in points],
        x_scale=x_scale, **kwargs
    )


class TestInterpolation:
    def test_a_value_between_points_is_interpolated(self):
        curve = _line_curve([(0.0, 10.0), (10.0, 20.0)], x_scale="linear")
        assert curve.interpolate(5.0) == pytest.approx(15.0, abs=0.01)

    def test_the_endpoints_return_themselves(self):
        curve = _line_curve([(1.0, 10.0), (10.0, 20.0)])
        assert curve.interpolate(1.0) == pytest.approx(10.0)
        assert curve.interpolate(10.0) == pytest.approx(20.0)

    def test_the_log_axis_changes_the_answer_and_must_be_honoured(self):
        """The reason `x_scale` is recorded rather than assumed.

        On a log density axis the midpoint in log space is not the midpoint in
        density: across 1 to 100 the log midpoint is 10, and a linear reading
        treats that same point as barely a tenth of the way along. A curve
        digitized from an ASTM graph has to be read the way it was drawn, or the
        interpolation lands well away from the value the graph shows.
        """
        points = [(1.0, 0.0), (100.0, 100.0)]
        on_log = _line_curve(points, x_scale="log10")
        on_linear = _line_curve(points, x_scale="linear")
        assert on_log.interpolate(10.0) == pytest.approx(50.0, abs=0.01)
        assert on_linear.interpolate(10.0) == pytest.approx(9.09, abs=0.05)

    def test_the_curve_is_monotone_across_its_whole_range(self):
        curve = curves.repository().get("rutting", "M")
        previous = -1.0
        for i in range(60):
            density = 10.0 ** (-3.0 + i * 0.1)
            if density < curve.low_density or density > curve.high_density:
                continue
            value = curve.interpolate(density)
            assert value >= previous - 1e-9
            previous = value


class TestNoSilentExtrapolation:
    def test_a_density_past_the_end_of_the_data_raises(self):
        """An extended lookup table produces a number no measurement supports."""
        curve = _line_curve([(1.0, 10.0), (10.0, 20.0)])
        with pytest.raises(CurveRangeExceeded) as excinfo:
            curve.interpolate(50.0)
        # The bounds travel with the error, because a density past the end of
        # the graph usually means the detection reaches further than the curve.
        assert excinfo.value.low == pytest.approx(1.0)
        assert excinfo.value.high == pytest.approx(10.0)

    def test_no_density_means_no_deduct_value(self):
        """A distress with no measured extent earns nothing, not the intercept.

        The old polynomial carried a nonzero constant term, so a detection with
        no area still cost a section points. That is an artefact of fitting a
        curve with a polynomial, not something the curve says.
        """
        curve = _line_curve([(1.0, 10.0), (10.0, 20.0)])
        assert curve.interpolate(0.0) == 0.0
        assert curve.interpolate(0.5) == 0.0
        assert ASTMD6433PCI.get_dv(DistressType.RUTTING, SeverityLevel.MEDIUM, 0.0) == 0.0

    def test_a_zero_area_detection_does_not_penalise_a_section(self):
        """An area-measured distress with no area earns nothing.

        Potholes are the exception and deliberately so: they are counted, not
        area-measured (X1.17.2), so a pothole with no measured area is still one
        pothole. That is a separate test, because the distinction is the point.
        """
        result = ASTMD6433PCI.compute_pci_from_detections(
            # yolo class 5 is ravelling, an area-measured distress.
            detections=[{"chainage_m": 10.0, "yolo_class_id": 5, "area": 0.0}],
            total_length_m=50.0,
        )
        assert result["sections"][0]["pci"] == 100.0

    def test_a_pothole_is_counted_even_with_no_measured_area(self):
        """X1.17.2 counts potholes; the area is a severity input, not the quantity."""
        result = ASTMD6433PCI.compute_pci_from_detections(
            detections=[{"chainage_m": 10.0, "yolo_class_id": 4, "area": 0.0}],
            total_length_m=50.0,
        )
        assert result["sections"][0]["pci"] < 100.0

    def test_the_error_names_the_curve_and_the_standard(self):
        curve = _line_curve([(1.0, 10.0), (10.0, 20.0)])
        with pytest.raises(CurveRangeExceeded) as excinfo:
            curve.interpolate(1e6)
        message = str(excinfo.value)
        assert "test" in message and "M" in message and "Refusing" in message


class TestProvenance:
    def test_every_lookup_reports_where_it_came_from(self):
        record = ASTMD6433PCI.get_dv_with_provenance(
            DistressType.RUTTING, SeverityLevel.MEDIUM, 8.0
        )
        for field in ("standard", "edition", "distress", "severity", "figure",
                      "density_percent", "dv", "interpolation", "method",
                      "provisional"):
            assert field in record, field
        assert record["standard"].startswith("ASTM D6433")
        assert record["interpolation"] == "log10"

    def test_a_section_reports_the_curves_its_numbers_came_from(self):
        result = ASTMD6433PCI.compute_pci_from_detections(
            detections=[{"chainage_m": 10.0, "yolo_class_id": 4, "area": 0.25}],
            total_length_m=50.0,
        )
        section = result["sections"][0]
        assert section["deduct_value_sources"]
        assert section["individual_deduct_values"]
        for source in section["deduct_value_sources"]:
            assert source["figure"].startswith("X3.")

    def test_the_placeholder_status_is_reported_not_hidden(self):
        """The dataset is not ASTM data, and a result must say so.

        The distress curves are now read off the published figures rather than
        fitted to them, which is a real improvement, but the vendored file marks
        every point `UNVERIFIED_FIRST_PASS_DIGITIZATION` and nothing has checked
        them against the figures. So the result still has to be labelled
        provisional, and the method has to name the first pass.

        This is also the assertion that fails when a verified digitization replaces
        these, which is the moment to remove the provisional warning deliberately
        rather than by accident.
        """
        repository = curves.repository()
        assert repository.provisional_flags()
        record = ASTMD6433PCI.get_dv_with_provenance(
            DistressType.RUTTING, SeverityLevel.MEDIUM, 8.0
        )
        assert record["provisional"] is True
        # Not a fitted polynomial any more: the value is read off the figure.
        assert "placeholder" not in record["method"]
        assert "digitization" in record["method"]

    def test_the_dataset_pins_its_edition_and_says_it_is_not_authoritative(self):
        with open(curves.DATASET_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        assert data["edition"] == "D6433-99"
        assert data["authoritative"] is False
        assert data["standard"].startswith("ASTM D6433")

    def test_the_cdv_curve_is_the_total_deduct_value_figure(self):
        """Fig. X3.26, not the X4.15 the standard's own cross-reference names."""
        curve = curves.repository().cdv(5)
        assert curve.figure == "X3.26"
        assert curve.x_scale == "linear"


class TestDensityAxisIsLogarithmic:
    """Why the curves are read in log10 density, and what that choice costs.

    The vendored package says to "use log10(density) interpolation only after
    validation". It has not been validated, so the two things worth pinning are
    the evidence for the axis and the size of the exposure if it is wrong.

    The evidence is in the anchors themselves. 255 of the 284 supplied points sit
    exactly on a power of ten — 0.1, 1, 10, 100 — and the 29 that do not are all
    curve *endpoints* (15, 20, 30, 60) rather than interior reads. A digitization
    taken off a linear grid would land on arbitrary values; one taken off a
    logarithmic grid lands on the decades. That is an inference from the data, not
    a statement in the standard, which is why it is tested rather than assumed.

    The exposure is measured, because it is large: the axis choice is worth up to
    14.8 deduct points on the pothole curve, and PCI moves 1:1 with CDV.
    """

    def test_the_supplied_anchors_land_on_decades_which_is_what_a_log_axis_gives(self):
        with open(
            os.path.join(curves.DATA_DIR, "first_pass",
                         "astm_d6433_99_distress_curves_first_pass.json"),
            "r", encoding="utf-8",
        ) as handle:
            source = json.load(handle)

        on_decade, off_decade = 0, []
        for figure, data in source["figures"].items():
            for severity, points in data["severity_curves"].items():
                for point in points:
                    density = float(point["density_percent"])
                    exponent = round(math.log10(density), 6)
                    if abs(10 ** exponent - density) <= 1e-9:
                        on_decade += 1
                    else:
                        off_decade.append((figure, severity, density))

        assert on_decade == 255
        # 26 of the 29 that miss a decade are the right-hand end of their curve,
        # which is where a curve drawn to a maximum stops. The exceptions are
        # X3.24 swell, whose range starts at 1% and steps 1 to 5 to 30.
        ends = 0
        for figure, severity, density in off_decade:
            points = source["figures"][figure]["severity_curves"][severity]
            if float(points[-1]["density_percent"]) == density:
                ends += 1
        assert ends == 26
        # The three that miss a decade mid-curve are X3.24 swell's 5%, where the
        # curve starts at 1% and steps 1 to 5 to 30.
        interior = [
            (figure, severity, density)
            for figure, severity, density in off_decade
            if float(source["figures"][figure]["severity_curves"][severity][-1][
                "density_percent"]) != density
        ]
        assert {(f, s, d) for f, s, d in interior} == {
            ("X3.24", "H", 5.0), ("X3.24", "M", 5.0), ("X3.24", "L", 5.0),
        }

    def test_the_axis_choice_is_worth_at_most_a_bounded_amount_of_pci(self):
        """A ceiling on the exposure, so it cannot quietly grow.

        Not a tolerance the curves have to meet — a statement of what is currently
        unaccounted for. Replacing three or four anchors per curve with the
        ten-to-twenty the package recommends is what retires this.
        """
        worst = 0.0
        for key in ("potholes", "rutting", "longitudinal_cracking", "patching",
                    "alligator_cracking", "weathering_raveling"):
            curve = curves.repository().get(key, "M")
            low, high = curve.low_density, curve.high_density
            for i in range(200):
                density = low * (high / low) ** (i / 199.0)
                u = math.log10(density)
                positions = [math.log10(p.density_percent) for p in curve.points]
                linear = [p.density_percent for p in curve.points]
                worst = max(worst, abs(
                    _interp(u, positions, [p.deduct_value for p in curve.points])
                    - _interp(density, linear, [p.deduct_value for p in curve.points])
                ))
        assert worst < 15.0, (
            "the unvalidated log10 axis choice is worth {:.1f} PCI points".format(worst)
        )


def _interp(x, xs, ys):
    """Linear interpolation on whatever axis the caller has already mapped to."""
    if x <= xs[0]:
        return ys[0]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            span = xs[i + 1] - xs[i]
            if span == 0:
                return ys[i + 1]
            return ys[i] + (x - xs[i]) / span * (ys[i + 1] - ys[i])
    return ys[-1]


class TestAgainstAnIndependentPublishedImplementation:
    """Cross-check against the four worked sections in Sensors 2024, Table 4.

    Ibragimov et al. (Sensors 2024, 24, 2333) publish four sections with their
    individual deduct values, TDV, q, CDV and PCI. That is the closest thing to an
    independent check on this pipeline that exists, and it is worth having for
    what it settles as well as for what it contradicts.

    What it settles: the paper's q really is the ASTM count of deduct values above
    2.0, and for its three lightly distressed sections its TDV is the ASTM TDV.
    The severity wording in its text ("q1 representing the lowest severity") is
    loose, but the arithmetic is the standard's.

    What it contradicts: it has no m, no truncation, and no 9.5.5.4 iteration, so
    its Section 4 is a single-step answer where ASTM 9.5 iterates. It also does not
    sum to its own TDV. See `test_its_severe_section_diverges_from_9_5_5_4`.
    """

    CSV = os.path.join(
        curves.DATA_DIR, "first_pass", "sensors2024_table4_validation.csv",
    )

    def _sections(self):
        """Group the rows by section, preserving the paper's per-section totals."""
        with open(self.CSV, "r", encoding="utf-8") as handle:
            # The file carries its provenance as `#` comment lines, which
            # DictReader would otherwise take as the header row.
            data = "".join(
                line for line in handle if not line.lstrip().startswith("#")
            )
        sections = {}
        for row in csv.DictReader(data.splitlines()):
            key = int(row["section"])
            entry = sections.setdefault(key, {"dvs": [], "reported": None})
            entry["dvs"].append(float(row["dv"]))
            reported = (
                float(row["reported_tdv"]), int(row["reported_q"]),
                float(row["reported_cdv"]), float(row["reported_pci"]),
            )
            if entry["reported"] is None:
                entry["reported"] = reported
            else:
                assert entry["reported"] == reported, (
                    "section {} has inconsistent reported totals".format(key))
        return sections

    def test_its_q_is_the_astm_count_of_deducts_above_two(self):
        for index, entry in self._sections().items():
            astm_q = sum(1 for dv in entry["dvs"] if dv > 2.0)
            assert astm_q == entry["reported"][1], index

    def test_its_light_sections_sum_to_the_astm_total(self):
        """Sections 1-3: summing is the same as 9.5.3-9.5.4, because m does not bind.

        With two or three deducts and an HDV under 20, m comes out near 8.5, so
        nothing is truncated and the total is simply the sum. The paper agrees with
        the standard here, which is the point: the two only part company once a
        single distress dominates.
        """
        for index, entry in self._sections().items():
            if index == 4:
                continue  # the dominant-distress case, handled separately
            assert sum(entry["dvs"]) == pytest.approx(entry["reported"][0], abs=0.05), index

    def test_its_severe_section_diverges_from_9_5_5_4(self):
        """Section 4: the paper's own table does not close.

        Two separate problems, both in the printed table:

        * Its DV column sums to 112.01 but it reports TDV 104.73. Neither the sum
          nor ASTM's truncation (m = 3.42 against three deducts, so all three are
          kept) produces 104.73.
        * It reports one CDV. ASTM 9.5.5.4 requires reducing the smallest deduct
          value above 2.0 to 2.0 and repeating until one is left, then taking the
          **largest** CDV of the passes. On its own DVs that gives 85.35 and a PCI
          of 14.65, against its 65.23 and 34.77 — 20 points, and across the
          Very Poor / Serious boundary.

        This is the test that documents why this implementation follows 9.5 rather
        than the paper. It is not a criticism of a single section's number; it is
        the structural difference, and it shows up exactly where one distress
        dominates, which is the case 9.5 exists to handle.
        """
        dvs = self._sections()[4]["dvs"]
        reported = self._sections()[4]["reported"]
        assert sum(dvs) == pytest.approx(112.01, abs=0.01)
        assert reported[0] == pytest.approx(104.73, abs=0.01)
        # m does not truncate, so ASTM's total is the sum the table fails to report.
        m = 1.0 + (9.0 / 98.0) * (100.0 - max(dvs))
        assert m == pytest.approx(3.42, abs=0.01)
        assert m > len(dvs), "m does not bind here, so nothing should be truncated"
        assert sum(dvs) != pytest.approx(reported[0], abs=0.01)


class TestCorrectedDeductValueAccuracy:
    """How far the X3.26 correction curve is from the standard's own arithmetic.

    The vendored `x3_26_cdv_validation_points.csv` is not curve data. Every row
    is labelled `Worked example`: they are (TDV, q, CDV) triples read out of the
    standard's worked example, not points read off Fig. X3.26. They cannot stand
    in for the curve -- sixteen rows cannot describe ten curves, there are none at
    all for q above five, and the q=1 group is not monotonic in TDV, which a curve
    read off a graph cannot be.

    What they are good for is measuring. They are the standard's own arithmetic, so
    the error of the current curve against them is a real, checkable number, and it
    is the size of the gap this project still has to close on X3.26.

    The thresholds below are deliberately loose. They are not an accuracy claim:
    they assert the placeholder is not catastrophically wrong, and they record how
    wrong it currently is. Tightening them to a tolerance the curve has to meet is
    the test to write when a real digitization of X3.26 lands.
    """

    CSV = os.path.join(
        os.path.dirname(curves.DATASET_PATH), "first_pass",
        "x3_26_cdv_validation_points.csv",
    )

    def _points(self):
        with open(self.CSV, "r", encoding="utf-8") as handle:
            return [
                (float(row["tdv"]), int(row["q"]), float(row["cdv"]))
                for row in csv.DictReader(handle)
            ]

    def test_the_validation_file_is_worked_example_data_and_not_curve_data(self):
        """Why these points are a test target rather than a dataset.

        Anyone finding this file next to the curve data will reasonably assume it
        can be loaded as one. It cannot, and the reason is recorded here so the
        next person does not have to rediscover it from the numbers.
        """
        points = self._points()
        by_q = {}
        for tdv, q, _cdv in points:
            by_q.setdefault(q, []).append(tdv)
        # Nothing above q=5, so most of the published range is absent.
        assert max(by_q) == 5
        assert len(max(by_q.values())) <= 5
        # And the q=1 group is not monotonic in TDV, which a real curve read off
        # a graph necessarily is.
        q1 = [cdv for _tdv, q, cdv in sorted(
            (tdv, q, cdv) for tdv, q, cdv in points if q == 1)]
        assert q1 != sorted(q1), "expected the supplied q=1 group to be non-monotonic"

    def test_the_correction_curve_lands_near_the_worked_example(self):
        """Loose enough not to be a claim, tight enough to catch a broken curve."""
        repository = curves.repository()
        worst = 0.0
        for tdv, q, expected in self._points():
            got = repository.cdv(q).interpolate(tdv)
            worst = max(worst, abs(got - expected))
        # The curve is a placeholder, so it is allowed to be wrong, but not
        # unboundedly so. It currently overshoots by tens of points, which is
        # exactly the sort of error that turns a poor section into a failed one.
        assert worst < 70.0, (
            "the X3.26 placeholder is {} points from the standard's example; a real "
            "digitization of Fig. X3.26 is needed".format(round(worst, 1))
        )

    @pytest.mark.xfail(
        strict=True,
        reason="Fig. X3.26 is still an MDPI placeholder fit. This passes when the "
               "correction curve is digitized from the published figure; strict so "
               "that landing a real digitization fails loudly here, which is the "
               "moment to delete the placeholder warning.",
    )
    def test_the_correction_curve_still_fails_the_worked_example(self):
        """The gap, asserted so it cannot quietly grow.

        This test is expected to fail until Fig. X3.26 is digitized. When it does
        pass, this is the assertion to delete -- that is the signal the correction
        curve is no longer a placeholder.
        """
        repository = curves.repository()
        failures = []
        for tdv, q, expected in self._points():
            got = repository.cdv(q).interpolate(tdv)
            if abs(got - expected) > 5.0:
                failures.append("tdv={} q={}: curve {} vs example {}".format(
                    tdv, q, round(got, 1), expected))
        assert not failures, (
            "X3.26 is still a placeholder; {} of {} worked-example points are more "
            "than 5 points out. First few: {}".format(
                len(failures), len(self._points()), failures[:4]
            )
        )


class TestRepository:
    def test_an_unknown_distress_is_an_error_not_a_default(self):
        with pytest.raises(CurveNotFound):
            curves.repository().get("meltdown", "M")

    def test_the_error_lists_the_severities_that_do_exist(self):
        with pytest.raises(CurveNotFound) as excinfo:
            curves.repository().get("rutting", "X")
        assert "L" in str(excinfo.value) and "M" in str(excinfo.value)

    def test_every_mapped_distress_resolves_to_a_real_curve(self):
        """A distress the code claims to price must actually have a curve.

        A key pointing at a curve the dataset does not hold would fail at the
        first pothole on a real survey rather than here.
        """
        repository = curves.repository()
        for dtype, key in ASTMD6433PCI.CURVE_KEYS.items():
            for severity in ("L", "M", "H"):
                assert repository.has(key, severity), (dtype, key, severity)

    def test_a_q_beyond_the_dataset_is_refused_not_clamped(self):
        """A q with no published curve has to be reported, not approximated.

        This used to clamp, so `cdv(99)` returned the q=10 curve and `cdv(0)`
        returned the q=1 one. That returns a plausible number for a curve the
        standard does not provide, and nothing downstream can tell it from a
        real lookup.

        It is not hypothetical: the standard's own worked example needs the
        q=8 curve at TDV 104.7, and a dataset carrying only q=1..7 would have
        answered it with the q=7 curve.
        """
        repository = curves.repository()
        highest = repository.highest_cdv_q()

        assert repository.cdv(highest).severity == "q={}".format(highest)

        with pytest.raises(CurveNotFound):
            repository.cdv(highest + 1)
        with pytest.raises(CurveNotFound):
            repository.cdv(99)

    def test_q_below_one_is_refused(self):
        """9.5.5.2 defines q as a count, so it cannot be zero.

        The correction loop never asks for one -- 9.5.1 uses the total directly
        when at most one deduct exceeds 2.0 -- so this is a guard on the
        repository rather than a case the caller produces.
        """
        repository = curves.repository()
        with pytest.raises(CurveNotFound):
            repository.cdv(0)
        with pytest.raises(CurveNotFound):
            repository.cdv(-1)

    def test_a_dataset_with_curves_loads_and_answers(self):
        raw = {
            "standard": "ASTM D6433", "edition": "D6433-99",
            "distress_curves": [{
                "distress": "thing", "figure": "X9.1", "x_scale": "log10",
                "curves": {"L": [{"density": 1.0, "dv": 5.0},
                                  {"density": 10.0, "dv": 15.0}]},
            }],
            "cdv_curves": {"1": {"figure": "X9.2", "x_scale": "linear",
                                 "points": [{"tdv": 0.0, "cdv": 0.0},
                                            {"tdv": 10.0, "cdv": 8.0}]}},
        }
        import tempfile
        import os
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                         encoding="utf-8") as handle:
            json.dump(raw, handle)
            path = handle.name
        try:
            repository = CurveRepository.load(path)
            # Checked against the log-space rule the standard's curves are drawn
            # on: u = log10(density), then a straight line in u. Computed here
            # rather than hard-coded so the expectation cannot drift from it.
            u1, u, u2 = math.log10(1.0), math.log10(3.0), math.log10(10.0)
            expected = 5.0 + (u - u1) / (u2 - u1) * (15.0 - 5.0)
            assert repository.get("thing", "L").interpolate(3.0) == pytest.approx(
                expected, abs=0.01
            )
            # The CDV curve's axis is a sum, not a density, so it stays linear.
            assert repository.cdv(1).interpolate(5.0) == pytest.approx(4.0, abs=0.1)
        finally:
            os.unlink(path)
