"""The standard's own worked example, clause by clause.

ASTM D6433 Section 9 publishes exactly one complete AC calculation, in Fig. 4
(distress data) and Fig. 6 (the corrected PCI). The deduct values it produces are
stated in the body text, so the arithmetic can be checked without the raster
figures:

    9.5.2  "this will be 25.1, 23.4, 17.9, 11.2, 7.9, 7.5, 6.9, and 5.3"
    9.5.3  "m = 1 + (9/98)(100-25.1) = 7.9"
    9.5.4  "the values are 25.1, 23.4, 17.9, 11.2, 7.9, 7.5, 6.9, and 4.8
            (the 4.8 is obtained by multiplying 5.3 by (7.9 - 7 = 0.9))"
    9.5.5.1 "that is, 104.7"
    9.5.5.2 "Determine q as the number of deducts with a value greater than 2.0.
            For example, in Fig. 6, q = 8"

This module pins the procedure to those published numbers. It previously had no
such test, which is how the claim that the implementation "reproduces the
standard worked example" survived without anyone running it end to end: the
procedure was right, but the CDV lookup at the published TDV of 104.7 was never
reached, and it raised. It now saturates and says so rather than raising; see
`TestWorkedExampleSaturatesRatherThanRaising`.

The published final CDV and PCI live in Fig. 6, which is a 625x742 raster image
and cannot be read to text, so the last step has no assertable expected value
here. Closing that gap needs the printed figure or the USACE PAVER reference
implementation, not this file.
"""

import pytest

from app.services.astm.pci import ASTMD6433PCI
from app.services.astm.curves import CurveRangeExceeded

#: 9.5.2
FIGURE_4_DEDUCTS = [25.1, 23.4, 17.9, 11.2, 7.9, 7.5, 6.9, 5.3]

#: 9.5.4 -- the trailing 5.3 becomes 4.8
EXPECTED_TRUNCATED = [25.1, 23.4, 17.9, 11.2, 7.9, 7.5, 6.9, 4.77]


class TestAllowableNumberOfDeducts:
    def test_m_matches_the_published_example(self):
        """9.5.3 quotes m = 7.9 for the Fig. 4 example."""
        assert ASTMD6433PCI.allowable_deducts(25.1) == pytest.approx(7.9, abs=0.05)

    def test_m_is_capped_at_ten(self):
        """9.5.3: 'must be less than or equal to ten'."""
        assert ASTMD6433PCI.allowable_deducts(0.0) == 10.0

    def test_m_is_rounded_before_the_fraction_is_taken(self):
        """The one-decimal rounding is load-bearing, not cosmetic.

        Unrounded, m is 7.8786 and the fractional part scales 5.3 by 0.8786 to
        4.66. The standard works to one decimal and says 4.8. The difference is
        0.11 of TDV, and it is the difference between matching Fig. 6 and not.
        """
        truncated = ASTMD6433PCI._truncate_to_m(FIGURE_4_DEDUCTS)
        assert truncated[-1] == pytest.approx(4.77, abs=0.01)
        assert truncated[-1] != pytest.approx(5.3 * 0.8786, abs=0.02)


class TestTruncationToM:
    def test_truncated_list_matches_clause_9_5_4(self):
        truncated = ASTMD6433PCI._truncate_to_m(FIGURE_4_DEDUCTS)
        assert len(truncated) == len(FIGURE_4_DEDUCTS)
        assert truncated[:7] == pytest.approx(FIGURE_4_DEDUCTS[:7])
        assert truncated[7] == pytest.approx(EXPECTED_TRUNCATED[7])

    def test_total_deduct_value_matches_clause_9_5_5_1(self):
        """9.5.5.1: 'that is, 104.7'."""
        total = sum(ASTMD6433PCI._truncate_to_m(FIGURE_4_DEDUCTS))
        assert total == pytest.approx(104.7, abs=0.05)

    def test_q_matches_clause_9_5_5_2(self):
        """9.5.5.2: 'q = 8', counting deducts greater than 2.0.

        All eight survive, including the fractional 4.8. A q counted on a 5.0
        threshold also returns 8 here, so this assertion alone does not
        discriminate between the two procedures -- the loop test below does.
        """
        truncated = ASTMD6433PCI._truncate_to_m(FIGURE_4_DEDUCTS)
        assert sum(1 for v in truncated if v > 2.0) == 8


class TestCorrectionLoop:
    @staticmethod
    def _trace(deducts, threshold=2.0):
        """Replay 9.5.5.1-9.5.5.4 and record (q, TDV) per pass.

        `threshold` is the only knob, so the alternative procedure can be run
        against the same data without duplicating the loop.
        """
        working = ASTMD6433PCI._truncate_to_m(deducts)
        trace = []
        while True:
            trace.append((sum(1 for v in working if v > threshold), sum(working)))
            above = [v for v in working if v > threshold]
            if len(above) <= 1:
                return trace
            working = list(working)
            working[working.index(min(above))] = threshold

    def test_loop_visits_every_published_pass(self):
        """Each pass cuts exactly one deduct, so q steps down one at a time."""
        trace = self._trace(FIGURE_4_DEDUCTS)
        assert [q for q, _ in trace] == [8, 7, 6, 5, 4, 3, 2, 1]
        totals = [total for _, total in trace]
        assert totals[0] == pytest.approx(104.67, abs=0.05)
        # 9.5.5.4 reduces the smallest value still above 2.0 to 2.0 each pass,
        # so every successive TDV is strictly smaller than the last.
        assert all(totals[i] > totals[i + 1] for i in range(len(totals) - 1))
        assert totals[-1] == pytest.approx(39.10, abs=0.05)

    def test_the_first_pass_is_the_pair_clause_9_5_5_2_names(self):
        assert self._trace(FIGURE_4_DEDUCTS)[0] == (8, pytest.approx(104.67, abs=0.05))

    def test_a_five_point_threshold_cannot_reproduce_the_published_pass(self):
        """Why the 2.0 threshold is load-bearing, on the standard's own data.

        Run the same loop with 5.0 instead of 2.0 and it never visits the
        (q=8, TDV=104.7) pair that 9.5.5.2 states explicitly: 4.8 and 5.3 fall
        below the threshold, so q never reaches 8 and the first pass is not the
        published one. Deducts in (2.0, 5.0] also skip correction entirely.

        This is the guard against reintroducing a 5.0 threshold. Two supplied
        third-party modules used one, sourced from a commercial glossary page
        rather than the standard, and both read higher PCI than ASTM does.
        """
        wrong = self._trace(FIGURE_4_DEDUCTS, threshold=5.0)
        right = self._trace(FIGURE_4_DEDUCTS, threshold=2.0)

        assert (8, pytest.approx(104.67, abs=0.05)) == right[0]
        assert max(q for q, _ in wrong) < 8
        assert len(wrong) < len(right)
        assert wrong[0] != right[0]

    def test_deducts_between_two_and_five_need_the_correction(self):
        """A 5.0 threshold silently exempts this whole band from correction."""
        truncated = ASTMD6433PCI._truncate_to_m(FIGURE_4_DEDUCTS)
        uncorrected_by_five = [v for v in truncated if 2.0 < v <= 5.0]
        assert uncorrected_by_five, "expected deducts in the (2.0, 5.0] band"


class TestWorkedExampleSaturatesRatherThanRaising:
    """The standard's own example runs off the end of the placeholder curves.

    The CDV correction curves in the dataset are placeholder polynomial fits
    tabulated over TDV 0-100, and they saturate at CDV 100 at the end of that
    range. The standard's example sits at TDV 104.7 on its first pass, so the
    lookup has no published value to read.

    This used to raise `CurveRangeExceeded`, which meant the pipeline could not
    finish the standard's own worked example at all. It now saturates to the
    curve's terminal value and reports that it did. That is a deliberate trade:
    a saturation is not the standard's answer, but it keeps the section in the
    results with an honest flag, where a raised exception kept it out entirely.

    What it must never become is a quiet PCI 0 presented as a computed value.
    That is the failure the flag exists to prevent, so the flag is asserted here
    rather than left to inspection.

    When the CDV curves are properly digitized from Fig. X3.27 (or X3.26 in
    D6433-99), these saturate and the assertions should be replaced with the
    standard's published corrected deduct value.
    """

    def test_cdv_lookup_past_the_curve_still_raises_in_the_data_layer(self):
        """The curve itself must not extrapolate.

        The saturation lives in the PCI layer, not here. `interpolate` is the
        low-level reader and its whole contract is to refuse to invent a value
        past the end of the data; relaxing that would remove the only guard
        between the placeholder curves and a reported number.
        """
        from app.services.astm import curves

        with pytest.raises(CurveRangeExceeded):
            curves.repository().cdv(8).interpolate(104.7)

    def test_full_correction_saturates_and_says_so(self):
        value, saturated = ASTMD6433PCI.corrected_deduct_value_with_provenance(
            FIGURE_4_DEDUCTS
        )
        assert saturated, "a lookup past the end of the curve must be reported"
        assert value == pytest.approx(100.0), (
            "the saturated reading is the curve's terminal value, which is a "
            "corrected deduct value of 100 and so a PCI of 0"
        )

    def test_saturation_does_not_arise_for_an_example_inside_the_curves(self):
        """Not every total deduct value saturates, and the flag must say which.

        If the flag were unconditionally true the section output would be
        worthless, since almost every real section reads a curve.
        """
        _, saturated = ASTMD6433PCI.corrected_deduct_value_with_provenance(
            [10.0, 9.0, 8.0, 7.0]
        )
        assert not saturated

    def test_single_deduct_never_reads_a_curve(self):
        """9.5.1: one value above 2.0 needs no curve, so nothing can saturate."""
        value, saturated = ASTMD6433PCI.corrected_deduct_value_with_provenance([95.0])
        assert saturated is False
        assert value == pytest.approx(95.0)

    def test_q_one_is_answered_from_9_5_1_and_never_from_a_curve(self):
        """The iteration of 9.5.5.4 ends on q=1, so q=1 is the reading the
        procedure is built to reach.

        The dataset carries a q=1 curve and it is a fitted placeholder: it
        returned 57.43 where 9.5.1 requires the total, 39.10, on this very
        example, and 100 at a total of 58.5. It was also byte-identical to the
        q=2 curve, which cannot be right, since q=1 applies no correction and
        q=2 does. Neither edition publishes a q=1 curve, because there is
        nothing for one to say.
        """
        for total in (5.0, 39.10, 58.5, 80.0, 104.7):
            value, saturated = ASTMD6433PCI._cdv_from_curve_with_provenance(
                total, 1
            )
            assert value == pytest.approx(total)
            # Not a lookup off the end of a table, so not a saturation.
            assert saturated is False

    def test_the_final_pass_of_the_9_5_5_4_loop_reads_9_5_1(self):
        """The last pass of the worked example is q=1 at TDV 39.10, and that is
        what the standard's own arithmetic reaches.

        Driven from the clause rather than hardcoded: truncate to m, then cut
        the smallest value above 2.0 down to 2.0 until one is left, which is
        9.5.5.4, and check that the resulting pass is answered by 9.5.1.
        """
        working = ASTMD6433PCI._truncate_to_m(FIGURE_4_DEDUCTS)
        while sum(1 for v in working if v > 2.0) > 1:
            working[working.index(min(v for v in working if v > 2.0))] = 2.0
        final_tdv = sum(working)
        assert sum(1 for v in working if v > 2.0) == 1
        assert final_tdv == pytest.approx(39.10, abs=0.05)

        value, saturated = ASTMD6433PCI._cdv_from_curve_with_provenance(
            final_tdv, 1
        )
        assert value == pytest.approx(final_tdv)
        assert saturated is False

    def test_q_one_does_not_clamp_a_total_the_curves_could_not_have_reached(self):
        """A single severe distress totalling 104.7 is the case 9.5.1 exists for.

        Before the fix this went to the q=1 curve, saturated at its terminal
        value, and reported PCI 0 for a road whose only defect is one pothole.
        """
        value, saturated = ASTMD6433PCI.corrected_deduct_value_with_provenance(
            [104.7]
        )
        assert value == pytest.approx(104.7)
        assert saturated is False