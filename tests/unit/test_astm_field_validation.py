"""The validation harness must not be able to produce a false reassurance.

Three ways a rank-correlation harness flatters itself, all of which showed up
in the version this replaced: a sample too small to support a coefficient, a
banded ground truth turned into a total order, and thresholds fitted on the
sections they are then scored against. Each one is pinned here.

The synthetic ground truth is built to be imperfect on purpose, so a harness
that reports success anyway is visibly wrong.
"""

import math

import pytest

from app.services.astm.field_validation import (
    GHANA_BANDS,
    MIN_SECTIONS_FOR_CORRELATION,
    ValidationReport,
    apply_band_thresholds,
    borderline,
    contingency,
    fit_band_thresholds,
    quadratic_weighted_kappa,
    spearman,
    validate,
)


# --------------------------------------------------------------------------
# the statistics
# --------------------------------------------------------------------------

class TestSpearman:
    def test_perfect_agreement_is_one(self):
        assert spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) == pytest.approx(1.0)

    def test_perfect_disagreement_is_minus_one(self):
        assert spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) == pytest.approx(-1.0)

    def test_ties_are_not_scored_as_a_perfect_ranking(self):
        """Two tied sections plus one high value.

        A tie is real information: the engine cannot separate two Poor
        sections from each other. Ranking them 1 and 2 by arbitrary index order
        would manufacture agreement that does not exist, so a tie must pull the
        coefficient below 1.0.
        """
        tied = spearman([1, 1, 3], [5, 7, 9])
        untied = spearman([1, 2, 3], [5, 7, 9])
        assert untied == pytest.approx(1.0)
        assert tied < untied
        assert tied == pytest.approx(0.8660254, abs=1e-6)

    def test_reversing_the_direction_flips_the_sign(self):
        forward = spearman([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        backward = spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
        assert forward * backward < 0


class TestWeightedKappa:
    def test_perfect_agreement_is_one(self):
        bands = list(GHANA_BANDS) * 4
        assert quadratic_weighted_kappa(bands, bands) == pytest.approx(1.0)

    def test_quadratic_weighting_costs_far_apart_bands_more(self):
        """Good vs Poor is a bigger error than Good vs Fair.

        Both versions below get three of four sections right, so they tie on
        plain accuracy. Only the weighting of the single mistake separates
        them. Discarding that distinction -- what happens if the ordinal bands
        are ranked into 1, 2, 3 -- is exactly what would let a neighbouring band
        slip through unscored.
        """
        truth = ["Good", "Fair", "Fair", "Fair"]
        near = ["Fair", "Fair", "Fair", "Fair"]     # Good misread as Fair
        far = ["Poor", "Fair", "Fair", "Fair"]      # Good misread as Poor
        assert sum(a == b for a, b in zip(truth, near)) == \
            sum(a == b for a, b in zip(truth, far)) == 3
        assert quadratic_weighted_kappa(truth, near) > \
            quadratic_weighted_kappa(truth, far)

    def test_a_single_class_is_not_scored(self):
        """Kappa divides by expected agreement.

        If every section carries one rating and the engine agrees with all of
        them, expected agreement is 1 and the coefficient is 0/0. Returning a
        number here would be a division that quietly produced nonsense, so
        this must come back undefined and be filtered out upstream.
        """
        assert math.isnan(
            quadratic_weighted_kappa(["Good"] * 5, ["Good"] * 5))

    def test_a_degenerate_split_is_zero_not_a_collapse(self):
        """The comparable case: perfect agreement is still not available.

        Both sides are constant but different, so nothing was agreed and
        nothing was expected. Zero is a real answer; nan is the undefined one.
        """
        assert quadratic_weighted_kappa(["Good"] * 5, ["Fair"] * 5) == 0.0

    def test_contingency_counts_every_section_once(self):
        table = contingency(["Good", "Fair", "Poor"], ["Fair", "Fair", "Poor"])
        assert sum(sum(row) for row in table) == 3


# --------------------------------------------------------------------------
# the guards
# --------------------------------------------------------------------------

class TestSmallSamplesAreRefused:
    def test_five_sections_produce_no_coefficient(self):
        """The case that would have printed SUCCESS on data built to give it.

        Five sections with a hand-assigned 1..5 rank and an index ascending in
        the same order correlate at exactly 1.0. That is arithmetic, not
        evidence, and the harness has to decline to say otherwise.
        """
        index_values = [18.4, 34.1, 56.8, 62.3, 88.5]
        bands = ["Poor", "Poor", "Fair", "Fair", "Good"]

        report = validate(index_values, bands)

        assert report.conclusive is False
        assert report.kappa is None
        assert report.rho is None
        assert any("needed" in note for note in report.notes)
        assert "INCONCLUSIVE" in report.verdict()

    def test_the_threshold_is_configurable_but_still_a_guard(self):
        report = validate([1, 2, 3], ["Poor", "Fair", "Good"], min_sections=3)
        assert report.sections == 3
        assert report.conclusive is True

    def test_a_single_ground_truth_band_is_refused(self):
        many = [10.0] * MIN_SECTIONS_FOR_CORRELATION
        report = validate(many, ["Good"] * MIN_SECTIONS_FOR_CORRELATION)
        assert report.conclusive is False
        assert any("single class" in note for note in report.notes)


class TestSyntheticDataIsScoredHonestly:
    """Deliberately imperfect: the engine gets two bands wrong and confuses one."""

    @pytest.fixture
    def sample(self):
        """Forty sections, plus two deliberate mistakes injected into the truth.

        Returned as `(values, ideal_bands, scored_bands)`. The distinction
        matters: `scored_bands` is what a field crew would actually record, and
        it is wrong twice.
        """
        values, ideal = [], []
        for i in range(40):
            value = 10.0 + i * 2.0
            if value < 45:
                ideal.append("Poor")
            elif value < 75:
                ideal.append("Fair")
            else:
                ideal.append("Good")
            values.append(value)
        scored = list(ideal)
        # Placed well inside a band, not next to a cut, so no choice of cut
        # can quietly repair them.
        scored[5] = "Good"      # value 20, a clearly poor section, over-rated
        scored[35] = "Fair"     # value 80, a clearly good section, under-rated
        return values, ideal, scored

    def test_a_nearly_perfect_engine_scores_well_against_held_out_cuts(
            self, sample):
        """Thresholds fitted elsewhere, then applied to the crew's ratings.

        `validate` fits when no thresholds are given, which is a training
        result and is near-perfect by construction. Passing cuts learned from a
        separate set is what makes this a validation result at all.
        """
        values, ideal, scored = sample
        cuts = fit_band_thresholds(values, ideal)

        report = validate(values, scored, thresholds=cuts)

        assert report.conclusive is True
        assert report.kappa > 0.8
        assert report.rho > 0.8

    def test_the_two_mistakes_appear_in_the_table(self, sample):
        """Against cuts the data did not choose, the errors stop hiding."""
        values, ideal, scored = sample
        cuts = fit_band_thresholds(values, ideal)

        report = validate(values, scored, thresholds=cuts)

        poor, fair, good = (GHANA_BANDS.index(b)
                            for b in ("Poor", "Fair", "Good"))
        # Rows are the crew's rating, columns the engine's.
        assert report.table[good][poor] == 1     # crew over-rated a Poor section
        assert report.table[fair][good] == 1    # crew under-rated a Good one

    def test_the_headline_number_stays_high_despite_the_errors(self, sample):
        """Why the contingency table cannot be skipped.

        Two misrated sections in forty barely move kappa. Reading only the
        coefficient would report a strong result while two specific calls the
        crew disagreed with stay buried in the table.
        """
        values, ideal, scored = sample
        cuts = fit_band_thresholds(values, ideal)

        report = validate(values, scored, thresholds=cuts)

        poor, good, fair = (GHANA_BANDS.index(b)
                            for b in ("Poor", "Good", "Fair"))
        errors = report.table[good][poor] + report.table[fair][good]
        assert errors == 2
        assert report.kappa > 0.85

    def test_a_fitted_run_reports_its_own_thresholds(self, sample):
        """Fitting inside `validate` is convenient and is still a training
        result. The report must show the cuts it used so a reader can tell."""
        values, _, scored = sample

        report = validate(values, scored)

        assert report.conclusive is True
        assert report.thresholds is not None
        assert apply_band_thresholds(values, *report.thresholds) is not None
        assert report.borderline_sections > 0

    def test_a_shuffled_engine_scores_badly(self, sample):
        """The check that proves the statistic can tell the difference."""
        values, ideal, scored = sample
        cuts = fit_band_thresholds(values, ideal)

        report = validate(list(reversed(values)), scored, thresholds=cuts)

        assert report.rho < 0.0
        assert report.kappa < 0.5


class TestRecallExposesAnUnpredictedBand:
    def test_a_band_the_engine_never_calls_is_visible(self):
        """A skewed network makes accuracy look good while Good is never used.

        Thirty Poor sections against ten Fair: an engine that calls everything
        Poor is right 75% of the time and never once recognises a fair road.
        Recall is the only number here that shows the gap.
        """
        bands = ["Poor"] * 30 + ["Fair"] * 10
        values = list(range(10, 50))

        # Cuts placed above every value: the engine calls the whole network
        # Poor. No fitting, so the result reflects only the predictions.
        report = validate(values, bands, thresholds=(1000.0, 2000.0))

        agreed = report.table[0][0]
        assert agreed / 40 == pytest.approx(0.75)
        assert report.recall["Poor"] == pytest.approx(1.0)
        assert report.recall["Fair"] == pytest.approx(0.0)
        assert report.recall["Good"] is None     # never present in the truth


# --------------------------------------------------------------------------
# thresholds
# --------------------------------------------------------------------------

class TestThresholdFitting:
    def test_the_fitted_cuts_reproduce_the_bands_they_were_fitted_to(self):
        """Not a claim about *where* the cuts land.

        With few sections many cut pairs score identically, so any assertion
        about the exact numbers pins an accident. What has to hold is that the
        chosen pair is good enough to separate the classes it was fitted on.
        """
        values = [10, 20, 30, 40, 60, 70, 80, 90]
        bands = ["Poor"] * 4 + ["Fair"] * 2 + ["Good"] * 2

        low, high = fit_band_thresholds(values, bands)

        assert low < high
        assert min(values) <= low and high <= max(values)
        assert apply_band_thresholds(values, low, high) == bands

    def test_the_cuts_are_not_returned_inside_the_wrong_gaps(self):
        """The fit is rewarded for kappa, so verify it is not picking a
        degenerate pair -- both cuts jammed into one gap would put every
        section in two bands and leave one empty."""
        values = [10, 20, 30, 40, 60, 70, 80, 90]
        bands = ["Poor"] * 4 + ["Fair"] * 2 + ["Good"] * 2

        low, high = fit_band_thresholds(values, bands)

        assigned = apply_band_thresholds(values, low, high)
        assert all(assigned.count(b) > 0 for b in ("Poor", "Fair", "Good"))

    def test_thresholds_must_be_supplied_to_score_held_out_sections(self):
        """Fitting and scoring on one set is a training result.

        `validate` fits when thresholds are absent, which is why the caller has
        to pass cuts learned elsewhere. This pins that the argument exists and
        is honoured, so a held-out evaluation is possible at all.
        """
        values = [10, 20, 30, 60, 70, 80]
        bands = ["Poor", "Poor", "Poor", "Fair", "Fair", "Good"]

        fitted = validate(values, bands)
        scored = validate(values, bands, thresholds=fitted.thresholds)

        assert fitted.thresholds == scored.thresholds

    def test_too_few_distinct_values_cannot_be_split(self):
        with pytest.raises(ValueError):
            fit_band_thresholds([5, 5, 5], ["Poor", "Fair", "Good"])

    def test_a_value_exactly_on_a_cut_goes_to_the_higher_band(self):
        assert apply_band_thresholds([40.0], 40.0, 70.0) == ["Fair"]
        assert apply_band_thresholds([39.9], 40.0, 70.0) == ["Poor"]


class TestBorderlineFlagging:
    def test_values_near_a_cut_are_flagged(self):
        assert borderline([38, 70], 40, 70) == [True, True]

    def test_values_well_inside_a_band_are_not(self):
        assert borderline([20, 55], 40, 70) == [False, False]

    def test_a_hard_threshold_alone_would_flip_on_a_tiny_change(self):
        """Why the flag exists: 0.1 of index should not move a whole band."""
        low, high = 40.0, 70.0
        assert apply_band_thresholds([39.95], low, high) != \
            apply_band_thresholds([40.05], low, high)
        assert any(borderline([39.95, 40.05], low, high))