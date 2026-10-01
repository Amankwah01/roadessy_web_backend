"""Structural faults in the CDV curve data, checked at load.

The CDV defect in this project was invisible for a long time because every
lookup returned a plausible number. The q=1 curve returned 57.43 where 9.5.1
requires 39.10, and the domain stopped below the standard's own worked example,
and neither raised: the mute in `app/core/config.py` is what kept them from
reaching a report, and it is a switch someone can flip.

`CurveRepository.self_check` is the belt to that braces. Each check states a
claim the dataset makes that the standard contradicts, so the defects are
reported by loading the data rather than discovered by trusting a PCI.
"""

import pytest

from app.services.astm import curves
from app.services.astm.curves import (
    WORKED_EXAMPLE_Q,
    WORKED_EXAMPLE_TDV,
    Curve,
    CurveNotFound,
    CurveRepository,
    CurvePoint,
)


def _cdv_curve(tdv_max=100.0, cdv_max=100.0, n=20):
    return Curve(
        distress="total_deduct_value",
        severity="q=test",
        figure="X3.27",
        x_scale="linear",
        points=[
            CurvePoint(tdv_max * i / (n - 1), cdv_max * i / (n - 1))
            for i in range(n)
        ],
    )


def _repo(cdv, distress=None):
    return CurveRepository(distress or {}, cdv, edition=curves.EDITION)


@pytest.fixture(scope="module")
def real_problems():
    """Faults reported by the dataset actually on disk."""
    return curves.repository().self_check()


class TestTheRealDatasetIsReportedAsFaulty:
    """These pin the known-bad state.

    They are written so that each one fails when a real Fig. X3.27 digitization
    lands and removes the fault it names. That is the point: a test that only
    ever passed would be a test that noticed nothing.
    """

    def test_a_q_one_curve_in_the_data_is_reported(self, real_problems):
        assert any("q=1 correction curve is present" in p for p in real_problems)

    def test_q_one_being_identical_to_q_two_is_reported(self, real_problems):
        assert any("identical" in p for p in real_problems)

    def test_a_domain_short_of_the_worked_example_is_reported(self, real_problems):
        assert any(
            "stop at TDV" in p and "104.7" in p for p in real_problems
        )


class TestTheWorkedExampleConstantsAreTheStandards:
    def test_the_tdv_is_the_sum_the_clause_publishes(self):
        """9.5.5.1 publishes 104.7 as the total of the eight retained values."""
        assert sum([25.1, 23.4, 17.9, 11.2, 7.9, 7.5, 6.9, 4.8]) == pytest.approx(
            WORKED_EXAMPLE_TDV, abs=0.05
        )

    def test_the_q_is_the_count_above_two(self):
        """9.5.5.2: eight of the eight retained values exceed 2.0."""
        values = [25.1, 23.4, 17.9, 11.2, 7.9, 7.5, 6.9, 4.8]
        assert sum(1 for v in values if v > 2.0) == WORKED_EXAMPLE_Q


class TestCleanDataPasses:
    def test_a_dataset_reaching_the_worked_example_reports_nothing(self):
        """A real digitization would be tabulated well past 104.7, and must not
        be reported as faulty. Otherwise the guard outlives the thing it guards."""
        repo = _repo({q: _cdv_curve(tdv_max=200.0) for q in range(2, 11)})
        assert repo.self_check() == []

    def test_distinct_q_one_and_q_two_pass(self):
        """If a digitization ever does carry a q=1 entry that differs from q=2,
        the identity fault clears. The q=1-presence fault would not, because
        9.5.1 still says no curve is consulted."""
        repo = _repo({
            1: _cdv_curve(cdv_max=50.0),
            2: _cdv_curve(cdv_max=100.0),
            **{q: _cdv_curve(tdv_max=200.0) for q in range(3, 11)},
        })
        problems = repo.self_check()
        assert not any("identical" in p for p in problems)


class TestEachFaultIsNamedPrecisely:
    def test_a_short_domain_names_the_curves_that_are_short(self):
        """The message has to say which q, or a reader has to go and look."""
        repo = _repo({
            2: _cdv_curve(tdv_max=200.0),
            3: _cdv_curve(tdv_max=200.0),
            8: _cdv_curve(tdv_max=100.0),
        })
        problems = repo.self_check()
        assert len(problems) == 1
        assert "q=8" in problems[0]
        assert "q=2" not in problems[0]

    def test_no_cdv_curves_at_all_is_reported_rather_than_passing(self):
        """An empty mapping has no fault to detect, and must not read as clean."""
        assert _repo({}).self_check() == [
            "No CDV correction curves are loaded at all."
        ]

    def test_the_q_one_message_explains_why_it_is_never_used(self):
        repo = _repo({1: _cdv_curve(tdv_max=200.0), 2: _cdv_curve(tdv_max=200.0)})
        message = repo.self_check()[0]
        assert "9.5.1" in message
        assert "never used" in message
