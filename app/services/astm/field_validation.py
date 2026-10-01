"""Checking the index against real road ratings, rather than against the standard.

The deduct-value curves are placeholder fits, so the PCI is not the ASTM number
and never will be without the printed figures. That is acceptable for screening
because screening does not need the absolute value -- it needs the index to
*rank* sections the way the people judging those roads rank them. This module is
how that gets checked, and how a local vocabulary gets attached to the result.

Two things it refuses to do, because both are easy ways to get a reassuring
number that means nothing:

* **It will not report a correlation from a sample too small to support one.**
  Rank correlation over a handful of sections swings arbitrarily, and a p-value
  beside it looks like evidence without being any.
* **It will not fit band thresholds on the same sections it scores.** That is a
  training result reported as a validation result. Thresholds are fitted on one
  split and evaluated on another.

On the statistic: MRH and GHA publish *bands* -- Good, Fair, Poor -- not
rankings. That is an ordinal categorical variable, so the headline is quadratic
weighted kappa over the confusion matrix, which knows that a Good/Fair
disagreement is smaller than a Good/Poor one. Ranking the bands into 1, 2, 3 and
correlating those would discard exactly that information, and inventing a unique
rank per section would claim a precision the ground truth does not have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

#: Ghana's road condition vocabulary, worst to best. MRH reports the national
#: network as a condition mix in these three terms.
GHANA_BANDS = ("Poor", "Fair", "Good")

#: Below this many sections no correlation coefficient is reported. Rank
#: correlation over a small sample is dominated by which sections happen to be
#: in it, and the coefficient moves by tenths when one section changes.
MIN_SECTIONS_FOR_CORRELATION = 30


# --------------------------------------------------------------------------
# statistics, implemented here so nothing new has to be depended on
# --------------------------------------------------------------------------

def _average_ranks(values: Sequence[float]) -> List[float]:
    """Ranks with ties averaged, which is what a rank correlation expects."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rho, as the Pearson correlation of the two rank vectors."""
    if len(x) != len(y):
        raise ValueError("inputs must be the same length")
    if len(x) < 2:
        return float("nan")
    rx, ry = _average_ranks(x), _average_ranks(y)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (
        sum((a - mx) ** 2 for a in rx) ** 0.5
        * sum((b - my) ** 2 for b in ry) ** 0.5
    )
    return num / den if den else float("nan")


def quadratic_weighted_kappa(truth: Sequence[str], predicted: Sequence[str],
                             bands: Sequence[str] = GHANA_BANDS) -> float:
    """Cohen's kappa with quadratic weights over the band confusion matrix.

    Quadratic weighting costs a Good/Poor disagreement far more than a
    Good/Fair one, which is the ordering the bands actually have. Unweighted
    kappa would treat them as equally wrong.
    """
    if len(truth) != len(predicted):
        raise ValueError("inputs must be the same length")
    if not truth:
        return float("nan")
    index = {b: i for i, b in enumerate(bands)}
    k = len(bands)
    observed = [[0.0] * k for _ in range(k)]
    for t, p in zip(truth, predicted):
        observed[index[t]][index[p]] += 1.0
    n = float(len(truth))
    for row in observed:
        for j in range(k):
            row[j] /= n

    row_marginals = [sum(row) for row in observed]
    col_marginals = [sum(observed[i][j] for i in range(k)) for j in range(k)]
    expected = [[row_marginals[i] * col_marginals[j] for j in range(k)]
                for i in range(k)]

    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = ((i - j) ** 2) / ((k - 1) ** 2)
            num += w * observed[i][j]
            den += w * expected[i][j]
    if den == 0.0:
        return float("nan")
    return 1.0 - (num / den)


def contingency(truth: Sequence[str], predicted: Sequence[str],
                bands: Sequence[str] = GHANA_BANDS) -> List[List[int]]:
    """Counts as a bands x bands table, worst row to best column."""
    index = {b: i for i, b in enumerate(bands)}
    table = [[0] * len(bands) for _ in bands]
    for t, p in zip(truth, predicted):
        table[index[t]][index[p]] += 1
    return table


# --------------------------------------------------------------------------
# band thresholds
# --------------------------------------------------------------------------

def fit_band_thresholds(index_values: Sequence[float],
                        truth_bands: Sequence[str],
                        vocabulary: Sequence[str] = GHANA_BANDS,
                        grid: Optional[Sequence[float]] = None
                        ) -> Tuple[float, float]:
    """Two cut points mapping the continuous index onto the three bands.

    Fitted by maximising quadratic weighted kappa against `truth_bands`, so the
    split follows the evidence rather than being chosen for looking tidy.
    Returns the two cut points, low-to-high.
    """
    if grid is None:
        grid = sorted(set(index_values))
    if len(grid) < 3:
        raise ValueError("need at least three distinct index values to fit two cuts")

    best = (-float("inf"), float(grid[0]), float(grid[-1]))
    for low in grid:
        for high in grid:
            if high <= low:
                continue
            predicted = apply_band_thresholds(index_values, low, high, vocabulary)
            kappa = quadratic_weighted_kappa(truth_bands, predicted, vocabulary)
            if kappa == kappa and kappa > best[0]:   # skip NaN
                best = (kappa, low, high)
    return best[1], best[2]


def apply_band_thresholds(index_values: Sequence[float], low: float,
                          high: float,
                          vocabulary: Sequence[str] = GHANA_BANDS) -> List[str]:
    return [vocabulary[0] if v < low else (vocabulary[1] if v < high else vocabulary[2])
            for v in index_values]


def borderline(values: Sequence[float], low: float, high: float,
               tolerance: float = 5.0) -> List[bool]:
    """Whether a value sits close enough to a cut to flip on a small change.

    A hard threshold turns a difference of 0.1 in the index into a whole band.
    On a screening output that manufactures disagreements that the underlying
    measurement cannot support, so near-boundary sections are flagged rather
    than silently assigned.
    """
    return [abs(v - low) <= tolerance or abs(v - high) <= tolerance
            for v in values]


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------

@dataclass
class ValidationReport:
    sections: int
    bands: Tuple[str, ...]
    kappa: Optional[float] = None
    rho: Optional[float] = None
    n_truth_bands: int = 0
    recall: Dict[str, Optional[float]] = field(default_factory=dict)
    table: List[List[int]] = field(default_factory=list)
    thresholds: Optional[Tuple[float, float]] = None
    borderline_sections: int = 0
    min_sections: int = MIN_SECTIONS_FOR_CORRELATION
    notes: List[str] = field(default_factory=list)

    @property
    def conclusive(self) -> bool:
        return (self.kappa is not None
                and self.sections >= self.min_sections
                and self.n_truth_bands >= 2)

    def verdict(self) -> str:
        if not self.conclusive:
            return ("INCONCLUSIVE - " + "; ".join(self.notes))
        parts = ["kappa={:.3f}".format(self.kappa)]
        if self.rho is not None:
            parts.append("rho={:.3f}".format(self.rho))
        if self.thresholds:
            parts.append("cuts={:.1f}/{:.1f}".format(*self.thresholds))
        return "  ".join(parts)


def validate(index_values: Sequence[float], truth_bands: Sequence[str],
             thresholds: Optional[Tuple[float, float]] = None,
             bands: Sequence[str] = GHANA_BANDS,
             min_sections: int = MIN_SECTIONS_FOR_CORRELATION
             ) -> ValidationReport:
    """Score the index against field ratings and apply the local vocabulary.

    `thresholds` are fitted on `index_values`/`truth_bands` when not supplied.
    Either way the caller is responsible for having held the scored sections
    out; `validate` cannot tell a training pair from a validation pair.
    """
    if len(index_values) != len(truth_bands):
        raise ValueError("index_values and truth_bands must be the same length")

    report = ValidationReport(sections=len(index_values), bands=tuple(bands),
                              min_sections=min_sections)
    present = [b for b in bands if any(t == b for t in truth_bands)]
    report.n_truth_bands = len(present)

    if report.sections < min_sections:
        report.notes.append(
            "only {} section(s); {} needed for a correlation to mean anything"
            .format(report.sections, min_sections))
    if report.n_truth_bands < 2:
        report.notes.append(
            "ground truth covers {} band(s); agreement cannot be assessed "
            "against a single class".format(report.n_truth_bands))
    if not _preconditions_met(report):
        return report

    if thresholds is None:
        thresholds = fit_band_thresholds(index_values, truth_bands, vocabulary=bands)
    report.thresholds = thresholds

    predicted = apply_band_thresholds(index_values, *thresholds, vocabulary=bands)
    report.table = contingency(truth_bands, predicted, bands)
    report.kappa = quadratic_weighted_kappa(truth_bands, predicted, bands)
    report.rho = spearman(
        [bands.index(b) for b in truth_bands], list(index_values))

    for i, band in enumerate(bands):
        actual = report.table[i][i]
        total = sum(report.table[i])
        report.recall[band] = (actual / total) if total else None

    report.borderline_sections = sum(
        borderline(index_values, *thresholds))
    if report.borderline_sections:
        report.notes.append(
            "{} of {} section(s) sit within 5 points of a band cut"
            .format(report.borderline_sections, report.sections))
    return report


def _preconditions_met(report: ValidationReport) -> bool:
    """Enough to attempt a coefficient at all.

    Deliberately does not consult `conclusive`: that property describes a
    finished report, and asking it before `kappa` has been computed would make
    it refuse every input, including good ones.
    """
    return (report.sections >= report.min_sections
            and report.n_truth_bands >= 2)