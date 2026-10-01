"""Deduct-value curves as reference data, not as formulas in code.

A deduct-value curve is a table of points read off a published graph. Writing it
as a polynomial in the source is wrong in a way that is easy to miss: the curve
is a lookup, the digitization is data, and the two have different provenance.
This module keeps them apart, so a curve can be replaced with a measured read of
the standard without touching anything that calls it.

Three things this enforces, because each has already been got wrong here:

* **The density axis is not linear.** ASTM's distress curves are drawn on a
  log density axis, so interpolating against raw density bends the curve the
  wrong way. Interpolation happens in log10(density), and the axis each curve
  was digitized on is recorded in the data rather than assumed.
* **No silent extrapolation.** A density outside the digitized range is an
  error, not a number. A lookup table extended past its data invents a value
  that then feeds a PCI, and nothing downstream can tell it from a measured one.
* **Every value carries its provenance.** Each curve names its standard, edition
  and figure, and whether it is an authoritative digitization or a placeholder.
  A deduct value that cannot say where it came from is not auditable.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Where the curve dataset lives. Kept beside this module so the data travels
#: with the code that reads it.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATASET_PATH = os.path.join(DATA_DIR, "curves_d6433_99.json")

#: The edition pinned by the dataset. Values from another edition of D6433 are
#: not interchangeable with these, so a lookup for a different edition is a miss
#: rather than a silent substitution.
STANDARD = "ASTM D6433"

#: Calculation authority: the edition whose clause text the procedure in `pci.py`
#: implements. Clause 9.6.2 (the multi-severity PCI correction) exists only from
#: D6433-20 onward, so a -99 procedure is demonstrably not the one being followed.
EDITION = "D6433-24"

#: The edition the curves in the dataset were actually digitized from. Kept
#: separate from `EDITION` on purpose, because the two are currently different
#: and collapsing them would hide it.
#:
#: The procedure clauses were compared clause by clause between the supplied
#: -24 text and the -99 text and are identical, apart from 9.6.2 being new. The
#: deduct-value *figures* are a different matter: X3.26 in -99 became X3.27 plus
#: a split of weathering from raveling in -24, so the curves may well differ and
#: that cannot be checked, because the figures are raster images. Until someone
#: reads them from the printed standard, every lookup mixes a -24 procedure with
#: -99 curve data, and a report has to say so.
CURVE_DATA_EDITION = "D6433-99"


#: Total deduct value in the standard's own worked example (9.5.5.1: the eight
#: retained deduct values of Fig. 6 sum to 104.7). Held as a named constant
#: because it is the smallest input the standard itself requires the correction
#: curves to answer, and a dataset whose curves stop short of it cannot
#: reproduce the one complete calculation the standard publishes.
WORKED_EXAMPLE_TDV = 104.7

#: q in that same example: eight of the eight retained deduct values exceed 2.0
#: (9.5.5.2), so it needs a q=8 curve read at TDV 104.7.
WORKED_EXAMPLE_Q = 8

#: TDV at which the q=1 answer is defined by 9.5.1 alone. When only one deduct
#: value exceeds 2.0 the total *is* the maximum corrected deduct value and no
#: curve is consulted, so there is no q=1 curve to be digitized and none is
#: published in either edition. A dataset carrying one is carrying a fit, not a
#: reading -- and the fit that shipped was both 41.5 too high at TDV 58.5 and
#: byte-identical to the q=2 curve.
#:
#: `ASTMD6433PCI._cdv_from_curve_with_provenance` short-circuits q <= 1 to the
#: total, so this curve is never read. It is called out here because its
#: presence in the dataset is itself evidence about how the data was produced.
Q_ONE_NEEDS_NO_CURVE = True


class CurveError(Exception):
    """Base class for curve lookup failures."""


class CurveNotFound(CurveError):
    """No curve is published for that distress and severity."""


class CurveRangeExceeded(CurveNotFound):
    """The requested density falls outside the digitized range of the curve.

    Raised instead of extrapolating. The reported bounds are part of the error
    because the usual cause is a detection extending past what the graph covers.

    Subclasses `CurveNotFound` because both mean "this lookup has no defensible
    answer from the data we hold", and callers already handle that case by
    falling back to a reading they can defend. It was a sibling of
    `CurveNotFound`, so `except CurveNotFound` did not catch it and a section
    whose total deduct value ran past the end of a curve took the whole request
    down instead of degrading the way the rest of the code degrades.
    """

    def __init__(self, distress: str, severity: str, density: float,
                 low: float, high: float) -> None:
        super().__init__(
            "Density {:.4g}% is outside the digitized range for {} {} "
            "({:.4g}% to {:.4g}%). Refusing to extrapolate.".format(
                density, distress, severity, low, high
            )
        )
        self.distress = distress
        self.severity = severity
        self.density = density
        self.low = low
        self.high = high


@dataclass(frozen=True)
class CurvePoint:
    density_percent: float
    deduct_value: float


@dataclass
class Curve:
    """One digitized severity curve, with the provenance of the digitization.

    `provisional` is the important field. A curve built from a placeholder fit
    is usable -- it keeps the pipeline working and it is better than nothing --
    but it is not the standard, and every value it produces is tagged so a
    report cannot present it as an ASTM figure.
    """

    distress: str
    severity: str
    figure: str
    points: List[CurvePoint]
    x_scale: str = "log10"
    standard: str = STANDARD
    edition: str = EDITION
    #: How the points were obtained: "digitized" when read off the published
    #: graph, otherwise whatever placeholder source produced them.
    method: str = "digitized"
    provisional: bool = False
    note: str = ""
    #: Set when the curve has no severity split, as polished aggregate has none.
    severities: Tuple[str, ...] = ()

    @property
    def low_density(self) -> float:
        return min(p.density_percent for p in self.points)

    @property
    def high_density(self) -> float:
        return max(p.density_percent for p in self.points)

    def provenance(self, density: float, deduct_value: float) -> Dict:
        """The audit record for one lookup (see the traceability requirement)."""
        return {
            "standard": self.standard,
            "edition": self.edition,
            "distress": self.distress,
            "severity": self.severity,
            "figure": self.figure,
            "density_percent": round(density, 4),
            "dv": round(deduct_value, 2),
            "interpolation": "log10" if self.x_scale == "log10" else "linear",
            "method": self.method,
            "provisional": self.provisional,
        }

    def _u(self, density: float) -> float:
        if self.x_scale != "log10":
            return density
        if density <= 0:
            raise CurveRangeExceeded(
                self.distress, self.severity, density,
                self.low_density, self.high_density,
            )
        return math.log10(density)

    def interpolate(self, density: float) -> float:
        """Deduct value at `density` percent, interpolated in log10 density.

        Below the digitized range the value is 0: a curve that starts at some
        small density and is asked about less than none of it is describing an
        absence, and the standard's curves all pass through the origin. Above
        the range this raises rather than continuing the line, because past the
        end of the published graph there is no measured value to continue from.
        """
        if density <= 0:
            return 0.0
        if density < self.low_density:
            return 0.0
        if density > self.high_density:
            raise CurveRangeExceeded(
                self.distress, self.severity, density,
                self.low_density, self.high_density,
            )

        points = self.points
        u = self._u(density)
        if u <= self._u(points[0].density_percent):
            return points[0].deduct_value
        if u >= self._u(points[-1].density_percent):
            return points[-1].deduct_value

        for left, right in zip(points, points[1:]):
            u_left = self._u(left.density_percent)
            u_right = self._u(right.density_percent)
            if u_left <= u <= u_right:
                if u_right == u_left:
                    return right.deduct_value
                fraction = (u - u_left) / (u_right - u_left)
                return left.deduct_value + fraction * (
                    right.deduct_value - left.deduct_value
                )
        return points[-1].deduct_value


class CurveRepository:
    """Every curve for one edition of the standard, addressable by type/severity."""

    def __init__(self, curves: Dict[Tuple[str, str], Curve], cdv_curves: Dict[int, Curve],
                 edition: str = EDITION) -> None:
        self._curves = curves
        self._cdv = cdv_curves
        self.edition = edition

    @classmethod
    def load(cls, path: str = DATASET_PATH) -> "CurveRepository":
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        curves: Dict[Tuple[str, str], Curve] = {}
        for entry in raw.get("distress_curves", []):
            severities = entry.get("severities") or ["L", "M", "H"]
            for severity, points in entry["curves"].items():
                curves[(entry["distress"], severity)] = Curve(
                    distress=entry["distress"],
                    severity=severity,
                    figure=entry.get("figure", ""),
                    points=[CurvePoint(float(p["density"]), float(p["dv"]))
                            for p in points],
                    x_scale=entry.get("x_scale", "log10"),
                    standard=raw.get("standard", STANDARD),
                    edition=raw.get("edition", EDITION),
                    method=entry.get("method", "digitized"),
                    provisional=bool(entry.get("provisional", False)),
                    note=entry.get("note", ""),
                    severities=tuple(severities),
                )
        cdv: Dict[int, Curve] = {}
        for q, entry in raw.get("cdv_curves", {}).items():
            cdv[int(q)] = Curve(
                distress="total_deduct_value",
                severity="q={}".format(q),
                figure=entry.get("figure", ""),
                points=[CurvePoint(float(p["tdv"]), float(p["cdv"]))
                        for p in entry["points"]],
                # The correction curve's horizontal axis is the total deduct
                # value, which is a linear sum, not a density. It stays linear.
                x_scale="linear",
                standard=raw.get("standard", STANDARD),
                edition=raw.get("edition", EDITION),
                method=entry.get("method", "digitized"),
                provisional=bool(entry.get("provisional", False)),
                note=entry.get("note", ""),
            )
        return cls(curves, cdv, raw.get("edition", EDITION))

    def get(self, distress: str, severity: str) -> Curve:
        try:
            return self._curves[(distress, severity)]
        except KeyError:
            raise CurveNotFound(
                "No {} curve for {} at severity {!r}. Published severities for "
                "this distress: {}.".format(
                    self.edition, distress, severity,
                    ", ".join(sorted(self.severities_for(distress))) or "none",
                )
            )

    def severities_for(self, distress: str) -> List[str]:
        return [s for (d, s) in self._curves if d == distress]

    def cdv(self, q: int) -> Curve:
        """The corrected-deduct-value curve for q values above 2.0 (Fig. X3.27).

        9.5.5.2 defines q as a count of deduct values greater than 2.0, so it is
        whatever the section happens to contain, not something bounded by this
        dataset. 9.5.3 caps m at ten, which bounds q at ten in practice.

        A q above the highest published curve used to be clamped down to it,
        which returned a plausible number for a curve that does not exist. The
        standard's own worked example sits at q=8, and a dataset carrying only
        q=1..7 would have quietly answered that with the q=7 curve. Refusing is
        the same choice `interpolate` makes past the end of a curve: a missing
        value is reported rather than invented.
        """
        if q < 1:
            raise CurveNotFound(
                "q is a count of deduct values above 2.0 (9.5.5.2) and cannot "
                "be below 1; got {!r}.".format(q)
            )
        highest = max(self._cdv)
        if q > highest:
            raise CurveNotFound(
                "No correction curve published for q={}. This dataset carries "
                "q={}..{}. Reading the q={} curve instead would return a value "
                "for a curve the standard does not provide.".format(
                    q, min(self._cdv), highest, highest
                )
            )
        return self._cdv[q]

    def highest_cdv_q(self) -> int:
        """The largest q this dataset has a correction curve for."""
        return max(self._cdv)

    def self_check(self) -> List[str]:
        """Structural faults in the CDV curves, as readable sentences.

        These are properties of the data, not of any lookup, so they are checked
        once at load rather than being rediscovered as a wrong PCI. Each one is
        a claim the dataset makes that the standard contradicts:

        * **A q=1 curve should not exist.** 9.5.1 answers q=1 with the total
          deduct value and consults no curve, so there is nothing for a q=1
          entry to say. Its presence means it was fitted rather than read.
        * **q=1 must not equal q=2.** They are different rules: one applies no
          correction, the other does. Identical data means one was copied onto
          the other.
        * **The curves must reach the standard's own worked example.** A dataset
          whose CDV curves stop below TDV 104.7 cannot answer the one complete
          calculation the standard publishes, and 9.5.5.3 has nothing to read
          there.

        Returns a list of sentences, empty when the data passes. Callers log
        these; none of them is fatal on its own, because the procedure is sound
        and the mute in `app/core/config.py` is what keeps the number unpublished
        while these are outstanding.
        """
        problems: List[str] = []
        if not self._cdv:
            return ["No CDV correction curves are loaded at all."]

        if 1 in self._cdv and Q_ONE_NEEDS_NO_CURVE:
            problems.append(
                "A q=1 correction curve is present, but 9.5.1 defines the q=1 "
                "answer as the total deduct value itself and consults no curve, "
                "so this entry is a fit rather than a reading and is never used."
            )

        if 1 in self._cdv and 2 in self._cdv:
            one = [(p.density_percent, p.deduct_value) for p in self._cdv[1].points]
            two = [(p.density_percent, p.deduct_value) for p in self._cdv[2].points]
            if one == two:
                problems.append(
                    "The q=1 and q=2 CDV curves are identical. 9.5.1 applies no "
                    "correction at q=1 while q=2 does, so they cannot be the "
                    "same data."
                )

        short = [
            q for q, curve in sorted(self._cdv.items())
            if curve.high_density < WORKED_EXAMPLE_TDV
        ]
        if short:
            problems.append(
                "CDV correction curves for q={} stop at TDV {:.1f}, below the "
                "standard's own worked example of {:.1f} (9.5.5.1), so a section "
                "matching that example has no published value to read and "
                "saturates instead.".format(
                    ", ".join(str(q) for q in short),
                    max(self._cdv[q].high_density for q in short),
                    WORKED_EXAMPLE_TDV,
                )
            )
        return problems

    def has(self, distress: str, severity: str) -> bool:
        return (distress, severity) in self._curves

    def provisional_flags(self) -> Dict[str, bool]:
        return {d: self._curves[(d, s)].provisional
                for (d, s) in sorted(self._curves)}


_REPOSITORY: Optional[CurveRepository] = None


def repository() -> CurveRepository:
    """The shared repository, loaded once."""
    global _REPOSITORY
    if _REPOSITORY is None:
        _REPOSITORY = CurveRepository.load()
        provisional = [d for d, flag in _REPOSITORY.provisional_flags().items() if flag]
        if provisional:
            logger.warning(
                "ASTM %s deduct-value curves for %s are PLACEHOLDER fits, not "
                "digitized from the published graphs. Every PCI they produce is "
                "provisional. Affected distresses: %s.",
                _REPOSITORY.edition, STANDARD, ", ".join(sorted(set(provisional))),
            )
        if _REPOSITORY.edition != EDITION:
            logger.warning(
                "Curve data is %s but the procedure is %s. The deduct-value "
                "figures are not known to be the same in both editions, so every "
                "PCI mixes editions until the curves are re-read from the "
                "printed standard.",
                _REPOSITORY.edition, EDITION,
            )
        for problem in _REPOSITORY.self_check():
            logger.warning("CDV correction-curve fault: %s", problem)
    return _REPOSITORY
