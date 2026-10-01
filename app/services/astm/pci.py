"""D6433-derived PCI for asphalt concrete pavements.

Implements clauses 9.1-9.6 of ASTM D 6433-24, *Standard Practice for Pavement
Condition Index Surveys*, with the severity and measurement rules from Appendix
X1 and the deduct-value curves from the published figures.

**What this number is.** A provisional index for relative condition screening
and deterioration tracking. **What it is not.** An ASTM D6433 compliance value,
and it should not be presented as one. See `ASTMD6433PCI.LIMITATIONS`, which is
attached to every result so a reader does not need this file to see why.

The limits are properties of the method, not of any run, and better data alone
does not clear them. The most important is a difference in what is being
measured: clause 7 designs a random sample per section and 11.1 defines section
PCI over the randomly surveyed sample units, whereas a video pass is a census
of the entire section. Those are different statistics, and no improvement to the
curve data makes them comparable.

The second is the curves themselves. They are anchor points digitized from
rasterized figures, carrying roughly +-1 to 3 deduct-value units of error, and
the calculation edition (D6433-24) is not the edition they were digitized from
(D6433-99). Authoritative values need USACE PAVER/MicroPAVER or a printed copy.
See `app/services/astm/data/README.md`.

This module previously followed Ibragimov et al., *Automated Pavement Condition
Index Assessment with Deep Learning and Image Analysis*, Sensors 2024, 24, 2333.
That paper is a sound reference for the detection pipeline and a partial one for
the arithmetic, and three things had to change to follow the standard instead:

* **Edition.** The paper implements D6433-18. The supplied standard is D6433-24,
  which carries clauses 9.6.2 and Appendix X1.4 that the earlier editions do not.
* **Severity bands.** The paper rates crack width low below 5 mm and medium above
  it, which is neither the standard (below 10 mm low, 10 to under 75 mm medium,
  75 mm and over high, X1.14.2) nor AASHTO R85. A 30 mm crack is high under the
  paper's bands and medium under this standard's.
* **The 9.5 correction.** The paper sums deduct values, reads a CDV from its
  Figure 4, and subtracts. It has no allowable-number-of-deducts m, no truncation
  to m with a fractional part, no q as the count of deduct values above 2.0, and
  no iteration reducing the smallest such value to 2.0 until only one is left.
  That is 9.5.2 to 9.5.5, and it is the part that stops several defects from
  compounding. The paper also uses q1-q7 as a severity index, where ASTM's q is a
  count, so its correction-curve lookup is not the one the standard defines.

**This module is muted by default and is not publishing a PCI.** The code is
intact; the switch is at the pipeline boundary (`PavementConditionFusioneer`),
so this module can still be exercised directly by tests. Setting
`ASTM_PCI_ENABLED=true` in the environment re-enables the pipeline, but the
curve data has to be replaced before the number means anything -- in particular
the `cdv_curves` domain, which currently saturates below the TDV of the
standard's own worked example. See `app/core/config.py` for the full list.
"""

import logging
import math

from app.services.astm import curves
from app.services.astm.curves import CurveNotFound, CurveRangeExceeded

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class DistressType(Enum):
    """Distress types per AASHTO R85-18 and ASTM D6433-18"""
    ALLIGATOR = "alligator"
    PATCHING = "patching"
    LONGITUDINAL = "longitudinal"
    TRANSVERSE = "transverse"
    POTHOLES = "potholes"
    RAVELLING = "ravelling"
    RUTTING = "rutting"
    SHOVING = "shoving"
    BLEEDING = "bleeding"
    CORRUGATION = "corrugation"
    DEPRESSION = "depression"


class SeverityLevel(Enum):
    """Severity levels per AASHTO R85-18 / ASTM D6433"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PCIRating(Enum):
    """PCI rating categories per ASTM D6433-18 Table 2"""
    GOOD = (86, 100, "Good")
    SATISFACTORY = (71, 85, "Satisfactory")
    FAIR = (56, 70, "Fair")
    POOR = (41, 55, "Poor")
    VERY_POOR = (26, 40, "Very Poor")
    SERIOUS = (11, 25, "Serious")
    FAILED = (0, 10, "Failed")

    @classmethod
    def from_pci(cls, pci: float) -> str:
        for rating in cls:
            lo, hi, label = rating.value
            if lo <= pci <= hi:
                return label
        return "Failed"


@dataclass
class Distress:
    """Individual distress instance"""
    type: DistressType
    severity: SeverityLevel
    length_m: float = 0.0
    area_m2: float = 0.0
    width_mm: float = 0.0
    depth_mm: Optional[float] = None
    chainage_start_m: float = 0.0
    chainage_end_m: float = 0.0
    #: The severity an inspector assigned to a distress the standard does not
    #: rate on a measurable extent. See `ASTMD6433PCI.resolve_severity`.
    #:
    #: This is a separate field from `severity` because `severity` is derived
    #: from the measurement for most types and was being ignored for the rest.
    #: For bumps, corrugation, railroad crossings, shoving and swells, X1.4
    #: defines the level by how the vehicle rides, which no detection box can
    #: measure. Left as None, that judgement has not been made and the fallback
    #: to LOW is counted and reported rather than passed off as a rating.
    rated_severity: Optional[SeverityLevel] = None


@dataclass
class SectionMetrics:
    """Per 50m section metrics"""
    section_index: int
    start_m: float
    end_m: float
    area_m2: float
    distresses: List[Distress]
    #: Width of the lane the section covers. Needed to put a length-measured
    #: distress on the same footing as the area-measured ones; see
    #: `compute_density`.
    lane_width_m: float = 3.25


class ASTMD6433PCI:
    """
    ASTM D6433-18 PCI Calculator per MDPI Sensors 2024 paper (Ibragimov et al.)
    
    Steps:
    1. Identify distress type, severity, extent per section
    2. Calculate density = (distress amount / section area) × 100
    3. Get DV from ASTM curves (digitized polynomial)
    4. Sum DV to get TDV
    5. Apply CDV correction curve
    6. PCI = 100 - CDV
    """
    
    # Standard section length (ASTM uses 50m or 100m; paper uses 50m)
    SECTION_LENGTH_M = 50.0
    
    # Standard lane width (3.0-3.5m per paper)
    LANE_WIDTH_M = 3.25
    
    # Section area = length × width
    SECTION_AREA_M2 = SECTION_LENGTH_M * LANE_WIDTH_M
    
    # Distress type mapping from YOLO classes to ASTM D6433 types
    YOLO_TO_AASHTO = {
        0: DistressType.BLEEDING,         # bleeding
        1: DistressType.CORRUGATION,      # corrugation
        2: DistressType.LONGITUDINAL,     # crack (longitudinal/transverse)
        3: DistressType.PATCHING,         # patching
        4: DistressType.POTHOLES,         # pothole
        5: DistressType.RAVELLING,        # ravelling
        6: DistressType.RUTTING,          # rutting
        7: DistressType.SHOVING,          # shoving
        8: DistressType.DEPRESSION,       # subgrade failure -> depression
    }
    
    #: Deduct-value curves are reference data, never formulas. They live in
    #: `app/services/astm/curves.py` and the dataset beside it, and are read
    #: through that repository, so a curve can be replaced with a read of the
    #: published graph without touching this file.
    #:
    #: The polynomial table that used to be here was never in ASTM D 6433. It
    #: was a fit from an MDPI paper, evaluated on a linear density axis where the
    #: published curves are logarithmic, and carrying a constant term so that a
    #: distress with no measured extent still cost points. It has been removed.
    #: The curves are now anchor points digitized from the published figures, and
    #: they are marked provisional because that digitization is a first pass.
    #:
    #: Regenerate the dataset with `python scripts/build_astm_curves.py`.

    #: What this number is, and what it is not. Served beside every PCI so a reader
    #: does not have to know the history to see it.
    #:
    #: The three limits below are properties of the method, not of any run. They
    #: cannot be cleared by better data alone, and the first one cannot be cleared
    #: at all without changing how sections are surveyed.
    LIMITATIONS = {
        "use": "relative condition screening and deterioration tracking",
        "not": "an ASTM D6433 compliance value",
        "method": "ASTM D6433-24 clauses 9.1-9.6 with Appendix X1 severity rules",
        "curves": "provisional first-pass digitization of the D6433-99 figures",
        "reasons": [
            "Clause 7 designs a random sample per section and 11.1 defines "
            "section PCI over 'the randomly surveyed sample units'. A video "
            "pass is a census of the whole section, so this is a different "
            "statistic from the one the standard defines, and no curve "
            "digitisation makes the two comparable.",
            "The deduct-value curves are a first-pass digitization of "
            "rasterized figures, carrying roughly 1 to 3 deduct-value units "
            "of error, and the calculation edition (D6433-24) is not the "
            "edition they were digitized from (D6433-99). Authoritative "
            "values require USACE PAVER/MicroPAVER or a printed copy. "
            "Comparison between sections is sound because the error is "
            "systematic rather than random.",
            "Six of the twenty AC distresses are rated on condition rather "
            "than on a measurable extent, and X1.4 defines five more by how "
            "the vehicle rides. Neither is derivable from a detection box. "
            "Where no severity rating is supplied the count is still "
            "deducted at LOW, which understates the distress, so any run "
            "using these is flagged as provisional on severity.",
        ],
        "ratings_are": "ASTM's seven bands, not Ghana's Good/Fair/Poor mix",
    }

    #: Distresses the standard rates on condition and ride quality rather than
    #: on a measurable extent. See `get_severity_from_width` for why they carry
    #: no automatic severity.
    CONDITION_RATED_TYPES = frozenset({
        DistressType.ALLIGATOR,
        DistressType.PATCHING,
        DistressType.RAVELLING,
        DistressType.CORRUGATION,
        DistressType.SHOVING,
        DistressType.BLEEDING,
    })

    #: The subset of the above that X1.4 rates on ride quality rather than on
    #: surface condition. X1.4.1 to X1.4.5 name five: bumps and sags,
    #: corrugation, railroad crossings, shoving, and swells. Only corrugation
    #: and shoving are in the current distress catalog, so the other three are
    #: absent rather than handled; adding them to `DistressType` without adding
    #: their curves would make them fail at the curve lookup instead.
    #:
    #: X1.4.6 defines the three levels by what the vehicle does:
    #:
    #:   L  vibrations noticeable, no reduction in speed needed
    #:   M  vibrations significant, some reduction in speed for comfort/safety
    #:   H  vibrations excessive, considerable reduction for comfort/safety
    #:
    #: Nothing in a detection box distinguishes those. They are an inspector's
    #: judgement made driving the section at the speed limit (X1.4.7), so they
    #: arrive as an input rather than being derived, and a section that has them
    #: without one owes a judgement.
    RIDE_QUALITY_TYPES = frozenset({
        DistressType.CORRUGATION,
        DistressType.SHOVING,
    })
    
    #: Severity levels in increasing order. 9.6.2 folds a level into the next one
    #: up, so the ordering is part of the procedure rather than presentation.
    SEVERITY_ORDER = (SeverityLevel.LOW, SeverityLevel.MEDIUM, SeverityLevel.HIGH)

    # PCI rating table (ASTM D6433-18 Table 2)
    PCI_RATINGS = [
        (86, 100, "Good"),
        (71, 85, "Satisfactory"),
        (56, 70, "Fair"),
        (41, 55, "Poor"),
        (26, 40, "Very Poor"),
        (11, 25, "Serious"),
        (0, 10, "Failed"),
    ]

    #: Which figure publishes each distress's curve, and the name the repository
    #: keys it under. Traceability only: the lookup itself goes through
    #: `curves.repository()`.
    CURVE_KEYS = {
        DistressType.ALLIGATOR: "alligator_cracking",
        DistressType.BLEEDING: "bleeding",
        DistressType.PATCHING: "patching",
        DistressType.LONGITUDINAL: "longitudinal_cracking",
        DistressType.TRANSVERSE: "transverse_cracking",
        DistressType.POTHOLES: "potholes",
        DistressType.RAVELLING: "weathering_raveling",
        DistressType.RUTTING: "rutting",
        DistressType.SHOVING: "shoving",
        DistressType.CORRUGATION: "corrugation",
        DistressType.DEPRESSION: "depression",
    }
    
    @classmethod
    def get_severity_from_width(
        cls,
        distress_type: DistressType,
        width_mm: float,
        depth_mm: Optional[float] = None,
    ) -> SeverityLevel:
        """Rate a distress per the measured criteria in ASTM D 6433-99.

        Only the distresses the standard rates on something measurable are rated
        here, and each on the quantity it actually names:

        * Cracking (X1.14.2) is rated on the non-filled crack *width*:
          under 10 mm is low, 10 to under 75 mm is medium, 75 mm and over is
          high. The old 5 mm and 20 mm steps came from neither this standard nor
          AASHTO R85, and they called a 30 mm crack high when this standard
          calls it medium.
        * Depressions (X1.10.1) are rated on maximum *depth*: 13 to 25 mm low,
          25 to 50 mm medium, over 50 mm high. Depth, not width.
        * Potholes (X1.17 and Table X1.1) need diameter *and* depth together.

        Alligator, patching, raveling, corrugation, shoving and bleeding are
        deliberately not rated here. The standard rates those on condition and
        on the ride quality the distress produces (X1.5.1, X1.15.1, X1.19.1 and
        the rest) -- whether pieces are spalled, whether a patch needs replacing,
        how the vehicle rides. None of that is readable off a detection box's
        extent, and a size threshold dressed up as a severity invents a rating
        the standard does not support. They are listed in CONDITION_RATED_TYPES
        so a report can say which ratings still owe a human judgement.
        """
        if distress_type in (DistressType.LONGITUDINAL, DistressType.TRANSVERSE):
            if width_mm < 10:
                return SeverityLevel.LOW
            if width_mm < 75:
                return SeverityLevel.MEDIUM
            return SeverityLevel.HIGH

        if distress_type == DistressType.DEPRESSION:
            if depth_mm is None:
                return SeverityLevel.LOW
            if depth_mm <= 25:
                return SeverityLevel.LOW
            if depth_mm <= 50:
                return SeverityLevel.MEDIUM
            return SeverityLevel.HIGH

        if distress_type == DistressType.POTHOLES:
            return cls._pothole_severity(width_mm, depth_mm)

        return SeverityLevel.LOW

    @classmethod
    def resolve_severity(cls, d: Distress) -> Tuple[SeverityLevel, str]:
        """Severity for one detection, and where that severity came from.

        Returns ``(severity, basis)`` where basis is:

        ``"measured"``
            Derived from a measurement the standard names for this type --
            crack width for X1.14.2, depth for depressions, diameter and depth
            together for potholes. Objective, so an inspector's opinion does
            not override it.
        ``"inspector"``
            The judgement the standard requires for a type it does not rate on
            a measurable extent, supplied in `Distress.rated_severity`. For the
            X1.4 types this is the ride-quality level.
        ``"unrated"``
            No judgement supplied, so it fell back to LOW.

        The fallback is the one worth being careful about. LOW is the mildest
        level, so defaulting to it understates the distress and flatters the
        PCI. That is the safer direction to be wrong in -- an overstatement
        would read better than the pavement deserves -- but it is still wrong,
        so every such detection is counted and the section's PCI is marked
        provisional rather than being reported as if the rating had been made.
        """
        if d.type not in cls.CONDITION_RATED_TYPES:
            return (
                cls.get_severity_from_width(d.type, d.width_mm, d.depth_mm),
                "measured",
            )
        if d.rated_severity is not None:
            return d.rated_severity, "inspector"
        return SeverityLevel.LOW, "unrated"

    @classmethod
    def _pothole_severity(
        cls, diameter_mm: float, depth_mm: Optional[float]
    ) -> SeverityLevel:
        """Pothole severity per ASTM D 6433-99 Table X1.1, keyed on diameter and depth.

        ===========  ==========  ===========  ==========
        Diameter mm  13-25 mm    >25-50 mm    >50 mm
        ===========  ==========  ===========  ==========
        100-200      L           L            L
        200-450      L           M            M
        450-750      L           M            H
        ===========  ==========  ===========  ==========

        Above 750 mm X1.17.1.2 drops diameter and rates on depth alone: 25 mm or
        less is medium, deeper is high.
        """
        if diameter_mm > 750:
            if depth_mm is not None and depth_mm > 25:
                return SeverityLevel.HIGH
            return SeverityLevel.MEDIUM
        if depth_mm is None:
            return SeverityLevel.LOW
        if depth_mm <= 25:
            return SeverityLevel.LOW
        if depth_mm <= 50:
            return SeverityLevel.MEDIUM if diameter_mm > 200 else SeverityLevel.LOW
        return SeverityLevel.HIGH if diameter_mm > 450 else SeverityLevel.MEDIUM
    
    @classmethod
    def _distress_amount(
        cls, distress: Distress, lane_width_m: float = 3.25
    ) -> float:
        """The quantity a distress type is measured by, per ASTM D6433-99 9.1/9.2.

        The measurement is distress-specific and is not always area. The
        standard's "How to Measure" clauses give: longitudinal and transverse
        cracking by length, alligator, patching, raveling, rutting, shoving,
        bleeding, corrugation and depression by area, and potholes by count.

        Every one of those is then divided by the *area* of the sample unit and
        multiplied by 100 to give a percent density (9.2). That only works if the
        quantity is in square metres, so a length has to be turned into the area
        of pavement it covers -- length times the lane width -- before it is
        divided. Dividing metres by square metres directly yields metres inverse,
        not a percentage, and produced densities over 100% for ordinary cracks:
        8 cracks over 50 m came out at 246%, which is not a density and is off the
        end of every published curve.

        A count has to be turned into an area the same way, which is what the
        per-unit figure below does.
        """
        measurement = cls.MEASUREMENT.get(distress.type, "area")
        if measurement == "length":
            return distress.length_m * lane_width_m
        if measurement == "area":
            return distress.area_m2
        if measurement == "count":
            per_unit = cls.COUNT_AREA_M2.get(distress.type, 0.5)
            return distress.length_m * per_unit
        return 0.0

    #: How each distress is measured in the field, and so how it becomes a
    #: density. Names the clause each comes from, for traceability.
    MEASUREMENT = {
        DistressType.LONGITUDINAL: "length",   # X1.14.3, linear metres
        DistressType.TRANSVERSE: "length",     # X1.14.3, linear metres
        DistressType.ALLIGATOR: "area",        # X1.5.2, square metres
        DistressType.PATCHING: "area",         # X1.15.2, square metres
        DistressType.RAVELLING: "area",        # X1.25, square metres
        DistressType.RUTTING: "area",          # X1.19.2, square metres
        DistressType.SHOVING: "area",          # X1.20.4, square metres
        DistressType.BLEEDING: "area",         # X1.6, square metres
        DistressType.CORRUGATION: "area",      # X1.9.2, square metres
        DistressType.DEPRESSION: "area",       # X1.10.2, square metres
        DistressType.POTHOLES: "count",        # X1.17.2, counted
    }

    #: Area a single counted unit is treated as covering, so a count can share a
    #: density with the area-measured distresses. Potholes are counted, not
    #: measured (X1.17.2).
    #:
    #: The factor is the standard's own, not a convenience. X1.17.1.2 converts a
    #: pothole larger than 750 mm to "the equivalent number of holes" by dividing
    #: its area by 0.5 m2 (5.5 ft2) -- so 0.5 m2 is the area one pothole counts as
    #: throughout the method. The 0.1 m2 this used to carry was a third of that
    #: and came from the placeholder fits, not the standard.
    #:
    #: It matters because it is what puts a pothole on the curve at all. At 0.1 m2
    #: a single hole in a 162.5 m2 section is a density of 0.06%, which is below
    #: the 0.1% the digitized pothole curve starts at, so one pothole priced at
    #: zero. At 0.5 m2 it is 0.31% and the curve answers.
    COUNT_AREA_M2 = {
        DistressType.POTHOLES: 0.5,
    }

    @classmethod
    def compute_density(
        cls, distress: Distress, section_area_m2: float, lane_width_m: float = 3.25
    ) -> float:
        """Density = (distress amount / section area) × 100 per ASTM D6433 Eq 8.

        This is the density of one detection. A section's density for a type is
        not the sum of these: see `compute_section_pci`, which adds up the
        amounts first and takes the density of the total.
        """
        if section_area_m2 <= 0:
            return 0.0
        amount = cls._distress_amount(distress, lane_width_m)
        return (amount / section_area_m2) * 100.0
    
    @classmethod
    def get_dv(cls, distress_type: DistressType, severity: SeverityLevel, density: float) -> float:
        """Deduct value for a type, severity and density percent, from the curve data.

        The curve is interpolated in log10(density), which is the axis ASTM draws
        its curves on. A density past the end of the digitized range raises
        rather than being extrapolated: a lookup table extended beyond its data
        produces a number that no measurement supports.
        """
        return cls.get_dv_with_provenance(distress_type, severity, density)["dv"]

    @classmethod
    def get_dv_with_provenance(
        cls, distress_type: DistressType, severity: SeverityLevel, density: float
    ) -> Dict:
        """As `get_dv`, plus the audit record of where the value came from.

        Reports the standard, edition, figure, interpolation method and whether
        the curve is a real digitization or a placeholder, so a reported PCI can
        be traced back to the exact curve that produced it.
        """
        key = cls.CURVE_KEYS.get(distress_type)
        if key is None:
            raise CurveNotFound(
                "No deduct-value curve is published for distress type {!r}.".format(
                    distress_type
                )
            )
        curve = curves.repository().get(key, severity.value[0].upper())
        record = curve.provenance(density, curve.interpolate(density))
        record["distress_type"] = distress_type.value
        return record
    
    @classmethod
    def allowable_deducts(cls, highest_dv: float) -> float:
        """``m`` = allowable number of deducts, ASTM D 6433-99 Eq 4 (9.5.3).

        ``m = 1 + (9/98) * (100 - HDV)``, capped at ten, where HDV is the highest
        individual deduct value. A road with one bad defect gets m near ten, so
        every other defect still counts; a road dominated by one severe distress
        gets m near one, and the rest are dropped rather than compounding a
        penalty the same damage already took.

        Rounded to one decimal, which is the precision the standard works to and
        which the fractional part of 9.5.4 depends on: unrounded, the worked
        example's m of 7.8786 turns its 5.3 into 4.66, where the standard's own
        7.9 turns it into 4.8.
        """
        return round(min(10.0, 1.0 + (9.0 / 98.0) * (100.0 - highest_dv)), 1)

    @classmethod
    def _truncate_to_m(cls, deduct_values: List[float]) -> List[float]:
        """Keep the m largest deduct values, including the fractional part (9.5.4).

        When m is 7.9 the list is the seven whole values plus a share of the
        eighth, scaled by the 0.9 -- the worked example in the standard turns a
        5.3 into 4.8 that way. If fewer than m values exist, all are used.
        """
        ordered = sorted(deduct_values, reverse=True)
        if not ordered:
            return []
        m = cls.allowable_deducts(ordered[0])
        whole = int(math.floor(m))
        fraction = m - whole
        kept = ordered[:whole]
        if fraction > 0.0 and len(ordered) > whole:
            kept.append(ordered[whole] * fraction)
        return kept

    @classmethod
    def corrected_deduct_value(cls, deduct_values: List[float]) -> float:
        """Maximum corrected deduct value, ASTM D 6433-99 9.5.1 to 9.5.5.

        Thin wrapper over `corrected_deduct_value_with_provenance` for callers
        that only want the number.
        """
        return cls.corrected_deduct_value_with_provenance(deduct_values)[0]

    @classmethod
    def corrected_deduct_value_with_provenance(
        cls, deduct_values: List[float]
    ) -> Tuple[float, bool]:
        """As `corrected_deduct_value`, plus whether any lookup ran off the curve.

        The standard does not scale a total by a factor for how many kinds of
        distress were found. It ranks the individual deduct values, keeps only
        the m that Eq 4 allows, sums them, and corrects that sum against the
        curve for q -- the number of retained values above 2.0. Any value above
        2.0 left over is then cut down to 2.0 and the whole thing recomputed, and
        the largest corrected value seen is the one used, so a single severe
        distress caps the damage instead of stacking.

        9.5.1: with no more than one value above 2.0, the total is the maximum
        corrected value outright and no curve is consulted.

        The boolean is the second half of the answer and the reason this variant
        exists. A total deduct value past the end of a published curve has no
        defensible reading in the data we hold, so rather than raise and take the
        request down, the reading is the curve's own terminal value and the fact
        is reported. The two are inseparable: a CDV that was clamped is a number
        the standard does not support at that input, and a caller that cannot see
        that will treat it as a measurement.
        """
        values = [float(v) for v in deduct_values if v > 0.0]
        if not values:
            return 0.0, False
        if sum(1 for v in values if v > 2.0) <= 1:
            return sum(values), False

        working = cls._truncate_to_m(values)
        best = 0.0
        saturated = False
        while True:
            total = sum(working)
            q = sum(1 for v in working if v > 2.0)
            cdv, clamped = cls._cdv_from_curve_with_provenance(total, q)
            saturated = saturated or clamped
            best = max(best, cdv)
            # 9.5.5.4: cut the smallest value still above 2.0 down to 2.0 and
            # repeat until only one is left. One value per pass, as the
            # standard says -- cutting every tied value at once would skip the
            # intermediate totals, and the answer is the largest of them.
            above = [v for v in working if v > 2.0]
            if len(above) <= 1:
                break
            smallest_above = min(above)
            first = working.index(smallest_above)
            working = list(working)
            working[first] = 2.0
        return best, saturated

    @classmethod
    def _cdv_from_curve(cls, total_deduct: float, q: int) -> float:
        """Read the AC correction curve for q, per 9.5.5.3 (Fig. X3.26).

        Thin wrapper over `_cdv_from_curve_with_provenance` for callers that only
        want the value.
        """
        return cls._cdv_from_curve_with_provenance(total_deduct, q)[0]

    @classmethod
    def _cdv_from_curve_with_provenance(
        cls, total_deduct: float, q: int
    ) -> Tuple[float, bool]:
        """Read the AC correction curve for q, and report whether it was clamped.

        The horizontal axis here is the total deduct value, which is a sum of
        deduct values rather than a density, so this curve is linear-logical in
        both directions and is interpolated as such.

        **q = 1 is answered from 9.5.1, not from a curve.** When only one deduct
        value exceeds 2.0 there is no compounding to correct, so the total *is*
        the maximum corrected deduct value. The iteration in 9.5.5.4 ends on
        exactly this case, so it is the reading the procedure is built to reach.

        This used to look up a q=1 curve, and the dataset has one, and it is
        wrong: as a placeholder fit it returns 57.43 at a total of 39.10 on the
        standard's own worked example, and up to 41.5 too high elsewhere. It was
        also byte-identical to the q=2 curve, which cannot be right, since q=1
        applies no correction and q=2 does. Neither the 1999 nor the 2024
        edition publishes a q=1 correction curve, because there is nothing for
        one to say. So the branch here is not a workaround for missing data; it
        is the clause, and consulting a curve for it was the bug.

        Past the end of a published curve the reading is its last point, which is
        a corrected deduct value of 100 and so a PCI of 0. That is the most damage
        the method can express, and it is also the right direction to saturate
        in: it never reports a road as better than the evidence supports. It is
        still not a value the standard publishes at that input, so it is
        returned alongside a flag.

        Note what is *not* being done here. An earlier proposal was to cap at
        CDV 100 for any total deduct value at or above 60. That threshold is
        wrong: the curves return a live, varying value across 60 to 100 -- at
        TDV 60 the digitized Figure X1.1 curves give roughly 54 to 68, not 100 --
        so capping there would report every heavily distressed section as PCI 0
        and destroy the ranking the pipeline exists to produce. The threshold
        here is the curve's own domain edge, which is the only point past which
        there is genuinely nothing left to read.
        """
        if q <= 1:
            # 9.5.1. Not clamped: the value is the standard's own answer, not a
            # reading taken off the end of a table.
            return total_deduct, False
        curve = curves.repository().cdv(q)
        try:
            return curve.interpolate(total_deduct), False
        except CurveRangeExceeded:
            terminal = max(p.deduct_value for p in curve.points)
            logger.warning(
                "Total deduct value %.1f is past the end of the q=%d correction "
                "curve (digitized to %.1f), so the reading is the curve's terminal "
                "value %.1f, i.e. PCI 0. That is a saturation, not a measurement, "
                "and the section is reported as saturated. The curve data is a "
                "placeholder fit, which is the underlying cause; see "
                "app/services/astm/data/README.md.",
                total_deduct, q, curve.high_density, terminal,
            )
            return terminal, True
    
    @classmethod
    def _capped_density(cls, amount_m2: float, area_m2: float) -> Tuple[float, bool]:
        """Percent density for 9.2, capped at the 100% the quantity can reach.

        9.2 defines density as a percent of the sample unit's area, so 100% is the
        most of a section that can be distressed. A detection set can still report
        more: two boxes over the same crack, or a length derived from a bounding
        area that reaches past the end of the section, both add up past the whole.

        That is a measurement artefact rather than an unknown value, and the
        published curves stop at 100% because there is nothing beyond it to read.
        So the density is capped and the fact is reported, rather than raising
        past the top of every curve. Raising was the alternative and it is much
        worse in practice: it discarded the whole section's PCI, including the
        distresses whose densities were perfectly ordinary.

        Returns the density and whether the cap was applied.
        """
        if area_m2 <= 0:
            return 0.0, False
        density = (amount_m2 / area_m2) * 100.0
        if density > 100.0:
            return 100.0, True
        return density, False

    @classmethod
    def _severity_densities(cls, section: SectionMetrics):
        """9.1 to 9.3 input: percent density for each type and severity level.

        9.1 sums the quantity of a distress type at each severity level over
        the whole sample unit, and 9.2 turns each of those totals into a
        percent density. The result is the state 9.6.2 operates on.
        """
        densities: Dict[Tuple[DistressType, SeverityLevel], float] = {}
        for d_type, distresses in cls._group_by_type(section).items():
            by_severity: Dict[SeverityLevel, List[Distress]] = {}
            for d in distresses:
                severity, _ = cls.resolve_severity(d)
                by_severity.setdefault(severity, []).append(d)
            for severity, group in by_severity.items():
                amount = sum(cls._distress_amount(d, section.lane_width_m)
                             for d in group)
                density, _ = cls._capped_density(amount, section.area_m2)
                densities[(d_type, severity)] = density
        return densities

    @classmethod
    def _merge_variants(cls, densities):
        """The severity configurations 9.6.2 requires be compared.

        A distress recorded at more than one severity level may in fact be one
        continuous extent that the severity definitions grade differently along
        its length. Merging the levels upward and re-reading the curves tests
        that reading against the as-found one.

        9.6.2.2, three severities, gives the list explicitly:

            (l, m, h)              as found
            (0, M, h)   M = l + m  low folded into medium
            (l, 0, H)   H = m + h  medium folded into high
            (0, m, H)   H = l + h  low folded straight into high
            (0, 0, H)   H = l+m+h  everything folded into high

        9.6.2.1, two severities, gives only (x1, x2) against (0, X2).
        """
        by_type: Dict[DistressType, Dict[SeverityLevel, float]] = {}
        for (d_type, severity), density in densities.items():
            by_type.setdefault(d_type, {})[severity] = density

        variants = [dict(densities)]

        def _variant(d_type, moves):
            """`moves` lists (from, into) pairs of severity levels to fold.

            Each source level is zeroed and its density added into its target,
            so folding low and medium both into high leaves high at their sum.
            """
            out = dict(densities)
            for src, _ in moves:
                out[(d_type, src)] = 0.0
            for src, dst in moves:
                out[(d_type, dst)] = (
                    out.get((d_type, dst), 0.0) + densities.get((d_type, src), 0.0)
                )
            return out

        for d_type, levels in by_type.items():
            present = [s for s in cls.SEVERITY_ORDER if s in levels]
            if len(present) == 2:
                variants.append(_variant(d_type, [(present[0], present[1])]))
            elif len(present) == 3:
                low, medium, high = present
                variants.append(_variant(d_type, [(low, medium)]))
                variants.append(_variant(d_type, [(medium, high)]))
                variants.append(_variant(d_type, [(low, high)]))
                variants.append(_variant(d_type, [
                    (low, high), (medium, high),
                ]))
        return variants

    @classmethod
    def _pci_from_densities(cls, densities):
        """PCI for one severity configuration, per 9.4 to 9.6.1.

        Returns ``(pci, max_cdv, saturated)``. Severities folded to zero are
        absent rather than free, so they are dropped before 9.5 ranks what is
        left, and `saturated` reports whether any 9.5.5.3 lookup ran off the end
        of its curve.
        """
        dvs = [
            cls.get_dv_with_provenance(d_type, severity, density)["dv"]
            for (d_type, severity), density in sorted(
                densities.items(), key=lambda kv: (kv[0][0].value, kv[0][1].value)
            )
            if density > 0.0
        ]
        max_cdv, saturated = cls.corrected_deduct_value_with_provenance(dvs)
        return (
            max(0.0, min(100.0, 100.0 - max_cdv)),
            max_cdv,
            saturated,
        )

    @classmethod
    def compute_section_pci(cls, section: SectionMetrics) -> Dict:
        """Compute PCI for a single section per ASTM D 6433-99 9.1 to 9.6.

        The individual deduct values are pooled across distress types here,
        because that is what the standard does: 9.5.2 ranks every deduct value
        in the section, not one list per distress type. A per-type subtotal that
        is then added up double-counts, because each type's correction curve is
        already fitted to the damage on its own.

        Pooling is across types, not across severities within one type. 9.3
        gives a deduct value per type *and severity level*, so a distress
        present at two levels contributes two ranked entries.
        """
        individual_dvs: List[float] = []
        breakdown: Dict[str, Dict] = {}
        provenance: List[Dict] = []
        provisional = 0
        density_capped: List[str] = []
        #: Types whose severity the standard leaves to a judgement, split by
        #: whether that judgement actually arrived.
        unrated_detections: List[str] = []
        ride_quality_rated: List[str] = []
        condition_rated: List[str] = []

        for d_type, distresses in cls._group_by_type(section).items():
            # One deduct value per distress type *and severity*, taken from the
            # total extent of that level in the section (9.1, 9.2, 9.3).
            #
            # Charging the curve once per individual detection instead pays the
            # curve's intercept once per pothole, and that intercept is most of
            # the value at low densities. Twenty potholes in one 50 m section
            # costed a TDV of 10.0 that way against an ASTM TDV of 0.53, taking
            # the section from 98.0 down to 89.1 for no reason in the road.
            by_severity: Dict[SeverityLevel, List[Distress]] = {}
            for d in distresses:
                severity, basis = cls.resolve_severity(d)
                by_severity.setdefault(severity, []).append(d)
                if basis == "unrated":
                    unrated_detections.append(d_type.value)
                elif basis == "inspector":
                    if d_type in cls.RIDE_QUALITY_TYPES:
                        ride_quality_rated.append(d_type.value)
                    else:
                        condition_rated.append(d_type.value)

            tdv = 0.0
            for severity, group in by_severity.items():
                amount = sum(cls._distress_amount(d, section.lane_width_m)
                             for d in group)
                density, capped = cls._capped_density(amount, section.area_m2)
                if capped:
                    density_capped.append(d_type.value)
                record = cls.get_dv_with_provenance(d_type, severity, density)
                provenance.append(record)
                tdv += record["dv"]
                # 9.3 gives one deduct value per distress type *and severity
                # level combination*, and 9.5.2 ranks those individual deduct
                # values against each other. So the two levels of one distress
                # are two entries, not their sum.
                #
                # Summing them first made a type present at two levels look
                # like a single severe distress: 13 sq ft low plus 14 sq ft
                # high alligator cracking became one deduct value of 32.72, so
                # it counted as a single deduct above 2.0 and 9.5.1 used that
                # total as the maximum CDV outright. Ranked separately, the two
                # are corrected against the q=2 curve and the section's PCI
                # comes out higher. Fig. 4 lists the two levels as separate
                # entries for exactly this reason.
                individual_dvs.append(record["dv"])

            breakdown[d_type.value] = {
                "tdv": round(tdv, 2),
                "count": len(distresses),
                "measurement": cls.MEASUREMENT.get(d_type, "area"),
            }

        # Only detections still owing a severity judgement make the section
        # provisional. One an inspector has rated is no longer provisional, which
        # is the whole point of accepting the rating.
        provisional = len(unrated_detections)

        # 9.6.2: where one distress carries more than one severity level, the
        # merged configurations have to be compared against the as-found one,
        # and the highest PCI wins. NOTE 1 under 12.1 records this being run
        # against the standard's own Fig. 4 and finding no correction needed;
        # that is a property of that data, not a general exemption.
        baseline_pci, baseline_cdv, baseline_saturated = cls._pci_from_densities(
            cls._severity_densities(section)
        )
        variants = cls._merge_variants(cls._severity_densities(section))
        best_pci, best_cdv, best_variant = baseline_pci, baseline_cdv, None
        #: Whether the variant that won had a saturated 9.5.5.3 lookup, or None
        #: while the as-found reading still stands. Tracked separately from
        #: `baseline_saturated` because a variant can be the reading that
        #: saturates, and reporting the baseline's flag would then be wrong.
        best_saturated = baseline_saturated
        for index, variant in enumerate(variants[1:], start=1):
            try:
                variant_pci, variant_cdv, variant_saturated = (
                    cls._pci_from_densities(variant)
                )
            except CurveNotFound:
                # A merged configuration can land on a (type, severity) pair
                # whose curve does not exist. The as-found reading stands.
                continue
            if variant_pci > best_pci:
                best_pci, best_cdv, best_variant, best_saturated = (
                    variant_pci, variant_cdv, index, variant_saturated,
                )

        pci, max_cdv = best_pci, best_cdv

        # Whether any lookup used a curve that is not a real digitization. A PCI
        # built on placeholder curves is arithmetic that works, but it is not
        # the standard, and the report has to say so rather than let a reader
        # assume otherwise.
        curves_provisional = any(r.get("provisional") for r in provenance)

        if provisional:
            logger.info(
                "Section %s has %d detection(s) of distresses ASTM D6433 rates on "
                "condition and ride quality rather than on a measurable extent, so "
                "their severity is a placeholder and the section PCI is provisional.",
                section.section_index,
                provisional,
            )

        if density_capped:
            logger.warning(
                "Section %s: %s reported more distress area than the section has "
                "(density above 100%%), so the density was capped at 100%%. The "
                "detections most likely overlap. The PCI is floored, not wrong, but "
                "the section should be checked.",
                section.section_index, ", ".join(sorted(set(density_capped))),
            )

        if best_saturated:
            logger.warning(
                "Section %s: the total deduct value ran past the end of the "
                "9.5.5.3 correction curve, so its PCI of %.2f is a saturation "
                "floor rather than a reading. Rank this section by total_cdv "
                "(%.2f), which is the part that is a real sum of deduct values.",
                section.section_index, pci, max_cdv,
            )

        return {
            "pci": round(pci, 2),
            "rating": cls._classify_pci(pci),
            "total_cdv": round(max_cdv, 2),
            "breakdown": breakdown,
            "individual_deduct_values": [round(v, 2) for v in individual_dvs],
            "deduct_value_sources": provenance,
            "curves_provisional": curves_provisional,
            "provisional_severity_detections": provisional,
            "severity_ratings": {
                "clause": "X1.4",
                "unrated": sorted(set(unrated_detections)),
                "ride_quality_rated": sorted(set(ride_quality_rated)),
                "condition_rated": sorted(set(condition_rated)),
                "note": "Distress types the standard does not rate on a measurable "
                        "extent. Severity comes from an inspector judgement "
                        "(Distress.rated_severity); where it is absent the level "
                        "defaults to LOW, which understates the distress.",
            },
            "density_capped_at_100pct": sorted(set(density_capped)),
            "cdv_curve_saturated": {
                "saturated": bool(best_saturated),
                "clause": "9.5.5.3",
                "note": "The total deduct value ran past the end of the "
                        "correction curve, so this PCI is the curve's terminal "
                        "value (PCI 0) rather than a reading the standard "
                        "publishes at this input. It is a floor, not a "
                        "measurement: no section here is better than this "
                        "says, and no section is worse than PCI 0 either. The "
                        "underlying cause is placeholder curve data, so compare "
                        "these sections by total_cdv and treat the PCI as an "
                        "ordering, not a value.",
            },
            "multi_severity_correction": {
                "applied": best_variant is not None,
                "variants_evaluated": len(variants),
                "baseline_pci": round(baseline_pci, 2),
                "reported_pci": round(pci, 2),
                "clause": "9.6.2",
            },
            "section_index": section.section_index,
            "start_m": section.start_m,
            "end_m": section.end_m,
        }

    @classmethod
    def _group_by_type(cls, section: SectionMetrics) -> Dict[DistressType, List[Distress]]:
        grouped: Dict[DistressType, List[Distress]] = {}
        for d in section.distresses:
            grouped.setdefault(d.type, []).append(d)
        return grouped
    
    @classmethod
    def _classify_pci(cls, pci: float) -> str:
        for lo, hi, label in cls.PCI_RATINGS:
            if lo <= pci <= hi:
                return label
        return "Failed"
    
    @classmethod
    def _detection_depth_mm(cls, det: Dict, d_type: DistressType) -> Optional[float]:
        """The measured depth of a detection in millimetres, or None if absent.

        The vision service reports pothole depth in metres under
        `pothole_depth_mm`, and rutting and depression depth already in
        millimetres under `depth_mm`. Both are accepted, in either unit, because
        the field name does not agree on the scale and guessing the scale here
        would put a 30 mm pothole in the low band or the high one.
        """
        raw = det.get("pothole_depth_mm")
        if raw is not None:
            metres = float(raw)
            # A depth of exactly zero is the detector saying "not measured",
            # which the severity tables treat as absent rather than as shallow.
            if metres == 0.0:
                return None
            # Vision works in metres on the pavement plane; a value past a few
            # metres is a unit slip, not a depth.
            return metres * 1000.0 if abs(metres) <= 10.0 else metres
        raw = det.get("depth_mm")
        if raw is not None:
            value = float(raw)
            # A depth of exactly zero is the detector saying "not measured",
            # which the severity tables treat as absent.
            return value if value > 0.0 else None
        return None

    @classmethod
    def compute_pci_from_detections(
        cls,
        detections: List[Dict],
        total_length_m: float,
        section_length_m: float = 50.0,
        lane_width_m: float = 3.25,
    ) -> Dict:
        """Main entry: compute PCI from detection list (e.g., from vision service)

        Sections are a fixed `section_length_m` (50 m per ASTM D6433) and their
        *count* follows `total_length_m`, the survey distance integrated from the
        vibration GPS. That is the survey's own statement of where the road ends,
        and it is the distance the IRI in each row was measured over, so a section
        is either wholly inside the survey or has no IRI to pair with. Adding
        sections past that point would emit rows carrying a PCI and no IRI, which
        reads as a measurement rather than as an artefact.

        Detection chainage comes from the vision GPS — a different logger over the
        same drive — and the two lengths do not agree: across the seven KNUST
        Obuasi trips they differ by 0.06% to 9.11%, the worst being 529 m. So
        detections near the end can fall outside the survey's sections. Those are
        counted and returned as `detections_outside_survey` rather than dropped,
        because a silently omitted defect is indistinguishable from a road without
        one, and the difference is a fraction of a percent on most trips and half
        a kilometre on the worst one.
        """
        detections = list(detections or [])
        n_sections = max(1, int(np.ceil(total_length_m / section_length_m)))
        survey_end_m = n_sections * section_length_m
        sections = []
        
        for i in range(n_sections):
            start = i * section_length_m
            end = min(start + section_length_m, total_length_m)
            area = (end - start) * lane_width_m
            
            section_distresses = []
            for det in detections:
                chainage = det.get("chainage_m", 0)
                if start <= chainage < end:
                    yolo_cls = det.get("yolo_class_id", 0)
                    d_type = cls.YOLO_TO_AASHTO.get(yolo_cls, DistressType.LONGITUDINAL)
                    
                    # Extract dimensions
                    width_mm = det.get("width", 0)  # width across lane in meters from vision
                    area_m2 = det.get("area", 0)
                    chainage_m = det.get("chainage_m", 0)
                    
                    # Compute actual length for linear distresses from area and width
                    if d_type in (DistressType.LONGITUDINAL, DistressType.TRANSVERSE):
                        # Linear distress: length = area / width (both from vision pinhole model)
                        # width_m is crack width across lane, area = length * width
                        width_m = width_mm if width_mm > 0 else 0.1  # fallback
                        length_m = area_m2 / width_m if area_m2 > 0 and width_m > 0 else 0.5
                    elif d_type in (DistressType.POTHOLES, DistressType.DEPRESSION):
                        # Count-based: use width as diameter proxy, each detection = 1 count
                        # Store count in length_m for density calculation
                        length_m = 1.0
                    else:
                        # Area-based distresses (alligator, patching, ravelling, rutting, shoving, bleeding, corrugation)
                        length_m = 0
                    
                    distress = Distress(
                        type=d_type,
                        severity=SeverityLevel.LOW,  # Will be overwritten
                        length_m=length_m,
                        area_m2=area_m2,
                        width_mm=width_mm * 1000.0,  # Convert m to mm for severity determination
                        # Table X1.1 rates a pothole on its diameter *and* its
                        # depth together, and X1.19.1 rates rutting on depth, so
                        # the measured depth has to reach the severity bands or
                        # both are decided on size alone. Vision measures it and
                        # reports it as `pothole_depth_mm` (metres, so scaled here
                        # to the millimetres the bands are written in); rutting
                        # and depression come through as `depth_mm`.
                        #
                        # Left unset this is not a neutral omission: on the
                        # digitized pothole curve a depth-blind pothole is capped
                        # at the low band, and the high band is worth roughly four
                        # times the low one.
                        depth_mm=cls._detection_depth_mm(det, d_type),
                        chainage_start_m=chainage_m,
                        chainage_end_m=chainage_m + length_m,
                    )
                    section_distresses.append(distress)
            
            sections.append(SectionMetrics(
                section_index=i,
                start_m=start,
                end_m=end,
                area_m2=area,
                distresses=section_distresses,
                lane_width_m=lane_width_m,
            ))
        
        # Compute PCI per section
        section_results = []
        total_cdv = 0.0
        for section in sections:
            result = cls.compute_section_pci(section)
            section_results.append(result)
            total_cdv += result.get("total_cdv", 0)
        
        # Overall PCI is the unweighted mean of the section PCIs, per ASTM
        # D6433: every 50 m section counts equally, and a section with no
        # detected distress scores 100 rather than being skipped. Skipping empty
        # sections would let a road's score depend on how the two GPS loggers
        # happened to differ, since each extra section pulls the mean up.
        overall_pci = np.mean([r["pci"] for r in section_results]) if section_results else 100.0

        # Detections the survey distance does not reach, reported rather than
        # dropped. A caller can see that the vision track ran further than the
        # road measured by the vibration logger, instead of inferring it from a
        # defect count that came out lower than the video seemed to show.
        outside = [
            float(d.get("chainage_m") or 0.0)
            for d in detections
            if float(d.get("chainage_m") or 0.0) >= survey_end_m
        ]
        if outside:
            logger.warning(
                f"{len(outside)} detection(s) fall beyond the {survey_end_m:.0f} m survey "
                f"distance and are not in any section (up to chainage "
                f"{max(outside):.0f} m). The vision and vibration GPS disagree on the "
                f"length of the survey.",
            )
        
        # Run-level provenance. Every section already carries its own
        # `curves_provisional` and `deduct_value_sources`, but a caller that
        # stores one number for a whole survey needs the caveat beside that
        # number, or the number travels alone and reads as an ASTM result.
        figures_used = sorted({
            source["figure"]
            for section in section_results
            for source in section.get("deduct_value_sources", [])
        })
        provisional_severities = sum(
            section.get("provisional_severity_detections", 0)
            for section in section_results
        )
        curves_provisional = any(
            section.get("curves_provisional", False) for section in section_results
        )
        capped_sections = [
            section["section_index"] for section in section_results
            if section.get("density_capped_at_100pct")
        ]
        #: Sections whose 9.5.5.3 lookup ran off the end of its curve. Reported at
        #: run level because the overall PCI averages section PCIs, so one
        #: saturated section silently drags the survey mean toward 0 and the
        #: caller has no way to see that from the mean alone.
        saturated_sections = [
            section["section_index"] for section in section_results
            if section.get("cdv_curve_saturated", {}).get("saturated")
        ]

        return {
            "overall_pci": round(overall_pci, 2),
            "rating": cls._classify_pci(float(overall_pci)),
            "sections": section_results,
            "total_length_m": total_length_m,
            "detections_outside_survey": len(outside),
            "survey_end_m": round(survey_end_m, 2),
            "provenance": {
                "standard": "{} {}".format(curves.STANDARD, curves.EDITION),
                "curve_data_edition": curves.CURVE_DATA_EDITION,
                # Two different claims, and they must not be conflated.
                #
                # `pci_is_provisional` is a property of the *method*: it is true
                # whenever the curve dataset is unverified, whether or not this
                # particular survey read a curve. A clean survey scores 100 and is
                # still produced by a method nobody has validated end to end, so
                # the flag stays true.
                #
                # `curves_provisional` is a property of *this run*: it says a
                # deduct value was actually read from unverified curve data. A
                # run with no distresses reads no curves, so it is false, and
                # that is the useful signal for "how much of this number came from
                # a digitized graph".
                "pci_is_provisional": bool(curves.repository().provisional_flags()),
                "curves_provisional": bool(curves_provisional),
                "provisional_severity_detections": provisional_severities,
                "figures_used": figures_used,
                # X3.26 is the -99 figure. In -24 it is X3.27, alongside a split
                # of weathering from raveling. The dataset is still -99 data, so
                # the -99 figure number is the accurate one here.
                "cdv_curve_figure": "X3.26",
                "cdv_curve_is_placeholder": True,
                "cdv_curve_figure_in_d6433_24": "X3.27",
                "cdv_saturated_sections": saturated_sections,
                "density_capped_sections": capped_sections,
                "limitations": cls.LIMITATIONS,
            },
        }


# Alias for backward compatibility
PCI_RATINGS = [
    (86, 100, "Good"),
    (71, 85, "Satisfactory"),
    (56, 70, "Fair"),
    (41, 55, "Poor"),
    (26, 40, "Very Poor"),
    (11, 25, "Serious"),
    (0, 10, "Failed"),
]


def compute_astm_pci_from_detections(
    detections: List[Dict],
    total_length_m: float,
    section_length_m: float = 50.0,
    lane_width_m: float = 3.25,
) -> Dict:
    """Convenience function matching paper's methodology"""
    return ASTMD6433PCI.compute_pci_from_detections(
        detections, total_length_m, section_length_m, 3.25
    )