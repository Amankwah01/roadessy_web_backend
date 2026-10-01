# app/core/config.py
import os

import numpy as np
from scipy.interpolate import PchipInterpolator

# ---------------------------------------------------------------------------
# ASTM D6433 PCI -- MUTED
# ---------------------------------------------------------------------------
# The PCI calculation in `app/services/astm/` is retained in full but is not
# currently published. It is switched off here rather than deleted, because the
# code is sound and the switch is what the calculation is missing: the
# deduct-value curves in `app/services/astm/data/curves_d6433_99.json` are
# PLACEHOLDER polynomial fits, not digitizations of the published figures, and
# the corrected-deduct-value curves are additionally wrong in a way that has
# been confirmed against the standard itself:
#
#   * The q correction curves are stored on a TDV domain of 0-100. The
#     standard's own worked example (clause 9.5.5.1) runs at TDV = 104.7, so
#     the stored data saturates on the standard's own example.
#   * q = 1 and q = 2 are the same curve, which cannot be right: the q=1 case
#     is CDV = TDV with no correction applied, and ASTM 9.5.1 returns the total
#     deduct value directly when at most one deduct exceeds 2.0.
#   * Figure X3.27 could not be digitized from the available copy of the -24
#     PDF. Its corrected-deduct-value axis measures 0-100, but the total-deduct-
#     value axis carries no tick labels in that raster, so the x scale can only
#     be assumed via the CDV = TDV diagonal -- which makes the calibration and
#     the measurement the same unknown, and the assumption is not stable
#     (it implies a TDV range of ~369 at the frame edge, against 200 in the
#     third-party table and 190 in an independent reproduction).
#   * The curves are only ~6 resolvable branches per column rather than the ten
#     that 9.5.3 implies, and their in-plot labels are illegible at that
#     raster resolution, so branches cannot be mapped to q values.
#
# So the number is withheld rather than shown with a caveat. A PCI that reads
# as a measurement is worse than no PCI, and the Roadessy pipeline is
# unaffected.
#
# One of those was a code bug, not a data bug, and is now fixed: the q=1 lookup
# consulted a fitted q=1 curve, but 9.5.1 defines the q=1 answer as the total
# deduct value itself and consults no curve, so the fitted curve returned up to
# 41.5 too high (57.43 where 9.5.1 requires 39.10, on the standard's own
# worked example) and was byte-identical to the q=2 curve. No digitization was
# needed to fix that; the clause determines the answer. The remaining gap is
# only the curve data.
#
# To re-enable: set ASTM_PCI_ENABLED=true, AND replace `cdv_curves` in
# curves_d6433_99.json with curves digitized from a vector or higher-resolution
# copy of ASTM D6433-24 Fig. X3.27 (with `method` changed from
# "placeholder_polynomial_fit" to "digitized" and `provisional` set false).
# Turning the flag on without replacing the data re-publishes the placeholders.
#
# `CurveRepository.self_check()` reports the outstanding data faults on load and
# names the ones that must clear first; see
# `app/services/astm/data/README.md`. In particular the CDV domain defect is
# deliberately reported rather than patched: extending a placeholder fit past
# its own end is how a fabricated curve gets mistaken for a digitized one.
#
# The IRI side of the ASTM pipeline (QCM per ASTM E1926/E950) is a real
# measurement from the vibration data and is NOT affected by the curve problem,
# so it keeps running. Set ASTM_QCM_IRI_ENABLED=false to mute that too.
def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


#: Publish the D6433-derived PCI. Default off; see the note above.
ASTM_PCI_ENABLED = _env_flag("ASTM_PCI_ENABLED", False)

#: Publish the ASTM QCM IRI. Independent of the PCI curve problem, so on.
ASTM_QCM_IRI_ENABLED = _env_flag("ASTM_QCM_IRI_ENABLED", True)

#: What a consumer should show in place of a PCI. Kept next to the flag so the
#: API, the reports and the exports all say the same thing.
ASTM_PCI_DISABLED_REASON = (
    "ASTM D6433 PCI is not published in this build. The deduct-value curves "
    "are placeholder fits rather than digitizations of the standard's figures, "
    "and the corrected-deduct-value curves saturate below the TDV of the "
    "standard's own worked example. The calculation code is retained and is "
    "disabled, not removed."
)

# Golden Quarter-Car Simulation (QCM) Constants
C_S = 6.0  # Suspension damping coefficient (1/s)
K_S = 623.0  # Suspension stiffness coefficient (1/s^2)
K_T = 653.0  # Tire stiffness coefficient (1/s^2)
M_R = 0.15  # Mass ratio (Unsprung mass / Sprung mass)
V_SPEED = 22.22  # Standard calibration velocity (80 km/h in m/s)
V_REF_KMH = 80.0  # Standard reference speed for IRI computation (km/h)

# IRI Segment Configuration
IRI_SEGMENT_LENGTH_M = 50.0  # Standard segment length for IRI reporting (m)

# Parametric PCHIP Spline Control Knots
MATRIX_KNOTS_X = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

PCHIP_ALLIGATOR = PchipInterpolator(MATRIX_KNOTS_X, np.array([5.0, 12.0, 22.0, 30.0, 45.0, 60.0, 72.0, 85.0, 100.0]))
PCHIP_LINEAR = PchipInterpolator(MATRIX_KNOTS_X, np.array([2.0, 6.0, 12.0, 18.0, 28.0, 42.0, 50.0, 68.0, 85.0]))
PCHIP_POTHOLE = PchipInterpolator(
    MATRIX_KNOTS_X,
    np.array([15.0, 28.0, 45.0, 55.0, 70.0, 85.0, 90.0, 98.0, 100.0]),
)
