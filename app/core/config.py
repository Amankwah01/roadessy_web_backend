# app/core/config.py
import os
from typing import Any, Dict

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
# measurement from the vibration data and is NOT affected by the curve problem.
# It is nevertheless off by default, because the track is published as a unit:
#
#   * A reader shown "ASTM QCM IRI" next to no ASTM PCI has been told one half of
#     a method and not the other, and will reasonably assume the half that is
#     missing is missing from the data rather than withheld from the build. The
#     mute reason says otherwise, but a reason beside a number does not stop the
#     number being quoted on its own.
#   * The QCM half is not free. It resamples to 100 Hz and runs a quarter-car
#     simulation, and it is the most expensive step in the inertial path for a
#     figure no report in this service consumes.
#
# Set ASTM_QCM_IRI_ENABLED=true to compute and publish it again. The calculation
# is retained in full either way.
def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


#: Where the pipeline writes everything it produces. One root for the whole
#: deployment, because the artifact routes serve exactly what the pipeline wrote
#: and a second tree next to it would be a second thing that can drift empty.
#:
#: Read through `app.core.env` so a value in `.env` is honoured whichever module
#: is imported first; `os.getenv` alone only sees the process environment, which
#: is how a configured output directory ends up silently ignored.
_DEFAULT_OUTPUT_DIR = r"F:\Adubofour\rodeessy_mphil\RoadSurveyOutputs"


def output_base_dir() -> "os.PathLike[str]":
    from pathlib import Path

    from app.core.env import get_env

    return Path(get_env("ROADESSY_OUTPUT_DIR", _DEFAULT_OUTPUT_DIR))


#: Publish the D6433-derived PCI. Default off; see the note above.
ASTM_PCI_ENABLED = _env_flag("ASTM_PCI_ENABLED", False)

#: Maximum upload size in bytes (default 5GB)
MAX_UPLOAD_BYTES = int(os.getenv("ROADESSY_MAX_UPLOAD_BYTES", str(5 * 1024 * 1024 * 1024)))

#: Publish the ASTM QCM IRI. Off by default too; see the note above. Sound
#: calculation, but it belongs to a track whose PCI half is withheld, and it is
#: the most expensive step in the inertial path.
ASTM_QCM_IRI_ENABLED = _env_flag("ASTM_QCM_IRI_ENABLED", False)

#: Run no part of the ASTM track at all.
#:
#: The two sub-switches above decide what the ASTM track *publishes*. This one
#: decides whether any of it is *computed*, so a deployment that does not want
#: ASTM pays none of its cost: no QCM IRI resampling, no ASTM interpretation of
#: the detections, no ASTM segments.
#:
#: The video is still decoded either way. Detection is shared — the Roadessy
#: track reuses the same detections under a different distress mapping — so
#: switching this off cannot skip `process_video_astm`, and does not try to.
#:
#: Default off. With both sub-switches off it is already inert, so this is
#: stated separately to make the intent readable at the top of the module
#: rather than something a reader has to derive from two other flags.
ASTM_TRACK_ENABLED = _env_flag("ASTM_TRACK_ENABLED", False)

#: Whether any ASTM work happens this run.
ASTM_TRACK_RUNNING = ASTM_TRACK_ENABLED and (ASTM_PCI_ENABLED or ASTM_QCM_IRI_ENABLED)

#: Why the ASTM track is absent, for the API and the reports to repeat.
ASTM_TRACK_DISABLED_REASON = (
    "The ASTM track is disabled in this build, so no ASTM PCI or ASTM QCM IRI is "
    "reported for this run. Condition figures come from the Roadessy track. The "
    "ASTM calculation code is retained and disabled, not removed."
)

#: What a consumer should show in place of a PCI. Kept next to the flag so the
#: API, the reports and the exports all say the same thing.
ASTM_PCI_DISABLED_REASON = (
    "ASTM D6433 PCI is not published in this build. The deduct-value curves "
    "are placeholder fits rather than digitizations of the standard's figures, "
    "and the corrected-deduct-value curves saturate below the TDV of the "
    "standard's own worked example. The calculation code is retained and is "
    "disabled, not removed."
)


def astm_track_state() -> Dict[str, Any]:
    """What this build does with the ASTM track, in one answer.

    Read once per run and logged, so "the pipeline reported no ASTM PCI and no
    ASTM IRI" is a line in the run's own log rather than something a reader has
    to infer from the absence of two numbers. The API answers the same question
    from this same function, so the log and the response cannot disagree about
    what the build does.
    """
    return {
        "running": ASTM_TRACK_RUNNING,
        "track_enabled": ASTM_TRACK_ENABLED,
        "pci_enabled": ASTM_PCI_ENABLED,
        "qcm_iri_enabled": ASTM_QCM_IRI_ENABLED,
        "reason": None if ASTM_TRACK_RUNNING else ASTM_TRACK_DISABLED_REASON,
        # The PCI carries its own, more specific reason, reported alongside rather
        # than replaced: a reader told the track is off still wants to know that
        # this half is withheld for a documented data fault, which is a different
        # situation from the track being switched off deliberately.
        "pci_reason": None if ASTM_PCI_ENABLED else ASTM_PCI_DISABLED_REASON,
    }

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
