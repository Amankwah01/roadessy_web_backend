"""One camera, and one answer about the ASTM track.

Both of these used to be answered in several places at once and the answers
disagreed. A defect's area is computed by turning pixels into metres through the
camera's mounting, so two fallbacks for "the caller said nothing" meant two
different pothole sizes from one recording — invisible in the reported IRI, which
is the same either way. The ASTM flags had the mirror problem: the log announced
two pipelines while one of them was switched off.
"""

import json

import pytest

from app.core.camera import (
    DEFAULT_CAMERA_CX,
    DEFAULT_CAMERA_CY,
    DEFAULT_CAMERA_FX,
    DEFAULT_CAMERA_FY,
    DEFAULT_CAMERA_HEIGHT_M,
    DEFAULT_CAMERA_TILT_DEG,
    normalise_calibration_payload,
    resolve_camera_geometry,
    serialize_distortion,
)
from app.core.config import astm_track_state


# ---------------------------------------------------------------------------
# One camera
# ---------------------------------------------------------------------------


def test_a_payload_that_says_nothing_gets_the_module_fallbacks():
    geometry = resolve_camera_geometry()

    assert geometry.height_m == DEFAULT_CAMERA_HEIGHT_M
    assert geometry.tilt_deg == DEFAULT_CAMERA_TILT_DEG
    assert (geometry.fx, geometry.fy) == (DEFAULT_CAMERA_FX, DEFAULT_CAMERA_FY)
    assert (geometry.cx, geometry.cy) == (DEFAULT_CAMERA_CX, DEFAULT_CAMERA_CY)
    assert geometry.distortion_coefficients is None


@pytest.mark.parametrize("payload", [None, "", "{}", "not json", "[]", '"a string"', "123", 0])
def test_anything_that_is_not_a_calibration_gets_the_same_fallbacks(payload):
    """A malformed payload must not become a different camera. Every one of these
    is an absent calibration, and an absent calibration has one answer."""
    geometry = resolve_camera_geometry(overrides=payload)

    assert geometry.as_tuple() == resolve_camera_geometry().as_tuple()


def test_the_flat_form_the_forms_send_is_read():
    geometry = resolve_camera_geometry(
        {"camera_height_meters": 1.6, "camera_tilt_degrees": 5.0}
    )

    assert geometry.height_m == 1.6
    assert geometry.tilt_deg == 5.0


def test_the_nested_intrinsics_are_read():
    geometry = resolve_camera_geometry(
        {
            "camera_height_meters": 1.6,
            "intrinsics": {
                "fx": 800.0,
                "fy": 801.0,
                "cx": 320.0,
                "cy": 240.0,
                "distortion_coefficients": [0.1, -0.02, 0.0, 0.0, 0.0],
            },
        }
    )

    assert (geometry.fx, geometry.fy, geometry.cx, geometry.cy) == (800.0, 801.0, 320.0, 240.0)
    assert json.loads(geometry.distortion_coefficients) == [0.1, -0.02, 0.0, 0.0, 0.0]


def test_the_checkerboard_response_can_be_handed_straight_back():
    """That route answers with a `camera_metadata` wrapper, and a caller who
    calibrated and then analyzed has exactly that object to hand over."""
    geometry = resolve_camera_geometry(
        {
            "camera_metadata": {
                "camera_height_meters": 1.6,
                "camera_tilt_degrees": 5.0,
                "intrinsics": {"fx": 800.0, "fy": 801.0, "cx": 320.0, "cy": 240.0},
            }
        }
    )

    assert geometry.height_m == 1.6
    assert geometry.tilt_deg == 5.0
    assert geometry.fx == 800.0
    assert geometry.cy == 240.0


def test_the_two_routes_that_size_defects_agree_on_the_same_payload():
    """The property this module exists for, stated as a test rather than left to
    a reader to verify: two entry points, one payload, one camera."""
    payload = {"camera_height_meters": 1.6, "camera_tilt_degrees": 5.0}

    via_test_route = resolve_camera_geometry(overrides=json.dumps(payload))
    via_pipeline = resolve_camera_geometry(overrides=payload)

    assert via_test_route.as_tuple() == via_pipeline.as_tuple()

    # And with nothing supplied at all, which is where they used to differ.
    assert resolve_camera_geometry().as_tuple() == resolve_camera_geometry(overrides=None).as_tuple()


def test_the_payload_wins_over_a_saved_profile():
    class Profile:
        camera_height_meters = 3.0
        camera_tilt_degrees = 30.0

    geometry = resolve_camera_geometry({"camera_height_meters": 1.6}, stored=Profile())

    assert geometry.height_m == 1.6, "the payload describes this upload's mount"
    assert geometry.tilt_deg == 30.0, "and the profile still answers what the payload did not"


def test_a_saved_profile_fills_what_the_payload_omits():
    class Profile:
        camera_height_meters = 3.0
        camera_tilt_degrees = 30.0

    geometry = resolve_camera_geometry(stored=Profile())

    assert geometry.height_m == 3.0
    assert geometry.tilt_deg == 30.0


def test_a_saved_profile_supplies_its_intrinsics():
    class Profile:
        camera_height_meters = None
        camera_tilt_degrees = None
        fx = 700.0
        fy = 701.0
        cx = 310.0
        cy = 241.0
        distortion_coefficients = "[0.1, 0.0, 0.0, 0.0, 0.0]"

    geometry = resolve_camera_geometry(stored=Profile())

    assert geometry.fx == 700.0
    assert geometry.cx == 310.0
    assert json.loads(geometry.distortion_coefficients) == [0.1, 0.0, 0.0, 0.0, 0.0]


def test_a_partial_lens_is_not_completed_from_the_other_source():
    """A focal length from one calibration and a principal point from another
    describes neither camera. Whichever source supplies the lens supplies all of
    it, and whatever it lacks stays on the defaults."""
    class Profile:
        camera_height_meters = None
        camera_tilt_degrees = None
        fx = 700.0
        fy = None
        cx = None
        cy = None
        distortion_coefficients = None

    geometry = resolve_camera_geometry(
        {"intrinsics": {"fx": 999.0, "fy": 998.0, "cx": 300.0, "cy": 200.0}},
        stored=Profile(),
    )

    assert geometry.fx == 999.0, "the payload describes this upload's lens"
    assert geometry.cx == 300.0, "so all four of its own come from it"
    assert geometry.cy == 200.0


def test_a_partial_lens_in_the_payload_does_not_borrow_from_the_profile():
    class Profile:
        camera_height_meters = None
        camera_tilt_degrees = None
        fx = 700.0
        fy = 701.0
        cx = 310.0
        cy = 241.0
        distortion_coefficients = None

    geometry = resolve_camera_geometry(
        {"intrinsics": {"fx": 999.0}},  # only a focal length
        stored=Profile(),
    )

    assert geometry.fx == 999.0
    assert geometry.fy == DEFAULT_CAMERA_FY, "not the profile's fy"
    assert geometry.cx == DEFAULT_CAMERA_CX, "not the profile's cx"
    assert geometry.cy == DEFAULT_CAMERA_CY, "not the profile's cy"


def test_the_profile_supplies_the_lens_when_the_payload_has_none():
    class Profile:
        camera_height_meters = None
        camera_tilt_degrees = None
        fx = 700.0
        fy = 701.0
        cx = 310.0
        cy = 241.0
        distortion_coefficients = None

    geometry = resolve_camera_geometry({"camera_height_meters": 1.6}, stored=Profile())

    assert geometry.height_m == 1.6
    assert geometry.fx == 700.0


def test_a_boolean_is_not_a_height():
    """`True` is an int in Python, so without the exclusion a malformed payload
    would calibrate the camera to one metre."""
    geometry = resolve_camera_geometry({"camera_height_meters": True, "camera_tilt_degrees": False})

    assert geometry.height_m == DEFAULT_CAMERA_HEIGHT_M
    assert geometry.tilt_deg == DEFAULT_CAMERA_TILT_DEG


def test_a_value_of_the_wrong_type_does_not_become_a_number():
    geometry = resolve_camera_geometry({"camera_height_meters": "high", "fx": "wide"})

    assert geometry.height_m == DEFAULT_CAMERA_HEIGHT_M
    assert geometry.fx == DEFAULT_CAMERA_FX


def test_the_tuple_and_the_kwargs_agree():
    geometry = resolve_camera_geometry({"camera_height_meters": 1.6, "intrinsics": {"fx": 800.0}})
    height, tilt, fx, fy, cx, cy, distortion = geometry.as_tuple()
    kwargs = geometry.as_kwargs()

    assert kwargs["camera_height_m"] == height
    assert kwargs["camera_tilt_deg"] == tilt
    assert kwargs["fx"] == fx
    assert kwargs["fy"] == fy
    assert kwargs["cx"] == cx
    assert kwargs["cy"] == cy
    assert kwargs["distortion_coefficients"] == distortion


# ---------------------------------------------------------------------------
# Distortion
# ---------------------------------------------------------------------------


def test_a_partly_malformed_coefficient_list_is_dropped_whole():
    """Half a calibration is not a better answer than none, and the caller cannot
    tell which half it got."""
    assert serialize_distortion([0.1, "x", 0.0]) is None
    assert serialize_distortion([]) is None
    assert serialize_distortion(None) is None
    assert serialize_distortion("") is None


def test_the_legacy_separated_string_is_understood():
    """A stored profile may hold the coefficients as text rather than as a list."""
    assert json.loads(serialize_distortion("0.1, -0.02, 0.0, 0.0, 0.0")) == [0.1, -0.02, 0.0, 0.0, 0.0]
    assert json.loads(serialize_distortion("0.1 -0.02 0.0")) == [0.1, -0.02, 0.0]


def test_a_string_that_only_looks_like_a_json_array_is_refused():
    assert serialize_distortion("[0.1, null]") is None
    assert serialize_distortion("[not json") is None
    assert json.loads(serialize_distortion("[0.1, 0.2]")) == [0.1, 0.2]


def test_the_normaliser_reports_only_what_the_payload_said():
    """Separating this from the resolver is what keeps "the caller sent 1.6" and
    "nothing was sent" distinguishable instead of the same number."""
    parsed = normalise_calibration_payload({"camera_height_meters": 1.6})

    assert parsed == {"camera_height_m": 1.6}
    assert normalise_calibration_payload(None) == {}
    assert normalise_calibration_payload("{}") == {}


def test_already_undistorted_survives_only_as_the_boolean_it_is():
    assert normalise_calibration_payload({"already_undistorted": True})["already_undistorted"] is True
    assert "already_undistorted" not in normalise_calibration_payload({"already_undistorted": "yes"})


# ---------------------------------------------------------------------------
# The ASTM track
# ---------------------------------------------------------------------------


def test_no_astm_work_happens_in_this_build():
    """Both halves off. The PCI half is withheld for a documented data fault; the
    IRI half is sound but belongs to the same track and is the most expensive
    step in the inertial path for a figure no report here consumes."""
    state = astm_track_state()

    assert state["running"] is False
    assert state["pci_enabled"] is False
    assert state["qcm_iri_enabled"] is False


def test_the_state_says_why_it_is_off_in_its_own_words():
    """A log line that says only "disabled" leaves the next reader to work out
    whether that was a decision or a fault."""
    state = astm_track_state()

    assert state["reason"], "an absent track needs a stated reason"
    assert state["pci_reason"], "and the PCI's own reason is different from the track's"
    assert state["reason"] != state["pci_reason"]
    assert "retained and disabled" in state["reason"]


def test_the_reason_is_absent_while_the_track_runs(monkeypatch):
    """Otherwise a switched-on build would still be shipping a mute reason to
    every consumer, and the reason is what makes the mute legible."""
    from app.core import config

    monkeypatch.setattr(config, "ASTM_TRACK_RUNNING", True)
    monkeypatch.setattr(config, "ASTM_PCI_ENABLED", True)

    state = config.astm_track_state()

    assert state["reason"] is None
    assert state["pci_reason"] is None
