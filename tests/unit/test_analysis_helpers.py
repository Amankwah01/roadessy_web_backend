"""The two helpers that decide what a survey run is allowed to claim.

`_resample_onto` and `_parse_calibration_metadata` both sit between untrusted
input and the numbers the pipeline computes. What they must not do is invent a
value: an interpolated position that was never measured, a camera height that
nobody supplied, or a boolean read out of a number.
"""

import json
import math

import numpy as np

from app.api.routes.analysis import _parse_calibration_metadata, _resample_onto
from app.core.camera import resolve_camera_geometry


# ---------------------------------------------------------------------------
# _resample_onto
# ---------------------------------------------------------------------------


def test_a_route_is_stretched_onto_the_asked_number_of_samples():
    """The two recorders start by hand and have unrelated rates, so the counts
    never match and one of them has to be resampled."""
    values = np.array([0.0, 10.0, 20.0, 30.0])

    resampled = _resample_onto(values, 7)

    assert resampled.size == 7
    assert resampled[0] == 0.0
    assert resampled[-1] == 30.0, "the last known fix must survive to the last sample"
    # Linear between fixes, so the midpoint of the series is its midpoint.
    assert math.isclose(float(resampled[3]), 15.0, abs_tol=1e-9)


def test_resampling_more_samples_than_there_are_fixes_interpolates():
    resampled = _resample_onto(np.array([0.0, 4.0]), 5)

    assert resampled.size == 5
    assert np.all(np.diff(resampled) > 0), "a single step would make the position a staircase"


def test_a_gap_between_two_fixes_is_interpolated_across():
    """Between two known fixes the route is smooth enough that interpolating is a
    closer answer than dropping the sample. The guarantee is not that gaps
    survive: it is that a stretch with no fixes at all stays unfilled."""
    values = np.array([1.0, np.nan, 3.0])

    resampled = _resample_onto(values, 3)

    assert resampled[0] == 1.0
    assert resampled[-1] == 3.0
    assert math.isclose(float(resampled[1]), 2.0, abs_tol=1e-9)


def test_a_route_with_no_fixes_at_all_is_all_nan_rather_than_all_zero():
    values = np.array([np.nan, np.nan, np.nan])

    resampled = _resample_onto(values, 4)

    assert np.isnan(resampled).all(), "no position must not read as position zero"


def test_one_fix_is_held_rather_than_interpolated_against_nothing():
    values = np.array([np.nan, 7.5, np.nan])

    resampled = _resample_onto(values, 3)

    assert np.allclose(resampled, 7.5), "a single fix cannot describe a slope, so it is held"


def test_an_empty_route_does_not_raise():
    assert _resample_onto(np.array([]), 5).size == 5
    assert np.isnan(_resample_onto(np.array([]), 5)).all()


def test_asking_for_no_samples_gives_no_samples():
    assert _resample_onto(np.array([1.0, 2.0]), 0).size == 0
    assert _resample_onto(np.array([1.0, 2.0]), -3).size == 0


def test_resampling_is_not_confused_by_infinities():
    """An infinity is a failed fix, not a very large distance."""
    values = np.array([0.0, np.inf, 2.0])

    resampled = _resample_onto(values, 3)

    assert np.isfinite(resampled).all()


# ---------------------------------------------------------------------------
# _parse_calibration_metadata
# ---------------------------------------------------------------------------


def test_no_metadata_yields_nothing_for_the_caller_to_read():
    """The defaults live at the call site, which does `calibration.get("fx", 1000.0)`
    and so on. Returning an empty dict rather than a half-filled one is what makes
    that `.get` reach its default at all."""
    assert _parse_calibration_metadata(None) == {}
    assert _parse_calibration_metadata("") == {}


def test_the_flat_form_of_camera_metadata_is_accepted():
    parsed = _parse_calibration_metadata(
        json.dumps({"camera_height_meters": 1.6, "camera_tilt_degrees": 5.0})
    )

    assert parsed["camera_height_m"] == 1.6
    assert parsed["camera_tilt_deg"] == 5.0


def test_the_short_form_is_accepted_too():
    parsed = _parse_calibration_metadata(json.dumps({"camera_height_m": 1.2, "camera_tilt_deg": 2.0}))

    assert parsed["camera_height_m"] == 1.2
    assert parsed["camera_tilt_deg"] == 2.0


def test_nested_intrinsics_are_read_and_serialised_for_the_detector():
    parsed = _parse_calibration_metadata(
        json.dumps(
            {
                "camera_height_meters": 1.6,
                "camera_tilt_degrees": 5.0,
                "intrinsics": {
                    "fx": 800.0,
                    "fy": 801.0,
                    "cx": 320.0,
                    "cy": 240.0,
                    "distortion_coefficients": [0.1, -0.02, 0.0, 0.0, 0.0],
                },
            }
        )
    )

    assert parsed["camera_height_m"] == 1.6
    assert parsed["camera_tilt_deg"] == 5.0
    assert parsed["fx"] == 800.0
    assert parsed["cy"] == 240.0
    # The detector takes the coefficients as a JSON string, which is the form the
    # rest of the pipeline passes them in.
    assert json.loads(parsed["distortion_coefficients"]) == [0.1, -0.02, 0.0, 0.0, 0.0]


def test_malformed_metadata_is_ignored_rather_than_fatal():
    """Refusing to run because a metadata string would not parse would hide the
    thing being tested, so the run continues and the call site supplies defaults."""
    for raw in ("not json at all", "{", "[]", '"a string"', "123"):
        assert _parse_calibration_metadata(raw) == {}, (
            "an unusable payload must resolve to nothing rather than to invented "
            "camera geometry"
        )


def test_a_value_of_the_wrong_type_is_left_out_so_the_resolver_defaults_it():
    """`"high"` is not a height in metres, and a boolean is not a number. The
    parser's job is to say what the payload said; what the pipeline assumes when
    it said nothing is `app.core.camera`'s single answer."""
    parsed = _parse_calibration_metadata(
        json.dumps({"camera_height_m": "high", "camera_tilt_deg": True, "fx": False})
    )

    assert "camera_height_m" not in parsed
    assert "camera_tilt_deg" not in parsed
    assert "fx" not in parsed

    from app.core.camera import DEFAULT_CAMERA_HEIGHT_M, DEFAULT_CAMERA_TILT_DEG

    assert resolve_camera_geometry(parsed).height_m == DEFAULT_CAMERA_HEIGHT_M
    assert resolve_camera_geometry(parsed).tilt_deg == DEFAULT_CAMERA_TILT_DEG


def test_a_partly_named_intrinsics_block_keeps_what_is_there():
    parsed = _parse_calibration_metadata(json.dumps({"intrinsics": {"fx": 700.0, "cy": "?"}}))

    assert parsed["fx"] == 700.0
    assert "cy" not in parsed, "a malformed component is left out, not defaulted to zero"


def test_already_undistorted_is_passed_through_as_the_boolean_it_is():
    assert _parse_calibration_metadata(json.dumps({"already_undistorted": True}))["already_undistorted"] is True
    assert (
        _parse_calibration_metadata(json.dumps({"already_undistorted": False}))["already_undistorted"]
        is False
    )
    assert "already_undistorted" not in _parse_calibration_metadata(json.dumps({"already_undistorted": "yes"}))


def test_a_distortion_list_with_a_non_number_in_it_is_dropped():
    parsed = _parse_calibration_metadata(
        json.dumps({"intrinsics": {"fx": 700.0, "distortion_coefficients": [0.1, "x", 0.0]}})
    )

    assert "distortion_coefficients" not in parsed, (
        "a partially malformed coefficient list must not be passed on as if it "
        "were a calibration"
    )
    assert parsed["fx"] == 700.0