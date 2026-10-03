"""Where the camera's geometry comes from, in one place.

A road video's defects are sized by turning pixels into metres, and that
conversion is driven by how the camera was mounted: how high above the pavement
it sat, how far it was tilted down, and the lens's own intrinsics. Three
separate places used to answer "what if the caller said nothing", and they
disagreed:

    the runs and batch forms      1.5 m, 0 deg
    /analysis/test-vision        2.5 m, 0 deg
    the survey pipeline           1.5 m, 15 deg
    the checkerboard form         1.6 m, 15 deg

The same video analyzed through two of those produced two different ground
scales, and therefore two different pothole areas and widths, while the reported
IRI stayed the same in both — so the disagreement was invisible in the headline
number and only visible in a defect size nobody thought to check.

One source now: the calibration the analysis payload carries, which the runs and
batch forms always populate from the device the operator picked. The constants
below are the fallback for a payload that omits a field, and nothing else decides
them.

They are the survey pipeline's former defaults, 1.5 m and 15 degrees, because
those are the values a downward-looking road camera is actually mounted at. A
tilt of 0 is not a neutral assumption here: the image-to-ground homography needs
the camera to be looking at the road at all, and `homography_from_payload`
refuses to build one without a tilt, so defaulting to 0 would silently drop
crack widths rather than merely estimate them coarsely.

Every consumer goes through `resolve_camera_geometry`. Adding a fourth caller
that reads these constants directly would reopen the problem this module exists
to close.
"""

from typing import Any, Dict, Mapping, Optional

#: Mounting height above the pavement, in metres.
DEFAULT_CAMERA_HEIGHT_M = 1.5

#: Downward tilt from horizontal, in degrees. Zero cannot reach the road.
DEFAULT_CAMERA_TILT_DEG = 15.0

#: Intrinsics. A plausible focal length rather than a measured one: reported
#: sizes are approximate without a calibration, but reporting none would report
#: no size at all.
DEFAULT_CAMERA_FX = 1000.0
DEFAULT_CAMERA_FY = 1000.0
DEFAULT_CAMERA_CX = 960.0
DEFAULT_CAMERA_CY = 540.0

#: Both spellings of each field, because callers send both. `*_meters` is what the
#: forms and the calibration schema use; the short form is what this service's own
#: dictionaries use.
_HEIGHT_KEYS = ("camera_height_m", "camera_height_meters")
_TILT_KEYS = ("camera_tilt_deg", "camera_tilt_degrees")
_INTRINSIC_KEYS = ("fx", "fy", "cx", "cy")


class CameraGeometry:
    """A resolved camera pose and lens, as the vision service takes it."""

    __slots__ = ("height_m", "tilt_deg", "fx", "fy", "cx", "cy", "distortion_coefficients")

    def __init__(
        self,
        height_m: float,
        tilt_deg: float,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        distortion_coefficients: Optional[str] = None,
    ) -> None:
        self.height_m = float(height_m)
        self.tilt_deg = float(tilt_deg)
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        # Already serialised, because that is what the vision service stores and
        # what a stored calibration profile holds.
        self.distortion_coefficients = distortion_coefficients

    def as_tuple(self) -> tuple:
        """The seven values in the order the pipeline has always passed them."""
        return (
            self.height_m,
            self.tilt_deg,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.distortion_coefficients,
        )

    def as_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments for `VisionAnalysisService`."""
        return {
            "camera_height_m": self.height_m,
            "camera_tilt_deg": self.tilt_deg,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "distortion_coefficients": self.distortion_coefficients,
        }

    def __repr__(self) -> str:  # pragma: no cover - for log lines only
        return (
            f"CameraGeometry(height_m={self.height_m}, tilt_deg={self.tilt_deg}, "
            f"fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy})"
        )


def _number(value: Any) -> Optional[float]:
    """A finite float, or None.

    Booleans are excluded deliberately: `True` is an int in Python, and a
    calibration field that arrives as `true` is a malformed request rather than a
    height of one metre.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _first_number(source: Mapping[str, Any], keys) -> Optional[float]:
    for key in keys:
        number = _number(source.get(key))
        if number is not None:
            return number
    return None


def serialize_distortion(value: Any) -> Optional[str]:
    """Distortion coefficients as the JSON array string OpenCV is given.

    Three forms are accepted: a list of numbers, a JSON array string, and the
    legacy string a stored profile may hold where the numbers are separated by
    commas or spaces.

    None when the value is absent or not wholly numeric. A partly malformed list
    is dropped whole rather than passed on: half a calibration is not a better
    answer than none, and the caller cannot tell which half it got.
    """
    import json

    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        numbers = [_number(item) for item in value]
        if not numbers or any(number is None for number in numbers):
            return None
        return json.dumps([float(number) for number in numbers])

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.startswith("["):
            # Already a JSON array, but it still has to be numbers: a string that
            # merely looks like one is how "null" would reach OpenCV.
            try:
                parsed = json.loads(text)
            except ValueError:
                return None
            return serialize_distortion(parsed) if isinstance(parsed, list) else None
        # Comma- or space-separated legacy form.
        parts = [part for part in text.replace(",", " ").split() if part]
        numbers = [_number(part) for part in parts]
        if not numbers or any(number is None for number in numbers):
            return None
        return json.dumps([float(number) for number in numbers])

    return None


def _candidate_blocks(payload: Mapping[str, Any]) -> list:
    """The mappings that may hold camera fields, most specific first.

    Three shapes are in use and all of them reach this service: the forms send
    `{camera_height_meters, camera_tilt_degrees, intrinsics: {...}}`; the
    checkerboard route answers `{camera_metadata: {..., intrinsics: {...}}}` and a
    caller may hand that response straight back as the payload; and a caller who
    saved a profile may flatten the intrinsics to the top level.
    """
    blocks: list = []
    for container in (payload, payload.get("camera_metadata")):
        if not isinstance(container, Mapping):
            continue
        nested = container.get("intrinsics")
        if isinstance(nested, Mapping):
            blocks.append(nested)
        blocks.append(container)
    return blocks


def normalise_calibration_payload(payload: Any) -> Dict[str, Any]:
    """The recognised camera fields of an analysis payload, and nothing else.

    Accepts the flat form (`camera_height_meters`), the nested intrinsics the
    checkerboard route produces, and both at once. Deliberately fills in no
    defaults: this says what the payload said, and `resolve_camera_geometry` says
    what the pipeline will assume. Splitting those keeps "the caller sent 1.5" and
    "nothing was sent" distinguishable instead of collapsing both into the same
    number.

    A payload that is absent or unparseable yields an empty dict rather than an
    error. Refusing to analyze a survey because a metadata string would not parse
    would hide the thing being analyzed.
    """
    if payload is None:
        return {}
    if isinstance(payload, (str, bytes, bytearray)):
        import json

        try:
            payload = json.loads(payload)
        except (TypeError, ValueError):
            return {}
    if not isinstance(payload, Mapping):
        return {}

    blocks = _candidate_blocks(payload)
    out: Dict[str, Any] = {}
    for keys, field in ((_HEIGHT_KEYS, "camera_height_m"), (_TILT_KEYS, "camera_tilt_deg")):
        for block in blocks:
            number = _first_number(block, keys)
            if number is not None:
                out[field] = number
                break

    for key in _INTRINSIC_KEYS:
        for block in blocks:
            number = _number(block.get(key))
            if number is not None:
                out[key] = number
                break

    for block in blocks:
        distortion = serialize_distortion(block.get("distortion_coefficients"))
        if distortion is not None:
            out["distortion_coefficients"] = distortion
            break

    already = payload.get("already_undistorted")
    if isinstance(already, bool):
        out["already_undistorted"] = already
    return out


def resolve_camera_geometry(
    overrides: Any = None,
    stored: Optional[Mapping[str, Any]] = None,
) -> CameraGeometry:
    """The camera the analysis will use, from the payload and any saved profile.

    Precedence is payload, then saved profile, then the module defaults. The
    payload wins because it describes the mount this upload was made from; a
    profile is the operator's remembered answer for a device, which is the best
    available when the payload says nothing; the constants are the last resort
    and are the same for every route.

    The mounting (height and tilt) is resolved per field, because a caller may
    know the mounting and not the lens. The intrinsics are not: they are taken as
    a group from whichever source has any of them. Taking fx from the payload and
    cy from a profile would build a camera matrix describing neither the lens nor
    the vehicle camera, and nothing downstream could tell.

    `stored` may be a `CalibrationProfile` row or any mapping with the same field
    names, so the caller does not have to reshape it first.
    """
    from_payload = normalise_calibration_payload(overrides)
    from_profile = normalise_calibration_payload(_profile_fields(stored))

    def pick(field: str, default: float) -> float:
        for source in (from_payload, from_profile):
            number = _number(source.get(field))
            if number is not None:
                return number
        return default

    # Whichever source carries any intrinsics supplies all four; the other is not
    # consulted for the ones it lacks.
    if any(key in from_payload for key in _INTRINSIC_KEYS):
        lens = from_payload
    elif any(key in from_profile for key in _INTRINSIC_KEYS):
        lens = from_profile
    else:
        lens = {}

    def lens_value(field: str, default: float) -> float:
        number = _number(lens.get(field))
        return default if number is None else number

    distortion = from_payload.get("distortion_coefficients") or from_profile.get(
        "distortion_coefficients"
    )

    return CameraGeometry(
        height_m=pick("camera_height_m", DEFAULT_CAMERA_HEIGHT_M),
        tilt_deg=pick("camera_tilt_deg", DEFAULT_CAMERA_TILT_DEG),
        fx=lens_value("fx", DEFAULT_CAMERA_FX),
        fy=lens_value("fy", DEFAULT_CAMERA_FY),
        cx=lens_value("cx", DEFAULT_CAMERA_CX),
        cy=lens_value("cy", DEFAULT_CAMERA_CY),
        distortion_coefficients=distortion,
    )


def _profile_fields(stored: Any) -> Dict[str, Any]:
    """A saved calibration profile's camera fields, as a mapping.

    Accepts a `CalibrationProfile` row or any mapping with the same names, so the
    caller does not reshape it first.

    Its intrinsics are kept as a group under one key. A profile that saved a focal
    length but no principal point did not record a calibration, so the half it
    lacks has to stay on the defaults rather than being completed from the
    payload's other half — a focal length from one calibration and a principal
    point from another is a camera matrix that describes neither camera.
    """
    if stored is None:
        return {}

    if isinstance(stored, Mapping):
        source: Dict[str, Any] = dict(stored)
    else:
        source = {
            name: getattr(stored, name)
            for name in (
                "camera_height_m",
                "camera_height_meters",
                "camera_tilt_deg",
                "camera_tilt_degrees",
                "fx",
                "fy",
                "cx",
                "cy",
                "distortion_coefficients",
            )
            if getattr(stored, name, None) is not None
        }

    present = {key: source[key] for key in _INTRINSIC_KEYS if source.get(key) is not None}
    if not present:
        return source

    if source.get("distortion_coefficients"):
        present["distortion_coefficients"] = source["distortion_coefficients"]
    remaining = {
        key: value
        for key, value in source.items()
        if key not in (*_INTRINSIC_KEYS, "distortion_coefficients")
    }
    remaining["intrinsics"] = present
    return remaining
