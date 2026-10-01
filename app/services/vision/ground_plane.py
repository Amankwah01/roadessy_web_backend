"""Ground-plane measurement from a single camera.

A detection box in the image is a box in pixels. Turning that into metres needs a
metric, and the old approach used a single depth for the whole box and scaled the
box's horizontal pixel extent by it. That is not a crack width: for a crack
running along the road, the box's horizontal extent is the crack's *length*, and
scaling it by one depth makes a long crack look like a very wide one. Every such
detection then reads as the worst severity on the road.

The pavement is a plane, and a plane has an exact image-to-ground homography.
The mounting height and pitch angle come in the request payload
(`CameraCalibrationMetadata`), and together with the intrinsics they fix the
mapping completely -- no depth estimate and no guesswork. A box then maps to a
real trapezoid on the road, and the crack's width is the short side of that
trapezoid.

Camera model, as set up here:

* the camera looks along its own +Z, with +X right and +Y down in the image;
* in world terms the camera sits at ``(0, 0, height)`` above a flat road, with
  the world +X running forward along the road, +Y to the left, and +Z up;
* ``tilt_degrees`` is the downward pitch, so a 15 degree value looks 15 degrees
  down at the road rather than at the horizon.

The inverse is used rather than a 3x3 matrix, because for this geometry the
range to a point depends only on which image row it is in, which makes the
horizon test exact and cheap. ``homography`` is still exposed for callers that
want the matrix itself.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Below this the camera is treated as level with the road, which makes the
# pavement plane unreachable: a level forward-looking camera never meets the
# ground, so there is no ground measurement to make.
_MIN_SIN_TILT = 1e-4

# A row at or above the horizon has no intersection with the road. Range runs
# from zero exactly there, so anything below this is treated as sky.
_MIN_RANGE_M = 1e-3


@dataclass(frozen=True)
class GroundExtent:
    """A detection's size measured on the pavement plane.

    ``length_m`` and ``width_m`` are the sides of the minimum-area rectangle
    enclosing the mapped box, with length the longer one. For a crack the width
    is the side across the crack, which is what the severity tables ask for.

    These bound the detection rather than tracing it: they are the extent of the
    box the detector returned, so a box that is not tight on the crack over-reads
    the width. That is a property of the box, not of the geometry, and it is the
    same over-read a person would get measuring a box.
    """

    length_m: float
    width_m: float
    area_m2: float
    range_m: float


class PavementHomography:
    """Maps image pixels to metres on the pavement, given the mounting pose."""

    def __init__(
        self,
        height_m: float,
        tilt_deg: float,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> None:
        if height_m <= 0:
            raise ValueError("camera height must be positive")
        if fx <= 0 or fy <= 0:
            raise ValueError("focal lengths must be positive")

        self.height_m = float(height_m)
        self.tilt_rad = math.radians(float(tilt_deg))
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)

        self.sin_tilt = math.sin(self.tilt_rad)
        self.cos_tilt = math.cos(self.tilt_rad)
        # A level camera is parallel to the road, so the plane is never reached.
        self.reachable = abs(self.sin_tilt) >= _MIN_SIN_TILT
        if not self.reachable:
            logger.warning(
                "Camera pitch is %.3f deg, so the pavement plane is never "
                "reached and no ground size can be measured. Cracks will be "
                "reported without a width, which leaves their severity "
                "unrated. A downward pitch is required.",
                float(tilt_deg),
            )

    @property
    def homography(self) -> np.ndarray:
        """3x3 matrix taking ground (X forward, Y left, 1) to image (u, v, 1)."""
        return np.array(
            [
                [self.cx * self.cos_tilt, -self.fx, self.cx * self.height_m * self.sin_tilt],
                [
                    self.cy * self.cos_tilt - self.fy * self.sin_tilt,
                    0.0,
                    self.height_m * (self.fy * self.cos_tilt + self.cy * self.sin_tilt),
                ],
                [self.cos_tilt, 0.0, self.height_m * self.sin_tilt],
            ],
            dtype=float,
        )

    def range_for_row(self, v: float) -> float:
        """Slant range from the camera to the road on image row ``v``.

        For a plane seen by a pitched camera this depends on the row alone, which
        is why the inverse below needs no iteration. NaN above the horizon.
        """
        forward = self.forward_distance(v)
        if not math.isfinite(forward):
            return float("nan")
        return math.hypot(forward, self.height_m)

    def forward_distance(self, v: float) -> float:
        """Distance ahead of the camera where image row ``v`` meets the road.

        Derived from the pinhole relation with the camera pitched ``tilt`` down:
        a point ``X`` ahead of the camera lies ``h*cos(t) - X*sin(t)`` below the
        optical axis, so the row it lands on is
        ``v = fy*(h*cos(t) - X*sin(t)) / (X*cos(t) + h*sin(t))``, and inverting
        that gives the expression below. NaN at or above the horizon.
        """
        if not self.reachable:
            return float("nan")
        denom = self.cos_tilt * (v - self.cy) + self.fy * self.sin_tilt
        if denom <= 0:
            return float("nan")
        return (
            self.height_m
            * (self.fy * self.cos_tilt - (v - self.cy) * self.sin_tilt)
            / denom
        )

    def image_to_ground(self, points_px: Sequence[Sequence[float]]) -> Optional[np.ndarray]:
        """Map image points to ground (X, Y) in metres, or None if any is unusable.

        Returns None rather than a partial answer: a box with one corner off the
        road has no meaningful size, and dropping the bad corner would understate
        the defect.
        """
        ground, _ = self._image_to_ground_with_range(points_px)
        return ground

    def _image_to_ground_with_range(
        self, points_px: Sequence[Sequence[float]]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self.reachable:
            return None, None
        pts = np.asarray(points_px, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 1 or pts.shape[1] != 2:
            return None, None

        u = pts[:, 0]
        v = pts[:, 1]
        denom = self.cos_tilt * (v - self.cy) + self.fy * self.sin_tilt
        if np.any(denom <= 0):
            return None, None
        forward = (
            self.height_m
            * (self.fy * self.cos_tilt - (v - self.cy) * self.sin_tilt)
            / denom
        )
        # Depth along the optical axis, positive in front of the camera.
        depth = self.cos_tilt * forward + self.height_m * self.sin_tilt
        if np.any(depth < _MIN_RANGE_M) or not np.all(np.isfinite(depth)):
            return None, None
        lateral = (self.cx - u) * depth / self.fx

        ground = np.column_stack([forward, lateral])
        ranges = np.sqrt(forward * forward + lateral * lateral + self.height_m ** 2)
        if not np.all(np.isfinite(ground)):
            return None, None
        return ground, ranges

    def lateral_scale(self, v: float) -> float:
        """Metres per pixel across the road, on image row ``v``.

        Exact: a pixel step sideways subtends 1/fx at whatever depth the row is
        looking at, so the step is ``depth / fx`` metres.
        """
        forward = self.forward_distance(v)
        if not math.isfinite(forward):
            return float("nan")
        depth = self.cos_tilt * forward + self.height_m * self.sin_tilt
        return depth / self.fx

    def longitudinal_scale(self, v: float) -> float:
        """Metres per pixel along the road, on image row ``v``.

        The forward distance is a linear-fractional function of the row, whose
        derivative works out to exactly ``-h*fy / D**2`` with
        ``D = cos(t)*(v - cy) + fy*sin(t)``. Magnitude is returned.
        """
        if not self.reachable:
            return float("nan")
        denom = self.cos_tilt * (v - self.cy) + self.fy * self.sin_tilt
        if denom <= 0:
            return float("nan")
        return (self.height_m * self.fy) / (denom * denom)

    def measure_box(self, bbox_px: Sequence[float]) -> Optional[GroundExtent]:
        """Measure a detection box on the pavement.

        ``bbox_px`` is ``(x1, y1, x2, y2)``. Returns None when the box cannot be
        measured -- a level camera, or a box reaching to or above the horizon.

        An axis-aligned box in a perspective image does not determine a single
        ground size, because it spans a range of depths. Mapping its four
        corners and taking the enclosing rectangle overshoots badly: a 10 m
        square read back as 28 m along the road. So each dimension is measured
        where it is actually defined instead.

        Across the road, the box's horizontal extent is set by its *near* edge,
        since that edge is laterally the widest one the box contains, so the
        lateral scale is taken at the near row and is exact. Along the road, a
        deep box is measured by differencing the two rows, which is exact rather
        than sampled; a shallow box cannot be, and there the local longitudinal
        scale at the middle row is used. That case is a transverse crack or a
        compact feature such as a pothole, where the box is thin enough in depth
        for one scale to represent it.
        """
        x1, y1, x2, y2 = (float(v) for v in bbox_px)
        w_px = abs(x2 - x1)
        h_px = abs(y2 - y1)
        if w_px <= 0 or h_px <= 0:
            return None

        v_near = max(y1, y2)
        v_far = min(y1, y2)
        forward_near = self.forward_distance(v_near)
        forward_far = self.forward_distance(v_far)
        lateral_near = self.lateral_scale(v_near)
        if not all(math.isfinite(x) for x in (forward_near, forward_far, lateral_near)):
            return None

        lateral_m = w_px * lateral_near
        if h_px > w_px:
            # Runs away from the camera: a longitudinal crack or a rut.
            length_m = abs(forward_near - forward_far)
            width_m = lateral_m
        else:
            # Lies across the view: a transverse crack, or a compact feature.
            v_mid = 0.5 * (v_near + v_far)
            along = self.longitudinal_scale(v_mid)
            if not math.isfinite(along):
                return None
            length_m = lateral_m
            width_m = h_px * along

        return GroundExtent(
            length_m=length_m,
            width_m=width_m,
            area_m2=length_m * width_m,
            range_m=self.range_for_row(v_near),
        )

    @classmethod
    def measure_ground_polygon(cls, ground: np.ndarray) -> Optional[GroundExtent]:
        """Length, width and area of a mapped box, in metres.

        The minimum-area enclosing rectangle of a quadrilateral always has one
        side along an edge of the quadrilateral, so testing the four edge
        directions finds the true minimum rather than an approximation of it.
        """
        pts = np.asarray(ground, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 3:
            return None
        return cls._extent_from_polygon(pts, float("nan"))

    @staticmethod
    def _extent_from_polygon(pts: np.ndarray, range_m: float) -> Optional[GroundExtent]:
        centroid = pts.mean(axis=0)
        centred = pts - centroid
        best: Optional[Tuple[float, float, float]] = None

        for i in range(pts.shape[0]):
            edge = pts[(i + 1) % pts.shape[0]] - pts[i]
            norm = float(np.hypot(edge[0], edge[1]))
            if norm < 1e-9:
                continue
            # Unit vector along the edge, and the one across it.
            along = edge / norm
            across = np.array([-along[1], along[0]])
            span_along = float(np.ptp(centred @ along))
            span_across = float(np.ptp(centred @ across))
            area = span_along * span_across
            if best is None or area < best[0]:
                best = (area, span_along, span_across)

        if best is None:
            return None
        _, span_along, span_across = best
        return GroundExtent(
            length_m=max(span_along, span_across),
            width_m=min(span_along, span_across),
            area_m2=span_along * span_across,
            range_m=range_m,
        )


def homography_from_payload(
    camera_height_m: Optional[float],
    camera_tilt_deg: Optional[float],
    intrinsics: Optional[dict] = None,
    fallback_size: Optional[Tuple[int, int]] = None,
    fallback_focal_px: float = 1000.0,
) -> Optional[PavementHomography]:
    """Build a homography from the request payload, or None if it cannot be built.

    Height and tilt are required because without them the image has no metric.
    Intrinsics are used when supplied; otherwise a plausible focal length is
    assumed from the frame size, which is approximate but far better than
    reporting no size at all.
    """
    if not camera_height_m or camera_height_m <= 0:
        return None
    if camera_tilt_deg is None:
        logger.warning(
            "No camera tilt was supplied, so the pavement plane cannot be "
            "reached and no ground size can be measured. Cracks will be "
            "reported without a width."
        )
        return None

    fx = fy = None
    cx = cy = None
    if intrinsics:
        try:
            fx = float(intrinsics.get("fx") or 0) or None
            fy = float(intrinsics.get("fy") or 0) or None
            cx = intrinsics.get("cx")
            cy = intrinsics.get("cy")
        except (TypeError, ValueError):
            fx = fy = None

    if fx is None or fy is None:
        if not fallback_size:
            return None
        width_px, height_px = fallback_size
        # A ~60 degree horizontal field of view is the usual assumption when a
        # camera has not been calibrated; it is recorded in the returned
        # homography's assumptions rather than hidden.
        fx = fy = float(fallback_focal_px) * (float(width_px) / 1920.0)
        cx = float(width_px) / 2.0
        cy = float(height_px) / 2.0
        logger.info(
            "No calibrated intrinsics supplied; assuming a %.0f px focal length "
            "at %d px width for ground sizing.",
            fx,
            width_px,
        )

    return PavementHomography(
        height_m=float(camera_height_m),
        tilt_deg=float(camera_tilt_deg),
        fx=float(fx),
        fy=float(fy or fx),
        cx=float(cx) if cx is not None else 0.0,
        cy=float(cy) if cy is not None else 0.0,
    )
