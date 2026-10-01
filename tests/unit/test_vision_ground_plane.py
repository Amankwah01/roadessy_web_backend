"""Ground sizing on the pavement plane, checked against known ground truth.

The point of these tests is that the numbers are right in metres, not merely
self-consistent. Each case lays a shape of known size on the road, projects it
through the same homography the service uses, and checks that measuring the
resulting pixel box returns that size.
"""

import math

import numpy as np
import pytest

from app.services.vision.ground_plane import PavementHomography

HEIGHT_M = 1.6
TILT_DEG = 15.0
FX = FY = 1200.0
CX, CY = 960.0, 540.0


@pytest.fixture
def pavement():
    return PavementHomography(HEIGHT_M, TILT_DEG, FX, FY, CX, CY)


def _project(pavement, x_m, y_m):
    """Ground metres to image pixels, the forward direction."""
    uvw = pavement.homography @ np.array([x_m, y_m, 1.0])
    return (uvw[0] / uvw[2], uvw[1] / uvw[2])


def _box_of(pavement, corners):
    pts = [_project(pavement, x, y) for x, y in corners]
    return (
        min(p[0] for p in pts), min(p[1] for p in pts),
        max(p[0] for p in pts), max(p[1] for p in pts),
    )


class TestHomographyRoundTrip:
    def test_ground_to_image_and_back_returns_the_same_metres(self, pavement):
        for x_m in (2.0, 5.0, 10.0, 20.0, 40.0):
            for y_m in (-1.5, 0.0, 1.5):
                u, v = _project(pavement, x_m, y_m)
                back = pavement.image_to_ground([(u, v)])
                assert back is not None
                assert back[0][0] == pytest.approx(x_m, abs=1e-6)
                assert back[0][1] == pytest.approx(y_m, abs=1e-6)

    def test_a_point_on_the_centre_line_stays_on_it(self, pavement):
        """Lateral position must not leak into forward position or vice versa."""
        for x_m in (3.0, 12.0, 30.0):
            _, v = _project(pavement, x_m, 0.0)
            back = pavement.image_to_ground([(CX, v)])[0]
            assert back[0] == pytest.approx(x_m, abs=1e-6)
            assert back[1] == pytest.approx(0.0, abs=1e-9)

    def test_range_satisfies_the_camera_triangle(self, pavement):
        """The camera is HEIGHT_M up, so range^2 = forward^2 + height^2."""
        for v in (700.0, 900.0, 1100.0):
            forward = pavement.forward_distance(v)
            assert math.hypot(forward, HEIGHT_M) == pytest.approx(
                pavement.range_for_row(v), abs=1e-9
            )

    def test_the_pitch_decides_where_the_road_meets_the_image(self, pavement):
        """A steeper camera puts the horizon higher and the road lower."""
        shallow = PavementHomography(HEIGHT_M, 5.0, FX, FY, CX, CY)
        steep = PavementHomography(HEIGHT_M, 45.0, FX, FY, CX, CY)
        horizon = lambda h: h.cy - h.fy * math.tan(h.tilt_rad)
        assert horizon(steep) < horizon(shallow)


class TestRefusals:
    def test_a_level_camera_cannot_reach_the_road(self):
        """A camera parallel to the pavement never meets it, so there is no size."""
        level = PavementHomography(HEIGHT_M, 0.0, FX, FY, CX, CY)
        assert not level.reachable
        assert level.measure_box((100, 600, 900, 1000)) is None
        assert level.image_to_ground([(CX, 900.0)]) is None

    def test_a_box_reaching_the_horizon_is_refused(self, pavement):
        horizon_v = pavement.cy - pavement.fy * math.tan(pavement.tilt_rad)
        assert pavement.image_to_ground([(CX, horizon_v - 5.0)]) is None
        assert pavement.image_to_ground([(CX, horizon_v + 5.0)]) is not None

    def test_a_box_touching_the_horizon_measures_nothing(self, pavement):
        horizon_v = pavement.cy - pavement.fy * math.tan(pavement.tilt_rad)
        assert pavement.measure_box((100.0, 100.0, 900.0, horizon_v)) is None

    def test_a_degenerate_box_has_no_size(self, pavement):
        assert pavement.measure_box((500.0, 700.0, 500.0, 900.0)) is None

    def test_a_non_positive_height_is_rejected(self):
        with pytest.raises(ValueError):
            PavementHomography(0.0, TILT_DEG, FX, FY, CX, CY)
        with pytest.raises(ValueError):
            PavementHomography(HEIGHT_M, TILT_DEG, 0.0, FY, CX, CY)


class TestSizesInMetres:
    def test_a_longitudinal_crack_measures_its_own_width(self, pavement):
        """The case the whole change exists for.

        A crack running along the road has a box that is tall and narrow. The
        old code scaled the box's horizontal pixel extent, which is the crack's
        *length*, so every one of these came out metres wide and rated the worst
        severity on the road. 50 mm has to come back as 50 mm.
        """
        corners = [(5.0, -0.025), (25.0, -0.025), (25.0, 0.025), (5.0, 0.025)]
        extent = pavement.measure_box(_box_of(pavement, corners))
        assert extent.length_m == pytest.approx(20.0, abs=0.05)
        assert extent.width_m == pytest.approx(0.050, abs=0.002)

    def test_a_transverse_crack_measures_across_its_own_run(self, pavement):
        corners = [(8.0, -1.25), (8.0, 1.25), (8.0 + 0.04, 1.25), (8.0 + 0.04, -1.25)]
        extent = pavement.measure_box(_box_of(pavement, corners))
        assert extent.length_m == pytest.approx(2.5, abs=0.05)
        assert extent.width_m == pytest.approx(0.040, abs=0.003)

    def test_a_pothole_reports_its_diameter(self, pavement):
        """Table X1.1 keys on diameter, so this has to land in the right band."""
        corners = [(6.0, -0.225), (6.0, 0.225), (6.0 + 0.45, 0.225), (6.0 + 0.45, -0.225)]
        extent = pavement.measure_box(_box_of(pavement, corners))
        assert max(extent.length_m, extent.width_m) == pytest.approx(0.45, abs=0.02)

    def test_a_crack_widens_measurably_with_its_true_width(self, pavement):
        """A wider crack must not measure the same as a hairline one."""
        def measure(true_width_m):
            half = true_width_m / 2.0
            return pavement.measure_box(
                _box_of(pavement, [(5.0, -half), (25.0, -half),
                                   (25.0, half), (5.0, half)])
            ).width_m

        assert measure(0.005) < measure(0.020) < measure(0.060)
        # And each lands near the truth, which is what the bands depend on.
        for true_width_m in (0.005, 0.020, 0.060):
            assert measure(true_width_m) == pytest.approx(true_width_m, rel=0.06)

    def test_the_same_crack_at_two_ranges_keeps_its_width(self, pavement):
        """Distance must not change the answer, only the pixel size of the box.

        Perspective makes a crack far away occupy fewer pixels; measuring it
        through the plane has to undo that. This is the property the old
        single-depth estimate could not hold.
        """
        def measure_at(x_m):
            return pavement.measure_box(
                _box_of(pavement, [(x_m, -0.03), (x_m + 20.0, -0.03),
                                   (x_m + 20.0, 0.03), (x_m, 0.03)])
            ).width_m

        near, far = measure_at(4.0), measure_at(25.0)
        assert near == pytest.approx(0.06, rel=0.05)
        assert far == pytest.approx(0.06, rel=0.05)
