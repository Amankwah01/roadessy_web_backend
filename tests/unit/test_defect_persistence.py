"""Defect records must carry a position, not only a chainage.

The vision service resolved a lat/lon for every detection from the interpolated
GPS track and carried it on the record as a GeoJSON point. `_persist_defects`
picked up chainage and confidence from those records and dropped the position,
so `tracked_defects.geom_location` was null for every row ever written and the
only way to put a defect on a map was to re-run its video.
"""
import pytest

from app.services.pipeline.heavy_compute import HeavyComputePipeline


def detection(lat=6.298762, lon=-1.652243, **extra):
    record = {
        "defect_type": "Pothole",
        "area": 0.25,
        "width": 0.5,
        "depth_mm": 30.0,
        "deduct_value": 8.0,
        "astm_distress_type": "Potholes",
        "ghana_distress_type": "Pothole",
        "yolo_class_id": 4,
        "confidence": 0.4,
        "chainage_m": 120.5,
        "geom": {"type": "Point", "coordinates": [lon, lat]},
    }
    record.update(extra)
    return record


def as_text(value):
    """Geometry values are WKT objects on PostGIS and plain text on SQLite."""
    return str(value)


class TestDetectionPoint:
    def test_a_positioned_detection_yields_a_point(self):
        point = HeavyComputePipeline._detection_point(detection())
        assert "POINT" in as_text(point).upper()

    def test_coordinates_are_lon_then_lat(self):
        """GeoJSON order, the reverse of the (lat, lon) used elsewhere here.

        Reading these the other way round puts a defect in the wrong hemisphere,
        which is invisible on a map until someone checks a known landmark.
        """
        point = as_text(HeavyComputePipeline._detection_point(detection(lat=6.298762, lon=-1.652243)))
        assert "-1.652243" in point
        assert "6.298762" in point
        assert point.index("-1.652243") < point.index("6.298762")

    def test_a_detection_without_gps_yields_nothing(self):
        """(0.0, 0.0) is the service's 'no GPS' value, in the Gulf of Guinea."""
        assert HeavyComputePipeline._detection_point(detection(lat=0.0, lon=0.0)) is None

    def test_a_detection_with_no_geom_yields_nothing(self):
        assert HeavyComputePipeline._detection_point({"chainage_m": 10.0}) is None

    def test_a_detection_with_null_geom_yields_nothing(self):
        assert HeavyComputePipeline._detection_point(detection(geom=None)) is None

    def test_empty_coordinates_yield_nothing(self):
        assert HeavyComputePipeline._detection_point(detection(geom={"type": "Point", "coordinates": []})) is None

    def test_unparseable_coordinates_yield_nothing(self):
        record = detection(geom={"type": "Point", "coordinates": ["north", "west"]})
        assert HeavyComputePipeline._detection_point(record) is None

    def test_one_dimensional_coordinates_yield_nothing(self):
        record = detection(geom={"type": "Point", "coordinates": [1.0]})
        assert HeavyComputePipeline._detection_point(record) is None

    def test_out_of_range_coordinates_yield_nothing(self):
        assert HeavyComputePipeline._detection_point(detection(lat=95.0, lon=-1.6)) is None
        assert HeavyComputePipeline._detection_point(detection(lat=6.2, lon=200.0)) is None

    @pytest.mark.parametrize("lon", [-179.9, -0.001, 0.001, 179.9])
    def test_legitimate_edge_longitudes_are_kept(self, lon):
        assert HeavyComputePipeline._detection_point(detection(lon=lon)) is not None

    def test_southern_and_western_hemispheres_are_kept(self):
        """A Ghana route is northern and western, but the column is not GH-only."""
        assert HeavyComputePipeline._detection_point(detection(lat=-33.86, lon=151.2)) is not None
