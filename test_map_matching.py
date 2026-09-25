#!/usr/bin/env python3
"""
Regression test for consolidated map matching functionality.

Covers:
- Road buffer creation
- Spatial snapping in WGS84
- Inertial GPS ingestion
- Vision detection ingestion
- Matching summary reporting
- WGS84 <-> UTM round-trip stability
- Haversine distance
- Linear chainage
- GPX trajectory parsing
- Point-to-line distance
- Batch snapping
- Buffer geometry creation
"""

from __future__ import annotations

import pytest

from app.services import MapMatchingIntegration, RoadNetworkBuffer, SpatialService
from app.services.spatial_service import (
    CoordinateSystem,
    MapMatchedPoint,
    RoadSegment,
    SpatialGeospatialService,
)


def assert_close(actual: float, expected: float, tolerance: float, label: str) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def build_inertial_shapefile_data() -> dict:
    return {
        "segments": [
            {
                "id": "SEG_A",
                "centerline": [(0.0000, 0.0000), (0.0010, 0.0000)],
                "name": "Primary Road",
                "width": 8.0,
            },
            {
                "id": "SEG_B",
                "centerline": [(0.0000, 0.0100), (0.0010, 0.0100)],
                "name": "Secondary Road",
                "width": 6.0,
            },
        ]
    }


def test_wgs84_round_trip() -> None:
    lon, lat = 5.6790, -0.1560
    easting, northing, zone, hemisphere = SpatialService.wgs84_to_utm(lon, lat)
    restored_lon, restored_lat = SpatialService.utm_to_wgs84(easting, northing, zone, hemisphere)

    assert_close(restored_lon, lon, 0.0002, "Longitude round-trip")
    assert_close(restored_lat, lat, 0.0002, "Latitude round-trip")


def test_road_buffer_snap() -> None:
    network = RoadNetworkBuffer(road_width_default=7.0, coordinate_system=CoordinateSystem.WGS84)
    segment = network.add_road_segment(
        segment_id="ROAD_001",
        centerline=[(0.0000, 0.0000), (0.0010, 0.0000)],
        road_name="Test Road",
        width_meters=8.0,
    )

    info = network.get_road_info("ROAD_001")
    assert info is not None, "Expected road info to exist"
    assert info["segment_id"] == "ROAD_001"
    assert info["road_name"] == "Test Road"
    assert info["buffer_distance_meters"] == 4.0
    assert "bounds" in info and info["bounds"] is not None

    match = network.snap_point_to_road((0.0005, 0.00001), search_radius_meters=100.0)
    assert match is not None, "Expected point to match road"
    assert match.snapped_segment_id == segment.segment_id
    assert match.is_within_buffer is True
    assert match.distance_to_road_meters < 5.0
    assert_close(match.snapped_point[0], 0.0005, 0.0002, "Snapped longitude")
    assert_close(match.snapped_point[1], 0.0, 0.0002, "Snapped latitude")


def test_map_matching_integration() -> None:
    integration = MapMatchingIntegration(build_inertial_shapefile_data())

    gps_points = [
        {"longitude": 0.0004, "latitude": 0.00002},
        {"lon": 0.0006, "lat": 0.00998},
        {"longitude": 0.1000, "latitude": 0.1000},
    ]
    gps_matches = integration.match_inertial_gps_data(gps_points)

    assert len(gps_matches) == 3
    assert gps_matches[0] is not None
    assert gps_matches[0].snapped_segment_id == "SEG_A"
    assert gps_matches[0].is_within_buffer is True
    assert gps_matches[1] is not None
    assert gps_matches[1].snapped_segment_id == "SEG_B"
    assert gps_matches[1].is_within_buffer is True
    assert gps_matches[2] is None

    vision_detections = [
        {"lon": 0.00045, "lat": 0.00001, "defect": "crack"},
        {"longitude": 0.00055, "latitude": 0.01001, "defect": "rutting"},
        {"lon": 0.50000, "lat": 0.50000, "defect": "pothole"},
    ]
    vision_matches = integration.match_vision_detections(vision_detections)

    assert len(vision_matches) == 3
    assert vision_matches[0] is not None
    assert vision_matches[0].snapped_segment_id == "SEG_A"
    assert vision_matches[1] is not None
    assert vision_matches[1].snapped_segment_id == "SEG_B"
    assert vision_matches[2] is None

    summary = integration.get_matching_summary(gps_matches + vision_matches)
    assert summary["total_points"] == 6
    assert summary["matched_points"] == 4
    assert summary["within_buffer"] == 4
    assert summary["success_rate"] == (4 / 6) * 100
    assert summary["avg_distance_to_road_meters"] >= 0.0
    assert summary["max_distance_to_road_meters"] >= summary["min_distance_to_road_meters"]

    buffer_meta = integration.create_road_buffer_polygon("SEG_A")
    assert buffer_meta is not None
    assert buffer_meta["segment_id"] == "SEG_A"
    assert "buffer_geometry" in buffer_meta


def test_haversine_distance() -> None:
    lon1, lat1 = 0.0, 0.0
    lon2, lat2 = 0.01, 0.0
    dist = SpatialGeospatialService.haversine_distance(lon1, lat1, lon2, lat2)
    assert dist > 0.0
    assert dist < 2000.0


def test_compute_linear_chainage() -> None:
    coords = [{"lon": 0.0, "lat": 0.0}, {"lon": 0.001, "lat": 0.0}, {"lon": 0.001, "lat": 0.001}]
    chainage = SpatialGeospatialService.compute_linear_chainage(coords)
    assert len(chainage) == 3
    assert chainage[0] == pytest.approx(0.0)
    assert chainage[-1] > chainage[0]


def test_point_to_line_distance() -> None:
    p = (0.0005, 0.0001)
    a = (0.0, 0.0)
    b = (0.001, 0.0)
    dist, t = SpatialGeospatialService.point_to_line_distance(p[0], p[1], a[0], a[1], b[0], b[1])
    assert dist >= 0.0
    assert 0.0 <= t <= 1.0


def test_road_network_batch_snap() -> None:
    network = RoadNetworkBuffer(road_width_default=7.0, coordinate_system=CoordinateSystem.WGS84)
    network.add_road_segment(
        segment_id="ROAD_BATCH",
        centerline=[(0.0000, 0.0000), (0.0010, 0.0000)],
        road_name="Batch Road",
        width_meters=8.0,
    )
    points = [(0.0005, 0.00001), (0.0005, 0.0100)]
    results = network.batch_snap_points(points)
    assert len(results) == 2
    assert results[0] is not None
    assert results[0].is_within_buffer is True
    assert results[1] is None


def test_road_network_get_all_segments() -> None:
    network = RoadNetworkBuffer(road_width_default=7.0, coordinate_system=CoordinateSystem.WGS84)
    network.add_road_segment(
        segment_id="SEG_1",
        centerline=[(0.0, 0.0), (0.001, 0.0)],
        road_name="Road 1",
        width_meters=8.0,
    )
    network.add_road_segment(
        segment_id="SEG_2",
        centerline=[(0.0, 0.01), (0.001, 0.01)],
        road_name="Road 2",
        width_meters=6.0,
    )
    segments = network.get_all_segments()
    assert isinstance(segments, dict)
    assert len(segments) == 2
    assert set(segments.keys()) == {"SEG_1", "SEG_2"}
    assert segments["SEG_1"]["road_name"] == "Road 1"
    assert segments["SEG_2"]["road_name"] == "Road 2"


def test_road_segment_validation() -> None:
    with pytest.raises(ValueError):
        RoadSegment(
            segment_id="BAD",
            centerline_coords=[],
            road_name="No Centerline",
            road_width_meters=8.0,
        )
    with pytest.raises(ValueError):
        RoadSegment(
            segment_id="BAD",
            centerline_coords=[(0.0, 0.0)],
            road_name="Zero Width",
            road_width_meters=0.0,
        )


def test_parse_gpx_trajectory(tmp_path) -> None:
    gpx_content = """<?xml version="1.0" encoding="UTF-8"?>
<gpx>
  <trk>
    <trkseg>
      <trkpt lat="1.0" lon="2.0"><time>2024-01-01T00:00:00Z</time></trkpt>
      <trkpt lat="1.001" lon="2.001"><time>2024-01-01T00:00:01Z</time></trkpt>
    </trkseg>
  </trk>
</gpx>"""
    gpx_path = tmp_path / "test.gpx"
    gpx_path.write_text(gpx_content)
    trajectory = SpatialGeospatialService.parse_gpx_trajectory(str(gpx_path))
    assert len(trajectory) == 2
    assert trajectory[0]["lat"] == 1.0
    assert trajectory[0]["lon"] == 2.0


def test_map_matching_summary_empty() -> None:
    integration = MapMatchingIntegration(build_inertial_shapefile_data())
    summary = integration.get_matching_summary([])
    assert summary["total_points"] == 0
    assert summary["matched_points"] == 0
    assert summary["success_rate"] == 0.0


def run_all_tests() -> None:
    test_wgs84_round_trip()
    test_road_buffer_snap()
    test_map_matching_integration()
    test_haversine_distance()
    test_compute_linear_chainage()
    test_point_to_line_distance()
    test_road_network_batch_snap()
    test_road_network_get_all_segments()
    test_road_segment_validation()
    print("test_map_matching.py: all assertions passed")


if __name__ == "__main__":
    run_all_tests()
