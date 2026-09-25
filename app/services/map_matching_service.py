# app/services/map_matching_service.py
"""
Compatibility wrapper for the consolidated spatial map matching implementation.

The actual implementation now lives in app.services.spatial_service so the
project has one authoritative map-matching engine.
"""

from app.services.spatial_service import (
    CoordinateSystem,
    MapMatchedPoint,
    MapMatchingIntegration,
    RoadNetworkBuffer,
    RoadSegment,
    SpatialGeospatialService,
)

__all__ = [
    "CoordinateSystem",
    "MapMatchedPoint",
    "MapMatchingIntegration",
    "RoadNetworkBuffer",
    "RoadSegment",
    "SpatialGeospatialService",
]
