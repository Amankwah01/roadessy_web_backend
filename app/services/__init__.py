# app/services/__init__.py

from app.services.fusion_service import PavementConditionFusioneer
from app.services.heavy_compute import HeavyComputePipeline, run_heavy_pipeline
from app.services.inertial_service import (
    InertialPipelineResult,
    InertialRefineryService,
    InertialRefineryService as InertialService,
)
from app.services.spatial_service import (
    CoordinateSystem,
    MapMatchedPoint,
    MapMatchingIntegration,
    RoadNetworkBuffer,
    RoadSegment,
    SpatialGeospatialService,
    SpatialService,
)
from app.services.vision_service import VisionAnalysisService


class OpticalService:
    """Lightweight compatibility shim for routes that import OpticalService."""

    pass


__all__ = [
    "InertialService",
    "InertialRefineryService",
    "InertialPipelineResult",
    "SpatialService",
    "SpatialGeospatialService",
    "CoordinateSystem",
    "MapMatchedPoint",
    "MapMatchingIntegration",
    "RoadNetworkBuffer",
    "RoadSegment",
    "VisionAnalysisService",
    "PavementConditionFusioneer",
    "HeavyComputePipeline",
    "run_heavy_pipeline",
    "OpticalService",
]