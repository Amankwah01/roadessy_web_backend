# app/services/__init__.py

from app.services.pipeline import HeavyComputePipeline, run_heavy_pipeline
from app.services.inertial import (
    InertialPipelineResult,
    InertialRefineryService,
    InertialRefineryService as InertialService,
)
from app.services.spatial import (
    CoordinateSystem,
    MapMatchedPoint,
    MapMatchingIntegration,
    RoadNetworkBuffer,
    RoadSegment,
    SpatialGeospatialService,
    SpatialService,
)
from app.services.vision import VisionAnalysisService
from app.services.fusion import PavementConditionFusioneer
from app.services.gps import UniversalGPSService
from app.services.imu import IMUProcessor, IMUState
from app.services.calibration import (
    CoordinateSystem as CalibCoordinateSystem,
    MapMatchedPoint as CalibMapMatchedPoint,
    MapMatchingIntegration as CalibMapMatchingIntegration,
    RoadNetworkBuffer as CalibRoadNetworkBuffer,
    RoadSegment as CalibRoadSegment,
    SpatialGeospatialService as CalibSpatialGeospatialService,
)
from app.services.astm import ASTMD6433PCI, compute_astm_pci_from_detections
from app.services.vision.depth import (
    DepthAnythingV2Estimator,
    GeometryAnchoredDepthScaler,
    MBTPAreaEstimator,
    RobustPotholeSegmentor,
    DepthEstimationService,
    EnhancedDimensionFusionEngine,
)


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
    "UniversalGPSService",
    "IMUProcessor",
    "IMUState",
    "ASTMD6433PCI",
    "compute_astm_pci_from_detections",
    "DepthAnythingV2Estimator",
    "GeometryAnchoredDepthScaler",
    "MBTPAreaEstimator",
    "RobustPotholeSegmentor",
    "DepthEstimationService",
    "EnhancedDimensionFusionEngine",
]