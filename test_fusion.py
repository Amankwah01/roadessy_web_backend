"""Test the fusion service with the new dual-pipeline architecture."""
from app.services.fusion_service import PavementConditionFusioneer
from app.services.inertial_service import IRISegmentResult, InertialPipelineResult, ASTMQCMResult
from app.services.vision_service import ASTMVisionResult, VideoQualityMetrics

# Simulate simple IRI result
simple_result = InertialPipelineResult(
    iri_m_km=2.5,
    total_distance_m=100.0,
    iri_segments=[IRISegmentResult(0, 50, 2.5, 10), IRISegmentResult(50, 100, 2.5, 10)],
    raw_timestamps=None,
    vertical_displacement=None,
)

# Simulate ASTM QCM result
astm_qcm = ASTMQCMResult(
    iri_m_km=3.0,
    total_distance_m=100.0,
    iri_segments=[IRISegmentResult(0, 50, 3.0, 10), IRISegmentResult(50, 100, 3.0, 10)],
    sample_rate_hz=100.0,
)

# Simulate empty vision detections (no defects)
empty_vision = ASTMVisionResult(
    detections=[],
    video_quality=VideoQualityMetrics(
        fps=0, resolution=(0, 0), motion_blur_score=0, illumination_score=0,
        gps_sync_quality=0, fps_flag="INSUFFICIENT", frames_analyzed=0, frames_skipped=0
    ),
    frame_sampling=None,
    astm_distress_mapping={}, ghana_distress_mapping={}
)

# Test Roadessy pipeline (defects only, IRI separate)
roadessy_result = PavementConditionFusioneer.compute_roadessy_pipeline(
    simple_result=simple_result,
    roadessy_vision=empty_vision,
    run_id='test',
)
print('Roadessy PCI:', roadessy_result["roadessy_pci"])
print('Simple IRI:', roadessy_result["simple_iri_m_km"])

# Test ASTM pipeline (defects only, IRI separate)
astm_result = PavementConditionFusioneer.compute_astm_pipeline(
    astm_qcm=astm_qcm,
    astm_vision=empty_vision,
    run_id='test',
)
print('ASTM PCI:', astm_result["astm_pci"])
print('ASTM QCM IRI:', astm_result["qcm_iri_m_km"])
print('Fusion test passed!')