# app/services/fusion_service.py
"""Fusion service for combining inertial IRI data and computer vision defect deductions
into Pavement Condition Index scores for two pipelines:

Pipeline 1: ASTM (International Standard) - MDPI Sensors 2024 paper methodology
- IRI: QCM at 100Hz upsampled (ASTM E1926/E950)
- Vision: MDPI Sensors 2024 paper methodology (AASHTO R85-18 + ASTM D6433-18)
- Output: ASTM PCI (IRI reported separately per ASTM D6433)

Pipeline 2: Roadessy (Custom/Ghana-aligned)
- IRI: Simple displacement-to-distance method (Enochs Thesis Eq 2.9/3.9)
- Vision: Ghana Manual 2023 Table 3 ratings + custom DV computation
- Output: Roadessy PCI = 100 - corrected(DV_sum)
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from app.core.logger import fusion_logger, monitor
from app.services.inertial_service import ASTMQCMResult, IRISegmentResult
from app.services.vision_service import ASTMVisionResult, VideoQualityMetrics
from app.services.astm_pci import ASTMD6433PCI, compute_astm_pci_from_detections


class PavementConditionFusioneer:
    """Fuses inertial IRI data and computer vision Deduct Values (DV) into 
    unified Pavement Condition Scores for two pipelines.
    """

    # ASTM D6433 Corrected Deduct Value (CDV) curves approximation
    _CORRECTION_FACTORS = {
        1: 1.00,
        2: 0.85,
        3: 0.72,
        4: 0.60,
        5: 0.52,
    }

    # Ghana Manual Table 3 rating weights per distress type
    _GHANA_DISTRESS_WEIGHTS = {
        "Potholes": 1.0,
        "Alligator Cracking": 0.9,
        "Rutting": 0.8,
        "Patching": 0.7,
        "Ravelling": 0.6,
        "Longitudinal Cracking": 0.5,
        "Transverse Cracking": 0.5,
        "Bleeding/Flushing": 0.4,
        "Corrugation/Shoving": 0.7,
        "Subgrade Failure": 1.0,
    }

    # PCI Classification (7-class system per Ghana Manual / ASTM D6433)
    _PCI_CLASSES = [
        (86, 100, "Good"),
        (71, 85, "Satisfactory"),
        (56, 70, "Fair"),
        (41, 55, "Poor"),
        (26, 40, "Very Poor"),
        (11, 25, "Serious"),
        (0, 10, "Failed"),
    ]

    # Roadessy correction factor (prototype calibration - verify against RMM manual)
    _ROADESSY_CORRECTION_FACTOR = 1.2

    @classmethod
    def _classify_pci(cls, pci: float) -> str:
        """Classify PCI score into 7-class system."""
        for lo, hi, label in cls._PCI_CLASSES:
            if lo <= pci <= hi:
                return label
        return "Failed"

    @classmethod
    def _compute_vision_pci(cls, tracking_registry: Dict[int, Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
        """Compute vision-based PCI from defect tracking registry (ASTM CDV method)."""
        deduct_values = []
        for tid, data in tracking_registry.items():
            dv = data.get("deduct_value")
            if dv is not None and dv > 0:
                deduct_values.append(float(dv))

        q_dvs = [dv for dv in deduct_values if dv >= 5.0]
        q_count = len(q_dvs)

        total_dv = float(sum(deduct_values))
        factor = cls._CORRECTION_FACTORS.get(min(q_count, 5), 0.45)
        corrected_dv = total_dv * factor

        vision_pci = max(0.0, 100.0 - corrected_dv)

        breakdown = {
            "video_dv": round(total_dv, 2),
            "total_dv": round(total_dv, 2),
            "correction_factor": factor,
            "corrected_dv": round(corrected_dv, 2),
            "defect_count": len(tracking_registry),
            "breached_types": len(set(d.get("defect_type") for d in tracking_registry.values())),
            "vision_pci": round(vision_pci, 2),
        }

        return vision_pci, breakdown

    @classmethod
    def _compute_astm_pci(
        cls,
        astm_qcm: ASTMQCMResult,
        astm_vision: ASTMVisionResult,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute ASTM PCI using paper's methodology (MDPI Sensors 2024 / ASTM D6433-18).
        
        PCI = 100 - CDV (Corrected Deduct Value)
        IRI reported SEPARATELY per ASTM D6433.
        """
        # Use paper's implementation
        result = ASTMD6433PCI.compute_pci_from_detections(
            detections=astm_vision.detections,
            total_length_m=astm_qcm.total_distance_m,
            section_length_m=50.0,
            lane_width_m=3.25,
        )
        
        overall_pci = result["overall_pci"]
        
        breakdown = {
            "astm_pci": round(overall_pci, 2),
            "astm_pci_class": ASTMD6433PCI._classify_pci(overall_pci),
            "qcm_iri_m_km": round(astm_qcm.iri_m_km, 4),
            "qcm_quality_flags": astm_qcm.quality_flags,
            "video_quality_fps_flag": astm_vision.video_quality.fps_flag if astm_vision.video_quality else "UNKNOWN",
            "sections": result["sections"],
            "total_length_m": result["total_length_m"],
        }
        
        return overall_pci, breakdown

    @classmethod
    def _compute_roadessy_pci(
        cls,
        simple_iri: float,
        simple_segments: List[IRISegmentResult],
        roadessy_vision: ASTMVisionResult,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute Roadessy PCI from defects ONLY (defects only, IRI separate).
        
        Roadessy approach (Ghana Manual 2023 aligned):
        - PCI = 100 - corrected(DV_sum) from defects only
        - IRI reported separately
        """
        # Aggregate Ghana ratings by distress type
        distress_ratings = {}
        for det in roadessy_vision.detections:
            ghana_type = det.get("ghana_distress_type", "Unknown")
            rating = det.get("ghana_rating_1to9", 1)
            weight = cls._GHANA_DISTRESS_WEIGHTS.get(ghana_type, 0.5)
            if ghana_type not in distress_ratings:
                distress_ratings[ghana_type] = {"ratings": [], "weight": weight}
            distress_ratings[ghana_type]["ratings"].append(rating)
        
        # Compute weighted DV sum
        dv_sum = 0.0
        rating_details = {}
        for distress_type, info in distress_ratings.items():
            avg_rating = np.mean(info["ratings"]) if info["ratings"] else 1
            weight = info["weight"]
            dv_sum += avg_rating * weight
            rating_details[distress_type] = {
                "avg_rating_1to9": round(avg_rating, 1),
                "weight": weight,
                "contribution": round(avg_rating * weight, 2)
            }
        
        # Roadessy: PCI from defects only, IRI separate
        # Correction factor per Ghana Manual 2023 RMM methodology
        # NOTE: Prototype calibration factor - verify against
        # [RMM] Road Condition Survey Manual 2023 when available.
        # Reference: F:\Adubofour\[RMM] Road Condtion Survey Manual 2023.pdf
        corrected_dv = dv_sum * cls._ROADESSY_CORRECTION_FACTOR
        roadessy_pci = max(0.0, min(100.0, 100.0 - corrected_dv))
        
        breakdown = {
            "roadessy_dv_sum": round(dv_sum, 2),
            "roadessy_corrected_dv": round(corrected_dv, 2),
            "roadessy_pci": round(roadessy_pci, 2),
            "roadessy_pci_class": cls._classify_pci(roadessy_pci),
            "simple_iri_m_km": round(simple_iri, 4),  # IRI reported separately
            "roadessy_rating_details": rating_details,
        }
        
        return roadessy_pci, breakdown

    @classmethod
    def _compute_astm_segment_pci(
        cls,
        astm_segments: List[IRISegmentResult],
        detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Compute ASTM PCI at 50m segment level using paper's methodology."""
        segment_pci = []
        max_segments = len(astm_segments) if astm_segments else 1
        
        for seg_idx in range(max_segments):
            seg_start = seg_idx * 50.0
            seg_end = (seg_idx + 1) * 50.0
            
            # Filter detections for this segment
            seg_detections = [
                d for d in detections
                if seg_start <= d.get("chainage_m", 0) < seg_end
            ]
            
            astm_iri = next(
                (s.iri_m_km for s in astm_segments 
                 if abs(s.start_distance_m - seg_start) < 1.0), 
                0
            )
            
            # Use paper's segment PCI computation
            result = ASTMD6433PCI.compute_pci_from_detections(
                detections=seg_detections,
                total_length_m=50.0,
                section_length_m=50.0,
                lane_width_m=3.25,
            )
            
            astm_pci_seg = result["overall_pci"]
            
            segment_pci.append({
                "segment_index": seg_idx,
                "start_m": seg_start,
                "end_m": seg_end,
                "astm_iri_m_km": round(astm_iri, 4),
                "astm_pci": round(astm_pci_seg, 2),
                "astm_pci_class": ASTMD6433PCI._classify_pci(astm_pci_seg),
            })
        
        return segment_pci

    @classmethod
    def _compute_roadessy_segment_pci(
        cls,
        simple_segments: List[IRISegmentResult],
        detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Compute Roadessy PCI at 50m segment level using defects only (IRI separate)."""
        segment_pci = []
        detections_by_segment = {}
        for det in detections:
            chainage = det.get("chainage_m", 0)
            seg_idx = int(chainage / 50.0)
            if seg_idx not in detections_by_segment:
                detections_by_segment[seg_idx] = []
            detections_by_segment[seg_idx].append(det)
        
        max_segments = max(
            len(simple_segments), 
            max(detections_by_segment.keys()) + 1 if detections_by_segment else 0
        )
        
        for seg_idx in range(max_segments):
            seg_start = seg_idx * 50.0
            seg_end = (seg_idx + 1) * 50.0
            simple_iri = next(
                (s.iri_m_km for s in simple_segments if abs(s.start_distance_m - seg_start) < 0.01), 0
            )
            seg_detections = detections_by_segment.get(seg_idx, [])
            seg_dv = sum(d.get("deduct_value", 0) for d in seg_detections)
            
            # Segment PCI from defects only, IRI separate
            corrected_dv = seg_dv * cls._ROADESSY_CORRECTION_FACTOR
            roadessy_pci_seg = max(0.0, min(100.0, 100.0 - corrected_dv))
            
            segment_pci.append({
                "segment_index": seg_idx,
                "start_m": seg_start,
                "end_m": seg_end,
                "simple_iri_m_km": round(simple_iri, 4),
                "roadessy_pci": round(roadessy_pci_seg, 2),
                "roadessy_pci_class": cls._classify_pci(roadessy_pci_seg),
            })
        
        return segment_pci

    @classmethod
    @monitor(stage="pavement_condition_fusion", domain="fusion")
    def compute_astm_pipeline(
        cls,
        astm_qcm: ASTMQCMResult,
        astm_vision: ASTMVisionResult,
        run_id: str = "",
    ) -> Dict[str, Any]:
        """
        Pipeline 1: ASTM Standard
        - Input: ASTM QCM IRI (100Hz) + ASTM Vision (MDPI)
        - Output: ASTM PCI (IRI reported separately per ASTM D6433)
        """
        astm_pci, astm_breakdown = cls._compute_astm_pci(astm_qcm, astm_vision)
        
        # 50m segment PCI for ASTM
        segment_pci = cls._compute_astm_segment_pci(
            astm_qcm.iri_segments,
            astm_vision.detections,
        )
        
        fusion_logger.info(
            f"ASTM Pipeline: PCI={astm_pci:.2f}, QCM_IRI={astm_qcm.iri_m_km:.2f}",
            run_id=run_id,
        )
        
        return {
            "astm_pci": round(astm_pci, 2),
            "astm_pci_class": cls._classify_pci(astm_pci),
            "qcm_iri_m_km": round(astm_qcm.iri_m_km, 4),
            "qcm_iri_segments_50m": [
                {"start_m": s.start_distance_m, "end_m": s.end_distance_m, "iri_m_km": round(s.iri_m_km, 4)}
                for s in astm_qcm.iri_segments
            ],
            "quality_flags": astm_qcm.quality_flags,
            "upsampling_info": astm_qcm.upsampling_info,
            "breakdown": astm_breakdown,
            "segment_pci_50m": segment_pci,
        }

    @classmethod
    @monitor(stage="pavement_condition_fusion", domain="fusion")
    def compute_roadessy_pipeline(
        cls,
        simple_result: Any,
        roadessy_vision: ASTMVisionResult,
        run_id: str = "",
    ) -> Dict[str, Any]:
        """
        Pipeline 2: Roadessy (Custom/Ghana-aligned)
        - Input: Simple IRI + Roadessy Vision (Ghana Table 3)
        - Output: Roadessy PCI = 100 - corrected(DV_sum) (IRI reported separately)
        """
        roadessy_pci, roadessy_breakdown = cls._compute_roadessy_pci(
            simple_result.iri_m_km, simple_result.iri_segments, roadessy_vision
        )
        
        # 50m segment PCI for Roadessy
        segment_pci = cls._compute_roadessy_segment_pci(
            simple_result.iri_segments,
            roadessy_vision.detections,
        )
        
        fusion_logger.info(
            f"Roadessy Pipeline: PCI={roadessy_pci:.2f}, Simple_IRI={simple_result.iri_m_km:.2f}",
            run_id=run_id,
        )
        
        return {
            "roadessy_pci": round(roadessy_pci, 2),
            "roadessy_pci_class": cls._classify_pci(roadessy_pci),
            "simple_iri_m_km": round(simple_result.iri_m_km, 4),
            "simple_iri_segments_50m": [
                {"start_m": s.start_distance_m, "end_m": s.end_distance_m, "iri_m_km": round(s.iri_m_km, 4)}
                for s in simple_result.iri_segments
            ],
            "breakdown": roadessy_breakdown,
            "segment_pci_50m": segment_pci,
        }

    # Legacy methods removed - use compute_astm_pipeline and compute_roadessy_pipeline instead