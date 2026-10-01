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

from app.core.config import (
    ASTM_PCI_DISABLED_REASON,
    ASTM_PCI_ENABLED,
    ASTM_QCM_IRI_ENABLED,
)
from app.core.logger import fusion_logger, monitor
from app.services.fusion.alignment import (
    MAX_PLAUSIBLE_STEP_M,
    apply_alignment,
    place_detections,
)
from app.services.inertial import ASTMQCMResult, IRISegmentResult
from app.services.vision import ASTMVisionResult, VideoQualityMetrics
from app.services.astm import ASTMD6433PCI, compute_astm_pci_from_detections


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
    # Descending lower bounds, one per class. Held as a table rather than a
    # list of (lo, hi) pairs because PCI is a continuous score: paired ranges
    # leave a one-unit gap at every boundary (85 < x < 86), and anything landing
    # in a gap matched no band at all and fell through to "Failed" — the worst
    # rating, for a road one point from Satisfactory.
    _PCI_BANDS = [
        (86.0, "Good"),
        (71.0, "Satisfactory"),
        (56.0, "Fair"),
        (41.0, "Poor"),
        (26.0, "Very Poor"),
        (11.0, "Serious"),
    ]

    # Roadessy correction factor (prototype calibration - verify against RMM manual)
    _ROADESSY_CORRECTION_FACTOR = 1.2

    @classmethod
    def _classify_pci(cls, pci: float) -> str:
        """Classify a PCI score into the 7-class system.

        A score above 100 is not a real PCI, so it is reported as Failed rather
        than the top class — an impossible value means something upstream went
        wrong, and calling it "Good" would read as a passing result.
        """
        if pci > 100.0:
            return "Failed"
        for lower_bound, label in cls._PCI_BANDS:
            if pci >= lower_bound:
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
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        Compute ASTM PCI per ASTM D6433-24 clauses 9.1-9.6.

        PCI = 100 - CDV (Corrected Deduct Value), with IRI reported separately.

        Returns ``(None, {...})`` when the PCI is muted, which is the default.
        See `app/core/config.py` for why the number is withheld and what has to
        be replaced before it can be published. The code below is intact and
        runs as soon as `ASTM_PCI_ENABLED` is set.

        This used to be described as following the Sensors 2024 paper's
        methodology. It does not: that paper has no allowable-number-of-deducts m,
        no truncation to m, and no 9.5.5.4 iteration, so on a section where one
        distress dominates it differs from the standard by about 20 PCI points.
        """
        if not ASTM_PCI_ENABLED:
            fusion_logger.warning(
                f"ASTM PCI muted: {ASTM_PCI_DISABLED_REASON}"
            )
            return None, {
                "enabled": False,
                "reason": ASTM_PCI_DISABLED_REASON,
                "qcm_iri_m_km": round(astm_qcm.iri_m_km, 4),
                "qcm_quality_flags": astm_qcm.quality_flags,
                "sections": [],
                "total_length_m": astm_qcm.total_distance_m,
            }

        result = ASTMD6433PCI.compute_pci_from_detections(
            detections=astm_vision.detections,
            total_length_m=astm_qcm.total_distance_m,
            section_length_m=50.0,
            lane_width_m=3.25,
        )

        overall_pci = result["overall_pci"]
        provenance = result["provenance"]

        breakdown = {
            "enabled": True,
            "astm_pci": round(overall_pci, 2),
            "astm_pci_class": ASTMD6433PCI._classify_pci(overall_pci),
            "qcm_iri_m_km": round(astm_qcm.iri_m_km, 4),
            "qcm_quality_flags": astm_qcm.quality_flags,
            "video_quality_fps_flag": astm_vision.video_quality.fps_flag if astm_vision.video_quality else "UNKNOWN",
            "provenance": provenance,
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

    @staticmethod
    def _detections_beyond_survey(detections: List[Dict[str, Any]], survey_end_m: float) -> int:
        """How many detections fall past the surveyed length of road.

        The survey's length comes from the vibration GPS and each detection's
        chainage from the vision GPS. Both describe the same drive, but the two
        loggers are started by hand and cover different stretches of it, so a
        detection near the end can have no 50 m row to sit in. Counting them
        keeps that visible in a log line instead of leaving a deficit that reads
        as a cleaner road than the footage shows.
        """
        if not detections or survey_end_m <= 0:
            return 0
        return sum(
            1 for det in detections if float(det.get("chainage_m") or 0.0) >= survey_end_m
        )

    @classmethod
    def _align_detections(
        cls,
        detections: List[Dict[str, Any]],
        inertial_track: Optional[Any],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], bool]:
        """Put the vision detections in the frame the 50 m IRI sections are cut from.

        The two 50 m tables are indexed by distance along the road, and the two
        distances come from two loggers started by hand. On the KNUST Obuasi
        17:12:30 trip the camera's GPS began 470 m *before* the vibration
        logger's, so a detection 500 m along the road reads as 970 m on the
        defect table — section 2 of the defect table is a different 50 m of road
        than section 2 of the IRI table, and comparing them compares two places.
        Placing each detection by its coordinates on the measured track puts both
        tables on one stretch of road.

        Without a track to align to, the detections are returned untouched and the
        alignment is reported as unaligned, so a caller can say the two tables are
        not comparable rather than presenting them as if they were.

        The third value says whether placement was actually attempted, which is
        not the same question as whether it succeeded.
        """
        if inertial_track is None:
            return list(detections), {
                "aligned": False,
                "reason": "no inertial GPS track, section indices are not comparable",
            }, False
        if not getattr(inertial_track, "is_contiguous", True):
            # Placing detections against a track that jumps would give confident
            # numbers measured from a road nobody drove. The reason is spelled
            # out because the fix is upstream: a logger that holds a stale fix.
            return list(detections), {
                "aligned": False,
                "reason": (
                    f"the inertial GPS track is not continuous: "
                    f"{getattr(inertial_track, 'teleport_count', 0)} jump(s) over "
                    f"{MAX_PLAUSIBLE_STEP_M:.0f} m, the largest "
                    f"{getattr(inertial_track, 'worst_step_m', 0.0):.0f} m. "
                    f"Its distance is not the driven road, so the section indices "
                    f"from the two sources cannot be compared."
                ),
            }, False
        alignment = place_detections(detections, inertial_track)
        summary = alignment.summary()
        if alignment.is_aligned:
            aligned = apply_alignment(detections, alignment)
        else:
            aligned = list(detections)
        return aligned, summary, True

    @staticmethod
    def _section_ready_detections(
        detections: List[Dict[str, Any]],
        alignment: Dict[str, Any],
        placement_attempted: bool,
    ) -> List[Dict[str, Any]]:
        """The detections whose section index means anything.

        Once a measured track has been supplied, a detection's chainage is only
        comparable if it was actually placed on that track. One that could not be
        — no coordinate, or a coordinate from a different road — still carries a
        distance measured from the camera's own start, and putting that into
        section 0 would assert "this pothole is in the first 50 m of the measured
        road" on no evidence at all. Those detections still count towards the
        survey's overall PCI, which is a deduct-value sum and needs no origin;
        they are only held out of the per-section table, and `alignment` reports
        how many.

        With no track at all there is nothing to be comparable *to*, so the
        detections are used as they are: that path has already warned that the
        section indices do not line up.

        A track that was supplied but could place nothing is the case that matters
        — the origins are known to differ and the projections all failed, so
        none of these distances can be trusted to mean a section of this road.
        """
        if not placement_attempted:
            return detections
        return [d for d in detections if d.get("chainage_source") == "gps_aligned"]

    @classmethod
    def _compute_astm_segment_pci(
        cls,
        astm_segments: List[IRISegmentResult],
        detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Compute ASTM PCI at 50m segment level using paper's methodology."""
        segment_pci = []
        # The table is the IRI's: one row per 50 m the survey actually measured.
        # Detection chainage comes from the vision GPS and IRI distance from the
        # vibration GPS, and the two disagree by 0.06-9.11% across the KNUST
        # Obuasi trips, so a detection can fall past the last row. Those are
        # reported by `_detections_beyond_survey` rather than quietly widening
        # the table, because a row past the survey's end would carry a PCI with no
        # IRI beside it.
        max_segments = len(astm_segments) if astm_segments else 0
        beyond = cls._detections_beyond_survey(detections, max_segments * 50.0)
        if beyond:
            fusion_logger.warning(
                f"{beyond} ASTM detection(s) fall beyond the {max_segments * 50.0:.0f} m survey "
                f"distance and are in no 50 m segment. Vision and vibration GPS disagree on survey length."
            )
        
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
            
            # Use paper's segment PCI computation.
            #
            # The chainages are rebased onto the segment first, because the
            # section calculation below compares them against a 0-50 m section
            # while the detections carry distance from the start of the survey.
            # Without the shift, only the first segment ever matched: a detection
            # 210 m along the road sat outside its own 50 m section, was treated
            # as being beyond the survey, and scored as a clean segment. That is
            # why the per-segment PCI looked uniformly 100 for most of a survey.
            result = ASTMD6433PCI.compute_pci_from_detections(
                detections=[
                    {**d, "chainage_m": float(d.get("chainage_m") or 0.0) - seg_start}
                    for d in seg_detections
                ],
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
        # Same rule as the ASTM table: the survey's own measured length defines
        # the rows, and detections past it are reported rather than given rows of
        # their own. This path used to widen the table to fit the furthest
        # detection while the ASTM path did not, so the two pipelines could report
        # a different number of segments for the same survey.
        max_segments = len(simple_segments) if simple_segments else 0
        detections_by_segment = {}
        beyond = cls._detections_beyond_survey(detections, max_segments * 50.0)
        if beyond:
            fusion_logger.warning(
                f"{beyond} Roadessy detection(s) fall beyond the "
                f"{max_segments * 50.0:.0f} m survey distance and are in no 50 m segment. "
                f"Vision and vibration GPS disagree on survey length."
            )
        for det in detections:
            chainage = float(det.get("chainage_m") or 0.0)
            seg_idx = int(chainage / 50.0)
            if seg_idx >= max_segments:
                continue
            detections_by_segment.setdefault(seg_idx, []).append(det)

        
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
        inertial_track: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Pipeline 1: ASTM Standard
        - Input: ASTM QCM IRI (100Hz) + ASTM Vision (MDPI)
        - Output: ASTM PCI (IRI reported separately per ASTM D6433)

        `inertial_track` is the measured road the IRI segments were cut from.
        Supplying it places the vision detections on that same stretch, so the
        per-segment PCI rows and the per-segment IRI rows describe the same
        50 m. Without it both tables are still produced, but their section
        indices come from two loggers with independent origins and `alignment`
        says so.
        """
        detections, alignment, placed = cls._align_detections(astm_vision.detections, inertial_track)
        aligned_vision = ASTMVisionResult(
            detections=detections,
            video_quality=astm_vision.video_quality,
            frame_sampling=astm_vision.frame_sampling,
            astm_distress_mapping=astm_vision.astm_distress_mapping,
            ghana_distress_mapping=astm_vision.ghana_distress_mapping,
        )
        if alignment.get("aligned"):
            fusion_logger.info(
                f"Aligned {alignment['placed']} detection(s) onto the measured track "
                f"(offset median {alignment['offset_median_m']:.0f} m, "
                f"spread {alignment['offset_spread_m']:.0f} m)",
                run_id=run_id,
            )
        else:
            fusion_logger.warning(
                f"Vision and IRI segment tables are not aligned: {alignment.get('reason')}. "
                f"Section indices from the two sources do not describe the same 50 m.",
                run_id=run_id,
            )

        astm_pci, astm_breakdown = cls._compute_astm_pci(astm_qcm, aligned_vision)

        # 50m segment PCI for ASTM
        segment_pci = (
            cls._compute_astm_segment_pci(
                astm_qcm.iri_segments,
                cls._section_ready_detections(detections, alignment, placed),
            )
            if ASTM_PCI_ENABLED
            else []
        )

        if astm_pci is None:
            fusion_logger.info(
                f"ASTM pipeline: PCI muted, QCM_IRI={astm_qcm.iri_m_km:.2f} "
                "(reported separately)",
                run_id=run_id,
            )
        else:
            fusion_logger.info(
                f"ASTM Pipeline: PCI={astm_pci:.2f}, QCM_IRI={astm_qcm.iri_m_km:.2f}",
                run_id=run_id,
            )

        return {
            "astm_pci": round(astm_pci, 2) if astm_pci is not None else None,
            "astm_pci_class": cls._classify_pci(astm_pci) if astm_pci is not None else None,
            "astm_pci_enabled": bool(astm_pci is not None),
            "qcm_iri_m_km": round(astm_qcm.iri_m_km, 4),
            "qcm_iri_segments_50m": [
                {"start_m": s.start_distance_m, "end_m": s.end_distance_m, "iri_m_km": round(s.iri_m_km, 4)}
                for s in astm_qcm.iri_segments
            ],
            # The PCI provenance rides with the PCI. It goes into the same JSON
            # column as the IRI quality flags rather than beside it, because a
            # stored float with no caveat is what this whole exercise exists to
            # avoid: the number reads as an ASTM result and is not one.
            "quality_flags": dict(
                astm_qcm.quality_flags or {},
                pci_provenance=astm_breakdown.get("provenance"),
            ),
            "upsampling_info": astm_qcm.upsampling_info,
            "breakdown": astm_breakdown,
            "segment_pci_50m": segment_pci,
            "alignment": alignment,
        }

    @classmethod
    @monitor(stage="pavement_condition_fusion", domain="fusion")
    def compute_roadessy_pipeline(
        cls,
        simple_result: Any,
        roadessy_vision: ASTMVisionResult,
        run_id: str = "",
        inertial_track: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Pipeline 2: Roadessy (Custom/Ghana-aligned)
        - Input: Simple IRI + Roadessy Vision (Ghana Table 3)
        - Output: Roadessy PCI = 100 - corrected(DV_sum) (IRI reported separately)

        `inertial_track` places the vision detections on the stretch of road the
        IRI segments were measured over, for the same reason and with the same
        effect as in `compute_astm_pipeline`.
        """
        detections, alignment, placed = cls._align_detections(roadessy_vision.detections, inertial_track)
        aligned_vision = ASTMVisionResult(
            detections=detections,
            video_quality=roadessy_vision.video_quality,
            frame_sampling=roadessy_vision.frame_sampling,
            astm_distress_mapping=roadessy_vision.astm_distress_mapping,
            ghana_distress_mapping=roadessy_vision.ghana_distress_mapping,
        )
        if not alignment.get("aligned"):
            fusion_logger.warning(
                f"Vision and IRI segment tables are not aligned: {alignment.get('reason')}. "
                f"Section indices from the two sources do not describe the same 50 m.",
                run_id=run_id,
            )

        roadessy_pci, roadessy_breakdown = cls._compute_roadessy_pci(
            simple_result.iri_m_km, simple_result.iri_segments, aligned_vision
        )
        
        # 50m segment PCI for Roadessy
        segment_pci = cls._compute_roadessy_segment_pci(
            simple_result.iri_segments,
            cls._section_ready_detections(detections, alignment, placed),
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
            "alignment": alignment,
        }

    # Legacy methods removed - use compute_astm_pipeline and compute_roadessy_pipeline instead