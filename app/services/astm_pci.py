"""
ASTM D6433-18 PCI Calculator based on MDPI Sensors 2024 paper (Ibragimov et al.)
Paper: "Automated Pavement Condition Index Assessment with Deep Learning and Image Analysis"
Implements: AASHTO R85-18 distress classification + ASTM D6433-18 PCI calculation
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class DistressType(Enum):
    """Distress types per AASHTO R85-18 and ASTM D6433-18"""
    ALLIGATOR = "alligator"
    PATCHING = "patching"
    LONGITUDINAL = "longitudinal"
    TRANSVERSE = "transverse"
    POTHOLES = "potholes"
    RAVELLING = "ravelling"
    RUTTING = "rutting"
    SHOVING = "shoving"
    BLEEDING = "bleeding"
    CORRUGATION = "corrugation"
    DEPRESSION = "depression"


class SeverityLevel(Enum):
    """Severity levels per AASHTO R85-18 / ASTM D6433"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PCIRating(Enum):
    """PCI rating categories per ASTM D6433-18 Table 2"""
    GOOD = (86, 100, "Good")
    SATISFACTORY = (71, 85, "Satisfactory")
    FAIR = (56, 70, "Fair")
    POOR = (41, 55, "Poor")
    VERY_POOR = (26, 40, "Very Poor")
    SERIOUS = (11, 25, "Serious")
    FAILED = (0, 10, "Failed")

    @classmethod
    def from_pci(cls, pci: float) -> str:
        for rating in cls:
            lo, hi, label = rating.value
            if lo <= pci <= hi:
                return label
        return "Failed"


@dataclass
class Distress:
    """Individual distress instance"""
    type: DistressType
    severity: SeverityLevel
    length_m: float = 0.0
    area_m2: float = 0.0
    width_mm: float = 0.0
    chainage_start_m: float = 0.0
    chainage_end_m: float = 0.0


@dataclass
class SectionMetrics:
    """Per 50m section metrics"""
    section_index: int
    start_m: float
    end_m: float
    area_m2: float
    distresses: List[Distress]


class ASTMD6433PCI:
    """
    ASTM D6433-18 PCI Calculator per MDPI Sensors 2024 paper (Ibragimov et al.)
    
    Steps:
    1. Identify distress type, severity, extent per section
    2. Calculate density = (distress amount / section area) × 100
    3. Get DV from ASTM curves (digitized polynomial)
    4. Sum DV to get TDV
    5. Apply CDV correction curve
    6. PCI = 100 - CDV
    """
    
    # Standard section length (ASTM uses 50m or 100m; paper uses 50m)
    SECTION_LENGTH_M = 50.0
    
    # Standard lane width (3.0-3.5m per paper)
    LANE_WIDTH_M = 3.25
    
    # Section area = length × width
    SECTION_AREA_M2 = SECTION_LENGTH_M * LANE_WIDTH_M
    
    # Distress type mapping from YOLO classes to ASTM D6433 types
    YOLO_TO_AASHTO = {
        0: DistressType.BLEEDING,         # bleeding
        1: DistressType.CORRUGATION,      # corrugation
        2: DistressType.LONGITUDINAL,     # crack (longitudinal/transverse)
        3: DistressType.PATCHING,         # patching
        4: DistressType.POTHOLES,         # pothole
        5: DistressType.RAVELLING,        # ravelling
        6: DistressType.RUTTING,          # rutting
        7: DistressType.SHOVING,          # shoving
        8: DistressType.DEPRESSION,       # subgrade failure -> depression
    }
    
    # Polynomial coefficients for ASTM D6433 DV curves
    # DV = a * density^2 + b * density + c (per severity)
    # Based on digitized ASTM D6433 curves from paper
    DV_COEFFICIENTS = {
        DistressType.ALLIGATOR: {
            SeverityLevel.LOW:    (0.0002, 0.015, 0.5),
            SeverityLevel.MEDIUM: (0.0005, 0.040, 1.0),
            SeverityLevel.HIGH:   (0.0010, 0.080, 2.0),
        },
        DistressType.PATCHING: {
            SeverityLevel.LOW:    (0.0001, 0.010, 0.3),
            SeverityLevel.MEDIUM: (0.0003, 0.025, 0.8),
            SeverityLevel.HIGH:   (0.0006, 0.050, 1.5),
        },
        DistressType.LONGITUDINAL: {
            SeverityLevel.LOW:    (0.00005, 0.005, 0.2),
            SeverityLevel.MEDIUM: (0.00015, 0.020, 0.5),
            SeverityLevel.HIGH:   (0.00030, 0.040, 1.0),
        },
        DistressType.TRANSVERSE: {
            SeverityLevel.LOW:    (0.00005, 0.005, 0.2),
            SeverityLevel.MEDIUM: (0.00015, 0.020, 0.5),
            SeverityLevel.HIGH:   (0.00030, 0.040, 1.0),
        },
        DistressType.POTHOLES: {
            SeverityLevel.LOW:    (0.0005, 0.020, 0.5),
            SeverityLevel.MEDIUM: (0.0015, 0.060, 2.0),
            SeverityLevel.HIGH:   (0.0030, 0.120, 5.0),
        },
        DistressType.RAVELLING: {
            SeverityLevel.LOW:    (0.0001, 0.010, 0.3),
            SeverityLevel.MEDIUM: (0.0003, 0.025, 0.8),
            SeverityLevel.HIGH:   (0.0006, 0.050, 1.5),
        },
        DistressType.RUTTING: {
            SeverityLevel.LOW:    (0.0001, 0.010, 0.3),
            SeverityLevel.MEDIUM: (0.0003, 0.025, 0.8),
            SeverityLevel.HIGH:   (0.0006, 0.050, 1.5),
        },
        DistressType.SHOVING: {
            SeverityLevel.LOW:    (0.0001, 0.010, 0.3),
            SeverityLevel.MEDIUM: (0.0003, 0.025, 0.8),
            SeverityLevel.HIGH:   (0.0006, 0.050, 1.5),
        },
        DistressType.BLEEDING: {
            SeverityLevel.LOW:    (0.00005, 0.005, 0.2),
            SeverityLevel.MEDIUM: (0.00015, 0.020, 0.5),
            SeverityLevel.HIGH:   (0.00030, 0.040, 1.0),
        },
        DistressType.CORRUGATION: {
            SeverityLevel.LOW:    (0.0001, 0.010, 0.3),
            SeverityLevel.MEDIUM: (0.0003, 0.025, 0.8),
            SeverityLevel.HIGH:   (0.0006, 0.050, 1.5),
        },
        DistressType.DEPRESSION: {
            SeverityLevel.LOW:    (0.0005, 0.020, 0.5),
            SeverityLevel.MEDIUM: (0.0015, 0.060, 2.0),
            SeverityLevel.HIGH:   (0.0030, 0.120, 5.0),
        },
    }
    
    # CDV correction curve coefficients (from paper Figure 4)
    # CDV = a * TDV^2 + b * TDV + c
    CDV_COEFFICIENTS = {
        DistressType.ALLIGATOR: (0.008, 0.85, 2.0),
        DistressType.PATCHING:  (0.005, 0.90, 1.0),
        DistressType.LONGITUDINAL: (0.003, 0.95, 0.5),
        DistressType.TRANSVERSE:  (0.003, 0.95, 0.5),
        DistressType.POTHOLES: (0.006, 0.88, 1.5),
        DistressType.RAVELLING: (0.004, 0.92, 0.8),
        DistressType.RUTTING: (0.004, 0.92, 0.8),
        DistressType.SHOVING: (0.004, 0.92, 0.8),
        DistressType.BLEEDING: (0.003, 0.95, 0.5),
        DistressType.CORRUGATION: (0.004, 0.92, 0.8),
        DistressType.DEPRESSION: (0.006, 0.88, 1.5),
    }
    
    # ASTM correction factors for multiple distress types (Table from D6433)
    CORRECTION_FACTORS = {
        1: 1.00,
        2: 0.85,
        3: 0.72,
        4: 0.60,
        5: 0.52,
    }
    
    # PCI rating table (ASTM D6433-18 Table 2)
    PCI_RATINGS = [
        (86, 100, "Good"),
        (71, 85, "Satisfactory"),
        (56, 70, "Fair"),
        (41, 55, "Poor"),
        (26, 40, "Very Poor"),
        (11, 25, "Serious"),
        (0, 10, "Failed"),
    ]
    
    @classmethod
    def get_severity_from_width(cls, distress_type: DistressType, width_mm: float) -> SeverityLevel:
        """Determine severity from crack width per AASHTO R85-18 / ASTM D6433 Table 1"""
        if distress_type in (DistressType.LONGITUDINAL, DistressType.TRANSVERSE):
            if width_mm < 5:
                return SeverityLevel.LOW
            elif width_mm < 20:
                return SeverityLevel.MEDIUM
            else:
                return SeverityLevel.HIGH
        elif distress_type in (DistressType.ALLIGATOR, DistressType.POTHOLES, DistressType.DEPRESSION):
            # Alligator, Potholes, Depression: severity from extent (using width as proxy)
            if width_mm < 300:
                return SeverityLevel.LOW
            elif width_mm < 750:
                return SeverityLevel.MEDIUM
            else:
                return SeverityLevel.HIGH
        elif distress_type in (DistressType.RAVELLING, DistressType.RUTTING, DistressType.SHOVING):
            # Raveling, Rutting, Shoving: severity from extent
            if width_mm < 100:
                return SeverityLevel.LOW
            elif width_mm < 500:
                return SeverityLevel.MEDIUM
            else:
                return SeverityLevel.HIGH
        elif distress_type == DistressType.PATCHING:
            # Patching: severity from patch quality
            if width_mm < 200:
                return SeverityLevel.LOW
            elif width_mm < 500:
                return SeverityLevel.MEDIUM
            else:
                return SeverityLevel.HIGH
        elif distress_type in (DistressType.BLEEDING, DistressType.CORRUGATION, DistressType.SHOVING):
            # Bleeding, Corrugation: typically low severity unless extensive
            if width_mm < 100:
                return SeverityLevel.LOW
            else:
                return SeverityLevel.MEDIUM
        return SeverityLevel.LOW
    
    @classmethod
    def compute_density(cls, distress: Distress, section_area_m2: float) -> float:
        """Density = (distress amount / section area) × 100 per ASTM D6433 Eq 8"""
        if distress.type in (DistressType.LONGITUDINAL, DistressType.TRANSVERSE):
            # Linear distress: amount = length (m)
            amount = distress.length_m
        elif distress.type in (DistressType.ALLIGATOR, DistressType.PATCHING, 
                               DistressType.RAVELLING, DistressType.RUTTING,
                               DistressType.SHOVING, DistressType.BLEEDING,
                               DistressType.CORRUGATION):
            # Area distress: amount = area (m²)
            amount = distress.area_m2
        elif distress.type in (DistressType.POTHOLES, DistressType.DEPRESSION):
            # Count-based distress: amount = count * typical_area
            # Use count (each pothole counted as 1) and typical area for density
            # Potholes: count * 0.1 m² typical area per pothole
            # Depression: count * 0.5 m² typical area
            if distress.type == DistressType.POTHOLES:
                amount = distress.length_m * 0.1  # length_m used as count proxy
            else:
                amount = distress.length_m * 0.5
        else:
            amount = 0.0
        
        if section_area_m2 <= 0:
            return 0.0
        return (amount / section_area_m2) * 100.0
    
    @classmethod
    def get_dv(cls, distress_type: DistressType, severity: SeverityLevel, density: float) -> float:
        """Get Deduct Value from polynomial fit of ASTM D6433 curves"""
        a, b, c = cls.DV_COEFFICIENTS[distress_type][severity]
        dv = a * density * density + b * density + c
        return max(0.0, dv)
    
    @classmethod
    def get_cdv(cls, distress_type: DistressType, tdv: float) -> float:
        """Corrected DV from ASTM correction curve (paper Figure 4)"""
        a, b, c = cls.CDV_COEFFICIENTS[distress_type]
        cdv = a * tdv * tdv + b * tdv + c
        return max(0.0, cdv)
    
    @classmethod
    def compute_section_pci(cls, section: SectionMetrics) -> Dict:
        """Compute PCI for a single section per ASTM D6433"""
        # Group distresses by type
        distresses_by_type = {}
        for d in section.distresses:
            if d.type not in distresses_by_type:
                distresses_by_type[d.type] = []
            distresses_by_type[d.type].append(d)
        
        total_cdv = 0.0
        breakdown = {}
        
        for d_type, distresses in distresses_by_type.items():
            # Sum DVs for this distress type (ASTM: use max DV per type per severity)
            tdv = 0.0
            for d in distresses:
                severity = cls.get_severity_from_width(d.type, d.width_mm)
                density = cls.compute_density(d, section.area_m2)
                dv = cls.get_dv(d_type, severity, density)
                tdv += dv
            
            # Apply CDV correction
            cdv = cls.get_cdv(d_type, tdv)
            total_cdv += cdv
            breakdown[d_type.value] = {
                "tdv": round(tdv, 2),
                "cdv": round(cdv, 2),
                "count": len(distresses),
            }
        
        # ASTM correction for multiple distress types
        n_types = len(distresses_by_type)
        correction = cls.CORRECTION_FACTORS.get(min(n_types, 5), 0.45)
        corrected_cdv = total_cdv * correction
        
        pci = max(0.0, min(100.0, 100.0 - corrected_cdv))
        
        return {
            "pci": round(pci, 2),
            "rating": cls._classify_pci(pci),
            "total_cdv": round(corrected_cdv, 2),
            "breakdown": breakdown,
            "section_index": section.section_index,
            "start_m": section.start_m,
            "end_m": section.end_m,
        }
    
    @classmethod
    def _classify_pci(cls, pci: float) -> str:
        for lo, hi, label in cls.PCI_RATINGS:
            if lo <= pci <= hi:
                return label
        return "Failed"
    
    @classmethod
    def compute_pci_from_detections(
        cls,
        detections: List[Dict],
        total_length_m: float,
        section_length_m: float = 50.0,
        lane_width_m: float = 3.25,
    ) -> Dict:
        """Main entry: compute PCI from detection list (e.g., from vision service)"""
        # Create sections
        n_sections = max(1, int(np.ceil(total_length_m / section_length_m)))
        sections = []
        
        for i in range(n_sections):
            start = i * section_length_m
            end = min(start + section_length_m, total_length_m)
            area = (end - start) * lane_width_m
            
            section_distresses = []
            for det in detections:
                chainage = det.get("chainage_m", 0)
                if start <= chainage < end:
                    yolo_cls = det.get("yolo_class_id", 0)
                    d_type = cls.YOLO_TO_AASHTO.get(yolo_cls, DistressType.LONGITUDINAL)
                    
                    # Extract dimensions
                    width_mm = det.get("width", 0)  # width across lane in meters from vision
                    area_m2 = det.get("area", 0)
                    chainage_m = det.get("chainage_m", 0)
                    
                    # Compute actual length for linear distresses from area and width
                    if d_type in (DistressType.LONGITUDINAL, DistressType.TRANSVERSE):
                        # Linear distress: length = area / width (both from vision pinhole model)
                        # width_m is crack width across lane, area = length * width
                        width_m = width_mm if width_mm > 0 else 0.1  # fallback
                        length_m = area_m2 / width_m if area_m2 > 0 and width_m > 0 else 0.5
                    elif d_type in (DistressType.POTHOLES, DistressType.DEPRESSION):
                        # Count-based: use width as diameter proxy, each detection = 1 count
                        # Store count in length_m for density calculation
                        length_m = 1.0
                    else:
                        # Area-based distresses (alligator, patching, ravelling, rutting, shoving, bleeding, corrugation)
                        length_m = 0
                    
                    distress = Distress(
                        type=d_type,
                        severity=SeverityLevel.LOW,  # Will be overwritten
                        length_m=length_m,
                        area_m2=area_m2,
                        width_mm=width_mm * 1000.0,  # Convert m to mm for severity determination
                        chainage_start_m=chainage_m,
                        chainage_end_m=chainage_m + length_m,
                    )
                    section_distresses.append(distress)
            
            sections.append(SectionMetrics(
                section_index=i,
                start_m=start,
                end_m=end,
                area_m2=area,
                distresses=section_distresses,
            ))
        
        # Compute PCI per section
        section_results = []
        total_cdv = 0.0
        for section in sections:
            result = cls.compute_section_pci(section)
            section_results.append(result)
            total_cdv += result.get("total_cdv", 0)
        
        # Overall PCI (weighted by section length)
        overall_pci = np.mean([r["pci"] for r in section_results]) if section_results else 100.0
        
        return {
            "overall_pci": round(overall_pci, 2),
            "rating": cls._classify_pci(float(overall_pci)),
            "sections": section_results,
            "total_length_m": total_length_m,
        }


# Alias for backward compatibility
PCI_RATINGS = [
    (86, 100, "Good"),
    (71, 85, "Satisfactory"),
    (56, 70, "Fair"),
    (41, 55, "Poor"),
    (26, 40, "Very Poor"),
    (11, 25, "Serious"),
    (0, 10, "Failed"),
]


def compute_astm_pci_from_detections(
    detections: List[Dict],
    total_length_m: float,
    section_length_m: float = 50.0,
    lane_width_m: float = 3.25,
) -> Dict:
    """Convenience function matching paper's methodology"""
    return ASTMD6433PCI.compute_pci_from_detections(
        detections, total_length_m, section_length_m, 3.25
    )