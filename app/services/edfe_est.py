"""
EnhancedDimensionFusionEngine - Fuse dimensions from multiple methods
Based on pothole-depth-fusion pipeline/edfe_est.py

Fuses three methods:
1. MBTP (Minimum Bounding Triangulated Pixel) - PRIMARY
2. Homography mapping - fallback
3. Depth map pinhole projection - fallback
"""
import numpy as np
import cv2
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import numpy.typing as npt

from app.services.geom_anchor import GeometryAnchoredDepthScaler
from app.services.mbtp_est import MBTPAreaEstimator, MBTPEstimate
from app.services.homography import HomographyMapper


@dataclass
class DimensionEstimate:
    """Fused dimension estimate with confidence weights."""
    width_m: float
    length_m: float
    area_m2: float
    volume_m3: float
    depth_m: float
    confidence: float
    method_weights: Dict[str, float]


class EnhancedDimensionFusionEngine:
    """
    Enhanced dimension fusion that prioritizes MBTP area estimation.
    Falls back to bounding box methods when MBTP fails.
    """
    
    def __init__(self, K: np.ndarray):
        self.K = K.copy().astype(np.float64)
        self.fx = float(K[0, 0])
        self.fy = float(K[1, 1])
        self.cx = float(K[0, 2])
        self.cy = float(K[1, 2])
        self.mbtp_estimator = MBTPAreaEstimator()
    
    def update_K(self, K: np.ndarray):
        self.K = K.copy().astype(np.float64)
        self.fx = float(K[0, 0])
        self.fy = float(K[1, 1])
        self.cx = float(K[0, 2])
        self.cy = float(K[1, 2])
    
    def fuse(self,
             bbox: Tuple[int, int, int, int],
             homography_mapper: Optional[HomographyMapper],
             pitch: float, roll: float, yaw: float,
             rgb_image: np.ndarray,
             camera_height_m: float,
             abs_depth: Optional[np.ndarray]) -> DimensionEstimate:
        """
        Fuse dimensions with MBTP as primary method.
        """
        results: Dict[str, Tuple[float, float, float]] = {}
        depth_val = 0.0
        
        # Method 1: MBTP (Primary)
        if abs_depth is not None:
            try:
                mbtp = self.mbtp_estimator.estimate(
                    rgb_image, abs_depth, bbox, self.K
                )
                if mbtp.area_m2 > 0:
                    # Convert MBTP area to width/length approximation
                    bbox_w = (bbox[2] - bbox[0]) * mbtp.depth_m / self.fx
                    bbox_h = (bbox[3] - bbox[1]) * mbtp.depth_m / self.fy
                    aspect_ratio = bbox_w / max(bbox_h, 0.001)
                    
                    if aspect_ratio > 1:
                        width_m = np.sqrt(mbtp.area_m2 * aspect_ratio)
                        length_m = mbtp.area_m2 / max(width_m, 0.001)
                    else:
                        length_m = np.sqrt(mbtp.area_m2 / max(aspect_ratio, 0.001))
                        width_m = mbtp.area_m2 / max(length_m, 0.001)
                    
                    results["mbtp"] = (float(width_m), float(length_m), float(mbtp.confidence))
                    depth_val = float(mbtp.depth_m)
            except Exception:
                pass
        
        # Method 2: Homography (fallback)
        if homography_mapper is not None:
            try:
                hd = homography_mapper.bbox_world_dims(bbox, pitch, roll, yaw)
                hc = homography_mapper.confidence(pitch, roll)
                if hd["width_m"] > 0 and hd["length_m"] > 0:
                    results["homography"] = (hd["width_m"], hd["length_m"], hc * 0.7)
            except Exception as exc:
                pass
        
        # Method 3: Depth map pinhole (fallback)
        if abs_depth is not None and "mbtp" not in results:
            try:
                from app.services.dav2e import DepthAnythingV2Estimator
                d_m = DepthAnythingV2Estimator.bbox_relative_depth(abs_depth, bbox)
                if d_m > 0:
                    depth_val = d_m
                    dc = self._depth_confidence(d_m, camera_height_m)
                    dw, dl = self._depth_dims(bbox, d_m)
                    if dw > 0 and dl > 0:
                        results["depth"] = (dw, dl, dc * 0.6)
            except Exception:
                pass
        
        if not results:
            return DimensionEstimate(
                width_m=0.0, length_m=0.0, area_m2=0.0, volume_m3=0.0,
                depth_m=depth_val, confidence=0.0, method_weights={}
            )
        
        total_conf = sum(v[2] for v in results.values()) or 1.0
        width_m = sum(v[0] * v[2] for v in results.values()) / total_conf
        length_m = sum(v[1] * v[2] for v in results.values()) / total_conf
        conf = min(1.0, total_conf / max(1, len(results)))
        
        if abs_depth is not None:
            try:
                mbtp = self.mbtp_estimator.estimate(rgb_image, abs_depth, bbox, self.K)
                volume_m3 = float(mbtp.volume_m3)
            except Exception:
                volume_m3 = self._compute_volume(bbox, abs_depth, depth_val)
        else:
            volume_m3 = 0.0
        
        weights = {k: round(v[2]/total_conf, 3) for k, v in results.items()}
        
        return DimensionEstimate(
            width_m=round(width_m, 4),
            length_m=round(length_m, 4),
            area_m2=round(width_m * length_m, 6),
            volume_m3=round(volume_m3, 6),
            depth_m=round(depth_val, 3),
            confidence=round(conf, 3),
            method_weights=weights
        )
    
    def _depth_dims(self, bbox, depth_m):
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * depth_m / self.fx, (y2 - y1) * depth_m / self.fy
    
    def _depth_confidence(self, depth_m, cam_h):
        ratio = depth_m / max(cam_h, 0.1)
        if ratio < 2: return 0.95
        elif ratio < 5: return 0.85
        elif ratio < 15: return 0.70
        elif ratio < 30: return 0.50
        else: return 0.30
    
    def _compute_volume(self, bbox, abs_depth, depth_val=0.0):
        """Fallback volume computation."""
        if abs_depth is None:
            return 0.0
        x1, y1, x2, y2 = bbox
        H, W = abs_depth.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W-1, x2), min(H-1, y2)
        if x1 >= x2 or y1 >= y2:
            return 0.0
        roi_depth = abs_depth[y1:y2, x1:x2]
        roi_mask = np.ones_like(roi_depth, dtype=bool)
        ground_ref = float(np.median(abs_depth))
        pixel_area = (depth_val / 1000.0 / self.fx) * (depth_val / 1000.0 / self.fy) if depth_val > 0 else 0.001
        depth_deficit = np.maximum(ground_ref - roi_depth, 0)
        return float(np.sum(depth_deficit[roi_mask > 0]) * pixel_area)


def create_fusion_engine(K: np.ndarray) -> EnhancedDimensionFusionEngine:
    """Factory function to create fusion engine."""
    return EnhancedDimensionFusionEngine(K)