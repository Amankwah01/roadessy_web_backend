"""
RobustPotholeSegmentor - Multi-modal pothole segmentation (Depth + RGB fusion)
Based on pothole-depth-fusion pipeline/rgbsegment.py
Handles both dry and water-filled potholes
"""
import numpy as np
import cv2
from typing import Tuple, Optional, List
from dataclasses import dataclass
import numpy.typing as npt

@dataclass
class SegmentationResult:
    """Result of pothole segmentation."""
    mask: Optional[npt.NDArray[np.uint8]]
    method: str  # "depth", "rgb", "hybrid", "fallback"
    confidence: float
    metadata: dict


class RobustPotholeSegmentor:
    """
    Multi-modal pothole segmentor that handles both dry and water-filled potholes.
    
    Strategies (in priority order):
    1. Depth-based segmentation (primary, works for dry potholes)
    2. RGB-based segmentation (fallback, works for water-filled)
    3. Hybrid fusion (combines both for best results)
    4. Fallback: Bounding box method
    
    Based on pothole-depth-fusion pipeline/rgbsegment.py
    """
    
    def __init__(self, 
                 depth_threshold_m: float = 0.02,      # 2cm depth threshold
                 min_area_pixels: int = 100,
                 use_rgb_fallback: bool = True):
        """
        Args:
            depth_threshold_m: Minimum depth difference to consider as pothole (2cm)
            min_area_pixels: Minimum connected component area to keep
            use_rgb_fallback: Enable RGB-based segmentation when depth fails
        """
        self.depth_threshold_m = depth_threshold_m
        self.min_area_pixels = min_area_pixels
        self.use_rgb_fallback = use_rgb_fallback
    
    def segment_pothole(self,
                        rgb_image: np.ndarray,
                        depth_map: np.ndarray,
                        bbox: Tuple[int, int, int, int],
                        ground_ref: float) -> Tuple[Optional[np.ndarray], str]:
        """
        Main segmentation method.
        
        Returns:
            mask: Binary mask (uint8) where 255 = pothole
            method: String indicating which method succeeded ("depth", "rgb", "hybrid", or "fallback")
        """
        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape
        
        # Ensure bbox within bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)
        
        if x1 >= x2 or y1 >= y2:
            return None, "invalid_bbox"
        
        # Extract ROIs
        roi_depth = depth_map[y1:y2, x1:x2].copy()
        roi_rgb = rgb_image[y1:y2, x1:x2].copy()
        
        # ============================================================
        # METHOD 1: Depth-based segmentation (primary)
        # ============================================================
        mask_depth = self._segment_by_depth(roi_depth, ground_ref)
        
        if mask_depth is not None and np.sum(mask_depth) >= self.min_area_pixels:
            return mask_depth, "depth"
        
        # ============================================================
        # METHOD 2: RGB-based segmentation (fallback for water)
        # ============================================================
        mask_rgb = None
        if self.use_rgb_fallback:
            mask_rgb = self._segment_by_rgb(roi_rgb, roi_depth, ground_ref)
            
            if mask_rgb is not None and np.sum(mask_rgb) >= self.min_area_pixels:
                return mask_rgb, "rgb_water"
        
        # ============================================================
        # METHOD 3: Hybrid fusion (combine both)
        # ============================================================
        if self.use_rgb_fallback and mask_depth is not None and mask_rgb is not None:
            mask_hybrid = self._fuse_masks(mask_depth, mask_rgb)
            if np.sum(mask_hybrid) >= self.min_area_pixels:
                return mask_hybrid, "hybrid"
        
        # ============================================================
        # FALLBACK: Bounding box method
        # ============================================================
        return None, "bounding_box_fallback"
    
    def _segment_by_depth(self, roi_depth: np.ndarray, ground_ref: float) -> Optional[np.ndarray]:
        """
        Segment pothole using depth discontinuity.
        Works well for dry potholes where depth drops significantly.
        """
        # Detect depth orientation (some cameras have inverted depth)
        h, w = roi_depth.shape
        is_inverted = np.median(roi_depth[3*h//4:, :]) > np.median(roi_depth[:h//4, :])
        
        if not is_inverted:
            # Normal: pothole is deeper than road
            mask = (ground_ref - roi_depth) > self.depth_threshold_m
        else:
            # Inverted: try both directions
            mask_normal = (ground_ref - roi_depth) > self.depth_threshold_m
            mask_inverted = (roi_depth - ground_ref) > self.depth_threshold_m
            mask = mask_normal | mask_inverted
        
        if np.sum(mask) < self.min_area_pixels:
            return None
        
        # Clean mask
        mask = self._clean_mask(mask.astype(np.uint8))
        
        return mask
    
    def _segment_by_rgb(self, roi_rgb: np.ndarray, roi_depth: np.ndarray, ground_ref: float) -> Optional[np.ndarray]:
        """
        Segment water-filled potholes using RGB cues.
        
        Water characteristics:
        - Darker than surrounding road
        - Smooth texture (low variance)
        - May have reflections (sky-colored patches)
        - Often has darker boundary (wet edge)
        """
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi_rgb, cv2.COLOR_BGR2GRAY)
        
        # ============================================================
        # CUE 1: Dark region detection
        # Water appears darker than surrounding road
        # ============================================================
        # Compute local brightness relative to surrounding
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        relative_brightness = blur.astype(np.float32) - gray.astype(np.float32)
        dark_mask = relative_brightness > 15  # Darker than local average
        
        # ============================================================
        # CUE 2: Texture analysis (water is smooth)
        # ============================================================
        # Compute local variance (texture)
        mean = cv2.GaussianBlur(gray, (5, 5), 0)
        variance = cv2.GaussianBlur(gray.astype(np.float32)**2, (5, 5), 0) - mean**2
        smooth_mask = variance < 50  # Low variance = smooth surface
        
        # ============================================================
        # CUE 3: Saturation analysis (water has low saturation)
        # ============================================================
        saturation = hsv[:, :, 1]
        low_saturation_mask = saturation < 50
        
        # ============================================================
        # CUE 4: Edge detection (water has weak internal edges)
        # ============================================================
        edges = cv2.Canny(gray, 50, 150)
        weak_edge_mask = edges < 10
        
        # ============================================================
        # CUE 5: Wet edge detection (darker ring around water)
        # ============================================================
        # Dilate the dark mask to find boundary
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated_dark = cv2.dilate(dark_mask.astype(np.uint8), kernel)
        wet_edge = (dilated_dark > 0) & (~dark_mask)
        
        # ============================================================
        # Combine cues
        # ============================================================
        # Primary: dark + smooth + low saturation
        primary_mask = dark_mask & smooth_mask & low_saturation_mask
        
        # Secondary: include weak edge regions
        secondary_mask = primary_mask & weak_edge_mask
        
        # Add wet edge boundary
        final_mask = secondary_mask | wet_edge
        
        # ============================================================
        # Post-processing
        # ============================================================
        final_mask = self._clean_mask(final_mask.astype(np.uint8))
        
        if np.sum(final_mask) < self.min_area_pixels:
            return None
        
        return final_mask
    
    def _segment_by_reflection(self, roi_rgb: np.ndarray, roi_depth: np.ndarray) -> Optional[np.ndarray]:
        """
        Specialized segmentation for potholes with sky/cloud reflections.
        """
        hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_BGR2HSV)
        
        # Sky colors typically have:
        # - Hue: 90-130 (blue/cyan)
        # - Saturation: low to medium (20-80)
        # - Value: high (150-255)
        sky_hue_mask = (hsv[:, :, 0] > 90) & (hsv[:, :, 0] < 130)
        sky_sat_mask = (hsv[:, :, 1] > 20) & (hsv[:, :, 1] < 80)
        sky_val_mask = hsv[:, :, 2] > 150
        
        sky_mask = sky_hue_mask & sky_sat_mask & sky_val_mask
        
        # Combine with depth info (sky reflection appears at ground depth)
        depth_mask = np.abs(roi_depth - np.median(roi_depth)) < 0.5  # Near ground level
        combined = sky_mask & depth_mask
        
        mask = self._clean_mask((combined * 255).astype(np.uint8))
        
        if np.sum(mask) < self.min_area_pixels:
            return None
        
        return mask
    
    def _segment_by_depth_simple(self, roi_depth: np.ndarray, ground_ref: float) -> Optional[np.ndarray]:
        """Segment pothole using depth discontinuity (simple version)."""
        mask = (ground_ref - roi_depth) > self.depth_threshold_m
        
        if np.sum(mask) < self.min_area_pixels:
            return None
        
        return self._clean_mask(mask.astype(np.uint8))
    
    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Clean binary mask with morphological operations."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Remove small regions
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < self.min_area_pixels:
                mask[labels == i] = 0
        
        return mask
    
    def _fuse_masks(self, mask_depth: np.ndarray, mask_rgb: np.ndarray) -> np.ndarray:
        """Fuse depth and RGB masks using OR operation."""
        if mask_depth is None:
            return mask_rgb
        if mask_rgb is None:
            return mask_depth
        
        # Union of both masks
        fused = cv2.bitwise_or(mask_depth, mask_rgb)
        
        # Clean
        return self._clean_mask(fused)
    
    def get_contour(self, mask: np.ndarray) -> List[np.ndarray]:
        """Extract contour from mask."""
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # Take largest contour
        largest = max(contours, key=cv2.contourArea)
        
        # Simplify contour (Douglas-Peucker)
        epsilon = 0.005 * cv2.arcLength(largest, True)
        simplified = cv2.approxPolyDP(largest, epsilon, True)
        
        return [simplified]


def create_segmentor(depth_threshold_m: float = 0.02,
                     min_area_pixels: int = 100,
                     use_rgb_fallback: bool = True) -> RobustPotholeSegmentor:
    """Factory function to create segmentor with default parameters."""
    return RobustPotholeSegmentor(
        depth_threshold_m=depth_threshold_m,
        min_area_pixels=min_area_pixels,
        use_rgb_fallback=use_rgb_fallback
    )