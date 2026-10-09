"""
MBTPAreaEstimator - Minimum Bounding Triangulated Pixel method for pothole area estimation.
Based on pothole-depth-fusion pipeline/mbtp_est.py

Paper: Uses triangulation of pothole contour pixels to compute accurate
area regardless of shape irregularity. Better than bounding box for
real-world potholes which are rarely rectangular.
"""
import numpy as np
import cv2
from typing import Tuple, List, Optional
from dataclasses import dataclass
import numpy.typing as npt

from app.services.vision.depth.rgbsegment import RobustPotholeSegmentor
from app.services.vision.depth.geom_anchor import GeometryAnchoredDepthScaler


@dataclass
class MBTPEstimate:
    """Result of MBTP estimation."""
    area_m2: float
    perimeter_m: float
    depth_m: float
    volume_m3: float
    convex_hull_area_m2: float
    irregularity_score: float
    confidence: float
    contour_points: List[List[float]]
    triangulation_map: Optional[np.ndarray] = None


class MBTPAreaEstimator:
    """
    Minimum Bounding Triangulated Pixel method for pothole area estimation.
    
    Paper: Uses triangulation of pothole contour pixels to compute accurate
    area regardless of shape irregularity. Better than bounding box for
    real-world potholes which are rarely rectangular.
    
    Steps:
    1. Segment pothole region from depth map (depth discontinuity)
    2. Extract contour of pothole
    3. Triangulate contour points (Delaunay triangulation)
    4. Sum triangle areas for total area
    5. Compute convex hull for shape irregularity metric
    """
    
    def __init__(self, 
                 depth_threshold_m: float = 0.02,  # 2cm depth threshold
                 min_area_pixels: int = 100,
                 contour_simplification_eps: float = 0.005,
                 use_rgb_segmentation: bool = True):
        self.depth_threshold_m = depth_threshold_m
        self.min_area_pixels = min_area_pixels
        self.contour_eps = contour_simplification_eps
        self.use_rgb_segmentation = use_rgb_segmentation
        
        # Initialize robust segmentor
        self.segmentor = RobustPotholeSegmentor(
            depth_threshold_m=depth_threshold_m,
            min_area_pixels=min_area_pixels,
            use_rgb_fallback=use_rgb_segmentation
        )


    def segment_pothole_region(self, 
                                depth_map: np.ndarray, 
                                bbox: Tuple[int, int, int, int],
                                ground_ref: float) -> np.ndarray:
        """
        Segment pothole region from depth map using depth discontinuity.
        
        Returns binary mask where True = pothole pixels.
        """
        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape
        
        # Extract ROI
        roi = depth_map[y1:y2, x1:x2].copy()
        
        # Check depth orientation
        if np.median(roi[3*roi.shape[0]//4:, :]) > np.median(roi[:roi.shape[0]//4, :]):
            # Inverted depth
            mask = (roi - ground_ref) > self.depth_threshold_m
        else:
            mask = (ground_ref - roi) > self.depth_threshold_m
        
        # Clean mask
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, 
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, 
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        
        # Remove small regions
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < self.min_area_pixels:
                mask[labels == i] = 0
        
        return mask


    def compute_contour(self, mask: np.ndarray) -> List[np.ndarray]:
        """Extract and simplify contour from mask."""
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # Take largest contour
        largest = max(contours, key=cv2.contourArea)
        
        # Simplify contour (Douglas-Peucker)
        epsilon = self.contour_eps * cv2.arcLength(largest, True)
        simplified = cv2.approxPolyDP(largest, epsilon, True)
        
        return [simplified]
    
    def triangulate_contour(self, contour: np.ndarray, image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Delaunay triangulation on contour points.
        
        Returns:
            - triangles: List of triangle indices
            - triangulation_map: Visualization of triangulation
        """
        h, w = image_shape
        points = contour.reshape(-1, 2).astype(np.float32)
        
        if len(points) < 3:
            return np.array([]), None
        
        # Add centroid to ensure full coverage
        centroid = np.mean(points, axis=0)
        points = np.vstack([points, centroid])
        
        # Delaunay triangulation
        rect = (0, 0, w, h)
        subdiv = cv2.Subdiv2D(rect)
        for p in points:
            subdiv.insert((float(p[0]), float(p[1])))
        
        triangles = subdiv.getTriangleList()
        triangles = triangles.reshape(-1, 3, 2).astype(np.int32)
        
        # Filter triangles outside contour
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        valid_triangles = []
        for tri in triangles:
            # Check if triangle centroid is inside contour
            cx = int(np.mean(tri[:, 0]))
            cy = int(np.mean(tri[:, 1]))
            if 0 <= cx < w and 0 <= cy < h and mask[cy, cx] > 0:
                valid_triangles.append(tri)
        
        # Create visualization
        viz = np.zeros((h, w, 3), dtype=np.uint8)
        for tri in valid_triangles:
            cv2.polylines(viz, [tri], True, (0, 255, 0), 1)
            cv2.fillPoly(viz, [tri], (0, 100, 0))
        
        return np.array(valid_triangles), viz
    
    def compute_triangle_area(self, triangle: np.ndarray, 
                              depth_map: np.ndarray,
                              fx: float, fy: float,
                              cx: float, cy: float,
                              ground_ref: float) -> float:
        """
        Compute real-world area of a triangle using pinhole projection.
        
        For each triangle vertex, compute:
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            Z = depth at that pixel
        
        Then compute 3D triangle area using cross product.
        """
        # Get depths at triangle vertices
        depths = []
        for pt in triangle:
            u, v = pt
            if 0 <= u < depth_map.shape[1] and 0 <= v < depth_map.shape[0]:
                z = depth_map[v, u]
                depths.append(z)
            else:
                depths.append(ground_ref)
        
        # Convert to 3D points
        points_3d = []
        for (u, v), z in zip(triangle, depths):
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points_3d.append([x, y, z])
        
        # Compute triangle area using cross product
        p1, p2, p3 = np.array(points_3d)
        v1 = p2 - p1
        v2 = p3 - p1
        area = 0.5 * np.linalg.norm(np.cross(v1, v2))
        
        return area
    
    def estimate(self,
                 rgb_image: np.ndarray,
                 depth_map: np.ndarray,
                 bbox: Tuple[int, int, int, int],
                 K: np.ndarray,
                 ground_ref: float = None) -> MBTPEstimate:
        """
        Main MBTP estimation method with RGB fallback.
        """
        h, w = depth_map.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        x1, y1, x2, y2 = bbox
        # YOLO boxes arrive as float pixel coordinates; every slice below
        # needs integers, so normalise once and keep the box inside the frame.
        x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
        x2, y2 = min(w - 1, int(round(x2))), min(h - 1, int(round(y2)))
        bbox = (x1, y1, x2, y2)
        
        # Compute ground reference if not provided
        if ground_ref is None:
            ring = 10
            rx1, ry1 = max(0, x1-ring), max(0, y1-ring)
            rx2, ry2 = min(w-1, x2+ring), min(h-1, y2+ring)
            
            border_mask = np.zeros(depth_map.shape, dtype=bool)
            border_mask[ry1:ry2+1, rx1:rx2+1] = True
            border_mask[y1:y2+1, x1:x2+1] = False
            
            ground_depths = depth_map[border_mask]
            if ground_depths.size == 0:
                ground_ref = float(np.median(depth_map))
            else:
                ground_ref = float(np.median(ground_depths))
        
        # Use robust segmentor
        mask, method = self.segmentor.segment_pothole(
            rgb_image, depth_map, bbox, ground_ref
        )
        
        # Log segmentation method
        if method != "depth":
            print(f"  [MBTP] Using {method} segmentation")
        
        if mask is None:
            print(f"  [MBTP] Segmentation failed ({method}), using bounding box fallback")
            return self._fallback_estimate(depth_map, bbox, K, ground_ref)
        
        # Extract contour
        contours = self.segmentor.get_contour(mask)
        if not contours:
            return self._fallback_estimate(depth_map, bbox, K, ground_ref)
        
        contour = contours[0]

        # The segmentor's contour is ROI-relative; translate it to full-image
        # pixel coordinates so triangulation samples the correct depth pixels.
        contour_img = contour + np.array([[[x1, y1]]])

        # Triangulate and compute area
        triangles, tri_viz = self.triangulate_contour(contour_img, (h, w))
        
        total_area = 0.0
        for tri in triangles:
            area = self.compute_triangle_area(tri, depth_map, fx, fy, cx, cy, ground_ref)
            total_area += area
        
        # Compute convex hull and volume
        hull = cv2.convexHull(contour)
        hull_area_pixels = cv2.contourArea(hull)
        hull_area_m2 = self._pixel_area_to_m2(hull_area_pixels, depth_map, bbox, fx, fy)
        irregularity = 1.0 - (total_area / max(hull_area_m2, 0.001))
        
        # Volume integration. The segmentor returns the mask at ROI size, so it
        # already aligns with roi_depth; slicing it again would misalign it.
        roi_depth = depth_map[y1:y2, x1:x2]
        roi_mask = mask
        depth_deficit = np.maximum(ground_ref - roi_depth, 0)
        pixel_area_m2 = (ground_ref / fx) * (ground_ref / fy)
        volume = float(np.sum(depth_deficit[roi_mask > 0]) * pixel_area_m2)
        
        # Confidence based on segmentation method
        method_confidence = {
            "depth": 0.9,
            "rgb_water": 0.7,
            "hybrid": 0.8,
            "bounding_box_fallback": 0.5
        }.get(method, 0.6)
        
        confidence = method_confidence * min(1.0, total_area / (hull_area_m2 + 0.001))
        
        return MBTPEstimate(
            area_m2=round(total_area, 6),
            perimeter_m=round(cv2.arcLength(contour, True) * (ground_ref / fx), 3),
            depth_m=round(np.mean(roi_depth[roi_mask > 0]) if np.any(roi_mask > 0) else ground_ref, 3),
            volume_m3=round(volume, 6),
            convex_hull_area_m2=round(hull_area_m2, 6),
            irregularity_score=round(irregularity, 3),
            confidence=round(confidence, 3),
            contour_points=contour.reshape(-1, 2).tolist(),
            triangulation_map=tri_viz
        )
    
    def _pixel_area_to_m2(self, pixel_area: float, depth_map: np.ndarray, 
                          bbox: Tuple[int, int, int, int], fx: float, fy: float) -> float:
        """Convert pixel area to square meters using average depth."""
        x1, y1, x2, y2 = bbox
        roi = depth_map[y1:y2, x1:x2]
        avg_depth = np.median(roi[roi > 0]) if np.any(roi > 0) else 5.0
        return pixel_area * (avg_depth / fx) * (avg_depth / fy)
    
    def _fallback_estimate(self, depth_map: np.ndarray, bbox: Tuple[int, int, int, int],
                           K: np.ndarray, ground_ref: float) -> MBTPEstimate:
        """Fallback to bounding box method when segmentation fails."""
        x1, y1, x2, y2 = bbox
        fx, fy = K[0, 0], K[1, 1]
        
        width_m = (x2 - x1) * ground_ref / K[0, 0]
        length_m = (y2 - y1) * ground_ref / K[1, 1]
        area = width_m * length_m
        
        return MBTPEstimate(
            area_m2=round(area, 6),
            perimeter_m=round(2 * (width_m + length_m), 3),
            depth_m=round(ground_ref, 3),
            volume_m3=0.0,
            convex_hull_area_m2=round(area, 6),
            irregularity_score=0.0,
            confidence=0.5,
            contour_points=[],
            triangulation_map=None
        )


def create_mbtp_estimator(depth_threshold_m: float = 0.02,
                          min_area_pixels: int = 100,
                          contour_simplification_eps: float = 0.005,
                          use_rgb_segmentation: bool = True) -> MBTPAreaEstimator:
    """Factory function to create MBTP estimator with default parameters."""
    return MBTPAreaEstimator(
        depth_threshold_m=depth_threshold_m,
        min_area_pixels=min_area_pixels,
        contour_simplification_eps=contour_simplification_eps,
        use_rgb_segmentation=use_rgb_segmentation
    )