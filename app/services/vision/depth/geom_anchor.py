"""
GeometryAnchoredDepthScaler - Convert relative inverse depth to metric depth using IMU anchors
Based on pothole-depth-fusion pipeline/geom_anchor.py
"""
import numpy as np
import cv2
from typing import Optional, Tuple, List
from dataclasses import dataclass
import numpy.typing as npt

from app.core.camera import (
    DEFAULT_CAMERA_CX,
    DEFAULT_CAMERA_CY,
    DEFAULT_CAMERA_FX,
    DEFAULT_CAMERA_FY,
    DEFAULT_CAMERA_HEIGHT_M,
)

FloatArray = npt.NDArray[np.float64]


@dataclass
class ScalerState:
    """State for the depth scaler."""
    scale: float = 1.0
    shift: float = 0.0
    confidence: float = 0.0


class GeometryAnchoredDepthScaler:
    """
    Scale relative inverse depth to metric depth using geometry anchors.
    
    Based on pothole-depth-fusion pipeline/geom_anchor.py
    Uses IMU pitch/roll and camera height to scale inverse depth to metric depth.
    """
    
    def __init__(self, 
                 camera_height_m: float = DEFAULT_CAMERA_HEIGHT_M,
                 pitch_deg: float = 0.0,
                 roll_deg: float = 0.0,
                 fx: float = DEFAULT_CAMERA_FX,
                 fy: float = DEFAULT_CAMERA_FY,
                 cx: float = DEFAULT_CAMERA_CX,
                 cy: float = DEFAULT_CAMERA_CY,
                 min_confidence: float = 0.1,
                 ema_alpha: float = 0.3):
        """
        Initialize the depth scaler.
        
        Args:
            camera_height_m: Camera height above ground (meters)
            pitch_deg: Camera pitch in degrees (positive = nose up)
            roll_deg: Camera roll in degrees (positive = right side up)
            fx, fy: Focal lengths in pixels
            cx, cy: Principal point in pixels
            min_confidence: Minimum confidence to accept a scale estimate
            ema_alpha: Exponential moving average smoothing factor
        """
        self.camera_height_m = camera_height_m
        self.pitch_deg = pitch_deg
        self.roll_deg = roll_deg
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.min_confidence = min_confidence
        self.ema_alpha = ema_alpha
        
        self._state = ScalerState()
        self._initialized = False
    
    def _ema(self, prev: float, new: float) -> float:
        """Exponential moving average."""
        return prev * (1 - self.ema_alpha) + new * self.ema_alpha
    
    def update_camera_params(self, 
                             camera_height_m: Optional[float] = None,
                             pitch_deg: Optional[float] = None,
                             roll_deg: Optional[float] = None):
        """Update camera parameters."""
        if camera_height_m is not None:
            self.camera_height_m = camera_height_m
        if pitch_deg is not None:
            self.pitch_deg = pitch_deg
        if roll_deg is not None:
            self.roll_deg = roll_deg
    
    def compute_ground_plane(self, 
                             pitch_deg: float, 
                             roll_deg: float, 
                             height_m: float) -> Tuple[np.ndarray, float]:
        """
        Compute ground plane normal and distance from camera pose.
        
        World (vehicle) coordinates: x right, y forward, z up
        Camera coordinates: x right, y down, z forward
        Camera is at world position [0, 0, height], looking along +y (forward)
        Ground plane in world: z = 0, normal [0, 0, 1] (pointing up)
        
        Returns:
            plane_normal: Normal vector of ground plane in camera coordinates (points UP from ground)
            plane_d: Plane distance from camera origin (d = height for z=0 plane)
        """
        # Base rotation from world (vehicle) to camera (no pitch/roll)
        # Camera x = world x (right)
        # Camera y = -world z (down = negative up)
        # Camera z = world y (forward)
        R_base = np.array([[1, 0, 0],
                           [0, 0, -1],
                           [0, 1, 0]], dtype=np.float64)
        
        # Vehicle rotations (pitch around x, roll around y)
        # Pitch: rotation around vehicle x-axis (which = camera x), positive = nose up
        pitch = np.radians(pitch_deg)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(pitch), -np.sin(pitch)],
                       [0, np.sin(pitch), np.cos(pitch)]], dtype=np.float64)
        
        # Roll: rotation around vehicle y-axis (which = camera z), positive = right side up
        roll = np.radians(roll_deg)
        Ry = np.array([[np.cos(roll), 0, np.sin(roll)],
                       [0, 1, 0],
                       [-np.sin(roll), 0, np.cos(roll)]], dtype=np.float64)
        
        # Vehicle rotation matrix (pitch then roll)
        R_vehicle = Ry @ Rx
        
        # World to camera rotation = base rotation @ vehicle rotation
        R_w2c = R_base @ R_vehicle
        
        # Ground normal in world coordinates (pointing up)
        n_world = np.array([0.0, 0.0, 1.0])
        
        # Ground normal in camera coordinates
        n_cam = R_w2c @ n_world
        
        # Plane distance: ground plane is z=0 in world, camera at [0, 0, height]
        # In camera coordinates: n_cam · P_c + d = 0
        # For point on ground at camera position: P_c = [0, 0, 0] (camera origin)
        # But ground plane doesn't pass through camera. The plane is z=0 in world.
        # Camera position in world: [0, 0, height]
        # Distance from camera to ground plane along world z = height
        # In camera coordinates, d = n_cam · (point_on_ground_in_cam)
        # A point on ground in world: [0, 0, 0], transform to camera: 
        # P_cam = R_w2c @ (P_world - C_world) where C_world = [0, 0, height]
        # But simpler: plane equation n_world · P_world = 0 (z=0)
        # P_world = C_world + R_c2w @ P_cam
        # n_world · (C_world + R_c2w @ P_cam) = 0
        # n_world · C_world + n_world · (R_c2w @ P_cam) = 0
        # n_world · C_world + (R_w2c @ n_world) · P_cam = 0
        # n_world · C_world + n_cam · P_cam = 0
        # So d = n_world · C_world = height (since n_world = [0,0,1], C_world = [0,0,height])
        d = float(height_m)
        
        return n_cam, float(d)
    
    def scale_inverse_depth(self, 
                            inv_depth: np.ndarray,
                            pitch_deg: float,
                            roll_deg: float,
                            height_m: float,
                            K: np.ndarray) -> np.ndarray:
        """
        Convert inverse depth to metric depth using ground plane geometry.
        
        Args:
            inv_depth: Inverse depth map (H, W) - higher values = closer
            pitch_deg: Camera pitch in degrees
            roll_deg: Camera roll in degrees  
            height_m: Camera height above ground
            K: Camera intrinsic matrix (3x3)
            
        Returns:
            Metric depth map in meters
        """
        n, d = self.compute_ground_plane(pitch_deg, roll_deg, height_m)
        self.n_ground = n
        
        h, w = inv_depth.shape
        u, v = np.meshgrid(np.arange(inv_depth.shape[1]), np.arange(inv_depth.shape[0]))
        
        # Unproject to camera coordinates (unnormalized)
        fx, fy = inv_depth.shape[1] / 2, inv_depth.shape[0] / 2  # approximate
        # Actually use proper intrinsics
        cx, cy = K[0, 2], K[1, 2]
        fx, fy = K[0, 0], K[1, 1]
        
        # Convert inverse depth to metric depth
        # For pinhole: X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy, Z = depth
        # Inverse depth: inv_d = 1/Z
        # So Z = 1/inv_d
        
        # But we need metric depth. Use ground plane intersection.
        # Ray direction in camera coordinates
        x_cam = (np.arange(inv_depth.shape[1]) - K[0, 2]) / K[0, 0]
        y_cam = (np.arange(inv_depth.shape[0]) - K[1, 2]) / K[1, 1]
        xx, yy = np.meshgrid(x_cam, y_cam)
        rays = np.stack([xx, yy, np.ones_like(xx)], axis=-1)  # (H, W, 3)
        
        # Normalize rays
        ray_norms = np.linalg.norm(rays, axis=-1, keepdims=True)
        rays_normalized = rays / (ray_norms + 1e-8)
        
        # Intersect with ground plane
        # Ray: p = t * ray_dir
        # Plane: n·p + d = 0
        # t = -d / (n·ray_dir)
        n_dot_ray = np.sum(rays_normalized * self.n_ground, axis=-1)
        
        # Avoid division by zero
        valid = np.abs(n_dot_ray) > 1e-6
        t = np.zeros_like(inv_depth)
        t[valid] = -d / n_dot_ray[valid]
        
        # Only keep intersections in front of camera (t > 0)
        valid = valid & (t > 0)
        
        # Metric depth = t (distance along ray)
        depth = t
        depth[~valid] = 0
        
        return depth
    
    def estimate_scale_from_ground_plane(self,
                                         depth: np.ndarray,
                                         K: np.ndarray,
                                         pitch_deg: float,
                                         roll_deg: float,
                                         height_m: float) -> Tuple[float, float]:
        """
        Estimate scale and shift to convert relative depth to metric depth.
        
        Uses ground plane geometry: intersect rays with ground plane.
        
        Returns:
            scale: Multiplicative factor
            shift: Additive offset
        """
        n, d = self.compute_ground_plane(pitch_deg, roll_deg, height_m)
        self.n_ground = n
        
        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Camera rays in normalized coordinates
        x_cam = (u - K[0, 2]) / K[0, 0]
        y_cam = (v - K[1, 2]) / K[1, 1]
        rays = np.stack([x_cam, y_cam, np.ones_like(u)], axis=-1)
        
        # Normalize rays
        ray_norms = np.linalg.norm(rays, axis=-1, keepdims=True)
        rays_norm = rays / (ray_norms + 1e-8)
        
        # Intersect with ground plane (use normalized rays for correct distance)
        n_dot_ray = np.sum(rays_norm * self.n_ground, axis=-1)
        valid = np.abs(n_dot_ray) > 1e-6
        
        t = np.zeros_like(depth)
        t[valid] = -d / n_dot_ray[valid]
        
        # Only keep intersections in front of camera (t > 0)
        valid = valid & (t > 0)
        
        # Use Z depth (distance along optical axis)
        z_depth = t * rays_norm[:, :, 2]
        
        # Calibrate using lower-middle region (v=0.5 to 0.8) where potholes typically appear
        # This avoids horizon (v~0.46) where ratio explodes and bottom (v>0.85) where ground is very close
        pothole_region = (v > h * 0.5) & (v < h * 0.85)
        valid_ground = valid & pothole_region & (depth > 0) & (z_depth > 0)
        
        if np.sum(valid_ground) < 10:
            return 1.0, 0.0
        
        # Compute scale ratios for ground pixels in pothole region
        ratios = z_depth[valid_ground] / depth[valid_ground]
        
        # Remove outliers (keep middle 80%)
        lo, hi = np.percentile(ratios, [10, 90])
        ratios_clean = ratios[(ratios >= lo) & (ratios <= hi)]
        
        if len(ratios_clean) < 10:
            return 1.0, 0.0
        
        # Use median for robustness
        scale = float(np.median(ratios_clean))
        shift = 0.0
        
        return scale, shift
    
    def update_state(self, scale: float, shift: float, confidence: float) -> None:
        """Update EMA state."""
        if confidence < self.min_confidence:
            return
        
        if not self._initialized:
            self._state.scale = scale
            self._state.shift = shift
            self._state.confidence = confidence
            self._initialized = True
        else:
            self._state.scale = self._ema(self._state.scale, scale)
            self._state.shift = self._ema(self._state.shift, shift)
            self._state.confidence = self._ema(self._state.confidence, confidence)
    
    def apply(self, depth: np.ndarray, K: np.ndarray,
              pitch_deg: float, roll_deg: float, height_m: float) -> np.ndarray:
        """
        Apply scaling to convert relative depth to metric depth.
        
        DepthAnythingV2 outputs relative depth (higher = farther).
        We fit scale/shift using ground plane geometry.
        
        Returns metric depth map in meters.
        """
        if not self._initialized:
            scale, shift = self.estimate_scale_from_ground_plane(
                depth, K, pitch_deg, roll_deg, height_m
            )
            self.update_state(scale, shift, 0.5)
        
        # DepthAnythingV2 outputs direct depth: metric = scale * relative_depth + shift
        metric_depth = self._state.scale * depth + self._state.shift
        return np.clip(metric_depth, 0, 100)  # Clip to reasonable range
    
    def get_state(self) -> ScalerState:
        return self._state
    
    def reset(self):
        self._initialized = False
        self._state = ScalerState()


# Convenience function
def create_depth_scaler(camera_height_m: float = 1.5,
                        pitch_deg: float = 0.0,
                        roll_deg: float = 0.0,
                        K: Optional[np.ndarray] = None,
                        fx: float = 1000.0, fy: float = 1000.0,
                        cx: float = 960.0, cy: float = 540.0) -> GeometryAnchoredDepthScaler:
    """Create a depth scaler with default parameters."""
    return GeometryAnchoredDepthScaler(
        camera_height_m=camera_height_m,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
        fx=fx, fy=fy, cx=cx, cy=cy
    )