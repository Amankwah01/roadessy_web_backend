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
        
        Returns:
            plane_normal: Normal vector of ground plane in camera coordinates
            plane_d: Plane distance from camera origin
        """
        pitch = np.radians(pitch_deg)
        roll = np.radians(roll_deg)
        
        # Camera coordinate system: x right, y down, z forward
        # Ground plane normal in camera coordinates
        # When pitch=0, roll=0: ground plane is y = -height (y-down coordinate)
        # Plane equation: n·x + d = 0
        # Normal points upward from ground
        
        # Rotation from world to camera
        # World: x right, y forward, z up
        # Camera: x right, y down, z forward
        # Pitch: rotation around x (positive = nose up)
        # Roll: rotation around z (positive = right side up)
        
        cp, sp = np.cos(np.radians(pitch_deg)), np.sin(np.radians(pitch_deg))
        cr, sr = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
        
        # Ground normal in world coordinates (pointing up)
        n_world = np.array([0.0, 0.0, 1.0])
        
        # World to camera rotation (pitch then roll)
        # Note: camera coordinate system has y-down, z-forward
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(-np.radians(pitch_deg)), -np.sin(-np.radians(pitch_deg))],
                       [0, np.sin(-np.radians(pitch_deg)), np.cos(-np.radians(pitch_deg))]])
        Rz = np.array([[np.cos(np.radians(roll_deg)), -np.sin(np.radians(roll_deg)), 0],
                       [np.sin(np.radians(roll_deg)), np.cos(np.radians(roll_deg)), 0],
                       [0, 0, 1]])
        
        # World to camera rotation (pitch then roll)
        R_w2c = Rz @ Rx
        
        # Ground normal in camera coordinates
        n_cam = R_w2c @ n_world
        
        # Plane distance: ground plane passes through (0, -height, 0) in world coords
        # In camera coordinates: point on ground = R_w2c @ [0, -height, 0] + t_cam
        # Camera at origin, so t_cam = [0, 0, 0]
        point_on_ground_world = np.array([0.0, -1.0, 0.0]) * self.camera_height_m
        point_on_ground_cam = R_w2c @ point_on_ground_world
        
        # Plane distance d = -n·p
        d = -np.dot(n_cam, point_on_ground_cam)
        
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
        
        # Metric depth = t (distance along ray)
        depth = t
        depth[~valid] = 0
        
        return depth
    
    def estimate_scale_from_ground_plane(self,
                                         inv_depth: np.ndarray,
                                         K: np.ndarray,
                                         pitch_deg: float,
                                         roll_deg: float,
                                         height_m: float) -> Tuple[float, float]:
        """
        Estimate scale and shift to convert inverse depth to metric depth.
        
        Uses ground plane geometry: intersect rays with ground plane.
        
        Returns:
            scale: Multiplicative factor
            shift: Additive offset
        """
        n, d = self.compute_ground_plane(pitch_deg, roll_deg, height_m)
        self.n_ground = n
        
        h, w = inv_depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Camera rays in normalized coordinates
        x_cam = (u - K[0, 2]) / K[0, 0]
        y_cam = (v - K[1, 2]) / K[1, 1]
        rays = np.stack([x_cam, y_cam, np.ones_like(u)], axis=-1)
        
        # Normalize rays
        ray_norms = np.linalg.norm(rays, axis=-1, keepdims=True)
        rays_norm = rays / (ray_norms + 1e-8)
        
        # Intersect with ground plane
        n_dot_ray = np.sum(rays * self.n_ground, axis=-1)
        valid = np.abs(n_dot_ray) > 1e-6
        
        t = np.zeros_like(inv_depth)
        t[valid] = -d / n_dot_ray[valid]
        
        # Now we have metric depth t from ground plane intersection
        # Compare with inverse depth to find scale
        valid_depth = inv_depth > 0
        valid_both = valid & valid_depth
        
        if np.sum(valid_both) < 10:
            return 1.0, 0.0
        
        # Linear regression: metric_depth = scale * (1/inv_depth) + shift
        inv_d = inv_depth[valid_both]
        metric_d = t[valid_both]
        
        # Robust linear fit
        A = np.column_stack([1.0 / inv_d, np.ones_like(inv_d)])
        try:
            scale_shift, _, _, _ = np.linalg.lstsq(A, metric_d, rcond=None)
            scale, shift = float(scale_shift[0]), float(scale_shift[1])
        except:
            scale, shift = 1.0, 0.0
        
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
    
    def apply(self, inv_depth: np.ndarray, K: np.ndarray,
              pitch_deg: float, roll_deg: float, height_m: float) -> np.ndarray:
        """
        Apply scaling to convert inverse depth to metric depth.
        
        Returns metric depth map in meters.
        """
        if not self._initialized:
            scale, shift = self.estimate_scale_from_ground_plane(
                inv_depth, K, pitch_deg, roll_deg, height_m
            )
            self.update_state(scale, shift, 0.5)
        
        metric_depth = self._state.scale / (inv_depth + 1e-6) + self._state.shift
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