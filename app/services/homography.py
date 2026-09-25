"""
HomographyMapper - Project image coordinates to world coordinates using camera pose
Based on pothole-depth-fusion pipeline/homography.py
"""
import numpy as np
from typing import Tuple, Dict, Optional
import numpy.typing as npt
import cv2

FloatArray = npt.NDArray[np.float64]


class HomographyMapper:
    """
    Maps image coordinates to world coordinates using camera pose and intrinsics.
    Based on pothole-depth-fusion pipeline/homography.py
    """
    
    def __init__(self, K: np.ndarray, camera_height_m: float = 1.5):
        """
        Initialize homography mapper.
        
        Args:
            K: Camera intrinsic matrix (3x3)
            camera_height_m: Camera height above ground (meters)
        """
        self.K = K.copy().astype(np.float64)
        self.camera_height_m = camera_height_m
        self.fx = float(K[0, 0])
        self.fy = float(K[1, 1])
        self.cx = float(K[0, 2])
        self.cy = float(K[1, 2])
    
    def update_camera_params(self, 
                             camera_height_m: Optional[float] = None,
                             K: Optional[np.ndarray] = None):
        """Update camera parameters."""
        if camera_height_m is not None:
            self.camera_height_m = camera_height_m
        if K is not None:
            self.K = K.copy().astype(np.float64)
            self.fx = float(K[0, 0])
            self.fy = float(K[1, 1])
            self.cx = float(K[0, 2])
            self.cy = float(K[1, 2])
    
    def bbox_world_dims(self,
                        bbox: Tuple[int, int, int, int],
                        pitch_deg: float,
                        roll_deg: float,
                        yaw_deg: float = 0.0) -> Dict[str, float]:
        """
        Compute world dimensions of a bounding box using homography.
        
        Args:
            bbox: Image bounding box [x1, y1, x2, y2]
            pitch_deg: Camera pitch in degrees
            roll_deg: Camera roll in degrees
            yaw_deg: Camera yaw in degrees
            
        Returns:
            Dictionary with width_m, length_m, area_m2
        """
        x1, y1, x2, y2 = bbox
        
        # Camera orientation
        pitch = np.radians(pitch_deg)
        roll = np.radians(roll_deg)
        yaw = np.radians(yaw_deg)
        
        # Camera matrix
        K = np.array([[self.fx, 0, self.cx],
                      [0, self.fy, self.cy],
                      [0, 0, 1]], dtype=np.float64)
        
        # Rotation matrices
        # Pitch: rotation around x-axis (nose up positive)
        # Roll: rotation around z-axis (right side up positive)
        # Yaw: rotation around y-axis (heading)
        
        # Camera to world rotation
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(-pitch), -np.sin(-pitch)],
                        [0, np.sin(-pitch), np.cos(-pitch)]])
        
        R_z = np.array([[np.cos(roll), -np.sin(roll), 0],
                        [np.sin(roll), np.cos(roll), 0],
                        [0, 0, 1]])
        
        R_y = np.array([[np.cos(-yaw), 0, np.sin(-yaw)],
                        [0, 1, 0],
                        [-np.sin(-yaw), 0, np.cos(-yaw)]])
        
        # Camera to world rotation: yaw -> pitch -> roll
        R_c2w = R_y @ R_x @ R_z
        
        # Camera position in world
        t_c = np.array([0.0, -self.camera_height_m, 0.0])  # Camera at height, looking down
        
        # Homography from ground plane (Z=0) to image
        # H = K * [r1 r2 t] where r1, r2 are first two columns of R_c2w
        R_c2w = R_c2w.T  # World to camera
        
        # Ground plane in camera coordinates
        # Ground plane: Y = -camera_height_m in world coordinates
        # In camera coordinates: n_c = R_c2w * [0, 1, 0]^T, d = camera_height_m
        
        n_ground_c = R_c2w[:, 1]  # Y-axis of R_c2w
        d_ground = self.camera_height_m
        
        # Four corners of bbox in image
        corners = np.array([
            [x1, y1, 1],
            [x2, y1, 1],
            [x2, y2, 1],
            [x1, y2, 1]
        ], dtype=np.float64).T
        
        # Back-project to 3D ground plane
        K_inv = np.linalg.inv(self.K)
        corners_h = np.array([[x1, y1, 1],
                              [x2, y1, 1],
                              [x2, y2, 1],
                              [x1, y2, 1]], dtype=np.float64).T
        
        # Ray directions in camera coordinates
        rays_c = np.linalg.inv(self.K) @ corners_h
        rays_c = rays_c / np.linalg.norm(rays_c, axis=0)
        
        # Ray directions in world coordinates
        rays_w = R_c2w @ rays_c
        
        # Camera position in world coordinates
        # Camera at (0, -camera_height_m, 0) in world coords
        t_cw = np.array([0.0, -self.camera_height_m, 0.0])
        
        # Intersect with ground plane Z=0
        # Ray: p = t_cw + t * ray_dir_w
        # Ground: Z = 0 => t = -t_cw[2] / ray_w[2]
        t = -t_cw[2] / rays_w[2, :]
        
        # Intersection points
        pts_w = t_cw[:, np.newaxis] + t * rays_w
        
        # Width and length
        # Order: top-left, top-right, bottom-right, bottom-left
        width_m = np.linalg.norm(pts_w[:, 1] - pts_w[:, 0])
        length_m = np.linalg.norm(pts_w[:, 2] - pts_w[:, 1])
        
        return {
            "width_m": float(width_m),
            "length_m": float(length_m),
            "area_m2": float(width_m * length_m)
        }
    
    def confidence(self, pitch_deg: float, roll_deg: float) -> float:
        """
        Compute confidence based on camera orientation.
        Low pitch/roll = high confidence.
        """
        pitch = abs(np.radians(pitch_deg))
        roll = abs(np.radians(roll_deg))
        
        # Confidence drops with pitch/roll
        pitch_conf = max(0.0, 1.0 - pitch / np.radians(30))
        roll_conf = max(0.0, 1.0 - roll / np.radians(30))
        
        return float(min(pitch_conf, roll_conf))
    
    def project_point(self,
                      u: float, v: float,
                      pitch_deg: float, roll_deg: float,
                      depth_m: float) -> Tuple[float, float]:
        """
        Project image point to world coordinates at given depth.
        """
        # Unproject to camera coordinates
        x_cam = (u - self.cx) / self.fx
        y_cam = (v - self.cy) / self.fy
        
        # Ray direction in camera coordinates
        ray_c = np.array([x_cam, y_cam, 1.0])
        ray_c = ray_c / np.linalg.norm(ray_c)
        
        # Transform to world
        pitch = np.radians(pitch_deg)
        roll = np.radians(roll_deg)
        
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(-pitch), -np.sin(-pitch)],
                        [0, np.sin(-pitch), np.cos(-pitch)]])
        R_z = np.array([[np.cos(0), -np.sin(0), 0],
                        [np.sin(0), np.cos(0), 0],
                        [0, 0, 1]])
        
        R_c2w = np.eye(3)  # Simplified - no yaw for now
        
        ray_w = ray_c  # Simplified
        
        # Scale by depth
        pt_w = np.array([0.0, -self.camera_height_m, 0.0]) + depth_m * ray_w
        
        return float(pt_w[0]), float(pt_w[1])


def create_homography_mapper(K: np.ndarray, camera_height_m: float = 1.5) -> HomographyMapper:
    """Factory function to create HomographyMapper."""
    return HomographyMapper(K, camera_height_m)