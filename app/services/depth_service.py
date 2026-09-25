# app/services/depth_service.py

from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from app.core.logger import vision_logger

FloatArray = NDArray[np.float64]


# =========================
# DATA STRUCTURES
# =========================


@dataclass
class DepthObservation:
    frame_index: int
    gray_frame: NDArray[np.uint8]
    bbox: List[float]
    polygon: Optional[NDArray[np.float32]]
    frame_time: Optional[float] = None


@dataclass
class DepthEstimationResult:
    max_depth_mm: Optional[float]
    mean_depth_mm: Optional[float]
    confidence: float
    method: str
    message: str
    matched_points: int = 0
    inlier_points: int = 0
    defect_points: int = 0
    road_plane_points: int = 0


# =========================
# CORE DEPTH SERVICE
# =========================


class DepthEstimationService:

    MIN_OBSERVATIONS = 2
    MIN_MATCHES = 30
    MIN_INLIERS = 20
    MIN_PLANE_POINTS = 15
    MIN_DEFECT_POINTS = 5

    # EMA smoothing storage (per track_id)
    _depth_state: Dict[int, float] = {}
    _alpha = 0.35  # smoothing factor

    # =========================
    # PUBLIC API
    # =========================

    @classmethod
    def estimate_track_depth(
        cls,
        observations: List[DepthObservation],
        camera_matrix: FloatArray,
        camera_height_meters: float,
        track_id: Optional[int] = None,
    ) -> DepthEstimationResult:

        if len(observations) < cls.MIN_OBSERVATIONS:
            vision_logger.warning("depth_estimation: insufficient observations", track_id=track_id, count=len(observations))
            return cls._empty("insufficient observations")

        # stable pair selection (improves SfM baseline)
        first, second = cls._select_pair(observations)

        # blur rejection (VERY IMPORTANT for dashcams)
        if cls._is_blurry(first.gray_frame) or cls._is_blurry(second.gray_frame):
            vision_logger.warning("depth_estimation: frame rejected due to motion blur", track_id=track_id)
            return cls._empty("frame rejected due to motion blur")

        pts1, pts2, match_count = cls._match_features(first.gray_frame, second.gray_frame)

        if len(pts1) < cls.MIN_MATCHES:
            vision_logger.warning("depth_estimation: insufficient feature matches", track_id=track_id, matched_points=match_count)
            return cls._empty("insufficient feature matches", matched_points=match_count)

        essential, inlier_mask = cv2.findEssentialMat(
            pts1,
            pts2,
            camera_matrix,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )

        if essential is None or inlier_mask is None:
            vision_logger.warning("depth_estimation: essential matrix failed", track_id=track_id, matched_points=match_count)
            return cls._empty("essential matrix failed", matched_points=match_count)

        _, R, t, pose_mask = cv2.recoverPose(
            essential,
            pts1,
            pts2,
            camera_matrix,
            mask=inlier_mask,
        )

        inliers = pose_mask.ravel().astype(bool)
        if np.count_nonzero(inliers) < cls.MIN_INLIERS:
            vision_logger.warning("depth_estimation: insufficient pose inliers", track_id=track_id, matched_points=match_count, inlier_points=int(np.count_nonzero(inliers)))
            return cls._empty(
                "insufficient pose inliers",
                matched_points=match_count,
                inlier_points=int(np.count_nonzero(inliers)),
            )

        pts1 = pts1[inliers]
        pts2 = pts2[inliers]

        points_3d, valid = cls._triangulate(camera_matrix, R, t, pts1, pts2)

        if len(points_3d) < cls.MIN_INLIERS:
            vision_logger.warning("depth_estimation: triangulation failed", track_id=track_id, inlier_points=len(points_3d))
            return cls._empty("triangulation failed", inlier_points=len(points_3d))

        pts1 = pts1[valid]

        # split road vs defect
        defect_mask = cls._points_inside_defect(pts1[: len(points_3d)], first)

        road_pts = points_3d[~defect_mask]
        defect_pts = points_3d[defect_mask]

        if len(road_pts) < cls.MIN_PLANE_POINTS:
            vision_logger.warning("depth_estimation: insufficient road plane points", track_id=track_id, road_points=len(road_pts))
            return cls._empty("insufficient road plane points")

        if len(defect_pts) < cls.MIN_DEFECT_POINTS:
            vision_logger.warning("depth_estimation: insufficient defect points", track_id=track_id, defect_points=len(defect_pts))
            return cls._empty("insufficient defect points")

        # =========================
        # RANSAC PLANE FIT (NEW)
        # =========================
        plane_n, plane_d = cls._ransac_plane_fit(road_pts)

        denom = max(np.linalg.norm(plane_n), 1e-9)
        camera_to_plane = abs(plane_d) / denom

        if camera_to_plane < 1e-6:
            return cls._empty("invalid plane scale")

        scale = camera_height_meters / camera_to_plane

        # signed distances
        dist = (defect_pts @ plane_n + plane_d) / denom
        depth_mm = np.abs(dist) * scale * 1000.0

        depth_mm = depth_mm[np.isfinite(depth_mm)]

        if len(depth_mm) == 0:
            return cls._empty("invalid depth values")

        max_depth = float(np.percentile(depth_mm, 95))
        mean_depth = float(np.mean(depth_mm))

        # =========================
        # TEMPORAL SMOOTHING (NEW)
        # =========================
        if track_id is not None:
            if track_id not in cls._depth_state:
                cls._depth_state[track_id] = max_depth
            else:
                prev = cls._depth_state[track_id]
                max_depth = cls._ema(prev, max_depth)
                cls._depth_state[track_id] = max_depth

        confidence = cls._confidence(match_count, len(points_3d), len(defect_pts), len(road_pts))

        vision_logger.metric(
            "depth_estimation",
            f"track_{track_id}",
            {
                "max_depth_mm": round(max_depth, 2),
                "mean_depth_mm": round(mean_depth, 2),
                "confidence": round(confidence, 3),
                "matched_points": match_count,
                "inlier_points": len(points_3d),
                "defect_points": len(defect_pts),
                "road_points": len(road_pts),
            },
            track_id=track_id,
        )

        return DepthEstimationResult(
            max_depth_mm=round(max_depth, 2),
            mean_depth_mm=round(mean_depth, 2),
            confidence=confidence,
            method="sfm_ransac_plane_ema",
            message="production depth estimation",
            matched_points=match_count,
            inlier_points=len(points_3d),
            defect_points=len(defect_pts),
            road_plane_points=len(road_pts),
        )

    # =========================
    # BLUR DETECTION (NEW)
    # =========================

    @staticmethod
    def _is_blurry(gray: NDArray[np.uint8], threshold: float = 80.0) -> bool:
        """Variance of Laplacian blur detector"""
        return bool(cv2.Laplacian(gray, cv2.CV_64F).var() < threshold)

    # =========================
    # FEATURE MATCHING
    # =========================

    @staticmethod
    def _match_features(gray1, gray2):
        # Robust ORB creation for OpenCV variants
        try:
            orb_ctor = getattr(cv2, "ORB_create", None)
            if orb_ctor is not None:
                orb = orb_ctor(3000)  # type: ignore[attr-defined]
            else:
                orb_class = getattr(cv2, "ORB", None)
                if orb_class is None or not hasattr(orb_class, "create"):
                    # ORB missing — return empty matches so caller can handle gracefully
                    empty = np.empty((0, 2), dtype=np.float64)
                    return empty, empty, 0
                orb = orb_class.create(3000)

            kp1, des1 = orb.detectAndCompute(gray1, None)
            kp2, des2 = orb.detectAndCompute(gray2, None)

            if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
                empty = np.empty((0, 2), dtype=np.float64)
                return empty, empty, 0

            # Choose matcher norm based on descriptor dtype
            try:
                if des1.dtype == np.uint8 and des2.dtype == np.uint8:
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
                else:
                    bf = cv2.BFMatcher(cv2.NORM_L2)

                matches = bf.knnMatch(des1, des2, k=2)
            except Exception:
                # Fallback: try matching with a safe cross-check matcher
                try:
                    bf = cv2.BFMatcher()
                    matches = bf.knnMatch(des1, des2, k=2)
                except Exception:
                    empty = np.empty((0, 2), dtype=np.float64)
                    return empty, empty, 0

            good = []
            for m_n in matches:
                if len(m_n) != 2:
                    continue
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good.append(m)

            if not good:
                empty = np.empty((0, 2), dtype=np.float64)
                return empty, empty, 0

            pts1 = np.array([kp1[m.queryIdx].pt for m in good], dtype=np.float64)
            pts2 = np.array([kp2[m.trainIdx].pt for m in good], dtype=np.float64)

            return pts1, pts2, len(good)
        except Exception:
            # Any runtime issue with feature extraction should gracefully degrade
            empty = np.empty((0, 2), dtype=np.float64)
            return empty, empty, 0

    # =========================
    # TRIANGULATION
    # =========================

    @staticmethod
    def _triangulate(K, R, t, p1, p2):

        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K @ np.hstack([R, t])

        homog = cv2.triangulatePoints(P1, P2, p1.T, p2.T)
        pts = (homog[:3] / homog[3]).T

        valid = np.isfinite(pts).all(axis=1) & (pts[:, 2] > 0)

        return pts[valid].astype(np.float64), valid

    # =========================
    # RANSAC PLANE FIT (NEW)
    # =========================

    @staticmethod
    def _ransac_plane_fit(points: FloatArray, iterations: int = 80):
        best_inliers: "np.ndarray" = np.empty(0, dtype=bool)
        best_n = None
        best_d = 0

        n = len(points)
        if n < 3:
            return np.array([0, 0, 1]), 0

        for _ in range(iterations):
            sample = points[np.random.choice(n, 3, replace=False)]

            n_vec = np.cross(sample[1] - sample[0], sample[2] - sample[0])
            if np.linalg.norm(n_vec) < 1e-6:
                continue

            n_vec = n_vec / np.linalg.norm(n_vec)
            d = -np.dot(n_vec, sample[0])

            dist = np.abs(points @ n_vec + d)
            inliers = dist < 0.05  # 5cm tolerance

            if np.sum(inliers) > len(best_inliers):
                best_inliers = inliers
                best_n = n_vec
                best_d = d

        if best_n is None:
            centroid = np.mean(points, axis=0)
            best_n = np.array([0, 0, 1])
            best_d = -np.dot(best_n, centroid)

        return best_n, best_d

    # =========================
    # TEMPORAL SMOOTHING (NEW)
    # =========================

    @staticmethod
    def _ema(prev, new):
        return prev * (1 - DepthEstimationService._alpha) + new * DepthEstimationService._alpha

    # =========================
    # DEFECT MASKING
    # =========================

    @staticmethod
    def _points_inside_defect(points, observation):

        if observation.polygon is not None and len(observation.polygon) >= 3:
            return np.array([cv2.pointPolygonTest(observation.polygon, (float(x), float(y)), False) >= 0 for x, y in points], dtype=bool)

        x1, y1, x2, y2 = observation.bbox
        return np.array([(x1 <= x <= x2) and (y1 <= y <= y2) for x, y in points], dtype=bool)

    # =========================
    # HELPERS
    # =========================

    @staticmethod
    def _select_pair(observations):
        return observations[0], observations[-1]

    @staticmethod
    def _confidence(m, inl, d, r):
        return float(min(m / 120, 1.0) * 0.3 + min(inl / 80, 1.0) * 0.3 + min(d / 12, 1.0) * 0.2 + min(r / 40, 1.0) * 0.2)

    @staticmethod
    def _empty(msg, matched_points=0, inlier_points=0):
        return DepthEstimationResult(
            max_depth_mm=None,
            mean_depth_mm=None,
            confidence=0.0,
            method="sfm_ransac_plane_ema",
            message=msg,
            matched_points=matched_points,
            inlier_points=inlier_points,
        )
