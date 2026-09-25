from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pandas as pd
import pytest

from app.services.depth_service import (
    DepthEstimationResult,
    DepthEstimationService,
    DepthObservation,
)


class TestDepthObservationDataclass:
    def test_create_observation(self):
        obs = DepthObservation(
            frame_index=1,
            gray_frame=np.zeros((100, 100), dtype=np.uint8),
            bbox=[10.0, 10.0, 50.0, 50.0],
            polygon=None,
            frame_time=0.5,
        )
        assert obs.frame_index == 1
        assert obs.frame_time == 0.5


class TestDepthEstimationResultDataclass:
    def test_create_result(self):
        result = DepthEstimationResult(
            max_depth_mm=10.0,
            mean_depth_mm=8.0,
            confidence=0.8,
            method="sfm",
            message="ok",
        )
        assert result.max_depth_mm == 10.0
        assert result.confidence == 0.8


class TestDepthEstimationServiceBlur:
    def test_is_blurry_returns_true_for_blank(self):
        blank = np.zeros((100, 100), dtype=np.uint8)
        assert DepthEstimationService._is_blurry(blank)

    def test_is_blurry_returns_false_for_texture(self):
        rng = np.random.default_rng(42)
        textured = rng.integers(0, 255, (100, 100), dtype=np.uint8)
        assert not DepthEstimationService._is_blurry(textured)


class TestDepthEstimationServiceMatchFeatures:
    def test_match_features_returns_shapes(self):
        img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        pts1, pts2, count = DepthEstimationService._match_features(img, img)
        assert pts1.shape[1] == 2
        assert pts2.shape[1] == 2
        assert count >= 0


class TestDepthEstimationServiceTriangulate:
    def test_triangulate_returns_valid(self):
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = 800.0
        K[1, 1] = 800.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0
        R = np.eye(3, dtype=np.float64)
        t = np.array([[0.1], [0], [0]], dtype=np.float64)
        p1 = np.array([[100.0, 200.0]], dtype=np.float64)
        p2 = np.array([[105.0, 200.0]], dtype=np.float64)

        pts3d, valid = DepthEstimationService._triangulate(K, R, t, p1, p2)
        assert len(pts3d) >= 0
        assert len(valid) == 1


class TestDepthEstimationServiceRansacPlaneFit:
    def test_ransac_plane_fit_planar_points(self):
        rng = np.random.default_rng(42)
        pts = np.column_stack([
            rng.uniform(-1, 1, 200),
            rng.uniform(-1, 1, 200),
            np.full(200, 0.5),
        ]).astype(np.float64)
        n, d = DepthEstimationService._ransac_plane_fit(pts)
        assert n.shape == (3,)
        assert isinstance(d, float)

    def test_ransac_plane_fit_insufficient_points(self):
        pts = np.array([[0, 0, 0]], dtype=np.float64)
        n, d = DepthEstimationService._ransac_plane_fit(pts)
        assert n.shape == (3,)


class TestDepthEstimationServiceEma:
    def test_ema_first_call(self):
        result = DepthEstimationService._ema(0.0, 10.0)
        assert result == pytest.approx(10.0 * DepthEstimationService._alpha)

    def test_ema_smoothing(self):
        prev = 5.0
        new = 15.0
        result = DepthEstimationService._ema(prev, new)
        expected = prev * (1 - DepthEstimationService._alpha) + new * DepthEstimationService._alpha
        assert result == pytest.approx(expected)


class TestDepthEstimationServicePointsInsideDefect:
    def test_points_inside_polygon(self):
        obs = DepthObservation(
            frame_index=0,
            gray_frame=np.zeros((100, 100), dtype=np.uint8),
            bbox=[0.0, 0.0, 10.0, 10.0],
            polygon=np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32),
        )
        pts = np.array([[5.0, 5.0], [20.0, 20.0]], dtype=np.float64)
        mask = DepthEstimationService._points_inside_defect(pts, obs)
        assert bool(mask[0]) is True
        assert bool(mask[1]) is False

    def test_points_inside_bbox_fallback(self):
        obs = DepthObservation(
            frame_index=0,
            gray_frame=np.zeros((100, 100), dtype=np.uint8),
            bbox=[0.0, 0.0, 10.0, 10.0],
            polygon=None,
        )
        pts = np.array([[5.0, 5.0], [20.0, 20.0]], dtype=np.float64)
        mask = DepthEstimationService._points_inside_defect(pts, obs)
        assert bool(mask[0]) is True
        assert bool(mask[1]) is False


class TestDepthEstimationServiceSelectPair:
    def test_select_pair_returns_first_and_last(self):
        obs = [
            DepthObservation(0, np.zeros((10, 10), dtype=np.uint8), [0, 0, 1, 1], None),
            DepthObservation(1, np.zeros((10, 10), dtype=np.uint8), [0, 0, 1, 1], None),
            DepthObservation(2, np.zeros((10, 10), dtype=np.uint8), [0, 0, 1, 1], None),
        ]
        first, second = DepthEstimationService._select_pair(obs)
        assert first.frame_index == 0
        assert second.frame_index == 2


class TestDepthEstimationServiceConfidence:
    def test_confidence_formula(self):
        conf = DepthEstimationService._confidence(120, 80, 12, 40)
        assert conf == pytest.approx(1.0)

    def test_confidence_partial(self):
        conf = DepthEstimationService._confidence(60, 40, 6, 20)
        assert 0.0 < conf < 1.0


class TestDepthEstimationServiceEmpty:
    def test_empty_result(self):
        result = DepthEstimationService._empty("test error", matched_points=10)
        assert result.max_depth_mm is None
        assert result.message == "test error"
        assert result.matched_points == 10
        assert result.confidence == 0.0


class TestDepthEstimationServiceEstimateTrackDepth:
    def test_insufficient_observations(self):
        observations = [
            DepthObservation(0, np.zeros((100, 100), dtype=np.uint8), [0, 0, 10, 10], None)
        ]
        K = np.eye(3, dtype=np.float64)
        result = DepthEstimationService.estimate_track_depth(observations, K, 1.5)
        assert result.message == "insufficient observations"

    def test_blurry_frame_rejection(self):
        observations = [
            DepthObservation(0, np.zeros((100, 100), dtype=np.uint8), [0, 0, 10, 10], None),
            DepthObservation(1, np.zeros((100, 100), dtype=np.uint8), [0, 0, 10, 10], None),
        ]
        K = np.eye(3, dtype=np.float64)
        result = DepthEstimationService.estimate_track_depth(observations, K, 1.5)
        assert "blur" in result.message.lower()

    def test_estimate_with_mocked_cv2(self):
        rng = np.random.default_rng(42)
        frames = [rng.integers(0, 255, (100, 100), dtype=np.uint8) for _ in range(3)]
        observations = [
            DepthObservation(i, frames[i], [10.0, 10.0, 50.0, 50.0], None)
            for i in range(3)
        ]
        K = np.array([[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1]], dtype=np.float64)

        mock_essential = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float64)
        mock_mask = np.ones((10, 1), dtype=np.uint8)
        mock_R = np.eye(3, dtype=np.float64)
        mock_t = np.array([[0.1], [0], [0]], dtype=np.float64)
        mock_pose_mask = np.ones((35, 1), dtype=np.int32)

        pts3d = np.column_stack([
            rng.uniform(-1, 1, 35),
            rng.uniform(-1, 1, 35),
            rng.uniform(0.1, 1.0, 35),
        ]).astype(np.float64)

        defect_mask = np.zeros(35, dtype=bool)
        defect_mask[:5] = True

        with patch("cv2.findEssentialMat", return_value=(mock_essential, mock_mask)), \
             patch("cv2.recoverPose", return_value=(None, mock_R, mock_t, mock_pose_mask)), \
             patch.object(DepthEstimationService, "_match_features", return_value=(np.ones((35, 2), dtype=np.float64), np.ones((35, 2), dtype=np.float64), 35)), \
             patch.object(DepthEstimationService, "_triangulate", return_value=(pts3d, np.ones(35, dtype=bool))), \
             patch.object(DepthEstimationService, "_points_inside_defect", return_value=defect_mask):
            result = DepthEstimationService.estimate_track_depth(observations, K, 1.5, track_id=1)
            assert result.max_depth_mm is not None
            assert result.confidence > 0.0
