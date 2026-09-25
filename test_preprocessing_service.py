# test_preprocessing_service.py
import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import pytest

from app.services.preprocessing_service import PreprocessingService


class TestPreprocessingServiceFrameSelection:
    def test_compute_frame_sharpness_sharp_vs_blurry(self):
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        sharp_score = PreprocessingService.compute_frame_sharpness(frame)
        blurry = cv2.blur(frame, (20, 20))
        blurry_score = PreprocessingService.compute_frame_sharpness(blurry)
        assert sharp_score > blurry_score

    def test_select_best_frames_per_second(self):
        scored = [
            {"frame_idx": 0, "timestamp": 0.1, "sharpness": 100.0},
            {"frame_idx": 1, "timestamp": 0.5, "sharpness": 200.0},
            {"frame_idx": 2, "timestamp": 1.1, "sharpness": 150.0},
            {"frame_idx": 3, "timestamp": 1.6, "sharpness": 300.0},
        ]
        selected = PreprocessingService.select_best_frames_per_second(scored)
        assert len(selected) == 2
        assert selected[0]["sharpness"] == 200.0
        assert selected[1]["sharpness"] == 300.0

    def test_select_best_frames_empty(self):
        assert PreprocessingService.select_best_frames_per_second([]) == []

    def test_create_video_from_frames_creates_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frame_path = os.path.join(tmpdir, "frame.png")
            frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(frame_path, frame)

            source = os.path.join(tmpdir, "source.mp4")
            out_path = os.path.join(tmpdir, "out.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
            writer = cv2.VideoWriter(source, fourcc, 1, (100, 100))
            writer.write(frame)
            writer.release()

            selected = [{"frame_idx": 0, "timestamp": 0.0, "sharpness": 100.0}]
            result = PreprocessingService.create_video_from_frames(selected, source, out_path, fps=1)
            assert os.path.exists(result)


class TestPreprocessingServiceVibrationMerge:
    def test_merge_vibration_data_merges_two_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df1 = pd.DataFrame({
                "RecIntvl": [0.02, 0.02],
                "acc_x": [1.0, 2.0],
                "speed": [10.0, 11.0],
            })
            df2 = pd.DataFrame({
                "RecIntvl": [0.02, 0.02],
                "acc_x": [3.0, 4.0],
                "speed": [12.0, 13.0],
            })
            path1 = os.path.join(tmpdir, "vib1.csv")
            path2 = os.path.join(tmpdir, "vib2.csv")
            df1.to_csv(path1, index=False)
            df2.to_csv(path2, index=False)

            output = os.path.join(tmpdir, "merged.csv")
            result = PreprocessingService.merge_vibration_data([path1, path2], output, resample_hz=50)

            assert os.path.exists(result)
            merged = pd.read_csv(result)
            assert len(merged) > 0
            assert "time" in merged.columns
            assert "acc_x" in merged.columns
            assert "speed" in merged.columns

    def test_merge_vibration_data_single_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                "RecIntvl": [0.02, 0.02, 0.02],
                "acc_x": [1.0, 2.0, 3.0],
                "speed": [10.0, 11.0, 12.0],
            })
            path = os.path.join(tmpdir, "vib.csv")
            df.to_csv(path, index=False)

            output = os.path.join(tmpdir, "merged.csv")
            result = PreprocessingService.merge_vibration_data([path], output)

            assert os.path.exists(result)
            merged = pd.read_csv(result)
            assert len(merged) == 3

    def test_merge_vibration_data_missing_recintvl_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({"acc_x": [1.0, 2.0], "speed": [10.0, 11.0]})
            path = os.path.join(tmpdir, "bad.csv")
            df.to_csv(path, index=False)

            with pytest.raises(ValueError, match="RecIntvl"):
                PreprocessingService.merge_vibration_data([path], os.path.join(tmpdir, "out.csv"))

    def test_merge_vibration_data_empty_list_raises(self):
        with pytest.raises(ValueError, match="No vibration files"):
            PreprocessingService.merge_vibration_data([], "out.csv")
