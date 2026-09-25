# app/services/preprocessing_service.py
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

from app.services.gps_service import UniversalGPSService


class PreprocessingService:
    """Stitches multiple dashcam videos, extracts/merges GPS trajectories, and selects representative frames."""

    @staticmethod
    def stitch_videos(video_paths: List[str], output_path: str, mode: str = "copy") -> str:
        if not video_paths:
            raise ValueError("No video paths provided for stitching")
        if len(video_paths) == 1:
            shutil.copy2(video_paths[0], output_path)
            return output_path

        file_list_path = output_path + ".txt"
        with open(file_list_path, "w", encoding="utf-8") as f:
            for path in video_paths:
                f.write(f"file '{path}'\n")

        try:
            if mode == "reencode":
                cmd = [
                    "ffmpeg",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", file_list_path,
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "18",
                    "-c:a", "aac",
                    "-y",
                    output_path,
                ]
            else:
                cmd = [
                    "ffmpeg",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", file_list_path,
                    "-c", "copy",
                    "-y",
                    output_path,
                ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg stitching failed: {result.stderr}")
        finally:
            if os.path.exists(file_list_path):
                os.remove(file_list_path)

        return output_path

    @staticmethod
    def extract_gps_trajectory(video_path: str) -> pd.DataFrame:
        try:
            gps_df = UniversalGPSService.extract_gps_from_video(video_path)
            gps_df["source_video"] = os.path.basename(video_path)
            return gps_df
        except Exception:
            return pd.DataFrame(columns=["time", "lat", "lon", "source_video"])

    @staticmethod
    def merge_gps_trajectories(
        gps_frames: List[Dict[str, Any]],
        output_csv: str,
    ) -> str:
        if not gps_frames:
            raise ValueError("No GPS frames provided for merging")

        merged_rows: List[pd.DataFrame] = []
        cumulative_offset = 0.0

        for idx, frame in enumerate(gps_frames):
            gps_df = frame.get("gps_df")
            video_path = frame.get("video_path", "")
            video_duration = frame.get("duration", 0.0)

            if gps_df is None or gps_df.empty:
                continue

            segment = gps_df.copy()
            if idx > 0:
                last_time = merged_rows[-1]["time"] if merged_rows else 0.0
                segment_time = segment["time"].to_numpy(dtype=float)
                if len(segment_time) > 0:
                    segment_duration = float(segment_time[-1]) - float(segment_time[0])
                    if segment_duration > 0:
                        cumulative_offset += segment_duration + 1.0
                    else:
                        cumulative_offset += video_duration + 1.0
                segment["time"] = segment["time"].astype(float) + cumulative_offset

            segment["source_video"] = os.path.basename(video_path)
            merged_rows.append(segment)

        if not merged_rows:
            raise ValueError("No valid GPS data found in any video")

        merged = pd.concat(merged_rows, ignore_index=True)
        merged = merged.sort_values("time").drop_duplicates(subset=["time"], keep="first")
        merged = merged.dropna(subset=["time", "lat", "lon"]).reset_index(drop=True)

        merged[["time", "lat", "lon", "source_video"]].to_csv(output_csv, index=False)
        return output_csv

    @staticmethod
    def get_video_duration(video_path: str) -> float:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except Exception:
            return 0.0

    @staticmethod
    def compute_frame_sharpness(frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def extract_and_score_frames(video_path: str) -> List[Dict[str, Any]]:
        cap = cv2.VideoCapture(video_path)
        scored: List[Dict[str, Any]] = []
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                sharpness = PreprocessingService.compute_frame_sharpness(frame)
                scored.append({
                    "frame_idx": frame_idx,
                    "timestamp": timestamp,
                    "sharpness": sharpness,
                })
                frame_idx += 1
        finally:
            cap.release()

        return scored

    @staticmethod
    def select_best_frames_per_second(scored_frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not scored_frames:
            return []

        best_by_second: Dict[int, Dict[str, Any]] = {}
        for frame in scored_frames:
            second = int(frame["timestamp"])
            if second not in best_by_second or frame["sharpness"] > best_by_second[second]["sharpness"]:
                best_by_second[second] = frame

        return [best_by_second[s] for s in sorted(best_by_second)]

    @staticmethod
    def create_video_from_frames(
        selected_frames: List[Dict[str, Any]],
        source_video: str,
        output_path: str,
        fps: int = 1,
    ) -> str:
        if not selected_frames:
            raise ValueError("No frames selected for downsampled video")

        temp_frames_dir = os.path.join(os.path.dirname(output_path), "temp_selected_frames")
        os.makedirs(temp_frames_dir, exist_ok=True)

        try:
            cap = cv2.VideoCapture(source_video)
            try:
                for idx, frame_info in enumerate(selected_frames):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_info["frame_idx"])
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    frame_path = os.path.join(temp_frames_dir, f"frame_{idx:06d}.png")
                    cv2.imwrite(frame_path, frame)
            finally:
                cap.release()

            frame_pattern = os.path.join(temp_frames_dir, "frame_%06d.png")
            cmd = [
                "ffmpeg",
                "-framerate", str(fps),
                "-i", frame_pattern,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-y",
                output_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg downsampled video creation failed: {result.stderr}")
        finally:
            if os.path.exists(temp_frames_dir):
                shutil.rmtree(temp_frames_dir)

        return output_path

    @staticmethod
    def _detect_vibration_column(df: pd.DataFrame, token: str) -> str:
        for column in df.columns:
            if token in column.lower():
                return str(column)
        return ""

    @classmethod
    def merge_vibration_data(
        cls,
        vibration_paths: List[str],
        output_csv: str,
        resample_hz: int = 50,
    ) -> str:
        if not vibration_paths:
            raise ValueError("No vibration files provided for merging")

        segments: List[pd.DataFrame] = []
        cumulative_offset = 0.0
        acc_col = ""
        speed_col = ""

        for path in vibration_paths:
            df = pd.read_csv(path)

            if "RecIntvl" not in df.columns:
                raise ValueError(f"RecIntvl column not found in {path}")

            acc_col = cls._detect_vibration_column(df, "acc")
            speed_col = cls._detect_vibration_column(df, "speed")

            if not acc_col:
                raise ValueError(f"Acceleration column not found in {path}")
            if not speed_col:
                raise ValueError(f"Speed column not found in {path}")

            df = df.copy()
            df["time"] = df["RecIntvl"].cumsum()

            if len(df) > 0:
                segment_duration = float(df["time"].iloc[-1])
                cumulative_offset += segment_duration + 1.0

            df["time"] = df["time"].astype(float) + cumulative_offset
            df["source_vibration"] = os.path.basename(path)
            segments.append(df)

        merged = pd.concat(segments, ignore_index=True)
        merged = merged.sort_values("time").drop_duplicates(subset=["time"], keep="first")
        merged = merged.reset_index(drop=True)

        if resample_hz and len(merged) > 1:
            t_min = float(merged["time"].iloc[0])
            t_max = float(merged["time"].iloc[-1])
            t_uniform = np.arange(t_min, t_max, 1.0 / resample_hz)

            merged = pd.DataFrame({
                "time": t_uniform,
                "RecIntvl": [1.0 / resample_hz] * len(t_uniform),
                acc_col: np.interp(t_uniform, merged["time"].to_numpy(dtype=float), merged[acc_col].to_numpy(dtype=float)),
                speed_col: np.interp(t_uniform, merged["time"].to_numpy(dtype=float), merged[speed_col].to_numpy(dtype=float)),
                "source_vibration": merged["source_vibration"].iloc[0] if len(merged) else "",
            })

        merged.to_csv(output_csv, index=False)
        return output_csv

    @classmethod
    def preprocess_run(
        cls,
        video_paths: List[str],
        output_dir: str,
        stitch_mode: str = "copy",
        downsample: bool = True,
        downsampled_fps: int = 1,
        vibration_paths: Optional[List[str]] = None,
        vibration_resample_hz: int = 50,
    ) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)

        stitched_path = os.path.join(output_dir, "stitched_video.mp4")
        downsampled_path = os.path.join(output_dir, "downsampled_video.mp4")
        gps_csv_path = os.path.join(output_dir, "merged_gps.csv")
        frame_manifest_path = os.path.join(output_dir, "frame_manifest.json")
        metadata_path = os.path.join(output_dir, "metadata.json")

        stitched_video = cls.stitch_videos(video_paths, stitched_path, mode=stitch_mode)

        gps_frames: List[Dict[str, Any]] = []
        for path in video_paths:
            gps_df = cls.extract_gps_trajectory(path)
            duration = cls.get_video_duration(path)
            gps_frames.append({
                "video_path": path,
                "gps_df": gps_df,
                "duration": duration,
            })

        try:
            merged_gps_csv = cls.merge_gps_trajectories(gps_frames, gps_csv_path)
            gps_df = pd.read_csv(merged_gps_csv)
            gps_points_count = len(gps_df)
        except Exception:
            gps_points_count = 0
            merged_gps_csv = None

        downsampled_video = None
        frame_manifest = None
        selected_frames_count = 0

        if downsample:
            try:
                scored_frames = cls.extract_and_score_frames(stitched_video)
                selected = cls.select_best_frames_per_second(scored_frames)
                selected_frames_count = len(selected)
                cls.create_video_from_frames(selected, stitched_video, downsampled_path, fps=downsampled_fps)
                downsampled_video = os.path.basename(downsampled_path)

                frame_manifest = {
                    "source_video": os.path.basename(stitched_video),
                    "downsampled_fps": downsampled_fps,
                    "total_scored_frames": len(scored_frames),
                    "selected_frames_count": selected_frames_count,
                    "frames": [
                        {
                            "second": int(f["timestamp"]),
                            "timestamp": round(f["timestamp"], 3),
                            "frame_idx": int(f["frame_idx"]),
                            "sharpness": round(f["sharpness"], 2),
                        }
                        for f in selected
                    ],
                }
                with open(frame_manifest_path, "w", encoding="utf-8") as f:
                    json.dump(frame_manifest, f, indent=2)
            except Exception:
                downsampled_video = None
                frame_manifest = None
                selected_frames_count = 0

        metadata = {
            "preprocess_id": os.path.basename(output_dir),
            "input_videos": [os.path.basename(p) for p in video_paths],
            "stitched_video": os.path.basename(stitched_video),
            "downsampled_video": downsampled_video,
            "downsampled_fps": downsampled_fps if downsample else None,
            "gps_csv": os.path.basename(merged_gps_csv) if merged_gps_csv else None,
            "gps_points_count": gps_points_count,
            "selected_frames_count": selected_frames_count,
            "stitch_mode": stitch_mode,
            "downsample": downsample,
        }

        merged_vibration_csv = None
        vibration_csv_count = 0

        if vibration_paths:
            try:
                merged_vibration_csv_path = os.path.join(output_dir, "merged_vibration.csv")
                cls.merge_vibration_data(vibration_paths, merged_vibration_csv_path, resample_hz=vibration_resample_hz)
                merged_vibration_csv = os.path.basename(merged_vibration_csv_path)
                vibration_df = pd.read_csv(merged_vibration_csv_path)
                vibration_csv_count = len(vibration_df)
            except Exception:
                merged_vibration_csv = None
                vibration_csv_count = 0

        metadata["vibration_csv"] = merged_vibration_csv
        metadata["vibration_csv_count"] = vibration_csv_count
        metadata["input_vibrations"] = [os.path.basename(p) for p in vibration_paths] if vibration_paths else []

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return {
            "preprocess_id": os.path.basename(output_dir),
            "stitched_video_path": stitched_video,
            "downsampled_video_path": downsampled_path if downsampled_video else None,
            "gps_csv_path": merged_gps_csv,
            "frame_manifest_path": frame_manifest_path if frame_manifest else None,
            "vibration_csv_path": os.path.join(output_dir, merged_vibration_csv) if merged_vibration_csv else None,
            "metadata": metadata,
        }
