import argparse
import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "defect.pt"


def infer_time_series(df: pd.DataFrame, time_column: Optional[str] = None) -> pd.Series:
    """Infer a time series from the CSV using a preferred time column when present."""
    if time_column and time_column in df.columns:
        return pd.to_numeric(df[time_column], errors="coerce")

    for candidate in ["time", "timestamp", "time_s", "seconds", "frame_time"]:
        if candidate in df.columns:
            return pd.to_numeric(df[candidate], errors="coerce")

    raise ValueError("No suitable time column found in CSV")


def align_detection_to_csv(detection_time_s: float, df: pd.DataFrame, time_series: pd.Series) -> int:
    """Return the nearest row index for a detection timestamp."""
    if time_series.empty:
        raise ValueError("CSV time series is empty")

    numeric_times = pd.to_numeric(time_series, errors="coerce")
    numeric_times = numeric_times.dropna()
    if numeric_times.empty:
        raise ValueError("CSV time series has no numeric values")

    diffs = (numeric_times - detection_time_s).abs()
    min_diff = diffs.min()
    idx = diffs[diffs == min_diff].index[-1]
    return int(idx)


def load_model(model_path: Optional[str] = None):
    model_file = Path(model_path or DEFAULT_MODEL_PATH)
    if not model_file.exists():
        raise FileNotFoundError(f"YOLO model not found: {model_file}")
    return YOLO(str(model_file))


def detect_defects_in_video(
    video_path: str,
    model,
    csv_path: Optional[str] = None,
    time_column: Optional[str] = None,
    conf: float = 0.35,
    output_csv: Optional[str] = None,
    output_video: Optional[str] = None,
    save_frames: bool = False,
    frames_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    csv_rows: Optional[pd.DataFrame] = None
    time_series: Optional[pd.Series] = None
    if csv_path:
        csv_rows = pd.read_csv(csv_path)
        time_series = infer_time_series(csv_rows, time_column)

    detections: List[Dict[str, Any]] = []
    frame_index = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    else:
        out = None

    if save_frames and frames_dir:
        os.makedirs(frames_dir, exist_ok=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_index += 1
        frame_time_s = frame_index / fps

        results = model(frame, stream=False, conf=conf, imgsz=640)
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf_score = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = result.names[cls_id]

                detection = {
                    "frame": frame_index,
                    "timestamp_s": frame_time_s,
                    "label": label,
                    "confidence": conf_score,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }

                if csv_rows is not None and time_series is not None:
                    csv_idx = align_detection_to_csv(frame_time_s, csv_rows, time_series)
                    csv_row = csv_rows.iloc[csv_idx].to_dict()
                    detection["csv_row_index"] = csv_idx
                    detection.update({f"csv_{k}": v for k, v in csv_row.items()})

                detections.append(detection)

                if out is not None:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{label} {conf_score:.2f}",
                        (int(x1), max(10, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

        if out is not None:
            out.write(frame)

        if save_frames and frames_dir:
            frame_path = Path(frames_dir) / f"frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)

    cap.release()
    if out is not None:
        out.release()

    if output_csv:
        pd.DataFrame(detections).to_csv(output_csv, index=False)

    return detections


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO defect detector on a video and align detections with a CSV")
    parser.add_argument("--video", required=True, help="Path to the input video")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path to the YOLO .pt model")
    parser.add_argument("--csv", help="Optional CSV with vibration/georeferencing data")
    parser.add_argument("--time-column", help="Optional column in the CSV to use as the time index")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--output-csv", help="Path to write detection results")
    parser.add_argument("--output-video", help="Path to write annotated video")
    parser.add_argument("--save-frames", action="store_true", help="Save every frame as a JPG")
    parser.add_argument("--frames-dir", help="Directory where frames should be saved")
    args = parser.parse_args()

    model = load_model(args.model)
    detections = detect_defects_in_video(
        video_path=args.video,
        model=model,
        csv_path=args.csv,
        time_column=args.time_column,
        conf=args.conf,
        output_csv=args.output_csv,
        output_video=args.output_video,
        save_frames=args.save_frames,
        frames_dir=args.frames_dir,
    )

    print(f"Detected {len(detections)} boxes")


if __name__ == "__main__":
    main()
