"""Run a YOLO defect model over a video and watch what it detects.

Written for answering "is this model actually seeing the defects in this road?"
on a specific clip, and for re-running that same check later after retraining.
It drives the loop itself rather than calling `process_video_astm`, for three
reasons, all of which matter here:

- Confidence is a parameter. The production call passes none, so Ultralytics
  defaults to 0.25; on real road video this model peaks near 0.13, so at the
  default it reports a clean road on footage that is visibly broken. The
  threshold has to be a dial you can turn.
- Tracking. The production path has none, so every frame's pothole is a new
  row. Counting distinct defects needs the IoU tracker in `vision_tracking`.
- Progress. Nothing in production reports per-frame progress, and a 31-minute
  clip with no output looks identical to a hung process.

The detection maths — undistortion, pixel-to-metre sizing, deduct values,
Ghana ratings, severity, class maps — is the production `VisionAnalysisService`,
called directly rather than reimplemented, so a number here means the same thing
it would mean in a real run.

Usage:
    python scripts/test_yolo_video.py VIDEO --model data/models/defect.pt --conf 0.05
    python scripts/test_yolo_video.py VIDEO --gps gps.csv --save-video --save-crops
    python scripts/test_yolo_video.py VIDEO --max-frames 60 --conf 0.01
"""

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.vision import VisionAnalysisService  # noqa: E402
from scripts.vision_tracking import (  # noqa: E402
    IoUTracker,
    Track,
    build_tracking_registry,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "data" / "models" / "defect.pt"

#: Colours cycled per class so the same defect type is the same colour in the
#: annotated video and in a crop, across runs.
CLASS_COLOURS = [
    (0, 0, 255), (0, 255, 255), (255, 128, 0), (0, 255, 0), (255, 0, 255),
    (0, 128, 255), (255, 255, 0), (128, 0, 255), (255, 128, 128),
]


def colour_for(class_id: int) -> Tuple[int, int, int]:
    return CLASS_COLOURS[class_id % len(CLASS_COLOURS)]


#: Severity bands per class, as (HIGH, MEDIUM) thresholds on that class's own
#: dominant dimension. Cracks and corrugation are linear, so width in mm is
#: what matters; the rest are areal, so area in m² is. Potholes carry no entry
#: because they already have a depth-based classification from the service.
#:
#: These are indicative thresholds for reading a model's output, not a
#: calibration — production rates defects from estimated depth, which this
#: harness deliberately does not run.
SEVERITY_BANDS: Dict[int, Tuple[str, str, str]] = {
    0: ("BLEEDING", "area_m2", 4.0, 1.0),        # Bleeding
    1: ("CORRUGATION", "width_mm", 50.0, 20.0),  # Corrugation
    2: ("CRACK", "width_mm", 10.0, 3.0),         # Crack
    3: ("PATCHING", "area_m2", 3.0, 1.0),         # Patching
    5: ("RAVELLING", "area_m2", 3.0, 1.0),       # Ravelling
    6: ("RUTTING", "area_m2", 2.0, 0.5),         # Rutting
    7: ("SHOVING", "width_mm", 75.0, 25.0),      # Shoving
    8: ("SUBGRADE", "area_m2", 5.0, 1.5),        # Subgrade Failure
}


def _severity_for(
    class_id: int, width_m: float, area_m2: float, pothole_severity: Optional[str]
) -> str:
    """HIGH / MEDIUM / LOW for any class, not only potholes."""
    if class_id == 4:
        return pothole_severity or "UNKNOWN"
    band = SEVERITY_BANDS.get(class_id)
    if band is None:
        return "UNKNOWN"
    name, dimension, high, medium = band
    value = width_m * 1000.0 if dimension == "width_mm" else area_m2
    if value >= high:
        return "HIGH"
    if value >= medium:
        return "MEDIUM"
    return "LOW"


def _pick_time_columns(fieldnames: Sequence[str]) -> List[str]:
    """Columns that might hold seconds, best guess first.

    These logger exports name their columns confusingly: in `vid_gps_*.csv` the
    header reads `Global_Second,Timestamp,...` but the values are the other way
    round — `Global_Second` carries a wall-clock timestamp and `Timestamp`
    carries the elapsed time. So the time column is chosen by looking at the
    values, not by trusting the header.
    """
    return [
        name
        for name in fieldnames
        if name and any(token in name.lower() for token in ("second", "time", "elapsed"))
    ]


def load_gps(path: str) -> List[Dict[str, Any]]:
    """Read a logger GPS CSV into the points the service's interpolator wants.

    The service interpolates on a `time` column, and these files carry seconds
    since the logger started. It is rebased to zero so it lines up with a video
    timestamp measured from its first frame, which is the only common origin the
    two files share. That rebase is load-bearing, not cosmetic: the same trip is
    exported twice in this dataset, once with the counter starting at 8241, and
    left unrebased its whole track sits outside the video's timeline so every
    detection resolves to the first fix.
    """
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []
        # Materialised once: a DictReader is a one-shot iterator, so probing it
        # for a usable time column and then reading it again would silently
        # yield nothing the second time.
        rows = list(reader)

        # Lat/lon are looked up by name, with the trailing space these exports
        # put on the last column stripped.
        fields = {(f or "").strip(): f for f in reader.fieldnames}
        lat_key = next((fields[n] for n in ("Latitude", "latitude", "lat") if n in fields), None)
        lon_key = next((fields[n] for n in ("Longitude", "longitude", "lon", "long") if n in fields), None)
        if not lat_key or not lon_key:
            return []

        time_key = None
        for candidate in _pick_time_columns(reader.fieldnames):
            if any(_as_seconds(row.get(candidate)) is not None for row in rows[:50]):
                time_key = candidate
                break
        if time_key is None:
            return []

        points: List[Dict[str, Any]] = []
        for row in rows:
            seconds = _as_seconds(row.get(time_key))
            lat = _as_float(row.get(lat_key))
            lon = _as_float(row.get(lon_key))
            if seconds is None or lat is None or lon is None:
                continue
            points.append({"time": seconds, "lat": lat, "lon": lon})

    if not points:
        return []
    base = min(p["time"] for p in points)
    for point in points:
        point["time"] -= base
    points.sort(key=lambda p: p["time"])
    return points


def _as_seconds(value: Any) -> Optional[float]:
    """Seconds from a plain number, or from an `H:MM:SS` / `MM:SS` elapsed clock.

    The same export family writes elapsed time both ways, so a probe that only
    accepts plain numbers rejects the `0:00:00` files outright and reports no
    GPS at all rather than a wrong one.
    """
    number = _as_float(value)
    if number is not None:
        return number
    parts = str(value).strip().split(":")
    if len(parts) not in (2, 3) or not all(re.fullmatch(r"\d{1,4}(?:\.\d+)?", p) for p in parts):
        return None
    return math.fsum(float(p) * 60.0**i for i, p in enumerate(reversed(parts)))


def _as_float(value: Any) -> Optional[float]:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _fmt_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return f"{seconds // 3600:d}h{(seconds % 3600) // 60:02d}m{seconds % 60:02d}s"


def _fmt_chainage(metres: float) -> str:
    return f"{metres / 1000.0:7.3f} km"


class ConsoleMonitor:
    """Live progress, so a long pass is watchable rather than silent.

    Rewrites one block in place on a TTY. When output is piped to a file there
    is no cursor to move, so it drops to periodic one-line status records — the
    same information, in a form a log file can hold.
    """

    def __init__(self, total_frames: int, processed_every: int = 1) -> None:
        self.total_frames = total_frames
        self.processed_every = processed_every
        self.tty = sys.stdout.isatty()
        self.started = time.time()
        self.frames_done = 0
        self._lines_drawn = 0
        self.events: List[str] = []

    def _elapsed(self) -> float:
        return time.time() - self.started

    def _eta(self) -> Optional[float]:
        if self.frames_done < 1 or self.total_frames <= 0:
            return None
        rate = self.frames_done / self._elapsed()
        if rate <= 0:
            return None
        return (self.total_frames - self.frames_done) / rate

    def update(
        self,
        frame_idx: int,
        detections: List[Dict[str, Any]],
        live_tracks: int,
        total_tracks: int,
        elapsed: float,
        note: str = "",
    ) -> None:
        self.frames_done += 1

        if not self.tty:
            if self.frames_done % max(1, self.processed_every) == 0 or note:
                eta = self._eta()
                print(
                    f"frame={frame_idx}/{self.total_frames} dets={len(detections)} "
                    f"live_tracks={live_tracks} total_tracks={total_tracks} "
                    f"elapsed={_fmt_duration(elapsed)}"
                    + (f" eta={_fmt_duration(eta)}" if eta else "")
                    + (f" | {note}" if note else ""),
                    flush=True,
                )
            return

        by_class = Counter(d["defect_type"] for d in detections)
        summary = ", ".join(f"{k}x{v}" for k, v in sorted(by_class.items())) or "none"
        eta = self._eta()

        lines = [
            f"  frame {frame_idx}/{self.total_frames}  ({self.frames_done * 100.0 / max(1, self.total_frames):5.1f}%)",
            f"  detections this frame : {len(detections)}  [{summary}]",
            f"  tracks live / total   : {live_tracks} / {total_tracks}",
            f"  elapsed / eta         : {_fmt_duration(elapsed)}"
            + (f" / {_fmt_duration(eta)}" if eta else ""),
        ]
        if note:
            lines.append(f"  {note}")

        # Erase the previous block, then draw this one.
        sys.stdout.write("\x1b[999D\x1b[J")
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()
        self._lines_drawn = len(lines)

    def clear(self) -> None:
        if self.tty and self._lines_drawn:
            sys.stdout.write("\x1b[999D\x1b[J")
            sys.stdout.flush()
            self._lines_drawn = 0


def annotate(
    frame: np.ndarray,
    pairs: Sequence[Tuple[Track, Dict[str, Any]]],
    class_names: Dict[int, str],
    show_track: bool = True,
) -> np.ndarray:
    """Draw boxes, class, confidence and track id onto a copy of the frame."""
    out = frame.copy()
    height, width = out.shape[:2]
    # The tracker works in the undistorted frame's coordinates, so boxes are
    # drawn here on the undistorted frame; the original is kept for the video.
    for track, det in pairs:
        x1, y1, x2, y2 = (int(v) for v in det["bbox"])
        colour = colour_for(det["yolo_class_id"])
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 3)

        name = class_names.get(det["yolo_class_id"], "Unknown")
        label = f"{name} {det['confidence']:.2f}"
        if show_track:
            label = f"#{track.track_id} {label}"

        scale = 0.8
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        ty = max(th + baseline + 2, y1)
        cv2.rectangle(out, (x1, ty - th - baseline - 2), (x1 + tw + 6, ty), colour, -1)
        cv2.putText(
            out, label, (x1 + 3, ty - baseline - 1),
            cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness, cv2.LINE_AA,
        )
    return out


def save_crop(
    frame: np.ndarray, det: Dict[str, Any], track: Track, out_dir: Path, margin: int = 40
) -> Optional[str]:
    """Write a padded crop around a track's clearest detection."""
    x1, y1, x2, y2 = (int(v) for v in det["bbox"])
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
    x2, y2 = min(w, x2 + margin), min(h, y2 + margin)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    crop = frame[y1:y2, x1:x2]
    name = f"track_{track.track_id:04d}_{det['defect_type'].lower().replace(' ', '_')}_f{det['frame_idx']}.jpg"
    path = out_dir / name
    cv2.imwrite(str(path), crop)
    return str(path)


def run(args: argparse.Namespace) -> int:
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"error: video not found: {video_path}", file=sys.stderr)
        return 2

    model_path = Path(args.model) if args.model else DEFAULT_MODEL
    if not model_path.exists():
        print(
            f"error: model not found: {model_path}\n"
            f"       pass --model, or place weights at {DEFAULT_MODEL}",
            file=sys.stderr,
        )
        return 2

    service = VisionAnalysisService(
        model_path=str(model_path),
        camera_height_m=args.camera_height,
        camera_tilt_deg=args.camera_tilt,
        fx=args.fx,
        fy=args.fy,
        # 4K footage: the production defaults are 1920x1080 principal points and
        # would put every ground estimate at the wrong place in the frame.
        cx=args.cx,
        cy=args.cy,
        distortion_coefficients=args.distortion,
        # Depth pulls a second neural net; off unless asked for, since it is
        # several times the cost and this harness is about what YOLO sees.
        enable_depth_estimation=args.depth,
    )
    if service.model is None:
        print(f"error: YOLO failed to load from {model_path}", file=sys.stderr)
        return 2

    class_names = dict(service.CLASS_NAMES)
    model_names = getattr(service.model, "names", {}) or {}
    if model_names and set(model_names) != set(class_names):
        print(
            f"warning: model classes {sorted(model_names)} do not match the service's "
            f"map {sorted(class_names)}; severity and rating rules may not apply correctly",
            file=sys.stderr,
        )

    gps_points = load_gps(args.gps) if args.gps else None
    if args.gps and not gps_points:
        print(f"warning: no usable GPS rows in {args.gps}", file=sys.stderr)
    gps_interp = service._create_gps_interpolator(gps_points) if gps_points else None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"error: could not open video: {video_path}", file=sys.stderr)
        return 2

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        total_frames = min(total_frames, args.max_frames)
    stride = max(1, int(round(src_fps / args.fps))) if args.fps > 0 else 1

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) / f"yolo_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"video     : {video_path}")
    print(f"model     : {model_path}")
    print(f"size      : {width}x{height} @ {src_fps:.2f} fps, {total_frames} frames "
          f"({_fmt_duration(total_frames / src_fps)} at 1x)")
    print(f"sampling  : every {stride} frame(s) -> ~{total_frames // stride} inferences")
    print(f"thresholds: conf>={args.conf} iou>={args.iou} imgsz={args.imgsz} device={args.device}")
    print(f"tracking  : IoU>={args.track_iou} miss<={args.max_missed}")
    print(f"output    : {out_dir}")
    print(f"gps       : {'yes' if gps_interp else 'no'}")
    print()

    tracker = IoUTracker(iou_threshold=args.track_iou, max_missed=args.max_missed)
    monitor = ConsoleMonitor(total_frames=total_frames, processed_every=args.report_every)

    writer = None
    if args.save_video:
        # Written in the detector's coordinates for consistency with the boxes;
        # mp4v is the broadly available fourcc for .mp4 output.
        target_size = None
        if args.video_scale != 1.0:
            target_size = (int(width * args.video_scale), int(height * args.video_scale))
        writer = cv2.VideoWriter(
            str(out_dir / "annotated.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(1.0, src_fps / stride),
            target_size or (width, height),
        )

    crops_dir = out_dir / "crops"
    if args.save_crops:
        crops_dir.mkdir(exist_ok=True)

    frame_records: List[Dict[str, Any]] = []
    crop_index: Dict[int, str] = {}
    frame_idx = 0
    processed = 0
    started = time.time()

    try:
        while processed < total_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            processed += 1
            working = service.undistort_frame(frame) if not args.no_undistort else frame
            timestamp = frame_idx / src_fps

            lat, lon = (0.0, 0.0)
            if gps_interp:
                lat, lon = gps_interp(timestamp)

            # The single call that matters. Confidence and IoU are passed here
            # because this is the only place they can be set.
            results = service.model(
                working, verbose=False, conf=args.conf, iou=args.iou,
                imgsz=args.imgsz, device=args.device,
            )

            detections: List[Dict[str, Any]] = []
            for result in results or []:
                boxes = getattr(result, "boxes", None)
                if boxes is None or not len(boxes):
                    continue
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = [float(v) for v in box.xyxy[0].tolist()]

                    width_m, length_m, area_m2 = service._pixel_to_real_world(
                        xyxy, (working.shape[0], working.shape[1])
                    )

                    pothole_dia = pothole_dep = pothole_sev = None
                    if cls_id == 4:
                        pothole_dia = width_m * 1000.0
                        # Fixed 30mm, as production does: this harness measures
                        # what the detector sees, not what a depth model infers.
                        pothole_dep = 30.0
                        pothole_sev = service._classify_pothole_severity(pothole_dia, pothole_dep)

                    if cls_id == 4:
                        ghana_rating = 7 if pothole_sev == "HIGH" else (4 if pothole_sev == "MEDIUM" else 2)
                    elif cls_id == 2:
                        crack_mm = width_m * 1000.0
                        ghana_rating = 7 if crack_mm > 10 else (4 if crack_mm > 3 else 2)
                    elif cls_id in (6, 7):
                        ghana_rating = 7 if area_m2 > 0.5 else (4 if area_m2 > 0.1 else 2)
                    else:
                        ghana_rating = 7 if area_m2 > 1.0 else (4 if area_m2 > 0.2 else 2)

                    # Severity for every class, not just potholes. Production
                    # only rates potholes (the depth model is pothole-specific),
                    # which leaves the other eight classes with a rating and no
                    # readable severity. Sized off each distress's own dominant
                    # dimension, since a wide crack and a broad patch of
                    # bleeding are not comparable by area.
                    severity = _severity_for(cls_id, width_m, area_m2, pothole_sev)

                    detections.append(
                        {
                            "frame_idx": frame_idx,
                            "yolo_class_id": cls_id,
                            "defect_type": class_names.get(cls_id, "Unknown"),
                            "astm_distress_type": service.YOLO_TO_AASHTO.get(cls_id, "Unknown"),
                            "ghana_distress_type": service.YOLO_TO_GHANA.get(cls_id, "Unknown"),
                            "confidence": round(conf, 4),
                            "bbox": xyxy,
                            "width": round(width_m, 3),
                            "length": round(length_m, 3),
                            "area": round(area_m2, 4),
                            "deduct_value": round(service._calculate_deduct_value(cls_id, area_m2), 2),
                            "ghana_rating_1to9": ghana_rating,
                            "chainage_m": round(timestamp * (args.assumed_speed_kmh / 3.6), 2),
                            "lat": lat,
                            "lon": lon,
                            "pothole_diameter_mm": round(pothole_dia, 2) if pothole_dia else None,
                            "pothole_depth_mm": pothole_dep,
                            "pothole_severity": pothole_sev,
                            "severity": severity,
                        }
                    )

            pairs = tracker.update(detections, frame_idx)
            live = len(tracker._active)  # noqa: SLF001 - harness needs live count
            total = len(tracker.tracks)

            if detections:
                frame_records.append(
                    {
                        "frame_idx": frame_idx,
                        "timestamp_s": round(timestamp, 2),
                        "chainage_m": round(timestamp * (args.assumed_speed_kmh / 3.6), 2),
                        "lat": round(lat, 6),
                        "lon": round(lon, 6),
                        "detections": [
                            {
                                "track_id": t.track_id,
                                **{k: v for k, v in d.items() if k != "bbox"},
                                "bbox": [round(b, 1) for b in d["bbox"]],
                            }
                            for t, d in pairs
                        ],
                    }
                )

            if args.save_crops:
                for track, det in pairs:
                    if track.track_id not in crop_index:
                        saved = save_crop(working, det, track, crops_dir)
                        if saved:
                            crop_index[track.track_id] = saved

            if writer is not None:
                annotated = annotate(working, pairs, class_names)
                if args.video_scale != 1.0:
                    annotated = cv2.resize(annotated, (int(width * args.video_scale), int(height * args.video_scale)))
                writer.write(annotated)

            elapsed = time.time() - started
            rate = processed / elapsed if elapsed > 0 else 0.0
            note = ""
            if rate > 0 and processed % 50 == 0:
                note = f"~{rate:.1f} frame/s, {elapsed:.0f}s elapsed"
            monitor.update(frame_idx, detections, live, total, elapsed, note)

            frame_idx += 1
    except KeyboardInterrupt:
        print("\ninterrupted; writing what has been collected so far")
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    monitor.clear()
    # Counted before merging so the report can show how many duplicates the
    # merge pass removed rather than only the final number.
    args.raw_track_count = len(tracker.finish(merge=False))
    tracks = tracker.finish(merge=not args.no_merge)
    elapsed = time.time() - started

    # ---- per-frame detail -------------------------------------------------
    frames_path = out_dir / "detections_per_frame.json"
    frames_path.write_text(json.dumps(frame_records, indent=2), encoding="utf-8")

    # ---- one row per distinct defect -------------------------------------
    tracks_path = out_dir / "tracks.csv"
    with open(tracks_path, "w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.writer(handle)
        writer_csv.writerow(
            [
                "track_id", "defect_type", "yolo_class_id", "hits", "first_frame", "last_frame",
                "best_confidence", "mean_confidence", "max_width_m", "max_length_m", "max_area_m2",
                "max_deduct_value", "total_deduct_value", "chainage_m", "severity",
                "pothole_severity", "crop",
            ]
        )
        for track in sorted(tracks, key=lambda t: t.first_frame):
            writer_csv.writerow(
                [
                    track.track_id,
                    class_names.get(track.yolo_class_id, "Unknown"),
                    track.yolo_class_id,
                    track.hits,
                    track.first_frame,
                    track.last_frame,
                    round(track.best_confidence, 4),
                    round(track.mean_confidence, 4),
                    round(track.max_width, 3),
                    round(track.max_length, 3),
                    round(track.max_area, 4),
                    round(track.max_deduct_value, 2),
                    round(track.total_deduct_value, 2),
                    round(track.start_chainage, 2),
                    track.severity,
                    "HIGH" if track.max_deduct_value >= 5 else "",
                    crop_index.get(track.track_id, ""),
                ]
            )

    # ---- PCI, through the production fusion service -----------------------
    # From the fusion service, not the vision one: PCI is computed by fusing
    # vision with IRI, and this harness has no IRI, so only the vision half is
    # reported. It is the same function production calls, so the number is
    # comparable.
    from app.services.fusion import PavementConditionFusioneer

    registry = build_tracking_registry(tracks, class_names)
    vision_pci, pci_breakdown = PavementConditionFusioneer._compute_vision_pci(registry)

    by_type = Counter(class_names.get(t.yolo_class_id, "Unknown") for t in tracks)
    by_severity = Counter(t.severity or "UNRATED" for t in tracks)
    # Per class and severity together, so a report cannot read as "12 cracks"
    # when 4 of them are severe and 8 are hairline.
    type_severity: Dict[str, Dict[str, int]] = {}
    for track in tracks:
        name = class_names.get(track.yolo_class_id, "Unknown")
        type_severity.setdefault(name, {})
        level = track.severity or "UNRATED"
        type_severity[name][level] = type_severity[name].get(level, 0) + 1

    all_confidences = [t.best_confidence for t in tracks if t.best_confidence > 0]
    potholes = [t for t in tracks if t.yolo_class_id == 4]

    summary = {
        "video": str(video_path),
        "model": str(model_path),
        "resolution": f"{width}x{height}",
        "source_fps": round(src_fps, 3),
        "stride": stride,
        "frames_read": frame_idx,
        "frames_processed": processed,
        "elapsed_s": round(elapsed, 1),
        "thresholds": {
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "track_iou": args.track_iou,
            "max_missed": args.max_missed,
            "merge_chainage_m": args.merge_distance,
        },
        "detections": {
            "total_frame_detections": sum(len(r["detections"]) for r in frame_records),
            "frames_with_detections": len(frame_records),
            "raw_tracks_before_merge": args.raw_track_count,
            "distinct_tracks": len(tracks),
            "duplicates_merged": max(0, args.raw_track_count - len(tracks)),
            "pothole_tracks": len(potholes),
            "by_type": dict(by_type),
            "by_severity": dict(by_severity),
            "by_type_and_severity": type_severity,
            "max_confidence": round(max(all_confidences), 4) if all_confidences else 0.0,
            "mean_track_confidence": round(sum(all_confidences) / len(all_confidences), 4) if all_confidences else 0.0,
        },
        "vision_pci": round(vision_pci, 2),
        "pci_breakdown": pci_breakdown,
        "gps_used": bool(gps_interp),
        "outputs": {
            "tracks_csv": str(tracks_path),
            "per_frame_json": str(frames_path),
            "annotated_video": str(out_dir / "annotated.mp4") if args.save_video else None,
            "crops_dir": str(crops_dir) if args.save_crops else None,
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ---- report -----------------------------------------------------------
    print("=" * 72)
    print(f"frames processed : {processed} of {total_frames} in {elapsed:.1f}s "
          f"({processed / elapsed if elapsed > 0 else 0:.1f}/s)")
    print(f"detections       : {summary['detections']['total_frame_detections']} across "
          f"{summary['detections']['frames_with_detections']} frames")
    print(f"distinct defects : {len(tracks)}")
    if not args.no_merge:
        print(f"  duplicates     : {args.raw_track_count} raw tracks -> {len(tracks)} "
              f"after merge ({summary['detections']['duplicates_merged']} folded in)")
    print(f"  potholes       : {len(potholes)}")
    if by_severity:
        print("  by severity    : " + ", ".join(
            f"{k}={v}" for k, v in sorted(by_severity.items(), key=lambda kv: -kv[1])
        ))
    if type_severity:
        # Every class, not just potholes: a count per type with a severity
        # breakdown, so a road that is mostly hairline cracks cannot be
        # confused with one carrying severe distress.
        print("  by type        :")
        for name in sorted(type_severity):
            levels = ", ".join(
                f"{k}={v}" for k, v in sorted(type_severity[name].items(), key=lambda kv: -kv[1])
            )
            print(f"      {name:<20} {by_type[name]:>3}  ({levels})")
    if all_confidences:
        print(f"confidence       : max {summary['detections']['max_confidence']:.3f}, "
              f"mean {summary['detections']['mean_track_confidence']:.3f}")
    print(f"vision PCI       : {vision_pci:.2f}  "
          f"(total DV {pci_breakdown['total_dv']:.2f}, "
          f"factor {pci_breakdown['correction_factor']})")
    if tracks:
        print()
        print(f"{'id':>4} {'type':<16} {'sev':<7} {'hits':>4} {'frames':>12} {'conf':>6} "
              f"{'width_m':>8} {'area_m2':>8} {'DV':>6} {'chainage':>14}")
        for track in sorted(tracks, key=lambda t: t.first_frame)[:25]:
            print(
                f"{track.track_id:>4} {class_names.get(track.yolo_class_id, '?'):<16} "
                f"{track.severity or '-':<7} "
                f"{track.hits:>4} {track.first_frame:>5}-{track.last_frame:<6} "
                f"{track.best_confidence:>6.3f} {track.max_width:>8.3f} "
                f"{track.max_area:>8.3f} {track.max_deduct_value:>6.2f} "
                f"{_fmt_chainage(track.start_chainage):>14}"
            )
        if len(tracks) > 25:
            print(f"     ... {len(tracks) - 25} more in {tracks_path.name}")
    if not tracks:
        print()
        print("No distinct defects tracked. If the video clearly has defects, try:")
        print(f"  --conf 0.01        (this run used {args.conf}; detections peak near 0.13)")
        print("  --max-frames 200   (check a longer stretch)")
    print()
    print(f"summary : {summary_path}")
    print(f"tracks  : {tracks_path}")
    print(f"frames  : {frames_path}")
    if args.save_video:
        print(f"video   : {out_dir / 'annotated.mp4'}")
    if args.save_crops:
        print(f"crops   : {crops_dir} ({len(crop_index)} tracks)")
    print("=" * 72)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a YOLO defect model over a video and report tracked detections.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", help="path to the video file")
    parser.add_argument("--model", default=None, help=f"weights file (default: {DEFAULT_MODEL})")
    parser.add_argument("--gps", default=None, help="logger GPS CSV for lat/lon on each detection")

    parser.add_argument("--conf", type=float, default=0.05,
                        help="detection confidence floor. The default 0.25 is above what this "
                             "model scores on real road video, so it reports nothing")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU inside the model. Lower than Ultralytics' 0.7, which "
                             "leaves two or three overlapping boxes on one defect — those "
                             "become separate tracks and inflate the defect count")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument("--device", default=None, help="torch device, e.g. 0 or cpu")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="frames per second to actually infer on; the rest are skipped")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = whole video)")

    parser.add_argument("--track-iou", type=float, default=0.3,
                        help="IoU for a detection to continue an existing track")
    parser.add_argument("--max-missed", type=int, default=5,
                        help="frames a track may go unseen before it is retired")
    parser.add_argument("--merge-distance", type=float, default=30.0,
                        help="metres along the road within which two same-class tracks are "
                             "treated as one defect seen twice")
    parser.add_argument("--no-merge", action="store_true",
                        help="skip the duplicate-merge pass and report raw tracks")

    parser.add_argument("--camera-height", type=float, default=2.5, help="camera height in metres")
    parser.add_argument("--camera-tilt", type=float, default=0.0, help="camera tilt in degrees")
    parser.add_argument("--fx", type=float, default=3000.0, help="focal length x, in pixels at 4K")
    parser.add_argument("--fy", type=float, default=3000.0, help="focal length y, in pixels at 4K")
    parser.add_argument("--cx", type=float, default=1920.0, help="principal point x")
    parser.add_argument("--cy", type=float, default=1080.0, help="principal point y")
    parser.add_argument("--distortion", default=None, help="JSON list of 5 distortion coefficients")
    parser.add_argument("--no-undistort", action="store_true",
                        help="skip undistortion; use raw frames")
    parser.add_argument("--assumed-speed-kmh", type=float, default=50.0,
                        help="speed used to turn video time into chainage when no GPS is supplied")
    parser.add_argument("--depth", action="store_true",
                        help="also run depth estimation (much slower)")

    parser.add_argument("--save-video", action="store_true", help="write annotated.mp4")
    parser.add_argument("--video-scale", type=float, default=0.25,
                        help="scale factor for the annotated video; 4K is unwatchable at full size")
    parser.add_argument("--save-crops", action="store_true", help="write a crop per tracked defect")
    parser.add_argument("--output", default=str(REPO_ROOT / "outputs" / "vision_tests"),
                        help="directory to write results into")
    parser.add_argument("--report-every", type=int, default=25,
                        help="status line every N frames when not attached to a terminal")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
