"""Check the production vision path against a real survey video.

Runs `VisionAnalysisService.process_video_with_depth` — the method the analysis
route calls — over a short clip of a real drive, with the real GPS CSV loaded
through the production GPS loader. The unit tests cover this code with stubbed
models and synthetic frames; this is the check that the assembled pipeline
still produces sensible output on the footage it will actually be given, and it
reports rather than asserts, because its job is diagnosis.

    python scripts/check_vision_pipeline.py --video PATH --gps PATH [--conf 0.10]

It exists because the failures it would have caught were all silent:

- `DefectDetection(frame_idx=...)` raised TypeError on the first detection of
  every run, and the caller saw a survey that found nothing rather than a crash.
- The video GPS writes the elapsed video clock as `H:MM:SS` strings (`0:00:00`,
  `0:37:21`) in a column named `Timestamp`, with the wall clock beside it in
  `Global_Second`. Reading that column as a number and then as a datetime left
  it at zero — `to_numeric` yields nothing for `0:00:00`, and dateutil reads it
  as midnight on an arbitrary date — so every frame in the video resolved to the
  first GPS fix, indistinguishable from a parked vehicle. Some exports instead
  carry a session counter in the thousands, which has the same effect unless the
  axis is rebased onto the first fix.
- Chainage derived from an assumed 10 m/s reported a plausible number for a
  survey that never went that fast.
"""
import argparse
import sys
import tempfile
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.gps.service import UniversalGPSService  # noqa: E402
from app.services.vision.service import DEFAULT_MODEL_PATH, VisionAnalysisService  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", required=True, help="survey video to check")
    parser.add_argument("--gps", help="GPS CSV for the same trip")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="YOLO checkpoint")
    parser.add_argument("--frames", type=int, default=40, help="clip length to check")
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.45)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print(f"model default: {DEFAULT_MODEL_PATH}")
    print(f"model exists : {Path(DEFAULT_MODEL_PATH).is_file()}")

    gps_points = None
    if args.gps:
        gps_df = UniversalGPSService.load_gps_csv(args.gps)
        gps_points = UniversalGPSService.dataframe_to_spatial_points(gps_df)
        print(
            f"gps points   : {len(gps_points)}  "
            f"time {gps_points[0]['time']:.1f}..{gps_points[-1]['time']:.1f}s"
        )
        print(
            f"gps first/last lat-lon: {gps_points[0]['lat']:.5f},{gps_points[0]['lon']:.5f}"
            f"  ->  {gps_points[-1]['lat']:.5f},{gps_points[-1]['lon']:.5f}"
        )

    with tempfile.TemporaryDirectory() as tmp:
        clip = str(Path(tmp) / "clip.mp4")
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"could not open video: {args.video}")
            return 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        written = 0
        for _ in range(args.frames):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
        cap.release()
        writer.release()
        print(f"clip         : {width}x{height} @ {fps:.3f} fps, {written} frames")

        service = VisionAnalysisService(
            model_path=args.model, enable_depth_estimation=False, conf=args.conf, iou=args.iou
        )
        print(f"model loaded : {service.model is not None}")
        if service.model is None:
            return 1

        result = service.process_video_with_depth(
            video_path=clip,
            gps_points=gps_points,
            run_id="check_vision_pipeline",
            target_fps=1.0,
            estimate_dense_depth=False,
        )

    detections = result.detections
    print(f"detections   : {len(detections)}")
    print(f"frame sampling: {result.frame_sampling}")
    print(f"by type      : {dict(Counter(d.get('defect_type') for d in detections))}")
    print(f"frame_idx    : {sorted({d.get('frame_idx') for d in detections})[:12]}")
    if any(d.get("frame_idx") is None for d in detections):
        print("WARNING: detections with no frame index")

    positions = {tuple(d["geom"]["coordinates"]) for d in detections if d.get("geom")}
    print(f"distinct positions: {len(positions)}")
    if positions and len(positions) == 1:
        print("WARNING: every detection resolved to one GPS fix - check the GPS time axis")

    chainages = [d.get("chainage_m") for d in detections]
    if chainages:
        print(f"chainage     : {min(chainages):.1f} .. {max(chainages):.1f} m")
        print(f"chainage src : {sorted({str(d.get('chainage_source')) for d in detections})}")
        if {d.get("chainage_source") for d in detections} == {"assumed_speed"} and gps_points:
            print("WARNING: GPS was supplied but chainage fell back to an assumed speed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
