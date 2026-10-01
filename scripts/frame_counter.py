import cv2
import csv
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_FOLDER = Path(r"F:\data_collection\KNUST_Kumasi_to_KNUST_Obuasi")

OUTPUT_CSV = VIDEO_FOLDER / "video_frame_integrity_report.csv"

VIDEO_EXTENSIONS = {
    ".mp4",
    ".MP4",
    ".mov",
    ".MOV",
    ".mkv",
    ".MKV",
    ".avi",
    ".AVI",
    ".m4v",
    ".M4V",
}


# ============================================================
# FFMPEG / FFPROBE
# ============================================================

def get_ffprobe_metadata(video_path):
    """
    Get reliable container/stream metadata using ffprobe.
    """

    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream="
        "codec_name,"
        "codec_long_name,"
        "width,"
        "height,"
        "r_frame_rate,"
        "avg_frame_rate,"
        "nb_frames,"
        "duration,"
        "bit_rate",
        "format="
        "duration,"
        "size,"
        "bit_rate",
        "-of", "json",
        str(video_path),
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )

        data = json.loads(result.stdout)

        stream = data.get("streams", [{}])[0]
        fmt = data.get("format", {})

        return stream, fmt

    except Exception as e:
        return {}, {
            "ffprobe_error": str(e)
        }


# ============================================================
# FRAME RATE
# ============================================================

def parse_fraction(value):
    """
    Convert FFprobe fraction such as 30000/1001 to float.
    """

    if not value:
        return 0.0

    try:
        if "/" in value:
            numerator, denominator = value.split("/")
            return float(numerator) / float(denominator)

        return float(value)

    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def parse_optional_int(value: int | float | str | None) -> Optional[int]:
    if value is None or not isinstance(value, (int, float, str)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_optional_float(value: int | float | str | None) -> Optional[float]:
    if value is None or not isinstance(value, (int, float, str)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ============================================================
# ACTUAL FRAME DECODING
# ============================================================

def decode_video(video_path):
    """
    Sequentially decode the entire video using OpenCV.

    Returns:
        decoded_frames
        first_decode_failure_frame
        last_successful_frame
        decode_error
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return {
            "decoded_frames": 0,
            "first_decode_failure_frame": None,
            "last_successful_frame": None,
            "decode_error": "Could not open video"
        }

    decoded_frames = 0
    first_decode_failure_frame = None
    last_successful_frame = None

    while True:

        ret, frame = cap.read()

        if not ret:

            # OpenCV does not distinguish EOF from
            # a decoding failure here.
            #
            # We record the frame position so it can
            # be compared with the expected frame count.

            current_position = int(
                cap.get(cv2.CAP_PROP_POS_FRAMES)
            )

            if current_position < 0:
                current_position = None

            first_decode_failure_frame = current_position

            break

        decoded_frames += 1
        last_successful_frame = decoded_frames - 1

    cap.release()

    return {
        "decoded_frames": decoded_frames,
        "first_decode_failure_frame": first_decode_failure_frame,
        "last_successful_frame": last_successful_frame,
        "decode_error": ""
    }


# ============================================================
# ANALYZE ONE VIDEO
# ============================================================

def analyze_video(video_path, index, total):

    print()
    print("=" * 80)
    print(f"[{index}/{total}] {video_path.name}")
    print("=" * 80)

    # --------------------------------------------------------
    # FFPROBE METADATA
    # --------------------------------------------------------

    stream, fmt = get_ffprobe_metadata(video_path)

    ffprobe_error = fmt.get("ffprobe_error", "")

    codec = stream.get("codec_name", "")
    codec_long = stream.get("codec_long_name", "")

    width = stream.get("width", 0)
    height = stream.get("height", 0)

    avg_fps = parse_fraction(
        stream.get("avg_frame_rate")
    )

    nominal_fps = parse_fraction(
        stream.get("r_frame_rate")
    )

    ffprobe_frames = parse_optional_int(stream.get("nb_frames"))
    stream_duration = parse_optional_float(stream.get("duration"))
    file_duration = parse_optional_float(fmt.get("duration"))

    file_size = video_path.stat().st_size

    # --------------------------------------------------------
    # OPENCV METADATA
    # --------------------------------------------------------

    cap = cv2.VideoCapture(str(video_path))

    opencv_opened = cap.isOpened()

    opencv_frames = None
    opencv_fps = None
    opencv_width = None
    opencv_height = None

    if opencv_opened:

        opencv_frames = parse_optional_int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        opencv_fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        opencv_width = parse_optional_int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        opencv_height = parse_optional_int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    # --------------------------------------------------------
    # ACTUAL DECODING
    # --------------------------------------------------------

    print("Decoding every frame...")

    decode_result = decode_video(video_path)

    decoded_frames = decode_result["decoded_frames"]

    # --------------------------------------------------------
    # DETERMINE FRAME DIFFERENCE
    # --------------------------------------------------------

    if ffprobe_frames is not None:

        frame_difference_ffprobe = (
            decoded_frames - ffprobe_frames
        )

    else:

        frame_difference_ffprobe = None

    if opencv_frames is not None:

        frame_difference_opencv = (
            decoded_frames - opencv_frames
        )

    else:

        frame_difference_opencv = None

    # --------------------------------------------------------
    # DETERMINE STATUS
    # --------------------------------------------------------

    status = "OK"

    if ffprobe_error:
        status = "FFPROBE_ERROR"

    elif not opencv_opened:
        status = "OPEN_ERROR"

    elif ffprobe_frames is not None:

        if decoded_frames < ffprobe_frames:
            status = "POSSIBLE_CORRUPTION"

        elif decoded_frames > ffprobe_frames:
            status = "FRAME_COUNT_MISMATCH"

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print(f"Resolution:          {width} x {height}")
    print(f"Codec:               {codec}")
    print(f"Average FPS:         {avg_fps:.6f}")
    print(f"Nominal FPS:         {nominal_fps:.6f}")

    print()
    print(f"FFprobe frames:      {ffprobe_frames}")
    print(f"OpenCV frames:       {opencv_frames}")
    print(f"Actually decoded:    {decoded_frames}")

    print()
    print(f"FFprobe difference:  {frame_difference_ffprobe}")
    print(f"OpenCV difference:   {frame_difference_opencv}")

    print()
    print(f"STATUS:              {status}")

    return {
        "filename": video_path.name,
        "full_path": str(video_path),

        "file_size_bytes": file_size,

        "width": width,
        "height": height,

        "codec": codec,
        "codec_long_name": codec_long,

        "ffprobe_avg_fps": avg_fps,
        "ffprobe_nominal_fps": nominal_fps,

        "ffprobe_frames": ffprobe_frames,
        "opencv_frames": opencv_frames,
        "decoded_frames": decoded_frames,

        "frame_difference_ffprobe":
            frame_difference_ffprobe,

        "frame_difference_opencv":
            frame_difference_opencv,

        "stream_duration_seconds":
            stream_duration,

        "file_duration_seconds":
            file_duration,

        "opencv_width":
            opencv_width,

        "opencv_height":
            opencv_height,

        "first_decode_failure_frame":
            decode_result["first_decode_failure_frame"],

        "last_successful_frame":
            decode_result["last_successful_frame"],

        "decode_error":
            decode_result["decode_error"],

        "ffprobe_error":
            ffprobe_error,

        "status": status,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("ROADESSY — VIDEO FRAME INTEGRITY CHECKER")
    print("=" * 80)

    if not VIDEO_FOLDER.exists():
        print()
        print(f"ERROR: Folder does not exist:")
        print(VIDEO_FOLDER)
        return

    videos = sorted(
        [
            p
            for p in VIDEO_FOLDER.iterdir()
            if p.is_file()
            and p.suffix in VIDEO_EXTENSIONS
        ]
    )

    if not videos:
        print()
        print("No video files found.")
        return

    print()
    print(f"Folder: {VIDEO_FOLDER}")
    print(f"Videos found: {len(videos)}")

    print()
    print("This process will:")
    print("  1. Inspect FFprobe metadata")
    print("  2. Inspect OpenCV metadata")
    print("  3. Decode EVERY frame")
    print("  4. Compare reported vs decoded frame counts")
    print("  5. Flag possible corruption")
    print()

    input("Press ENTER to begin...")

    results = []

    start_time = datetime.now()

    for index, video_path in enumerate(videos, start=1):

        result = analyze_video(
            video_path,
            index,
            len(videos)
        )

        results.append(result)

    # ========================================================
    # WRITE CSV
    # ========================================================

    fieldnames = [
        "filename",
        "full_path",
        "file_size_bytes",

        "width",
        "height",

        "codec",
        "codec_long_name",

        "ffprobe_avg_fps",
        "ffprobe_nominal_fps",

        "ffprobe_frames",
        "opencv_frames",
        "decoded_frames",

        "frame_difference_ffprobe",
        "frame_difference_opencv",

        "stream_duration_seconds",
        "file_duration_seconds",

        "opencv_width",
        "opencv_height",

        "first_decode_failure_frame",
        "last_successful_frame",

        "decode_error",
        "ffprobe_error",

        "status",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(results)

    # ========================================================
    # SUMMARY
    # ========================================================

    elapsed = datetime.now() - start_time

    total_ffprobe_frames = sum(
        r["ffprobe_frames"] or 0
        for r in results
    )

    total_decoded_frames = sum(
        r["decoded_frames"] or 0
        for r in results
    )

    problematic = [
        r for r in results
        if r["status"] != "OK"
    ]

    print()
    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(f"Videos checked:       {len(results):,}")
    print(f"FFprobe frames:       {total_ffprobe_frames:,}")
    print(f"Decoded frames:       {total_decoded_frames:,}")

    print()
    print(f"Problematic videos:   {len(problematic):,}")

    if problematic:

        print()
        print("PROBLEMATIC FILES")
        print("-" * 80)

        for r in problematic:

            print(
                f"{r['filename']}: "
                f"{r['status']} | "
                f"reported={r['ffprobe_frames']} | "
                f"decoded={r['decoded_frames']}"
            )

    else:

        print()
        print("ALL VIDEOS PASSED THE FRAME INTEGRITY CHECK.")

    print()
    print(f"Elapsed time:         {elapsed}")
    print(f"CSV report:           {OUTPUT_CSV}")

    print("=" * 80)


if __name__ == "__main__":
    main()