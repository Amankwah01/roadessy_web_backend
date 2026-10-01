# app/services/gps_service.py
import json
import os
import re
import subprocess
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd


class UniversalGPSService:
    """Loads GPS from external files or video metadata into time/lat/lon records."""

    #: Every source the loader accepts, for the UI to offer and for the API to
    #: validate against. "overlay" reads the position and time burned into the
    #: video itself, which is a source in its own right and not only a clock:
    #: the camera's GPS log is clean, while the vibration logger's holds a stale
    #: fix for most of a trip and then jumps kilometres.
    GPS_SOURCES = ("auto", "overlay", "video", "csv", "vib_csv", "gpx", "srt", "json")

    @staticmethod
    def detect_gps_source(
        video_path: Optional[str] = None,
        external_file: Optional[str] = None,
        vibration_csv: Optional[str] = None,
    ) -> str:
        if external_file:
            suffix = Path(external_file).suffix.lower()
            if suffix == ".gpx":
                return "gpx"
            if suffix == ".csv":
                return "csv"
            if suffix == ".srt":
                return "srt"
            if suffix == ".json":
                return "json"
            raise ValueError(f"Unsupported external GPS format: {suffix}")

        if vibration_csv:
            return "vib_csv"

        if video_path:
            return "video"

        raise ValueError("No GPS source provided")

    @classmethod
    def load_gps_universal(
        cls,
        video_path: Optional[str] = None,
        external_file: Optional[str] = None,
        source: Optional[str] = None,
        vibration_csv: Optional[str] = None,
    ) -> pd.DataFrame:
        gps_source = source if source and source != "auto" else cls.detect_gps_source(video_path, external_file, vibration_csv)

        if gps_source not in cls.GPS_SOURCES:
            raise ValueError(
                f"Unsupported GPS source: {gps_source}. "
                f"Expected one of {', '.join(cls.GPS_SOURCES)}."
            )
        if gps_source == "overlay":
            if not video_path:
                raise ValueError("The burned-in overlay can only be read from a video")
            return cls.load_overlay_gps(video_path)
        if gps_source == "video":
            if not video_path:
                raise ValueError("Video GPS extraction requires a video path")
            return cls.extract_gps_from_video(video_path)
        if gps_source in ("csv", "vib_csv"):
            return cls.load_gps_csv(
                cls._required_external_file(external_file or vibration_csv),
                prefer_recintvl=gps_source == "vib_csv",
            )
        if gps_source == "gpx":
            return cls.load_gpx(cls._required_external_file(external_file))
        if gps_source == "srt":
            return cls.load_srt_gps(cls._required_external_file(external_file))
        if gps_source == "json":
            return cls.load_gps_json(cls._required_external_file(external_file))

        raise ValueError(f"Unsupported GPS source: {gps_source}")

    @staticmethod
    def load_overlay_gps(video_path: str, stride: int = 10) -> pd.DataFrame:
        """The track as the video's own burned-in telemetry recorded it.

        Offered as a source in its own right, not only as a clock: it is the one
        position record that came from the same camera as the footage, so it
        cannot disagree with the video about where the vehicle was the way a
        separate logger can. It is read from every frame's time and a stride of
        positions, which costs about two seconds per position read and is
        reported rather than hidden.
        """
        if not os.path.exists(video_path):
            raise ValueError(f"Video not found for overlay GPS: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Video could not be opened for overlay GPS: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        def frame_at(index: int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            return frame if ok else None

        try:
            from app.services.vision.overlay import read_overlay_track

            track = read_overlay_track(frame_at, total_frames, stride=stride)
        finally:
            cap.release()

        if not track:
            raise ValueError(
                "The burned-in telemetry could not be read from this video, so it "
                "cannot serve as the GPS source. The overlay has to be visible on "
                "the frame; a video without it is not an error in the video."
            )
        return pd.DataFrame(track)

    @staticmethod
    def _required_external_file(external_file: Optional[str]) -> str:
        if not external_file:
            raise ValueError("External GPS source selected without an external file")
        return external_file

    @staticmethod
    def extract_gps_from_video(video_path: str) -> pd.DataFrame:
        """
        Extract GPS data from video metadata using exiftool.
        Returns a DataFrame with time, lat, lon columns.
        """
        try:
            bundled_exiftool = Path(__file__).resolve().parents[3] / "tools" / "exiftool" / "exiftool.exe"
            possible_paths = [str(bundled_exiftool)] if bundled_exiftool.exists() else []
            if os.name == "nt":
                possible_paths.extend([
                    r"C:\ExifTool\exiftool.exe",
                    r"C:\Program Files\ExifTool\exiftool.exe",
                ])
            possible_paths.append("exiftool")

            exiftool_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    exiftool_path = path
                    break

            if not exiftool_path:
                raise RuntimeError(
                    "exiftool not found. Install it or place exiftool.exe in tools/exiftool/"
                )

            cmd = [
                exiftool_path,
                "-json",
                "-gps:all",
                "-api",
                "LargeFileSupport=1",
                video_path,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )

            if result.returncode != 0:
                raise RuntimeError(f"exiftool failed: {result.stderr}")

            metadata = json.loads(result.stdout)
            if not metadata:
                raise ValueError("No metadata found in video")

            gps_points = []
            video_metadata = metadata[0]

            lat = video_metadata.get("GPSLatitude")
            lon = video_metadata.get("GPSLongitude")
            timestamp = video_metadata.get("GPSDateTime") or video_metadata.get("CreateDate") or video_metadata.get("ModifyDate")

            if lat is None or lon is None:
                raise ValueError("No GPS coordinates found in video metadata")

            if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                gps_points.append({
                    "time": pd.to_datetime(timestamp).timestamp() if timestamp else 0.0,
                    "lat": float(lat),
                    "lon": float(lon),
                })
            else:
                def _convert_to_decimal(coord, ref):
                    if isinstance(coord, list) and len(coord) == 3:
                        return float(coord[0]) + float(coord[1]) / 60.0 + float(coord[2]) / 3600.0
                    return float(coord) # type: ignore

                lat_decimal = _convert_to_decimal(lat, video_metadata.get("GPSLatitudeRef", "N"))
                lon_decimal = _convert_to_decimal(lon, video_metadata.get("GPSLongitudeRef", "E"))

                if video_metadata.get("GPSLatitudeRef", "N") == "S":
                    lat_decimal *= -1
                if video_metadata.get("GPSLongitudeRef", "E") == "W":
                    lon_decimal *= -1

                gps_points.append({
                    "time": pd.to_datetime(timestamp).timestamp() if timestamp else 0.0,
                    "lat": lat_decimal,
                    "lon": lon_decimal,
                })

            return UniversalGPSService._normalize_gps_dataframe(pd.DataFrame(gps_points))

        except Exception as exc:
            raise RuntimeError(f"Failed to extract GPS from video: {str(exc)}") from exc

    @staticmethod
    def load_gpx(path: str) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        tree = ET.parse(path)
        root = tree.getroot()

        for elem in root.iter():
            if "}" in elem.tag:
                elem.tag = elem.tag.split("}", 1)[1]

        for point in root.findall(".//trkpt"):
            time_node = point.find("time")
            rows.append(
                {
                    "time": pd.to_datetime(time_node.text).timestamp() if time_node is not None and time_node.text else len(rows),
                    "lat": point.attrib.get("lat"),
                    "lon": point.attrib.get("lon"),
                }
            )

        return UniversalGPSService._normalize_gps_dataframe(pd.DataFrame(rows))

    #: Columns these logger exports use for the time axis, tried in this order
    #: only when a file has more than one column that *looks* like a clock. The
    #: order is a tie-break between equally readable columns, not a claim about
    #: which one is right — see `_score_time_column`.
    _TIME_COLUMN_NAMES = (
        "timestamp",
        "time",
        "elapsed",
        "elapsed_time",
        "seconds",
        "second",
        "global_second",
        "recintvl",
        "gpstime",
        "gps_time",
        "datetime",
        "utc",
    )

    @staticmethod
    def _elapsed_clock_to_seconds(series: pd.Series) -> pd.Series:
        """Convert `H:MM:SS(.fff)` / `MM:SS` elapsed strings to seconds.

        Some exports write elapsed time as a clock string (`0:00:00`, `0:37:21`)
        rather than a number. `to_numeric` yields nothing for those, and handing
        them to `to_datetime` is worse than useless: dateutil reads `0:00:00` as
        midnight on an arbitrary date, so the column comes back looking valid
        while its absolute offset is meaningless. It only survives because the
        axis is rebased afterwards.
        """
        text = series.astype(str).str.strip()
        parts = text.str.split(":", n=2, expand=True)
        out = pd.Series(np.nan, index=series.index, dtype="float64")
        if parts.shape[1] < 2:
            return out
        hours, minutes, seconds = (pd.to_numeric(parts[i], errors="coerce") for i in range(parts.shape[1]))
        out = hours.fillna(0.0) * 3600.0 + minutes.fillna(0.0) * 60.0
        if parts.shape[1] == 3:
            out = out + seconds.fillna(0.0)
        matches = text.str.match(r"^\d{1,4}(?::\d{1,2}){1,2}(?:\.\d+)?$").fillna(False)
        return out.where(matches)

    @staticmethod
    def _epoch_to_seconds(series: pd.Series) -> pd.Series:
        """Convert a datetime-like column to seconds, NaN where unparseable."""
        try:
            with warnings.catch_warnings():
                # Reaching here means the column is neither numbers nor an
                # elapsed clock, so the format genuinely cannot be inferred.
                # That is a probe, not a failure: the caller only keeps the
                # result if enough rows parsed, and the warning would otherwise
                # fire on every load of these logger exports.
                warnings.simplefilter("ignore", UserWarning)
                parsed = pd.to_datetime(series, errors="coerce", utc=True)
        except Exception:
            return pd.Series(np.nan, index=series.index, dtype="float64")
        if not parsed.notna().any():
            return pd.Series(np.nan, index=series.index, dtype="float64")
        naive = parsed.dt.tz_localize(None) if parsed.dt.tz is not None else parsed
        scale = 1e9 if str(naive.dtype).startswith("datetime64[ns") else 1e6
        return pd.Series(naive.astype("int64").to_numpy() / scale, index=series.index)

    @staticmethod
    def _score_time_column(series: Optional[pd.Series]) -> Tuple[Optional[pd.Series], int]:
        """Read one column as a time axis, preferring the most literal reading.

        These exports put plain seconds, `H:MM:SS` elapsed clocks and wall-clock
        datetimes into columns whose names say otherwise, and two of them can be
        present in the same file. So the column is chosen by what its *values*
        are, with a higher score for a reading that needs no interpretation:
        plain seconds (3) beats an elapsed clock (2) beats a datetime (1).
        """
        if series is None or len(series) == 0:
            return None, 0

        numeric = pd.to_numeric(series, errors="coerce")
        if float(numeric.notna().mean()) >= 0.9:
            return numeric, 3

        clock = UniversalGPSService._elapsed_clock_to_seconds(series)
        if float(clock.notna().mean()) >= 0.9:
            return clock, 2

        epoch = UniversalGPSService._epoch_to_seconds(series)
        if float(epoch.notna().mean()) >= 0.9:
            return epoch, 1

        return None, 0

    @classmethod
    def _resolve_time_axis(cls, df: pd.DataFrame, prefer_recintvl: bool) -> Tuple[Optional[pd.Series], str]:
        """Find the column that actually holds seconds, and say how it was read.

        Returns `(None, "fallback")` when no column in the file can be read as a
        clock, which is deliberately different from "read as seconds": a caller
        that had to invent a 1 Hz axis should not present it as measured time.
        """
        # The RecIntvl column on these vibration CSVs is the *device sampling
        # interval* (1.0 s for the KNUST dataset), and only that CSV type opts
        # into accumulating it. Integrating it is a different question from
        # reading a time column, so it is handled before the value scan.
        if prefer_recintvl:
            recintvl_column = next(
                (column for column in df.columns if column.strip().lower() == "recintvl"),
                None,
            )
            if recintvl_column is not None:
                intervals = pd.to_numeric(df[recintvl_column], errors="coerce").dropna()
                if len(intervals) == len(df) and len(df) > 0:
                    return pd.Series(
                        intervals.cumsum().to_numpy() - intervals.iloc[0],
                        index=df.index,
                    ), "recintvl"

        best: Optional[pd.Series] = None
        best_score = 0
        best_name = ""
        seen: Set[str] = set()
        # RecIntvl first when asked for, otherwise the documented order.
        ordered = list(df.columns)
        if prefer_recintvl:
            ordered.sort(key=lambda column: 0 if column.strip().lower() == "recintvl" else 1)

        for column in ordered:
            lowered = column.strip().lower()
            if lowered in seen or lowered not in cls._TIME_COLUMN_NAMES:
                continue
            seen.add(lowered)
            candidate, score = cls._score_time_column(df[column])
            if score > best_score:
                best, best_score, best_name = candidate, score, column
        return best, best_name or "fallback"

    @staticmethod
    def load_gps_csv(path: str, prefer_recintvl: bool = False) -> pd.DataFrame:
        df = pd.read_csv(path)
        df.columns = [str(column).strip() for column in df.columns]
        lower_columns = {column.lower(): column for column in df.columns}
        rename_map = {}
        for canonical, candidates in {
            "lat": ["lat", "latitude"],
            "lon": ["lon", "lng", "longitude"],
        }.items():
            for candidate in candidates:
                if candidate in lower_columns:
                    rename_map[lower_columns[candidate]] = canonical
                    break
        renamed = df.rename(columns=rename_map)
        time_series, _ = UniversalGPSService._resolve_time_axis(df, prefer_recintvl)
        if time_series is None:
            # Fallback: assume 1 Hz
            time_series = pd.Series(np.arange(len(renamed), dtype=np.float64), index=renamed.index)
        renamed["time"] = time_series
        return UniversalGPSService._normalize_gps_dataframe(renamed)

    @staticmethod
    def load_srt_gps(path: str) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        current_time: Optional[float] = None

        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                timestamp_match = re.search(r"(\d+):(\d+):(\d+)[,.](\d+)", line)
                if timestamp_match:
                    current_time = (
                        int(timestamp_match.group(1)) * 3600
                        + int(timestamp_match.group(2)) * 60
                        + int(timestamp_match.group(3))
                        + int(timestamp_match.group(4)) / 1000.0
                    )

                lat = re.search(r"(?:Lat|Latitude)[:=]?\s*([-0-9.]+)", line, re.IGNORECASE)
                lon = re.search(r"(?:Lon|Lng|Longitude)[:=]?\s*([-0-9.]+)", line, re.IGNORECASE)
                if lat and lon and current_time is not None:
                    rows.append({"time": current_time, "lat": float(lat.group(1)), "lon": float(lon.group(1))})

        return UniversalGPSService._normalize_gps_dataframe(pd.DataFrame(rows))

    @staticmethod
    def load_gps_json(path: str) -> pd.DataFrame:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if isinstance(payload, dict):
            payload = payload.get("gps") or payload.get("points") or payload.get("features") or []
        return UniversalGPSService._normalize_gps_dataframe(pd.DataFrame(payload))

    @staticmethod
    def _normalize_gps_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValueError("GPS source did not contain any points")
        missing = {"time", "lat", "lon"} - set(df.columns)
        if missing:
            raise ValueError(f"GPS source missing required columns: {', '.join(sorted(missing))}")

        normalized = df[["time", "lat", "lon"]].copy()
        normalized["time"] = pd.to_numeric(normalized["time"], errors="coerce")
        normalized["lat"] = pd.to_numeric(normalized["lat"], errors="coerce")
        normalized["lon"] = pd.to_numeric(normalized["lon"], errors="coerce")
        normalized = normalized.dropna(subset=["time", "lat", "lon"]).sort_values("time")
        normalized = normalized.drop_duplicates(subset=["time"], keep="first").reset_index(drop=True)

        if len(normalized) == 0:
            raise ValueError("GPS source did not contain valid time/lat/lon rows")

        # Rebase onto the first fix. The loaders disagree about origin — the
        # logger counters are absolute session seconds that can start in the
        # thousands, GPX and video metadata are epoch seconds, and only the
        # SRT and some CSV columns are already relative — while every consumer
        # measures video time from frame zero. Left unrebased, a logger that
        # started at 8241 s puts its whole track outside the video's timeline, so
        # every detection resolves to the first fix and the run reports the
        # vehicle as stationary for the entire survey.
        base = float(normalized["time"].iloc[0])
        if base != 0.0:
            normalized["time"] = normalized["time"] - base
        return normalized

    @staticmethod
    def dataframe_to_spatial_points(df: pd.DataFrame) -> List[Dict[str, Any]]:
        return [
            {
                "time": float(row.time),  # type: ignore[arg-type]
                "timestamp": str(row.time),
                "lat": float(row.lat),  # type: ignore[arg-type]
                "lon": float(row.lon),  # type: ignore[arg-type]
                "elevation": 0.0,
            }
            for row in df.itertuples(index=False)
        ]

    @staticmethod
    def extract_frames(video_path: str) -> List[Dict[str, Any]]:
        cap = cv2.VideoCapture(video_path)
        frames: List[Dict[str, Any]] = []
        frame_index = 0
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        previous_time = -1.0

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break
                position_time = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                frame_time = (
                    position_time
                    if np.isfinite(position_time) and position_time > previous_time
                    else frame_index / fps
                )
                previous_time = frame_time
                frames.append(
                    {
                        "frame": frame_index,
                        "time": frame_time,
                        "frame_data": frame,
                    }
                )
                frame_index += 1
        finally:
            cap.release()

        return frames

    @staticmethod
    def sync_frames_to_gps(
        frames: List[Dict[str, Any]], gps: pd.DataFrame, video_time_offset_s: float = 0.0
    ) -> List[Dict[str, float]]:
        """Interpolate GPS at video timestamps using an explicit clock offset.

        Frames outside the GPS time range are retained but marked ``gps_in_range``
        rather than silently appearing to have a valid endpoint position.
        """
        gps_t = gps["time"].to_numpy(dtype=np.float64)
        if len(gps_t) == 0:
            return []
        gps_t_relative = gps_t - gps_t[0]
        lat = gps["lat"].to_numpy(dtype=np.float64)
        lon = gps["lon"].to_numpy(dtype=np.float64)

        output = []
        for item in frames:
            frame_time = float(item["time"]) + video_time_offset_s
            output.append(
                {
                    "frame": int(item["frame"]),
                    "time": frame_time,
                    "lat": float(np.interp(frame_time, gps_t_relative, lat)),
                    "lon": float(np.interp(frame_time, gps_t_relative, lon)),
                    "gps_in_range": bool(gps_t_relative[0] <= frame_time <= gps_t_relative[-1]),
                }
            )
        return output

    @staticmethod
    def frame_gps_sync_diagnostics(frame_gps_map: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(frame_gps_map)
        matched = sum(1 for item in frame_gps_map if item.get("gps_in_range"))
        return {
            "frame_count": total,
            "frames_with_gps": matched,
            "coverage_ratio": round(matched / total, 4) if total else 0.0,
            "status": "synchronized" if total and matched == total else "partial_or_unverified",
        }
