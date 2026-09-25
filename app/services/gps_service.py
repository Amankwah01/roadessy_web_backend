# app/services/gps_service.py
import json
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd


class UniversalGPSService:
    """Loads GPS from external files or video metadata into time/lat/lon records."""

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

    @staticmethod
    def load_gps_csv(path: str, prefer_recintvl: bool = False) -> pd.DataFrame:
        df = pd.read_csv(path)
        df.columns = [str(column).strip() for column in df.columns]
        lower_columns = {column.lower(): column for column in df.columns}
        rename_map = {}
        # Prefer the parsed Timestamp column for the time axis. The RecIntvl
        # column on these logger CSVs is the *device sampling interval*
        # (1.0 s for the KNUST dataset), but the actual acquisition
        # timestamps resolve to 1 ms spacing. Using RecIntvl as the time
        # axis shrinks distances by ~1000x and inflates IRI by the
        # same factor.
        if prefer_recintvl:
            time_candidates = ["recintvl", "time", "timestamp", "gpstime", "gps_time", "global_second"]
        else:
            time_candidates = ["timestamp", "time", "gpstime", "gps_time", "global_second"]
        for canonical, candidates in {
            "time": time_candidates,
            "lat": ["lat", "latitude"],
            "lon": ["lon", "lng", "longitude"],
        }.items():
            for candidate in candidates:
                if candidate in lower_columns:
                    rename_map[lower_columns[candidate]] = canonical
                    break
        renamed = df.rename(columns=rename_map)
        recintvl_column = next(
            (column for column in df.columns if column.strip().lower() == "recintvl"),
            None,
        )
        time_resolved = False
        if recintvl_column and prefer_recintvl:
            intervals = pd.to_numeric(df[recintvl_column], errors="coerce").fillna(0.0)
            if len(intervals) > 0:
                renamed["time"] = intervals.cumsum() - intervals.iloc[0]
            else:
                renamed["time"] = 0.0
            time_resolved = True
        elif "time" in renamed.columns:
            time_series = pd.to_numeric(renamed["time"], errors="coerce")
            if not time_series.notna().any():
                try:
                    parsed = pd.to_datetime(renamed["time"], errors="coerce", utc=True)
                    if parsed.notna().any():
                        # Handle timezone-aware timestamps: get seconds since epoch correctly
                        if parsed.dt.tz is not None:
                            parsed = parsed.dt.tz_convert('UTC').dt.tz_localize(None)
                        # Get seconds since epoch correctly (handle us/ns resolution)
                        arr = parsed.array
                        ns = arr._ndarray.view('int64')
                        # Detect resolution: datetime64[ns] gives ns, datetime64[us] gives us
                        if parsed.dtype == 'datetime64[ns, UTC]' or parsed.dtype == 'datetime64[ns]':
                            time_series = ns / 1e9
                        else:
                            # datetime64[us] -> microseconds since epoch
                            time_series = ns / 1e6
                        # Convert to relative time
                        time_series = time_series - time_series[0]
                except Exception:
                    pass
            renamed["time"] = time_series
            time_resolved = True
        if not time_resolved:
            # Fallback: assume 1 Hz
            renamed["time"] = np.arange(len(renamed), dtype=np.float64)
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
