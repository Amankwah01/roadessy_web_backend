import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pyproj import Transformer
import shapely
from shapely.geometry import LineString, Point, mapping
from shapely.ops import linemerge, transform, nearest_points
from shapely.wkt import loads as wkt_loads
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.logger import monitor, spatial_logger


class CoordinateSystem(str, Enum):
    WGS84 = "WGS84"


@dataclass
class RoadSegment:
    segment_id: str
    centerline_coords: List[Tuple[float, float]]
    road_name: str
    road_width_meters: float

    def __post_init__(self) -> None:
        if len(self.centerline_coords) < 2:
            raise ValueError("A road segment requires at least two centerline coordinates")
        if self.road_width_meters <= 0:
            raise ValueError("Road width must be greater than zero")


@dataclass
class MapMatchedPoint:
    snapped_segment_id: str
    is_within_buffer: bool
    distance_to_road_meters: float
    snapped_point: Tuple[float, float]
    original_point: Optional[Tuple[float, float]] = None
    chainage_m: Optional[float] = None

    @property
    def segment_id(self) -> str:
        return self.snapped_segment_id

    @property
    def distance_meters(self) -> float:
        return self.distance_to_road_meters


class RoadNetworkBuffer:
    def __init__(self, road_width_default: float = 7.0, coordinate_system: CoordinateSystem = CoordinateSystem.WGS84):
        if road_width_default <= 0:
            raise ValueError("Road width must be greater than zero")
        self.road_width_default = float(road_width_default)
        self.coordinate_system = coordinate_system
        self._segments: Dict[str, RoadSegment] = {}

    def add_road_segment(
        self,
        segment_id: str,
        centerline: Sequence[Tuple[float, float]],
        road_name: str,
        width_meters: float,
    ) -> RoadSegment:
        centerline_coords: List[Tuple[float, float]] = [
            (float(point[0]), float(point[1])) for point in centerline
        ]
        segment = RoadSegment(segment_id, centerline_coords, road_name, width_meters)
        self._segments[segment.segment_id] = segment
        return segment

    def get_road_info(self, segment_id: str) -> Optional[Dict[str, Any]]:
        segment = self._segments.get(segment_id)
        if segment is None:
            return None
        longitudes = [point[0] for point in segment.centerline_coords]
        latitudes = [point[1] for point in segment.centerline_coords]
        return {
            "segment_id": segment.segment_id,
            "road_name": segment.road_name,
            "buffer_distance_meters": segment.road_width_meters / 2.0,
            "bounds": (min(longitudes), min(latitudes), max(longitudes), max(latitudes)),
        }

    def get_all_segments(self) -> Dict[str, Dict[str, Any]]:
        return {
            segment_id: {
                "road_name": segment.road_name,
                "width_meters": segment.road_width_meters,
                "centerline": list(segment.centerline_coords),
            }
            for segment_id, segment in self._segments.items()
        }

    def snap_point_to_road(
        self,
        point: Tuple[float, float],
        search_radius_meters: Optional[float] = None,
    ) -> Optional[MapMatchedPoint]:
        if len(point) != 2:
            return None
        lon, lat = float(point[0]), float(point[1])
        transformers = SpatialService._metric_transformer([(lon, lat)])
        if transformers is None:
            return None
        forward, inverse = transformers

        metric_point = Point(*forward.transform(lon, lat))
        best = None
        search_radius = float(search_radius_meters) if search_radius_meters is not None else self.road_width_default

        for segment in self._segments.values():
            lons, lats = zip(*segment.centerline_coords)
            xs, ys = forward.transform(lons, lats)
            metric_line = LineString(zip(xs, ys))

            snapped_metric, _ = nearest_points(metric_line, metric_point)
            distance_m = float(metric_point.distance(snapped_metric))

            if distance_m > search_radius:
                continue

            snapped_lon, snapped_lat = inverse.transform(snapped_metric.x, snapped_metric.y)
            candidate = MapMatchedPoint(
                snapped_segment_id=segment.segment_id,
                is_within_buffer=distance_m <= segment.road_width_meters / 2.0,
                distance_to_road_meters=distance_m,
                snapped_point=(float(snapped_lon), float(snapped_lat)),
                original_point=(lon, lat),
                chainage_m=float(metric_line.project(snapped_metric)),
            )
            if best is None or candidate.distance_to_road_meters < best.distance_to_road_meters:
                best = candidate

        return best

    def batch_snap_points(self, points: Sequence[Tuple[float, float]]) -> List[Optional[MapMatchedPoint]]:
        return [self.snap_point_to_road(point) for point in points]

    def create_road_buffer_polygon(self, segment_id: str) -> Optional[Dict[str, Any]]:
        segment = self._segments.get(segment_id)
        if segment is None:
            return None
        transformers = SpatialService._metric_transformer(segment.centerline_coords)
        if transformers is None:
            return None
        forward, inverse = transformers

        lons, lats = zip(*segment.centerline_coords)
        xs, ys = forward.transform(lons, lats)
        metric_line = LineString(zip(xs, ys))

        buffer_metric = metric_line.buffer(segment.road_width_meters / 2.0)
        buffer_wgs84 = transform(inverse.transform, buffer_metric)
        return {
            "segment_id": segment_id,
            "buffer_geometry": mapping(buffer_wgs84),
        }


class MapMatchingIntegration:
    def __init__(self, road_data: Dict[str, Any]):
        self.network = RoadNetworkBuffer(road_width_default=7.0)
        for segment_data in road_data.get("segments", []):
            self.network.add_road_segment(
                segment_id=str(segment_data["id"]),
                centerline=segment_data.get("centerline", []),
                road_name=segment_data.get("name", ""),
                width_meters=float(segment_data.get("width", self.network.road_width_default)),
            )

    @staticmethod
    def _point_coordinates(point: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        lon = point.get("lon", point.get("longitude"))
        lat = point.get("lat", point.get("latitude"))
        if lon is None or lat is None:
            return None
        return float(lon), float(lat)

    def match_inertial_gps_data(self, points: Sequence[Dict[str, Any]]) -> List[Optional[MapMatchedPoint]]:
        return [self.network.snap_point_to_road(coordinates) if (coordinates := self._point_coordinates(point)) else None for point in points]

    def match_vision_detections(self, points: Sequence[Dict[str, Any]]) -> List[Optional[MapMatchedPoint]]:
        return self.match_inertial_gps_data(points)

    def get_matching_summary(self, matches: Sequence[Optional[MapMatchedPoint]]) -> Dict[str, Any]:
        matched = [match for match in matches if match is not None]
        distances = [match.distance_to_road_meters for match in matched]
        within_buffer = sum(match.is_within_buffer for match in matched)
        return {
            "total_points": len(matches),
            "matched_points": len(matched),
            "within_buffer": within_buffer,
            "success_rate": (len(matched) / len(matches) * 100.0) if matches else 0.0,
            "avg_distance_to_road_meters": sum(distances) / len(distances) if distances else 0.0,
            "min_distance_to_road_meters": min(distances) if distances else 0.0,
            "max_distance_to_road_meters": max(distances) if distances else 0.0,
        }

    def create_road_buffer_polygon(self, segment_id: str) -> Optional[Dict[str, Any]]:
        return self.network.create_road_buffer_polygon(segment_id)


class SpatialService:
    """Spatial map matching engine against road network data."""

    DEFAULT_MAX_DISTANCE_M = 100.0

    #: The widest stretch of run-relative chainage a station is interpolated
    #: across. Beyond it the matched points either side are describing different
    #: places — the run left the road, or the match jumped to another road — and
    #: a value read across that gap would be an invention. A detection in such a
    #: gap is reported as unmatched instead.
    MAX_STATION_INTERPOLATION_GAP_M = 250.0

    @classmethod
    def station_profile(
        cls,
        gps_df: pd.DataFrame,
        map_match_result: Optional[Dict[str, Any]],
    ) -> Optional[List[Tuple[float, float]]]:
        """Pairs of (chainage along this run, station along the road).

        A detection's own chainage is distance travelled since the survey
        started, so it is zero at wherever the vehicle happened to be put on the
        road. Map matching projects each GPS fix onto the road centerline and
        reports its position measured from the start of *that road*, which is a
        fixed frame: the same pothrup is at the same station on every survey of
        it, whether the run started at the beginning of the road or in the
        middle. This pairs the two so one can be read off the other.

        Returns None when the run was not matched, because a station cannot be
        had without a road to measure along.
        """
        if not map_match_result:
            return None
        snapped = map_match_result.get("snapped_coords") or []
        if len(snapped) < 2:
            return None

        points = cls._valid_gps_points(gps_df)
        # The snapped points are index-aligned with the same sequence, so the
        # run-relative distance for each is the cumulative distance over it.
        if len(points) != len(snapped):
            return None
        relative = [entry["chainage_m"] for entry in cls._raw_chainage(points)]

        pairs = [
            (float(distance), float(point["chainage_m"]))
            for distance, point in zip(relative, snapped)
            if point.get("matched")
        ]
        if len(pairs) < 2:
            return None
        pairs.sort(key=lambda pair: pair[0])
        return cls._rescale_to_gps_metric(pairs, relative)

    @classmethod
    def _rescale_to_gps_metric(
        cls,
        pairs: List[Tuple[float, float]],
        relative: Sequence[float],
    ) -> List[Tuple[float, float]]:
        """Put the stations on the same distance scale as the chainage.

        Two definitions of a metre are in play here. A detection's chainage is a
        Haversine sum on a sphere of radius 6371 km, which is within about
        0.001% of true ground distance at these latitudes. The station is a
        projection onto the road centerline carried out in UTM, and UTM's
        k0 factor plus the projection's own distortion makes it about 0.55%
        short of the same ground. Left alone, the two drift apart by roughly
        30 m over a 5 km run, so a station would slowly stop agreeing with the
        chainage it is supposed to sit beside.

        The fit rescales the stations so that the measured run and the matched
        road agree about how long the stretch between them is. The scale is
        applied from the road's own beginning rather than from wherever this
        survey joined it: the projection's scale error runs the whole length of
        the road, so anchoring the correction at the road's origin is the only
        choice that does not privilege where this particular run happened to
        start, and it keeps stations from two surveys of the same road directly
        comparable.
        """
        matched_distances = [distance for distance, _ in pairs]
        if matched_distances[-1] <= matched_distances[0]:
            return pairs

        gps_span = matched_distances[-1] - matched_distances[0]
        station_span = pairs[-1][1] - pairs[0][1]
        if station_span <= 0:
            return pairs
        scale = gps_span / station_span
        if abs(scale - 1.0) < 1e-6:
            return pairs

        rescaled = [(distance, station * scale) for distance, station in pairs]
        spatial_logger.info(
            "Road stations rescaled onto the GPS distance scale",
            scale=round(scale, 6),
            gps_span_m=round(gps_span, 2),
            projected_span_m=round(station_span, 2),
        )
        return rescaled

    @classmethod
    def station_for_chainage(
        cls,
        profile: Sequence[Tuple[float, float]],
        chainage_m: Optional[float],
    ) -> Optional[float]:
        """Read a station off the profile, or None if it cannot be read.

        Outside the range the match actually covers the answer is None rather
        than the nearest station. Holding a defect at the station of the last
        matched point would place it at the end of the road, which is a claim
        about where it is, not a smaller version of one.
        """
        if chainage_m is None or not profile:
            return None
        chainage = float(chainage_m)

        if chainage < profile[0][0] or chainage > profile[-1][0]:
            return None

        for (d0, s0), (d1, s1) in zip(profile[:-1], profile[1:]):
            if d0 <= chainage <= d1:
                # A hole in the matched points, not a missing interpolation.
                if d1 - d0 > cls.MAX_STATION_INTERPOLATION_GAP_M:
                    return None
                if d1 == d0:
                    return s0
                return s0 + (chainage - d0) * (s1 - s0) / (d1 - d0)
        return None

    @classmethod
    def apply_absolute_stations(
        cls,
        detections: Sequence[Dict[str, Any]],
        gps_df: pd.DataFrame,
        map_match_result: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Give every detection its station along the matched road.

        `chainage_m` is deliberately left alone. The 50 m PCI table and the IRI
        segments are both cut from it, and they start at the beginning of the
        run; replacing it with a road station would silently move every section
        boundary. The station is added beside it, with the offset between the
        two, so a reader can see that a defect at chainage 120 m is at station
        4180 m without either number having been overwritten.
        """
        profile = cls.station_profile(gps_df, map_match_result)
        road_id = (map_match_result or {}).get("matched_road_id")

        enriched: List[Dict[str, Any]] = []
        for detection in detections:
            record = dict(detection)
            station = cls.station_for_chainage(profile, detection.get("chainage_m")) if profile else None
            record["station_m"] = round(station, 2) if station is not None else None
            record["station_road_id"] = road_id if station is not None else None
            record["station_source"] = (
                "map_matched" if station is not None
                else ("unmatched_road" if map_match_result else "no_road_matched")
            )
            if station is not None and detection.get("chainage_m") is not None:
                record["chainage_offset_m"] = round(
                    station - float(detection["chainage_m"]), 2
                )
            else:
                record["chainage_offset_m"] = None
            enriched.append(record)
        return enriched

    @staticmethod
    def compute_linear_chainage(coords_df: Union[pd.DataFrame, List[Dict[str, float]]]) -> Union[pd.DataFrame, List[float]]:
        if isinstance(coords_df, list):
            coords_df = pd.DataFrame(coords_df)
            return_list = True
        else:
            return_list = False

        if coords_df.empty or "lat" not in coords_df.columns or "lon" not in coords_df.columns:
            return coords_df if not return_list else []

        lats = pd.to_numeric(coords_df["lat"], errors="coerce").to_numpy(dtype=float)
        lons = pd.to_numeric(coords_df["lon"], errors="coerce").to_numpy(dtype=float)

        if len(lats) == 0:
            return coords_df if not return_list else []

        phi1 = np.radians(lats[:-1])
        phi2 = np.radians(lats[1:])
        delta_phi = np.radians(lats[1:] - lats[:-1])
        delta_lambda = np.radians(lons[1:] - lons[:-1])

        a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2  # type: ignore[operator]
        distances = 2.0 * 6371000.0 * np.arcsin(np.sqrt(np.maximum(0.0, a)))

        chainages = np.insert(np.cumsum(np.nan_to_num(distances)), 0, 0.0)

        if return_list:
            return chainages.tolist()

        result = coords_df.copy()
        result["chainage_m"] = chainages
        return result

    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        radius_m = 6371000.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        value = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
        return 2.0 * radius_m * math.asin(math.sqrt(value))

    @staticmethod
    def wgs84_to_utm(lon: float, lat: float) -> Tuple[float, float, int, str]:
        zone = min(60, max(1, int((lon + 180.0) // 6.0) + 1))
        hemisphere = "north" if lat >= 0 else "south"
        epsg = (32600 if lat >= 0 else 32700) + zone
        easting, northing = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(lon, lat)
        return float(easting), float(northing), zone, hemisphere

    @staticmethod
    def utm_to_wgs84(easting: float, northing: float, zone: int, hemisphere: str) -> Tuple[float, float]:
        epsg = (32600 if hemisphere.lower().startswith("n") else 32700) + zone
        lon, lat = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True).transform(easting, northing)
        return float(lon), float(lat)

    @staticmethod
    def point_to_line_distance(
        x: float, y: float, x1: float, y1: float, x2: float, y2: float
    ) -> Tuple[float, float]:
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0.0 and dy == 0.0:
            return float(math.hypot(x - x1, y - y1)), 0.0
        t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
        t = min(1.0, max(0.0, float(t)))
        projection_x = x1 + t * dx
        projection_y = y1 + t * dy
        return float(math.hypot(x - projection_x, y - projection_y)), t

    @staticmethod
    def parse_gpx_trajectory(path: str) -> List[Dict[str, float]]:
        root = ET.parse(path).getroot()
        trajectory = []
        for element in root.iter():
            if element.tag.rsplit("}", 1)[-1] != "trkpt":
                continue
            lat = element.attrib.get("lat")
            lon = element.attrib.get("lon")
            if lat is not None and lon is not None:
                trajectory.append({"lat": float(lat), "lon": float(lon)})
        return trajectory

    @staticmethod
    def _valid_gps_points(gps_df: pd.DataFrame) -> List[Tuple[float, float]]:
        if "lat" not in gps_df.columns or "lon" not in gps_df.columns:
            return []
        lats = pd.to_numeric(gps_df["lat"], errors="coerce").to_numpy(dtype=float)
        lons = pd.to_numeric(gps_df["lon"], errors="coerce").to_numpy(dtype=float)
        valid_mask = np.isfinite(lats) & np.isfinite(lons) & (lats >= -90.0) & (lats <= 90.0) & (lons >= -180.0) & (lons <= 180.0)
        return list(zip(lons[valid_mask], lats[valid_mask]))

    @staticmethod
    def _trajectory_geometry(points: Sequence[Tuple[float, float]]):
        if not points:
            return None
        if len(points) == 1:
            return Point(points[0])
        return LineString(points)

    @staticmethod
    def _metric_transformer(points: Sequence[Tuple[float, float]]):
        if not points:
            return None
        mean_lon = sum(point[0] for point in points) / len(points)
        mean_lat = sum(point[1] for point in points) / len(points)
        zone = min(60, max(1, int((mean_lon + 180.0) // 6.0) + 1))
        epsg = (32600 if mean_lat >= 0 else 32700) + zone
        return (
            Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True),
            Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True),
        )

    @staticmethod
    def _normalise_centerline(geometry):
        if geometry is None or geometry.is_empty:
            return None
        if geometry.geom_type == "MultiLineString":
            merged = linemerge(geometry)
            if merged.geom_type == "LineString":
                return merged
        if geometry.geom_type in {"LineString", "MultiLineString"}:
            return geometry
        return None

    @staticmethod
    def _get_road_centerline(db: Session, road_id: str) -> Optional[str]:
        try:
            result = db.execute(
                text("""
                    SELECT ST_AsText(ST_SetSRID(geom, 4326)) AS wkt
                    FROM gh_roads
                    WHERE gid::VARCHAR = :road_id
                """),
                {"road_id": str(road_id)},
            ).first()
            return result[0] if result and result[0] else None
        except Exception as exc:
            spatial_logger.warning(f"Failed to fetch centerline for road_id={road_id}: {exc}")
            return None

    @staticmethod
    def _find_nearest_road(db: Session, gps_df: pd.DataFrame, max_distance_m: float = DEFAULT_MAX_DISTANCE_M) -> Optional[str]:
        points = SpatialService._valid_gps_points(gps_df)
        trajectory = SpatialService._trajectory_geometry(points)
        if trajectory is None or trajectory.is_empty:
            return None
        trajectory_wkt = trajectory.wkt

        try:
            result = db.execute(
                text("""
                    WITH trajectory AS (
                        SELECT ST_SetSRID(ST_GeomFromText(:trajectory_wkt), 4326) AS geom
                    )
                    SELECT g.gid::VARCHAR AS road_id
                    FROM gh_roads AS g
                    CROSS JOIN trajectory AS t
                    WHERE ST_DWithin(g.geom::geography, t.geom::geography, :max_distance_m)
                    ORDER BY ST_Distance(g.geom::geography, t.geom::geography) ASC
                    LIMIT 1
                """),
                {
                    "trajectory_wkt": trajectory_wkt,
                    "max_distance_m": float(max_distance_m),
                },
            ).first()
            if result and result[0]:
                spatial_logger.info("Auto-matched trajectory to road_id=%s", result[0])
                return str(result[0])
            return None
        except Exception as exc:
            spatial_logger.warning(f"Failed to auto-match road: {exc}")
            return None

    @staticmethod
    def _snap_to_centerline(points: Sequence[Tuple[float, float]], centerline_wkt: str, max_distance_m: float) -> Tuple[List[Dict[str, Any]], Any, float]:
        centerline = SpatialService._normalise_centerline(wkt_loads(centerline_wkt))
        if centerline is None:
            return [], None, 0.0
        transformers = SpatialService._metric_transformer(points)
        if transformers is None:
            return [], centerline, 0.0
        forward, inverse = transformers

        metric_centerline = transform(forward.transform, centerline)

        lons, lats = zip(*points)
        xs, ys = forward.transform(lons, lats)

        snapped_coords = []
        for index, (x, y, orig_lon, orig_lat) in enumerate(zip(xs, ys, lons, lats)):
            metric_point = Point(x, y)
            snapped_metric, _ = nearest_points(metric_centerline, metric_point)
            distance_m = float(metric_point.distance(snapped_metric))
            chainage_m = float(metric_centerline.project(snapped_metric))
            snapped_lon, snapped_lat = inverse.transform(snapped_metric.x, snapped_metric.y)
            snapped_coords.append({
                "index": index,
                "lat": float(snapped_lat),
                "lon": float(snapped_lon),
                "orig_lat": float(orig_lat),
                "orig_lon": float(orig_lon),
                "chainage_m": chainage_m,
                "distance_m": distance_m,
                "matched": distance_m <= max_distance_m,
            })
        return snapped_coords, centerline, float(metric_centerline.length)

    @staticmethod
    def _raw_chainage(points: Sequence[Tuple[float, float]]) -> List[Dict[str, Any]]:
        if not points:
            return []

        lons, lats = zip(*points)
        lats_arr = np.array(lats, dtype=float)
        lons_arr = np.array(lons, dtype=float)

        phi1 = np.radians(lats_arr[:-1])
        phi2 = np.radians(lats_arr[1:])
        delta_phi = np.radians(lats_arr[1:] - lats_arr[:-1])
        delta_lambda = np.radians(lons_arr[1:] - lons_arr[:-1])

        a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2  # type: ignore[operator]
        distances = 2.0 * 6371000.0 * np.arcsin(np.sqrt(np.maximum(0.0, a)))
        cum_distances = np.insert(np.cumsum(np.nan_to_num(distances)), 0, 0.0)

        coords = []
        for index, (lon, lat, chainage) in enumerate(zip(lons, lats, cum_distances)):
            coords.append({
                "index": index,
                "lat": float(lat),
                "lon": float(lon),
                "orig_lat": float(lat),
                "orig_lon": float(lon),
                "chainage_m": float(chainage),
                "distance_m": None,
                "matched": False,
            })
        return coords

    @classmethod
    @monitor(stage="spatial_map_matching", domain="spatial")
    def map_match_trajectory(
        cls,
        gps_df: pd.DataFrame,
        centerline_wkt: Optional[str] = None,
        road_id: Optional[str] = None,
        run_id: str = "",
        db: Optional[Session] = None,
        max_distance_m: float = DEFAULT_MAX_DISTANCE_M,
    ) -> Dict[str, Any]:
        all_points = SpatialService._valid_gps_points(gps_df)
        total_points = len(gps_df)
        trajectory_geom = SpatialService._trajectory_geometry(all_points) if all_points else None
        trajectory_wkt = trajectory_geom.wkt if trajectory_geom is not None else None
        if not all_points:
            return {
                "matched_points": 0,
                "total_points": total_points,
                "unmatched_points": total_points,
                "matched_road_id": str(road_id) if road_id else None,
                "match_status": "unmatched",
                "match_method": "provided" if road_id else "none",
                "centerline_wkt": None,
                "centerline_geojson": None,
                "trajectory_wkt": None,
                "snapped_trajectory_wkt": None,
                "snapped_coords": [],
                "chainage_min": 0.0,
                "chainage_max": 0.0,
                "matched_distance_m": 0.0,
                "road_length_m": 0.0,
                "max_distance_m": float(max_distance_m),
            }

        effective_road_id = str(road_id) if road_id else None
        resolved_centerline_wkt = centerline_wkt
        match_method = "provided" if effective_road_id else "auto"
        if db:
            if effective_road_id:
                resolved_centerline_wkt = cls._get_road_centerline(db, effective_road_id) or resolved_centerline_wkt
                if resolved_centerline_wkt is None:
                    spatial_logger.warning(f"Provided road_id={effective_road_id} was not found in gh_roads")
            else:
                effective_road_id = cls._find_nearest_road(db, gps_df, max_distance_m)
                if effective_road_id:
                    resolved_centerline_wkt = cls._get_road_centerline(db, effective_road_id)
        elif not resolved_centerline_wkt:
            match_method = "none"

        if resolved_centerline_wkt:
            snapped_coords, centerline_geom, road_length_m = cls._snap_to_centerline(all_points, resolved_centerline_wkt, max_distance_m)
            matched = [point for point in snapped_coords if point["matched"]]
            chainages = [point["chainage_m"] for point in matched]
            snapped_trajectory = LineString([(point["lon"], point["lat"]) for point in matched]) if len(matched) >= 2 else None
            result = {
                "matched_points": len(matched),
                "total_points": total_points,
                "unmatched_points": total_points - len(matched),
                "matched_road_id": effective_road_id,
                "match_status": "matched" if matched else "unmatched",
                "match_method": match_method,
                "centerline_wkt": resolved_centerline_wkt,
                "centerline_geojson": mapping(centerline_geom) if centerline_geom is not None else None,
                "trajectory_wkt": trajectory_wkt,
                "snapped_trajectory_wkt": snapped_trajectory.wkt if snapped_trajectory is not None else None,
                "snapped_coords": snapped_coords,
                "chainage_min": float(min(chainages)) if chainages else 0.0,
                "chainage_max": float(max(chainages)) if chainages else 0.0,
                "matched_distance_m": float(max(chainages) - min(chainages)) if chainages else 0.0,
                "road_length_m": road_length_m,
                "max_distance_m": float(max_distance_m),
            }
            spatial_logger.info(
                "Map matching completed",
                run_id=run_id,
                matched_points=len(matched),
                total_points=total_points,
                road_id=effective_road_id,
            )
            return result

        raw_coords = cls._raw_chainage(all_points)
        return {
            "matched_points": 0,
            "total_points": total_points,
            "unmatched_points": total_points,
            "matched_road_id": effective_road_id,
            "match_status": "unmatched",
            "match_method": match_method,
            "centerline_wkt": None,
            "centerline_geojson": None,
            "trajectory_wkt": trajectory_wkt,
            "snapped_trajectory_wkt": None,
            "snapped_coords": raw_coords,
            "chainage_min": 0.0,
            "chainage_max": float(raw_coords[-1]["chainage_m"]) if raw_coords else 0.0,
            "matched_distance_m": float(raw_coords[-1]["chainage_m"]) if raw_coords else 0.0,
            "road_length_m": 0.0,
            "max_distance_m": float(max_distance_m),
        }


SpatialGeospatialService = SpatialService