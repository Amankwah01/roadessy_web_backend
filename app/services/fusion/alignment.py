"""Placing vision detections on the same stretch of road as the IRI segments.

A survey is measured twice: the vibration logger produces the IRI and defines
the road by the distance it travelled, while the camera produces the defects and
measures distance from wherever *its* GPS started. The two loggers are started
by hand, so the second one commonly begins before the first and stops before it,
and their accumulated distances differ because of that as well as through noise.
On the KNUST Obuasi 17:12:30 trip the camera's GPS begins 470 m before the
vibration logger's, which puts every detection's chainage 470 m high relative
to the IRI's.

Indexing both by `chainage // 50` therefore assumes a common origin that the
data does not guarantee. When it is wrong, section 2 of the IRI table describes
a different 50 m of road than section 2 of the defect table, and a reader
comparing the two is comparing two different places.

This module projects each detection onto the inertial track by *physical
position* — finding where on the measured road that coordinate actually lies —
rather than by assuming both chains started in the same place. Where a
projection is impossible, the two chains are reported as unaligned instead of
being silently treated as if they agreed.

The two tables stay separate. A defect and a roughness reading are different
measurements, and this supplies the common ground needed to line them up, not a
reason to merge them.
"""
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Mean Earth radius, matching the value the inertial distance integral uses.
EARTH_RADIUS_M = 6371008.8

#: How far a detection may sit from the measured track and still be considered
#: to lie on it. Generous relative to GPS noise (a few metres) so a genuinely
#: on-road detection is not rejected, but far below the ~11 km spacing between
#: junctions, so a detection from a different trip is not absorbed into this one.
ON_TRACK_TOLERANCE_M = 150.0

#: Local equirectangular projection is accurate to well under a metre over the
#: few kilometres a survey covers, which is far finer than GPS error, so one
#: scale factor at the track's midpoint is enough to compare coordinates.
_LAT_TO_M = 111320.0

#: The largest gap between consecutive fixes that a vehicle driving a road can
#: produce. The survey GPS logs at 1 Hz and the fastest measured step across the
#: KNUST Obuasi trips is 62 m, so 200 m is clear of real driving while sitting far
#: below the multi-kilometre jumps a logger with a stale fix produces.
MAX_PLAUSIBLE_STEP_M = 200.0


@dataclass
class InertialTrack:
    """The measured road: its coordinates and the distance along it.

    `distance_m[i]` is the distance from the start of the survey to point `i`,
    which is the same quantity the 50 m IRI sections are cut from. Keeping the
    two together is the point: a detection projected onto this track lands on
    the road the IRI was measured along, not merely on a road.
    """

    lat: np.ndarray
    lon: np.ndarray
    distance_m: np.ndarray

    def __post_init__(self) -> None:
        if len(self.lat) == 0:
            raise ValueError("InertialTrack needs at least one point")
        steps = np.diff(self.distance_m) if len(self.distance_m) > 1 else np.array([])
        self.worst_step_m = float(steps.max()) if len(steps) else 0.0
        self.teleport_count = int((steps > MAX_PLAUSIBLE_STEP_M).sum())
        # A logger holding a stale fix reports the *same* coordinate over and
        # over: most steps are exactly zero and the distance arrives in a few
        # large ones. That is not a smooth drive, and a track built from it is
        # not the road, even when no single step is large enough to disqualify.
        self.frozen_fraction = float((steps == 0.0).mean()) if len(steps) else 0.0
        self.median_step_m = float(np.median(steps)) if len(steps) else 0.0

    @property
    def is_contiguous(self) -> bool:
        """Whether the fixes form one continuous drive.

        A logger that holds a stale fix and then snaps back reports a straight
        line across country between two of its points, and its cumulative
        distance counts that straight line as driven road. The survey distance
        this track supplies is then not the survey length at all, and anything
        projected onto it is placed against a road nobody drove — so the track
        is reported as unusable rather than quietly used.

        Every KNUST Obuasi vibration log trips this: their GPS holds still for
        most of the trip (median step 0.00 m) and then jumps, contributing
        between 0.2 km and 13.1 km of apparent distance from a handful of
        teleports. The camera's GPS over the same trips is clean throughout.
        """
        return self.teleport_count == 0

    @property
    def total_distance_m(self) -> float:
        return float(self.distance_m[-1]) if len(self.distance_m) else 0.0

    def __len__(self) -> int:
        return len(self.lat)


def build_inertial_track(
    latitudes: Optional[Sequence[float]],
    longitudes: Optional[Sequence[float]],
) -> Optional[InertialTrack]:
    """Cumulative distance along a lat/lon track, the frame the IRI uses.

    Returns None when there is no usable coordinate pair, so callers can tell
    "no track" from "a track of zero length".
    """
    if latitudes is None or longitudes is None:
        return None
    lat = np.asarray(latitudes, dtype=float)
    lon = np.asarray(longitudes, dtype=float)
    usable = np.isfinite(lat) & np.isfinite(lon)
    if usable.sum() < 2:
        return None
    lat, lon = lat[usable], lon[usable]

    # A track of constant coordinates is a logger with no fix, not a stationary
    # vehicle; treating it as zero length would make every detection "off-track".
    if float(np.ptp(lat)) == 0.0 and float(np.ptp(lon)) == 0.0:
        return None

    return InertialTrack(lat=lat, lon=lon, distance_m=cumulative_distance_m(lat, lon))


def cumulative_distance_m(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Distance from the first point to each successive point, in metres."""
    if len(lat) < 2:
        return np.zeros(len(lat), dtype=float)
    lat1, lat2 = np.radians(lat[:-1]), np.radians(lat[1:])
    d_lat = lat2 - lat1
    d_lon = np.radians(lon[1:] - lon[:-1])
    a = np.sin(d_lat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon / 2.0) ** 2
    steps = 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return np.concatenate([[0.0], np.cumsum(steps)])


def _metres_xy(
    lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Project to metres about a reference point.

    An equirectangular projection about the track's own midpoint: over a few
    kilometres the distortion is centimetres, orders of magnitude below the GPS
    error it is being used to compare against.
    """
    x = (lon - lon0) * _LAT_TO_M * math.cos(math.radians(lat0))
    y = (lat - lat0) * _LAT_TO_M
    return x, y


@dataclass
class DetectionPlacement:
    """Where one detection sits on the measured road."""

    detection_index: int
    distance_m: Optional[float]
    offset_m: Optional[float]
    off_track_m: Optional[float]


@dataclass
class ChainageAlignment:
    """The result of placing the vision detections on the measured road."""

    placements: List[DetectionPlacement] = field(default_factory=list)
    placed_count: int = 0
    unplaced_count: int = 0
    track_length_m: float = 0.0
    track_quality: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_aligned(self) -> bool:
        """True when at least one detection could be placed.

        False means the two chains share no usable coordinates, so their section
        indices cannot be compared and a caller should say so rather than
        present two tables that look aligned.
        """
        return self.placed_count > 0

    def offsets(self) -> List[float]:
        return [p.offset_m for p in self.placements if p.offset_m is not None]

    def summary(self) -> Dict[str, Any]:
        """Offset statistics, for the report and for the logs.

        The spread matters as much as the mean: a consistent offset just shifts
        the table, while a large spread means the two chains do not describe the
        same road and no single shift can align them.
        """
        offsets = self.offsets()
        if not offsets:
            return {
                "aligned": False,
                "placed": 0,
                "unplaced": len(self.placements),
                "reason": "no detection could be placed on the measured track",
            }
        return {
            "aligned": True,
            "placed": self.placed_count,
            "unplaced": self.unplaced_count,
            "track_length_m": round(self.track_length_m, 2),
            "offset_mean_m": round(float(np.mean(offsets)), 2),
            "offset_median_m": round(float(np.median(offsets)), 2),
            "offset_spread_m": round(float(np.percentile(offsets, 95) - np.percentile(offsets, 5)), 2),
            "track_quality": self.track_quality,
        }

def place_detections(
    detections: Sequence[Dict[str, Any]],
    track: Optional[InertialTrack],
) -> ChainageAlignment:
    """Project each detection's position onto the measured track.

    Uses the position the vision service already resolved from the camera's own
    GPS rather than its chainage, because a chainage is only comparable between
    two runs once both are known to share an origin — which is exactly what is
    in question.
    """
    alignment = ChainageAlignment()
    if track is None or len(track) < 2:
        for index, _ in enumerate(detections):
            alignment.placements.append(DetectionPlacement(index, None, None, None))
        alignment.unplaced_count = len(detections)
        return alignment

    alignment.track_length_m = track.total_distance_m
    # Carried into the report so a reader can judge the frame the sections were
    # cut in. A frozen logger still produces confident section indices, so the
    # quality of the underlying track has to travel with the answer.
    alignment.track_quality = {
        "contiguous": track.is_contiguous,
        "teleport_count": track.teleport_count,
        "worst_step_m": round(track.worst_step_m, 1),
        "median_step_m": round(track.median_step_m, 2),
        "frozen_fraction": round(track.frozen_fraction, 3),
    }
    mid = len(track) // 2
    track_x, track_y = _metres_xy(track.lat, track.lon, float(track.lat[mid]), float(track.lon[mid]))
    track_points = np.column_stack([track_x, track_y])

    for index, detection in enumerate(detections):
        coordinate = _detection_coordinate(detection)
        if coordinate is None:
            alignment.placements.append(DetectionPlacement(index, None, None, None))
            continue
        lat, lon = coordinate
        x, y = _metres_xy(
            np.array([lat]), np.array([lon]), float(track.lat[mid]), float(track.lon[mid])
        )
        offsets_sq = (track_points[:, 0] - float(x[0])) ** 2 + (track_points[:, 1] - float(y[0])) ** 2
        nearest = int(np.argmin(offsets_sq))
        off_track_m = float(math.sqrt(offsets_sq[nearest]))
        if off_track_m > ON_TRACK_TOLERANCE_M:
            alignment.placements.append(DetectionPlacement(index, None, None, off_track_m))
            continue
        distance = float(track.distance_m[nearest])
        alignment.placements.append(
            DetectionPlacement(
                detection_index=index,
                distance_m=distance,
                offset_m=distance - float(detection.get("chainage_m") or 0.0),
                off_track_m=off_track_m,
            )
        )
        alignment.placed_count += 1

    alignment.unplaced_count = len(detections) - alignment.placed_count
    return alignment


def _detection_coordinate(detection: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """The (lat, lon) a vision detection was recorded at, if it has one."""
    geom = detection.get("geom") or {}
    coordinates = geom.get("coordinates") if isinstance(geom, dict) else None
    if not coordinates or len(coordinates) < 2:
        return None
    try:
        # GeoJSON order: longitude first.
        lon = float(coordinates[0])
        lat = float(coordinates[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return None
    if lat == 0.0 and lon == 0.0:
        # The vision service's "no GPS" value, which is a real coordinate.
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return lat, lon


def apply_alignment(
    detections: Sequence[Dict[str, Any]],
    alignment: ChainageAlignment,
) -> List[Dict[str, Any]]:
    """Return detections with `chainage_m` expressed in the measured road's frame.

    The original chainage is kept as `vision_chainage_m`: it is the right
    distance along the camera's own run, and discarding it would lose the only
    record of where the video actually was. `chainage_source` then says which
    frame the value is in, so nothing downstream has to guess.
    """
    aligned: List[Dict[str, Any]] = []
    for placement in alignment.placements:
        detection = dict(detections[placement.detection_index])
        detection["vision_chainage_m"] = detection.get("chainage_m")
        if placement.distance_m is not None:
            detection["chainage_m"] = placement.distance_m
            detection["chainage_source"] = "gps_aligned"
            detection["chainage_offset_m"] = round(placement.offset_m or 0.0, 2)
        else:
            detection["chainage_source"] = (
                "off_track" if placement.off_track_m is not None else "unaligned"
            )
        aligned.append(detection)
    return aligned
