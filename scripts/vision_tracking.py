"""Cross-frame defect tracking for the YOLO video test harness.

Production has no tracker: `heavy_compute` numbers detections with
`enumerate(detections)`, so `track_id` is a position in one frame's list and
nothing associates a pothole in frame 40 with the same pothole in frame 41.
That makes a per-frame detector unusable for answering the question this
harness exists to answer — how many distinct defects are in this road, and how
bad are they.

This is a plain IoU association with a "missed this many frames, then it's a
new defect" rule. It is deliberately simple and inspectable rather than
tuned: the point is to watch a model's behaviour, not to ship a tracker.
Association is greedy by descending IoU, which is fine at the handful of boxes
per frame this detector emits and is far easier to reason about than an
assignment solver.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

BBox = Tuple[float, float, float, float]

#: Severity labels in ascending order, so "the worst reading seen" is a max
#: rather than whichever happened to come last.
_SEVERITY_ORDER = {"": -1, "LOW": 0, "MEDIUM": 1, "HIGH": 2}


def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Intersection over union of two xyxy boxes. Zero if they do not overlap."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


@dataclass
class Track:
    """One defect followed across frames."""

    track_id: int
    yolo_class_id: int
    bbox: BBox
    first_frame: int
    last_frame: int
    hits: int = 1
    best_confidence: float = 0.0
    confidences: List[float] = field(default_factory=list)
    areas: List[float] = field(default_factory=list)
    widths: List[float] = field(default_factory=list)
    lengths: List[float] = field(default_factory=list)
    deduct_values: List[float] = field(default_factory=list)
    chainages: List[float] = field(default_factory=list)
    frames: List[int] = field(default_factory=list)
    #: Single representative sample per track, kept rather than averaged so a
    #: crop of the clearest frame can be written out for review.
    best_frame_idx: int = 0
    best_bbox: Optional[BBox] = None
    #: Severity of the strongest reading seen, for any class — not just
    #: potholes. Kept as a maximum so a faint frame cannot downgrade a defect
    #: that was clearly severe once.
    severity: str = ""

    @property
    def mean_confidence(self) -> float:
        return sum(self.confidences) / len(self.confidences) if self.confidences else 0.0

    @property
    def max_area(self) -> float:
        return max(self.areas) if self.areas else 0.0

    @property
    def max_width(self) -> float:
        return max(self.widths) if self.widths else 0.0

    @property
    def max_length(self) -> float:
        return max(self.lengths) if self.lengths else 0.0

    @property
    def max_deduct_value(self) -> float:
        return max(self.deduct_values) if self.deduct_values else 0.0

    @property
    def total_deduct_value(self) -> float:
        return sum(self.deduct_values) if self.deduct_values else 0.0

    @property
    def duration_frames(self) -> int:
        return self.last_frame - self.first_frame + 1

    @property
    def start_chainage(self) -> float:
        return self.chainages[0] if self.chainages else 0.0


class IoUTracker:
    """Greedy IoU association across frames.

    `max_missed` is what stops one pothole scrolling past the camera from being
    counted a hundred times as it drifts between frames, and equally what stops
    a gap in the video from welding two separate defects into one.
    """

    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 5, max_lost: int = 60) -> None:
        self.iou_threshold = iou_threshold
        #: Frames a track may go unmatched and still be resumed.
        self.max_missed = max_missed
        #: Frames after which an unmatched track is retired for good.
        self.max_lost = max_lost
        self._next_id = 1
        self._active: Dict[int, Track] = {}
        self._retired: List[Track] = []

    @property
    def tracks(self) -> List[Track]:
        return self._retired + list(self._active.values())

    def _new_track(self, yolo_class_id: int, bbox: BBox, frame_idx: int) -> Track:
        """Create a track for a detection nothing matched.

        `hits` starts at 0 because every creation is followed by an `_absorb`
        for the frame that spawned it; starting at 1 would count the first
        sighting twice.
        """
        track = Track(
            track_id=self._next_id,
            yolo_class_id=yolo_class_id,
            bbox=bbox,
            first_frame=frame_idx,
            last_frame=frame_idx,
            hits=0,
            best_frame_idx=frame_idx,
            best_bbox=bbox,
        )
        self._next_id += 1
        self._active[track.track_id] = track
        return track

    def update(self, detections: List[Dict], frame_idx: int) -> List[Tuple[Track, Dict]]:
        """Associate this frame's detections with existing tracks.

        Returns (track, detection) pairs so the caller can annotate or persist
        using the same identity the summary reports.
        """
        for track_id, existing in list(self._active.items()):
            if frame_idx - existing.last_frame > self.max_missed:
                self._retire(track_id)

        if not detections:
            return []

        pairs: List[Tuple[float, int, int]] = []
        for det_index, det in enumerate(detections):
            bbox = det["bbox"]
            for track_id, track in self._active.items():
                score = iou(bbox, track.bbox)
                if score >= self.iou_threshold:
                    pairs.append((score, det_index, track_id))

        # Best overlap wins; each detection and each track is claimed once.
        pairs.sort(key=lambda item: item[0], reverse=True)
        det_to_track: Dict[int, Track] = {}
        used_tracks: set = set()
        for score, det_index, track_id in pairs:
            if det_index in det_to_track or track_id in used_tracks:
                continue
            det_to_track[det_index] = self._active[track_id]
            used_tracks.add(track_id)

        result: List[Tuple[Track, Dict]] = []
        for det_index, det in enumerate(detections):
            track = det_to_track.get(det_index)
            if track is None:
                track = self._new_track(det["yolo_class_id"], det["bbox"], frame_idx)
            self._absorb(track, det, det["bbox"], frame_idx)
            result.append((track, det))

        return result

    def _absorb(self, track: Track, det: Dict, bbox: BBox, frame_idx: int) -> None:
        confidence = float(det.get("confidence", 0.0))
        track.bbox = bbox
        track.last_frame = frame_idx
        track.hits += 1
        track.confidences.append(confidence)
        track.areas.append(float(det.get("area", 0.0)))
        track.widths.append(float(det.get("width", 0.0)))
        track.lengths.append(float(det.get("length", 0.0)))
        track.deduct_values.append(float(det.get("deduct_value", 0.0)))
        track.chainages.append(float(det.get("chainage_m", 0.0)))
        track.frames.append(frame_idx)
        if confidence > track.best_confidence:
            track.best_confidence = confidence
            track.best_frame_idx = frame_idx
            track.best_bbox = bbox
        severity = det.get("severity") or ""
        if _SEVERITY_ORDER.get(severity, -1) > _SEVERITY_ORDER.get(track.severity, -1):
            track.severity = severity

    def _retire(self, track_id: int) -> None:
        track = self._active.pop(track_id, None)
        if track is not None:
            self._retired.append(track)

    def finish(self, merge: bool = True) -> List[Track]:
        """Close every open track, then optionally fold duplicates together.

        `merge` runs `merge_duplicates` over the result — see that function for
        why a single pass of frame-to-frame association is not enough on its
        own.
        """
        for track_id in list(self._active):
            self._retire(track_id)
        tracks = self.tracks
        return merge_duplicates(tracks) if merge else tracks


def _has_chainage(track: Track) -> bool:
    """Whether this track has a usable position along the road.

    Chainage is all zeros when the harness was run without GPS or an assumed
    speed, and "distance 0" is not evidence that two defects are in the same
    place — merging on it would collapse every same-class pair in the video
    into one. Treated as unknown so merging falls back to image overlap alone.
    """
    return any(value != 0.0 for value in track.chainages)


def merge_duplicates(
    tracks: List[Track],
    chainage_tolerance_m: float = 30.0,
    bbox_iou: float = 0.3,
) -> List[Track]:
    """Fold tracks that are the same physical defect seen more than once.

    Frame-to-frame association alone undercounts, for two separate reasons
    that need the same remedy:

    - A weak detection flickers. This model scores around 0.05-0.18, so a
      defect clears the threshold for a few frames, drops out, and comes back.
      The gap is longer than any `max_missed` worth using, so each appearance
      became its own track.
    - Non-maximum suppression at the default IoU leaves several overlapping
      boxes for one defect, and the tracker faithfully tracks each.

    Merging is deliberately conservative and uses the strongest available
    evidence: the same class, and either the same patch of image (IoU) or the
    same place along the road (chainage). Chainage is what makes this work when
    the camera has moved on — a rutting patch re-detected 10 frames later sits
    at nearly the same distance even though the pixels have shifted.

    Tracks that overlap in *time* are never merged: two boxes present in the
    same frame are two defects, however much they overlap.
    """
    if not tracks:
        return []

    ordered = sorted(tracks, key=lambda t: t.first_frame)
    merged: List[Track] = []
    consumed: set = set()

    for candidate in ordered:
        if candidate.track_id in consumed:
            continue

        group = [candidate]
        consumed.add(candidate.track_id)

        for other in ordered:
            if other.track_id in consumed:
                continue
            if other.yolo_class_id != candidate.yolo_class_id:
                continue
            # Same moment in the video: these are simultaneous, separate defects.
            if other.first_frame <= candidate.last_frame and candidate.first_frame <= other.last_frame:
                continue
            if other.first_frame < candidate.first_frame:
                continue

            same_place = (
                _has_chainage(candidate)
                and _has_chainage(other)
                and abs(other.start_chainage - candidate.start_chainage) <= chainage_tolerance_m
            )
            same_pixels = (
                candidate.best_bbox is not None
                and other.best_bbox is not None
                and iou(candidate.best_bbox, other.best_bbox) >= bbox_iou
            )
            if same_place or same_pixels:
                group.append(other)
                consumed.add(other.track_id)

        if len(group) == 1:
            merged.append(candidate)
            continue

        merged.append(_combine(group))

    return merged


def _combine(group: List[Track]) -> Track:
    """Fold several tracks into the earliest, keeping the strongest readings.

    Confidence, size and deduct value are combined as maxima rather than
    averages: a defect seen clearly once has not been made mild by the frames
    where it was faint or missed.
    """
    primary = min(group, key=lambda t: t.first_frame)
    merged = Track(
        track_id=primary.track_id,
        yolo_class_id=primary.yolo_class_id,
        bbox=primary.bbox,
        first_frame=min(t.first_frame for t in group),
        last_frame=max(t.last_frame for t in group),
        # Zero, not the dataclass default of 1: every track in the group
        # already carries its own hits and is added below, so seeding with 1
        # would overcount every merged track by one.
        hits=0,
        best_frame_idx=primary.best_frame_idx,
        best_bbox=primary.best_bbox,
    )
    for track in group:
        merged.hits += track.hits
        merged.confidences.extend(track.confidences)
        merged.areas.extend(track.areas)
        merged.widths.extend(track.widths)
        merged.lengths.extend(track.lengths)
        merged.deduct_values.extend(track.deduct_values)
        merged.chainages.extend(track.chainages)
        merged.frames.extend(track.frames)
        if track.best_confidence > merged.best_confidence:
            merged.best_confidence = track.best_confidence
            merged.best_frame_idx = track.best_frame_idx
            merged.best_bbox = track.best_bbox
        if _SEVERITY_ORDER.get(track.severity, -1) > _SEVERITY_ORDER.get(merged.severity, -1):
            merged.severity = track.severity
    return merged


def build_tracking_registry(tracks: List[Track], class_names: Dict[int, str]) -> Dict[int, Dict]:
    """The `{track_id: {...}}` shape the fusion service consumes.

    `_compute_vision_pci` reads only `deduct_value` and `defect_type`, and
    carries the per-defect maximum into the penalty, so this passes each track's
    worst reading rather than its average — averaging would let a defect seen
    clearly once look mild.
    """
    registry: Dict[int, Dict] = {}
    for track in tracks:
        registry[track.track_id] = {
            "defect_type": class_names.get(track.yolo_class_id, "Unknown"),
            "deduct_value": track.max_deduct_value,
            "max_area": track.max_area,
            "max_width": track.max_width,
            "mean_confidence": track.mean_confidence,
            "best_confidence": track.best_confidence,
            "severity": track.severity,
            "hits": track.hits,
            "first_frame": track.first_frame,
            "last_frame": track.last_frame,
            "chainage_m": track.start_chainage,
        }
    return registry
