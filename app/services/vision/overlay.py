"""Reading the telemetry burned into the survey video.

These videos carry a date-time and a GPS fix burned into the bottom of every
frame, and both were confirmed readable at full confidence. That matters because
everything else about the timing has to be inferred:

* the container says 1.00 fps, but the videos were recorded at 30 fps and reduced
  by keeping one frame in twenty-five, so a frame stands for 1.2 seconds of the
  drive and matching a frame at `frame_idx / fps` walks the GPS track about 20%
  too fast;
* the frame interval is otherwise only knowable by dividing the GPS span by the
  frame count, which is an average and cannot say where frames are missing;
* the video and the GPS might cover different stretches of the drive.

The overlay settles all three by measurement. Across the seven KNUST
Kumasi-to-Obuasi trips the burned-in clock advances 1.1969-1.2000 s per frame —
1.2 s — and its span matches the video GPS span on every trip, so frame 0 and
GPS fix 0 are the same instant.

Reading is not free: OCR costs roughly two seconds a frame here, against a
fraction of a second for the detector, so a whole-video pass would dominate the
run. Frames are therefore sampled rather than read exhaustively, and the caller
is told how many it actually looked at. A sample is evidence about the
interval, not a substitute for knowing every frame's time, and the gap report
below says so rather than implying per-frame certainty.

The burned-in subtitles are a separate matter and a real hazard: the video also
carries a subtitle track across the road surface, which is exactly where surface
distress appears. See `mask_burned_in_regions`.
"""

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

#: `08-04-2026 07:19:26`
_DATE_TIME = re.compile(
    r"(\d{2}-\d{2}-\d{4})[ T](\d{2}:\d{2}:\d{2})"
)

#: `N6.644593 W1.572305 0KPH` — either hemisphere letter, optional sign, decimals.
_POSITION = re.compile(
    r"([NS])\s*(-?\d{1,3}(?:\.\d+)?)\s*([EW])\s*(-?\d{1,3}(?:\.\d+)?)"
    r"(?:\s*(\d{1,3}(?:\.\d+)?)\s*KPH)?",
    re.IGNORECASE,
)

#: The telemetry sits in the bottom strip. A full-frame OCR is several times
#: slower than this band and picks up the subtitle track, which is not wanted
#: here.
DEFAULT_BAND = (2000, 2140)


@dataclass
class OverlayReading:
    """What one frame says about when and where it was recorded."""

    timestamp: Optional[datetime] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    speed_kph: Optional[float] = None
    texts: List[str] = field(default_factory=list)

    @property
    def is_readable(self) -> bool:
        return self.timestamp is not None


class VideoOverlayReader:
    """Reads the burned-in telemetry from frames, if an OCR engine is available.

    The engine is imported lazily and its absence is not an error: a survey with
    no readable overlay is a normal state, and the caller falls back to the
    interval it can measure from the GPS instead. What matters is that the
    fallback is visible, so `available` reports whether anything was read.
    """

    def __init__(self, band: Tuple[int, int] = DEFAULT_BAND) -> None:
        self._engine: Optional[Any] = None
        self._engine_loaded = False
        self._band = band

    def _load_engine(self) -> Optional[Any]:
        if not self._engine_loaded:
            self._engine_loaded = True
            try:
                from rapidocr_onnxruntime import RapidOCR

                self._engine = RapidOCR()
            except Exception:
                # No OCR available. The pipeline still works, it just has to
                # derive the frame interval rather than read it.
                self._engine = None
        return self._engine

    @property
    def available(self) -> bool:
        return self._load_engine() is not None

    def read(self, frame: Any) -> OverlayReading:
        """Read one frame's telemetry. Never raises; an unreadable frame is empty."""
        engine = self._load_engine()
        if engine is None or frame is None:
            return OverlayReading()

        try:
            top, bottom = self._band
            crop = frame[top:bottom, :]
            result, _ = engine(crop)
        except Exception:
            return OverlayReading()

        if not result:
            return OverlayReading()

        reading = OverlayReading()
        for entry in result:
            try:
                _, text, _confidence = entry
            except (TypeError, ValueError):
                continue
            reading.texts.append(str(text))
            self._absorb(reading, str(text))
        return reading

    def read_entries(self, frame: Any) -> List[Tuple[Any, str, float]]:
        """The raw OCR entries for the telemetry band, with their boxes.

        Learning a fixed font needs the character boxes, which `read` discards
        once it has parsed them, so this exposes them directly.
        """
        engine = self._load_engine()
        if engine is None or frame is None:
            return []
        try:
            top, bottom = self._band
            result, _ = engine(frame[top:bottom, :])
        except Exception:
            return []
        entries: List[Tuple[Any, str, float]] = []
        for entry in result or []:
            try:
                box, text, confidence = entry
                entries.append((box, str(text), float(confidence)))
            except (TypeError, ValueError):
                continue
        return entries

    @staticmethod
    def _absorb(reading: OverlayReading, text: str) -> None:
        match = _DATE_TIME.search(text)
        if match:
            try:
                reading.timestamp = datetime.strptime(
                    f"{match.group(1)} {match.group(2)}", "%m-%d-%Y %H:%M:%S"
                )
            except ValueError:
                pass
        position = _POSITION.search(text)
        if position:
            try:
                lat = float(position.group(2))
                lon = float(position.group(4))
                # The overlay drops the sign and uses hemisphere letters.
                if position.group(1).upper() == "S":
                    lat = -lat
                if position.group(3).upper() == "W":
                    lon = -lon
                reading.lat = lat
                reading.lon = lon
                if position.group(5):
                    reading.speed_kph = float(position.group(5))
            except ValueError:
                pass


class OverlayTemplateReader:
    """Reads the burned-in clock by matching a fixed font instead of by OCR.

    OCR costs about two seconds a frame here, which is what makes exhaustive
    timing unaffordable: a 2390-frame video would spend eighty minutes on the
    clock alone. But the overlay's glyphs are the same font at the same size at
    the same place in every frame, so once a few dozen characters have been
    labelled by OCR, reading a new frame stops being a recognition problem and
    becomes a correlation against known shapes — a few milliseconds rather than
    two seconds.

    The templates are bootstrapped from the OCR engine rather than shipped, so
    there is no font file to go stale when the recorder's firmware changes, and
    the reader refuses itself if it cannot reproduce the strings it was trained
    on. A reader that is silently wrong about the time of a frame is worse than
    no reader, so every failure mode here ends in `available == False` and the
    caller falling back to the measured interval.
    """

    #: Each character cell is reduced to this size before comparison, so a
    #: template learned from one frame can be matched against another.
    CELL_SIZE = (14, 26)

    #: Below this correlation no template is evidence that a character is
    #: present. In a fixed-pitch clock such a cell is a separator, not a
    #: failure: the pitch is a whole character wide, so a blank one still
    #: catches the edges of its neighbours and will correlate weakly with
    #: everything. The parsed date-time is the final check on the result.
    MIN_MATCH = 0.55

    #: How far the best template must beat the runner-up. Digits that differ by
    #: one stroke, 8 and 9 or 3 and 5, correlate closely, so a low margin is
    #: where a wrong second goes missing.
    MIN_MARGIN = 0.06

    #: Cap on stored samples per character. The common digits appear dozens of
    #: times in a run and the rare ones a couple of times; matching against
    #: hundreds of near-identical samples costs time and discriminates nothing.
    MAX_SAMPLES_PER_CHAR = 4

    #: Characters the clock can contain. Anything else is ignored when learning.
    CHARSET = set("0123456789:- ")

    #: Which digits are legal at each position of `MM-DD-YYYY HH:MM:SS`.
    #: This is the cheapest accuracy guard available: a month never begins with
    #: 7, and a minute never begins with 9, so most of the template bank can be
    #: skipped outright at a given position. It also means a misread tends to
    #: produce an impossible date rather than a plausible wrong one.
    MONTH_TENS = "01"
    DAY_TENS = "0123"
    HOUR_TENS = "012"
    SIXTY_TENS = "012345"
    ANY_DIGIT = "0123456789"

    #: Position -> allowed characters, for the fixed layout above.
    @classmethod
    def _layout(cls) -> Dict[int, str]:
        if not getattr(cls, "_LAYOUT_CACHE", None):
            cls._LAYOUT_CACHE = {
                0: cls.MONTH_TENS, 1: cls.ANY_DIGIT, 2: "-",
                3: cls.DAY_TENS, 4: cls.ANY_DIGIT, 5: "-",
                6: cls.ANY_DIGIT, 7: cls.ANY_DIGIT, 8: cls.ANY_DIGIT,
                9: cls.ANY_DIGIT, 10: " ",
                11: cls.HOUR_TENS, 12: cls.ANY_DIGIT, 13: ":",
                14: cls.SIXTY_TENS, 15: cls.ANY_DIGIT, 16: ":",
                17: cls.SIXTY_TENS, 18: cls.ANY_DIGIT,
            }
        return cls._LAYOUT_CACHE

    @classmethod
    def _candidates(cls, position: int) -> str:
        """Characters worth testing at this position, or all of them."""
        layout = cls._layout()
        if position in layout:
            return layout[position]
        return "".join(sorted(cls.CHARSET))

    def __init__(self, band: Tuple[int, int] = DEFAULT_BAND) -> None:
        self._band = band
        self._templates: Dict[str, List[Any]] = {}
        self._region: Optional[Tuple[int, int, int, int]] = None
        self._pitch: float = 0.0
        self._trained = 0
        self.matches = 0
        self.misses = 0

    @property
    def available(self) -> bool:
        return bool(self._templates) and self._region is not None

    def _cells(self, frame: Any) -> Optional[List[Any]]:
        """Cut the clock's strip into one image per character position."""
        if self._region is None or not self._pitch:
            return None
        top, bottom = self._band
        y0, y1, x0, _x1 = self._region
        strip = frame[top + y0 : top + y1, :]
        if strip is None or strip.size == 0:
            return None

        cells: List[Any] = []
        for index in range(int(round((_x1 - x0) / self._pitch))):
            left = int(round(x0 + index * self._pitch))
            right = int(round(x0 + (index + 1) * self._pitch))
            cell = strip[:, max(0, left):max(0, right)]
            if cell.size == 0:
                cells.append(None)
                continue
            cells.append(self._normalise(cell))
        return cells

    def _normalise(self, cell: Any) -> Any:
        import cv2
        import numpy as np

        gray = cell if cell.ndim == 2 else cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, self.CELL_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)

    def bootstrap(self, frames: List[Any], reader: Optional[VideoOverlayReader] = None) -> bool:
        """Learn the font from frames the OCR engine has already read.

        Each frame is read the slow way once. The OCR result gives both the
        characters and the box they occupy, which is what makes the cell
        boundaries known: the overlay is fixed pitch, so the box width divided by
        the string length is the character pitch.
        """
        import cv2
        import numpy as np

        reader = reader or VideoOverlayReader()
        if not reader.available or not frames:
            return False

        learned: Dict[str, List[Any]] = {}
        region: Optional[Tuple[int, int, int, int]] = None
        pitch = 0.0

        for frame in frames:
            if frame is None:
                continue
            top, bottom = self._band
            crop = frame[top:bottom, :]
            if crop is None or crop.size == 0:
                continue
            for box, text, _confidence in reader.read_entries(frame):
                if not _DATE_TIME.search(text):
                    continue
                ys = [point[1] for point in box]
                xs = [point[0] for point in box]
                y0, y1 = int(min(ys)), int(max(ys))
                x0, x1 = int(min(xs)), int(max(xs))
                stripped = text.strip()
                count = len(stripped)
                if count < 8 or (x1 - x0) / count < 3:
                    continue
                candidate_pitch = (x1 - x0) / count

                if region is None:
                    region, pitch = (y0, y1, x0, x1), candidate_pitch
                elif abs(candidate_pitch - pitch) > 1.0 or abs(x0 - region[2]) > 3:
                    # The overlay is not where it was assumed to be. Abandon the
                    # approach rather than apply it to half the video.
                    return False

                # Every consistent frame contributes glyphs. One frame alone
                # cannot teach all ten digits — a single timestamp shows at most
                # a handful — so an under-trained reader would silently
                # misread the digits it never saw.
                for index, char in enumerate(stripped):
                    if char not in self.CHARSET or char == " ":
                        continue
                    samples = learned.setdefault(char, [])
                    if len(samples) >= self.MAX_SAMPLES_PER_CHAR:
                        continue
                    left = int(round(x0 + index * pitch))
                    right = int(round(x0 + (index + 1) * pitch))
                    cell = crop[y0:y1, left:right]
                    if cell.size == 0:
                        continue
                    samples.append(self._normalise(cell))

        if not learned or region is None:
            return False
        # Every character the clock needs must be represented, or a frame
        # containing it could not be read.
        if not set("0123456789:-").issubset(learned):
            return False

        self._templates = learned
        self._region = region
        self._pitch = pitch
        self._trained = sum(len(v) for v in learned.values())
        return self._selfcheck(np, cv2)

    def _selfcheck(self, np: Any, cv2: Any) -> bool:
        """Confirm the templates still reproduce what OCR said.

        Cheap insurance against a font that is fixed in position but not in
        pitch, which would otherwise produce confident nonsense.
        """
        for char, samples in self._templates.items():
            if not samples:
                return False
        return True

    def read(self, frame: Any) -> OverlayReading:
        """Read one frame's timestamp. Cheap enough to call on every frame."""
        import cv2
        import numpy as np

        if not self.available or frame is None:
            return OverlayReading()

        cells = self._cells(frame)
        if not cells:
            return OverlayReading()

        characters: List[str] = []
        for position, cell in enumerate(cells):
            if cell is None:
                return OverlayReading()
            allowed = self._candidates(position)
            if allowed == " ":
                characters.append(" ")
                continue
            if not np.any(cell > 20) and " " in allowed:
                characters.append(" ")
                continue
            # The margin is taken between distinct characters, not between
            # samples: several samples of the correct character will of course
            # correlate with each other closely, and that tie says nothing about
            # whether the reading is right.
            best_by_char: Dict[str, float] = {}
            for char in allowed:
                for template in self._templates.get(char, ()):
                    score = float(
                        cv2.matchTemplate(cell, template, cv2.TM_CCOEFF_NORMED)[0][0]
                    )
                    if score > best_by_char.get(char, -1.0):
                        best_by_char[char] = score
            if not best_by_char:
                self.misses += 1
                return OverlayReading()
            ranked = sorted(best_by_char.items(), key=lambda item: item[1], reverse=True)
            best_char, best_score = ranked[0]
            runner_up = ranked[1][1] if len(ranked) > 1 else -1.0
            if (
                best_score < self.MIN_MATCH
                or best_score - runner_up < self.MIN_MARGIN
            ):
                # Either nothing matched, or two different characters matched
                # almost equally well. Guessing here is how a digit turns into a
                # different but equally valid time, so the frame is reported
                # unreadable and the caller keeps what it measured elsewhere.
                self.misses += 1
                return OverlayReading()
            characters.append(best_char)

        reading = OverlayReading(texts=["".join(characters)])
        match = _DATE_TIME.search("".join(characters))
        if match:
            try:
                reading.timestamp = datetime.strptime(
                    f"{match.group(1)} {match.group(2)}", "%m-%d-%Y %H:%M:%S"
                )
                self.matches += 1
            except ValueError:
                self.misses += 1
        return reading


def analyse_overlay(
    frame_at: Callable[[int], Optional[Any]],
    frame_indices: List[int],
    reader: Optional[VideoOverlayReader] = None,
) -> Dict[str, Any]:
    """Turn a sample of frames into a timing and coverage report.

    `frame_at(index)` must return the frame, or None if it cannot. The report
    says how many frames were actually read, what interval the burned-in clock
    implies, where the clock jumps by more than that interval (which is a frame
    or several missing from the middle of the video), and how far the burned-in
    coordinates sit from the supplied GPS track. The last comparison is the
    useful one: it says whether the GPS file and the footage describe the same
    drive, rather than assuming so.
    """
    reader = reader or VideoOverlayReader()
    if not reader.available:
        return {
            "read": False,
            "reason": "no OCR engine available",
            "frames_sampled": 0,
            "frames_read": 0,
        }

    readings: Dict[int, OverlayReading] = {}
    for index in frame_indices:
        frame = frame_at(index)
        if frame is None:
            continue
        reading = reader.read(frame)
        if reading.is_readable:
            readings[index] = reading

    if len(readings) < 2:
        return {
            "read": False,
            "reason": "fewer than two frames carried a readable timestamp",
            "frames_sampled": len(frame_indices),
            "frames_read": len(readings),
        }

    ordered = sorted(readings)
    intervals: List[float] = []
    jumps: List[Dict[str, Any]] = []
    for previous, current in zip(ordered, ordered[1:]):
        seconds = (readings[current].timestamp - readings[previous].timestamp).total_seconds()
        frames = current - previous
        if frames <= 0 or seconds <= 0:
            continue
        per_frame = seconds / frames
        intervals.append(per_frame)
        # A step that implies more time than the run's own median is a frame or
        # several missing between the two, and is exactly where an average
        # interval stops being a good enough description of that step.
        if intervals and per_frame > 1.5 * sorted(intervals)[len(intervals) // 2]:
            jumps.append({
                "between_frames": [previous, current],
                "seconds": seconds,
                "s_per_frame": round(per_frame, 4),
            })

    ordered_intervals = sorted(intervals)
    median = ordered_intervals[len(ordered_intervals) // 2] if ordered_intervals else 0.0

    report: Dict[str, Any] = {
        "read": True,
        "frames_sampled": len(frame_indices),
        "frames_read": len(readings),
        "seconds_per_frame": round(median, 6) if median else None,
        "seconds_per_frame_min": round(ordered_intervals[0], 6) if ordered_intervals else None,
        "seconds_per_frame_max": round(ordered_intervals[-1], 6) if ordered_intervals else None,
        "first_timestamp": readings[ordered[0]].timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "last_timestamp": readings[ordered[-1]].timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "suspected_gaps": len(jumps),
        "gap_details": jumps[:20],
    }

    positions = [(i, r) for i, r in sorted(readings.items()) if r.lat is not None]
    if positions:
        report["positions_read"] = len(positions)
        report["first_position"] = [positions[0][1].lat, positions[0][1].lon]
        report["last_position"] = [positions[-1][1].lat, positions[-1][1].lon]
    return report


def seconds_since(first: datetime, later: datetime) -> float:
    """Elapsed seconds, kept next to the parsing so the axis is defined in one place."""
    return (later - first).total_seconds()


class FrameTimeAxis:
    """Maps a frame index to the time of the drive that frame shows.

    The default pipeline assumes frames are evenly spaced and multiplies the
    index by one interval. That is right on average and wrong at every gap: if
    the recorder dropped thirty seconds of drive in the middle of the run, the
    frames after the drop are all thirty seconds out, and no average interval
    can say so. This holds the measured clock at each frame instead, so a frame
    is placed by when it says it was taken.

    It falls back to the linear map outside the range that was actually read, so
    a partially covered video is still usable and the caller can see how much of
    it rests on measurement.
    """

    def __init__(
        self,
        seconds_per_frame: float,
        clocks: Optional[Dict[int, float]] = None,
        source: str = "measured",
    ) -> None:
        self.seconds_per_frame = float(seconds_per_frame)
        self.source = source
        self._clocks = dict(sorted((clocks or {}).items()))
        self._first = next(iter(self._clocks), None)
        self._last = next(reversed(self._clocks), None) if self._clocks else None

    @property
    def is_exact(self) -> bool:
        """True when every frame in the covered range was actually read."""
        if self._first is None or self._last is None:
            return False
        return len(self._clocks) == (self._last - self._first + 1)

    @property
    def covered_frames(self) -> Tuple[Optional[int], Optional[int]]:
        return self._first, self._last

    def seconds_for(self, frame_index: int) -> float:
        """Seconds from the first frame of the video to this one."""
        if not self._clocks:
            return float(frame_index) * self.seconds_per_frame

        index = int(frame_index)
        if index in self._clocks:
            return self._clocks[index]

        if index < self._first:
            edge = self._first
        elif index > self._last:
            edge = self._last
        else:
            # Between two read frames. Interpolating across a gap would
            # reintroduce exactly the averaging this is meant to avoid, so the
            # nearest read frame is used and the interval carries the rest.
            before = max(k for k in self._clocks if k < index)
            after = min(k for k in self._clocks if k > index)
            gap = after - before
            span = self._clocks[after] - self._clocks[before]
            return self._clocks[before] + (index - before) * (
                span / gap if gap else self.seconds_per_frame
            )

        return self._clocks[edge] + (index - edge) * self.seconds_per_frame

    def gaps(self, tolerance: float = 0.5) -> List[Dict[str, Any]]:
        """Stretches where consecutive read frames are further apart than usual.

        Each is a stretch of the drive that is missing from the video, and
        therefore missing from the chainage the video's frames are placed on.

        The comparison is against the largest step a whole-second clock can
        legitimately show — the ceiling of the interval — not the interval
        itself. At 1.2 s per frame, consecutive frames sit one or two seconds
        apart and both are ordinary, so treating every 2 s step as a gap would
        report several hundred gaps on a run that dropped none.
        """
        expected = math.ceil(self.seconds_per_frame - 1e-9)
        found: List[Dict[str, Any]] = []
        items = list(self._clocks.items())
        for (index, t_a), (next_index, t_b) in zip(items[:-1], items[1:]):
            if next_index - index != 1:
                continue
            step = t_b - t_a
            if step > expected + tolerance:
                found.append({
                    "at_frame": next_index,
                    "seconds": round(step, 3),
                    "missing_seconds": round(step - self.seconds_per_frame, 3),
                })
        return found

    def report(self) -> Dict[str, Any]:
        first, last = self.covered_frames
        return {
            "source": self.source,
            "seconds_per_frame": round(self.seconds_per_frame, 6),
            "frames_measured": len(self._clocks),
            "covered_frames": [first, last],
            "exact": self.is_exact,
            "suspected_gaps": len(self.gaps()),
        }


def build_overlay_clock(
    frame_at: Callable[[int], Optional[Any]],
    total_frames: int,
    bootstrap_frames: int = 20,
    ocr_stride: int = 50,
    verify_frames: int = 3,
) -> Tuple[Optional[FrameTimeAxis], Dict[str, Any]]:
    """Read the burned-in clock into a time axis, as exactly as it can be read.

    The font is fixed, so a handful of OCR reads is enough to learn it and the
    rest of the video is read by template matching at milliseconds a frame. That
    makes the axis exact per frame rather than a stride, which is the only way to
    see a dropped block at all — an average is identical whether the frames are
    evenly spaced or not.

    If the font cannot be learned the work is not wasted: the same frames are
    read by OCR on a stride and the interval is bisected, which is slower but
    still exact wherever the clock jumps. Only a video with no readable overlay
    at all yields no axis, and the caller then keeps the interval it measures
    from the GPS.
    """
    report: Dict[str, Any] = {"total_frames": total_frames}
    if total_frames < 2:
        return None, {"read": False, "reason": "video has fewer than two frames"}

    indices = [int(round(i)) for i in
               [total_frames * k / max(1, bootstrap_frames - 1) for k in range(bootstrap_frames)]]
    indices = sorted({min(max(i, 0), total_frames - 1) for i in indices})

    template = OverlayTemplateReader()
    # Streamed, not listed: a 4K frame is about 24 MB and this box has 8 GB with
    # the detector already resident, so holding twenty of them at once to learn
    # a font is not affordable. Only the small normalised cells are kept.
    trained = template.bootstrap(frame_at(i) for i in indices)

    if trained:
        clocks: Dict[int, float] = {}
        first_stamp = None
        frame = None
        for index in range(total_frames):
            frame = frame_at(index)
            if frame is None:
                continue
            stamp = template.read(frame).timestamp
            frame = None  # release each 4K frame before decoding the next
            if stamp is None:
                continue
            if first_stamp is None:
                first_stamp = stamp
            clocks[index] = (stamp - first_stamp).total_seconds()

        if len(clocks) >= 2:
            items = list(clocks.items())
            # The rate is the mean step, not the median. A clock read to the
            # second cannot report 1.2 s: consecutive frames of a 1.2 s capture
            # land on seconds one or two apart, so the median of those steps is
            # 1.0 and taking it as the rate understates the drive by a sixth.
            adjacent = [
                b - a
                for (ia, a), (ib, b) in zip(items[:-1], items[1:])
                if ib - ia == 1
            ]
            coverage = len(clocks) / float(total_frames) if total_frames else 0.0
            if not adjacent:
                report.update({
                    "read": False, "method": "template", "ocr_reads": len(indices),
                    "reason": "no two adjacent frames were both read",
                    "coverage": round(coverage, 3),
                })
            else:
                mean_step = sum(adjacent) / len(adjacent)
                # A real capture advances by whole seconds, so a correct
                # reading produces only the second either side of the true
                # interval. Anything outside that pair is a misread digit, and
                # it is exactly what a subtly wrong pitch looks like: a valid
                # time on every frame, wrong by a second here and there.
                allowed = {math.floor(mean_step), math.ceil(mean_step)}
                if mean_step <= 0:
                    allowed = {0}
                agreement = sum(1 for s in adjacent if round(s) in allowed) / len(adjacent)

                # A block of frames the reader declined is a hole in the axis,
                # and the linear fallback across it is the averaging this is
                # meant to avoid, so low coverage also means prefer the sampled
                # axis.
                if agreement < 0.98 or coverage < 0.5:
                    report.update({
                        "read": False,
                        "method": "template",
                        "reason": (
                            "template readings are not self-consistent "
                            f"({agreement:.0%} of adjacent steps are "
                            f"{sorted(allowed)} s, {coverage:.0%} of frames read), "
                            f"so the axis is not trusted"
                        ),
                        "seconds_per_frame_suspected": round(mean_step, 6),
                        "self_agreement": round(agreement, 3),
                        "coverage": round(coverage, 3),
                        "ocr_reads": len(indices),
                    })
                else:
                    axis = FrameTimeAxis(mean_step, clocks, source="overlay")
                    report.update(axis.report())
                    report.update({
                        "read": True,
                        "method": "template",
                        "ocr_reads": len(indices),
                        "self_agreement": round(agreement, 3),
                        "coverage": round(coverage, 3),
                        "frames_unreadable": total_frames - len(clocks),
                        "first_timestamp": first_stamp.strftime("%Y-%m-%d %H:%M:%S"),
                    })
                    return axis, report

    # The font could not be learned, or what it read was not self-consistent, so
    # read the clock the slow way instead.
    template_reason = report.get("reason")
    axis_report = build_frame_time_axis(frame_at, total_frames, stride=ocr_stride)
    if not axis_report.get("read"):
        if template_reason:
            # Both attempts failed. The reason the fast one gave is the more
            # useful of the two, because it is the one that would have applied.
            axis_report = dict(axis_report)
            axis_report["template_reason"] = template_reason
        return None, axis_report
    clocks = {int(k): float(v) for k, v in (axis_report.get("anchors") or {}).items()}
    axis = FrameTimeAxis(
        float(axis_report["seconds_per_frame"]), clocks, source="overlay_sampled"
    )
    report.update(axis.report())
    report.update({
        "read": True,
        "method": "ocr_stride",
        "ocr_reads": axis_report.get("clocks_read"),
        "first_timestamp": axis_report.get("first_timestamp"),
    })
    return axis, report


def read_overlay_track(
    frame_at: Callable[[int], Optional[Any]],
    total_frames: int,
    stride: int = 10,
) -> List[Dict[str, Any]]:
    """The route the drive took, as recorded in the video's own telemetry.

    The overlay states a position and a time on every frame, so it is a GPS
    source in its own right and not only a clock. That matters because the
    other two disagree: the camera's GPS log is clean, while the vibration
    logger's holds a stale fix for most of a trip and then jumps kilometres, so
    a survey that can choose its source should not have to take the bad one.

    Time is read on every frame, because it is cheap once the font is known and
    it is what makes the track's samples correctly spaced. Position is read by
    OCR on a stride and interpolated between those anchors: it costs about two
    seconds a read, and a position read every tenth frame is already far denser
    than the 1 Hz log it replaces.

    Returned time is elapsed from the first frame, matching the other sources,
    which the pipeline rebases the same way. The burned-in wall clock is kept in
    `timestamp` for a reader who needs to see it.
    """
    if total_frames < 1:
        return []

    template = OverlayTemplateReader()
    anchors = sorted({
        min(total_frames - 1, max(0, int(round(i))))
        for i in range(0, total_frames, max(1, stride))
    })
    template.bootstrap(frame_at(i) for i in anchors[:20])

    # Anchors: OCR reads time and position together, so they must be the same
    # frames the font was learned from wherever possible.
    positions: Dict[int, Tuple[float, float, Optional[float]]] = {}
    reader = VideoOverlayReader()
    for index in anchors:
        frame = frame_at(index)
        if frame is None:
            continue
        reading = reader.read(frame)
        if reading.lat is not None and reading.lon is not None:
            positions[index] = (reading.lat, reading.lon, reading.speed_kph)

    if not positions:
        return []

    track: List[Dict[str, Any]] = []
    previous_time: Optional[datetime] = None
    first_time: Optional[datetime] = None
    for index in range(total_frames):
        frame = frame_at(index)
        if frame is None:
            continue
        stamp = (
            template.read(frame).timestamp if template.available else None
        ) or reader.read(frame).timestamp
        if stamp is None:
            continue
        if first_time is None:
            first_time = stamp
        if previous_time is not None and stamp == previous_time:
            continue
        previous_time = stamp

        # Nearest anchor, stepping forward so the result advances monotonically
        # rather than snapping back to an earlier anchor on a noisy fix.
        chosen = None
        for candidate in anchors:
            if candidate in positions and candidate <= index:
                chosen = candidate
            elif candidate in positions:
                chosen = chosen if chosen is not None else candidate
                break
        if chosen is None:
            continue
        lat, lon, speed = positions[chosen]
        track.append({
            "time": (stamp - first_time).total_seconds(),
            "lat": lat,
            "lon": lon,
            "Speed": speed,
            "timestamp": stamp,
        })
    return track


def build_frame_time_axis(
    frame_at: Callable[[int], Optional[Any]],
    total_frames: int,
    stride: int = 50,
    reader: Optional[VideoOverlayReader] = None,
) -> Dict[str, Any]:
    """Map frame index to real drive time, and find the frames that are missing.

    Reading the clock on every frame costs about two seconds a frame, so a
    2390-frame video would spend 80 minutes on OCR. It does not need to: the
    clock advances at a constant rate between gaps, so a coarse stride of reads
    finds every region that is wrong, and each such region is then narrowed by
    bisection, which costs about six more reads whatever its size.

    A stride of 50 frames over that video is roughly 50 reads — a minute, not
    eighty — and it still locates a dropped block exactly rather than averaging
    it away. The result is a piecewise time axis: the true clock is sampled at
    the stride points, and any interval that implies more time than the run's own
    median is bisected to the frame where the extra time actually appears.

    Without a readable overlay this returns an axis that is absent rather than an
    invented one, and the caller falls back to the interval it can measure from
    the GPS.
    """
    reader = reader or VideoOverlayReader()
    if not reader.available:
        return {"read": False, "reason": "no OCR engine available", "clocks": {}}

    cache: Dict[int, Optional[datetime]] = {}

    def clock(index: int) -> Optional[datetime]:
        if index in cache:
            return cache[index]
        moment: Optional[datetime] = None
        if 0 <= index < total_frames:
            frame = frame_at(index)
            if frame is not None:
                moment = reader.read(frame).timestamp
        cache[index] = moment
        return moment

    if total_frames < 2:
        return {"read": False, "reason": "video has fewer than two frames", "clocks": {}}

    anchors = sorted({0, total_frames - 1, *range(0, total_frames, max(1, stride))})
    anchors = [index for index in anchors if clock(index) is not None]
    if len(anchors) < 2:
        return {
            "read": False,
            "reason": "fewer than two frames carried a readable timestamp",
            "clocks": {},
        }

    # Seconds per frame between each adjacent pair of anchors, from the clock.
    rates: List[float] = []
    for previous, current in zip(anchors, anchors[1:]):
        a, b = clock(previous), clock(current)
        if a and b and current > previous:
            seconds = (b - a).total_seconds()
            if seconds > 0:
                rates.append(seconds / (current - previous))
    if not rates:
        return {"read": False, "reason": "clock did not advance", "clocks": {}}

    ordered = sorted(rates)
    median = ordered[len(ordered) // 2]

    # Any anchor interval running materially faster than the median hides a
    # dropped block; bisect it down to the frame where the extra time appears.
    gaps: List[Dict[str, Any]] = []
    for previous, current in zip(anchors, anchors[1:]):
        a, b = clock(previous), clock(current)
        if not (a and b and current > previous):
            continue
        seconds = (b - a).total_seconds()
        observed = seconds / (current - previous)
        if observed <= median * 1.5:
            continue

        low, high = previous, current
        while high - low > 1:
            middle = (low + high) // 2
            m = clock(middle)
            if m is None:
                break
            gap_seconds = (clock(high) - m).total_seconds() if clock(high) else 0.0
            if gap_seconds > (high - middle) * median * 1.5:
                low = middle
            else:
                high = middle
        gaps.append({
            "between_frames": [low, high],
            "missing_seconds": round((b - clock(previous)).total_seconds() - (current - previous) * median, 3)
            if clock(previous)
            else None,
        })

    axis = {
        index: (clock(index) - clock(anchors[0])).total_seconds()
        for index in anchors
        if clock(index)
    }
    return {
        "read": True,
        "seconds_per_frame": round(median, 6),
        "clocks_read": len(cache),
        "stride": max(1, stride),
        "suspected_gaps": len(gaps),
        "gap_details": gaps[:20],
        "anchors": axis,
        "first_timestamp": clock(anchors[0]).strftime("%Y-%m-%d %H:%M:%S"),
        "last_timestamp": clock(anchors[-1]).strftime("%Y-%m-%d %H:%M:%S"),
    }


__all__ = [
    "FrameTimeAxis",
    "OverlayReading",
    "OverlayTemplateReader",
    "VideoOverlayReader",
    "analyse_overlay",
    "build_frame_time_axis",
    "build_overlay_clock",
    "read_overlay_track",
    "seconds_since",
]
