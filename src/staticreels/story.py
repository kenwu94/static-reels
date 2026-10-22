"""Story files: load the JSON, validate it, and do the timeline math.

A story is a list of segments played back to back. Segment start times are
not stored in the JSON; each one is the running total of the durations before
it, so editing a duration can never leave a gap or an overlap. Times inside a
segment (text lines, handle) are relative to that segment's start. Audio cue
times are absolute (seconds from the start of the video).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SEGMENT_TYPES = ("static_text", "ai_clip", "countdown", "end_card", "spectrogram", "blueprint")
BACKGROUNDS = ("static", "dark", "black")  # static_text backgrounds, brightest to darkest
# Cues with a start and an end. room/muffle/silence shape the whole mix in their range.
RANGE_CUES = ("drone", "hiss", "silence", "grind", "breathing", "panic", "paper", "tick", "riser",
              "drawtones", "room", "muffle", "day")
# Cues with a list of times.
POINT_CUES = ("heartbeat", "knock", "creak", "bedsprings", "tape", "snap", "beep", "sub", "thud", "sample")
DEFAULT_GLITCH = {"static_text": 0.15, "ai_clip": 0.0, "countdown": 0.5, "end_card": 0.1,
                  "spectrogram": 0.05, "blueprint": 0.0}
EPS = 1e-6
_MISSING = object()


class StoryError(ValueError):
    """The story file is malformed. The message says which field is wrong."""


@dataclass
class Line:
    text: str
    start: float  # seconds after the segment starts when typing begins
    end: float | None = None  # seconds after the segment starts when it disappears (None = segment end)
    cps: float = 28.0  # typewriter speed, characters per second

    @property
    def typed_by(self) -> float:
        """Segment-relative time at which the whole line has been typed."""
        return self.start + len(self.text) / self.cps

    def chars_at(self, t: float) -> int:
        """How many characters are visible at segment time t (0 before the line starts or after it ends)."""
        if t < self.start or (self.end is not None and t >= self.end):
            return 0
        return min(len(self.text), int((t - self.start) * self.cps) + 1)


@dataclass
class Overlay:
    rec_dot: bool = False
    timestamp: str | None = None


@dataclass
class Plan:
    """A floor plan drawn by the spectrogram and blueprint segments."""

    bedroom: float = 1.0  # how much of the bedroom is left, 1 = full width, 0 = gone (studio)
    label: str | None = None  # dimension written across the bedroom, e.g. 11'4"
    outline: str | None = None  # dashed person outline: None, "bed" or "wall"


@dataclass
class Segment:
    index: int
    type: str
    duration: float
    start: float = 0.0
    lines: list[Line] = field(default_factory=list)
    fade_in: float = 0.0
    fade_out: float = 0.0
    glitch: float = 0.0  # 0 = never glitch, 1 = constantly broken
    background: str = "static"  # static_text only: static / dark / black
    # ai_clip only
    clip: str | None = None
    prompt: str | None = None
    overlay: Overlay = field(default_factory=Overlay)
    cut_to_black: float = 0.0  # seconds of hard black at the end of the clip
    # countdown only (numbers or short text like 2'6")
    digits: list[str] = field(default_factory=list)
    # end_card only
    handle: str | None = None
    handle_start: float = 0.0
    # spectrogram / blueprint only
    plan: Plan = field(default_factory=Plan)
    reveal: float = 0.0  # spectrogram: seconds for the plan to scroll in
    pencil: str | None = None  # blueprint: handwritten pencil note
    pencil_start: float = 0.0

    @property
    def end(self) -> float:
        return self.start + self.duration

    @property
    def label(self) -> str:
        return f"segment {self.index + 1} ({self.type}, {self.start:g}-{self.end:g}s)"


@dataclass
class Cue:
    type: str
    start: float = 0.0
    end: float = 0.0
    times: list[float] = field(default_factory=list)
    gain: float = 1.0
    gain_to: float | None = None  # range cues: gain at the end of the range (rising/falling envelope)
    pan: float = 0.0  # -1 = hard left, 0 = centre, 1 = hard right
    size: float = 0.5  # room: 0 = no echo (tiny/sealed), 1 = large echoey room
    cutoff: float = 1500.0  # muffle: low-pass frequency in Hz
    period: float = 4.5  # breathing: seconds per breath
    file: str | None = None  # sample: path to a recorded sound


@dataclass
class Story:
    title: str
    fps: int
    width: int
    height: int
    segments: list[Segment]
    cues: list[Cue]
    voiceover: str | None = None
    font: str | None = None
    caption: dict = field(default_factory=dict)
    source: Path | None = None

    @property
    def duration(self) -> float:
        return sum(s.duration for s in self.segments)

    @property
    def total_frames(self) -> int:
        return round(self.duration * self.fps)

    @property
    def aspect(self) -> str:
        g = math.gcd(self.width, self.height)
        return f"{self.width // g}:{self.height // g}"

    @property
    def ai_segments(self) -> list[Segment]:
        return [s for s in self.segments if s.type == "ai_clip"]

    def frame_range(self, seg: Segment) -> tuple[int, int]:
        """[first, end) frame indices of a segment.

        Rounding the absolute boundaries (rather than each duration) means the
        per-segment frame counts always add up to total_frames, even with
        fractional durations.
        """
        return round(seg.start * self.fps), round(seg.end * self.fps)

    def segment_at(self, t: float) -> Segment:
        """The segment playing at absolute time t."""
        for seg in self.segments:
            if seg.start <= t < seg.end:
                return seg
        if abs(t - self.duration) < EPS:
            return self.segments[-1]
        raise ValueError(f"t={t} is outside the story (0-{self.duration:g}s)")


# --------------------------------------------------------------------------- parsing helpers


def _get(d: dict, key: str, where: str, kind: type, default: Any = _MISSING) -> Any:
    """Fetch d[key], checking its type; raise StoryError naming the field if it's wrong."""
    if key not in d or d[key] is None:
        if default is _MISSING:
            raise StoryError(f"{where}: missing required field '{key}'")
        return default
    value = d[key]
    if kind is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise StoryError(f"{where}.{key}: expected a number, got {value!r}")
        return float(value)
    if kind is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise StoryError(f"{where}.{key}: expected a whole number, got {value!r}")
        return value
    if kind is str:
        if not isinstance(value, str) or not value.strip():
            raise StoryError(f"{where}.{key}: expected a non-empty string, got {value!r}")
        return value
    if not isinstance(value, kind):
        raise StoryError(f"{where}.{key}: expected {kind.__name__}, got {type(value).__name__}")
    return value


def parse_size(value: Any) -> tuple[int, int]:
    """Accept "1080x1920" or [1080, 1920]. Both sides must be even (required by yuv420p)."""
    if isinstance(value, str):
        m = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", value)
        if not m:
            raise StoryError(f"size: expected something like \"1080x1920\", got {value!r}")
        w, h = int(m[1]), int(m[2])
    elif isinstance(value, list) and len(value) == 2 and all(isinstance(v, int) and not isinstance(v, bool) for v in value):
        w, h = value
    else:
        raise StoryError(f"size: expected \"WIDTHxHEIGHT\" or [width, height], got {value!r}")
    if w <= 0 or h <= 0 or w % 2 or h % 2:
        raise StoryError(f"size: width and height must be positive even numbers, got {w}x{h}")
    return w, h


def _parse_lines(raw: Any, where: str, duration: float) -> list[Line]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise StoryError(f"{where}.lines: expected a list")
    lines = []
    for i, item in enumerate(raw):
        w = f"{where}.lines[{i}]"
        if not isinstance(item, dict):
            raise StoryError(f"{w}: expected an object like {{\"text\": ..., \"start\": ...}}")
        line = Line(
            text=_get(item, "text", w, str),
            start=_get(item, "start", w, float),
            end=_get(item, "end", w, float, None),
            cps=_get(item, "cps", w, float, 28.0),
        )
        if not 0 <= line.start < duration:
            raise StoryError(f"{w}.start: {line.start:g}s is outside the segment (0-{duration:g}s)")
        if line.end is not None and not line.start < line.end <= duration + EPS:
            raise StoryError(f"{w}.end: must be after start and no later than {duration:g}s")
        if line.cps <= 0:
            raise StoryError(f"{w}.cps: must be positive")
        visible_until = line.end if line.end is not None else duration
        if line.typed_by > visible_until + EPS:
            raise StoryError(
                f"{w}: typing takes until {line.typed_by:.2f}s but the line is only visible until "
                f"{visible_until:g}s; start it earlier, shorten it, or raise cps"
            )
        lines.append(line)
    return lines


def _parse_segment(raw: Any, index: int, start: float) -> Segment:
    where = f"segments[{index}]"
    if not isinstance(raw, dict):
        raise StoryError(f"{where}: expected an object")
    kind = _get(raw, "type", where, str)
    if kind not in SEGMENT_TYPES:
        raise StoryError(f"{where}.type: {kind!r} is not one of {', '.join(SEGMENT_TYPES)}")
    duration = _get(raw, "duration", where, float)
    if duration <= 0:
        raise StoryError(f"{where}.duration: must be greater than 0")

    seg = Segment(
        index=index,
        type=kind,
        duration=duration,
        start=start,
        lines=_parse_lines(raw.get("lines"), where, duration),
        fade_in=_get(raw, "fade_in", where, float, 0.0),
        fade_out=_get(raw, "fade_out", where, float, 0.0),
        glitch=_get(raw, "glitch", where, float, DEFAULT_GLITCH[kind]),
    )
    if seg.fade_in < 0 or seg.fade_out < 0 or seg.fade_in + seg.fade_out > duration + EPS:
        raise StoryError(f"{where}: fade_in + fade_out must be between 0 and the segment duration")
    if not 0 <= seg.glitch <= 1:
        raise StoryError(f"{where}.glitch: must be between 0 and 1")

    if kind == "ai_clip":
        seg.clip = _get(raw, "clip", where, str)
        seg.prompt = _get(raw, "prompt", where, str)
        ov = _get(raw, "overlay", where, dict, {})
        seg.overlay = Overlay(
            rec_dot=_get(ov, "rec_dot", f"{where}.overlay", bool, False),
            timestamp=_get(ov, "timestamp", f"{where}.overlay", str, None),
        )
        seg.cut_to_black = _get(raw, "cut_to_black", where, float, 0.0)
        if not 0 <= seg.cut_to_black < duration:
            raise StoryError(f"{where}.cut_to_black: must be at least 0 and shorter than the segment")
    elif kind == "countdown":
        digits = _get(raw, "digits", where, list)
        ok = all((isinstance(d, int) and not isinstance(d, bool)) or (isinstance(d, str) and d.strip()) for d in digits)
        if not digits or not ok:
            raise StoryError(f"{where}.digits: expected a non-empty list of numbers or short text, e.g. [5, 4, 3] or [\"2'6\\\"\"]")
        seg.digits = [str(d) for d in digits]
    elif kind == "end_card":
        seg.handle = _get(raw, "handle", where, str, None)
        seg.handle_start = _get(raw, "handle_start", where, float, 0.0)
        if not 0 <= seg.handle_start < duration:
            raise StoryError(f"{where}.handle_start: must be inside the segment (0-{duration:g}s)")
    elif kind in ("spectrogram", "blueprint"):
        p = _get(raw, "plan", where, dict, {})
        w = f"{where}.plan"
        seg.plan = Plan(
            bedroom=_get(p, "bedroom", w, float, 1.0),
            label=_get(p, "label", w, str, None),
            outline=_get(p, "outline", w, str, None),
        )
        if not 0 <= seg.plan.bedroom <= 1:
            raise StoryError(f"{w}.bedroom: must be between 0 (gone) and 1 (full width)")
        if seg.plan.outline not in (None, "bed", "wall"):
            raise StoryError(f"{w}.outline: must be \"bed\", \"wall\" or null")
        seg.reveal = _get(raw, "reveal", where, float, round(duration * 0.8, 3))
        if not 0 < seg.reveal <= duration:
            raise StoryError(f"{where}.reveal: must be greater than 0 and no longer than the segment")
        seg.pencil = _get(raw, "pencil", where, str, None)
        seg.pencil_start = _get(raw, "pencil_start", where, float, 0.0)
        if not 0 <= seg.pencil_start < duration:
            raise StoryError(f"{where}.pencil_start: must be inside the segment (0-{duration:g}s)")
    if kind == "static_text":
        seg.background = _get(raw, "background", where, str, "static")
        if seg.background not in BACKGROUNDS:
            raise StoryError(f"{where}.background: {seg.background!r} is not one of {', '.join(BACKGROUNDS)}")
    return seg


def _parse_cue(raw: Any, index: int, total: float) -> Cue:
    where = f"audio_cues[{index}]"
    if not isinstance(raw, dict):
        raise StoryError(f"{where}: expected an object")
    kind = _get(raw, "type", where, str)
    if kind not in RANGE_CUES + POINT_CUES:
        raise StoryError(f"{where}.type: {kind!r} is not one of {', '.join(RANGE_CUES + POINT_CUES)}")
    cue = Cue(
        type=kind,
        gain=_get(raw, "gain", where, float, 1.0),
        pan=_get(raw, "pan", where, float, 0.0),
        size=_get(raw, "size", where, float, 0.5),
        cutoff=_get(raw, "cutoff", where, float, 1500.0),
        period=_get(raw, "period", where, float, 4.5),
    )
    if cue.gain < 0:
        raise StoryError(f"{where}.gain: must not be negative")
    if not -1 <= cue.pan <= 1:
        raise StoryError(f"{where}.pan: must be between -1 (left) and 1 (right)")
    if not 0 <= cue.size <= 1:
        raise StoryError(f"{where}.size: must be between 0 and 1")
    if not 50 <= cue.cutoff <= 20000:
        raise StoryError(f"{where}.cutoff: must be between 50 and 20000 Hz")
    if cue.period <= 0.3:
        raise StoryError(f"{where}.period: must be longer than 0.3 seconds")
    if kind == "sample":
        cue.file = _get(raw, "file", where, str)

    if kind in RANGE_CUES:
        cue.start = _get(raw, "start", where, float)
        cue.end = _get(raw, "end", where, float)
        if not 0 <= cue.start < cue.end <= total + EPS:
            raise StoryError(
                f"{where}: needs 0 <= start < end <= {total:g}s (the video length), "
                f"got start={cue.start:g}, end={cue.end:g}"
            )
        cue.gain_to = _get(raw, "gain_to", where, float, None)
        if cue.gain_to is not None and cue.gain_to < 0:
            raise StoryError(f"{where}.gain_to: must not be negative")
    else:
        times = _get(raw, "times", where, list)
        if not times:
            raise StoryError(f"{where}.times: expected at least one time")
        for i, t in enumerate(times):
            if isinstance(t, bool) or not isinstance(t, (int, float)) or not 0 <= t < total:
                raise StoryError(f"{where}.times[{i}]: {t!r} is not a time within the video (0-{total:g}s)")
        cue.times = [float(t) for t in times]
    return cue


def parse_story(data: Any, source: Path | None = None) -> Story:
    """Build a validated Story from already-decoded JSON."""
    if not isinstance(data, dict):
        raise StoryError("the story file must contain a JSON object")
    title = _get(data, "title", "story", str)
    fps = _get(data, "fps", "story", int, 30)
    if fps <= 0:
        raise StoryError("fps: must be greater than 0")
    width, height = parse_size(data.get("size", "1080x1920"))

    raw_segments = _get(data, "segments", "story", list)
    if not raw_segments:
        raise StoryError("segments: the story needs at least one segment")
    segments, t = [], 0.0
    for i, raw in enumerate(raw_segments):
        seg = _parse_segment(raw, i, t)
        segments.append(seg)
        t = seg.end
    total = t

    declared = _get(data, "duration", "story", float, None)
    if declared is not None and abs(declared - total) > EPS:
        raise StoryError(f"duration: segment durations add up to {total:g}s but \"duration\" says {declared:g}s")

    cues = [_parse_cue(c, i, total) for i, c in enumerate(_get(data, "audio_cues", "story", list, []))]

    caption = _get(data, "caption", "story", dict, {})
    hashtags = caption.get("hashtags", [])
    if not isinstance(hashtags, list) or not all(isinstance(h, str) for h in hashtags):
        raise StoryError("caption.hashtags: expected a list of strings")

    return Story(
        title=title,
        fps=fps,
        width=width,
        height=height,
        segments=segments,
        cues=cues,
        voiceover=_get(data, "voiceover", "story", str, None),
        font=_get(data, "font", "story", str, None),
        caption=caption,
        source=source,
    )


def load_story(path: str | Path) -> Story:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise StoryError(f"story file not found: {path}") from None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise StoryError(f"{path}: invalid JSON at line {e.lineno}, column {e.colno}: {e.msg}") from None
    return parse_story(data, source=path)
