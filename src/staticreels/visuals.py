"""Code-made visuals: TV static, glitchy typewriter text, countdown, end card, clip treatment.

Frames are float32 RGB arrays shaped (height, width, 3) with values 0-255.
They are converted to uint8 only at the very end (render_segment), and are
generated one at a time and streamed straight into ffmpeg.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import ff
from .story import Line, Overlay, Segment

TEXT_COLOR = (236, 234, 226)
FONT_SIZE = 62

# --------------------------------------------------------------------------- fonts


class FontNotFoundError(RuntimeError):
    """No usable font file could be found or loaded."""


_WIN_FONTS = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
SYSTEM_MONOSPACE = {
    "Windows": [_WIN_FONTS / "consola.ttf", _WIN_FONTS / "lucon.ttf", _WIN_FONTS / "cour.ttf"],
    "Darwin": [
        Path("/System/Library/Fonts/Menlo.ttc"),
        Path("/System/Library/Fonts/Monaco.ttf"),
        Path("/System/Library/Fonts/Supplemental/Courier New.ttf"),
    ],
    "Linux": [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
        Path("/usr/share/fonts/TTF/DejaVuSansMono.ttf"),
        Path("/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"),
    ],
}


def resolve_font(explicit: str | None = None, project_root: Path | None = None) -> Path:
    """Pick the text font.

    Order: explicit (--font or the story's "font") > STATICREELS_FONT > the first
    file in assets/fonts/ > a system monospace font. An explicitly requested font
    that doesn't exist is an error rather than a silent fallback.
    """
    for label, value in (("font setting", explicit), ("STATICREELS_FONT", os.environ.get("STATICREELS_FONT"))):
        if value:
            path = Path(value).expanduser()
            if not path.is_absolute() and project_root is not None:
                path = project_root / path
            if path.is_file():
                return path
            raise FontNotFoundError(f"The {label} points to {value!r}, but no such file exists.")
    if project_root is not None:
        fonts_dir = project_root / "assets" / "fonts"
        bundled = sorted([*fonts_dir.glob("*.ttf"), *fonts_dir.glob("*.otf")])
        if bundled:
            return bundled[0]
    for path in SYSTEM_MONOSPACE.get(platform.system(), []):
        if path.is_file():
            return path
    raise FontNotFoundError(
        "No usable font found. Put a .ttf/.otf file in assets/fonts/, or set STATICREELS_FONT in .env "
        "to the full path of a font file (a monospace font such as DejaVu Sans Mono looks best)."
    )


@lru_cache(maxsize=32)
def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except OSError as e:
        raise FontNotFoundError(f"Could not load font {path}: {e}") from None


# --------------------------------------------------------------------------- building blocks


@dataclass
class Sprite:
    """A pre-rendered RGBA strip covering rows [y0, y0 + rows) of the frame, ready to composite."""

    y0: int
    rgba: np.ndarray  # float32 (rows, width, 4); colour 0-255, alpha 0-1

    @classmethod
    def from_image(cls, img: Image.Image) -> "Sprite | None":
        """Crop a full-frame RGBA image to the rows that contain anything."""
        bbox = img.getbbox()
        if bbox is None:
            return None
        y0, y1 = bbox[1], bbox[3]
        arr = np.asarray(img.crop((0, y0, img.width, y1)), dtype=np.float32)
        arr[..., 3] /= 255.0
        return cls(y0, arr)

    def draw(self, frame: np.ndarray, opacity: float = 1.0) -> None:
        rows = frame[self.y0 : self.y0 + self.rgba.shape[0]]
        a = self.rgba[..., 3:4] * opacity
        rows *= 1.0 - a
        rows += self.rgba[..., :3] * a


class Static:
    """Animated TV static: chunky noise, light scanlines, vignette and a slow rolling bar."""

    TINT = np.array([0.93, 0.97, 1.04], dtype=np.float32)  # slightly cold, like an old CRT

    def __init__(self, width: int, height: int):
        self.w, self.h = width, height
        y = np.linspace(-1, 1, height, dtype=np.float32)[:, None]
        x = np.linspace(-1, 1, width, dtype=np.float32)[None, :]
        r2 = (x**2 + y**2) / 2  # 0 in the centre, 1 in the corners
        self.vignette = (1.0 - 0.6 * r2**1.2).astype(np.float32)
        scan = np.where(np.arange(height) % 3 == 0, 0.72, 1.0).astype(np.float32)[:, None]
        self.shade = self.vignette * scan
        self.rows = np.arange(height, dtype=np.float32)[:, None]

    def frame(self, rng: np.random.Generator, t: float, brightness: float = 1.0) -> np.ndarray:
        # Noise at half resolution, blown up 2x: reads as TV "snow" and compresses a bit better.
        n = rng.integers(0, 256, (self.h // 2 + 1, self.w // 2 + 1), dtype=np.uint8)
        n = n.repeat(2, axis=0).repeat(2, axis=1)[: self.h, : self.w].astype(np.float32)
        bar_y = (t * 420.0) % (self.h + 600) - 300  # a soft bright band rolling down the screen
        bar = 1.0 + 0.35 * np.exp(-(((self.rows - bar_y) / 140.0) ** 2))
        g = n * (self.shade * bar * np.float32(brightness))
        return g[..., None] * self.TINT


def _wrap(text: str, font: ImageFont.FreeTypeFont, max_width: float) -> list[tuple[int, str]]:
    """Greedy word wrap. Returns (offset of the row's first character in text, row text)."""
    rows, start, current, pos = [], 0, "", 0
    for word in text.split(" "):
        candidate = f"{current} {word}" if current else word
        if current and font.getlength(candidate) > max_width:
            rows.append((start, current))
            start, current = pos, word
        else:
            current = candidate
        pos += len(word) + 1
    rows.append((start, current))
    return rows


class TextBlock:
    """A segment's lines, laid out once (so typing never reflows them) and drawn at any time t."""

    def __init__(self, lines: list[Line], font_path: Path, width: int, height: int, *,
                 size: int = FONT_SIZE, center_y: float = 0.5, color: tuple = TEXT_COLOR):
        self.lines, self.w, self.h, self.size, self.color = lines, width, height, size, color
        self.font = load_font(str(font_path), size)
        self.band: tuple[int, int] | None = None
        self.rows: list[tuple[int, int, str, int, int]] = []  # (line index, char offset, text, x, y)
        self._key, self._sprite = None, None
        if not lines:
            return

        ascent, descent = self.font.getmetrics()
        self.row_h = int((ascent + descent) * 1.12)
        gap = int(self.row_h * 0.45)  # extra space between separate lines
        wrapped = [(li, off, txt) for li, line in enumerate(lines) for off, txt in _wrap(line.text, self.font, width * 0.84)]
        total = len(wrapped) * self.row_h + gap * (len(lines) - 1)
        top = y = int(height * center_y - total / 2)
        prev = None
        for li, off, txt in wrapped:
            if prev is not None and li != prev:
                y += gap
            x = int((width - self.font.getlength(txt)) / 2)
            self.rows.append((li, off, txt, x, y))
            y += self.row_h
            prev = li

        pad = 50
        self.band = (max(0, top - pad), min(height, y + pad))
        # A soft dark backdrop behind the text keeps it readable over bright static.
        v = np.linspace(-1, 1, self.band[1] - self.band[0], dtype=np.float32)
        weight = np.clip(1.3 * (1 - v**4), 0, 1)
        self.backdrop = (1.0 - 0.55 * weight)[:, None, None]

    def sprite(self, t: float) -> Sprite | None:
        counts = tuple(line.chars_at(t) for line in self.lines)
        if counts == self._key:
            return self._sprite
        self._key = counts
        if not any(counts):
            self._sprite = None
            return None
        img = Image.new("RGBA", (self.w, self.h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        fill = (*self.color, 255)
        for li, off, txt, x, y in self.rows:
            n = counts[li] - off
            if n <= 0:
                continue
            shown = txt[:n]
            d.text((x, y), shown, font=self.font, fill=fill, stroke_width=4, stroke_fill=(0, 0, 0, 255))
            still_typing = counts[li] < len(self.lines[li].text)
            if still_typing and n <= len(txt):  # block cursor after the last typed character
                cx = x + self.font.getlength(shown) + 4
                d.rectangle([cx, y + self.row_h * 0.12, cx + self.size * 0.5, y + self.row_h * 0.8], fill=(*self.color, 210))
        self._sprite = Sprite.from_image(img)
        return self._sprite

    def draw(self, frame: np.ndarray, t: float) -> None:
        sprite = self.sprite(t)
        if sprite is None:
            return
        y0, y1 = self.band
        frame[y0:y1] *= self.backdrop
        sprite.draw(frame)


class Glitcher:
    """Occasional bursts of RGB split + horizontal jitter. intensity 0 = never, 1 = constantly broken."""

    def __init__(self, rng: np.random.Generator, intensity: float):
        self.rng, self.intensity, self.frames_left = rng, intensity, 0

    def __call__(self, frame: np.ndarray, band: tuple[int, int] | None = None, force: bool = False) -> bool:
        rng = self.rng
        if self.frames_left == 0:
            if not force and (self.intensity <= 0 or rng.random() >= 0.15 * self.intensity):
                return False
            self.frames_left = int(rng.integers(2, 5))  # a burst lasts a few frames
        self.frames_left -= 1

        y0, y1 = band if band else (0, frame.shape[0])
        strength = 0.4 + self.intensity
        region = frame[y0:y1]
        dx = int(rng.integers(4, 10 + int(26 * strength)))
        region[..., 0] = np.roll(region[..., 0], dx, axis=1)  # red one way...
        region[..., 2] = np.roll(region[..., 2], -dx, axis=1)  # ...blue the other
        for _ in range(int(rng.integers(1, 4))):  # a few torn horizontal slices
            ya = int(rng.integers(y0, max(y0 + 1, y1 - 8)))
            hh = int(rng.integers(6, 48))
            shift = int(rng.choice([-1, 1]) * rng.integers(10, 30 + int(80 * strength)))
            frame[ya : ya + hh] = np.roll(frame[ya : ya + hh], shift, axis=1)
        return True


class RecOverlay:
    """Found-footage camera overlay: viewfinder corners, blinking REC dot, timestamp.

    Kept out of the bottom ~17% and away from the right edge where the Reels UI sits.
    """

    def __init__(self, width: int, height: int, font_path: Path, overlay: Overlay):
        font = load_font(str(font_path), 50)
        white = (240, 240, 235, 215)
        left, right, top, bottom, arm, thick = 60, width - 60, 130, height - 330, 80, 5
        row = top + 75

        def build(dot_on: bool) -> Sprite:
            img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            for x, y, sx, sy in ((left, top, 1, 1), (right, top, -1, 1), (left, bottom, 1, -1), (right, bottom, -1, -1)):
                d.line([(x, y + sy * arm), (x, y), (x + sx * arm, y)], fill=white, width=thick)
            if overlay.rec_dot:
                if dot_on:
                    d.ellipse([left + 30, row - 17, left + 64, row + 17], fill=(230, 30, 30, 255))
                d.text((left + 80, row), "REC", font=font, anchor="lm", fill=white)
            if overlay.timestamp:
                d.text((right - 30, row), overlay.timestamp, font=font, anchor="rm", fill=white)
            return Sprite.from_image(img)

        self.on, self.off = build(True), build(False)

    def draw(self, frame: np.ndarray, t: float) -> None:
        (self.on if (t % 1.0) < 0.6 else self.off).draw(frame)  # blink once per second


# --------------------------------------------------------------------------- segment renderers


@dataclass
class Ctx:
    """Everything the segment renderers share."""

    width: int
    height: int
    fps: int
    font_path: Path
    ffmpeg: str | None = None
    seed: int = 307
    static: Static = field(init=False)

    def __post_init__(self):
        self.static = Static(self.width, self.height)

    def rng(self, salt: int) -> np.random.Generator:
        """A deterministic generator per segment, so re-renders look identical."""
        return np.random.default_rng([self.seed, salt])


def render_segment(seg: Segment, ctx: Ctx, n_frames: int, clip_path: Path | None = None) -> Iterator[np.ndarray]:
    """Yield exactly n_frames uint8 RGB frames for one segment, with fades applied."""
    makers = {"static_text": _static_text, "countdown": _countdown, "end_card": _end_card, "ai_clip": _ai_clip,
              "spectrogram": _spectrogram, "blueprint": _blueprint}
    frames = makers[seg.type](seg, ctx, n_frames, ctx.rng(seg.index), clip_path)
    try:
        for i, frame in zip(range(n_frames), frames):
            t = i / ctx.fps
            fade = 1.0
            if seg.fade_in > 0:
                fade = min(fade, t / seg.fade_in)
            if seg.fade_out > 0:
                fade = min(fade, (seg.duration - t) / seg.fade_out)
            if fade < 1.0:
                frame *= max(fade, 0.0)
            np.clip(frame, 0, 255, out=frame)
            yield frame.astype(np.uint8)
    finally:
        frames.close()


def _static_text(seg, ctx, n, rng, clip_path):
    text = TextBlock(seg.lines, ctx.font_path, ctx.width, ctx.height)
    glitch = Glitcher(rng, seg.glitch)
    for i in range(n):
        t = i / ctx.fps
        if seg.background == "black":
            frame = np.zeros((ctx.height, ctx.width, 3), dtype=np.float32)
        elif seg.background == "dark":  # near-black with a faint crawl of noise
            frame = ctx.static.frame(rng, t, brightness=0.17)
        else:
            frame = ctx.static.frame(rng, t, brightness=0.6 + 0.05 * rng.standard_normal())
        text.draw(frame, t)
        glitch(frame, text.band)
        yield frame


def _countdown(seg, ctx, n, rng, clip_path):
    w, h = ctx.width, ctx.height
    text = TextBlock(seg.lines, ctx.font_path, w, h, center_y=0.70)
    glitch = Glitcher(rng, seg.glitch)
    size = int(h * 0.36)
    widest = max(load_font(str(ctx.font_path), size).getlength(d) for d in seg.digits)
    if widest > 0.84 * w:  # text like 2'6" is wider than a single digit: shrink to fit
        size = int(size * 0.84 * w / widest)
    font = load_font(str(ctx.font_path), size)
    sprites = {}
    for d in set(seg.digits):
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(img).text((w / 2, h * 0.42), d, font=font, anchor="mm",
                                 fill=(245, 245, 240, 255), stroke_width=8, stroke_fill=(0, 0, 0, 255))
        sprites[d] = Sprite.from_image(img)
    slot = seg.duration / len(seg.digits)
    for i in range(n):
        t = i / ctx.fps
        k = min(int(t / slot), len(seg.digits) - 1)
        in_slot = t - k * slot
        flash = in_slot < 2.0 / ctx.fps  # static flares on each new digit
        frame = ctx.static.frame(rng, t, brightness=1.35 if flash else 0.55)
        if in_slot < slot * 0.7 and rng.random() > 0.12:  # digit flashes on, flickers, then drops out
            sprites[seg.digits[k]].draw(frame)
        text.draw(frame, t)
        glitch(frame, None, force=flash)
        yield frame


def _end_card(seg, ctx, n, rng, clip_path):
    w, h = ctx.width, ctx.height
    text = TextBlock(seg.lines, ctx.font_path, w, h, center_y=0.42)
    handle = None
    if seg.handle:
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(img).text((w / 2, h * 0.64), seg.handle, font=load_font(str(ctx.font_path), 46),
                                 anchor="mm", fill=(200, 200, 196, 255))
        handle = Sprite.from_image(img)
    glitch = Glitcher(rng, seg.glitch)
    for i in range(n):
        t = i / ctx.fps
        frame = ctx.static.frame(rng, t, brightness=0.13)  # near-black with a faint crawl of noise
        frame *= rng.uniform(0.9, 1.0)
        text.draw(frame, t)
        if handle and t >= seg.handle_start:
            handle.draw(frame, opacity=min(1.0, (t - seg.handle_start) / 0.5))
        glitch(frame, text.band)
        yield frame


def clip_decode_cmd(ffmpeg: str, clip: Path, width: int, height: int, fps: int, duration: float) -> list[str]:
    """ffmpeg command that turns any clip into exactly-sized raw frames for one segment.

    Scale up to cover the frame, centre-crop, conform the frame rate, pad by
    cloning the last frame (so short clips freeze instead of ending early),
    then trim; plus a slight desaturation and contrast boost. Grain and
    overlays are added afterwards in numpy.
    """
    vf = ",".join([
        f"scale={width}:{height}:force_original_aspect_ratio=increase",
        f"crop={width}:{height}",
        f"fps={fps}",
        f"tpad=stop_mode=clone:stop_duration={duration:.3f}",
        "eq=saturation=0.72:contrast=1.15:brightness=-0.03",
        "format=rgb24",
    ])
    return [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(clip), "-an", "-sn", "-vf", vf,
            "-t", f"{duration:.3f}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]


def _ai_clip(seg, ctx, n, rng, clip_path):
    if clip_path is None or ctx.ffmpeg is None:
        raise ValueError(f"{seg.label}: needs a clip file and ffmpeg")
    w, h = ctx.width, ctx.height
    rec = RecOverlay(w, h, ctx.font_path, seg.overlay) if (seg.overlay.rec_dot or seg.overlay.timestamp) else None
    text = TextBlock(seg.lines, ctx.font_path, w, h) if seg.lines else None
    vignette = ctx.static.vignette[..., None]
    source = ff.read_frames(clip_decode_cmd(ctx.ffmpeg, clip_path, w, h, ctx.fps, seg.duration), w, h)
    last = None
    try:
        for i in range(n):
            t = i / ctx.fps
            src = next(source, None)
            if src is None:  # decoder ran short (rounding): hold the last frame
                if last is None:
                    raise ff.FfmpegError(f"ffmpeg produced no frames from {clip_path}")
                src = last
            last = src
            if seg.cut_to_black and t >= seg.duration - seg.cut_to_black:
                yield np.zeros((h, w, 3), dtype=np.float32)  # hard cut to black
                continue
            frame = src.astype(np.float32)
            frame *= vignette * np.float32(rng.uniform(0.93, 1.04))  # vignette + light flicker
            frame += rng.standard_normal((h, w), dtype=np.float32)[..., None] * 13.0  # film grain
            if i < 3:  # a burst of static on the cut in
                mix = (0.75, 0.45, 0.2)[i]
                frame = frame * (1 - mix) + ctx.static.frame(rng, t) * mix
            if rec:
                rec.draw(frame, t)
            if text:
                text.draw(frame, t)
            yield frame
    finally:
        source.close()


# --------------------------------------------------------------------------- spectrogram & blueprint

# "Inferno"-style colour ramp (black -> purple -> orange -> pale yellow), as spectrogram apps use.
_RAMP = np.array([[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66],
                  [245, 125, 21], [250, 193, 39], [252, 255, 164]], dtype=np.float32)
_LUT = np.stack([np.interp(np.linspace(0, 1, 256), np.linspace(0, 1, len(_RAMP)), _RAMP[:, c]) for c in range(3)],
                axis=-1).astype(np.float32)

_HANDWRITING = [_WIN_FONTS / "Inkfree.ttf", _WIN_FONTS / "segoepr.ttf",
                Path("/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf")]


def _spectrogram(seg, ctx, n, rng, clip_path):
    """A spectrogram scrolling in left to right; the 'noise' resolves into a floor plan."""
    from .plans import draw_plan  # local import: plans uses this module's font loader

    w, h = ctx.width, ctx.height
    W, H = int(w * 0.86) // 2 * 2, int(h * 0.40) // 2 * 2
    x0, y0 = (w - W) // 2, int(h * 0.13)
    mask, _ = draw_plan(W, H, seg.plan, ctx.font_path)
    lines = np.asarray(mask.filter(ImageFilter.GaussianBlur(1.0)), dtype=np.float32) / 255.0

    # Background "sound": brighter toward the bottom (low frequencies), grainy over time.
    base = (0.12 + 0.35 * np.linspace(0, 1, H, dtype=np.float32) ** 1.5)[:, None]
    grain = rng.random((H // 2 + 1, W // 2 + 1)).astype(np.float32).repeat(2, 0).repeat(2, 1)[:H, :W]
    value = np.clip(base * (0.4 + 0.6 * grain) + 0.95 * lines, 0, 1)
    full = _LUT[(value * 255).astype(np.uint8)]

    # Axis labels and a file name, like the app would show.
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    small = load_font(str(ctx.font_path), 22)
    grey = (150, 150, 150, 255)
    d.text((x0, y0 - 18), "rec_0307.m4a   spectrogram", font=small, anchor="lb", fill=grey)
    for k, label in enumerate(("20k", "15k", "10k", "5k", "0 Hz")):
        d.text((x0 + 6, y0 + 8 + k * (H - 30) / 4), label, font=small, anchor="lt" if k < 4 else "lb", fill=grey)
    for k, label in enumerate(("0:00", "0:30", "1:00")):
        d.text((x0 + k * W / 2, y0 + H + 10), label, font=small, anchor=("lt", "mt", "rt")[k], fill=grey)
    labels = Sprite.from_image(img)

    text = TextBlock(seg.lines, ctx.font_path, w, h, size=54, center_y=0.71)
    glitch = Glitcher(rng, seg.glitch)
    for i in range(n):
        t = i / ctx.fps
        frame = ctx.static.frame(rng, t, brightness=0.15)
        box = frame[y0 : y0 + H, x0 : x0 + W]
        c = int(W * min(1.0, t / seg.reveal))
        box[:] = 4.0
        box[:, :c] = full[:, :c] * np.float32(rng.uniform(0.93, 1.03))
        if c < W:
            box[:, c : c + 3] = 235.0  # playhead
        labels.draw(frame)
        text.draw(frame, t)
        glitch(frame, (y0, y0 + H))
        yield frame


def _blueprint(seg, ctx, n, rng, clip_path):
    """An old floor plan in brown ink on aged paper, slowly pushing in, with an optional pencil note."""
    from .plans import draw_plan

    w, h = ctx.width, ctx.height
    W, H = int(w * 0.86) // 2 * 2, int(h * 0.46) // 2 * 2
    x0, y0 = (w - W) // 2, int(h * 0.11)
    mask, anchor = draw_plan(W, H, seg.plan, ctx.font_path)
    ink = np.asarray(mask.filter(ImageFilter.GaussianBlur(0.6)), dtype=np.float32)[..., None] / 255.0

    # Paper: cream, blotchy, foxed at the edges, folded in quarters.
    small = rng.random((9, 8)).astype(np.float32)
    blotch = np.asarray(Image.fromarray((small * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC), np.float32) / 255
    yy = np.linspace(-1, 1, H, dtype=np.float32)[:, None]
    xx = np.linspace(-1, 1, W, dtype=np.float32)[None, :]
    edge = 1 - 0.35 * np.clip(np.maximum(np.abs(xx), np.abs(yy)) - 0.75, 0, 1) / 0.25
    shade = (0.86 + 0.14 * blotch) * edge
    for frac, axis in ((0.5, 0), (0.5, 1)):
        k = int((H if axis == 0 else W) * frac)
        if axis == 0:
            shade[k - 1 : k + 1, :] *= 0.82
            shade[k + 1 : k + 3, :] *= 1.06
        else:
            shade[:, k - 1 : k + 1] *= 0.82
            shade[:, k + 1 : k + 3] *= 1.06
    paper = shade[..., None] * np.array([226, 214, 184], np.float32)
    paper = paper * (1 - 0.9 * ink) + np.array([52, 40, 28], np.float32) * 0.9 * ink

    title = Image.new("L", (W, H), 0)
    ImageDraw.Draw(title).text((W - 30, H - 22), "APT. 4B  -  FLOOR PLAN  -  1926", font=load_font(str(ctx.font_path), 20),
                               anchor="rb", fill=255)
    t_mask = np.asarray(title, np.float32)[..., None] / 255
    paper = paper * (1 - 0.8 * t_mask) + np.array([52, 40, 28], np.float32) * 0.8 * t_mask

    # Pencil note: grey, handwritten, slightly crooked, with a line pointing at the outline.
    pencil = None
    if seg.pencil:
        hand = next((p for p in _HANDWRITING if p.is_file()), ctx.font_path)
        note = Image.new("L", (W, H), 0)
        nd = ImageDraw.Draw(note)
        tx, ty = anchor[0] + 0.22 * W, anchor[1] - 0.02 * H
        nd.text((tx, ty), seg.pencil, font=load_font(str(hand), int(H * 0.07)), anchor="lm", fill=255)
        nd.line([(tx - 12, ty + 4), (anchor[0] + 0.04 * W, anchor[1] + 2)], fill=200, width=3)
        note = note.rotate(-4, center=(tx, ty), resample=Image.BICUBIC)
        pencil = np.asarray(note, np.float32)[..., None] / 255.0

    text = TextBlock(seg.lines, ctx.font_path, w, h, size=54, center_y=0.71)
    for i in range(n):
        t = i / ctx.fps
        sheet = paper
        if pencil is not None and t >= seg.pencil_start:
            o = min(1.0, (t - seg.pencil_start) / 0.6) * 0.8
            sheet = paper * (1 - o * pencil) + np.float32(88) * o * pencil
        s = 1 + 0.05 * t / seg.duration  # slow push-in
        cw, ch = W / s, H / s
        img = Image.fromarray(np.clip(sheet, 0, 255).astype(np.uint8))
        img = img.resize((W, H), Image.BILINEAR, box=((W - cw) / 2, (H - ch) / 2, (W + cw) / 2, (H + ch) / 2))
        frame = ctx.static.frame(rng, t, brightness=0.1)
        frame[y0 : y0 + H, x0 : x0 + W] = np.asarray(img, np.float32)
        frame[y0 : y0 + H, x0 : x0 + W] += rng.standard_normal((H, W), dtype=np.float32)[..., None] * 7  # grain
        text.draw(frame, t)
        yield frame
