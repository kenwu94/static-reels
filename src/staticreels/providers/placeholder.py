"""Code-made clips: dark, grainy little scenes picked from keywords in the prompt
(bedroom pan, breathing wallpaper, hallway with a figure).

Two uses:
- placeholders (render --placeholders): labelled "PLACEHOLDER" and only 80% of
  the segment long, so the real clip path (upscale, crop, freeze the last
  frame) is exercised too. Not for posting.
- final footage (render --procedural): no label, full segment length.

Either way they are rendered at 720x1280 and go through the same clip
treatment as AI footage.
"""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .. import ff
from ..visuals import Sprite, load_font
from .base import Provider


def _lowfreq(rng: np.random.Generator, h: int, w: int, cells: tuple[int, int]) -> np.ndarray:
    """Smooth blotchy noise in 0..1: a small random grid scaled up with bicubic interpolation."""
    small = (rng.random(cells) * 255).astype(np.uint8)
    return np.asarray(Image.fromarray(small).resize((w, h), Image.BICUBIC), dtype=np.float32) / 255.0


class _Scene:
    def __init__(self, kind: str, w: int, h: int, rng: np.random.Generator, extras: frozenset = frozenset()):
        self.kind, self.w, self.h, self.rng, self.extras = kind, w, h, rng, extras
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        self.u, self.v = (x - w / 2) / (w / 2), (y - h / 2) / (h / 2)  # -1..1 across the frame
        self.dark_edges = 1 - 0.5 * (self.u**2 + self.v**2) / 2
        self.tex = _lowfreq(rng, h, int(w * 1.6), (14, 22))  # wider than the frame, for panning
        self.blotch = _lowfreq(rng, h, w, (10, 6))

    def frame(self, t: float, length: float) -> np.ndarray:
        rng, u, v, p = self.rng, self.u, self.v, t / length
        flicker = 0.85 + 0.15 * rng.random()
        if rng.random() < 0.04:
            flicker *= 0.45  # an occasional deep dip
        if self.kind == "wallpaper":
            z = 1 + 0.12 * p + 0.015 * np.sin(2 * np.pi * 0.6 * t)  # slow push-in plus a "breath"
            px, py = u / z * 360, v / z * 640
            motif = np.sin(px / 17) * np.sin(py / 17) + 0.6 * np.sin((px + py) / 11) * np.cos((px - py) / 13)
            stripes = 0.5 + 0.5 * np.sin(px / 45)
            paper = 0.45 + 0.25 * np.clip(motif, 0, 1) * stripes + 0.1 * stripes
            peel = np.where(self.blotch > 0.7, 0.35, 1.0)
            lum, tint = paper * peel * 0.3 * self.dark_edges * flicker, (1.0, 0.86, 0.64)
            if "torn" in self.extras:  # a seam splitting open from the inside, widening over the clip
                # Brighter, with softer peeling, so the black tear is the darkest thing in frame.
                lum = paper * np.where(self.blotch > 0.7, 0.7, 1.0) * 0.42 * self.dark_edges * flicker
                edge = np.abs(u / z - 0.05 - 0.03 * np.sin(v / z * 9))
                width = 0.01 + 0.09 * p
                lum = np.where(edge < width, 0.0, lum + 0.25 * np.exp(-(((edge - width) / 0.01) ** 2)) * lum)
                if p > 0.65:  # two faint points of light in the gap, late and dim
                    dx = np.abs(u / z - 0.05) - 0.022  # two glints either side of the tear's centre
                    eyes = np.exp(-(dx**2 + (v / z + 0.1) ** 2) / 0.0001)
                    lum = lum + 0.5 * min(1.0, (p - 0.65) / 0.2) * eyes * (edge < width)
        elif self.kind == "door":  # floor-level view of a closed door; feet appear in the light under it
            z = 1 + 0.1 * p
            # Shifted up so the light under the door sits at ~2/3 height, clear of the Reels caption area.
            uz, vz = u / z, (v + 0.3) / z
            room = (0.02 + 0.03 * self.blotch) * self.dark_edges
            in_door = (np.abs(uz) < 0.55) & (vz > -1.3) & (vz < 0.62)
            panels = 0.035 * ((np.abs(np.abs(uz) - 0.27) < 0.2) & (np.abs(np.abs(vz + 0.35) - 0.45) < 0.38))
            underlit = 1 + 1.5 * np.exp(-np.clip(0.62 - vz, 0, None) / 0.18)  # the door glows faintly from below
            lum = np.where(in_door, (0.1 + panels) * underlit, room)
            handle = 0.3 * np.exp(-(((uz - 0.42) / 0.03) ** 2 + ((vz - 0.0 - 0.006 * np.sin(t * 40) * (p > 0.75)) / 0.017) ** 2))
            gap = (np.abs(uz) < 0.55) & (np.abs(vz - 0.635) < 0.012)
            spill = 0.3 * np.exp(-np.maximum(vz - 0.647, 0) / 0.05) * (vz > 0.647) * np.exp(-((uz / 0.6) ** 6))
            shadow = 1.0
            if p > 0.3:  # two shadows of feet break the line of light, and stay
                feet = (np.abs(np.abs(uz) - 0.14) < 0.06) * min(1.0, (p - 0.3) / 0.15)
                shadow = 1 - 0.9 * feet
            light = (0.9 * gap + spill) * shadow * flicker
            lum, tint = lum + handle + light, (1.0, 0.82, 0.58)
        elif self.kind == "hallway":
            z = 1 + 0.3 * p  # slow zoom down the corridor
            uz, vz = u / z, (v + 0.05) / z
            d = np.maximum(np.abs(uz) / 0.5, np.abs(vz))  # 0 at the vanishing point, ~1 at the walls
            lum = 0.03 + 0.28 * np.clip(d, 0, 1.2) ** 2
            far = d < 0.14
            lum = np.where(far, 0.09, lum)
            if "looking_back" not in self.extras:  # from the far end, we *are* the figure
                figure = np.exp(-((uz / 0.03) ** 4 + ((vz - 0.01) / 0.1) ** 4))
                lum = lum * (1 - 0.85 * figure * far)
            door = 0.25 * np.exp(-(((uz - 0.27) / 0.008) ** 2)) * (np.abs(vz) < 0.28)  # light from a door ajar
            if "looking_back" in self.extras:
                door = door * max(0.0, 1 - 1.4 * p)  # the door swings shut and the light goes
            lum, tint = (lum + door) * flicker, (0.9, 0.92, 1.0)
        else:  # bedroom: pan across a dim room, blue TV glow from off-screen left
            x0 = int(p * (self.tex.shape[1] - self.w))
            base = self.tex[:, x0 : x0 + self.w]
            glow = 0.14 * np.exp(-(((u + 1.1) / 0.6) ** 2)) * flicker
            lum, tint = (0.05 + 0.2 * base) * self.dark_edges + glow, (0.78, 0.86, 1.05)
        rgb = lum[..., None].astype(np.float32) * np.array(tint, dtype=np.float32) * 255
        rgb += rng.standard_normal((self.h, self.w), dtype=np.float32)[..., None] * 10
        return rgb


def _scene_kind(prompt: str) -> tuple[str, frozenset]:
    """Pick a scene, plus optional variations, from keywords in the prompt."""
    p = prompt.lower()
    extras = frozenset(name for key, name in (("tear", "torn"), ("looking back", "looking_back")) if key in p)
    if "wallpaper" in p:
        return "wallpaper", extras
    if "hallway" in p or "corridor" in p:
        return "hallway", extras
    if "door" in p:
        return "door", extras
    return "room", extras


def _dims(aspect: str) -> tuple[int, int]:
    a, b = (int(x) for x in aspect.split(":"))
    if b >= a:
        return 720, round(720 * b / a / 2) * 2
    return round(720 * a / b / 2) * 2, 720


class PlaceholderProvider(Provider):
    name = "placeholder"

    def __init__(self, out_dir: str | Path, ffmpeg: str, font_path: Path, fps: int = 24, *, final: bool = False):
        self.out_dir, self.ffmpeg, self.font_path, self.fps = Path(out_dir), ffmpeg, font_path, fps
        self.final = final  # True: no label, full length (used as real footage)

    def generate(self, prompt: str, aspect: str, duration: float, image: Path | None = None,
                 *, out_name: str | None = None) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        seed = zlib.crc32(prompt.encode("utf-8"))
        out = self.out_dir / (out_name or f"placeholder_{seed:08x}.mp4")
        w, h = _dims(aspect)
        # Placeholders are deliberately short to exercise freeze-last-frame; final clips cover the segment.
        length = duration if self.final else max(1.0, duration * 0.8)
        rng = np.random.default_rng(seed)
        kind, extras = _scene_kind(prompt)
        scene = _Scene(kind, w, h, rng, extras)

        label = None
        if not self.final:
            img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            ImageDraw.Draw(img).text((w / 2, 40), f"PLACEHOLDER  {out.name}", font=load_font(str(self.font_path), 28),
                                     anchor="mt", fill=(255, 255, 255, 170))
            label = Sprite.from_image(img)

        args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p"]
        with ff.FrameWriter(self.ffmpeg, out, w, h, self.fps, args) as writer:
            for i in range(round(length * self.fps)):
                frame = scene.frame(i / self.fps, length)
                if label:
                    label.draw(frame)
                writer.write(np.clip(frame, 0, 255).astype(np.uint8))
        return out
