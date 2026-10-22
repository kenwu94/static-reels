"""Turn a story into a finished MP4, and verify the result.

Rendering streams every frame of every segment, in order, into one ffmpeg
encoder that also muxes the finished audio track. One encode, no concat step,
so segment boundaries can't drift out of sync with the sound.
"""

from __future__ import annotations

import io
import json
import math
import struct
import time
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import audio, ff
from .providers import ManualProvider, PlaceholderProvider
from .publish import write_caption_template
from .story import Story
from .visuals import Ctx, FontNotFoundError, load_font, render_segment, resolve_font

# H.264 High / yuv420p with a bitrate cap: full-frame static is close to incompressible,
# and without the cap those segments balloon the file for no visible gain after Instagram re-encodes.
VIDEO_ARGS = ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-maxrate", "14M", "-bufsize", "28M",
              "-pix_fmt", "yuv420p", "-profile:v", "high", "-g", "60"]
AUDIO_ARGS = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
KNOCK_SHAKE = (18, -10, 5)  # vertical pixel offsets for the frames right after each knock


class RenderError(RuntimeError):
    """Rendering can't start or finish (missing clips, voiceover, ...)."""


@dataclass
class RenderResult:
    video: Path
    caption: Path
    caption_created: bool
    sidecar: Path
    seconds: float


def _resolve_voiceover(story: Story, project_root: Path) -> Path | None:
    if not story.voiceover:
        return None
    p = Path(story.voiceover)
    candidates = [p] if p.is_absolute() else [project_root / p] + ([story.source.parent / p] if story.source else [])
    for c in candidates:
        if c.is_file():
            return c
    raise RenderError(f"voiceover file not found: {story.voiceover}")


def load_samples(story: Story, project_root: Path, ffmpeg: str) -> dict[str, np.ndarray]:
    """Decode every recorded sound used by a `sample` cue (paths relative to the project or story)."""
    samples = {}
    for cue in story.cues:
        if cue.type != "sample" or cue.file in samples:
            continue
        p = Path(cue.file)
        candidates = [p] if p.is_absolute() else [project_root / p] + ([story.source.parent / p] if story.source else [])
        found = next((c for c in candidates if c.is_file()), None)
        if found is None:
            raise RenderError(f"sample file not found: {cue.file}")
        samples[cue.file] = audio.decode_mono(ffmpeg, found)
    return samples


def _knock_shake(story: Story) -> dict[int, int]:
    """Frame index -> vertical shake offset, for the frames following each knock."""
    shake = {}
    for cue in story.cues:
        if cue.type == "knock":
            for t in cue.times:
                f = round(t * story.fps)
                for k, dy in enumerate(KNOCK_SHAKE):
                    shake[f + k] = dy
    return shake


CLIP_SOURCES = ("files", "placeholder", "procedural")


def render(story: Story, *, project_root: Path, clips_dir: Path, out_path: Path, clip_source: str = "files",
           font: str | None = None, log: Callable[[str], None] = print) -> RenderResult:
    """clip_source: "files" = your AI clips from clips_dir; "placeholder" = labelled code-made
    stand-ins (not postable); "procedural" = the same code-made scenes as final footage."""
    if clip_source not in CLIP_SOURCES:
        raise ValueError(f"clip_source must be one of {CLIP_SOURCES}")
    started = time.perf_counter()
    ffmpeg = ff.find_tool("ffmpeg")
    font_path = resolve_font(font or story.font, project_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    build = out_path.parent / ".build" / out_path.stem  # intermediates (gitignored with output/)
    build.mkdir(parents=True, exist_ok=True)
    log(f"Font:  {font_path}")

    # 1. Clips: your AI files from clips/, or code-made scenes (labelled placeholders or final procedural).
    if clip_source != "files":
        final = clip_source == "procedural"
        provider = PlaceholderProvider(build / clip_source, ffmpeg, font_path, final=final)
        clips = {}
        for seg in story.ai_segments:
            log(f"{'Procedural' if final else 'Placeholder'} clip: {seg.clip}")
            clips[seg.clip] = provider.generate(seg.prompt, story.aspect, seg.duration, out_name=seg.clip)
    else:
        manual = ManualProvider(clips_dir)
        missing = manual.missing(story)
        if missing:
            story_arg = story.source or "<story.json>"
            raise RenderError(
                f"missing AI clip(s) in {clips_dir}: {', '.join(missing)}\n"
                f"Run `staticreels prompts {story_arg}` to see what to generate, "
                "or add --placeholders to render with code-made stand-ins."
            )
        clips = manual.paths(story)

    # 2. Sound: synthesize the bed, mix in any voiceover, normalize loudness.
    log("Audio: synthesizing and normalizing ...")
    voiceover = _resolve_voiceover(story, project_root)
    samples = load_samples(story, project_root, ffmpeg)
    bed = audio.write_wav(build / "bed.wav", audio.synth_bed(story, samples=samples))
    track = audio.finalize(ffmpeg, bed, build / "audio.wav", story.duration, voiceover)

    # 3. Video: every frame of every segment, streamed into a single encoder.
    log(f"Video: {story.width}x{story.height} @ {story.fps} fps, {story.total_frames} frames")
    ctx = Ctx(story.width, story.height, story.fps, font_path, ffmpeg)
    shake = _knock_shake(story)
    output_args = ["-map", "0:v:0", "-map", "1:a:0", *VIDEO_ARGS, "-r", str(story.fps), *AUDIO_ARGS,
                   "-movflags", "+faststart", "-t", f"{story.duration:.3f}"]
    try:
        with ff.FrameWriter(ffmpeg, out_path, story.width, story.height, story.fps, output_args,
                            extra_inputs=["-i", str(track)]) as writer:
            for seg in story.segments:
                f0, f1 = story.frame_range(seg)
                t0 = time.perf_counter()
                for i, frame in enumerate(render_segment(seg, ctx, f1 - f0, clip_path=clips.get(seg.clip))):
                    dy = shake.get(f0 + i)
                    if dy:
                        frame = np.roll(frame, dy, axis=0)
                    writer.write(frame)
                log(f"  [{seg.index + 1}/{len(story.segments)}] {seg.type:<11} {seg.start:5.1f}-{seg.end:5.1f}s"
                    f"  {f1 - f0:4d} frames  ({time.perf_counter() - t0:.1f}s)")
    except BaseException:
        out_path.unlink(missing_ok=True)  # never leave a half-written video behind
        raise

    # 4. Caption template (never overwrites your edits) and a sidecar describing this render.
    ai_clips = sorted(clips) if clip_source == "files" else []
    caption = out_path.with_suffix(".caption.txt")
    created = write_caption_template(story, caption, ai_clips=ai_clips)
    sidecar = out_path.with_suffix(".render.json")
    sidecar.write_text(json.dumps({
        "story": str(story.source) if story.source else None,
        "rendered_at": datetime.now().isoformat(timespec="seconds"),
        "clip_source": clip_source,
        "placeholders": clip_source == "placeholder",
        "clips": {name: str(p) for name, p in clips.items()},
        "ai_generated_clips": ai_clips,
        "duration": story.duration,
    }, indent=2), encoding="utf-8")
    return RenderResult(out_path, caption, created, sidecar, time.perf_counter() - started)


def split_parts(story: Story, video: Path, out_dir: Path, log: Callable[[str], None] = print) -> list[Path]:
    """Cut a finished render into one MP4 per segment (same encode settings, frame-accurate).

    Cuts use the same rounded frame boundaries as the render, and a 10 ms audio
    fade at each end keeps the cuts from clicking.
    """
    ffmpeg = ff.find_tool("ffmpeg")
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for seg in story.segments:
        f0, f1 = story.frame_range(seg)
        start, dur = f0 / story.fps, (f1 - f0) / story.fps
        out = out_dir / f"{seg.index + 1:02d}_{seg.type}.mp4"
        ff.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.3f}", "-i", str(video),
                "-t", f"{dur:.3f}", "-af", f"afade=t=in:d=0.01,afade=t=out:st={dur - 0.01:.3f}:d=0.01",
                *VIDEO_ARGS, "-r", str(story.fps), *AUDIO_ARGS, "-movflags", "+faststart", str(out)])
        log(f"  {out}  ({seg.start:g}-{seg.end:g}s)")
        parts.append(out)
    return parts


# --------------------------------------------------------------------------- verification


@dataclass
class VideoReport:
    path: Path
    rows: list[tuple[str, str, str]]  # (status, check, detail); status is OK / FAIL / WARN / INFO
    probe: dict

    @property
    def ok(self) -> bool:
        return all(status != "FAIL" for status, _, _ in self.rows)


def top_level_atoms(path: Path) -> list[str]:
    """Names of the top-level MP4 boxes, in file order (used to confirm +faststart)."""
    names = []
    with open(path, "rb") as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            size, kind = struct.unpack(">I4s", header)
            names.append(kind.decode("latin-1"))
            if size == 1:  # 64-bit size follows
                size = struct.unpack(">Q", f.read(8))[0] - 16
            elif size == 0:  # box runs to end of file
                break
            else:
                size -= 8
            f.seek(size, 1)
    return names


def _rate(text: str | None) -> float:
    try:
        return float(Fraction(text))
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def inspect_video(path: str | Path, *, size: tuple[int, int] = (1080, 1920), fps: int = 30,
                  measure_audio: bool = True) -> VideoReport:
    """Check a finished MP4 against what an Instagram Reel upload needs from this pipeline."""
    path = Path(path)
    if not path.is_file():
        raise RenderError(f"video not found: {path}")
    data = ff.probe(ff.find_tool("ffprobe"), path)
    streams, fmt = data.get("streams", []), data.get("format", {})
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    rows: list[tuple[str, str, str]] = []

    def need(name: str, good: bool, detail: str) -> None:
        rows.append(("OK" if good else "FAIL", name, detail))

    if v is None:
        need("video stream", False, "none found")
    else:
        w, h = v.get("width"), v.get("height")
        need("resolution", (w, h) == size, f"{w}x{h} (want {size[0]}x{size[1]})")
        rate = _rate(v.get("avg_frame_rate")) or _rate(v.get("r_frame_rate"))
        need("frame rate", abs(rate - fps) < 0.01, f"{rate:.3f} fps (want {fps})")
        need("video codec", v.get("codec_name") == "h264", f"{v.get('codec_name')} ({v.get('profile')})")
        need("pixel format", v.get("pix_fmt") == "yuv420p", str(v.get("pix_fmt")))
    if a is None:
        need("audio stream", False, "none found")
    else:
        need("audio codec", a.get("codec_name") == "aac",
             f"{a.get('codec_name')}, {a.get('sample_rate')} Hz, {a.get('channels')} ch")
    atoms = top_level_atoms(path)
    faststart = "moov" in atoms and "mdat" in atoms and atoms.index("moov") < atoms.index("mdat")
    need("faststart", faststart, "moov before mdat" if faststart else f"box order: {' '.join(atoms)}")
    duration = float(fmt.get("duration", 0))
    need("duration", duration > 0, f"{duration:.2f} s")
    size_bytes = int(fmt.get("size", path.stat().st_size))
    bitrate = int(fmt.get("bit_rate", 0))
    rows.append(("INFO", "file size", f"{size_bytes / 1e6:.1f} MB, {bitrate / 1e6:.1f} Mbit/s overall"))
    if measure_audio and a is not None:
        m = audio.measure_loudness(ff.find_tool("ffmpeg"), path)
        lufs, tp = float(m["input_i"]), float(m["input_tp"])
        rows.append(("OK" if abs(lufs - audio.TARGET_LUFS) <= 1.0 else "WARN", "loudness",
                     f"{lufs:.1f} LUFS integrated (target {audio.TARGET_LUFS:g})"))
        rows.append(("OK" if tp <= -1.0 else "WARN", "true peak", f"{tp:.1f} dBTP (want <= -1.0)"))
    return VideoReport(path, rows, data)


def contact_sheet(video: str | Path, out_png: str | Path, *, count: int = 6, cols: int = 3,
                  thumb_width: int = 360, project_root: Path | None = None) -> Path:
    """Save a grid of `count` evenly spaced frames (taken from the middle of equal time slices)."""
    video, out_png = Path(video), Path(out_png)
    ffmpeg = ff.find_tool("ffmpeg")
    duration = float(ff.probe(ff.find_tool("ffprobe"), video)["format"]["duration"])
    thumbs = []
    for i in range(count):
        t = duration * (i + 0.5) / count
        proc = ff.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(video),
                       "-frames:v", "1", "-f", "image2pipe", "-c:v", "png", "-"], binary=True)
        img = Image.open(io.BytesIO(proc.stdout)).convert("RGB")
        thumbs.append((t, img.resize((thumb_width, round(thumb_width * img.height / img.width)), Image.LANCZOS)))

    tw, th = thumbs[0][1].size
    gap, label_h, rows = 20, 40, math.ceil(count / cols)
    sheet = Image.new("RGB", (cols * tw + (cols + 1) * gap, rows * (th + label_h) + (rows + 1) * gap), (18, 18, 20))
    draw = ImageDraw.Draw(sheet)
    try:
        font = load_font(str(resolve_font(None, project_root)), 24)
    except FontNotFoundError:
        font = ImageFont.load_default()
    for k, (t, thumb) in enumerate(thumbs):
        r, c = divmod(k, cols)
        x, y = gap + c * (tw + gap), gap + r * (th + label_h + gap)
        sheet.paste(thumb, (x, y))
        draw.text((x, y + th + 8), f"{t:.1f}s", font=font, fill=(200, 200, 200))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    return out_png
