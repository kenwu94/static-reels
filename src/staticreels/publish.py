"""Publishing, safe by default.

`staticreels publish` is a dry run unless --yes is passed, and even then it
shows the caption and schedule and asks for a typed confirmation. The only
backend is "manual": it never talks to Instagram or any other service, it
prints what to do. A scheduler backend (Ayrshare, Metricool, ...) should only
be added after reading that service's current API docs.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable

from .story import Story

NOTES_MARKER = "### NOTES (not posted) ###"
CAPTION_LIMIT = 2200  # Instagram's caption length limit, in characters


class PublishError(RuntimeError):
    """Publishing was refused or can't proceed."""


# --------------------------------------------------------------------------- caption files


def caption_template(story: Story, ai_clips: list[str] | None = None) -> str:
    """ai_clips: the AI-generated clip names actually used (default: every ai_clip in the story)."""
    cap = story.caption
    hook = cap.get("hook") or story.title
    tags = " ".join(cap.get("hashtags", []))
    if ai_clips is None:
        ai_clips = [s.clip for s in story.ai_segments]
    clips = ", ".join(ai_clips)
    lines = [
        hook,
        "",
        tags,
        "",
        NOTES_MARKER,
        "Everything from the marker line down is ignored by `staticreels publish`.",
        "- The first line is the hook: it's what people see before \"more\", so keep it short.",
    ]
    if clips:
        lines += [
            f"- AI LABEL: this Reel uses AI-generated clips ({clips}).",
            "  Before posting, check Instagram's current AI-labelling rules and turn on the",
            "  AI label in the upload screen where it's required, e.g. if any clip shows",
            "  realistic AI-generated people or realistic-looking events.",
        ]
    else:
        lines.append("- No AI-generated footage: every visual and sound in this Reel was made by code.")
    lines.append("- The Reels UI covers the bottom and right edge of the video: check the preview before sharing.")
    return "\n".join(lines) + "\n"


def write_caption_template(story: Story, path: Path, ai_clips: list[str] | None = None) -> bool:
    """Write the template unless you've edited the file (so your edits survive re-renders).

    A file that is still exactly one of the generated templates is refreshed, so
    switching clip sources keeps the AI-label note accurate.
    """
    text = caption_template(story, ai_clips)
    if path.exists():
        current = path.read_text(encoding="utf-8")
        untouched = current in (caption_template(story), caption_template(story, []))
        if current == text or not untouched:
            return False
    path.write_text(text, encoding="utf-8")
    return True


def read_caption(path: str | Path) -> tuple[str, str]:
    """Return (caption to post, notes). Notes are everything from NOTES_MARKER down."""
    path = Path(path)
    if not path.is_file():
        raise PublishError(f"caption file not found: {path}")
    caption, _, notes = path.read_text(encoding="utf-8").partition(NOTES_MARKER)
    return caption.strip(), notes.strip()


# --------------------------------------------------------------------------- backends


class ManualBackend:
    """Doesn't upload anything: prints the steps to post it yourself."""

    name = "manual"

    def publish(self, video: Path, caption: str, when: datetime | None, out: Callable[[str], None]) -> None:
        out("")
        out("Manual upload: this tool has not sent anything anywhere. To post it:")
        out(f"  1. Get {video.name} onto your phone, or open Meta Business Suite on a computer.")
        out("  2. Instagram app (Professional account): tap +, choose Reel, pick the video.")
        out("     Or in Meta Business Suite, create a reel and upload the file.")
        out("  3. Paste the caption shown above.")
        out("  4. If the AI-label note in your caption file applies, turn the AI label on.")
        if when:
            out(f"  5. Schedule it for {when:%Y-%m-%d %H:%M} (the app and Business Suite both have a schedule option).")
        else:
            out("  5. Share it (or schedule it with the app's or Business Suite's schedule option).")


BACKENDS = {"manual": ManualBackend}


# --------------------------------------------------------------------------- the command


def parse_schedule(text: str | None) -> datetime | None:
    """"2026-10-31 21:00" (local time) -> datetime. Must be in the future."""
    if not text:
        return None
    try:
        when = datetime.fromisoformat(text)
    except ValueError:
        raise PublishError(f"--schedule: expected something like \"2026-10-31 21:00\", got {text!r}") from None
    now = datetime.now(when.tzinfo) if when.tzinfo else datetime.now()
    if when <= now:
        raise PublishError(f"--schedule: {text} is in the past")
    return when


def _read_sidecar(video: Path) -> dict:
    sidecar = video.with_suffix(".render.json")
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def run_publish(video: str | Path, caption_file: str | Path, *, schedule: str | None = None,
                backend: str = "manual", yes: bool = False, ask: Callable[[str], str] = input,
                out: Callable[[str], None] = print) -> int:
    from .assemble import RenderError, inspect_video  # local import: assemble imports this module
    from .ff import FfmpegError

    video = Path(video)
    if not video.is_file():
        raise PublishError(f"video not found: {video}")
    if backend not in BACKENDS:
        raise PublishError(f"unknown backend {backend!r}; available: {', '.join(BACKENDS)}")
    caption, notes = read_caption(caption_file)
    if not caption:
        raise PublishError(f"{caption_file}: the caption (above {NOTES_MARKER}) is empty")
    when = parse_schedule(schedule)

    blockers, warnings = [], []
    meta = _read_sidecar(video)
    if meta.get("placeholders"):
        blockers.append("this video was rendered with --placeholders (code-made stand-ins, not your AI clips)")
    try:
        report = inspect_video(video, measure_audio=False)
        failed = [f"{name}: {detail}" for status, name, detail in report.rows if status == "FAIL"]
        if failed:
            blockers.append("`staticreels check` failed: " + "; ".join(failed))
    except (FfmpegError, RenderError) as e:
        warnings.append(f"couldn't verify the video ({e})")
    if len(caption) > CAPTION_LIMIT:
        blockers.append(f"caption is {len(caption)} characters; Instagram's limit is {CAPTION_LIMIT}")
    hashtags = sum(word.startswith("#") for word in caption.split())

    out(f"Video:    {video}")
    out(f"Backend:  {backend}")
    out(f"Schedule: {when:%Y-%m-%d %H:%M} (local time)" if when else "Schedule: post now")
    out(f"Caption:  {len(caption)} characters, {hashtags} hashtag(s)")
    out("-" * 60)
    out(caption)
    out("-" * 60)
    if notes:
        out("Notes from the caption file (not posted):")
        for line in notes.splitlines():
            out(f"  {line}")
    for w in warnings:
        out(f"WARNING: {w}")
    for b in blockers:
        out(f"BLOCKED: {b}")

    if not yes:
        out("\nDRY RUN: nothing was published." + ("" if blockers else " Re-run with --yes to go ahead."))
        return 0
    if blockers:
        raise PublishError("refusing to publish until the BLOCKED items above are fixed")
    answer = ask("\nPublish this video with the caption and schedule above? Type 'publish' to confirm: ")
    if answer.strip().lower() != "publish":
        out("Cancelled. Nothing was published.")
        return 1
    BACKENDS[backend]().publish(video, caption, when, out)
    return 0
