"""Command-line entry point.

    staticreels prompts STORY             print the AI clip prompts and which clips are still missing
    staticreels render  STORY [--placeholders | --procedural] [--parts]
    staticreels check   VIDEO             ffprobe the result against the Reel spec
    staticreels sheet   VIDEO             contact-sheet PNG of evenly spaced frames
    staticreels publish VIDEO --caption-file FILE [--schedule ...] [--yes]

Run it from the project folder: clips/, output/, assets/ and .env are found
relative to the current directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from .assemble import RenderError, contact_sheet, inspect_video, render, split_parts
from .ff import FfmpegError
from .providers import ManualProvider
from .publish import PublishError, run_publish
from .story import StoryError, load_story
from .visuals import FontNotFoundError


def cmd_prompts(args) -> int:
    story = load_story(args.story)
    provider = ManualProvider(args.clips_dir)
    print(f"{story.title}: {len(story.ai_segments)} AI clip(s) to generate, aspect {story.aspect}\n")
    for seg in story.ai_segments:
        print(f"== {seg.clip}   ({seg.label})")
        provider.generate(seg.prompt, story.aspect, seg.duration, out_name=seg.clip)
        print()
    missing = provider.missing(story)
    if missing:
        print(f"Still missing in {args.clips_dir}/: {', '.join(missing)}")
    else:
        print(f"All clips present in {args.clips_dir}/. Next: staticreels render {args.story}")
    return 0


def cmd_render(args) -> int:
    story = load_story(args.story)
    out = Path(args.out) if args.out else Path("output") / f"{Path(args.story).stem}.mp4"
    source = "placeholder" if args.placeholders else "procedural" if args.procedural else "files"
    note = {"placeholder": "  [PLACEHOLDER clips]", "procedural": "  [procedural clips]", "files": ""}[source]
    print(f"Rendering {story.title!r}: {story.duration:g}s, {len(story.segments)} segments{note}")
    result = render(story, project_root=Path.cwd(), clips_dir=Path(args.clips_dir), out_path=out,
                    clip_source=source, font=args.font)
    print(f"\nDone in {result.seconds:.0f}s: {result.video}")
    print(f"Caption template: {result.caption}" + ("" if result.caption_created else " (kept your existing file)"))
    if args.parts:
        print("Parts:")
        split_parts(story, result.video, out.with_name(f"{out.stem}_parts"))
    print(f"Next: staticreels check {result.video}")
    return 0


def cmd_check(args) -> int:
    report = inspect_video(args.video, measure_audio=not args.no_loudness)
    print(f"{report.path}")
    for status, name, detail in report.rows:
        print(f"  [{status:<4}] {name:<13} {detail}")
    if args.raw:
        import json
        print(json.dumps(report.probe, indent=2))
    print("\nPASS: ready to upload as a Reel." if report.ok else "\nFAIL: see the FAIL rows above.")
    return 0 if report.ok else 1


def cmd_sheet(args) -> int:
    video = Path(args.video)
    out = Path(args.out) if args.out else video.with_name(f"{video.stem}_contact.png")
    print(contact_sheet(video, out, count=args.frames, project_root=Path.cwd()))
    return 0


def cmd_publish(args) -> int:
    return run_publish(args.video, args.caption_file, schedule=args.schedule, backend=args.backend, yes=args.yes)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="staticreels", description="Story JSON + AI clips -> vertical horror Reels.")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("prompts", help="print the AI clip prompts for a story")
    s.add_argument("story")
    s.add_argument("--clips-dir", default="clips")
    s.set_defaults(func=cmd_prompts)

    s = sub.add_parser("render", help="build output/<story>.mp4")
    s.add_argument("story")
    clips = s.add_mutually_exclusive_group()
    clips.add_argument("--placeholders", action="store_true",
                       help="labelled code-made stand-ins instead of clips/ (for testing; can't be published)")
    clips.add_argument("--procedural", action="store_true",
                       help="use the code-made scenes as the final footage (no AI clips needed)")
    s.add_argument("--parts", action="store_true", help="also save each segment as its own MP4")
    s.add_argument("--clips-dir", default="clips")
    s.add_argument("--out", help="output path (default: output/<story name>.mp4)")
    s.add_argument("--font", help="font file for on-screen text (overrides story and .env)")
    s.set_defaults(func=cmd_render)

    s = sub.add_parser("check", help="verify a rendered MP4 with ffprobe")
    s.add_argument("video")
    s.add_argument("--raw", action="store_true", help="also print ffprobe's full JSON")
    s.add_argument("--no-loudness", action="store_true", help="skip the loudness measurement")
    s.set_defaults(func=cmd_check)

    s = sub.add_parser("sheet", help="save a contact-sheet PNG of evenly spaced frames")
    s.add_argument("video")
    s.add_argument("--frames", type=int, default=6)
    s.add_argument("--out", help="PNG path (default: next to the video)")
    s.set_defaults(func=cmd_sheet)

    s = sub.add_parser("publish", help="publish a Reel (dry run unless --yes)")
    s.add_argument("video")
    s.add_argument("--caption-file", required=True)
    s.add_argument("--schedule", help='local date/time, e.g. "2026-10-31 21:00"')
    s.add_argument("--backend", default="manual")
    s.add_argument("--yes", action="store_true", help="actually go ahead (still asks you to confirm)")
    s.set_defaults(func=cmd_publish)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # don't crash on non-ASCII captions in old Windows consoles
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv(Path.cwd() / ".env")
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (StoryError, FfmpegError, FontNotFoundError, RenderError, PublishError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
