"""The default provider: you generate clips yourself; this prints what to make and checks for the files."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Callable

from ..story import Story
from .base import Provider


class ManualProvider(Provider):
    name = "manual"

    def __init__(self, clips_dir: str | Path, out: Callable[[str], None] = print):
        self.clips_dir = Path(clips_dir)
        self.out = out

    def generate(self, prompt: str, aspect: str, duration: float, image: Path | None = None,
                 *, out_name: str | None = None) -> Path:
        """Print the prompt and where to save the result. Doesn't wait for the file."""
        if not out_name:
            raise ValueError("ManualProvider.generate needs out_name (the filename the story expects)")
        path = self.clips_dir / out_name
        self.out(f"  Save as:  {path}")
        self.out(f"  Aspect:   {aspect}    Length: {duration:g}s or longer "
                 "(longer is trimmed, shorter freezes on its last frame)")
        if image:
            self.out(f"  Reference image: {image}")
        self.out("  Prompt:")
        for row in textwrap.wrap(prompt, 76):
            self.out(f"    {row}")
        return path

    def paths(self, story: Story) -> dict[str, Path]:
        return {seg.clip: self.clips_dir / seg.clip for seg in story.ai_segments}

    def missing(self, story: Story) -> list[str]:
        """Clip filenames the story needs that aren't in clips_dir yet (or are empty)."""
        return [name for name, p in self.paths(story).items() if not p.is_file() or p.stat().st_size == 0]
