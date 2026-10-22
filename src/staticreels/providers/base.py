"""The provider interface.

A provider turns a text prompt (and optionally a reference image) into a video
clip. The default "manual" provider doesn't generate anything itself: it tells
you what to make in your AI tool and where to save it.

Real API providers (Kling, Dreamina, Google Flow, ...) are deliberately not
included. Add one only after reading that service's current official API docs;
keep its API key in .env, never in code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class Provider(ABC):
    name = "base"

    @abstractmethod
    def generate(self, prompt: str, aspect: str, duration: float, image: Path | None = None,
                 *, out_name: str | None = None) -> Path:
        """Produce (or ask for) a clip and return the path where it is, or will be, saved.

        prompt:   the text prompt from the story
        aspect:   e.g. "9:16"
        duration: seconds the segment needs (longer clips are trimmed, shorter ones freeze)
        image:    optional reference/first-frame image
        out_name: the filename the story expects, e.g. "clip1.mp4"
        """
