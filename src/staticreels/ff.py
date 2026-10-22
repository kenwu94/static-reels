"""Thin helpers around the ffmpeg / ffprobe command-line tools.

Everything goes through subprocess; no video libraries. Raw RGB frames are
streamed through pipes so nothing intermediate has to be written to disk.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np


class FfmpegError(RuntimeError):
    """ffmpeg/ffprobe is missing or a command failed."""


def install_hint() -> str:
    """How to install ffmpeg on the current OS."""
    system = platform.system()
    if system == "Windows":
        return (
            "Install ffmpeg on Windows with:\n"
            "  winget install --id Gyan.FFmpeg -e\n"
            "or download a 'release essentials' zip from https://www.gyan.dev/ffmpeg/builds/,\n"
            "unzip it (e.g. to C:\\ffmpeg) and add C:\\ffmpeg\\bin to your PATH.\n"
            "Then open a NEW terminal and run: ffmpeg -version\n"
            "(Or set STATICREELS_FFMPEG / STATICREELS_FFPROBE in .env to the .exe paths.)"
        )
    if system == "Darwin":
        return "Install ffmpeg on macOS with Homebrew:\n  brew install ffmpeg"
    return (
        "Install ffmpeg with your distro's package manager, for example:\n"
        "  Debian/Ubuntu: sudo apt install ffmpeg\n"
        "  Fedora:        sudo dnf install ffmpeg\n"
        "  Arch:          sudo pacman -S ffmpeg"
    )


def find_tool(name: str) -> str:
    """Locate "ffmpeg" or "ffprobe": STATICREELS_FFMPEG / STATICREELS_FFPROBE first, then PATH."""
    var = f"STATICREELS_{name.upper()}"
    override = os.environ.get(var)
    if override:
        found = shutil.which(override) or (override if Path(override).is_file() else None)
        if not found:
            raise FfmpegError(f"{var} is set to {override!r}, but no such program exists.")
        return found
    found = shutil.which(name)
    if not found:
        raise FfmpegError(f"{name} was not found on PATH.\n{install_hint()}")
    return found


def _tail(text: str, lines: int = 15) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])


def run(cmd: list[str], *, binary: bool = False) -> subprocess.CompletedProcess:
    """Run a command to completion; raise FfmpegError with the end of stderr if it fails."""
    if binary:
        proc = subprocess.run(cmd, capture_output=True, stdin=subprocess.DEVNULL)
        err = proc.stderr.decode("utf-8", "replace")
    else:
        proc = subprocess.run(cmd, capture_output=True, stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace")
        err = proc.stderr
    if proc.returncode != 0:
        raise FfmpegError(f"{Path(cmd[0]).name} failed (exit {proc.returncode}):\n{_tail(err)}")
    return proc


def probe(ffprobe: str, path: str | Path) -> dict:
    """ffprobe's JSON description of a media file (format + streams)."""
    proc = run([ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)])
    return json.loads(proc.stdout)


class FrameWriter:
    """Pipe raw RGB frames into an ffmpeg encoder process.

    Use as a context manager: on a clean exit the encoder is finished and
    checked; on an exception it is killed so no half-written file is mistaken
    for a finished one.
    """

    def __init__(self, ffmpeg: str, out_path: str | Path, width: int, height: int, fps: int,
                 output_args: list[str], extra_inputs: list[str] = ()):
        self._shape = (height, width, 3)
        self._log = tempfile.TemporaryFile()
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-framerate", str(fps), "-i", "-",
            *extra_inputs, *output_args, str(out_path),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._log)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != self._shape or frame.dtype != np.uint8:
            raise ValueError(f"expected a uint8 frame shaped {self._shape}, got {frame.dtype} {frame.shape}")
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except OSError:  # BrokenPipeError, or EINVAL on Windows: the encoder died
            self._proc.wait()
            raise FfmpegError(f"ffmpeg encoder stopped unexpectedly:\n{self._read_log()}") from None

    def _read_log(self) -> str:
        self._log.seek(0)
        return _tail(self._log.read().decode("utf-8", "replace"))

    def close(self) -> None:
        self._proc.stdin.close()
        if self._proc.wait() != 0:
            raise FfmpegError(f"ffmpeg encoder failed:\n{self._read_log()}")

    def abort(self) -> None:
        if self._proc.poll() is None:
            self._proc.kill()
        self._proc.wait()

    def __enter__(self) -> "FrameWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if exc_type is None:
                self.close()
            else:
                self.abort()
        finally:
            self._log.close()


def read_frames(cmd: list[str], width: int, height: int) -> Iterator[np.ndarray]:
    """Run an ffmpeg command that writes rawvideo rgb24 to stdout and yield its frames.

    Stopping early (closing the generator) kills ffmpeg; that is not an error.
    """
    size = width * height * 3
    log = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, stdin=subprocess.DEVNULL)
    finished = False
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
        finished = True
    finally:
        if not finished and proc.poll() is None:
            proc.kill()
        proc.stdout.close()
        rc = proc.wait()
        log.seek(0)
        err = log.read().decode("utf-8", "replace")
        log.close()
    if rc != 0:
        raise FfmpegError(f"ffmpeg failed while decoding (exit {rc}):\n{_tail(err)}")
