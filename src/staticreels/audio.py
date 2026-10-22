"""Code-made sound design, synthesized with numpy and written to WAV.

Every cue in the story becomes a mono sound placed in a stereo field (`pan`,
so the wall can always be on the listener's left). Then three cues shape the
whole mix over their time range:

- room:    adds room tone, and echo whose length follows `size` (the bedroom
           sounds smaller part by part as the story goes on)
- muffle:  low-passes everything, like hearing it through plaster
- silence: dead silence (also kills room tone), with click-free edges

The result is written as 16-bit 48 kHz stereo. ffmpeg then mixes in an
optional voiceover (ducking the bed under it) and normalizes loudness with a
two-pass loudnorm.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Callable

import numpy as np

from . import ff
from .story import Cue, Story

SR = 48_000
TARGET_LUFS = -14.0
TRUE_PEAK = -2.0  # dBTP ceiling; sharp transients (snaps) overshoot a little in the AAC encode


# --------------------------------------------------------------------------- helpers


def _filter(x: np.ndarray, sr: int, low: float | None = None, high: float | None = None, order: int = 4) -> np.ndarray:
    """Zero-phase band-limit in the frequency domain, with Butterworth-shaped slopes."""
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    gain = np.ones_like(f)
    if high:
        gain /= np.sqrt(1 + (f / high) ** (2 * order))
    if low:
        gain /= np.sqrt(1 + (low / np.maximum(f, 1e-3)) ** (2 * order))
    return np.fft.irfft(spec * gain, n=len(x))


def _fftconv(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Linear convolution via FFT (output length len(x) + len(h) - 1)."""
    n = len(x) + len(h) - 1
    nfft = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(h, nfft), nfft)[:n]


def _fades(n: int, sr: int, fade_in: float, fade_out: float) -> np.ndarray:
    env = np.ones(n)
    a, b = min(n, int(fade_in * sr)), min(n, int(fade_out * sr))
    if a:
        env[:a] = np.linspace(0, 1, a)
    if b:
        env[n - b :] *= np.linspace(1, 0, b)
    return env


def _place(dst: np.ndarray, src: np.ndarray, start: float, sr: int) -> None:
    """Add src into dst starting at `start` seconds, clipped to dst's length (works for mono or stereo)."""
    i0 = int(round(start * sr))
    if i0 >= len(dst):
        return
    m = min(len(src), len(dst) - i0)
    dst[i0 : i0 + m] += src[:m]


def _normalize(x: np.ndarray) -> np.ndarray:
    peak = np.abs(x).max()
    return x / peak if peak > 0 else x


def _rms_norm(x: np.ndarray, target: float = 0.25) -> np.ndarray:
    """Scale noise-like sounds to a common loudness (peak normalizing them would make them too quiet)."""
    rms = np.sqrt(np.mean(np.square(x)))
    return x * (target / rms) if rms > 0 else x


def _pan(mono: np.ndarray, pan: float) -> np.ndarray:
    """Equal-power pan: -1 = left only, 0 = centre, 1 = right only. Returns (n, 2)."""
    theta = (pan + 1) * np.pi / 4
    return np.stack([mono * np.cos(theta), mono * np.sin(theta)], axis=1)


def _resonate(excitation: np.ndarray, sr: int, freqs, decays, gains) -> np.ndarray:
    """Ring a bank of decaying sine resonators (wood, metal, plastic...) with an impulse signal."""
    t = np.arange(int(max(decays) * 6 * sr)) / sr
    ir = sum(g * np.sin(2 * np.pi * f * t) * np.exp(-t / d) for f, d, g in zip(freqs, decays, gains))
    return _fftconv(excitation, ir)[: len(excitation)]


def _pulse_train(sr: int, seconds: float, rate: Callable[[float], float], rng: np.random.Generator,
                 jitter: float = 0.15) -> tuple[np.ndarray, list[float]]:
    """Irregular impulses whose rate follows rate(t): the stick-slip motion behind creaks and springs."""
    n = int(seconds * sr)
    out, times, t = np.zeros(n), [], 0.0
    while t < seconds:
        i = int(t * sr)
        if i < n:
            out[i] = 1.0
            times.append(t)
        t += (1.0 / rate(t)) * rng.uniform(1 - jitter, 1 + jitter)
    return out, times


def _slow_random(rng: np.random.Generator, n: int, sr: int, step: float) -> np.ndarray:
    """A smooth random curve in 0..1 with a new value every `step` seconds."""
    k = max(2, int(n / sr / step) + 2)
    return np.interp(np.arange(n) / sr, np.linspace(0, n / sr, k), rng.random(k))


# --------------------------------------------------------------------------- sustained sounds


def drone(seconds: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Detuned low sines (40-80 Hz) beating against each other, with a slow LFO swell."""
    t = np.arange(int(seconds * sr)) / sr
    voices = [(41.2, 1.0), (41.9, 0.9), (55.0, 0.7), (55.7, 0.6), (73.4, 0.4), (74.3, 0.35)]
    sig = sum(a * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) for f, a in voices)
    sig /= sum(a for _, a in voices)
    lfo = 0.6 + 0.4 * np.sin(2 * np.pi * 0.08 * t + rng.uniform(0, 2 * np.pi))
    # Soft saturation adds harmonics, so the drone is still felt on phone speakers that can't play 40 Hz.
    sig = np.tanh(2.2 * sig * lfo)
    return _normalize(sig) * _fades(len(t), sr, 1.5, 1.5)


def hiss(seconds: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Band-passed noise with sparse crackle: the sound of the wall."""
    n = int(seconds * sr)
    noise = _filter(rng.standard_normal(n), sr, low=1500, high=9000)
    noise /= noise.std() + 1e-9
    crackle = (rng.random(n) < 40 / sr) * rng.standard_normal(n) * 4.0
    return (0.25 * noise + 0.15 * crackle) * _fades(n, sr, 0.03, 0.03)


def grind(seconds: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Plaster grinding: low rumble + dense grit, surging in uneven lurches."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    rumble = _filter(rng.standard_normal(n), sr, low=35, high=420)
    rumble /= rumble.std() + 1e-9
    grit = (rng.random(n) < 1400 / sr) * rng.standard_normal(n)
    grit = _filter(grit, sr, low=700, high=4500)
    grit /= grit.std() + 1e-9
    env = 0.25 + 0.75 * _slow_random(rng, n, sr, 0.18) ** 1.5
    for _ in range(max(1, int(seconds * 0.8))):  # occasional lurches
        env += 1.2 * np.exp(-(((t - rng.uniform(0, seconds)) / 0.12) ** 2))
    return _rms_norm((0.7 * rumble + 0.45 * grit) * env * _fades(n, sr, 0.08, 0.15))


def breathing(seconds: float, sr: int, rng: np.random.Generator, period: float = 4.5, panic: bool = False) -> np.ndarray:
    """Slow, wet breaths: filtered-noise inhales and longer, lower exhales. panic=True: fast and shaky."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    inhale = _filter(rng.standard_normal(n), sr, low=700, high=4200)
    exhale = _filter(rng.standard_normal(n), sr, low=200, high=2200)
    throat = _filter(rng.standard_normal(n), sr, low=80, high=350)
    inhale, exhale, throat = (x / (x.std() + 1e-9) for x in (inhale, exhale, throat))
    env_in, env_out = np.zeros(n), np.zeros(n)
    c = rng.uniform(0, 0.3) * period
    while c < seconds:
        p = period * rng.uniform(0.88, 1.12)
        a, b = int(p * 0.38 * sr), int(p * 0.48 * sr)
        i0 = int(c * sr)
        e0 = i0 + a + int(0.05 * p * sr)
        if i0 < n:
            seg = np.sin(np.pi * np.linspace(0, 1, a) ** 0.8)  # inhale swells, then stops
            env_in[i0 : i0 + a] += seg[: n - i0]
        if e0 < n:
            seg = np.sin(np.pi * np.linspace(0, 1, b) ** 0.6) ** 1.2  # exhale: quick start, long tail
            env_out[e0 : e0 + b] += seg[: n - e0]
        c += p
    sig = 0.55 * inhale * env_in + 0.8 * exhale * env_out + 0.25 * throat * env_out
    if panic:
        sig *= 1 + 0.35 * np.sin(2 * np.pi * 9 * t)  # a shaky, juddering breath
    return _rms_norm(sig * _fades(n, sr, 0.05, 0.03))


def paper(seconds: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Wallpaper crinkling: sparse crackles in slow waves over a faint papery hiss."""
    n = int(seconds * sr)
    density = _slow_random(rng, n, sr, 0.4)
    out = np.zeros(n)
    for _ in range(rng.poisson(seconds * 45)):
        i = int(rng.random() * n)
        if rng.random() > density[i]:
            continue
        m = int(rng.uniform(0.002, 0.018) * sr)
        burst = rng.standard_normal(m) * np.exp(-np.arange(m) / (m / 3)) * rng.uniform(0.2, 1.0)
        out[i : i + m] += burst[: n - i]
    out = _filter(out, sr, low=1200, high=10000)
    bed = _filter(rng.standard_normal(n), sr, low=2500, high=9000) * 0.05 * density
    return _rms_norm(out + bed, 0.2) * _fades(n, sr, 0.05, 0.1)


def tick(seconds: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """A wall clock: tick, tock, once a second."""
    n = int(seconds * sr)
    exc = np.zeros(n)
    for k, t in enumerate(np.arange(0.1, seconds, 1.0)):
        exc[int(t * sr)] = 1.0 if k % 2 == 0 else 0.7
    out = _resonate(exc, sr, [3200, 2450, 5100], [0.006, 0.008, 0.003], [1, 0.6, 0.3])
    return _normalize(_filter(out, sr, low=800))


def riser(seconds: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Shepard tone: octave-spaced sines that seem to rise forever without getting higher."""
    t = np.arange(int(seconds * sr)) / sr
    out, octaves, per_octave = np.zeros(len(t)), 7, 2.2
    for k in range(octaves):
        x = (k + t / per_octave) % octaves
        f = 55 * 2**x
        amp = np.exp(-(((x - octaves / 2) / (octaves / 4)) ** 2))
        out += amp * np.sin(2 * np.pi * np.cumsum(f) / sr)
    env = np.linspace(0.25, 1, len(t)) ** 1.5 * _fades(len(t), sr, 0.3, 0.05)
    return _normalize(out) * env


def drawtones(seconds: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Faint pure tones and glides: what lines drawn in a spectrogram sound like."""
    n = int(seconds * sr)
    out, c = np.zeros(n), 0.0
    while c < seconds:
        length = rng.uniform(0.35, 1.3)
        f0 = float(np.exp(rng.uniform(np.log(900), np.log(6000))))
        f1 = f0 * 2 ** rng.uniform(-1, 1) if rng.random() < 0.6 else f0  # a sweep or a steady line
        m = min(int(length * sr), n - int(c * sr))
        if m > 0:
            x = np.linspace(0, 1, m)
            f = f0 * (f1 / f0) ** x
            out[int(c * sr) : int(c * sr) + m] += np.sin(np.pi * x) ** 0.5 * np.sin(2 * np.pi * np.cumsum(f) / sr) * rng.uniform(0.5, 1)
        c += rng.uniform(0.12, 0.45)
    return _normalize(out) * 0.8 * _fades(n, sr, 0.05, 0.05)


def day(seconds: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Daytime: distant traffic rumble with the odd car passing."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    brown = _filter(np.cumsum(rng.standard_normal(n)), sr, low=25, high=350)
    brown /= brown.std() + 1e-9
    cars = np.zeros(n)
    band = _filter(rng.standard_normal(n), sr, low=150, high=1200)
    band /= band.std() + 1e-9
    for _ in range(int(seconds / 3) + 1):
        cars += np.exp(-(((t - rng.uniform(0, seconds)) / 1.2) ** 2))
    return _rms_norm(0.8 * brown + 0.6 * band * cars) * _fades(n, sr, 0.4, 0.4)


def room_tone(seconds: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """The near-silent low hum of an indoor space."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    tone = _filter(rng.standard_normal(n), sr, low=30, high=260)
    tone /= tone.std() + 1e-9
    return _rms_norm(tone + 0.15 * np.sin(2 * np.pi * 60 * t)) * _fades(n, sr, 0.05, 0.05)


# --------------------------------------------------------------------------- one-shot sounds


def heartbeat(sr: int) -> np.ndarray:
    """One "lub-dub": two pitch-dropping sine thumps 0.21 s apart."""
    t = np.arange(int(0.22 * sr)) / sr
    freq = 42 + 38 * np.exp(-t / 0.025)
    thump = np.sin(2 * np.pi * np.cumsum(freq) / sr) * (1 - np.exp(-t / 0.003)) * np.exp(-t / 0.06)
    out = np.zeros(int(0.5 * sr))
    out[: len(thump)] += thump
    i = int(0.21 * sr)
    out[i : i + len(thump)] += 0.7 * thump
    return _normalize(np.tanh(2.5 * out))


def knock(sr: int, rng: np.random.Generator) -> np.ndarray:
    """A knuckle on a door: low-passed noise burst + a woody body tone + a few room reflections."""
    n = int(0.35 * sr)
    t = np.arange(n) / sr
    pitch = rng.uniform(0.92, 1.08)  # every knock slightly different
    burst = _normalize(_filter(rng.standard_normal(n) * np.exp(-t / 0.006), sr, high=1400 * pitch))
    body = np.sin(2 * np.pi * 118 * pitch * t) * np.exp(-t / 0.028) + 0.6 * np.sin(2 * np.pi * 72 * pitch * t) * np.exp(-t / 0.05)
    hit = 0.7 * burst + body
    out = hit.copy()
    for delay, g in ((0.019, 0.4), (0.043, 0.25), (0.087, 0.14), (0.15, 0.07)):
        d = int(delay * sr)
        out[d:] += g * hit[: n - d]
    return _normalize(np.tanh(1.5 * _filter(out, sr, low=40, high=3500)))


def creak(sr: int, rng: np.random.Generator, seconds: float = 1.8) -> np.ndarray:
    """A long wooden creak: stick-slip pulses speeding up and slowing down, ringing a few wood modes."""
    r0, r1 = rng.uniform(25, 45), rng.uniform(70, 140)
    exc, times = _pulse_train(sr, seconds, lambda t: r0 + (r1 - r0) * np.sin(np.pi * t / seconds), rng)
    for t in times:  # louder in the middle of the creak
        exc[int(t * sr)] *= np.sin(np.pi * t / seconds) ** 0.5 * rng.uniform(0.6, 1.0)
    f = rng.uniform(0.85, 1.15)
    out = _resonate(exc, sr, [260 * f, 455 * f, 790 * f, 1350 * f], [0.03, 0.022, 0.015, 0.008], [1, 0.7, 0.45, 0.25])
    return _normalize(_filter(out, sr, low=120, high=5000))


def bedsprings(sr: int, rng: np.random.Generator) -> np.ndarray:
    """Someone settling onto an old mattress: a soft thump and a burst of ringing springs."""
    seconds = 0.9
    exc, times = _pulse_train(sr, seconds, lambda t: 12 + 45 * np.exp(-t / 0.3), rng, jitter=0.3)
    for t in times:
        exc[int(t * sr)] *= np.exp(-t / 0.35)
    f = rng.uniform(0.95, 1.05)
    metal = _resonate(exc, sr, [640 * f, 1390 * f, 2280 * f, 3170 * f], [0.12, 0.09, 0.06, 0.04], [1, 0.8, 0.6, 0.4])
    t = np.arange(len(exc)) / sr
    body = 0.8 * np.sin(2 * np.pi * 70 * t) * np.exp(-t / 0.07)
    return _normalize(0.6 * _normalize(metal) + body)


def tape(sr: int, rng: np.random.Generator, pull: bool = True) -> np.ndarray:
    """A tape measure: ratcheting pull-out (optional) then the hard snap of it retracting."""
    snap_at = 0.55 if pull else 0.0
    n = int((snap_at + 0.35) * sr)
    out = np.zeros(n)
    if pull:
        exc, _ = _pulse_train(sr, 0.5, lambda t: 130, rng, jitter=0.2)
        exc *= np.linspace(0.4, 1.0, len(exc))
        ratchet = _resonate(exc, sr, [3100, 4700, 6900], [0.004, 0.003, 0.002], [1, 0.7, 0.5])
        whirr = _filter(rng.standard_normal(len(exc)), sr, low=1500, high=6000) * 0.04
        _place(out, _normalize(ratchet) * 0.6 + whirr, 0.02, sr)
    m = int(0.35 * sr)
    t = np.arange(m) / sr
    impulse = np.zeros(m)
    impulse[0] = 1.0
    click = _filter(rng.standard_normal(m) * np.exp(-t / 0.0015), sr, low=1500)
    ring = _resonate(impulse, sr, [2950, 4380, 6120, 1720], [0.05, 0.035, 0.025, 0.06], [1, 0.7, 0.5, 0.6])
    thump = np.sin(2 * np.pi * 95 * t) * np.exp(-t / 0.035)
    _place(out, 0.8 * _normalize(click) + 0.6 * _normalize(ring) + 0.5 * thump, snap_at, sr)
    return _normalize(out)


def beep(sr: int) -> np.ndarray:
    """A phone's start-recording beep."""
    t = np.arange(int(0.09 * sr)) / sr
    note = lambda f: np.sin(2 * np.pi * f * t) * _fades(len(t), sr, 0.005, 0.02)  # noqa: E731
    return np.concatenate([note(1320), np.zeros(int(0.03 * sr)), note(1760)])


def sub(sr: int, rng: np.random.Generator) -> np.ndarray:
    """A deep sub-bass hit that drops in pitch: the 'reveal' boom."""
    t = np.arange(int(1.8 * sr)) / sr
    f = 26 + 24 * np.exp(-t / 0.25)
    body = np.tanh(1.8 * np.sin(2 * np.pi * np.cumsum(f) / sr) * (1 - np.exp(-t / 0.004)) * np.exp(-t / 0.55))
    hit = _filter(rng.standard_normal(len(t)), sr, high=120) * np.exp(-t / 0.05)
    return _normalize(body + 0.5 * _normalize(hit))


def thud(sr: int, rng: np.random.Generator) -> np.ndarray:
    """A soft, dead thud with no ring and no echo, like something closing inside a wall."""
    t = np.arange(int(0.45 * sr)) / sr
    burst = _filter(rng.standard_normal(len(t)) * np.exp(-t / 0.02), sr, high=300)
    return _normalize(_normalize(burst) + np.sin(2 * np.pi * 62 * t) * np.exp(-t / 0.09))


# --------------------------------------------------------------------------- mixing


def _room_ir(size: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """A synthetic room echo: a few early reflections and a noise tail whose length follows size."""
    t = np.arange(int((0.08 + 1.4 * size) * sr)) / sr
    h = rng.standard_normal(len(t)) * np.exp(-t / (0.02 + 0.32 * size))
    h = _filter(h, sr, high=3000 + 5000 * size)
    for delay, g in ((0.007, 0.5), (0.013, 0.35), (0.021, 0.25)):
        h[int(delay * (0.5 + size) * sr)] += g * rng.choice([-1, 1])
    h[0] = 0.0
    return h / np.sqrt(np.sum(h**2))


def _ramp(n: int, cue: Cue) -> np.ndarray:
    """Gain envelope for a range cue: constant, or gain -> gain_to (rises accelerate)."""
    g0, g1 = cue.gain, cue.gain if cue.gain_to is None else cue.gain_to
    x = np.linspace(0, 1, n)
    return g0 + (g1 - g0) * (x**2 if g1 > g0 else x)


def _silence(mute: np.ndarray, start: float, end: float, sr: int, ramp: float = 0.008) -> None:
    """Zero the mix between start and end, with short ramps so it doesn't click."""
    i0, i1, r = int(round(start * sr)), min(len(mute), int(round(end * sr))), int(ramp * sr)
    mute[i0:i1] = 0.0
    a = max(0, i0 - r)
    mute[a:i0] = np.minimum(mute[a:i0], np.linspace(1, 0, i0 - a))
    b = min(len(mute), i1 + r)
    mute[i1:b] = np.minimum(mute[i1:b], np.linspace(0, 1, b - i1))


RANGE_SOUNDS: dict[str, Callable] = {
    "drone": lambda s, sr, rng, cue: drone(s, sr, rng),
    "hiss": lambda s, sr, rng, cue: hiss(s, sr, rng),
    "grind": lambda s, sr, rng, cue: grind(s, sr, rng),
    "breathing": lambda s, sr, rng, cue: breathing(s, sr, rng, cue.period),
    "panic": lambda s, sr, rng, cue: breathing(s, sr, rng, 0.95, panic=True),
    "paper": lambda s, sr, rng, cue: paper(s, sr, rng),
    "tick": lambda s, sr, rng, cue: tick(s, sr, rng),
    "riser": lambda s, sr, rng, cue: riser(s, sr, rng),
    "drawtones": lambda s, sr, rng, cue: drawtones(s, sr, rng),
    "day": lambda s, sr, rng, cue: day(s, sr, rng),
}
POINT_SOUNDS: dict[str, Callable] = {
    "heartbeat": lambda sr, rng: heartbeat(sr),
    "knock": knock,
    "creak": creak,
    "bedsprings": bedsprings,
    "tape": lambda sr, rng: tape(sr, rng, pull=True),
    "snap": lambda sr, rng: tape(sr, rng, pull=False),
    "beep": lambda sr, rng: beep(sr),
    "sub": sub,
    "thud": thud,
}


def synth_bed(story: Story, sr: int = SR, seed: int = 1307, samples: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """Render every audio cue into one (n, 2) float32 stereo track as long as the video, peaking at 0.8.

    samples maps a `sample` cue's file to its decoded mono audio (see assemble.load_samples).
    """
    n = int(round(story.duration * sr))
    dry = np.zeros((n, 2))  # sources that get room echo
    beds = np.zeros((n, 2))  # room tones (no echo on echo)
    rng = np.random.default_rng(seed)
    rooms, muffles, mute = [], [], np.ones(n)
    for cue in story.cues:
        if cue.type == "room":
            rooms.append(cue)
            tone = room_tone(cue.end - cue.start, sr, rng)
            _place(beds, _pan(tone * _ramp(len(tone), cue), 0.0), cue.start, sr)
        elif cue.type == "muffle":
            muffles.append(cue)
        elif cue.type == "silence":
            _silence(mute, cue.start, cue.end, sr)
        elif cue.type in RANGE_SOUNDS:
            sound = RANGE_SOUNDS[cue.type](cue.end - cue.start, sr, rng, cue)
            _place(dry, _pan(sound * _ramp(len(sound), cue), cue.pan), cue.start, sr)
        else:
            for t in cue.times:
                if cue.type == "sample":
                    sound = (samples or {})[cue.file]
                else:
                    sound = POINT_SOUNDS[cue.type](sr, rng)
                _place(dry, _pan(cue.gain * sound, cue.pan), t, sr)

    # Room echo: each room cue reverberates what plays during its range (the tail rings on past the end).
    wet = np.zeros((n, 2))
    for cue in rooms:
        if cue.size <= 0.01:
            continue
        i0, i1 = int(round(cue.start * sr)), int(round(cue.end * sr))
        seg = dry[i0:i1] * _fades(i1 - i0, sr, 0.02, 0.02)[:, None]
        for ch in (0, 1):  # a different echo per ear gives width
            echo = _fftconv(seg[:, ch], _room_ir(cue.size, sr, rng)) * 0.55 * cue.size
            _place(wet[:, ch], echo, cue.start, sr)
    mix = dry + wet + beds

    # Muffle: low-pass the whole mix in range, crossfading in and out.
    for cue in muffles:
        i0, i1 = int(round(cue.start * sr)), int(round(cue.end * sr))
        w = _fades(i1 - i0, sr, 0.04, 0.04)[:, None]
        muffled = np.stack([_filter(mix[i0:i1, ch], sr, high=cue.cutoff, order=2) for ch in (0, 1)], axis=1)
        mix[i0:i1] = mix[i0:i1] * (1 - w) + muffled * w

    mix *= mute[:, None]
    peak = np.abs(mix).max()
    if peak > 0:
        mix *= 0.8 / peak  # headroom only; loudnorm sets the final level
    return mix.astype(np.float32)


def write_wav(path: str | Path, audio: np.ndarray, sr: int = SR) -> Path:
    """Write a float signal (-1..1), mono (n,) or stereo (n, 2), as a stereo 16-bit WAV."""
    path = Path(path)
    if audio.ndim == 1:
        audio = np.repeat(audio[:, None], 2, axis=1)
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.ascontiguousarray(pcm).tobytes())
    return path


# --------------------------------------------------------------------------- ffmpeg stages


def decode_mono(ffmpeg: str, path: str | Path, sr: int = SR) -> np.ndarray:
    """Decode any audio file ffmpeg understands to mono float at sr (used for `sample` cues)."""
    proc = ff.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path), "-vn",
                   "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"], binary=True)
    return np.frombuffer(proc.stdout, dtype=np.float32).astype(np.float64)


def measure_loudness(ffmpeg: str, path: str | Path, target: float = TARGET_LUFS, true_peak: float = TRUE_PEAK) -> dict:
    """Run loudnorm in analysis mode and return its JSON (input_i, input_tp, input_lra, ...)."""
    proc = ff.run([
        ffmpeg, "-hide_banner", "-nostats", "-i", str(path), "-vn",
        "-af", f"loudnorm=I={target}:TP={true_peak}:LRA=11:print_format=json", "-f", "null", "-",
    ])
    text = proc.stderr
    start, end = text.rfind("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ff.FfmpegError("could not find loudnorm's measurement in ffmpeg's output")
    return json.loads(text[start : end + 1])


def finalize(ffmpeg: str, bed_wav: Path, out_wav: Path, duration: float, voiceover: Path | None = None,
             target: float = TARGET_LUFS, true_peak: float = TRUE_PEAK) -> Path:
    """Optionally mix a voiceover over the bed (ducking the bed), then loudness-normalize."""
    src = bed_wav
    if voiceover is not None:
        mixed = out_wav.with_name("mixed.wav")
        graph = (
            "[0:a]aformat=sample_rates=48000:channel_layouts=stereo[bed];"
            "[1:a]aformat=sample_rates=48000:channel_layouts=stereo,asplit=2[key][vo];"
            # The voice drives a compressor on the bed, pushing the bed down while it speaks.
            "[bed][key]sidechaincompress=threshold=0.02:ratio=8:attack=15:release=350[ducked];"
            "[ducked][vo]amix=inputs=2:duration=first:normalize=0[out]"
        )
        ff.run([ffmpeg, "-hide_banner", "-y", "-i", str(bed_wav), "-i", str(voiceover),
                "-filter_complex", graph, "-map", "[out]", "-t", f"{duration:.3f}", "-c:a", "pcm_f32le", str(mixed)])
        src = mixed  # float WAV, so the sum can't clip before normalization

    m = measure_loudness(ffmpeg, src, target, true_peak)
    if m.get("input_i") in (None, "-inf"):  # silent track: nothing to normalize
        af = "anull"
    else:
        af = (
            f"loudnorm=I={target}:TP={true_peak}:LRA=11"
            f":measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}"
            f":measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true"
        )
    ff.run([ffmpeg, "-hide_banner", "-y", "-i", str(src), "-af", af,
            "-ar", str(SR), "-ac", "2", "-c:a", "pcm_s16le", str(out_wav)])
    return out_wav
