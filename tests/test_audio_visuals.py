"""Smoke tests for the synthesized audio and code-made visuals (no ffmpeg needed)."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from staticreels import audio  # noqa: E402
from staticreels.story import POINT_CUES, RANGE_CUES, Line, load_story, parse_story  # noqa: E402
from staticreels.visuals import Ctx, FontNotFoundError, Static, TextBlock, render_segment, resolve_font  # noqa: E402

SEED = ROOT / "tests" / "fixtures" / "sample_story.json"


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


class AudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.story = load_story(SEED)
        cls.bed = audio.synth_bed(cls.story)

    def window(self, start, end):
        return self.bed[int(start * audio.SR) : int(end * audio.SR)]

    def test_length_and_range(self):
        self.assertEqual(self.bed.shape, (30 * audio.SR, 2))
        self.assertTrue(np.all(np.isfinite(self.bed)))
        self.assertLessEqual(np.abs(self.bed).max(), 0.8 + 1e-6)

    def test_silence_cue_is_silent(self):
        self.assertLess(np.abs(self.window(25.6, 26.3)).max(), 1e-6)

    def test_knocks_stand_out(self):
        knock = rms(self.window(27.2, 27.3))
        before = rms(self.window(26.8, 27.15))
        self.assertGreater(knock, 2 * before)

    def test_hiss_rises_through_the_countdown(self):
        # Compare high-frequency energy (where the hiss lives) early vs late in the countdown.
        def hf(x):
            return rms(np.diff(x, axis=0))
        self.assertGreater(hf(self.window(20.4, 20.9)), 2 * hf(self.window(17.1, 17.6)))


def tiny_story(cues, duration=2):
    return parse_story({"title": "t", "segments": [{"type": "end_card", "duration": duration}], "audio_cues": cues})


class NewSoundTests(unittest.TestCase):
    def test_every_sound_type_renders(self):
        cues = [{"type": k, "start": 0.1, "end": 1.9} for k in RANGE_CUES if k not in ("silence",)]
        cues += [{"type": k, "times": [0.2]} for k in POINT_CUES if k != "sample"]
        bed = audio.synth_bed(tiny_story(cues))
        self.assertTrue(np.all(np.isfinite(bed)))
        self.assertGreater(rms(bed), 0.01)

    def test_hard_left_pan_leaves_right_ear_silent(self):
        bed = audio.synth_bed(tiny_story([{"type": "hiss", "start": 0, "end": 2, "pan": -1}]))
        self.assertGreater(rms(bed[:, 0]), 100 * rms(bed[:, 1]))

    def test_room_size_controls_echo_tail(self):
        def tail(size):
            cues = [{"type": "snap", "times": [0.2]}, {"type": "room", "start": 0, "end": 2, "size": size, "gain": 0}]
            bed = audio.synth_bed(tiny_story(cues))
            hit = np.abs(bed).max()
            return rms(bed[int(1.0 * audio.SR) : int(1.4 * audio.SR)]) / hit  # 0.8-1.2 s after the snap
        self.assertGreater(tail(0.9), 5 * tail(0.05))

    def test_muffle_removes_highs(self):
        def highs(cues):
            bed = audio.synth_bed(tiny_story(cues))[:, 0]
            return rms(np.diff(bed[int(0.5 * audio.SR) : int(1.5 * audio.SR)])) / rms(bed)
        hiss = {"type": "hiss", "start": 0, "end": 2}
        self.assertLess(highs([hiss, {"type": "muffle", "start": 0, "end": 2, "cutoff": 800}]), 0.5 * highs([hiss]))

    def test_sample_cue_uses_the_decoded_sound(self):
        story = tiny_story([{"type": "sample", "file": "x.wav", "times": [0.5]}])
        bed = audio.synth_bed(story, samples={"x.wav": np.ones(4800)})
        self.assertGreater(rms(bed[int(0.5 * audio.SR) : int(0.6 * audio.SR)]), 0.1)
        self.assertLess(rms(bed[: int(0.4 * audio.SR)]), 1e-6)


class VisualTests(unittest.TestCase):
    def test_static_frame(self):
        frame = Static(1080, 1920).frame(np.random.default_rng(0), 0.5)
        self.assertEqual(frame.shape, (1920, 1080, 3))
        self.assertEqual(frame.dtype, np.float32)
        self.assertGreater(frame.std(), 10)  # it's noise, not a flat colour

    def test_text_wraps_inside_the_frame(self):
        try:
            font = resolve_font(None, ROOT)
        except FontNotFoundError:
            self.skipTest("no font available on this machine")
        lines = [Line("The knocking came from my side of the door.", start=0)]
        block = TextBlock(lines, font, 1080, 1920)
        self.assertGreater(len(block.rows), 1)  # long line wrapped
        for _, _, text, x, _ in block.rows:
            self.assertGreaterEqual(x, 0)
            self.assertLessEqual(x + block.font.getlength(text), 1080)
        self.assertIsNone(block.sprite(-1))
        self.assertIsNotNone(block.sprite(10))

    def test_new_segment_types_render_frames(self):
        try:
            font = resolve_font(None, ROOT)
        except FontNotFoundError:
            self.skipTest("no font available on this machine")
        story = parse_story({"title": "t", "fps": 10, "size": "540x960", "segments": [
            {"type": "spectrogram", "duration": 1, "plan": {"bedroom": 0.5, "label": "9'7\"", "outline": "bed"},
             "lines": [{"text": "hi", "start": 0}]},
            {"type": "blueprint", "duration": 1, "plan": {"bedroom": 0, "outline": "wall"}, "pencil": "TENANT"},
            {"type": "countdown", "duration": 1, "digits": ["2'6\"", "1'8\""]},
            {"type": "static_text", "duration": 1, "background": "black", "lines": [{"text": "x", "start": 0}]},
        ]})
        ctx = Ctx(540, 960, 10, font)
        for seg in story.segments:
            frames = list(render_segment(seg, ctx, 10))
            self.assertEqual(len(frames), 10, seg.type)
            self.assertEqual(frames[-1].shape, (960, 540, 3))
            self.assertGreater(frames[-1].max(), 100, f"{seg.type} drew nothing visible")

    def test_missing_explicit_font_is_a_clear_error(self):
        with self.assertRaises(FontNotFoundError) as ctx:
            resolve_font("nope/not-a-font.ttf", ROOT)
        self.assertIn("not-a-font.ttf", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
