"""Story parsing, validation and timeline math. Run: python -m unittest discover -s tests"""

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # so the tests run even without `pip install -e .`

from staticreels.story import Line, StoryError, load_story, parse_story  # noqa: E402

SEED = ROOT / "tests" / "fixtures" / "sample_story.json"
SEED_DATA = json.loads(SEED.read_text(encoding="utf-8"))


def seed_with(change) -> dict:
    """A deep copy of the seed story with `change(data)` applied."""
    data = copy.deepcopy(SEED_DATA)
    change(data)
    return data


class SeedStoryTests(unittest.TestCase):
    def setUp(self):
        self.story = load_story(SEED)

    def test_top_level_fields(self):
        self.assertEqual(self.story.fps, 30)
        self.assertEqual((self.story.width, self.story.height), (1080, 1920))
        self.assertEqual(self.story.aspect, "9:16")
        self.assertIsNone(self.story.voiceover)

    def test_segment_order(self):
        self.assertEqual(
            [s.type for s in self.story.segments],
            ["static_text", "ai_clip", "static_text", "ai_clip", "countdown", "ai_clip", "end_card"],
        )

    def test_segment_starts_match_the_script(self):
        self.assertEqual([s.start for s in self.story.segments], [0, 3, 8, 12, 17, 21, 26])
        self.assertEqual([s.end for s in self.story.segments], [3, 8, 12, 17, 21, 26, 30])

    def test_durations_sum_to_the_video_length(self):
        self.assertEqual(sum(s.duration for s in self.story.segments), 30)
        self.assertEqual(self.story.duration, 30)
        self.assertEqual(self.story.total_frames, 900)

    def test_frame_ranges_tile_the_video_exactly(self):
        expected_start = 0
        for seg in self.story.segments:
            f0, f1 = self.story.frame_range(seg)
            self.assertEqual(f0, expected_start)
            self.assertGreater(f1, f0)
            expected_start = f1
        self.assertEqual(expected_start, self.story.total_frames)

    def test_audio_cues_fall_within_the_video(self):
        total = self.story.duration
        self.assertTrue(self.story.cues)
        for cue in self.story.cues:
            if cue.times:
                for t in cue.times:
                    self.assertTrue(0 <= t < total, f"{cue.type} at {t}")
            else:
                self.assertTrue(0 <= cue.start < cue.end <= total, f"{cue.type} {cue.start}-{cue.end}")

    def test_seed_cues_match_the_brief(self):
        by_type = {c.type: c for c in self.story.cues}
        self.assertEqual((by_type["drone"].start, by_type["drone"].end), (0, 30))
        knocks = by_type["knock"].times
        self.assertEqual(len(knocks), 3)
        self.assertTrue(all(27 <= t <= 29 for t in knocks))
        countdown = next(s for s in self.story.segments if s.type == "countdown")
        rising = [c for c in self.story.cues if c.type == "hiss" and c.start == countdown.start]
        self.assertEqual(len(rising), 1)
        self.assertGreater(rising[0].gain_to, rising[0].gain)

    def test_ai_clips(self):
        clips = self.story.ai_segments
        self.assertEqual([s.clip for s in clips], ["clip1.mp4", "clip2.mp4", "clip3.mp4"])
        for seg in clips:
            self.assertIn("vertical 9:16", seg.prompt)
        self.assertTrue(clips[0].overlay.rec_dot)
        self.assertEqual(clips[0].overlay.timestamp, "3:07 AM")

    def test_every_line_finishes_typing_inside_its_segment(self):
        for seg in self.story.segments:
            for line in seg.lines:
                self.assertLessEqual(line.typed_by, seg.duration, f"{seg.label}: {line.text!r}")

    def test_end_card_handle(self):
        self.assertEqual(self.story.segments[-1].handle, "@static.in.the.walls")

    def test_segment_at(self):
        self.assertEqual(self.story.segment_at(0).index, 0)
        self.assertEqual(self.story.segment_at(2.999).index, 0)
        self.assertEqual(self.story.segment_at(3.0).index, 1)
        self.assertEqual(self.story.segment_at(19).type, "countdown")
        self.assertEqual(self.story.segment_at(30).index, 6)
        with self.assertRaises(ValueError):
            self.story.segment_at(30.5)


class AllStoriesTests(unittest.TestCase):
    def test_every_story_file_validates_and_tiles_the_video(self):
        files = sorted((ROOT / "stories").glob("*.json"))
        self.assertGreaterEqual(len(files), 2)
        for path in files:
            with self.subTest(story=path.name):
                story = load_story(path)
                self.assertEqual(story.frame_range(story.segments[-1])[1], story.total_frames)
                names = [s.clip for s in story.ai_segments]
                self.assertEqual(len(names), len(set(names)), "clip filenames must be unique")
                self.assertEqual(story.duration, 30)
                for seg in story.segments:
                    for line in seg.lines:
                        self.assertLessEqual(line.typed_by, seg.duration, f"{seg.label}: {line.text!r}")

    def test_floor_plan_series_shrinks_the_bedroom(self):
        widths = []
        for k in (1, 2, 4):
            story = load_story(ROOT / "stories" / f"floor_plan_part{k}.json")
            widths += [s.plan.bedroom for s in story.segments if s.type in ("spectrogram", "blueprint")]
        self.assertEqual(widths, sorted(widths, reverse=True))
        self.assertEqual(widths[-1], 0)


class NewFieldValidationTests(unittest.TestCase):
    def assertRejected(self, data, *fragments):
        with self.assertRaises(StoryError) as ctx:
            parse_story(data)
        for fragment in fragments:
            self.assertIn(fragment, str(ctx.exception))

    def story(self, segment=None, cue=None):
        return {"title": "t", "segments": [segment or {"type": "end_card", "duration": 2}],
                "audio_cues": [cue] if cue else []}

    def test_text_countdown_values(self):
        seg = parse_story(self.story({"type": "countdown", "duration": 2, "digits": ["2'6\"", 3]})).segments[0]
        self.assertEqual(seg.digits, ["2'6\"", "3"])

    def test_bad_new_fields(self):
        self.assertRejected(self.story(cue={"type": "hiss", "start": 0, "end": 1, "pan": -2}), "pan")
        self.assertRejected(self.story(cue={"type": "room", "start": 0, "end": 1, "size": 3}), "size")
        self.assertRejected(self.story(cue={"type": "sample", "times": [1]}), "file")
        self.assertRejected(self.story({"type": "spectrogram", "duration": 2, "plan": {"bedroom": 2}}), "bedroom")
        self.assertRejected(self.story({"type": "blueprint", "duration": 2, "plan": {"outline": "sofa"}}), "outline")
        self.assertRejected(self.story({"type": "static_text", "duration": 2, "background": "pink"}), "background")


class TimelineMathTests(unittest.TestCase):
    def test_fractional_durations_never_drift(self):
        durations = [1.37, 2.11, 0.52, 3.333, 0.7, 4.01]
        data = {"title": "t", "fps": 30, "segments": [{"type": "end_card", "duration": d} for d in durations]}
        story = parse_story(data)
        counts = [f1 - f0 for f0, f1 in map(story.frame_range, story.segments)]
        self.assertEqual(sum(counts), story.total_frames)
        self.assertEqual(story.total_frames, round(sum(durations) * 30))

    def test_typewriter_character_counts(self):
        line = Line("abcdefghij", start=1.0, cps=10.0)
        self.assertEqual(line.chars_at(0.99), 0)
        self.assertEqual(line.chars_at(1.0), 1)
        self.assertEqual(line.chars_at(1.45), 5)
        self.assertEqual(line.chars_at(5.0), 10)
        self.assertAlmostEqual(line.typed_by, 2.0)
        self.assertEqual(Line("abc", start=0, end=1.0).chars_at(1.0), 0)  # gone at its end time


class ValidationTests(unittest.TestCase):
    def assertRejected(self, change, *fragments):
        with self.assertRaises(StoryError) as ctx:
            parse_story(seed_with(change))
        for fragment in fragments:
            self.assertIn(fragment, str(ctx.exception))

    def test_declared_duration_must_match_segments(self):
        self.assertRejected(lambda d: d.update(duration=31), "add up to 30s", "31s")

    def test_cue_past_the_end_is_rejected(self):
        self.assertRejected(lambda d: d["audio_cues"][-1]["times"].append(30.5), "audio_cues[6].times[3]")

    def test_negative_cue_time_is_rejected(self):
        self.assertRejected(lambda d: d["audio_cues"][4]["times"].insert(0, -1), "audio_cues[4].times[0]")

    def test_range_cue_must_be_ordered(self):
        self.assertRejected(lambda d: d["audio_cues"][1].update(start=3, end=2), "audio_cues[1]")

    def test_range_cue_past_the_end_is_rejected(self):
        self.assertRejected(lambda d: d["audio_cues"][0].update(end=31), "audio_cues[0]", "video length")

    def test_line_starting_outside_its_segment(self):
        self.assertRejected(lambda d: d["segments"][0]["lines"][0].update(start=3.5), "segments[0].lines[0].start")

    def test_line_that_cant_finish_typing(self):
        self.assertRejected(lambda d: d["segments"][0]["lines"][1].update(cps=5), "segments[0].lines[1]", "typing")

    def test_unknown_segment_type(self):
        self.assertRejected(lambda d: d["segments"][0].update(type="montage"), "segments[0].type")

    def test_ai_clip_needs_a_prompt_and_a_file(self):
        self.assertRejected(lambda d: d["segments"][1].pop("prompt"), "segments[1]", "prompt")
        self.assertRejected(lambda d: d["segments"][1].pop("clip"), "segments[1]", "clip")

    def test_countdown_needs_digits(self):
        self.assertRejected(lambda d: d["segments"][4].update(digits=[]), "segments[4].digits")

    def test_bad_sizes(self):
        self.assertRejected(lambda d: d.update(size="1080 by 1920"), "size")
        self.assertRejected(lambda d: d.update(size="1081x1920"), "even")

    def test_zero_duration_segment(self):
        self.assertRejected(lambda d: d["segments"][2].update(duration=0), "segments[2].duration")

    def test_glitch_out_of_range(self):
        self.assertRejected(lambda d: d["segments"][2].update(glitch=2), "segments[2].glitch")

    def test_missing_file_and_bad_json(self):
        with self.assertRaises(StoryError):
            load_story(ROOT / "stories" / "does_not_exist.json")


if __name__ == "__main__":
    unittest.main()
