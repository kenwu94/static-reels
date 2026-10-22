"""Publishing stays a dry run unless asked, and asks before doing anything."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from staticreels import publish  # noqa: E402
from staticreels.story import load_story  # noqa: E402

SEED = ROOT / "tests" / "fixtures" / "sample_story.json"


class CaptionTests(unittest.TestCase):
    def test_template_has_hook_hashtags_and_ai_reminder(self):
        text = publish.caption_template(load_story(SEED))
        caption, _, notes = text.partition(publish.NOTES_MARKER)
        self.assertTrue(caption.startswith("I've been hearing static in my walls."))
        self.assertIn("#horror", caption)
        self.assertNotIn("AI", caption)  # the reminder is in the notes, not the posted caption
        self.assertIn("AI LABEL", notes)
        self.assertIn("clip1.mp4", notes)

    def test_template_never_overwrites(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "c.txt"
            path.write_text("my edits", encoding="utf-8")
            self.assertFalse(publish.write_caption_template(load_story(SEED), path))
            self.assertEqual(path.read_text(encoding="utf-8"), "my edits")

    def test_procedural_render_has_no_ai_label_note(self):
        _, _, notes = publish.caption_template(load_story(SEED), ai_clips=[]).partition(publish.NOTES_MARKER)
        self.assertNotIn("AI LABEL", notes)
        self.assertIn("No AI-generated footage", notes)

    def test_untouched_template_is_refreshed_when_clip_source_changes(self):
        story = load_story(SEED)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "c.txt"
            publish.write_caption_template(story, path)  # AI-clips version
            self.assertTrue(publish.write_caption_template(story, path, ai_clips=[]))
            self.assertIn("No AI-generated footage", path.read_text(encoding="utf-8"))

    def test_schedule_in_the_past_is_rejected(self):
        with self.assertRaises(publish.PublishError):
            publish.parse_schedule("2001-01-01 09:00")
        self.assertIsNone(publish.parse_schedule(None))


class RunPublishTests(unittest.TestCase):
    """Uses a fake 'video' plus a placeholder sidecar; inspect_video's failure is tolerated or blocks."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.video = d / "x.mp4"
        self.video.write_bytes(b"not really a video")
        self.caption = d / "caption.txt"
        self.caption.write_text(publish.caption_template(load_story(SEED)), encoding="utf-8")
        self.lines = []

    def tearDown(self):
        self.tmp.cleanup()

    def run_publish(self, **kw):
        def ask(prompt):
            raise AssertionError("dry run must not prompt")
        kw.setdefault("ask", ask)
        return publish.run_publish(self.video, self.caption, out=self.lines.append, **kw)

    def test_dry_run_by_default(self):
        self.assertEqual(self.run_publish(), 0)
        self.assertTrue(any("DRY RUN" in line for line in self.lines))

    def test_placeholder_render_is_blocked(self):
        self.video.with_suffix(".render.json").write_text(json.dumps({"placeholders": True}), encoding="utf-8")
        with self.assertRaises(publish.PublishError):
            self.run_publish(yes=True)


if __name__ == "__main__":
    unittest.main()
