import json
import tempfile
import unittest
from pathlib import Path

from translator.state import (
    PROGRESS_VERSION,
    load_progress,
    merge_reference_profile,
    merge_story_state,
    new_reference_profile,
    new_story_state,
    reference_profile_for_prompt,
    save_progress,
    story_state_for_prompt,
)


class StoryStateTests(unittest.TestCase):
    def test_merge_story_state_deduplicates_named_entities(self):
        state = new_story_state({"title": "Book", "author": "Author"})
        state["characters"] = [{"name": "太郎", "description": "学生"}]
        patch = {
            "chapter_summary": "太郎再次出现。",
            "characters": [{"name": "太郎", "description": "高中生", "aliases": ["タロウ"]}],
            "glossary": [{"source": "東京", "target": "东京", "note": "地名"}],
            "locations": ["学校", "学校"],
        }

        merged = merge_story_state(state, patch, recent_summary_limit=5)

        self.assertEqual(len(merged["characters"]), 1)
        self.assertEqual(merged["characters"][0]["name"], "太郎")
        self.assertEqual(merged["characters"][0]["description"], "高中生")
        self.assertEqual(merged["locations"], ["学校"])
        self.assertEqual(merged["recent_summaries"], ["太郎再次出现。"])

    def test_merge_reference_profile_applies_caps_and_deduplicates(self):
        profile = new_reference_profile({"title": "Series Ref"}, "中文")
        patch = {
            "series_notes": ["沿用既刊译名", "沿用既刊译名"],
            "style_notes": [f"风格{i}" for i in range(20)],
            "characters": [
                {
                    "name": f"角色{i}",
                    "aliases": ["别名", "别名"],
                    "role": "角色",
                    "usage_note": "说明",
                    "example_sentences": ["例句A", "例句A", "例句B", "例句C"],
                }
                for i in range(45)
            ],
            "terms": [
                {
                    "term": f"术语{i}",
                    "category": "专有名词",
                    "usage_note": "说明",
                    "example_sentences": ["例句1", "例句2", "例句3"],
                }
                for i in range(85)
            ],
        }

        merged = merge_reference_profile(profile, patch)

        self.assertEqual(merged["series_notes"], ["沿用既刊译名"])
        self.assertEqual(len(merged["style_notes"]), 12)
        self.assertEqual(len(merged["characters"]), 40)
        self.assertEqual(len(merged["terms"]), 80)
        self.assertEqual(merged["characters"][0]["example_sentences"], ["例句A", "例句B"])

    def test_story_state_for_prompt_compacts_large_state(self):
        state = new_story_state({"title": "Book", "author": "Author"})
        state["characters"] = [
            {"name": f"角色{i}", "description": f"描述{i}" * 20, "aliases": [f"别名{i}", f"别名{i}扩展"]}
            for i in range(30)
        ]
        state["glossary"] = [
            {"source": f"术语{i}", "target": f"译法{i}", "note": f"说明{i}" * 20}
            for i in range(80)
        ]
        state["events"] = [f"事件{i}" * 20 for i in range(60)]
        state["style_notes"] = [f"风格{i}" * 20 for i in range(40)]
        state["open_questions"] = [f"疑问{i}" * 20 for i in range(20)]
        state["recent_summaries"] = [f"摘要{i}" * 80 for i in range(10)]

        prompt_state = story_state_for_prompt(state, recent_summary_limit=5)

        self.assertLessEqual(len(prompt_state["characters"]), 18)
        self.assertLessEqual(len(prompt_state["glossary"]), 40)
        self.assertLessEqual(len(prompt_state["events"]), 24)
        self.assertLessEqual(len(prompt_state["style_notes"]), 16)
        self.assertLessEqual(len(prompt_state["open_questions"]), 8)
        self.assertEqual(len(prompt_state["recent_summaries"]), 5)
        self.assertTrue(all(len(item) <= 220 for item in prompt_state["recent_summaries"]))
        serialized = json.dumps(prompt_state, ensure_ascii=False)
        self.assertLess(len(serialized), 12000)

    def test_reference_profile_for_prompt_compacts_large_profile(self):
        profile = new_reference_profile({"title": "Series Ref"}, "中文")
        profile["series_notes"] = [f"系列说明{i}" * 20 for i in range(20)]
        profile["style_notes"] = [f"风格说明{i}" * 20 for i in range(20)]
        profile["characters"] = [
            {
                "name": f"角色{i}",
                "aliases": [f"别名{i}", f"别名扩展{i}", f"别名额外{i}"],
                "role": f"角色{i}身份" * 10,
                "usage_note": f"说明{i}" * 20,
                "example_sentences": [f"例句{i}A" * 20, f"例句{i}B" * 20],
            }
            for i in range(40)
        ]
        profile["terms"] = [
            {
                "term": f"术语{i}",
                "category": "专有名词",
                "usage_note": f"术语说明{i}" * 20,
                "example_sentences": [f"术语例句{i}" * 20],
            }
            for i in range(80)
        ]

        prompt_profile = reference_profile_for_prompt(profile)

        self.assertLessEqual(len(prompt_profile["series_notes"]), 8)
        self.assertLessEqual(len(prompt_profile["style_notes"]), 8)
        self.assertLessEqual(len(prompt_profile["characters"]), 24)
        self.assertLessEqual(len(prompt_profile["terms"]), 48)
        serialized = json.dumps(prompt_profile, ensure_ascii=False)
        self.assertLess(len(serialized), 12000)

    def test_load_progress_migrates_legacy_document_records(self):
        legacy_progress = {
            "version": 2,
            "input_path": "/tmp/input.epub",
            "output_path": "/tmp/output.epub",
            "source_language": "日语",
            "target_language": "中文",
            "book": {"title": "Book", "author": "Author"},
            "story_state": new_story_state({"title": "Book", "author": "Author"}),
            "documents": {
                "chapter1.xhtml": {
                    "file_name": "chapter1.xhtml",
                    "item_id": "chapter1",
                    "status": "done",
                    "source_hash": "hash-1",
                    "translated_html": "<html></html>",
                    "segment_count": 3,
                    "reviews": [],
                    "story_state_after": {},
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            progress_path = Path(tmp_dir) / "progress.json"
            progress_path.write_text(json.dumps(legacy_progress, ensure_ascii=False), encoding="utf-8")

            migrated = load_progress(progress_path)

        self.assertEqual(migrated["version"], PROGRESS_VERSION)
        self.assertFalse(migrated["reference_enabled"])
        self.assertEqual(migrated["reference_phase"]["status"], "disabled")
        self.assertIn("summary_phase", migrated)
        self.assertIn("translation_phase", migrated)
        self.assertIn("reference_phase", migrated)
        self.assertEqual(migrated["documents"]["chapter1.xhtml"]["translation_status"], "done")
        self.assertEqual(migrated["documents"]["chapter1.xhtml"]["summary_status"], "pending")

    def test_save_progress_omits_rebuildable_translated_html(self):
        progress = {
            "version": PROGRESS_VERSION,
            "input_path": "/tmp/input.epub",
            "output_path": "/tmp/output.epub",
            "source_language": "日语",
            "target_language": "中文",
            "book": {"title": "Book", "author": "Author"},
            "story_state": new_story_state({"title": "Book", "author": "Author"}),
            "documents": {
                "chapter1.xhtml": {
                    "file_name": "chapter1.xhtml",
                    "item_id": "chapter1",
                    "source_hash": "hash-1",
                    "segment_count": 2,
                    "batch_count": 2,
                    "summary_status": "done",
                    "summary_patch": {},
                    "translation_context_snapshot": {"events": ["事件"]},
                    "translation_status": "done",
                    "translated_html": "<html>large translated body</html>",
                    "translated_batches": {
                        "batch_0001": {
                            "batch_index": 1,
                            "translations": {"seg_0001": "译文一"},
                            "review": {},
                        },
                        "batch_0002": {
                            "batch_index": 2,
                            "translations": {"seg_0002": "译文二"},
                            "review": {},
                        },
                    },
                    "reviews": [],
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            progress_path = Path(tmp_dir) / "progress.json"
            save_progress(progress_path, progress)
            saved = json.loads(progress_path.read_text(encoding="utf-8"))

        saved_record = saved["documents"]["chapter1.xhtml"]
        self.assertEqual(saved_record["translation_status"], "done")
        self.assertEqual(saved_record["translated_html"], "")
        self.assertEqual(saved_record["translated_batches"]["batch_0001"]["translations"]["seg_0001"], "译文一")


if __name__ == "__main__":
    unittest.main()
