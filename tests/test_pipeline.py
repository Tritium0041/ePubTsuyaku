import copy
import tempfile
import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import patch

from ebooklib import epub

from translator.config import PipelineConfig
from translator.pipeline import run_translation_pipeline, run_translation_pipeline_with_retries
from translator.state import load_progress


def build_sample_epub(path: Path, chapters=None, *, title: str = "Demo Book", language: str = "ja") -> None:
    chapters = chapters or [
        ("第一章", ["太郎は学校へ行った。", "花子に会った。"]),
        ("第二章", ["二人は图书馆で本を読んだ。"]),
    ]

    book = epub.EpubBook()
    book.set_identifier("demo-book")
    book.set_title(title)
    book.set_language(language)
    book.add_author("Tester")

    chapter_items = []
    for index, (title, paragraphs) in enumerate(chapters, start=1):
        chapter = epub.EpubHtml(title=title, file_name=f"chapter{index}.xhtml", lang="ja")
        body = "\n".join(f"<p>{paragraph}</p>" for paragraph in paragraphs)
        chapter.content = f"""
        <?xml version='1.0' encoding='utf-8'?>
        <html xmlns="http://www.w3.org/1999/xhtml">
          <body>
            <h1>{title}</h1>
            {body}
          </body>
        </html>
        """
        book.add_item(chapter)
        chapter_items.append(chapter)

    book.toc = tuple(chapter_items)
    book.spine = ["nav", *chapter_items]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    epub.write_epub(str(path), book, {})


def make_config(
    *,
    input_path: Path,
    output_path: Path,
    progress_path: Path,
    provider: str = "mock",
    translation_workers: int = 4,
    max_batch_chars: int = 1000,
    max_batch_segments: int = 64,
    max_review_retries: int = 0,
    min_review_score: int = 85,
    reset_progress: bool = False,
    reference_input_path: Optional[Path] = None,
    target_language: str = "中文",
) -> PipelineConfig:
    model = "mock-model" if provider == "mock" else "demo-model"
    return PipelineConfig(
        input_path=input_path,
        output_path=output_path,
        progress_path=progress_path,
        reference_input_path=reference_input_path,
        source_language="日语",
        target_language=target_language,
        provider=provider,
        api_key=None if provider == "mock" else "test-key",
        base_url=None if provider == "mock" else "https://example.com/v1",
        model=model,
        summary_model=model,
        translation_model=model,
        review_model=model,
        translation_workers=translation_workers,
        max_batch_chars=max_batch_chars,
        max_batch_segments=max_batch_segments,
        max_review_retries=max_review_retries,
        min_review_score=min_review_score,
        recent_summary_limit=3,
        title_suffix="（中文译本）",
        reset_progress=reset_progress,
    )


def empty_summary_response() -> dict:
    return {
        "chapter_summary": "",
        "characters": [],
        "time_context": [],
        "locations": [],
        "events": [],
        "concepts": [],
        "glossary": [],
        "style_notes": [],
        "open_questions": [],
    }


def ok_review_response() -> dict:
    return {
        "score": 100,
        "needs_retry": False,
        "major_issues": [],
        "minor_issues": [],
        "term_updates": [],
        "style_updates": [],
        "corrected_segments": {},
        "retry_feedback": "",
    }


class PipelineIntegrationTests(unittest.TestCase):
    def test_mock_pipeline_translates_and_rebuilds_epub(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "input.epub"
            output_path = tmp_path / "output.epub"
            progress_path = tmp_path / "progress.json"

            build_sample_epub(input_path)
            config = make_config(
                input_path=input_path,
                output_path=output_path,
                progress_path=progress_path,
                provider="mock",
                translation_workers=4,
            )

            result = run_translation_pipeline(config)

            self.assertTrue(output_path.exists())
            self.assertTrue(progress_path.exists())
            self.assertEqual(result["processed_count"], 2)
            self.assertEqual(result["summary_completed_count"], 2)

            translated_book = epub.read_epub(str(output_path))
            chapter1 = translated_book.get_item_with_href("chapter1.xhtml")
            content = chapter1.get_content().decode("utf-8")
            nav = translated_book.get_item_with_id("nav").get_content().decode("utf-8")
            self.assertIn("[中文] 太郎は学校へ行った。", content)
            self.assertIn("[中文] 第一章", nav)
            self.assertIn("[中文] 第二章", nav)
            self.assertIn("Demo Book（中文译本）", translated_book.title)

    def test_pipeline_does_not_retry_when_score_is_low_but_no_retry_requested(self):
        shared = {"translate_calls": 0}

        class FakeLLMClient:
            def extract_reference_patch(self, *args, **kwargs):
                return {"series_notes": [], "style_notes": [], "characters": [], "terms": []}

            def summarize(self, *args, **kwargs):
                return empty_summary_response()

            def translate(self, *args, **kwargs):
                shared["translate_calls"] += 1
                segments = kwargs["segments"]
                return {segment["id"]: f"[中文] {segment['text']}" for segment in segments}

            def review(self, *args, **kwargs):
                payload = ok_review_response()
                payload["score"] = 10
                return payload

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "input.epub"
            output_path = tmp_path / "output.epub"
            progress_path = tmp_path / "progress.json"

            build_sample_epub(input_path)
            config = make_config(
                input_path=input_path,
                output_path=output_path,
                progress_path=progress_path,
                provider="openai-compatible",
                translation_workers=1,
                max_review_retries=3,
                min_review_score=90,
            )

            with patch("translator.pipeline._build_llm_client", side_effect=lambda *_: FakeLLMClient()):
                run_translation_pipeline(config)

            self.assertEqual(shared["translate_calls"], 2)

    def test_summary_phase_saves_context_snapshots_without_future_leak(self):
        class FakeLLMClient:
            def extract_reference_patch(self, *args, **kwargs):
                return {"series_notes": [], "style_notes": [], "characters": [], "terms": []}

            def summarize(self, *args, **kwargs):
                joined = " ".join(segment["text"] for segment in kwargs["segments"])
                if "事件甲" in joined:
                    event = "事件甲"
                else:
                    event = "事件乙"
                return {
                    "chapter_summary": event,
                    "characters": [],
                    "time_context": [],
                    "locations": [],
                    "events": [event],
                    "concepts": [],
                    "glossary": [],
                    "style_notes": [],
                    "open_questions": [],
                }

            def translate(self, *args, **kwargs):
                return {segment["id"]: f"[中文] {segment['text']}" for segment in kwargs["segments"]}

            def review(self, *args, **kwargs):
                return ok_review_response()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "input.epub"
            output_path = tmp_path / "output.epub"
            progress_path = tmp_path / "progress.json"

            build_sample_epub(
                input_path,
                chapters=[
                    ("第一章", ["事件甲。"]),
                    ("第二章", ["事件乙。"]),
                ],
            )
            config = make_config(
                input_path=input_path,
                output_path=output_path,
                progress_path=progress_path,
                provider="openai-compatible",
                translation_workers=1,
            )

            with patch("translator.pipeline._build_llm_client", side_effect=lambda *_: FakeLLMClient()):
                run_translation_pipeline(config)

            progress = load_progress(progress_path)
            chapter1 = progress["documents"]["chapter1.xhtml"]
            chapter2 = progress["documents"]["chapter2.xhtml"]

            self.assertEqual(chapter1["summary_patch"]["events"], ["事件甲"])
            self.assertEqual(chapter1["translation_context_snapshot"]["events"], ["事件甲"])
            self.assertEqual(chapter2["translation_context_snapshot"]["events"], ["事件甲", "事件乙"])

    def test_large_document_summary_is_chunked_and_aggregated_before_translation(self):
        shared = {"summary_inputs": [], "translation_story_state": None}

        class FakeLLMClient:
            def extract_reference_patch(self, *args, **kwargs):
                return {"series_notes": [], "style_notes": [], "characters": [], "terms": []}

            def summarize(self, *args, **kwargs):
                joined = " ".join(segment["text"] for segment in kwargs["segments"])
                shared["summary_inputs"].append(joined)
                event = "前半事件" if "前半事件" in joined else "后半事件"
                style = "前半风格" if "前半事件" in joined else "后半风格"
                return {
                    "chapter_summary": event,
                    "characters": [],
                    "time_context": [],
                    "locations": [],
                    "events": [event],
                    "concepts": [],
                    "glossary": [],
                    "style_notes": [style],
                    "open_questions": [],
                }

            def translate(self, *args, **kwargs):
                shared["translation_story_state"] = copy.deepcopy(kwargs["story_state"])
                return {segment["id"]: f"[中文] {segment['text']}" for segment in kwargs["segments"]}

            def review(self, *args, **kwargs):
                return ok_review_response()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "input.epub"
            output_path = tmp_path / "output.epub"
            progress_path = tmp_path / "progress.json"

            build_sample_epub(
                input_path,
                chapters=[
                    (
                        "第一章",
                        [
                            "前半事件。" * 20,
                            "前半事件。" * 20,
                            "后半事件。" * 20,
                            "后半事件。" * 20,
                        ],
                    ),
                ],
            )
            config = make_config(
                input_path=input_path,
                output_path=output_path,
                progress_path=progress_path,
                provider="openai-compatible",
                translation_workers=1,
                max_batch_chars=120,
                max_batch_segments=1,
            )

            with patch("translator.pipeline._build_llm_client", side_effect=lambda *_: FakeLLMClient()):
                run_translation_pipeline(config)

            self.assertGreaterEqual(len(shared["summary_inputs"]), 2)
            self.assertIsNotNone(shared["translation_story_state"])
            self.assertIn("前半事件", shared["translation_story_state"]["events"])
            self.assertIn("后半事件", shared["translation_story_state"]["events"])
            self.assertIn("前半风格", shared["translation_story_state"]["style_notes"])
            self.assertIn("后半风格", shared["translation_story_state"]["style_notes"])

    def test_summary_chunk_structured_output_failure_falls_back_to_smaller_summaries(self):
        shared = {"summary_batch_sizes": []}

        class FakeLLMClient:
            def extract_reference_patch(self, *args, **kwargs):
                return {"series_notes": [], "style_notes": [], "characters": [], "terms": []}

            def summarize(self, *args, **kwargs):
                batch_size = len(kwargs["segments"])
                shared["summary_batch_sizes"].append(batch_size)
                if batch_size > 1:
                    raise RuntimeError("严格 schema 调用失败：没有找到完整的 JSON。；JSON mode 也连续 3 次失败：没有找到完整的 JSON。")
                joined = " ".join(segment["text"] for segment in kwargs["segments"])
                return {
                    "chapter_summary": joined,
                    "characters": [],
                    "time_context": [],
                    "locations": [],
                    "events": [joined],
                    "concepts": [],
                    "glossary": [],
                    "style_notes": [],
                    "open_questions": [],
                }

            def translate(self, *args, **kwargs):
                return {segment["id"]: f"[中文] {segment['text']}" for segment in kwargs["segments"]}

            def review(self, *args, **kwargs):
                return ok_review_response()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "input.epub"
            output_path = tmp_path / "output.epub"
            progress_path = tmp_path / "progress.json"

            build_sample_epub(
                input_path,
                chapters=[("第一章", ["第一段。", "第二段。", "第三段。"])],
            )
            config = make_config(
                input_path=input_path,
                output_path=output_path,
                progress_path=progress_path,
                provider="openai-compatible",
                translation_workers=1,
                max_batch_segments=3,
            )

            with patch("translator.pipeline._build_llm_client", side_effect=lambda *_: FakeLLMClient()):
                result = run_translation_pipeline(config)

            self.assertEqual(result["processed_count"], 1)
            self.assertTrue(any(size > 1 for size in shared["summary_batch_sizes"]))
            self.assertTrue(any(size == 1 for size in shared["summary_batch_sizes"]))

    def test_review_updates_do_not_pollute_following_document_translation_context(self):
        shared = {
            "second_doc_glossary": None,
            "second_doc_style_notes": None,
        }

        class FakeLLMClient:
            def extract_reference_patch(self, *args, **kwargs):
                return {"series_notes": [], "style_notes": [], "characters": [], "terms": []}

            def summarize(self, *args, **kwargs):
                return empty_summary_response()

            def translate(self, *args, **kwargs):
                segments = kwargs["segments"]
                joined = " ".join(segment["text"] for segment in segments)
                if "第二章" in joined:
                    shared["second_doc_glossary"] = copy.deepcopy(kwargs["story_state"].get("glossary", []))
                    shared["second_doc_style_notes"] = copy.deepcopy(kwargs["story_state"].get("style_notes", []))
                return {segment["id"]: f"[中文] {segment['text']}" for segment in segments}

            def review(self, *args, **kwargs):
                joined = " ".join(segment["text"] for segment in kwargs["source_segments"])
                if "第一章" in joined:
                    payload = ok_review_response()
                    payload["term_updates"] = [{"source": "动态术语", "target": "术语", "note": "只影响当前批"}]
                    payload["style_updates"] = ["动态风格"]
                    return payload
                return ok_review_response()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "input.epub"
            output_path = tmp_path / "output.epub"
            progress_path = tmp_path / "progress.json"

            build_sample_epub(
                input_path,
                chapters=[
                    ("第一章", ["太郎は学校へ行った。"]),
                    ("第二章", ["花子は本を読んだ。"]),
                ],
            )
            config = make_config(
                input_path=input_path,
                output_path=output_path,
                progress_path=progress_path,
                provider="openai-compatible",
                translation_workers=1,
            )

            with patch("translator.pipeline._build_llm_client", side_effect=lambda *_: FakeLLMClient()):
                run_translation_pipeline(config)

            self.assertEqual(shared["second_doc_glossary"], [])
            self.assertEqual(shared["second_doc_style_notes"], [])

    def test_pipeline_resume_uses_saved_batches_and_only_translates_missing_work(self):
        class FailingClient:
            def __init__(self, shared):
                self.shared = shared

            def extract_reference_patch(self, *args, **kwargs):
                return {"series_notes": [], "style_notes": [], "characters": [], "terms": []}

            def summarize(self, *args, **kwargs):
                return empty_summary_response()

            def translate(self, *args, **kwargs):
                joined = " ".join(segment["text"] for segment in kwargs["segments"])
                self.shared["translated_inputs"].append(joined)
                if "第二段。" in joined:
                    raise RuntimeError("boom on second paragraph")
                return {segment["id"]: f"[中文] {segment['text']}" for segment in kwargs["segments"]}

            def review(self, *args, **kwargs):
                return ok_review_response()

        class StableClient(FailingClient):
            def translate(self, *args, **kwargs):
                joined = " ".join(segment["text"] for segment in kwargs["segments"])
                self.shared["translated_inputs"].append(joined)
                return {segment["id"]: f"[中文] {segment['text']}" for segment in kwargs["segments"]}

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "input.epub"
            output_path = tmp_path / "output.epub"
            progress_path = tmp_path / "progress.json"

            build_sample_epub(
                input_path,
                chapters=[
                    ("第一章", ["第一段。", "第二段。", "第三段。"]),
                ],
            )
            config = make_config(
                input_path=input_path,
                output_path=output_path,
                progress_path=progress_path,
                provider="openai-compatible",
                translation_workers=1,
                max_batch_segments=1,
            )

            first_shared = {"translated_inputs": []}
            with patch("translator.pipeline._build_llm_client", side_effect=lambda *_: FailingClient(first_shared)):
                with self.assertRaisesRegex(RuntimeError, "boom on second paragraph"):
                    run_translation_pipeline(config)

            progress = load_progress(progress_path)
            record = progress["documents"]["chapter1.xhtml"]
            translated_batch_keys = sorted(record["translated_batches"].keys())
            self.assertIn("batch_0001", translated_batch_keys)
            self.assertIn("batch_0002", translated_batch_keys)
            self.assertNotIn("batch_0003", translated_batch_keys)

            second_shared = {"translated_inputs": []}
            with patch("translator.pipeline._build_llm_client", side_effect=lambda *_: StableClient(second_shared)):
                result = run_translation_pipeline(config)

            self.assertEqual(result["processed_count"], 1)
            self.assertTrue(all("第一段。" not in item for item in second_shared["translated_inputs"]))
            self.assertTrue(all("第一章" not in item for item in second_shared["translated_inputs"]))
            self.assertLessEqual(len(second_shared["translated_inputs"]), 2)
            self.assertTrue(any("第二段。" in item for item in second_shared["translated_inputs"]))
            self.assertEqual(result["translation_completed_batch_count"], 4)

    def test_auto_resume_retries_retryable_run_errors_and_reuses_progress(self):
        shared = {"translated_inputs": [], "failed_once": False}

        class FlakyClient:
            def extract_reference_patch(self, *args, **kwargs):
                return {"series_notes": [], "style_notes": [], "characters": [], "terms": []}

            def summarize(self, *args, **kwargs):
                return empty_summary_response()

            def translate(self, *args, **kwargs):
                joined = " ".join(segment["text"] for segment in kwargs["segments"])
                shared["translated_inputs"].append(joined)
                if not shared["failed_once"] and "第二段。" in joined:
                    shared["failed_once"] = True
                    raise RuntimeError("模型连续 3 次未返回合法 JSON: mock boom")
                return {segment["id"]: f"[中文] {segment['text']}" for segment in kwargs["segments"]}

            def review(self, *args, **kwargs):
                return ok_review_response()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "input.epub"
            output_path = tmp_path / "output.epub"
            progress_path = tmp_path / "progress.json"

            build_sample_epub(
                input_path,
                chapters=[
                    ("第一章", ["第一段。", "第二段。", "第三段。"]),
                ],
            )
            config = make_config(
                input_path=input_path,
                output_path=output_path,
                progress_path=progress_path,
                provider="openai-compatible",
                translation_workers=1,
                max_batch_segments=1,
                reset_progress=True,
            )

            with patch("translator.pipeline._build_llm_client", side_effect=lambda *_: FlakyClient()):
                result = run_translation_pipeline_with_retries(config)

            self.assertEqual(result["processed_count"], 1)
            self.assertEqual(result["auto_resume_count"], 1)
            self.assertEqual(result["run_attempt_count"], 2)
            self.assertTrue(output_path.exists())
            self.assertEqual(shared["translated_inputs"].count("第一章"), 1)
            self.assertEqual(shared["translated_inputs"].count("第一段。"), 1)
            self.assertGreaterEqual(shared["translated_inputs"].count("第二段。"), 2)

            progress = load_progress(progress_path)
            record = progress["documents"]["chapter1.xhtml"]
            self.assertEqual(record["translation_status"], "done")
            self.assertIn("batch_0001", record["translated_batches"])
            self.assertIn("batch_0002", record["translated_batches"])
            self.assertIn("batch_0003", record["translated_batches"])
            self.assertIn("batch_0004", record["translated_batches"])

    def test_auto_resume_does_not_retry_non_retryable_errors(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "input.epub"
            output_path = tmp_path / "output.epub"
            progress_path = tmp_path / "progress.json"

            build_sample_epub(input_path)
            config = make_config(
                input_path=input_path,
                output_path=output_path,
                progress_path=progress_path,
                provider="mock",
            )

            with patch(
                "translator.pipeline.run_translation_pipeline",
                side_effect=RuntimeError("缺少 API Key。"),
            ) as run_once:
                with self.assertRaisesRegex(RuntimeError, "缺少 API Key"):
                    run_translation_pipeline_with_retries(config)

            self.assertEqual(run_once.call_count, 1)

    def test_translation_output_is_identical_between_one_and_four_workers(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "input.epub"
            output_path_one = tmp_path / "output-one.epub"
            output_path_four = tmp_path / "output-four.epub"
            progress_one = tmp_path / "progress-one.json"
            progress_four = tmp_path / "progress-four.json"

            build_sample_epub(
                input_path,
                chapters=[
                    ("第一章", ["段落一。", "段落二。", "段落三。", "段落四。", "段落五。", "段落六。"]),
                ],
            )

            result_one = run_translation_pipeline(
                make_config(
                    input_path=input_path,
                    output_path=output_path_one,
                    progress_path=progress_one,
                    provider="mock",
                    translation_workers=1,
                    max_batch_segments=1,
                )
            )
            result_four = run_translation_pipeline(
                make_config(
                    input_path=input_path,
                    output_path=output_path_four,
                    progress_path=progress_four,
                    provider="mock",
                    translation_workers=4,
                    max_batch_segments=1,
                )
            )

            book_one = epub.read_epub(str(output_path_one))
            book_four = epub.read_epub(str(output_path_four))
            chapter_one = book_one.get_item_with_href("chapter1.xhtml").get_content().decode("utf-8")
            chapter_four = book_four.get_item_with_href("chapter1.xhtml").get_content().decode("utf-8")

            self.assertEqual(result_one["translation_total_batch_count"], result_four["translation_total_batch_count"])
            self.assertGreater(result_one["translation_total_batch_count"], 1)
            self.assertEqual(chapter_one, chapter_four)

    def test_pipeline_without_reference_does_not_pass_reference_profile(self):
        shared = {
            "summary_reference_profile": [],
            "translation_reference_profile": [],
            "review_reference_profile": [],
        }

        class FakeLLMClient:
            def extract_reference_patch(self, *args, **kwargs):
                raise AssertionError("reference phase should not run without reference epub")

            def summarize(self, *args, **kwargs):
                shared["summary_reference_profile"].append(kwargs.get("reference_profile"))
                return empty_summary_response()

            def translate(self, *args, **kwargs):
                shared["translation_reference_profile"].append(kwargs.get("reference_profile"))
                return {segment["id"]: f"[中文] {segment['text']}" for segment in kwargs["segments"]}

            def review(self, *args, **kwargs):
                shared["review_reference_profile"].append(kwargs.get("reference_profile"))
                return ok_review_response()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "input.epub"
            output_path = tmp_path / "output.epub"
            progress_path = tmp_path / "progress.json"

            build_sample_epub(input_path)
            config = make_config(
                input_path=input_path,
                output_path=output_path,
                progress_path=progress_path,
                provider="openai-compatible",
                translation_workers=1,
            )

            with patch("translator.pipeline._build_llm_client", side_effect=lambda *_: FakeLLMClient()):
                run_translation_pipeline(config)

            self.assertTrue(shared["summary_reference_profile"])
            self.assertTrue(shared["translation_reference_profile"])
            self.assertTrue(shared["review_reference_profile"])
            self.assertTrue(all(item is None for item in shared["summary_reference_profile"]))
            self.assertTrue(all(item is None for item in shared["translation_reference_profile"]))
            self.assertTrue(all(item is None for item in shared["review_reference_profile"]))

    def test_reference_profile_is_injected_without_polluting_story_state(self):
        shared = {
            "summary_reference_profile": None,
            "translation_reference_profile": None,
            "review_reference_profile": None,
            "translation_story_state": None,
        }

        class FakeLLMClient:
            def extract_reference_patch(self, *args, **kwargs):
                return {
                    "series_notes": ["系列人名一般保留既刊译名"],
                    "style_notes": ["前作对话偏口语，吐槽感要保留"],
                    "characters": [
                        {
                            "name": "小鸟游",
                            "aliases": ["会长"],
                            "role": "学生会长",
                            "usage_note": "系列常用译名",
                            "example_sentences": ["小鸟游会长叹了口气。"],
                        }
                    ],
                    "terms": [
                        {
                            "term": "学生会",
                            "category": "组织",
                            "usage_note": "沿用既刊译法",
                            "example_sentences": ["学生会里一片寂静。"],
                        }
                    ],
                }

            def summarize(self, *args, **kwargs):
                shared["summary_reference_profile"] = copy.deepcopy(kwargs.get("reference_profile"))
                return empty_summary_response()

            def translate(self, *args, **kwargs):
                shared["translation_reference_profile"] = copy.deepcopy(kwargs.get("reference_profile"))
                shared["translation_story_state"] = copy.deepcopy(kwargs.get("story_state"))
                return {segment["id"]: f"[中文] {segment['text']}" for segment in kwargs["segments"]}

            def review(self, *args, **kwargs):
                shared["review_reference_profile"] = copy.deepcopy(kwargs.get("reference_profile"))
                return ok_review_response()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "input.epub"
            reference_path = tmp_path / "reference.epub"
            output_path = tmp_path / "output.epub"
            progress_path = tmp_path / "progress.json"

            build_sample_epub(input_path)
            build_sample_epub(
                reference_path,
                chapters=[("前作", ["小鸟游会长叹了口气。", "学生会里一片寂静。"])],
                title="Series Vol.2",
                language="zh",
            )
            config = make_config(
                input_path=input_path,
                output_path=output_path,
                progress_path=progress_path,
                provider="openai-compatible",
                translation_workers=1,
                reference_input_path=reference_path,
            )

            with patch("translator.pipeline._build_llm_client", side_effect=lambda *_: FakeLLMClient()):
                run_translation_pipeline(config)

            self.assertEqual(shared["summary_reference_profile"]["characters"][0]["name"], "小鸟游")
            self.assertEqual(shared["translation_reference_profile"]["terms"][0]["term"], "学生会")
            self.assertEqual(shared["review_reference_profile"]["style_notes"][0], "前作对话偏口语，吐槽感要保留")
            self.assertEqual(shared["translation_story_state"]["glossary"], [])
            self.assertEqual(shared["translation_story_state"]["style_notes"], [])

            progress = load_progress(progress_path)
            self.assertEqual(progress["reference_phase"]["reference_profile"]["characters"][0]["name"], "小鸟游")
            self.assertTrue(progress["reference_enabled"])

    def test_reference_change_invalidates_progress_and_reruns_pipeline(self):
        class FakeLLMClient:
            def __init__(self, shared, marker):
                self.shared = shared
                self.marker = marker

            def extract_reference_patch(self, *args, **kwargs):
                self.shared["reference_calls"] += 1
                return {
                    "series_notes": [f"marker:{self.marker}"],
                    "style_notes": [],
                    "characters": [],
                    "terms": [],
                }

            def summarize(self, *args, **kwargs):
                self.shared["summary_calls"] += 1
                return empty_summary_response()

            def translate(self, *args, **kwargs):
                self.shared["translate_calls"] += 1
                return {segment["id"]: f"[中文] {segment['text']}" for segment in kwargs["segments"]}

            def review(self, *args, **kwargs):
                return ok_review_response()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "input.epub"
            reference_a = tmp_path / "reference-a.epub"
            reference_b = tmp_path / "reference-b.epub"
            output_path = tmp_path / "output.epub"
            progress_path = tmp_path / "progress.json"

            build_sample_epub(input_path)
            build_sample_epub(reference_a, chapters=[("前作A", ["惯用译名A。"])], title="Series A", language="zh")
            build_sample_epub(reference_b, chapters=[("前作B", ["惯用译名B。"])], title="Series B", language="zh")

            first_shared = {"reference_calls": 0, "summary_calls": 0, "translate_calls": 0}
            with patch(
                "translator.pipeline._build_llm_client",
                side_effect=lambda *_: FakeLLMClient(first_shared, "A"),
            ):
                run_translation_pipeline(
                    make_config(
                        input_path=input_path,
                        output_path=output_path,
                        progress_path=progress_path,
                        provider="openai-compatible",
                        translation_workers=1,
                        reference_input_path=reference_a,
                    )
                )

            first_progress = load_progress(progress_path)
            first_fingerprint = first_progress["reference_fingerprint"]

            second_shared = {"reference_calls": 0, "summary_calls": 0, "translate_calls": 0}
            with patch(
                "translator.pipeline._build_llm_client",
                side_effect=lambda *_: FakeLLMClient(second_shared, "B"),
            ):
                run_translation_pipeline(
                    make_config(
                        input_path=input_path,
                        output_path=output_path,
                        progress_path=progress_path,
                        provider="openai-compatible",
                        translation_workers=1,
                        reference_input_path=reference_b,
                    )
                )

            second_progress = load_progress(progress_path)
            self.assertGreater(second_shared["reference_calls"], 0)
            self.assertGreater(second_shared["summary_calls"], 0)
            self.assertGreater(second_shared["translate_calls"], 0)
            self.assertNotEqual(first_fingerprint, second_progress["reference_fingerprint"])
            self.assertEqual(Path(second_progress["reference_input_path"]).resolve(), reference_b.resolve())

    def test_reference_phase_resume_reuses_completed_reference_documents(self):
        class FailingClient:
            def __init__(self, shared):
                self.shared = shared

            def extract_reference_patch(self, *args, **kwargs):
                joined = " ".join(segment["text"] for segment in kwargs["segments"])
                self.shared["reference_inputs"].append(joined)
                if "参考二" in joined:
                    raise RuntimeError("boom on reference chapter two")
                return {
                    "series_notes": [],
                    "style_notes": [joined[:20]],
                    "characters": [],
                    "terms": [],
                }

            def summarize(self, *args, **kwargs):
                return empty_summary_response()

            def translate(self, *args, **kwargs):
                return {segment["id"]: f"[中文] {segment['text']}" for segment in kwargs["segments"]}

            def review(self, *args, **kwargs):
                return ok_review_response()

        class StableClient(FailingClient):
            def extract_reference_patch(self, *args, **kwargs):
                joined = " ".join(segment["text"] for segment in kwargs["segments"])
                self.shared["reference_inputs"].append(joined)
                return {
                    "series_notes": [],
                    "style_notes": [joined[:20]],
                    "characters": [],
                    "terms": [],
                }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "input.epub"
            reference_path = tmp_path / "reference.epub"
            output_path = tmp_path / "output.epub"
            progress_path = tmp_path / "progress.json"

            build_sample_epub(input_path)
            build_sample_epub(
                reference_path,
                chapters=[
                    ("前作一", ["参考一。"]),
                    ("前作二", ["参考二。"]),
                ],
                title="Series Ref",
                language="zh",
            )
            config = make_config(
                input_path=input_path,
                output_path=output_path,
                progress_path=progress_path,
                provider="openai-compatible",
                translation_workers=1,
                reference_input_path=reference_path,
            )

            first_shared = {"reference_inputs": []}
            with patch("translator.pipeline._build_llm_client", side_effect=lambda *_: FailingClient(first_shared)):
                with self.assertRaisesRegex(RuntimeError, "boom on reference chapter two"):
                    run_translation_pipeline(config)

            progress = load_progress(progress_path)
            self.assertEqual(progress["reference_documents"]["chapter1.xhtml"]["status"], "done")
            self.assertEqual(progress["reference_documents"]["chapter2.xhtml"]["status"], "pending")

            second_shared = {"reference_inputs": []}
            with patch("translator.pipeline._build_llm_client", side_effect=lambda *_: StableClient(second_shared)):
                run_translation_pipeline(config)

            self.assertTrue(any("参考二。" in item for item in second_shared["reference_inputs"]))
            self.assertTrue(all("参考一。" not in item for item in second_shared["reference_inputs"]))

    def test_reference_language_mismatch_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "input.epub"
            reference_path = tmp_path / "reference.epub"
            output_path = tmp_path / "output.epub"
            progress_path = tmp_path / "progress.json"

            build_sample_epub(input_path)
            build_sample_epub(
                reference_path,
                chapters=[("Reference", ["This is an English translation."])],
                title="English Ref",
                language="en",
            )

            config = make_config(
                input_path=input_path,
                output_path=output_path,
                progress_path=progress_path,
                provider="mock",
                translation_workers=1,
                reference_input_path=reference_path,
            )

            with self.assertRaisesRegex(RuntimeError, "参考 EPUB 语言与目标语言不匹配"):
                run_translation_pipeline(config)


if __name__ == "__main__":
    unittest.main()
