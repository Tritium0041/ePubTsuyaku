import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from ebooklib import epub

from translator.webapp import create_app, discover_epub_files


def build_sample_epub(
    path: Path,
    chapters=None,
    *,
    title: str = "UI Demo",
    language: str = "ja",
) -> None:
    chapters = chapters or [("Chapter 1", ["太郎は学校へ行った。"])]

    book = epub.EpubBook()
    book.set_identifier("ui-demo")
    book.set_title(title)
    book.set_language(language)
    book.add_author("Tester")

    chapter_items = []
    for index, (chapter_title, paragraphs) in enumerate(chapters, start=1):
        chapter = epub.EpubHtml(title=chapter_title, file_name=f"chapter{index}.xhtml", lang="ja")
        body = "\n".join(f"<p>{paragraph}</p>" for paragraph in paragraphs)
        chapter.content = f"""
        <html xmlns="http://www.w3.org/1999/xhtml">
          <body>
            <h1>{chapter_title}</h1>
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


class WebAppTests(unittest.TestCase):
    def _wait_for_job_completion(self, client, timeout: float = 8.0):
        deadline = time.time() + timeout
        snapshot = None
        while time.time() < deadline:
            snapshot = client.get("/api/status").get_json()
            if snapshot["job"] and snapshot["job"]["status"] in {"completed", "failed"}:
                return snapshot
            time.sleep(0.1)
        return snapshot

    def test_web_ui_uses_updated_deepseek_defaults(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            app = create_app(project_root=root)
            client = app.test_client()

            response = client.get("/")
            html = response.get_data(as_text=True)

            self.assertEqual(response.status_code, 200)
            self.assertIn('value="deepseek-v4-flash"', html)
            self.assertIn('value="4000"', html)
            self.assertIn('id="reference_existing_file"', html)
            self.assertIn('id="reference_manual_input_path"', html)
            self.assertIn("参考提取", html)
            self.assertIn('id="translation_workers" name="translation_workers" type="number" min="1" step="1" value="4"', html)
            self.assertIn('"deepseek_model": "deepseek-v4-flash"', html)

    def test_discover_epub_files_skips_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "epubOutput").mkdir()
            build_sample_epub(root / "book.epub")
            build_sample_epub(root / "epubOutput" / "translated.epub")

            files = discover_epub_files(root)

            self.assertEqual(len(files), 1)
            self.assertTrue(files[0]["path"].endswith("book.epub"))

    def test_web_ui_can_start_mock_job(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "epubOutput").mkdir()
            sample_path = root / "demo.epub"
            build_sample_epub(sample_path)

            app = create_app(project_root=root)
            client = app.test_client()

            response = client.post(
                "/start",
                data={
                    "existing_file": str(sample_path),
                    "source_language": "日语",
                    "target_language": "中文",
                    "provider_preset": "mock",
                    "model": "mock-model",
                    "translation_workers": "2",
                    "max_batch_chars": "800",
                    "max_batch_segments": "16",
                    "max_review_retries": "0",
                    "min_review_score": "90",
                    "recent_summary_limit": "3",
                    "title_suffix": "（中文译本）",
                    "reset_progress": "on",
                },
                follow_redirects=True,
            )

            self.assertEqual(response.status_code, 200)

            snapshot = self._wait_for_job_completion(client)
            self.assertIsNotNone(snapshot)
            self.assertEqual(snapshot["job"]["status"], "completed")
            output_path = Path(snapshot["job"]["output_path"])
            self.assertTrue(output_path.exists())

    def test_web_ui_can_start_mock_job_with_reference_epub(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "epubOutput").mkdir()
            sample_path = root / "demo.epub"
            reference_path = root / "reference.epub"
            build_sample_epub(sample_path)
            build_sample_epub(reference_path, title="Series Ref", language="zh")

            app = create_app(project_root=root)
            client = app.test_client()

            response = client.post(
                "/start",
                data={
                    "existing_file": str(sample_path),
                    "reference_manual_input_path": str(reference_path),
                    "source_language": "日语",
                    "target_language": "中文",
                    "provider_preset": "mock",
                    "model": "mock-model",
                    "translation_workers": "2",
                    "max_batch_chars": "800",
                    "max_batch_segments": "16",
                    "max_review_retries": "0",
                    "min_review_score": "90",
                    "recent_summary_limit": "3",
                    "title_suffix": "（中文译本）",
                    "reset_progress": "on",
                },
                follow_redirects=True,
            )

            self.assertEqual(response.status_code, 200)

            snapshot = self._wait_for_job_completion(client)
            self.assertIsNotNone(snapshot)
            self.assertEqual(snapshot["job"]["status"], "completed")
            self.assertTrue(snapshot["job"]["reference_enabled"])
            self.assertTrue(snapshot["job"]["reference_input_path"].endswith("reference.epub"))

    def test_web_ui_preserves_last_form_values_after_redirect(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "epubOutput").mkdir()
            sample_path = root / "demo.epub"
            build_sample_epub(sample_path)

            app = create_app(project_root=root)
            client = app.test_client()

            response = client.post(
                "/start",
                data={
                    "manual_input_path": str(sample_path),
                    "source_language": "日语",
                    "target_language": "简体中文",
                    "provider_preset": "mock",
                    "model": "mock-model",
                    "base_url": "",
                    "translation_workers": "7",
                    "max_batch_chars": "900",
                    "max_batch_segments": "23",
                    "max_review_retries": "1",
                    "min_review_score": "88",
                    "recent_summary_limit": "4",
                    "title_suffix": "（测试译本）",
                    "reset_progress": "on",
                },
                follow_redirects=True,
            )

            html = response.get_data(as_text=True)

            self.assertEqual(response.status_code, 200)
            self.assertIn(f'value="{sample_path}"', html)
            self.assertIn('<option value="mock" selected>', html)
            self.assertIn('value="简体中文"', html)
            self.assertIn('id="translation_workers" name="translation_workers" type="number" min="1" step="1" value="7"', html)
            self.assertIn('value="900"', html)
            self.assertIn('value="23"', html)
            self.assertIn('value="（测试译本）"', html)
            snapshot = self._wait_for_job_completion(client)
            self.assertIsNotNone(snapshot)
            self.assertEqual(snapshot["job"]["status"], "completed")

    def test_uploaded_epub_resume_reuses_progress_on_same_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "epubOutput").mkdir()
            sample_path = root / "demo.epub"
            build_sample_epub(sample_path)
            upload_bytes = sample_path.read_bytes()

            app = create_app(project_root=root)
            client = app.test_client()

            first_response = client.post(
                "/start",
                data={
                    "upload_file": (io.BytesIO(upload_bytes), "demo.epub"),
                    "source_language": "日语",
                    "target_language": "中文",
                    "provider_preset": "mock",
                    "model": "mock-model",
                    "translation_workers": "1",
                    "max_batch_chars": "800",
                    "max_batch_segments": "16",
                    "max_review_retries": "0",
                    "min_review_score": "90",
                    "recent_summary_limit": "3",
                    "title_suffix": "（中文译本）",
                    "reset_progress": "on",
                },
                content_type="multipart/form-data",
                follow_redirects=True,
            )

            self.assertEqual(first_response.status_code, 200)
            first_snapshot = self._wait_for_job_completion(client)
            self.assertIsNotNone(first_snapshot)
            self.assertEqual(first_snapshot["job"]["status"], "completed")

            second_response = client.post(
                "/start",
                data={
                    "upload_file": (io.BytesIO(upload_bytes), "demo.epub"),
                    "source_language": "日语",
                    "target_language": "中文",
                    "provider_preset": "mock",
                    "model": "mock-model",
                    "translation_workers": "1",
                    "max_batch_chars": "800",
                    "max_batch_segments": "16",
                    "max_review_retries": "0",
                    "min_review_score": "90",
                    "recent_summary_limit": "3",
                    "title_suffix": "（中文译本）",
                },
                content_type="multipart/form-data",
                follow_redirects=True,
            )

            self.assertEqual(second_response.status_code, 200)
            second_snapshot = self._wait_for_job_completion(client)
            self.assertIsNotNone(second_snapshot)
            self.assertEqual(second_snapshot["job"]["status"], "completed")
            self.assertEqual(second_snapshot["job"]["progress_path"], first_snapshot["job"]["progress_path"])
            self.assertEqual(second_snapshot["job"]["input_path"], first_snapshot["job"]["input_path"])
            self.assertEqual(second_snapshot["job"]["output_path"], first_snapshot["job"]["output_path"])
            self.assertEqual(second_snapshot["job"]["processed_count"], 0)
            self.assertEqual(second_snapshot["job"]["skipped_count"], 1)

    def test_web_ui_auto_resumes_retryable_failures(self):
        shared = {"failed_once": False}

        class FlakyClient:
            def extract_reference_patch(self, *args, **kwargs):
                return {"series_notes": [], "style_notes": [], "characters": [], "terms": []}

            def summarize(self, *args, **kwargs):
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

            def translate(self, *args, **kwargs):
                joined = " ".join(segment["text"] for segment in kwargs["segments"])
                if not shared["failed_once"] and "第二段。" in joined:
                    shared["failed_once"] = True
                    raise RuntimeError("模型连续 3 次未返回合法 JSON: mock boom")
                return {segment["id"]: f"[中文] {segment['text']}" for segment in kwargs["segments"]}

            def review(self, *args, **kwargs):
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

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "epubOutput").mkdir()
            sample_path = root / "demo.epub"
            build_sample_epub(
                sample_path,
                chapters=[("第一章", ["第一段。", "第二段。", "第三段。"])],
            )

            app = create_app(project_root=root)
            client = app.test_client()

            with patch("translator.pipeline._build_llm_client", side_effect=lambda *_: FlakyClient()):
                response = client.post(
                    "/start",
                    data={
                        "existing_file": str(sample_path),
                        "source_language": "日语",
                        "target_language": "中文",
                        "provider_preset": "openai-compatible",
                        "api_key": "test-key",
                        "base_url": "https://example.com/v1",
                        "model": "demo-model",
                        "translation_workers": "1",
                        "max_batch_chars": "800",
                        "max_batch_segments": "1",
                        "max_review_retries": "0",
                        "min_review_score": "90",
                        "recent_summary_limit": "3",
                        "title_suffix": "（中文译本）",
                        "reset_progress": "on",
                    },
                    follow_redirects=True,
                )

                self.assertEqual(response.status_code, 200)
                snapshot = self._wait_for_job_completion(client)

            self.assertIsNotNone(snapshot)
            self.assertEqual(snapshot["job"]["status"], "completed")
            self.assertEqual(snapshot["job"]["result"]["auto_resume_count"], 1)
            self.assertTrue(any("准备自动续跑" in line for line in snapshot["job"]["logs"]))
            self.assertTrue(any("自动续跑成功" in line for line in snapshot["job"]["logs"]))


if __name__ == "__main__":
    unittest.main()
