import unittest

from translator.epub_utils import apply_translations
from translator.epub_utils import batch_segments
from translator.epub_utils import extract_document_title
from translator.epub_utils import prepare_document


class BatchSegmentsTests(unittest.TestCase):
    def test_batch_segments_respects_segment_count_limit(self):
        segments = [{"id": f"seg_{index:04d}", "text": "a"} for index in range(5)]

        batches = batch_segments(segments, max_batch_chars=100, max_batch_segments=2)

        self.assertEqual([len(batch) for batch in batches], [2, 2, 1])


class PrepareDocumentTests(unittest.TestCase):
    def test_prepare_document_merges_ruby_text_into_single_segment(self):
        class FakeItem:
            file_name = "chapter1.xhtml"
            id = "chapter1"

            def get_content(self):
                return """
        <html xmlns="http://www.w3.org/1999/xhtml">
          <body>
            <p>僕──<ruby>常盤<rt>ときわ</rt>孤<rt>こ</rt>太<rt>た</rt>郎<rt>ろう</rt></ruby>をからかった。</p>
          </body>
        </html>
        """.encode("utf-8")

        plan = prepare_document(FakeItem())

        self.assertEqual(len(plan.segments), 1)
        self.assertEqual(plan.segments[0]["text"], "僕──常盤孤太郎をからかった。")

    def test_extract_document_title_prefers_translated_heading(self):
        html = """
        <html xmlns="http://www.w3.org/1999/xhtml">
          <head><title>原始标题</title></head>
          <body>
            <h1>[中文] 第一章</h1>
            <p>正文。</p>
          </body>
        </html>
        """

        title = extract_document_title(html, fallback="Fallback")

        self.assertEqual(title, "[中文] 第一章")

    def test_prepare_document_falls_back_to_br_separated_text(self):
        class FakeItem:
            file_name = "volume.xhtml"
            id = "volume"

            def get_content(self):
                body = "<h2>目录</h2>" + "<br/>".join(
                    ["第一章", "雨一直下。", "润奈坐在窗边。", "诗暮叹了口气。"] * 80
                )
                return f"""
        <html xmlns="http://www.w3.org/1999/xhtml">
          <body>
            <div>{body}</div>
          </body>
        </html>
        """.encode("utf-8")

        plan = prepare_document(FakeItem())

        self.assertGreater(len(plan.segments), 100)
        self.assertIn("雨一直下。", [segment["text"] for segment in plan.segments])

        translated = {segment["id"]: f"译文{index}" for index, segment in enumerate(plan.segments, start=1)}
        rendered = apply_translations(plan, translated)

        self.assertIn("<p>译文1</p>", rendered)
        self.assertIn('data-epub-tsuyaku-fallback="br-text"', rendered)


if __name__ == "__main__":
    unittest.main()
