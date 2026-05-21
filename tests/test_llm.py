import unittest

from translator.llm import OpenAICompatibleLLMClient, _extract_translation_map, _parse_json_from_text


class TranslationPayloadTests(unittest.TestCase):
    def test_parse_json_from_text_tolerates_trailing_garbage_after_object(self):
        payload = _parse_json_from_text('{"ok": true} trailing noise')

        self.assertEqual(payload, {"ok": True})

    def test_extract_translation_map_supports_keyed_object(self):
        payload = {"translations": {"seg_0001": "第一句", "seg_0002": "第二句"}}

        translation_map = _extract_translation_map(payload)

        self.assertEqual(translation_map["seg_0001"], "第一句")
        self.assertEqual(translation_map["seg_0002"], "第二句")

    def test_translate_repairs_missing_segments_with_followup_request(self):
        client = OpenAICompatibleLLMClient(
            api_key="test-key",
            base_url="https://example.com/v1",
            model="demo-model",
        )
        responses = iter(
            [
                {"translations": {"seg_0001": "第一句"}},
                {"translations": {"seg_0002": "第二句"}},
            ]
        )
        requested_tool_names = []

        def fake_call_json(
            system_prompt,
            user_prompt,
            model,
            temperature,
            max_tokens=None,
            schema=None,
            tool_name="return_json",
            tool_description="",
        ):
            requested_tool_names.append(tool_name)
            return next(responses)

        client._call_json = fake_call_json  # type: ignore[method-assign]

        result = client.translate(
            book_metadata={"title": "Demo", "author": "Tester", "identifier": "demo", "language": "ja"},
            story_state={},
            segments=[
                {"id": "seg_0001", "text": "太郎は学校へ行った。"},
                {"id": "seg_0002", "text": "花子に会った。"},
            ],
            source_language="日语",
            target_language="中文",
        )

        self.assertEqual(
            result,
            {
                "seg_0001": "第一句",
                "seg_0002": "第二句",
            },
        )
        self.assertEqual(requested_tool_names, ["return_translations", "return_translations"])


if __name__ == "__main__":
    unittest.main()
