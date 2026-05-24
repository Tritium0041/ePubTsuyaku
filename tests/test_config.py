import os
import unittest
from unittest.mock import patch

from translator.config import resolve_provider_settings


class ProviderSettingsTests(unittest.TestCase):
    def test_auto_provider_prefers_epub_tsuyaku_environment(self):
        with patch.dict(
            os.environ,
            {
                "EPUB_TSUYAKU_API_KEY": "new-key",
                "EPUB_TSUYAKU_BASE_URL": "https://example.com/v1",
                "EPUB_TSUYAKU_MODEL": "new-model",
                "EPUB_TRANSLATOR_API_KEY": "legacy-key",
                "EPUB_TRANSLATOR_BASE_URL": "https://legacy.example.com/v1",
                "EPUB_TRANSLATOR_MODEL": "legacy-model",
            },
            clear=True,
        ):
            provider, api_key, base_url, model = resolve_provider_settings(
                provider="auto",
                api_key_env=None,
                explicit_base_url=None,
                explicit_model=None,
            )

        self.assertEqual(provider, "openai-compatible")
        self.assertEqual(api_key, "new-key")
        self.assertEqual(base_url, "https://example.com/v1")
        self.assertEqual(model, "new-model")

    def test_auto_provider_keeps_legacy_epub_translator_environment(self):
        with patch.dict(
            os.environ,
            {
                "EPUB_TRANSLATOR_API_KEY": "legacy-key",
                "EPUB_TRANSLATOR_BASE_URL": "https://legacy.example.com/v1",
                "EPUB_TRANSLATOR_MODEL": "legacy-model",
            },
            clear=True,
        ):
            provider, api_key, base_url, model = resolve_provider_settings(
                provider="auto",
                api_key_env=None,
                explicit_base_url=None,
                explicit_model=None,
            )

        self.assertEqual(provider, "openai-compatible")
        self.assertEqual(api_key, "legacy-key")
        self.assertEqual(base_url, "https://legacy.example.com/v1")
        self.assertEqual(model, "legacy-model")


if __name__ == "__main__":
    unittest.main()
