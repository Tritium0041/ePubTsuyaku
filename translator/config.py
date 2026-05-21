from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


ALIYUN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_BETA_BASE_URL = "https://api.deepseek.com/beta"


@dataclass
class PipelineConfig:
    input_path: Path
    output_path: Path
    progress_path: Path
    reference_input_path: Optional[Path]
    source_language: str
    target_language: str
    provider: str
    api_key: Optional[str]
    base_url: Optional[str]
    model: str
    summary_model: str
    translation_model: str
    review_model: str
    translation_workers: int = 4
    auto_resume_retries: int = 2
    max_batch_chars: int = 3500
    max_batch_segments: int = 64
    max_review_retries: int = 2
    min_review_score: int = 85
    recent_summary_limit: int = 5
    title_suffix: Optional[str] = None
    reset_progress: bool = False


def resolve_provider_settings(
    provider: str,
    api_key_env: Optional[str],
    explicit_base_url: Optional[str],
    explicit_model: Optional[str],
) -> tuple[str, Optional[str], Optional[str], str]:
    if provider == "mock":
        return "mock", None, None, explicit_model or "mock-model"

    if api_key_env:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"环境变量 {api_key_env} 未设置。")
        model = explicit_model or os.environ.get("EPUB_TRANSLATOR_MODEL") or os.environ.get("OPENAI_MODEL") or "qwen-max"
        base_url = explicit_base_url or os.environ.get("EPUB_TRANSLATOR_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        return "openai-compatible", api_key, base_url, model

    env_api_key = os.environ.get("EPUB_TRANSLATOR_API_KEY")
    if env_api_key:
        model = explicit_model or os.environ.get("EPUB_TRANSLATOR_MODEL") or os.environ.get("OPENAI_MODEL") or "qwen-max"
        base_url = explicit_base_url or os.environ.get("EPUB_TRANSLATOR_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        return "openai-compatible", env_api_key, base_url, model

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        model = explicit_model or os.environ.get("OPENAI_MODEL") or "gpt-4.1-mini"
        base_url = explicit_base_url or os.environ.get("OPENAI_BASE_URL")
        return "openai-compatible", openai_api_key, base_url, model

    aliyun_api_key = os.environ.get("ALIYUN_API_KEY")
    if aliyun_api_key:
        model = explicit_model or os.environ.get("EPUB_TRANSLATOR_MODEL") or "qwen-max"
        base_url = explicit_base_url or os.environ.get("EPUB_TRANSLATOR_BASE_URL") or ALIYUN_BASE_URL
        return "openai-compatible", aliyun_api_key, base_url, model

    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
    if deepseek_api_key:
        model = explicit_model or os.environ.get("EPUB_TRANSLATOR_MODEL") or "deepseek-v4-flash"
        base_url = explicit_base_url or os.environ.get("EPUB_TRANSLATOR_BASE_URL") or DEEPSEEK_BASE_URL
        return "openai-compatible", deepseek_api_key, base_url, model

    raise RuntimeError(
        "没有找到可用的 API Key。请设置 EPUB_TRANSLATOR_API_KEY / OPENAI_API_KEY / ALIYUN_API_KEY / DEEPSEEK_API_KEY，或使用 --provider mock。"
    )
