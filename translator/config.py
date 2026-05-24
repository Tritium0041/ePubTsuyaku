from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


ALIYUN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_BETA_BASE_URL = "https://api.deepseek.com/beta"
PROJECT_API_KEY_ENV = "EPUB_TSUYAKU_API_KEY"
PROJECT_BASE_URL_ENV = "EPUB_TSUYAKU_BASE_URL"
PROJECT_MODEL_ENV = "EPUB_TSUYAKU_MODEL"
LEGACY_PROJECT_API_KEY_ENV = "EPUB_TRANSLATOR_API_KEY"
LEGACY_PROJECT_BASE_URL_ENV = "EPUB_TRANSLATOR_BASE_URL"
LEGACY_PROJECT_MODEL_ENV = "EPUB_TRANSLATOR_MODEL"


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
    review_workers: int = 0
    reference_workers: int = 0
    auto_resume_retries: int = 2
    max_batch_chars: int = 3500
    max_batch_segments: int = 64
    max_review_retries: int = 2
    min_review_score: int = 85
    recent_summary_limit: int = 5
    title_suffix: Optional[str] = None
    reset_progress: bool = False


def first_env(*names: str) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def project_model(default: str, *, include_openai: bool = True) -> str:
    names = [PROJECT_MODEL_ENV, LEGACY_PROJECT_MODEL_ENV]
    if include_openai:
        names.append("OPENAI_MODEL")
    return first_env(*names) or default


def project_base_url(*, include_openai: bool = True) -> Optional[str]:
    names = [PROJECT_BASE_URL_ENV, LEGACY_PROJECT_BASE_URL_ENV]
    if include_openai:
        names.append("OPENAI_BASE_URL")
    return first_env(*names)


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
        model = explicit_model or project_model("qwen-max")
        base_url = explicit_base_url or project_base_url()
        return "openai-compatible", api_key, base_url, model

    env_api_key = first_env(PROJECT_API_KEY_ENV, LEGACY_PROJECT_API_KEY_ENV)
    if env_api_key:
        model = explicit_model or project_model("qwen-max")
        base_url = explicit_base_url or project_base_url()
        return "openai-compatible", env_api_key, base_url, model

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        model = explicit_model or project_model("gpt-4.1-mini")
        base_url = explicit_base_url or project_base_url()
        return "openai-compatible", openai_api_key, base_url, model

    aliyun_api_key = os.environ.get("ALIYUN_API_KEY")
    if aliyun_api_key:
        model = explicit_model or project_model("qwen-max", include_openai=False)
        base_url = explicit_base_url or project_base_url(include_openai=False) or ALIYUN_BASE_URL
        return "openai-compatible", aliyun_api_key, base_url, model

    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
    if deepseek_api_key:
        model = explicit_model or project_model("deepseek-v4-flash", include_openai=False)
        base_url = explicit_base_url or project_base_url(include_openai=False) or DEEPSEEK_BASE_URL
        return "openai-compatible", deepseek_api_key, base_url, model

    raise RuntimeError(
        "没有找到可用的 API Key。请设置 EPUB_TSUYAKU_API_KEY / OPENAI_API_KEY / ALIYUN_API_KEY / DEEPSEEK_API_KEY，或使用 --provider mock。"
    )
