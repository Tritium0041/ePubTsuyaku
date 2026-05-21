from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

from .config import PipelineConfig, resolve_provider_settings
from .pipeline import run_translation_pipeline_with_retries


def _find_default_input(project_root: Path) -> Path:
    test_book_dir = project_root / "testBook"
    if not test_book_dir.exists():
        raise RuntimeError("请通过 --input 指定待翻译 epub 文件。")

    epub_files = sorted(test_book_dir.glob("*.epub"))
    if not epub_files:
        raise RuntimeError("testBook/ 下没有找到 epub 文件，请通过 --input 指定。")
    return epub_files[0]


def _sanitize_filename_part(value: str) -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", value.strip(), flags=re.UNICODE)
    return cleaned.strip("_") or "translated"


def _resolve_output_path(project_root: Path, input_path: Path, output_arg: Optional[str], target_language: str) -> Path:
    suffix = _sanitize_filename_part(target_language)
    if not output_arg:
        return project_root / "epubOutput" / f"{input_path.stem}.{suffix}.epub"

    output_path = Path(output_arg)
    if not output_path.is_absolute():
        output_path = project_root / output_path

    if output_path.suffix.lower() == ".epub":
        return output_path
    return output_path / f"{input_path.stem}.{suffix}.epub"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="使用 LLM 对 EPUB 进行章节级翻译、校对和重建。")
    parser.add_argument("--input", help="待翻译的 epub 路径。默认读取 testBook/ 下的第一个 epub。")
    parser.add_argument("--reference-epub", help="可选：前作精翻参考 epub 路径。")
    parser.add_argument("--output", help="输出 epub 路径，默认写入 epubOutput/。")
    parser.add_argument("--source-lang", default="日语", help="源语言，默认日语。")
    parser.add_argument("--target-lang", default="中文", help="目标语言，默认中文。")
    parser.add_argument(
        "--provider",
        choices=("auto", "openai-compatible", "mock"),
        default="auto",
        help="LLM 提供方。mock 用于本地联调。",
    )
    parser.add_argument("--api-key-env", help="指定从哪个环境变量读取 API Key。")
    parser.add_argument("--base-url", help="OpenAI 兼容接口的 base_url。")
    parser.add_argument("--model", help="默认模型名。")
    parser.add_argument("--summary-model", help="要素提取阶段模型名。")
    parser.add_argument("--translation-model", help="初翻阶段模型名。")
    parser.add_argument("--review-model", help="校对阶段模型名。")
    parser.add_argument("--translation-workers", type=int, default=4, help="翻译阶段的并发 worker 数。")
    parser.add_argument("--auto-resume-retries", type=int, default=2, help="任务失败时自动续跑的额外次数。")
    parser.add_argument("--progress-file", default="progress.json", help="断点续跑文件路径。")
    parser.add_argument("--max-batch-chars", type=int, default=3500, help="单批次最多传给模型的字符数。")
    parser.add_argument("--max-batch-segments", type=int, default=64, help="单批次最多包含的片段数。")
    parser.add_argument("--max-review-retries", type=int, default=2, help="校对不达标时的最大重试次数。")
    parser.add_argument("--min-review-score", type=int, default=85, help="低于该分数时触发重试。")
    parser.add_argument("--recent-summary-limit", type=int, default=5, help="向后续批次传递的最近摘要数量。")
    parser.add_argument("--title-suffix", help="输出书名后缀，例如 （中文译本）。")
    parser.add_argument("--reset-progress", action="store_true", help="忽略现有 progress 文件，从头开始。")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[1]
    input_path = Path(args.input) if args.input else _find_default_input(project_root)
    if not input_path.is_absolute():
        input_path = project_root / input_path
    if not input_path.exists():
        raise RuntimeError(f"输入文件不存在: {input_path}")

    reference_input_path = Path(args.reference_epub) if args.reference_epub else None
    if reference_input_path is not None:
        if not reference_input_path.is_absolute():
            reference_input_path = project_root / reference_input_path
        if not reference_input_path.exists():
            raise RuntimeError(f"参考输入文件不存在: {reference_input_path}")

    output_path = _resolve_output_path(project_root, input_path, args.output, args.target_lang)
    progress_path = Path(args.progress_file)
    if not progress_path.is_absolute():
        progress_path = project_root / progress_path

    provider_name, api_key, base_url, default_model = resolve_provider_settings(
        provider=args.provider,
        api_key_env=args.api_key_env,
        explicit_base_url=args.base_url,
        explicit_model=args.model,
    )
    model = args.model or default_model

    config = PipelineConfig(
        input_path=input_path,
        output_path=output_path,
        progress_path=progress_path,
        reference_input_path=reference_input_path,
        source_language=args.source_lang,
        target_language=args.target_lang,
        provider=provider_name,
        api_key=api_key,
        base_url=base_url,
        model=model,
        summary_model=args.summary_model or model,
        translation_model=args.translation_model or model,
        review_model=args.review_model or model,
        translation_workers=max(1, args.translation_workers),
        auto_resume_retries=max(0, args.auto_resume_retries),
        max_batch_chars=args.max_batch_chars,
        max_batch_segments=args.max_batch_segments,
        max_review_retries=args.max_review_retries,
        min_review_score=args.min_review_score,
        recent_summary_limit=args.recent_summary_limit,
        title_suffix=args.title_suffix,
        reset_progress=args.reset_progress,
    )

    result = run_translation_pipeline_with_retries(config)
    print("")
    print("Translation finished:")
    print(f"  input: {result['input_path']}")
    print(f"  output: {result['output_path']}")
    print(f"  progress: {result['progress_path']}")
    print(f"  run attempts: {result.get('run_attempt_count', 1)}")
    print(f"  auto resumes: {result.get('auto_resume_count', 0)}")
    print(f"  processed documents: {result['processed_count']}")
    print(f"  reused documents: {result['skipped_count']}")
    print(f"  retried batches: {result['retried_batches']}")
    return 0
