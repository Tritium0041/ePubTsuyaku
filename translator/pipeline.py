from __future__ import annotations

import copy
import hashlib
import re
import threading
from concurrent.futures import FIRST_COMPLETED, CancelledError, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import ebooklib
from ebooklib import epub

from .config import PipelineConfig
from .epub_utils import (
    apply_translations,
    batch_segments,
    extract_document_title,
    extract_book_metadata,
    iter_spine_documents,
    prepare_document,
    set_item_content,
)
from .llm import MockLLMClient, OpenAICompatibleLLMClient
from .state import (
    create_progress_document,
    empty_reference_patch,
    empty_summary_patch,
    get_reference_document_record,
    get_document_record,
    load_progress,
    make_batch_key,
    merge_reference_profile,
    merge_story_state,
    new_reference_profile,
    new_story_state,
    reference_profile_for_prompt,
    save_progress,
    story_state_for_prompt,
    upsert_reference_document_record,
    upsert_document_record,
)

DC_NAMESPACE = "http://purl.org/dc/elements/1.1/"
OUTPUT_LAYOUT_CSS_ID = "epub_tsuyaku_output_layout"
OUTPUT_LAYOUT_CSS_NAME = "Styles/epub-tsuyaku-output-layout.css"
OUTPUT_LAYOUT_CSS = """
html,
body {
  direction: ltr;
  writing-mode: horizontal-tb;
  -webkit-writing-mode: horizontal-tb;
}
""".strip()

RETRYABLE_RUN_ERROR_SNIPPETS = (
    "严格 schema 调用失败",
    "json mode",
    "合法 json",
    "合法翻译 json",
    "json解析",
    "json 解析",
    "jsondecodeerror",
    "unterminated string",
    "extra data",
    "expecting value",
    "invalid control character",
    "缺少片段译文",
    "仍有未补齐的片段译文",
    "apiconnectionerror",
    "connection error",
    "read timeout",
    "timeout",
    "timed out",
    "ratelimit",
    "rate limit",
    "too many requests",
    "server error",
    "service unavailable",
    "temporarily unavailable",
    "overloaded",
)

NON_RETRYABLE_RUN_ERROR_SNIPPETS = (
    "缺少 api key",
    "没有找到可用的 api key",
    "需要填写 api key",
    "环境变量",
    "输入文件不存在",
    "参考输入文件不存在",
    "参考 epub 不存在",
    "参考输入文件必须是 .epub",
    "参考 epub 语言与目标语言不匹配",
    "请通过 --input 指定待翻译 epub 文件",
    "testbook/ 下没有找到 epub 文件",
    "请至少选择一个 epub 文件",
)

RETRYABLE_EXCEPTION_NAME_SNIPPETS = (
    "jsondecodeerror",
    "timeout",
    "connection",
    "ratelimit",
    "internalservererror",
    "servererror",
    "serviceunavailable",
)

STRUCTURED_OUTPUT_FAILURE_SNIPPETS = (
    "严格 schema 调用失败",
    "json mode",
    "合法 json",
    "没有找到完整的 json",
    "unterminated string",
    "extra data",
    "expecting value",
    "invalid control character",
)


@dataclass
class PreparedDocument:
    index: int
    item: Any
    plan: Any
    batches: List[List[Dict[str, str]]]
    record: Dict[str, Any]


@dataclass
class ReferenceContext:
    enabled: bool
    input_path: Optional[Path]
    fingerprint: str
    book: Any
    book_metadata: Dict[str, str]


def _build_llm_client(config: PipelineConfig):
    if config.provider == "mock":
        return MockLLMClient()
    if not config.api_key:
        raise RuntimeError("缺少 API Key。")
    return OpenAICompatibleLLMClient(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
        summary_model=config.summary_model,
        translation_model=config.translation_model,
        review_model=config.review_model,
    )


def _compute_file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _language_family(value: str) -> str:
    lowered = (value or "").strip().lower()
    if not lowered:
        return ""
    if any(token in lowered for token in ("zh", "chinese", "中文", "汉语", "漢語", "简体", "繁体", "繁體")):
        return "zh"
    if any(token in lowered for token in ("ja", "jp", "japanese", "日语", "日文", "日本語")):
        return "ja"
    if any(token in lowered for token in ("en", "english", "英语", "英文")):
        return "en"
    if any(token in lowered for token in ("ko", "korean", "韩语", "韓語", "한국어")):
        return "ko"
    return ""


def _validate_reference_book_language(reference_book_metadata: Dict[str, str], target_language: str) -> None:
    metadata_language = str(reference_book_metadata.get("language") or "")
    metadata_family = _language_family(metadata_language)
    target_family = _language_family(target_language)
    if metadata_family and target_family and metadata_family != target_family:
        raise RuntimeError(
            f"参考 EPUB 语言与目标语言不匹配：reference={metadata_language or metadata_family}, target={target_language}"
        )


def _build_reference_context(config: PipelineConfig) -> ReferenceContext:
    if config.reference_input_path is None:
        return ReferenceContext(
            enabled=False,
            input_path=None,
            fingerprint="",
            book=None,
            book_metadata={},
        )

    reference_path = config.reference_input_path.expanduser().resolve()
    if not reference_path.exists():
        raise RuntimeError(f"参考 EPUB 不存在: {reference_path}")
    if reference_path.suffix.lower() != ".epub":
        raise RuntimeError("参考输入文件必须是 .epub。")

    reference_book = epub.read_epub(str(reference_path))
    reference_book_metadata = extract_book_metadata(reference_book)
    _validate_reference_book_language(reference_book_metadata, config.target_language)
    reference_hash = _compute_file_sha1(reference_path)
    reference_fingerprint = hashlib.sha1(
        f"{reference_hash}\n{config.target_language}".encode("utf-8")
    ).hexdigest()
    return ReferenceContext(
        enabled=True,
        input_path=reference_path,
        fingerprint=reference_fingerprint,
        book=reference_book,
        book_metadata=reference_book_metadata,
    )


def _create_progress_for_run(
    config: PipelineConfig,
    book_metadata: Dict[str, str],
    reference_context: ReferenceContext,
) -> Dict[str, Any]:
    return create_progress_document(
        input_path=config.input_path,
        output_path=config.output_path,
        source_language=config.source_language,
        target_language=config.target_language,
        book_metadata=book_metadata,
        reference_enabled=reference_context.enabled,
        reference_input_path=reference_context.input_path,
        reference_fingerprint=reference_context.fingerprint,
        reference_book=reference_context.book_metadata,
    )


def _validate_or_create_progress(
    config: PipelineConfig,
    book_metadata: Dict[str, str],
    reference_context: ReferenceContext,
) -> Dict[str, Any]:
    if config.reset_progress and config.progress_path.exists():
        config.progress_path.unlink()

    progress = load_progress(config.progress_path)
    if progress is None:
        return _create_progress_for_run(config, book_metadata, reference_context)

    same_task = (
        progress.get("input_path") == str(config.input_path)
        and progress.get("source_language") == config.source_language
        and progress.get("target_language") == config.target_language
        and bool(progress.get("reference_enabled")) == reference_context.enabled
        and str(progress.get("reference_fingerprint") or "") == reference_context.fingerprint
    )
    if not same_task:
        return _create_progress_for_run(config, book_metadata, reference_context)

    progress["output_path"] = str(config.output_path)
    progress["book"] = dict(book_metadata)
    progress["reference_enabled"] = reference_context.enabled
    progress["reference_input_path"] = str(reference_context.input_path) if reference_context.input_path else ""
    progress["reference_fingerprint"] = reference_context.fingerprint
    progress["reference_book"] = dict(reference_context.book_metadata)
    same_pair = (
        progress.get("source_language") == config.source_language
        and progress.get("target_language") == config.target_language
    )
    if not same_pair:
        return _create_progress_for_run(config, book_metadata, reference_context)
    return progress


def _iter_exception_chain(exc: BaseException) -> List[BaseException]:
    chain: List[BaseException] = []
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return chain


def _exception_summary(exc: BaseException) -> str:
    message = str(exc).strip()
    return message or exc.__class__.__name__


def is_retryable_run_error(exc: BaseException) -> bool:
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        return False

    chain = _iter_exception_chain(exc)
    lowered_messages = [f"{item.__class__.__name__}: {item}".lower() for item in chain]
    lowered_names = [item.__class__.__name__.lower() for item in chain]

    for message in lowered_messages:
        if any(snippet in message for snippet in NON_RETRYABLE_RUN_ERROR_SNIPPETS):
            return False

    if any(any(snippet in name for snippet in RETRYABLE_EXCEPTION_NAME_SNIPPETS) for name in lowered_names):
        return True

    return any(
        any(snippet in message for snippet in RETRYABLE_RUN_ERROR_SNIPPETS)
        for message in lowered_messages
    )


def _is_structured_output_failure(exc: BaseException) -> bool:
    return any(
        any(snippet in f"{item.__class__.__name__}: {item}".lower() for snippet in STRUCTURED_OUTPUT_FAILURE_SNIPPETS)
        for item in _iter_exception_chain(exc)
    )


def _translated_segments_as_list(batch: List[Dict[str, str]], translated_map: Dict[str, str]) -> List[Dict[str, str]]:
    return [{"id": segment["id"], "translation": translated_map[segment["id"]]} for segment in batch]


def _apply_review_corrections(translated_map: Dict[str, str], review_payload: Dict[str, Any]) -> Dict[str, str]:
    corrected = dict(translated_map)
    for segment_id, translation in (review_payload.get("corrected_segments") or {}).items():
        corrected[segment_id] = translation
    return corrected


def _ensure_book_item_identifiers(book: Any) -> None:
    for index, item in enumerate(book.get_items(), start=1):
        identifier = getattr(item, "id", None) or getattr(item, "uid", None) or getattr(item, "file_name", None)
        if not identifier:
            identifier = f"item_{index}"
        identifier = re.sub(r"[^0-9A-Za-z_]+", "_", str(identifier)).strip("_") or f"item_{index}"
        if hasattr(item, "id") and not getattr(item, "id", None):
            item.id = identifier
        if hasattr(item, "uid") and not getattr(item, "uid", None):
            item.uid = identifier


def _assign_identifier(target: Any, fallback: str) -> None:
    identifier = getattr(target, "uid", None) or getattr(target, "id", None) or getattr(target, "href", None) or fallback
    identifier = re.sub(r"[^0-9A-Za-z_]+", "_", str(identifier)).strip("_") or fallback
    if hasattr(target, "id") and not getattr(target, "id", None):
        target.id = identifier
    if hasattr(target, "uid") and not getattr(target, "uid", None):
        target.uid = identifier


def _normalize_toc(entries: Any, prefix: str = "toc") -> Any:
    normalized = []
    for index, entry in enumerate(entries or [], start=1):
        fallback = f"{prefix}_{index}"
        if isinstance(entry, tuple) and len(entry) == 2:
            head, children = entry
            _assign_identifier(head, fallback)
            normalized.append((head, tuple(_normalize_toc(children, prefix=fallback))))
            continue
        _assign_identifier(entry, fallback)
        normalized.append(entry)
    return normalized


def _toc_target_candidates(entry: Any) -> List[str]:
    candidates: List[str] = []
    href = getattr(entry, "href", None)
    file_name = getattr(entry, "file_name", None)
    item_id = getattr(entry, "id", None) or getattr(entry, "uid", None)
    for value in (href, file_name, item_id):
        if not value:
            continue
        text = str(value)
        candidates.append(text)
        if "#" in text:
            candidates.append(text.split("#", 1)[0])
    return candidates


def _translate_toc_entry(entry: Any, title_lookup: Dict[str, str], fallback: str) -> Any:
    translated_title = ""
    for candidate in _toc_target_candidates(entry):
        translated_title = title_lookup.get(candidate, "")
        if translated_title:
            break
    translated_title = translated_title or getattr(entry, "title", None) or fallback

    if isinstance(entry, epub.Link):
        translated = epub.Link(entry.href, translated_title, getattr(entry, "uid", None))
        _assign_identifier(translated, fallback)
        return translated

    if isinstance(entry, epub.Section):
        translated = epub.Section(translated_title, getattr(entry, "href", "") or "")
        _assign_identifier(translated, fallback)
        return translated

    if hasattr(entry, "title"):
        entry.title = translated_title
    _assign_identifier(entry, fallback)
    return entry


def _rewrite_toc_titles(entries: Any, title_lookup: Dict[str, str], prefix: str = "toc") -> Any:
    rewritten = []
    for index, entry in enumerate(entries or [], start=1):
        fallback = f"{prefix}_{index}"
        if isinstance(entry, tuple) and len(entry) == 2:
            head, children = entry
            rewritten_head = _translate_toc_entry(head, title_lookup, fallback)
            rewritten_children = tuple(_rewrite_toc_titles(children, title_lookup, prefix=fallback))
            rewritten.append((rewritten_head, rewritten_children))
            continue
        rewritten.append(_translate_toc_entry(entry, title_lookup, fallback))
    return rewritten


def _build_translated_title_lookup(book: Any) -> Dict[str, str]:
    title_lookup: Dict[str, str] = {}
    for item in book.get_items():
        file_name = getattr(item, "file_name", None)
        item_id = getattr(item, "id", None) or getattr(item, "uid", None)
        if not file_name and not item_id:
            continue
        translated_title = extract_document_title(
            item.get_content() if hasattr(item, "get_content") else "",
            fallback=getattr(item, "title", "") or "",
        )
        if not translated_title:
            continue
        if hasattr(item, "title"):
            item.title = translated_title
        if file_name:
            title_lookup[str(file_name)] = translated_title
        if item_id:
            title_lookup[str(item_id)] = translated_title
    return title_lookup


def _set_book_title(book: Any, title: str) -> None:
    book.title = title
    metadata = getattr(book, "metadata", None)
    if isinstance(metadata, dict):
        metadata.setdefault(DC_NAMESPACE, {})
        metadata[DC_NAMESPACE]["title"] = [(title, {})]


def _normalize_output_language_code(target_language: str) -> str:
    normalized = str(target_language or "").strip().lower()
    if any(token in normalized for token in ("中文", "汉语", "漢語", "chinese", "zh")):
        return "zh-CN" if "繁" not in normalized and "tw" not in normalized and "hk" not in normalized else "zh-TW"
    return target_language or "zh-CN"


def _prepare_output_book_metadata(book: Any, target_language: str) -> None:
    language_code = _normalize_output_language_code(target_language)
    if hasattr(book, "set_language"):
        book.set_language(language_code)
    else:
        book.language = language_code

    metadata = getattr(book, "metadata", None)
    if isinstance(metadata, dict):
        metadata.setdefault(DC_NAMESPACE, {})
        metadata[DC_NAMESPACE]["language"] = [(language_code, {})]

    if hasattr(book, "set_direction"):
        book.set_direction("ltr")
    else:
        book.direction = "ltr"

    layout_item = None
    if hasattr(book, "get_item_with_id"):
        layout_item = book.get_item_with_id(OUTPUT_LAYOUT_CSS_ID)
    if layout_item is None:
        layout_item = epub.EpubItem(
            uid=OUTPUT_LAYOUT_CSS_ID,
            file_name=OUTPUT_LAYOUT_CSS_NAME,
            media_type="text/css",
            content=OUTPUT_LAYOUT_CSS,
        )
        if hasattr(book, "add_item"):
            book.add_item(layout_item)
    elif hasattr(layout_item, "set_content"):
        layout_item.set_content(OUTPUT_LAYOUT_CSS)

    for item in book.get_items():
        if getattr(item, "get_type", lambda: None)() != ebooklib.ITEM_DOCUMENT:
            continue
        if hasattr(item, "set_language"):
            item.set_language(language_code)
        elif hasattr(item, "lang"):
            item.lang = language_code
        if hasattr(item, "direction"):
            item.direction = "ltr"
        if layout_item is not None and hasattr(item, "add_item"):
            item.add_item(layout_item)


def _create_document_record(plan: Any, batch_count: int) -> Dict[str, Any]:
    return {
        "file_name": plan.file_name,
        "item_id": plan.item_id,
        "source_hash": plan.source_hash,
        "segment_count": len(plan.segments),
        "batch_count": batch_count,
        "summary_status": "pending",
        "summary_patch": empty_summary_patch(),
        "translation_context_snapshot": {},
        "translation_status": "pending",
        "translated_batches": {},
        "translated_html": "",
        "reviews": [],
    }


def _create_reference_document_record(plan: Any) -> Dict[str, Any]:
    return {
        "file_name": plan.file_name,
        "item_id": plan.item_id,
        "source_hash": plan.source_hash,
        "segment_count": len(plan.segments),
        "status": "pending",
        "patch": empty_reference_patch(),
    }


def _ensure_reference_document_record(progress: Dict[str, Any], plan: Any) -> Dict[str, Any]:
    record = get_reference_document_record(progress, plan.file_name)
    if (
        record is None
        or record.get("source_hash") != plan.source_hash
        or int(record.get("segment_count", -1) or -1) != len(plan.segments)
    ):
        record = _create_reference_document_record(plan)
        upsert_reference_document_record(progress, record)
        return record

    record["file_name"] = plan.file_name
    record["item_id"] = plan.item_id
    record["source_hash"] = plan.source_hash
    record["segment_count"] = len(plan.segments)
    upsert_reference_document_record(progress, record)
    return record


def _ensure_document_record(progress: Dict[str, Any], plan: Any, batch_count: int) -> Dict[str, Any]:
    record = get_document_record(progress, plan.file_name)
    if (
        record is None
        or record.get("source_hash") != plan.source_hash
        or int(record.get("segment_count", -1) or -1) != len(plan.segments)
        or int(record.get("batch_count", -1) or -1) != batch_count
    ):
        record = _create_document_record(plan, batch_count)
        upsert_document_record(progress, record)
        return record

    record["file_name"] = plan.file_name
    record["item_id"] = plan.item_id
    record["source_hash"] = plan.source_hash
    record["segment_count"] = len(plan.segments)
    record["batch_count"] = batch_count
    if not plan.segments and not record.get("translated_html"):
        record["translated_html"] = plan.raw_html
    upsert_document_record(progress, record)
    return record


def prepare_documents(book: Any, progress: Dict[str, Any], config: PipelineConfig) -> List[PreparedDocument]:
    prepared_documents: List[PreparedDocument] = []
    for index, item in enumerate(iter_spine_documents(book), start=1):
        plan = prepare_document(item)
        batches = (
            batch_segments(
                plan.segments,
                config.max_batch_chars,
                max_batch_segments=config.max_batch_segments,
            )
            if plan.segments
            else []
        )
        record = _ensure_document_record(progress, plan, len(batches))
        prepared_documents.append(
            PreparedDocument(
                index=index,
                item=item,
                plan=plan,
                batches=batches,
                record=record,
            )
        )
    return prepared_documents


def prepare_reference_documents(book: Any, progress: Dict[str, Any]) -> List[PreparedDocument]:
    prepared_documents: List[PreparedDocument] = []
    for index, item in enumerate(iter_spine_documents(book), start=1):
        plan = prepare_document(item)
        record = _ensure_reference_document_record(progress, plan)
        prepared_documents.append(
            PreparedDocument(
                index=index,
                item=item,
                plan=plan,
                batches=[],
                record=record,
            )
        )
    return prepared_documents


def _mark_summary_done(
    record: Dict[str, Any],
    plan: Any,
    patch: Dict[str, Any],
    translation_context_snapshot: Dict[str, Any],
) -> None:
    record["item_id"] = plan.item_id
    record["source_hash"] = plan.source_hash
    record["segment_count"] = len(plan.segments)
    record["summary_status"] = "done"
    record["summary_patch"] = patch
    record["translation_context_snapshot"] = translation_context_snapshot
    if not plan.segments:
        record["translation_status"] = "done"
        record["translated_html"] = plan.raw_html


def _mark_reference_done(
    record: Dict[str, Any],
    plan: Any,
    patch: Dict[str, Any],
) -> None:
    record["item_id"] = plan.item_id
    record["source_hash"] = plan.source_hash
    record["segment_count"] = len(plan.segments)
    record["status"] = "done"
    record["patch"] = patch


def _reference_patch_from_profile(reference_profile: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "series_notes": copy.deepcopy(reference_profile.get("series_notes", [])),
        "style_notes": copy.deepcopy(reference_profile.get("style_notes", [])),
        "characters": copy.deepcopy(reference_profile.get("characters", [])),
        "terms": copy.deepcopy(reference_profile.get("terms", [])),
    }


def _document_summary_patch_from_state(document_state: Dict[str, Any]) -> Dict[str, Any]:
    chapter_summaries = [str(item).strip() for item in document_state.get("recent_summaries", []) if str(item).strip()]
    chapter_summary = " ".join(chapter_summaries[-3:]).strip()
    if len(chapter_summary) > 480:
        chapter_summary = chapter_summary[:477].rstrip() + "..."
    return {
        "chapter_summary": chapter_summary,
        "characters": copy.deepcopy(document_state.get("characters", [])),
        "time_context": copy.deepcopy(document_state.get("time_context", [])),
        "locations": copy.deepcopy(document_state.get("locations", [])),
        "events": copy.deepcopy(document_state.get("events", [])),
        "concepts": copy.deepcopy(document_state.get("concepts", [])),
        "glossary": copy.deepcopy(document_state.get("glossary", [])),
        "style_notes": copy.deepcopy(document_state.get("style_notes", [])),
        "open_questions": copy.deepcopy(document_state.get("open_questions", [])),
    }


def _extract_reference_document_patch(
    *,
    config: PipelineConfig,
    reference_client: Any,
    book_metadata: Dict[str, str],
    base_reference_profile: Dict[str, Any],
    segments: List[Dict[str, str]],
    log: Callable[[str], None],
) -> Dict[str, Any]:
    if not segments:
        return empty_reference_patch()

    reference_batches = batch_segments(
        segments,
        config.max_batch_chars,
        max_batch_segments=config.max_batch_segments,
    )
    if len(reference_batches) == 1:
        prompt_profile = reference_profile_for_prompt(base_reference_profile)
        return reference_client.extract_reference_patch(
            book_metadata=book_metadata,
            reference_profile=prompt_profile,
            segments=segments,
            target_language=config.target_language,
        )

    working_profile = copy.deepcopy(base_reference_profile)
    document_profile = new_reference_profile(book_metadata, config.target_language)

    for batch_index, batch in enumerate(reference_batches, start=1):
        log(f"  - reference chunk {batch_index}/{len(reference_batches)}")
        prompt_profile = reference_profile_for_prompt(working_profile)
        chunk_patch = reference_client.extract_reference_patch(
            book_metadata=book_metadata,
            reference_profile=prompt_profile,
            segments=batch,
            target_language=config.target_language,
        )
        working_profile = merge_reference_profile(working_profile, chunk_patch)
        document_profile = merge_reference_profile(document_profile, chunk_patch)

    return _reference_patch_from_profile(document_profile)


def _extract_reference_document_worker(
    *,
    config: PipelineConfig,
    prepared: PreparedDocument,
    book_metadata: Dict[str, str],
    base_reference_profile: Dict[str, Any],
    thread_local: threading.local,
    log: Callable[[str], None],
    emit: Callable[[str], None],
    total_document_count: int,
) -> Dict[str, Any]:
    plan = prepared.plan
    emit(
        "reference_document_started",
        index=prepared.index,
        file_name=plan.file_name,
        segment_count=len(plan.segments),
        total_document_count=total_document_count,
    )
    if not plan.segments:
        log(f"[reference-pass] {prepared.index}. {plan.file_name} (no text)")
        patch = empty_reference_patch()
    else:
        log(f"[reference] {prepared.index}. {plan.file_name} ({len(plan.segments)} segments)")
        reference_client = getattr(thread_local, "reference_client", None)
        if reference_client is None:
            reference_client = _build_llm_client(config)
            thread_local.reference_client = reference_client
        patch = _extract_reference_document_patch(
            config=config,
            reference_client=reference_client,
            book_metadata=book_metadata,
            base_reference_profile=base_reference_profile,
            segments=plan.segments,
            log=log,
        )

    return {
        "index": prepared.index,
        "file_name": plan.file_name,
        "patch": patch,
        "segment_count": len(plan.segments),
    }


def _split_segments_balanced(segments: List[Dict[str, str]]) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    if len(segments) <= 1:
        return list(segments), []

    total_chars = sum(len(segment.get("text", "")) for segment in segments)
    target_chars = max(1, total_chars // 2)
    running_chars = 0
    split_index = 1
    for index, segment in enumerate(segments[:-1], start=1):
        running_chars += len(segment.get("text", ""))
        split_index = index
        if running_chars >= target_chars:
            break
    return segments[:split_index], segments[split_index:]


def _merge_summary_patches(
    *,
    book_metadata: Dict[str, str],
    patches: List[Dict[str, Any]],
    recent_summary_limit: int,
) -> Dict[str, Any]:
    combined_state = new_story_state(book_metadata)
    for patch in patches:
        combined_state = merge_story_state(combined_state, patch, recent_summary_limit)
    return _document_summary_patch_from_state(combined_state)


def _summarize_segments_resilient(
    *,
    config: PipelineConfig,
    summary_client: Any,
    book_metadata: Dict[str, str],
    base_story_state: Dict[str, Any],
    reference_profile: Optional[Dict[str, Any]],
    segments: List[Dict[str, str]],
    log: Callable[[str], None],
    depth: int = 0,
) -> Dict[str, Any]:
    prompt_state = story_state_for_prompt(base_story_state, config.recent_summary_limit)
    try:
        return summary_client.summarize(
            book_metadata=book_metadata,
            story_state=prompt_state,
            segments=segments,
            source_language=config.source_language,
            target_language=config.target_language,
            reference_profile=reference_profile,
        )
    except Exception as exc:
        if not _is_structured_output_failure(exc) or len(segments) <= 1:
            raise

        left_segments, right_segments = _split_segments_balanced(segments)
        if not right_segments:
            raise

        indent = "  " * max(1, depth + 1)
        log(
            f"{indent}- summary fallback split {len(segments)} -> "
            f"{len(left_segments)} + {len(right_segments)}: {_exception_summary(exc)}"
        )

        left_patch = _summarize_segments_resilient(
            config=config,
            summary_client=summary_client,
            book_metadata=book_metadata,
            base_story_state=base_story_state,
            reference_profile=reference_profile,
            segments=left_segments,
            log=log,
            depth=depth + 1,
        )
        mid_state = merge_story_state(base_story_state, left_patch, config.recent_summary_limit)
        right_patch = _summarize_segments_resilient(
            config=config,
            summary_client=summary_client,
            book_metadata=book_metadata,
            base_story_state=mid_state,
            reference_profile=reference_profile,
            segments=right_segments,
            log=log,
            depth=depth + 1,
        )
        return _merge_summary_patches(
            book_metadata=book_metadata,
            patches=[left_patch, right_patch],
            recent_summary_limit=config.recent_summary_limit,
        )


def _summarize_document_patch(
    *,
    config: PipelineConfig,
    summary_client: Any,
    book_metadata: Dict[str, str],
    base_story_state: Dict[str, Any],
    reference_profile: Optional[Dict[str, Any]],
    segments: List[Dict[str, str]],
    log: Callable[[str], None],
) -> Dict[str, Any]:
    if not segments:
        return empty_summary_patch()

    summary_batches = batch_segments(
        segments,
        config.max_batch_chars,
        max_batch_segments=config.max_batch_segments,
    )
    if len(summary_batches) == 1:
        return _summarize_segments_resilient(
            config=config,
            summary_client=summary_client,
            book_metadata=book_metadata,
            base_story_state=base_story_state,
            reference_profile=reference_profile,
            segments=segments,
            log=log,
        )

    working_state = copy.deepcopy(base_story_state)
    document_state = new_story_state(book_metadata)

    for batch_index, batch in enumerate(summary_batches, start=1):
        log(f"  - summary chunk {batch_index}/{len(summary_batches)}")
        chunk_patch = _summarize_segments_resilient(
            config=config,
            summary_client=summary_client,
            book_metadata=book_metadata,
            base_story_state=working_state,
            reference_profile=reference_profile,
            segments=batch,
            log=log,
        )
        working_state = merge_story_state(working_state, chunk_patch, config.recent_summary_limit)
        document_state = merge_story_state(document_state, chunk_patch, config.recent_summary_limit)

    return _document_summary_patch_from_state(document_state)


def run_reference_phase(
    config: PipelineConfig,
    prepared_documents: List[PreparedDocument],
    progress: Dict[str, Any],
    reference_book_metadata: Dict[str, str],
    log: Callable[[str], None],
    emit: Callable[[str], None],
    reference_fingerprint: str,
) -> Dict[str, Any]:
    total_document_count = len(prepared_documents)
    reference_profile = new_reference_profile(reference_book_metadata, config.target_language)
    reference_completed_count = 0
    reference_workers = max(1, config.reference_workers or config.translation_workers)
    thread_local = threading.local()

    progress["reference_phase"]["status"] = "running"
    progress["reference_phase"]["completed_count"] = 0
    progress["reference_phase"]["total_document_count"] = total_document_count
    progress["reference_phase"]["reference_profile"] = copy.deepcopy(reference_profile)
    save_progress(config.progress_path, progress)
    emit(
        "reference_phase_started",
        total_document_count=total_document_count,
        reference_book=reference_book_metadata,
        reference_fingerprint=reference_fingerprint,
    )

    pending_documents: Dict[str, PreparedDocument] = {}
    futures: Dict[Future[Any], PreparedDocument] = {}
    executor = ThreadPoolExecutor(max_workers=reference_workers)
    executor_shutdown = False
    base_reference_profile = copy.deepcopy(reference_profile)
    result_buffer: Dict[int, Dict[str, Any]] = {}
    next_result_index = 1

    def persist_reference_result(result: Dict[str, Any], *, reused: bool) -> None:
        nonlocal reference_profile, reference_completed_count
        prepared = pending_documents.get(result["file_name"])
        if prepared is None:
            return

        record = prepared.record
        patch = dict(result.get("patch") or empty_reference_patch())
        reference_profile = merge_reference_profile(reference_profile, patch)
        if not reused:
            _mark_reference_done(record, prepared.plan, patch)
        upsert_reference_document_record(progress, record)

        reference_completed_count += 1
        progress["reference_phase"]["completed_count"] = reference_completed_count
        progress["reference_phase"]["reference_profile"] = copy.deepcopy(reference_profile)
        save_progress(config.progress_path, progress)
        emit(
            "reference_document_done",
            index=prepared.index,
            file_name=prepared.plan.file_name,
            segment_count=len(prepared.plan.segments),
            reference_completed_count=reference_completed_count,
            total_document_count=total_document_count,
            reused=reused,
        )

    def drain_reference_results() -> None:
        nonlocal next_result_index
        while next_result_index in result_buffer:
            result = result_buffer.pop(next_result_index)
            persist_reference_result(result, reused=bool(result.get("reused")))
            pending_documents.pop(result["file_name"], None)
            next_result_index += 1

    try:
        for prepared in prepared_documents:
            plan = prepared.plan
            record = prepared.record
            can_reuse_patch = (
                record.get("status") == "done"
                and record.get("source_hash") == plan.source_hash
                and record.get("segment_count") == len(plan.segments)
                and isinstance(record.get("patch"), dict)
            )
            pending_documents[plan.file_name] = prepared

            if can_reuse_patch:
                log(f"[reference-skip] {prepared.index}. {plan.file_name}")
                result_buffer[prepared.index] = {
                    "index": prepared.index,
                    "file_name": plan.file_name,
                    "patch": record.get("patch"),
                    "segment_count": len(plan.segments),
                    "reused": True,
                }
                drain_reference_results()
                continue

            future = executor.submit(
                _extract_reference_document_worker,
                config=config,
                prepared=prepared,
                book_metadata=reference_book_metadata,
                base_reference_profile=base_reference_profile,
                thread_local=thread_local,
                log=log,
                emit=emit,
                total_document_count=total_document_count,
            )
            futures[future] = prepared

        first_error: Optional[BaseException] = None
        while futures:
            done, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                prepared = futures.pop(future)
                try:
                    result_buffer[prepared.index] = future.result()
                except CancelledError:
                    continue
                except BaseException as exc:  # noqa: BLE001
                    if first_error is None:
                        first_error = exc

            drain_reference_results()

            if first_error is not None:
                for pending_future in list(futures.keys()):
                    pending_future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)
                executor_shutdown = True
                raise first_error
    finally:
        if not executor_shutdown:
            executor.shutdown(wait=True, cancel_futures=False)

    progress["reference_phase"]["status"] = "done"
    progress["reference_phase"]["completed_count"] = reference_completed_count
    progress["reference_phase"]["total_document_count"] = total_document_count
    progress["reference_phase"]["reference_profile"] = copy.deepcopy(reference_profile)
    save_progress(config.progress_path, progress)
    emit(
        "reference_phase_finished",
        reference_completed_count=reference_completed_count,
        total_document_count=total_document_count,
        reference_book=reference_book_metadata,
        reference_fingerprint=reference_fingerprint,
    )
    return reference_profile

def run_summary_phase(
    config: PipelineConfig,
    prepared_documents: List[PreparedDocument],
    progress: Dict[str, Any],
    book_metadata: Dict[str, str],
    reference_profile: Optional[Dict[str, Any]],
    log: Callable[[str], None],
    emit: Callable[[str], None],
) -> Dict[str, Any]:
    total_document_count = len(prepared_documents)
    summary_state = new_story_state(book_metadata)
    summary_client = _build_llm_client(config)
    summary_completed_count = 0

    progress["summary_phase"]["status"] = "running"
    progress["summary_phase"]["completed_count"] = 0
    progress["summary_phase"]["story_state"] = copy.deepcopy(summary_state)
    progress["story_state"] = copy.deepcopy(summary_state)
    save_progress(config.progress_path, progress)
    emit(
        "summary_phase_started",
        total_document_count=total_document_count,
    )

    for prepared in prepared_documents:
        plan = prepared.plan
        record = prepared.record
        can_reuse_summary = (
            record.get("summary_status") == "done"
            and record.get("source_hash") == plan.source_hash
            and record.get("segment_count") == len(plan.segments)
            and isinstance(record.get("summary_patch"), dict)
            and bool(record.get("translation_context_snapshot"))
        )

        if can_reuse_summary:
            log(f"[summary-skip] {prepared.index}. {plan.file_name}")
            summary_state = merge_story_state(summary_state, record.get("summary_patch"), config.recent_summary_limit)
            record["translation_context_snapshot"] = story_state_for_prompt(summary_state, config.recent_summary_limit)
            if not plan.segments and not record.get("translated_html"):
                record["translated_html"] = plan.raw_html
                record["translation_status"] = "done"
            upsert_document_record(progress, record)
            summary_completed_count += 1
            progress["summary_phase"]["completed_count"] = summary_completed_count
            progress["summary_phase"]["story_state"] = copy.deepcopy(summary_state)
            progress["story_state"] = copy.deepcopy(summary_state)
            save_progress(config.progress_path, progress)
            emit(
                "summary_document_done",
                index=prepared.index,
                file_name=plan.file_name,
                segment_count=len(plan.segments),
                summary_completed_count=summary_completed_count,
                total_document_count=total_document_count,
                reused=True,
            )
            continue

        if not plan.segments:
            log(f"[summary-pass] {prepared.index}. {plan.file_name} (no text)")
            patch = empty_summary_patch()
        else:
            log(f"[summary] {prepared.index}. {plan.file_name} ({len(plan.segments)} segments)")
            emit(
                "summary_document_started",
                index=prepared.index,
                file_name=plan.file_name,
                segment_count=len(plan.segments),
                total_document_count=total_document_count,
            )
            patch = _summarize_document_patch(
                config=config,
                summary_client=summary_client,
                book_metadata=book_metadata,
                segments=plan.segments,
                base_story_state=summary_state,
                reference_profile=reference_profile,
                log=log,
            )

        summary_state = merge_story_state(summary_state, patch, config.recent_summary_limit)
        translation_context_snapshot = story_state_for_prompt(summary_state, config.recent_summary_limit)
        _mark_summary_done(record, plan, patch, translation_context_snapshot)
        upsert_document_record(progress, record)

        summary_completed_count += 1
        progress["summary_phase"]["completed_count"] = summary_completed_count
        progress["summary_phase"]["story_state"] = copy.deepcopy(summary_state)
        progress["story_state"] = copy.deepcopy(summary_state)
        save_progress(config.progress_path, progress)
        emit(
            "summary_document_done",
            index=prepared.index,
            file_name=plan.file_name,
            segment_count=len(plan.segments),
            summary_completed_count=summary_completed_count,
            total_document_count=total_document_count,
            reused=False,
        )

    progress["summary_phase"]["status"] = "done"
    progress["summary_phase"]["completed_count"] = summary_completed_count
    progress["summary_phase"]["story_state"] = copy.deepcopy(summary_state)
    progress["story_state"] = copy.deepcopy(summary_state)
    save_progress(config.progress_path, progress)
    emit(
        "summary_phase_finished",
        summary_completed_count=summary_completed_count,
        total_document_count=total_document_count,
    )
    return summary_state


def _sorted_translated_batch_entries(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    entries = list((record.get("translated_batches") or {}).values())
    valid_entries = [entry for entry in entries if isinstance(entry, dict) and int(entry.get("batch_index", 0) or 0) > 0]
    return sorted(valid_entries, key=lambda entry: int(entry.get("batch_index", 0) or 0))


def _completed_batch_indices(record: Dict[str, Any], total_batches: int) -> set[int]:
    if record.get("translation_status") == "done" and record.get("translated_html"):
        return set(range(1, total_batches + 1))

    completed = set()
    for entry in _sorted_translated_batch_entries(record):
        batch_index = int(entry.get("batch_index", 0) or 0)
        if 1 <= batch_index <= total_batches and isinstance(entry.get("translations"), dict) and entry.get("translations"):
            completed.add(batch_index)
    return completed


def _has_complete_batch_results(record: Dict[str, Any], total_batches: int) -> bool:
    return len(_completed_batch_indices(record, total_batches)) == total_batches


def _collect_translated_map(record: Dict[str, Any]) -> Dict[str, str]:
    translated_map: Dict[str, str] = {}
    for entry in _sorted_translated_batch_entries(record):
        for segment_id, translation in dict(entry.get("translations") or {}).items():
            translated_map[str(segment_id)] = str(translation)
    return translated_map


def _collect_review_payloads(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    reviews: List[Dict[str, Any]] = []
    for entry in _sorted_translated_batch_entries(record):
        review = entry.get("review")
        if isinstance(review, dict):
            reviews.append(review)
    return reviews


def _store_batch_result(
    record: Dict[str, Any],
    batch_index: int,
    translations: Dict[str, str],
    review_payload: Dict[str, Any],
) -> None:
    record.setdefault("translated_batches", {})[make_batch_key(batch_index)] = {
        "batch_index": batch_index,
        "translations": dict(translations),
        "review": dict(review_payload),
    }
    record["reviews"] = _collect_review_payloads(record)
    record["translation_status"] = "partial"


def _ensure_document_translation_complete(plan: Any, translated_map: Dict[str, str]) -> None:
    missing = [segment["id"] for segment in plan.segments if not translated_map.get(segment["id"])]
    if missing:
        raise RuntimeError(f"{plan.file_name} 仍有未补齐的片段译文: {', '.join(missing)}")


def _finalize_translated_document(
    prepared: PreparedDocument,
    progress: Dict[str, Any],
) -> None:
    record = prepared.record
    translated_map = _collect_translated_map(record)
    _ensure_document_translation_complete(prepared.plan, translated_map)
    translated_html = apply_translations(prepared.plan, translated_map)
    set_item_content(prepared.item, translated_html)
    record["translation_status"] = "done"
    record["translated_html"] = translated_html
    record["reviews"] = _collect_review_payloads(record)
    upsert_document_record(progress, record)


def _apply_stored_document_translation(prepared: PreparedDocument, progress: Dict[str, Any]) -> None:
    record = prepared.record
    if not prepared.plan.segments:
        set_item_content(prepared.item, prepared.plan.raw_html)
        return

    translated_html = str(record.get("translated_html") or "")
    if translated_html:
        set_item_content(prepared.item, translated_html)
        return

    _finalize_translated_document(prepared, progress)


def _get_thread_local_llm_client(config: PipelineConfig, thread_local: threading.local):
    llm_client = getattr(thread_local, "llm_client", None)
    if llm_client is None:
        llm_client = _build_llm_client(config)
        thread_local.llm_client = llm_client
    return llm_client


def _translate_batch_once_worker(
    *,
    config: PipelineConfig,
    book_metadata: Dict[str, str],
    prepared: PreparedDocument,
    batch_index: int,
    batch: List[Dict[str, str]],
    prompt_state: Dict[str, Any],
    prompt_reference_profile: Optional[Dict[str, Any]],
    retry_feedback: Optional[str],
    attempt: int,
    retry_callback: Callable[[PreparedDocument, int, int, str], None],
    batch_started_callback: Callable[[PreparedDocument, int], None],
    batch_finished_callback: Callable[[], None],
    thread_local: threading.local,
) -> Dict[str, Any]:
    batch_started_callback(prepared, batch_index)
    try:
        llm_client = _get_thread_local_llm_client(config, thread_local)
        batch_translation = llm_client.translate(
            book_metadata=book_metadata,
            story_state=prompt_state,
            segments=batch,
            source_language=config.source_language,
            target_language=config.target_language,
            retry_feedback=retry_feedback,
            reference_profile=prompt_reference_profile,
        )
        return {
            "file_name": prepared.plan.file_name,
            "batch_index": batch_index,
            "batch": batch,
            "prompt_state": prompt_state,
            "prompt_reference_profile": prompt_reference_profile,
            "translations": batch_translation,
            "attempt": attempt,
            "retry_feedback": retry_feedback,
        }
    finally:
        batch_finished_callback()


def _review_batch_worker(
    *,
    config: PipelineConfig,
    book_metadata: Dict[str, str],
    prepared: PreparedDocument,
    batch_index: int,
    batch: List[Dict[str, str]],
    translations: Dict[str, str],
    prompt_state: Dict[str, Any],
    prompt_reference_profile: Optional[Dict[str, Any]],
    attempt: int,
    thread_local: threading.local,
) -> Dict[str, Any]:
    llm_client = _get_thread_local_llm_client(config, thread_local)
    translated_segments = _translated_segments_as_list(batch, translations)
    review_payload = llm_client.review(
        book_metadata=book_metadata,
        story_state=prompt_state,
        source_segments=batch,
        translated_segments=translated_segments,
        source_language=config.source_language,
        target_language=config.target_language,
        reference_profile=prompt_reference_profile,
    )
    corrected_translations = _apply_review_corrections(translations, review_payload)
    return {
        "file_name": prepared.plan.file_name,
        "batch_index": batch_index,
        "batch": batch,
        "prompt_state": prompt_state,
        "prompt_reference_profile": prompt_reference_profile,
        "translations": corrected_translations,
        "review": review_payload,
        "attempt": attempt,
    }


def run_parallel_translation_phase(
    config: PipelineConfig,
    prepared_documents: List[PreparedDocument],
    progress: Dict[str, Any],
    book_metadata: Dict[str, str],
    reference_profile: Optional[Dict[str, Any]],
    log: Callable[[str], None],
    emit: Callable[[str], None],
) -> Dict[str, int]:
    total_document_count = len(prepared_documents)
    total_batch_count = sum(len(prepared.batches) for prepared in prepared_documents)
    completed_batch_count = 0
    processed_count = 0
    skipped_count = 0
    completed_count = 0
    retried_batches = 0

    active_workers_lock = threading.Lock()
    counters_lock = threading.Lock()
    active_workers = 0
    translation_thread_local = threading.local()
    review_thread_local = threading.local()
    review_workers = max(1, config.review_workers or config.translation_workers)

    def current_active_workers() -> int:
        with active_workers_lock:
            return active_workers

    def batch_started_callback(prepared: PreparedDocument, batch_index: int) -> None:
        nonlocal active_workers
        with active_workers_lock:
            active_workers += 1
            worker_count = active_workers
        log(f"[translate] {prepared.index}. {prepared.plan.file_name} batch {batch_index}/{len(prepared.batches)}")
        emit(
            "translation_batch_started",
            index=prepared.index,
            file_name=prepared.plan.file_name,
            batch_index=batch_index,
            total_batches=len(prepared.batches),
            batch_segment_count=len(prepared.batches[batch_index - 1]),
            active_workers=worker_count,
            translation_workers=config.translation_workers,
            review_workers=review_workers,
        )

    def batch_finished_callback() -> None:
        nonlocal active_workers
        with active_workers_lock:
            active_workers = max(0, active_workers - 1)

    def retry_callback(prepared: PreparedDocument, batch_index: int, attempt: int, retry_feedback: str) -> None:
        nonlocal retried_batches
        with counters_lock:
            retried_batches += 1
            retry_count = retried_batches
        log(f"    retry {attempt}: {retry_feedback}")
        emit(
            "translation_batch_retry",
            index=prepared.index,
            file_name=prepared.plan.file_name,
            batch_index=batch_index,
            total_batches=len(prepared.batches),
            attempt=attempt,
            retried_batches=retry_count,
            retry_feedback=retry_feedback,
            active_workers=current_active_workers(),
            translation_workers=config.translation_workers,
            review_workers=review_workers,
        )

    progress["translation_phase"]["status"] = "running"
    progress["translation_phase"]["total_batch_count"] = total_batch_count
    prompt_reference_profile = reference_profile_for_prompt(reference_profile) if reference_profile else None

    pending_documents: Dict[str, PreparedDocument] = {}
    translation_futures: Dict[Future[Any], PreparedDocument] = {}
    review_futures: Dict[Future[Any], PreparedDocument] = {}

    for prepared in prepared_documents:
        record = prepared.record
        batch_count = len(prepared.batches)
        completed_indices = _completed_batch_indices(record, batch_count)
        completed_batch_count += len(completed_indices)

    progress["translation_phase"]["completed_batch_count"] = completed_batch_count
    save_progress(config.progress_path, progress)
    emit(
        "translation_phase_started",
        total_document_count=total_document_count,
        total_batch_count=total_batch_count,
        completed_batch_count=completed_batch_count,
        translation_workers=config.translation_workers,
        review_workers=review_workers,
    )
    log(f"[translate] total batches: {total_batch_count}, workers: {config.translation_workers}, review workers: {review_workers}")

    def finalize_document(prepared: PreparedDocument, *, reused: bool) -> None:
        nonlocal processed_count, skipped_count, completed_count
        record = prepared.record
        if reused:
            _apply_stored_document_translation(prepared, progress)
            skipped_count += 1
            completed_count += 1
            log(f"[skip] {prepared.index}. {prepared.plan.file_name}")
            emit(
                "translation_document_reused",
                index=prepared.index,
                file_name=prepared.plan.file_name,
                completed_count=completed_count,
                skipped_count=skipped_count,
                processed_count=processed_count,
                total_document_count=total_document_count,
            )
            return

        _finalize_translated_document(prepared, progress)
        save_progress(config.progress_path, progress)
        processed_count += 1
        completed_count += 1
        log(f"[done] {prepared.index}. {prepared.plan.file_name}")
        emit(
            "translation_document_done",
            index=prepared.index,
            file_name=prepared.plan.file_name,
            completed_count=completed_count,
            processed_count=processed_count,
            skipped_count=skipped_count,
            total_document_count=total_document_count,
        )

    translation_executor = ThreadPoolExecutor(max_workers=max(1, config.translation_workers))
    review_executor = ThreadPoolExecutor(max_workers=review_workers)
    executors_shutdown = False

    def persist_batch_result(result: Dict[str, Any]) -> None:
        nonlocal completed_batch_count
        prepared = pending_documents[result["file_name"]]
        record = prepared.record
        _store_batch_result(
            record=record,
            batch_index=int(result["batch_index"]),
            translations=dict(result["translations"]),
            review_payload=dict(result["review"] or {}),
        )
        upsert_document_record(progress, record)
        completed_batch_count += 1
        progress["translation_phase"]["completed_batch_count"] = completed_batch_count
        save_progress(config.progress_path, progress)
        emit(
            "translation_batch_done",
            index=prepared.index,
            file_name=prepared.plan.file_name,
            batch_index=int(result["batch_index"]),
            total_batches=len(prepared.batches),
            completed_batch_count=completed_batch_count,
            total_batch_count=total_batch_count,
            active_workers=current_active_workers(),
            translation_workers=config.translation_workers,
            review_workers=review_workers,
        )
        if len(_completed_batch_indices(record, len(prepared.batches))) == len(prepared.batches):
            finalize_document(prepared, reused=False)
            pending_documents.pop(prepared.plan.file_name, None)

    def should_retry_review(result: Dict[str, Any]) -> bool:
        review_payload = dict(result.get("review") or {})
        needs_retry = bool(review_payload.get("needs_retry"))
        score = int(review_payload.get("score", 0) or 0)
        has_material_feedback = bool(
            review_payload.get("major_issues") or review_payload.get("retry_feedback")
        )
        return needs_retry or (score < config.min_review_score and has_material_feedback)

    def review_retry_feedback(result: Dict[str, Any]) -> str:
        review_payload = dict(result.get("review") or {})
        return str(
            review_payload.get("retry_feedback")
            or "; ".join((review_payload.get("major_issues") or [])[:3])
            or "score too low"
        )

    def submit_translation(
        *,
        prepared: PreparedDocument,
        batch_index: int,
        batch: List[Dict[str, str]],
        prompt_state: Dict[str, Any],
        retry_feedback: Optional[str] = None,
        attempt: int = 0,
    ) -> None:
        future = translation_executor.submit(
            _translate_batch_once_worker,
            config=config,
            book_metadata=book_metadata,
            prepared=prepared,
            batch_index=batch_index,
            batch=batch,
            prompt_state=prompt_state,
            prompt_reference_profile=prompt_reference_profile,
            retry_feedback=retry_feedback,
            attempt=attempt,
            retry_callback=retry_callback,
            batch_started_callback=batch_started_callback,
            batch_finished_callback=batch_finished_callback,
            thread_local=translation_thread_local,
        )
        translation_futures[future] = prepared

    def submit_review(translation_result: Dict[str, Any]) -> None:
        prepared = pending_documents[translation_result["file_name"]]
        future = review_executor.submit(
            _review_batch_worker,
            config=config,
            book_metadata=book_metadata,
            prepared=prepared,
            batch_index=int(translation_result["batch_index"]),
            batch=list(translation_result["batch"]),
            translations=dict(translation_result["translations"]),
            prompt_state=dict(translation_result["prompt_state"]),
            prompt_reference_profile=translation_result.get("prompt_reference_profile"),
            attempt=int(translation_result.get("attempt", 0) or 0),
            thread_local=review_thread_local,
        )
        review_futures[future] = prepared

    try:
        for prepared in prepared_documents:
            record = prepared.record
            if not prepared.plan.segments:
                set_item_content(prepared.item, prepared.plan.raw_html)
                completed_count += 1
                emit(
                    "translation_document_done",
                    index=prepared.index,
                    file_name=prepared.plan.file_name,
                    completed_count=completed_count,
                    processed_count=processed_count,
                    skipped_count=skipped_count,
                    total_document_count=total_document_count,
                )
                continue

            if record.get("translation_status") == "done" and _has_complete_batch_results(record, len(prepared.batches)):
                finalize_document(prepared, reused=True)
                continue

            completed_indices = _completed_batch_indices(record, len(prepared.batches))
            if len(completed_indices) == len(prepared.batches):
                finalize_document(prepared, reused=False)
                continue

            pending_documents[prepared.plan.file_name] = prepared
            pending_batch_count = len(prepared.batches) - len(completed_indices)
            log(
                f"[translate-doc] {prepared.index}. {prepared.plan.file_name} "
                f"({pending_batch_count}/{len(prepared.batches)} batches pending)"
            )
            emit(
                "translation_document_started",
                index=prepared.index,
                file_name=prepared.plan.file_name,
                segment_count=len(prepared.plan.segments),
                total_batches=len(prepared.batches),
                pending_batches=pending_batch_count,
                total_document_count=total_document_count,
            )

            translation_context_snapshot = dict(record.get("translation_context_snapshot") or {})
            prompt_state = story_state_for_prompt(translation_context_snapshot, config.recent_summary_limit)
            for batch_index, batch in enumerate(prepared.batches, start=1):
                if batch_index in completed_indices:
                    continue
                submit_translation(
                    prepared=prepared,
                    batch_index=batch_index,
                    batch=batch,
                    prompt_state=prompt_state,
                )

        first_error: Optional[BaseException] = None
        while translation_futures or review_futures:
            active_futures = list(translation_futures.keys()) + list(review_futures.keys())
            done, _ = wait(active_futures, return_when=FIRST_COMPLETED)
            successful_translation_results: List[Dict[str, Any]] = []
            successful_review_results: List[Dict[str, Any]] = []
            for future in done:
                future_kind = "translation" if future in translation_futures else "review"
                if future_kind == "translation":
                    translation_futures.pop(future, None)
                else:
                    review_futures.pop(future, None)
                try:
                    result = future.result()
                except CancelledError:
                    continue
                except BaseException as exc:  # noqa: BLE001
                    if first_error is None:
                        first_error = exc
                    continue
                if future_kind == "translation":
                    successful_translation_results.append(result)
                else:
                    successful_review_results.append(result)

            for result in successful_translation_results:
                submit_review(result)

            for result in successful_review_results:
                if should_retry_review(result) and int(result.get("attempt", 0) or 0) < config.max_review_retries:
                    next_attempt = int(result.get("attempt", 0) or 0) + 1
                    feedback = review_retry_feedback(result)
                    prepared = pending_documents[result["file_name"]]
                    retry_callback(prepared, int(result["batch_index"]), next_attempt, feedback)
                    submit_translation(
                        prepared=prepared,
                        batch_index=int(result["batch_index"]),
                        batch=list(result["batch"]),
                        prompt_state=dict(result["prompt_state"]),
                        retry_feedback=feedback,
                        attempt=next_attempt,
                    )
                    continue
                persist_batch_result(result)

            if first_error is not None:
                finished_review_futures = [
                    future for future in list(review_futures.keys()) if future.done() and not future.cancelled()
                ]
                for review_future in finished_review_futures:
                    review_futures.pop(review_future, None)
                    try:
                        persist_batch_result(review_future.result())
                    except BaseException:  # noqa: BLE001
                        pass

                for pending_future in list(translation_futures.keys()) + list(review_futures.keys()):
                    pending_future.cancel()
                translation_executor.shutdown(wait=True, cancel_futures=True)
                review_executor.shutdown(wait=True, cancel_futures=True)
                executors_shutdown = True
                translation_futures.clear()
                review_futures.clear()
                raise first_error
    finally:
        if not executors_shutdown:
            translation_executor.shutdown(wait=True, cancel_futures=False)
            review_executor.shutdown(wait=True, cancel_futures=False)

    progress["translation_phase"]["status"] = "done"
    progress["translation_phase"]["completed_batch_count"] = completed_batch_count
    progress["translation_phase"]["completed_document_count"] = completed_count
    save_progress(config.progress_path, progress)
    emit(
        "translation_phase_finished",
        total_document_count=total_document_count,
        completed_count=completed_count,
        processed_count=processed_count,
        skipped_count=skipped_count,
        total_batch_count=total_batch_count,
        completed_batch_count=completed_batch_count,
        retried_batches=retried_batches,
    )
    return {
        "processed_count": processed_count,
        "skipped_count": skipped_count,
        "completed_count": completed_count,
        "completed_batch_count": completed_batch_count,
        "total_batch_count": total_batch_count,
        "retried_batches": retried_batches,
    }


def run_translation_pipeline(
    config: PipelineConfig,
    log_func: Optional[Callable[[str], None]] = None,
    status_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    def log(message: str) -> None:
        if log_func is None:
            print(message)
            return
        log_func(message)

    def emit(event: str, **payload: Any) -> None:
        if status_callback is None:
            return
        body = {"event": event}
        body.update(payload)
        status_callback(body)

    book = epub.read_epub(str(config.input_path))
    book_metadata = extract_book_metadata(book)
    reference_context = _build_reference_context(config)
    progress = _validate_or_create_progress(config, book_metadata, reference_context)
    prepared_documents = prepare_documents(book, progress, config)
    prepared_reference_documents = (
        prepare_reference_documents(reference_context.book, progress) if reference_context.enabled else []
    )
    save_progress(config.progress_path, progress)

    emit(
        "run_started",
        total_document_count=len(prepared_documents),
        book_metadata=book_metadata,
        input_path=str(config.input_path),
        output_path=str(config.output_path),
        progress_path=str(config.progress_path),
        reference_enabled=reference_context.enabled,
        reference_input_path=str(reference_context.input_path) if reference_context.input_path else "",
        reference_book=reference_context.book_metadata,
        reference_fingerprint=reference_context.fingerprint,
    )

    reference_profile: Optional[Dict[str, Any]] = None
    if reference_context.enabled:
        reference_title = reference_context.book_metadata.get("title") or reference_context.input_path.name
        log(f"reference: {reference_title}")
        log(f"reference fingerprint: {reference_context.fingerprint}")
        reference_profile = run_reference_phase(
            config=config,
            prepared_documents=prepared_reference_documents,
            progress=progress,
            reference_book_metadata=reference_context.book_metadata,
            log=log,
            emit=emit,
            reference_fingerprint=reference_context.fingerprint,
        )
    else:
        log("reference: disabled")

    run_summary_phase(
        config=config,
        prepared_documents=prepared_documents,
        progress=progress,
        book_metadata=book_metadata,
        reference_profile=reference_profile,
        log=log,
        emit=emit,
    )

    translation_result = run_parallel_translation_phase(
        config=config,
        prepared_documents=prepared_documents,
        progress=progress,
        book_metadata=book_metadata,
        reference_profile=reference_profile,
        log=log,
        emit=emit,
    )

    if config.title_suffix:
        _set_book_title(book, f"{book_metadata['title']}{config.title_suffix}")

    _prepare_output_book_metadata(book, config.target_language)
    translated_title_lookup = _build_translated_title_lookup(book)
    _ensure_book_item_identifiers(book)
    book.toc = tuple(_rewrite_toc_titles(_normalize_toc(book.toc), translated_title_lookup))
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    epub.write_epub(str(config.output_path), book, {"spine_direction": True})

    result = {
        "input_path": str(config.input_path),
        "output_path": str(config.output_path),
        "progress_path": str(config.progress_path),
        "processed_count": translation_result["processed_count"],
        "skipped_count": translation_result["skipped_count"],
        "completed_count": translation_result["completed_count"],
        "retried_batches": translation_result["retried_batches"],
        "summary_completed_count": len(prepared_documents),
        "translation_completed_batch_count": translation_result["completed_batch_count"],
        "translation_total_batch_count": translation_result["total_batch_count"],
        "translation_workers": config.translation_workers,
        "review_workers": max(1, config.review_workers or config.translation_workers),
        "reference_workers": max(1, config.reference_workers or config.translation_workers),
        "reference_enabled": reference_context.enabled,
        "reference_completed_count": len(prepared_reference_documents) if reference_context.enabled else 0,
    }
    emit("run_finished", **result)
    return result


def run_translation_pipeline_with_retries(
    config: PipelineConfig,
    log_func: Optional[Callable[[str], None]] = None,
    status_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    def log(message: str) -> None:
        if log_func is None:
            print(message)
            return
        log_func(message)

    total_attempts = max(1, int(config.auto_resume_retries or 0) + 1)

    for attempt_index in range(total_attempts):
        attempt_number = attempt_index + 1
        attempt_config = config if attempt_index == 0 else replace(config, reset_progress=False)

        if attempt_index > 0:
            log(
                f"[auto-resume] 开始第 {attempt_number}/{total_attempts} 次运行，"
                f"复用 progress: {attempt_config.progress_path}"
            )

        try:
            result = run_translation_pipeline(
                attempt_config,
                log_func=log_func,
                status_callback=status_callback,
            )
        except Exception as exc:
            failure_message = _exception_summary(exc)
            retryable = is_retryable_run_error(exc)
            can_retry = retryable and attempt_index < total_attempts - 1
            if can_retry:
                log(f"任务失败，准备自动续跑 {attempt_number}/{total_attempts - 1}: {failure_message}")
                log(f"自动续跑将复用现有 progress: {config.progress_path}")
                continue
            if retryable and attempt_index > 0:
                log(f"任务失败，自动续跑次数已用尽: {failure_message}")
            raise

        result["run_attempt_count"] = attempt_number
        result["auto_resume_count"] = attempt_index
        if attempt_index > 0:
            log(f"自动续跑成功，本次在第 {attempt_number} 次运行完成。")
        return result

    raise RuntimeError("自动续跑包装器提前结束，未获得有效结果。")
