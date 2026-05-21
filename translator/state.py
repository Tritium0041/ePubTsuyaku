from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROGRESS_VERSION = 4
LIST_FIELDS = (
    "time_context",
    "locations",
    "events",
    "concepts",
    "style_notes",
    "open_questions",
)
REFERENCE_SERIES_NOTE_LIMIT = 12
REFERENCE_STYLE_NOTE_LIMIT = 12
REFERENCE_CHARACTER_LIMIT = 40
REFERENCE_TERM_LIMIT = 80
REFERENCE_EXAMPLE_LIMIT = 2
PROMPT_RECENT_SUMMARY_CHAR_LIMIT = 160
PROMPT_CHARACTER_LIMIT = 14
PROMPT_CHARACTER_KEEP_HEAD = 10
PROMPT_CHARACTER_ALIAS_LIMIT = 4
PROMPT_CHARACTER_NAME_CHAR_LIMIT = 32
PROMPT_CHARACTER_DESCRIPTION_CHAR_LIMIT = 64
PROMPT_GLOSSARY_LIMIT = 24
PROMPT_GLOSSARY_KEEP_HEAD = 12
PROMPT_GLOSSARY_TEXT_CHAR_LIMIT = 40
PROMPT_GLOSSARY_NOTE_CHAR_LIMIT = 60
PROMPT_LIST_LIMITS = {
    "time_context": 8,
    "locations": 8,
    "events": 16,
    "concepts": 12,
    "style_notes": 12,
    "open_questions": 6,
}
PROMPT_LIST_KEEP_HEAD = {
    "time_context": 4,
    "locations": 2,
    "events": 0,
    "concepts": 0,
    "style_notes": 6,
    "open_questions": 0,
}
PROMPT_LIST_CHAR_LIMITS = {
    "time_context": 40,
    "locations": 40,
    "events": 80,
    "concepts": 80,
    "style_notes": 100,
    "open_questions": 100,
}
PROMPT_REFERENCE_SERIES_NOTE_LIMIT = 6
PROMPT_REFERENCE_STYLE_NOTE_LIMIT = 6
PROMPT_REFERENCE_CHARACTER_LIMIT = 12
PROMPT_REFERENCE_CHARACTER_KEEP_HEAD = 8
PROMPT_REFERENCE_TERM_LIMIT = 16
PROMPT_REFERENCE_TERM_KEEP_HEAD = 8
PROMPT_REFERENCE_ALIAS_LIMIT = 3
PROMPT_REFERENCE_ROLE_CHAR_LIMIT = 36
PROMPT_REFERENCE_USAGE_NOTE_CHAR_LIMIT = 48


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _trim_text(value: Any, max_chars: int) -> str:
    text = normalize_text(value)
    if not text or max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def _windowed_items(items: List[Any], limit: int, keep_head: int = 0) -> List[Any]:
    if limit <= 0 or not items:
        return []
    if len(items) <= limit:
        return list(items)

    head_count = min(limit, max(0, keep_head))
    tail_count = max(0, limit - head_count)
    selected_indexes: List[int] = list(range(head_count))
    if tail_count > 0:
        tail_start = max(head_count, len(items) - tail_count)
        selected_indexes.extend(range(tail_start, len(items)))

    seen = set()
    result: List[Any] = []
    for index in selected_indexes:
        if index in seen:
            continue
        seen.add(index)
        result.append(items[index])
    return result[:limit]


def _compact_prompt_strings(
    items: Iterable[Any],
    *,
    limit: int,
    max_chars: int,
    keep_head: int = 0,
) -> List[str]:
    cleaned_items = [normalize_text(item) for item in items if normalize_text(item)]
    selected = _windowed_items(cleaned_items, limit=limit, keep_head=keep_head)
    return [_trim_text(item, max_chars) for item in selected if _trim_text(item, max_chars)]


def _compact_prompt_characters(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected = _windowed_items(list(items), limit=PROMPT_CHARACTER_LIMIT, keep_head=PROMPT_CHARACTER_KEEP_HEAD)
    compacted: List[Dict[str, Any]] = []
    for item in selected:
        if not isinstance(item, dict):
            continue
        name = _trim_text(item.get("name"), PROMPT_CHARACTER_NAME_CHAR_LIMIT)
        if not name:
            continue
        aliases = _compact_prompt_strings(
            item.get("aliases", []),
            limit=PROMPT_CHARACTER_ALIAS_LIMIT,
            max_chars=PROMPT_CHARACTER_NAME_CHAR_LIMIT,
        )
        compacted.append(
            {
                "name": name,
                "description": _trim_text(item.get("description"), PROMPT_CHARACTER_DESCRIPTION_CHAR_LIMIT),
                "aliases": aliases,
            }
        )
    return compacted


def _compact_prompt_glossary(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected = _windowed_items(list(items), limit=PROMPT_GLOSSARY_LIMIT, keep_head=PROMPT_GLOSSARY_KEEP_HEAD)
    compacted: List[Dict[str, Any]] = []
    for item in selected:
        if not isinstance(item, dict):
            continue
        source = _trim_text(item.get("source"), PROMPT_GLOSSARY_TEXT_CHAR_LIMIT)
        target = _trim_text(item.get("target"), PROMPT_GLOSSARY_TEXT_CHAR_LIMIT)
        if not source or not target:
            continue
        compacted.append(
            {
                "source": source,
                "target": target,
                "note": _trim_text(item.get("note"), PROMPT_GLOSSARY_NOTE_CHAR_LIMIT),
            }
        )
    return compacted


def _compact_reference_characters(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected = _windowed_items(
        list(items),
        limit=PROMPT_REFERENCE_CHARACTER_LIMIT,
        keep_head=PROMPT_REFERENCE_CHARACTER_KEEP_HEAD,
    )
    compacted: List[Dict[str, Any]] = []
    for item in selected:
        if not isinstance(item, dict):
            continue
        name = _trim_text(item.get("name"), PROMPT_CHARACTER_NAME_CHAR_LIMIT)
        if not name:
            continue
        compacted.append(
            {
                "name": name,
                "aliases": _compact_prompt_strings(
                    item.get("aliases", []),
                    limit=PROMPT_REFERENCE_ALIAS_LIMIT,
                    max_chars=PROMPT_CHARACTER_NAME_CHAR_LIMIT,
                ),
                "role": _trim_text(item.get("role"), PROMPT_REFERENCE_ROLE_CHAR_LIMIT),
                "usage_note": _trim_text(item.get("usage_note"), PROMPT_REFERENCE_USAGE_NOTE_CHAR_LIMIT),
            }
        )
    return compacted


def _compact_reference_terms(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected = _windowed_items(
        list(items),
        limit=PROMPT_REFERENCE_TERM_LIMIT,
        keep_head=PROMPT_REFERENCE_TERM_KEEP_HEAD,
    )
    compacted: List[Dict[str, Any]] = []
    for item in selected:
        if not isinstance(item, dict):
            continue
        term = _trim_text(item.get("term"), PROMPT_GLOSSARY_TEXT_CHAR_LIMIT)
        if not term:
            continue
        compacted.append(
            {
                "term": term,
                "category": _trim_text(item.get("category"), PROMPT_GLOSSARY_TEXT_CHAR_LIMIT),
                "usage_note": _trim_text(item.get("usage_note"), PROMPT_REFERENCE_USAGE_NOTE_CHAR_LIMIT),
            }
        )
    return compacted


def new_story_state(book_metadata: Dict[str, str]) -> Dict[str, Any]:
    return {
        "book_title": book_metadata.get("title", ""),
        "author": book_metadata.get("author", ""),
        "characters": [],
        "time_context": [],
        "locations": [],
        "events": [],
        "concepts": [],
        "glossary": [],
        "style_notes": [],
        "open_questions": [],
        "recent_summaries": [],
    }


def new_reference_profile(
    book_metadata: Optional[Dict[str, str]],
    target_language: str,
) -> Dict[str, Any]:
    metadata = dict(book_metadata or {})
    return {
        "book_title": metadata.get("title", ""),
        "target_language": target_language,
        "series_notes": [],
        "style_notes": [],
        "characters": [],
        "terms": [],
    }


def empty_summary_patch() -> Dict[str, Any]:
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


def empty_reference_patch() -> Dict[str, Any]:
    return {
        "series_notes": [],
        "style_notes": [],
        "characters": [],
        "terms": [],
    }


def make_batch_key(batch_index: int) -> str:
    return f"batch_{batch_index:04d}"


def _dedupe_strings(existing: Iterable[str], incoming: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in list(existing) + list(incoming):
        cleaned = normalize_text(value)
        if not cleaned:
            continue
        marker = cleaned.casefold()
        if marker in seen:
            continue
        seen.add(marker)
        result.append(cleaned)
    return result


def _merge_named_items(
    existing: Iterable[Dict[str, Any]],
    incoming: Iterable[Dict[str, Any]],
    key_name: str,
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    def put(item: Dict[str, Any]) -> None:
        key = normalize_text(item.get(key_name))
        if not key:
            return
        marker = key.casefold()
        current = merged.get(marker, {key_name: key})
        for field, value in item.items():
            if field == key_name:
                current[field] = key
                continue
            if isinstance(value, list):
                current[field] = _dedupe_strings(current.get(field, []), value)
            else:
                cleaned = normalize_text(value)
                if cleaned:
                    current[field] = cleaned
        merged[marker] = current
        if marker not in order:
            order.append(marker)

    for item in existing:
        put(item)
    for item in incoming:
        put(item)

    return [merged[marker] for marker in order]


def _finalize_reference_character(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": normalize_text(item.get("name")),
        "aliases": _dedupe_strings([], item.get("aliases", [])),
        "role": normalize_text(item.get("role")),
        "usage_note": normalize_text(item.get("usage_note")),
        "example_sentences": _dedupe_strings([], item.get("example_sentences", []))[:REFERENCE_EXAMPLE_LIMIT],
    }


def _finalize_reference_term(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "term": normalize_text(item.get("term")),
        "category": normalize_text(item.get("category")),
        "usage_note": normalize_text(item.get("usage_note")),
        "example_sentences": _dedupe_strings([], item.get("example_sentences", []))[:REFERENCE_EXAMPLE_LIMIT],
    }


def merge_story_state(
    current_state: Dict[str, Any],
    patch: Optional[Dict[str, Any]],
    recent_summary_limit: int,
) -> Dict[str, Any]:
    result = copy.deepcopy(current_state)
    if not patch:
        return result

    result["characters"] = _merge_named_items(
        result.get("characters", []),
        patch.get("characters", []),
        "name",
    )
    result["glossary"] = _merge_named_items(
        result.get("glossary", []),
        patch.get("glossary", []),
        "source",
    )

    for field in LIST_FIELDS:
        result[field] = _dedupe_strings(result.get(field, []), patch.get(field, []))

    summary = normalize_text(patch.get("chapter_summary"))
    if summary:
        recent = result.get("recent_summaries", [])
        recent.append(summary)
        result["recent_summaries"] = recent[-recent_summary_limit:]
    else:
        result["recent_summaries"] = result.get("recent_summaries", [])[-recent_summary_limit:]

    return result


def merge_reference_profile(
    current_profile: Dict[str, Any],
    patch: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    result = copy.deepcopy(current_profile)
    if not patch:
        return result

    book_title = normalize_text(result.get("book_title")) or normalize_text(patch.get("book_title"))
    target_language = normalize_text(result.get("target_language")) or normalize_text(patch.get("target_language"))
    result["book_title"] = book_title
    result["target_language"] = target_language
    result["series_notes"] = _dedupe_strings(
        result.get("series_notes", []),
        patch.get("series_notes", []),
    )[:REFERENCE_SERIES_NOTE_LIMIT]
    result["style_notes"] = _dedupe_strings(
        result.get("style_notes", []),
        patch.get("style_notes", []),
    )[:REFERENCE_STYLE_NOTE_LIMIT]
    result["characters"] = [
        _finalize_reference_character(item)
        for item in _merge_named_items(result.get("characters", []), patch.get("characters", []), "name")
        if normalize_text(item.get("name"))
    ][:REFERENCE_CHARACTER_LIMIT]
    result["terms"] = [
        _finalize_reference_term(item)
        for item in _merge_named_items(result.get("terms", []), patch.get("terms", []), "term")
        if normalize_text(item.get("term"))
    ][:REFERENCE_TERM_LIMIT]
    return result


def story_state_for_prompt(state: Dict[str, Any], recent_summary_limit: int) -> Dict[str, Any]:
    prompt_state = {
        "book_title": normalize_text(state.get("book_title")),
        "author": normalize_text(state.get("author")),
        "characters": _compact_prompt_characters(state.get("characters", [])),
        "glossary": _compact_prompt_glossary(state.get("glossary", [])),
        "recent_summaries": _compact_prompt_strings(
            state.get("recent_summaries", []),
            limit=max(1, recent_summary_limit),
            max_chars=PROMPT_RECENT_SUMMARY_CHAR_LIMIT,
        ),
    }
    for field in LIST_FIELDS:
        prompt_state[field] = _compact_prompt_strings(
            state.get(field, []),
            limit=PROMPT_LIST_LIMITS[field],
            max_chars=PROMPT_LIST_CHAR_LIMITS[field],
            keep_head=PROMPT_LIST_KEEP_HEAD[field],
        )
    return prompt_state


def reference_profile_for_prompt(profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = profile or {}
    return {
        "book_title": normalize_text(payload.get("book_title")),
        "target_language": normalize_text(payload.get("target_language")),
        "series_notes": _compact_prompt_strings(
            payload.get("series_notes", []),
            limit=PROMPT_REFERENCE_SERIES_NOTE_LIMIT,
            max_chars=PROMPT_LIST_CHAR_LIMITS["style_notes"],
        ),
        "style_notes": _compact_prompt_strings(
            payload.get("style_notes", []),
            limit=PROMPT_REFERENCE_STYLE_NOTE_LIMIT,
            max_chars=PROMPT_LIST_CHAR_LIMITS["style_notes"],
        ),
        "characters": _compact_reference_characters(payload.get("characters", [])),
        "terms": _compact_reference_terms(payload.get("terms", [])),
    }


def _new_reference_phase(
    reference_book: Optional[Dict[str, str]],
    target_language: str,
    enabled: bool,
) -> Dict[str, Any]:
    return {
        "status": "pending" if enabled else "disabled",
        "completed_count": 0,
        "total_document_count": 0,
        "reference_profile": new_reference_profile(reference_book, target_language),
    }


def _new_summary_phase(book_metadata: Dict[str, str]) -> Dict[str, Any]:
    story_state = new_story_state(book_metadata)
    return {
        "status": "pending",
        "completed_count": 0,
        "story_state": story_state,
    }


def _new_translation_phase() -> Dict[str, Any]:
    return {
        "status": "pending",
        "completed_document_count": 0,
        "completed_batch_count": 0,
        "total_batch_count": 0,
    }


def _normalize_translated_batches(value: Any) -> Dict[str, Dict[str, Any]]:
    if isinstance(value, dict):
        normalized: Dict[str, Dict[str, Any]] = {}
        for key, entry in value.items():
            if not isinstance(entry, dict):
                continue
            batch_index = int(entry.get("batch_index") or 0)
            if batch_index <= 0:
                match = re.search(r"(\d+)$", str(key))
                batch_index = int(match.group(1)) if match else 0
            if batch_index <= 0:
                continue
            normalized[make_batch_key(batch_index)] = {
                "batch_index": batch_index,
                "translations": dict(entry.get("translations") or {}),
                "review": dict(entry.get("review") or {}),
            }
        return normalized

    if isinstance(value, list):
        normalized = {}
        for entry in value:
            if not isinstance(entry, dict):
                continue
            batch_index = int(entry.get("batch_index") or 0)
            if batch_index <= 0:
                continue
            normalized[make_batch_key(batch_index)] = {
                "batch_index": batch_index,
                "translations": dict(entry.get("translations") or {}),
                "review": dict(entry.get("review") or {}),
            }
        return normalized

    return {}


def _normalize_document_record(record: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(record or {})
    normalized.setdefault("file_name", "")
    normalized.setdefault("item_id", "")
    normalized.setdefault("source_hash", "")
    normalized["segment_count"] = int(normalized.get("segment_count", 0) or 0)
    normalized["batch_count"] = int(normalized.get("batch_count", 0) or 0)
    normalized.setdefault("summary_status", "pending")
    normalized["summary_patch"] = dict(normalized.get("summary_patch") or empty_summary_patch())
    normalized["translation_context_snapshot"] = dict(normalized.get("translation_context_snapshot") or {})
    normalized.setdefault("translation_status", "pending")
    normalized["translated_batches"] = _normalize_translated_batches(normalized.get("translated_batches"))
    normalized["translated_html"] = str(normalized.get("translated_html") or "")
    normalized["reviews"] = list(normalized.get("reviews") or [])

    if normalized.get("status") == "done" and normalized["translated_html"]:
        normalized["translation_status"] = "done"

    return normalized


def _normalize_reference_document_record(record: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(record or {})
    normalized.setdefault("file_name", "")
    normalized.setdefault("item_id", "")
    normalized.setdefault("source_hash", "")
    normalized["segment_count"] = int(normalized.get("segment_count", 0) or 0)
    normalized.setdefault("status", "pending")
    normalized["patch"] = dict(normalized.get("patch") or empty_reference_patch())
    return normalized


def _recompute_phase_counts(progress: Dict[str, Any]) -> None:
    reference_documents = progress.get("reference_documents", {}) or {}
    reference_completed = 0
    reference_total = 0
    for record in reference_documents.values():
        if not isinstance(record, dict):
            continue
        reference_total += 1
        if record.get("status") == "done":
            reference_completed += 1

    documents = progress.get("documents", {}) or {}
    summary_completed = 0
    translation_completed_documents = 0
    translation_completed_batches = 0
    translation_total_batches = 0

    for record in documents.values():
        if not isinstance(record, dict):
            continue
        batch_count = int(record.get("batch_count", 0) or 0)
        translation_total_batches += batch_count

        if record.get("summary_status") == "done":
            summary_completed += 1

        translated_batches = _normalize_translated_batches(record.get("translated_batches"))
        if record.get("translation_status") == "done":
            translation_completed_documents += 1
            translation_completed_batches += batch_count
            continue

        translation_completed_batches += min(batch_count, len(translated_batches))

    reference_phase = progress.setdefault("reference_phase", {})
    summary_phase = progress.setdefault("summary_phase", {})
    translation_phase = progress.setdefault("translation_phase", {})
    reference_phase["completed_count"] = reference_completed
    reference_phase["total_document_count"] = reference_total
    summary_phase["completed_count"] = summary_completed
    translation_phase["completed_document_count"] = translation_completed_documents
    translation_phase["completed_batch_count"] = translation_completed_batches
    translation_phase["total_batch_count"] = translation_total_batches


def _migrate_progress_document(progress: Dict[str, Any]) -> Dict[str, Any]:
    book_metadata = dict(progress.get("book") or {})
    target_language = str(progress.get("target_language") or "")
    reference_input_path = str(progress.get("reference_input_path") or "")
    reference_fingerprint = str(progress.get("reference_fingerprint") or "")
    reference_book = dict(progress.get("reference_book") or {})
    reference_enabled = bool(progress.get("reference_enabled"))
    if reference_input_path or reference_fingerprint or reference_book:
        reference_enabled = True

    reference_phase = dict(progress.get("reference_phase") or {})
    summary_phase = dict(progress.get("summary_phase") or {})
    translation_phase = dict(progress.get("translation_phase") or {})
    story_state = dict(summary_phase.get("story_state") or progress.get("story_state") or new_story_state(book_metadata))

    migrated = {
        "version": PROGRESS_VERSION,
        "input_path": progress.get("input_path", ""),
        "output_path": progress.get("output_path", ""),
        "source_language": progress.get("source_language", ""),
        "target_language": target_language,
        "book": book_metadata,
        "reference_enabled": reference_enabled,
        "reference_input_path": reference_input_path,
        "reference_fingerprint": reference_fingerprint,
        "reference_book": reference_book,
        "story_state": story_state,
        "reference_phase": _new_reference_phase(reference_book, target_language, reference_enabled),
        "summary_phase": _new_summary_phase(book_metadata),
        "translation_phase": _new_translation_phase(),
        "reference_documents": {},
        "documents": {},
    }

    migrated["reference_phase"].update(reference_phase)
    migrated["summary_phase"].update(summary_phase)
    migrated["summary_phase"]["story_state"] = story_state
    migrated["translation_phase"].update(translation_phase)
    migrated["reference_phase"]["reference_profile"] = merge_reference_profile(
        new_reference_profile(reference_book, target_language),
        migrated["reference_phase"].get("reference_profile") or {},
    )

    if reference_enabled:
        migrated["reference_phase"]["status"] = str(migrated["reference_phase"].get("status") or "pending")
    else:
        migrated["reference_phase"]["status"] = "disabled"
        migrated["reference_phase"]["completed_count"] = 0
        migrated["reference_phase"]["total_document_count"] = 0
        migrated["reference_phase"]["reference_profile"] = new_reference_profile({}, target_language)

    reference_documents = progress.get("reference_documents", {}) or {}
    for file_name, record in reference_documents.items():
        normalized = _normalize_reference_document_record(record)
        normalized["file_name"] = normalized.get("file_name") or str(file_name)
        migrated["reference_documents"][normalized["file_name"]] = normalized

    documents = progress.get("documents", {}) or {}
    for file_name, record in documents.items():
        normalized = _normalize_document_record(record)
        normalized["file_name"] = normalized.get("file_name") or str(file_name)
        migrated["documents"][normalized["file_name"]] = normalized

    _recompute_phase_counts(migrated)
    return migrated


def load_progress(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return _migrate_progress_document(payload)


def create_progress_document(
    input_path: Path,
    output_path: Path,
    source_language: str,
    target_language: str,
    book_metadata: Dict[str, str],
    *,
    reference_enabled: bool = False,
    reference_input_path: Optional[Path] = None,
    reference_fingerprint: str = "",
    reference_book: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    summary_phase = _new_summary_phase(book_metadata)
    normalized_reference_book = dict(reference_book or {})
    return {
        "version": PROGRESS_VERSION,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "source_language": source_language,
        "target_language": target_language,
        "book": book_metadata,
        "reference_enabled": bool(reference_enabled),
        "reference_input_path": str(reference_input_path) if reference_input_path else "",
        "reference_fingerprint": str(reference_fingerprint or ""),
        "reference_book": normalized_reference_book,
        "story_state": copy.deepcopy(summary_phase["story_state"]),
        "reference_phase": _new_reference_phase(normalized_reference_book, target_language, bool(reference_enabled)),
        "summary_phase": summary_phase,
        "translation_phase": _new_translation_phase(),
        "reference_documents": {},
        "documents": {},
    }


def save_progress(path: Path, payload: Dict[str, Any]) -> None:
    normalized = _migrate_progress_document(payload)
    normalized["version"] = PROGRESS_VERSION
    normalized["story_state"] = copy.deepcopy(normalized["summary_phase"]["story_state"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(normalized, handle, ensure_ascii=False, indent=2)


def get_reference_document_record(progress: Dict[str, Any], file_name: str) -> Optional[Dict[str, Any]]:
    record = progress.get("reference_documents", {}).get(file_name)
    if not isinstance(record, dict):
        return None
    normalized = _normalize_reference_document_record(record)
    progress.setdefault("reference_documents", {})[file_name] = normalized
    return normalized


def upsert_reference_document_record(progress: Dict[str, Any], record: Dict[str, Any]) -> None:
    normalized = _normalize_reference_document_record(record)
    progress.setdefault("reference_documents", {})[normalized["file_name"]] = normalized
    _recompute_phase_counts(progress)


def get_document_record(progress: Dict[str, Any], file_name: str) -> Optional[Dict[str, Any]]:
    record = progress.get("documents", {}).get(file_name)
    if not isinstance(record, dict):
        return None
    normalized = _normalize_document_record(record)
    progress.setdefault("documents", {})[file_name] = normalized
    return normalized


def upsert_document_record(progress: Dict[str, Any], record: Dict[str, Any]) -> None:
    normalized = _normalize_document_record(record)
    progress.setdefault("documents", {})[normalized["file_name"]] = normalized
    _recompute_phase_counts(progress)
