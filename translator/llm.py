from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from openai import OpenAI

from .config import DEEPSEEK_BETA_BASE_URL
from .epub_utils import batch_segments
from .prompts import (
    build_reference_prompts,
    build_review_prompts,
    build_summary_prompts,
    build_translation_prompts,
)


def _parse_json_from_text(text: str) -> Any:
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()

    if not candidate:
        raise ValueError("模型返回为空。")

    if candidate[0] in "{[":
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    start_index = None
    opening = None
    for index, char in enumerate(candidate):
        if char in "{[":
            start_index = index
            opening = char
            break
    if start_index is None:
        raise ValueError("未找到 JSON 起始位置。")

    closing = "}" if opening == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for index in range(start_index, len(candidate)):
        char = candidate[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return json.loads(candidate[start_index : index + 1])

    raise ValueError("没有找到完整的 JSON。")


def _coerce_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _extract_message_text(content: Any) -> str:
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            else:
                item_text = getattr(item, "text", None)
                if item_text:
                    text_parts.append(str(item_text))
        return "".join(text_parts).strip()
    return _coerce_string(content)


def _strict_object_schema(properties: Dict[str, Any], required: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or list(properties.keys()),
        "additionalProperties": False,
    }


def _string_array_schema() -> Dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}}


def _glossary_entry_schema() -> Dict[str, Any]:
    return _strict_object_schema(
        {
            "source": {"type": "string"},
            "target": {"type": "string"},
            "note": {"type": "string"},
        }
    )


def _character_entry_schema() -> Dict[str, Any]:
    return _strict_object_schema(
        {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "aliases": {"type": "array", "items": {"type": "string"}},
        }
    )


def _reference_character_schema() -> Dict[str, Any]:
    return _strict_object_schema(
        {
            "name": {"type": "string"},
            "aliases": {"type": "array", "items": {"type": "string"}},
            "role": {"type": "string"},
            "usage_note": {"type": "string"},
            "example_sentences": {"type": "array", "items": {"type": "string"}},
        }
    )


def _reference_term_schema() -> Dict[str, Any]:
    return _strict_object_schema(
        {
            "term": {"type": "string"},
            "category": {"type": "string"},
            "usage_note": {"type": "string"},
            "example_sentences": {"type": "array", "items": {"type": "string"}},
        }
    )


def _summary_schema() -> Dict[str, Any]:
    return _strict_object_schema(
        {
            "chapter_summary": {"type": "string"},
            "characters": {"type": "array", "items": _character_entry_schema()},
            "time_context": _string_array_schema(),
            "locations": _string_array_schema(),
            "events": _string_array_schema(),
            "concepts": _string_array_schema(),
            "glossary": {"type": "array", "items": _glossary_entry_schema()},
            "style_notes": _string_array_schema(),
            "open_questions": _string_array_schema(),
        }
    )


SUMMARY_FIELD_LIMITS = {
    "characters": 8,
    "time_context": 6,
    "locations": 6,
    "events": 10,
    "concepts": 10,
    "glossary": 12,
    "style_notes": 6,
    "open_questions": 6,
}


def _trim_value(value: Any, max_chars: int) -> str:
    text = _coerce_string(value)
    if not text or max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def _dedupe_string_items(items: List[Any], *, max_items: int, max_chars: int) -> List[str]:
    result: List[str] = []
    seen = set()
    for item in items or []:
        cleaned = _trim_value(item, max_chars)
        if not cleaned:
            continue
        marker = cleaned.casefold()
        if marker in seen:
            continue
        seen.add(marker)
        result.append(cleaned)
        if len(result) >= max_items:
            break
    return result


def _normalize_summary_characters(items: List[Any]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    seen = set()
    for item in items or []:
        if not isinstance(item, dict):
            continue
        name = _trim_value(item.get("name"), 32)
        if not name:
            continue
        marker = name.casefold()
        if marker in seen:
            continue
        seen.add(marker)
        result.append(
            {
                "name": name,
                "description": _trim_value(item.get("description"), 96),
                "aliases": _dedupe_string_items(item.get("aliases", []) or [], max_items=6, max_chars=24),
            }
        )
        if len(result) >= SUMMARY_FIELD_LIMITS["characters"]:
            break
    return result


def _normalize_summary_glossary(items: List[Any]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    seen = set()
    for item in items or []:
        if not isinstance(item, dict):
            continue
        source = _trim_value(item.get("source"), 48)
        target = _trim_value(item.get("target"), 48)
        if not source or not target:
            continue
        marker = source.casefold()
        if marker in seen:
            continue
        seen.add(marker)
        result.append(
            {
                "source": source,
                "target": target,
                "note": _trim_value(item.get("note"), 96),
            }
        )
        if len(result) >= SUMMARY_FIELD_LIMITS["glossary"]:
            break
    return result


def _reference_patch_schema() -> Dict[str, Any]:
    return _strict_object_schema(
        {
            "series_notes": _string_array_schema(),
            "style_notes": _string_array_schema(),
            "characters": {"type": "array", "items": _reference_character_schema()},
            "terms": {"type": "array", "items": _reference_term_schema()},
        }
    )


def _review_schema() -> Dict[str, Any]:
    return _strict_object_schema(
        {
            "score": {"type": "integer"},
            "needs_retry": {"type": "boolean"},
            "major_issues": _string_array_schema(),
            "minor_issues": _string_array_schema(),
            "term_updates": {"type": "array", "items": _glossary_entry_schema()},
            "style_updates": _string_array_schema(),
            "corrected_segments": {
                "type": "array",
                "items": _strict_object_schema(
                    {
                        "id": {"type": "string"},
                        "translation": {"type": "string"},
                    }
                ),
            },
            "retry_feedback": {"type": "string"},
        }
    )


def _translation_schema(expected_ids: List[str]) -> Dict[str, Any]:
    translation_properties = {segment_id: {"type": "string"} for segment_id in expected_ids}
    return _strict_object_schema(
        {
            "translations": _strict_object_schema(translation_properties, required=expected_ids),
        }
    )


def _validate_summary_payload(payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("summary 返回值必须是对象。")
    payload["chapter_summary"] = _trim_value(payload.get("chapter_summary"), 320)
    payload["characters"] = _normalize_summary_characters(payload.get("characters", []) or [])
    payload["time_context"] = _dedupe_string_items(
        payload.get("time_context", []) or [],
        max_items=SUMMARY_FIELD_LIMITS["time_context"],
        max_chars=48,
    )
    payload["locations"] = _dedupe_string_items(
        payload.get("locations", []) or [],
        max_items=SUMMARY_FIELD_LIMITS["locations"],
        max_chars=48,
    )
    payload["events"] = _dedupe_string_items(
        payload.get("events", []) or [],
        max_items=SUMMARY_FIELD_LIMITS["events"],
        max_chars=120,
    )
    payload["concepts"] = _dedupe_string_items(
        payload.get("concepts", []) or [],
        max_items=SUMMARY_FIELD_LIMITS["concepts"],
        max_chars=120,
    )
    payload["glossary"] = _normalize_summary_glossary(payload.get("glossary", []) or [])
    payload["style_notes"] = _dedupe_string_items(
        payload.get("style_notes", []) or [],
        max_items=SUMMARY_FIELD_LIMITS["style_notes"],
        max_chars=120,
    )
    payload["open_questions"] = _dedupe_string_items(
        payload.get("open_questions", []) or [],
        max_items=SUMMARY_FIELD_LIMITS["open_questions"],
        max_chars=120,
    )


def _validate_reference_patch(payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("reference patch 返回值必须是对象。")
    for field in ("series_notes", "style_notes", "characters", "terms"):
        payload.setdefault(field, [])


def _extract_translation_map(payload: Dict[str, Any]) -> Dict[str, str]:
    if not isinstance(payload, dict):
        raise ValueError("translation 返回值必须是对象。")

    translation_map: Dict[str, str] = {}
    translations = payload.get("translations")
    if isinstance(translations, dict):
        for segment_id, translation in translations.items():
            clean_id = _coerce_string(segment_id)
            if clean_id:
                translation_map[clean_id] = _coerce_string(translation)
        return translation_map

    segments = payload.get("segments", [])
    if not isinstance(segments, list):
        raise ValueError("translations 必须是对象，或 segments 必须是列表。")

    for item in segments:
        if not isinstance(item, dict):
            continue
        segment_id = _coerce_string(item.get("id"))
        translation = _coerce_string(item.get("translation"))
        if segment_id:
            translation_map[segment_id] = translation
    return translation_map


def _normalize_translation_payload(payload: Dict[str, Any], expected_ids: List[str]) -> Dict[str, str]:
    translation_map = _extract_translation_map(payload)

    missing = [segment_id for segment_id in expected_ids if not translation_map.get(segment_id)]
    if missing:
        raise ValueError(f"缺少片段译文: {', '.join(missing)}")
    return translation_map


def _normalize_review_payload(payload: Dict[str, Any], expected_ids: List[str]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("review 返回值必须是对象。")
    corrected_map: Dict[str, str] = {}
    for item in payload.get("corrected_segments", []) or []:
        if not isinstance(item, dict):
            continue
        segment_id = _coerce_string(item.get("id"))
        translation = _coerce_string(item.get("translation"))
        if segment_id and segment_id in expected_ids and translation:
            corrected_map[segment_id] = translation

    return {
        "score": int(payload.get("score", 0) or 0),
        "needs_retry": bool(payload.get("needs_retry", False)),
        "major_issues": [item for item in payload.get("major_issues", []) or [] if _coerce_string(item)],
        "minor_issues": [item for item in payload.get("minor_issues", []) or [] if _coerce_string(item)],
        "term_updates": payload.get("term_updates", []) or [],
        "style_updates": payload.get("style_updates", []) or [],
        "corrected_segments": corrected_map,
        "retry_feedback": _coerce_string(payload.get("retry_feedback")),
    }


class BaseLLMClient:
    def extract_reference_patch(
        self,
        book_metadata: Dict[str, str],
        reference_profile: Dict[str, Any],
        segments: List[Dict[str, str]],
        target_language: str,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def summarize(
        self,
        book_metadata: Dict[str, str],
        story_state: Dict[str, Any],
        segments: List[Dict[str, str]],
        source_language: str,
        target_language: str,
        reference_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def translate(
        self,
        book_metadata: Dict[str, str],
        story_state: Dict[str, Any],
        segments: List[Dict[str, str]],
        source_language: str,
        target_language: str,
        retry_feedback: Optional[str] = None,
        reference_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        raise NotImplementedError

    def review(
        self,
        book_metadata: Dict[str, str],
        story_state: Dict[str, Any],
        source_segments: List[Dict[str, str]],
        translated_segments: List[Dict[str, str]],
        source_language: str,
        target_language: str,
        reference_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class OpenAICompatibleLLMClient(BaseLLMClient):
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str],
        model: str,
        summary_model: Optional[str] = None,
        translation_model: Optional[str] = None,
        review_model: Optional[str] = None,
        timeout: int = 300,
    ) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.base_url = base_url
        self.model = model
        self.summary_model = summary_model or model
        self.translation_model = translation_model or model
        self.review_model = review_model or model
        self.timeout = timeout
        self.is_deepseek = self._is_official_deepseek_base_url(base_url)
        self.strict_client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BETA_BASE_URL) if self.is_deepseek else None

    @staticmethod
    def _is_official_deepseek_base_url(base_url: Optional[str]) -> bool:
        if not base_url:
            return False
        parsed = urlparse(base_url)
        hostname = (parsed.hostname or "").lower()
        return hostname == "api.deepseek.com"

    @staticmethod
    def _estimate_translation_max_tokens(segments: List[Dict[str, str]]) -> int:
        total_chars = sum(len(segment.get("text", "")) for segment in segments)
        total_segments = len(segments)
        return min(8192, max(1536, total_chars * 3 + total_segments * 24))

    @staticmethod
    def _estimate_review_max_tokens(segments: List[Dict[str, str]]) -> int:
        total_chars = sum(len(segment.get("translation", "")) for segment in segments)
        return min(4096, max(1024, total_chars * 2 + len(segments) * 32))

    def _chat_create(self, *, use_strict_client: bool = False, **kwargs: Any):
        client = self.strict_client if use_strict_client and self.strict_client is not None else self.client
        if self.is_deepseek:
            extra_body = dict(kwargs.get("extra_body") or {})
            extra_body.setdefault("thinking", {"type": "disabled"})
            kwargs["extra_body"] = extra_body
        return client.chat.completions.create(timeout=self.timeout, **kwargs)

    def _complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
    ) -> str:
        request_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            request_kwargs["response_format"] = response_format
        response = self._chat_create(**request_kwargs)
        return _extract_message_text(response.choices[0].message.content)

    def _call_strict_tool(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float,
        tool_name: str,
        tool_description: str,
        schema: Dict[str, Any],
        max_tokens: Optional[int] = None,
    ) -> Any:
        if self.strict_client is None:
            raise RuntimeError("当前模型未启用严格 schema 模式。")

        request_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_description,
                        "strict": True,
                        "parameters": schema,
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": tool_name}},
            "parallel_tool_calls": False,
        }
        if max_tokens is not None:
            request_kwargs["max_tokens"] = max_tokens

        response = self._chat_create(use_strict_client=True, **request_kwargs)
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []
        if not tool_calls:
            content = _extract_message_text(getattr(message, "content", ""))
            raise ValueError(f"严格 schema 调用未返回 tool_call：{content or 'empty content'}")
        arguments = _coerce_string(getattr(tool_calls[0].function, "arguments", ""))
        if not arguments:
            raise ValueError("严格 schema 调用未返回 arguments。")
        return _parse_json_from_text(arguments)

    def _call_json(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None,
        schema: Optional[Dict[str, Any]] = None,
        tool_name: str = "return_json",
        tool_description: str = "Return a json object that matches the requested schema.",
    ) -> Any:
        strict_error = None
        if schema is not None and self.strict_client is not None:
            try:
                return self._call_strict_tool(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=model,
                    temperature=temperature,
                    tool_name=tool_name,
                    tool_description=tool_description,
                    schema=schema,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                strict_error = exc

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        last_error = None
        for _ in range(3):
            response_text = self._complete(
                messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"} if self.is_deepseek else None,
            )
            try:
                return _parse_json_from_text(response_text)
            except Exception as exc:
                last_error = exc
                messages.append({"role": "assistant", "content": response_text})
                messages.append(
                    {
                        "role": "user",
                        "content": f"上一条回复不是合法 json 或缺少必需字段：{exc}。请重新输出完整且合法的 json object，不要附带任何解释。",
                    }
                )
        if strict_error is not None:
            raise RuntimeError(f"严格 schema 调用失败：{strict_error}；JSON mode 也连续 3 次失败：{last_error}")
        raise RuntimeError(f"模型连续 3 次未返回合法 JSON: {last_error}")

    def extract_reference_patch(
        self,
        book_metadata: Dict[str, str],
        reference_profile: Dict[str, Any],
        segments: List[Dict[str, str]],
        target_language: str,
    ) -> Dict[str, Any]:
        system_prompt, user_prompt = build_reference_prompts(
            book_metadata,
            reference_profile,
            segments,
            target_language,
        )
        payload = self._call_json(
            system_prompt,
            user_prompt,
            model=self.summary_model,
            temperature=0.0,
            max_tokens=2048,
            schema=_reference_patch_schema(),
            tool_name="return_reference_patch",
            tool_description="Return a structured previous-volume reference patch json object.",
        )
        _validate_reference_patch(payload)
        return payload

    def summarize(
        self,
        book_metadata: Dict[str, str],
        story_state: Dict[str, Any],
        segments: List[Dict[str, str]],
        source_language: str,
        target_language: str,
        reference_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        system_prompt, user_prompt = build_summary_prompts(
            book_metadata,
            story_state,
            segments,
            source_language,
            target_language,
            reference_profile=reference_profile,
        )
        payload = self._call_json(
            system_prompt,
            user_prompt,
            model=self.summary_model,
            temperature=0.0,
            max_tokens=2048,
            schema=_summary_schema(),
            tool_name="return_summary",
            tool_description="Return a structured chapter summary json object.",
        )
        _validate_summary_payload(payload)
        return payload

    def _translate_once(
        self,
        book_metadata: Dict[str, str],
        story_state: Dict[str, Any],
        segments: List[Dict[str, str]],
        source_language: str,
        target_language: str,
        retry_feedback: Optional[str],
        reference_profile: Optional[Dict[str, Any]],
    ) -> Dict[str, str]:
        system_prompt, user_prompt = build_translation_prompts(
            book_metadata,
            story_state,
            segments,
            source_language,
            target_language,
            retry_feedback=retry_feedback,
            reference_profile=reference_profile,
        )
        payload = self._call_json(
            system_prompt,
            user_prompt,
            model=self.translation_model,
            temperature=0.1,
            max_tokens=self._estimate_translation_max_tokens(segments),
            schema=_translation_schema([item["id"] for item in segments]),
            tool_name="return_translations",
            tool_description="Return all translations as a json object keyed by segment id.",
        )
        return _extract_translation_map(payload)

    def _repair_missing_translations(
        self,
        book_metadata: Dict[str, str],
        story_state: Dict[str, Any],
        segments: List[Dict[str, str]],
        source_language: str,
        target_language: str,
        retry_feedback: Optional[str],
        reference_profile: Optional[Dict[str, Any]],
    ) -> Dict[str, str]:
        repaired: Dict[str, str] = {}
        repair_feedback = "上一轮返回里有部分 id 缺失。请只补齐本轮给出的这些 id，并保证每个 id 都返回译文。"
        if retry_feedback:
            repair_feedback = f"{retry_feedback}\n\n{repair_feedback}"
        for chunk in batch_segments(segments, max_batch_chars=900, max_batch_segments=24):
            repaired.update(
                self._translate_once(
                    book_metadata=book_metadata,
                    story_state=story_state,
                    segments=chunk,
                    source_language=source_language,
                    target_language=target_language,
                    retry_feedback=repair_feedback,
                    reference_profile=reference_profile,
                )
            )
        return repaired

    def translate(
        self,
        book_metadata: Dict[str, str],
        story_state: Dict[str, Any],
        segments: List[Dict[str, str]],
        source_language: str,
        target_language: str,
        retry_feedback: Optional[str] = None,
        reference_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        expected_ids = [item["id"] for item in segments]
        translation_map = self._translate_once(
            book_metadata=book_metadata,
            story_state=story_state,
            segments=segments,
            source_language=source_language,
            target_language=target_language,
            retry_feedback=retry_feedback,
            reference_profile=reference_profile,
        )
        missing_ids = [segment_id for segment_id in expected_ids if not translation_map.get(segment_id)]
        if missing_ids:
            repair_segments = [segment for segment in segments if segment["id"] in set(missing_ids)]
            translation_map.update(
                self._repair_missing_translations(
                    book_metadata=book_metadata,
                    story_state=story_state,
                    segments=repair_segments,
                    source_language=source_language,
                    target_language=target_language,
                    retry_feedback=retry_feedback,
                    reference_profile=reference_profile,
                )
            )
        missing_after_repair = [segment_id for segment_id in expected_ids if not translation_map.get(segment_id)]
        if missing_after_repair:
            raise RuntimeError(f"仍有未补齐的片段译文: {', '.join(missing_after_repair)}")
        return {segment_id: translation_map[segment_id] for segment_id in expected_ids}

    def review(
        self,
        book_metadata: Dict[str, str],
        story_state: Dict[str, Any],
        source_segments: List[Dict[str, str]],
        translated_segments: List[Dict[str, str]],
        source_language: str,
        target_language: str,
        reference_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        system_prompt, user_prompt = build_review_prompts(
            book_metadata,
            story_state,
            source_segments,
            translated_segments,
            source_language,
            target_language,
            reference_profile=reference_profile,
        )
        payload = self._call_json(
            system_prompt,
            user_prompt,
            model=self.review_model,
            temperature=0.0,
            max_tokens=self._estimate_review_max_tokens(translated_segments),
            schema=_review_schema(),
            tool_name="return_review",
            tool_description="Return a structured review json object.",
        )
        expected_ids = [item["id"] for item in source_segments]
        return _normalize_review_payload(payload, expected_ids)


class MockLLMClient(BaseLLMClient):
    def extract_reference_patch(
        self,
        book_metadata: Dict[str, str],
        reference_profile: Dict[str, Any],
        segments: List[Dict[str, str]],
        target_language: str,
    ) -> Dict[str, Any]:
        return {
            "series_notes": [],
            "style_notes": [f"参考语言：{target_language}"],
            "characters": [],
            "terms": [],
        }

    def summarize(
        self,
        book_metadata: Dict[str, str],
        story_state: Dict[str, Any],
        segments: List[Dict[str, str]],
        source_language: str,
        target_language: str,
        reference_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        preview = " / ".join(segment["text"] for segment in segments[:2])
        return {
            "chapter_summary": f"Mock summary: {preview[:120]}",
            "characters": [],
            "time_context": [],
            "locations": [],
            "events": [preview[:120]] if preview else [],
            "concepts": [],
            "glossary": [],
            "style_notes": [f"目标语言：{target_language}"],
            "open_questions": [],
        }

    def translate(
        self,
        book_metadata: Dict[str, str],
        story_state: Dict[str, Any],
        segments: List[Dict[str, str]],
        source_language: str,
        target_language: str,
        retry_feedback: Optional[str] = None,
        reference_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        prefix = f"[{target_language}] "
        return {segment["id"]: prefix + segment["text"] for segment in segments}

    def review(
        self,
        book_metadata: Dict[str, str],
        story_state: Dict[str, Any],
        source_segments: List[Dict[str, str]],
        translated_segments: List[Dict[str, str]],
        source_language: str,
        target_language: str,
        reference_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
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
