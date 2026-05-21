from __future__ import annotations

import argparse
import hashlib
import os
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ebooklib import epub
from flask import Flask, flash, jsonify, redirect, render_template, request, send_file, url_for

from .config import (
    ALIYUN_BASE_URL,
    DEEPSEEK_BASE_URL,
    PipelineConfig,
    resolve_provider_settings,
)
from .epub_utils import extract_book_metadata, iter_spine_documents
from .pipeline import run_translation_pipeline_with_retries


def _sanitize_filename_part(value: str) -> str:
    import re

    cleaned = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", value.strip(), flags=re.UNICODE)
    return cleaned.strip("_") or "translated"


def _default_model_for_preset(provider_preset: str) -> str:
    if provider_preset == "deepseek":
        return "deepseek-v4-flash"
    if provider_preset == "aliyun":
        return "qwen-max"
    if provider_preset == "mock":
        return "mock-model"
    return "gpt-4.1-mini"


def _default_form_state() -> Dict[str, Any]:
    return {
        "existing_file": "",
        "manual_input_path": "",
        "reference_existing_file": "",
        "reference_manual_input_path": "",
        "source_language": "日语",
        "target_language": "中文",
        "provider_preset": "deepseek",
        "api_key": "",
        "model": "deepseek-v4-flash",
        "base_url": DEEPSEEK_BASE_URL,
        "summary_model": "",
        "translation_model": "",
        "review_model": "",
        "translation_workers": "4",
        "max_batch_chars": "4000",
        "max_batch_segments": "64",
        "max_review_retries": "2",
        "min_review_score": "85",
        "recent_summary_limit": "5",
        "title_suffix": "（中文译本）",
        "reset_progress": True,
    }


def _form_state_from_form_data(form_data: Dict[str, str]) -> Dict[str, Any]:
    state = _default_form_state()
    for key in (
        "existing_file",
        "manual_input_path",
        "reference_existing_file",
        "reference_manual_input_path",
        "source_language",
        "target_language",
        "provider_preset",
        "model",
        "base_url",
        "summary_model",
        "translation_model",
        "review_model",
        "translation_workers",
        "max_batch_chars",
        "max_batch_segments",
        "max_review_retries",
        "min_review_score",
        "recent_summary_limit",
        "title_suffix",
    ):
        value = form_data.get(key)
        if value is not None:
            state[key] = value
    state["reset_progress"] = form_data.get("reset_progress") == "on"
    return state


def _hash_uploaded_file(file_storage: Any) -> str:
    stream = getattr(file_storage, "stream", None)
    if stream is None:
        raise RuntimeError("上传文件流不可用。")
    digest = hashlib.sha1()
    stream.seek(0)
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    stream.seek(0)
    return digest.hexdigest()


def _safe_upload_name(original_name: str, content_hash: str) -> str:
    file_name = Path(original_name or "upload.epub").name
    stem = file_name.rsplit(".", 1)[0]
    suffix = Path(file_name).suffix.lower() or ".epub"
    safe_stem = _sanitize_filename_part(stem)
    return f"{safe_stem}.{content_hash[:12]}{suffix}"


def discover_epub_files(project_root: Path) -> List[Dict[str, str]]:
    seen = set()
    result: List[Dict[str, str]] = []
    excluded = {".git", "__pycache__", "epubOutput", ".webui"}

    for path in sorted(project_root.rglob("*.epub")):
        relative_parts = path.relative_to(project_root).parts
        if any(part in excluded for part in relative_parts):
            continue
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(
            {
                "path": resolved,
                "label": str(path.relative_to(project_root)),
                "name": path.name,
            }
        )
    return result


def _count_documents(input_path: Path) -> int:
    book = epub.read_epub(str(input_path))
    return len(list(iter_spine_documents(book)))


def _summarize_book(input_path: Path) -> Dict[str, Any]:
    book = epub.read_epub(str(input_path))
    metadata = extract_book_metadata(book)
    metadata["document_count"] = len(list(iter_spine_documents(book)))
    return metadata


def resolve_web_provider_settings(
    provider_preset: str,
    api_key: Optional[str],
    base_url: Optional[str],
    model: Optional[str],
) -> tuple[str, Optional[str], Optional[str], str]:
    clean_key = (api_key or "").strip() or None
    clean_base_url = (base_url or "").strip() or None
    clean_model = (model or "").strip() or None

    if provider_preset == "mock":
        return "mock", None, None, clean_model or "mock-model"

    if provider_preset == "deepseek":
        effective_key = clean_key or os.environ.get("DEEPSEEK_API_KEY")
        if not effective_key:
            raise RuntimeError("DeepSeek 模式需要填写 API Key，或提前设置 DEEPSEEK_API_KEY。")
        return "openai-compatible", effective_key, clean_base_url or DEEPSEEK_BASE_URL, clean_model or "deepseek-v4-flash"

    if provider_preset == "aliyun":
        effective_key = clean_key or os.environ.get("ALIYUN_API_KEY")
        if not effective_key:
            raise RuntimeError("阿里云模式需要填写 API Key，或提前设置 ALIYUN_API_KEY。")
        return "openai-compatible", effective_key, clean_base_url or ALIYUN_BASE_URL, clean_model or "qwen-max"

    if provider_preset == "openai-compatible":
        effective_key = clean_key or os.environ.get("EPUB_TRANSLATOR_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not effective_key:
            raise RuntimeError("OpenAI 兼容模式需要填写 API Key，或提前设置 EPUB_TRANSLATOR_API_KEY / OPENAI_API_KEY。")
        return "openai-compatible", effective_key, clean_base_url, clean_model or "gpt-4.1-mini"

    return resolve_provider_settings(
        provider="auto",
        api_key_env=None,
        explicit_base_url=clean_base_url,
        explicit_model=clean_model,
    )


@dataclass
class JobState:
    id: str
    status: str
    input_path: str
    input_label: str
    output_path: str
    progress_path: str
    source_language: str
    target_language: str
    model: str
    provider_preset: str
    base_url: Optional[str]
    title_suffix: str
    reference_input_path: str = ""
    reference_input_label: str = ""
    translation_workers: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    reference_enabled: bool = False
    reference_fingerprint: str = ""
    reference_book: Dict[str, Any] = field(default_factory=dict)
    reference_completed_count: int = 0
    reference_total_count: int = 0
    total_document_count: int = 0
    summary_completed_count: int = 0
    summary_total_count: int = 0
    translation_completed_batch_count: int = 0
    translation_total_batch_count: int = 0
    active_workers: int = 0
    completed_count: int = 0
    processed_count: int = 0
    skipped_count: int = 0
    retried_batches: int = 0
    current_phase: str = "idle"
    current_document: str = ""
    current_document_index: int = 0
    current_segment_count: int = 0
    current_batch_index: int = 0
    current_batch_total: int = 0
    last_error: str = ""
    result: Dict[str, Any] = field(default_factory=dict)
    book_metadata: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)

    def to_public_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["download_url"] = url_for("download_output", job_id=self.id) if self.status == "completed" else None
        payload["can_start_new_job"] = self.status in {"idle", "completed", "failed"}
        return payload


class JobManager:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.webui_root = project_root / ".webui"
        self.upload_dir = self.webui_root / "uploads"
        self.progress_dir = self.webui_root / "progress"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._current_job: Optional[JobState] = None
        self._thread: Optional[threading.Thread] = None
        self._last_form_state: Dict[str, Any] = _default_form_state()

    def _get_job(self, job_id: str) -> Optional[JobState]:
        if self._current_job and self._current_job.id == job_id:
            return self._current_job
        return None

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            files = discover_epub_files(self.project_root)
            if self._current_job is None:
                return {
                    "job": None,
                    "available_files": files,
                    "form_state": dict(self._last_form_state),
                }
            return {
                "job": self._current_job.to_public_dict(),
                "available_files": files,
                "form_state": dict(self._last_form_state),
            }

    def _append_log(self, job_id: str, message: str) -> None:
        with self._lock:
            job = self._get_job(job_id)
            if job is None:
                return
            timestamp = time.strftime("%H:%M:%S")
            job.logs.append(f"[{timestamp}] {message}")
            job.logs = job.logs[-400:]

    def _apply_event(self, job_id: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            job = self._get_job(job_id)
            if job is None:
                return

            event = payload.get("event")
            if event == "run_started":
                job.total_document_count = int(payload.get("total_document_count", job.total_document_count) or 0)
                job.summary_total_count = job.total_document_count
                job.reference_enabled = bool(payload.get("reference_enabled", job.reference_enabled))
                job.reference_input_path = str(payload.get("reference_input_path") or job.reference_input_path)
                job.reference_fingerprint = str(payload.get("reference_fingerprint") or job.reference_fingerprint)
                job.reference_book = dict(payload.get("reference_book") or job.reference_book)
                return

            if event == "reference_phase_started":
                job.current_phase = "reference"
                job.reference_total_count = int(payload.get("total_document_count", job.reference_total_count) or 0)
                job.reference_book = dict(payload.get("reference_book") or job.reference_book)
                job.reference_fingerprint = str(payload.get("reference_fingerprint") or job.reference_fingerprint)
                return

            if event == "reference_document_started":
                job.current_phase = "reference"
                job.current_document = payload.get("file_name", "")
                job.current_document_index = int(payload.get("index", 0) or 0)
                job.current_segment_count = int(payload.get("segment_count", 0) or 0)
                job.current_batch_index = 0
                job.current_batch_total = 0
                return

            if event == "reference_document_done":
                job.current_phase = "reference"
                job.current_document = payload.get("file_name", "")
                job.reference_completed_count = int(
                    payload.get("reference_completed_count", job.reference_completed_count) or 0
                )
                job.reference_total_count = int(payload.get("total_document_count", job.reference_total_count) or 0)
                return

            if event == "reference_phase_finished":
                job.current_phase = "summary"
                job.reference_completed_count = int(
                    payload.get("reference_completed_count", job.reference_completed_count) or 0
                )
                job.reference_total_count = int(payload.get("total_document_count", job.reference_total_count) or 0)
                return

            if event == "summary_phase_started":
                job.current_phase = "summary"
                job.summary_total_count = int(payload.get("total_document_count", job.summary_total_count) or 0)
                return

            if event == "summary_document_started":
                job.current_phase = "summary"
                job.current_document = payload.get("file_name", "")
                job.current_document_index = int(payload.get("index", 0) or 0)
                job.current_segment_count = int(payload.get("segment_count", 0) or 0)
                job.current_batch_index = 0
                job.current_batch_total = int(payload.get("total_batches", 0) or 0)
                return

            if event == "summary_document_done":
                job.current_phase = "summary"
                job.current_document = payload.get("file_name", "")
                job.summary_completed_count = int(
                    payload.get("summary_completed_count", job.summary_completed_count) or 0
                )
                return

            if event == "summary_phase_finished":
                job.current_phase = "translation"
                job.summary_completed_count = int(
                    payload.get("summary_completed_count", job.summary_completed_count) or 0
                )
                return

            if event == "translation_phase_started":
                job.current_phase = "translation"
                job.translation_total_batch_count = int(
                    payload.get("total_batch_count", job.translation_total_batch_count) or 0
                )
                job.translation_completed_batch_count = int(
                    payload.get("completed_batch_count", job.translation_completed_batch_count) or 0
                )
                job.translation_workers = int(payload.get("translation_workers", job.translation_workers) or 0)
                return

            if event == "translation_document_started":
                job.current_phase = "translation"
                job.current_document = payload.get("file_name", "")
                job.current_document_index = int(payload.get("index", 0) or 0)
                job.current_segment_count = int(payload.get("segment_count", 0) or 0)
                job.current_batch_total = int(payload.get("total_batches", 0) or 0)
                return

            if event == "translation_batch_started":
                job.current_phase = "translation"
                job.current_document = payload.get("file_name", "")
                job.current_document_index = int(payload.get("index", 0) or 0)
                job.current_batch_index = int(payload.get("batch_index", 0) or 0)
                job.current_batch_total = int(payload.get("total_batches", 0) or 0)
                job.active_workers = int(payload.get("active_workers", job.active_workers) or 0)
                job.translation_workers = int(payload.get("translation_workers", job.translation_workers) or 0)
                return

            if event == "translation_batch_retry":
                job.retried_batches = int(payload.get("retried_batches", job.retried_batches) or 0)
                job.active_workers = int(payload.get("active_workers", job.active_workers) or 0)
                return

            if event == "translation_batch_done":
                job.translation_completed_batch_count = int(
                    payload.get("completed_batch_count", job.translation_completed_batch_count) or 0
                )
                job.translation_total_batch_count = int(
                    payload.get("total_batch_count", job.translation_total_batch_count) or 0
                )
                job.active_workers = int(payload.get("active_workers", job.active_workers) or 0)
                return

            if event == "translation_document_reused":
                job.completed_count = int(payload.get("completed_count", job.completed_count) or 0)
                job.skipped_count = int(payload.get("skipped_count", job.skipped_count) or 0)
                job.processed_count = int(payload.get("processed_count", job.processed_count) or 0)
                return

            if event == "translation_document_done":
                job.completed_count = int(payload.get("completed_count", job.completed_count) or 0)
                job.processed_count = int(payload.get("processed_count", job.processed_count) or 0)
                job.skipped_count = int(payload.get("skipped_count", job.skipped_count) or 0)
                return

            if event == "translation_phase_finished":
                job.translation_completed_batch_count = int(
                    payload.get("completed_batch_count", job.translation_completed_batch_count) or 0
                )
                job.translation_total_batch_count = int(
                    payload.get("total_batch_count", job.translation_total_batch_count) or 0
                )
                job.completed_count = int(payload.get("completed_count", job.completed_count) or 0)
                job.processed_count = int(payload.get("processed_count", job.processed_count) or 0)
                job.skipped_count = int(payload.get("skipped_count", job.skipped_count) or 0)
                job.retried_batches = int(payload.get("retried_batches", job.retried_batches) or 0)
                job.active_workers = 0
                return

            if event == "run_finished":
                job.result = {key: value for key, value in payload.items() if key != "event"}

    def _path_label(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.project_root))
        except ValueError:
            return str(path)

    def _resolve_input_source(
        self,
        job_id: str,
        form_data: Dict[str, str],
        file_storage: Any,
        *,
        upload_error_prefix: str,
        manual_key: str,
        existing_key: str,
        required: bool,
    ) -> tuple[Optional[Path], str]:
        uploaded = file_storage and getattr(file_storage, "filename", "")
        if uploaded:
            if not str(file_storage.filename).lower().endswith(".epub"):
                raise RuntimeError(f"{upload_error_prefix}文件必须是 .epub。")
            content_hash = _hash_uploaded_file(file_storage)
            destination = self.upload_dir / _safe_upload_name(file_storage.filename, content_hash)
            file_storage.save(destination)
            return destination.resolve(), file_storage.filename

        manual_path = (form_data.get(manual_key) or "").strip()
        if manual_path:
            input_path = Path(manual_path).expanduser()
            if not input_path.is_absolute():
                input_path = (self.project_root / input_path).resolve()
            if not input_path.exists():
                raise RuntimeError(f"{upload_error_prefix}文件不存在: {input_path}")
            if input_path.suffix.lower() != ".epub":
                raise RuntimeError(f"{upload_error_prefix}文件必须是 .epub。")
            return input_path, str(input_path)

        existing_file = (form_data.get(existing_key) or "").strip()
        if existing_file:
            input_path = Path(existing_file)
            if not input_path.is_absolute():
                input_path = (self.project_root / input_path).resolve()
            if not input_path.exists():
                raise RuntimeError(f"选择的{upload_error_prefix}文件不存在: {input_path}")
            return input_path, self._path_label(input_path)

        if required:
            raise RuntimeError("请至少选择一个 EPUB 文件，或上传一个新文件。")
        return None, ""

    def start_job(self, form_data: Dict[str, str], files: Any) -> JobState:
        with self._lock:
            if self._current_job and self._current_job.status == "running":
                raise RuntimeError("当前已有翻译任务在运行，请等待完成后再启动新任务。")

            self._last_form_state = _form_state_from_form_data(form_data)
            job_id = uuid.uuid4().hex[:10]
            input_path, input_label = self._resolve_input_source(
                job_id,
                form_data,
                files.get("upload_file"),
                upload_error_prefix="输入",
                manual_key="manual_input_path",
                existing_key="existing_file",
                required=True,
            )
            reference_input_path, reference_input_label = self._resolve_input_source(
                job_id,
                form_data,
                files.get("reference_upload_file"),
                upload_error_prefix="参考 EPUB ",
                manual_key="reference_manual_input_path",
                existing_key="reference_existing_file",
                required=False,
            )
            if input_path is None:
                raise RuntimeError("请至少选择一个 EPUB 文件，或上传一个新文件。")
            provider_preset = (form_data.get("provider_preset") or "deepseek").strip()
            api_key = (form_data.get("api_key") or "").strip() or None
            model = (form_data.get("model") or "").strip() or _default_model_for_preset(provider_preset)
            base_url = (form_data.get("base_url") or "").strip() or None
            provider, resolved_api_key, resolved_base_url, resolved_model = resolve_web_provider_settings(
                provider_preset=provider_preset,
                api_key=api_key,
                base_url=base_url,
                model=model,
            )

            source_language = (form_data.get("source_language") or "日语").strip() or "日语"
            target_language = (form_data.get("target_language") or "中文").strip() or "中文"
            title_suffix = (form_data.get("title_suffix") or "（中文译本）").strip()
            logical_stem = _sanitize_filename_part(Path(input_label or input_path.name).stem)
            progress_name = f"{logical_stem}.{_sanitize_filename_part(target_language)}.json"
            progress_path = self.progress_dir / progress_name
            output_path = self.project_root / "epubOutput" / f"{logical_stem}.{_sanitize_filename_part(target_language)}.epub"

            translation_workers = max(1, int(form_data.get("translation_workers") or 4))
            max_batch_chars = int(form_data.get("max_batch_chars") or 4000)
            max_batch_segments = int(form_data.get("max_batch_segments") or 64)
            max_review_retries = int(form_data.get("max_review_retries") or 2)
            min_review_score = int(form_data.get("min_review_score") or 85)
            recent_summary_limit = int(form_data.get("recent_summary_limit") or 5)
            reset_progress = form_data.get("reset_progress") == "on"

            config = PipelineConfig(
                input_path=input_path,
                output_path=output_path,
                progress_path=progress_path,
                reference_input_path=reference_input_path,
                source_language=source_language,
                target_language=target_language,
                provider=provider,
                api_key=resolved_api_key,
                base_url=resolved_base_url,
                model=resolved_model,
                summary_model=(form_data.get("summary_model") or "").strip() or resolved_model,
                translation_model=(form_data.get("translation_model") or "").strip() or resolved_model,
                review_model=(form_data.get("review_model") or "").strip() or resolved_model,
                translation_workers=translation_workers,
                auto_resume_retries=2,
                max_batch_chars=max_batch_chars,
                max_batch_segments=max_batch_segments,
                max_review_retries=max_review_retries,
                min_review_score=min_review_score,
                recent_summary_limit=recent_summary_limit,
                title_suffix=title_suffix or None,
                reset_progress=reset_progress,
            )

            book_metadata = _summarize_book(input_path)
            reference_book_metadata = _summarize_book(reference_input_path) if reference_input_path else {}
            job = JobState(
                id=job_id,
                status="queued",
                input_path=str(input_path),
                input_label=input_label,
                output_path=str(output_path),
                progress_path=str(progress_path),
                source_language=source_language,
                target_language=target_language,
                model=resolved_model,
                provider_preset=provider_preset,
                base_url=resolved_base_url,
                title_suffix=title_suffix,
                reference_input_path=str(reference_input_path) if reference_input_path else "",
                reference_input_label=reference_input_label,
                translation_workers=translation_workers,
                reference_enabled=reference_input_path is not None,
                reference_book=reference_book_metadata,
                reference_total_count=int(reference_book_metadata.get("document_count", 0) or 0),
                total_document_count=book_metadata.get("document_count", 0),
                summary_total_count=book_metadata.get("document_count", 0),
                book_metadata=book_metadata,
            )
            self._current_job = job
            self._thread = threading.Thread(target=self._run_job, args=(job_id, config), daemon=True)
            self._thread.start()
            return job

    def _run_job(self, job_id: str, config: PipelineConfig) -> None:
        with self._lock:
            job = self._get_job(job_id)
            if job is not None:
                job.status = "running"
                job.started_at = time.time()

        self._append_log(job_id, f"准备翻译: {config.input_path.name}")
        if config.reference_input_path is not None:
            self._append_log(job_id, f"前作参考: {config.reference_input_path}")
        self._append_log(job_id, f"模型: {config.model}")
        self._append_log(job_id, f"输出文件: {config.output_path}")

        try:
            result = run_translation_pipeline_with_retries(
                config,
                log_func=lambda message: self._append_log(job_id, message),
                status_callback=lambda payload: self._apply_event(job_id, payload),
            )
        except Exception as exc:
            failure_message = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self._append_log(job_id, failure_message)
            self._append_log(job_id, traceback.format_exc())
            with self._lock:
                job = self._get_job(job_id)
                if job is not None:
                    job.status = "failed"
                    job.finished_at = time.time()
                    job.last_error = failure_message
                    job.active_workers = 0
            return

        with self._lock:
            job = self._get_job(job_id)
            if job is not None:
                job.status = "completed"
                job.finished_at = time.time()
                job.result = result
                job.completed_count = int(result.get("completed_count", job.completed_count) or 0)
                job.processed_count = int(result.get("processed_count", job.processed_count) or 0)
                job.skipped_count = int(result.get("skipped_count", job.skipped_count) or 0)
                job.retried_batches = int(result.get("retried_batches", job.retried_batches) or 0)
                job.summary_completed_count = int(result.get("summary_completed_count", job.summary_completed_count) or 0)
                job.translation_completed_batch_count = int(
                    result.get("translation_completed_batch_count", job.translation_completed_batch_count) or 0
                )
                job.translation_total_batch_count = int(
                    result.get("translation_total_batch_count", job.translation_total_batch_count) or 0
                )
                job.translation_workers = int(result.get("translation_workers", job.translation_workers) or 0)
                job.reference_enabled = bool(result.get("reference_enabled", job.reference_enabled))
                job.reference_completed_count = int(result.get("reference_completed_count", job.reference_completed_count) or 0)
                job.active_workers = 0
        self._append_log(job_id, "翻译完成。")

    def send_output(self, job_id: str):
        with self._lock:
            job = self._get_job(job_id)
            if job is None:
                raise FileNotFoundError("未找到任务。")
            output_path = Path(job.output_path)
        if not output_path.exists():
            raise FileNotFoundError("输出文件不存在。")
        return send_file(output_path, as_attachment=True, download_name=output_path.name)


def create_app(project_root: Optional[Path] = None) -> Flask:
    root = (project_root or Path(__file__).resolve().parents[1]).resolve()
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.secret_key = "epub-translator-webui"
    manager = JobManager(root)
    app.config["JOB_MANAGER"] = manager
    app.config["PROJECT_ROOT"] = root

    @app.get("/")
    def index():
        snapshot = manager.snapshot()
        return render_template(
            "index.html",
            state=snapshot,
            provider_defaults={
                "deepseek_model": "deepseek-v4-flash",
                "aliyun_model": "qwen-max",
                "openai_model": "gpt-4.1-mini",
                "deepseek_base_url": DEEPSEEK_BASE_URL,
                "aliyun_base_url": ALIYUN_BASE_URL,
            },
        )

    @app.post("/start")
    def start():
        manager._last_form_state = _form_state_from_form_data(request.form.to_dict())
        try:
            job = manager.start_job(request.form.to_dict(), request.files)
        except Exception as exc:
            flash(str(exc), "error")
            return redirect(url_for("index"))
        flash(f"任务已启动：{job.input_label}", "success")
        return redirect(url_for("index"))

    @app.get("/api/status")
    def api_status():
        return jsonify(manager.snapshot())

    @app.get("/download/<job_id>")
    def download_output(job_id: str):
        try:
            return manager.send_output(job_id)
        except FileNotFoundError as exc:
            flash(str(exc), "error")
            return redirect(url_for("index"))

    return app


def build_web_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动 EPUB Translator Web UI。")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认 127.0.0.1。")
    parser.add_argument("--port", type=int, default=7860, help="监听端口，默认 7860。")
    parser.add_argument("--debug", action="store_true", help="是否开启 Flask debug。")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_web_parser()
    args = parser.parse_args(argv)
    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)
    return 0
