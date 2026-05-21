from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import ebooklib
from bs4 import BeautifulSoup
from bs4.element import Comment, Declaration, Doctype, NavigableString, ProcessingInstruction, Tag


SEGMENT_BLOCK_TAGS = {
    "p",
    "li",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "dt",
    "dd",
    "th",
    "td",
    "blockquote",
    "caption",
    "figcaption",
}

SKIPPED_TEXT_TAGS = {"script", "style", "meta", "title", "rt", "rp"}
TITLE_CANDIDATE_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6", "title")


@dataclass
class SegmentBlock:
    segment_id: str
    element: Tag
    source_text: str


@dataclass
class DocumentPlan:
    file_name: str
    item_id: str
    raw_html: str
    soup: BeautifulSoup
    blocks: List[SegmentBlock]
    segments: List[Dict[str, str]]
    source_hash: str


def content_to_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _clean_text_value(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def compute_sha1(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def safe_metadata_value(book: Any, namespace: str, name: str, fallback: str = "") -> str:
    try:
        values = book.get_metadata(namespace, name)
    except Exception:
        return fallback
    if not values:
        return fallback
    return str(values[0][0])


def extract_book_metadata(book: Any) -> Dict[str, str]:
    return {
        "title": safe_metadata_value(book, "DC", "title", "Untitled"),
        "author": safe_metadata_value(book, "DC", "creator", "Unknown"),
        "identifier": safe_metadata_value(book, "DC", "identifier", ""),
        "language": safe_metadata_value(book, "DC", "language", ""),
    }


def _resolve_spine_item(book: Any, entry: Any) -> Any:
    candidate = entry[0] if isinstance(entry, (list, tuple)) else entry
    if hasattr(candidate, "get_type"):
        return candidate
    if hasattr(book, "get_item_with_id"):
        item = book.get_item_with_id(candidate)
        if item is not None:
            return item
    if isinstance(candidate, str) and hasattr(book, "get_item_with_href"):
        item = book.get_item_with_href(candidate)
        if item is not None:
            return item
    return None


def iter_spine_documents(book: Any) -> Iterable[Any]:
    seen = set()
    spine = getattr(book, "spine", []) or []
    for entry in spine:
        item = _resolve_spine_item(book, entry)
        if item is None:
            continue
        if item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        key = getattr(item, "file_name", None) or getattr(item, "id", None)
        if getattr(item, "id", "") == "nav" or str(getattr(item, "file_name", "")).lower().endswith("nav.xhtml"):
            continue
        if key in seen:
            continue
        seen.add(key)
        yield item

    if seen:
        return

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        yield item


def _is_translatable_node(node: NavigableString) -> bool:
    if not isinstance(node, NavigableString):
        return False
    if isinstance(node, (Comment, Doctype, Declaration, ProcessingInstruction)):
        return False
    parent = getattr(node, "parent", None)
    if parent is None:
        return False
    for ancestor in (parent, *parent.parents):
        name = getattr(ancestor, "name", "").lower()
        if name in SKIPPED_TEXT_TAGS:
            return False
    return bool(str(node).strip())


def _nearest_segment_container(node: NavigableString) -> Optional[Tag]:
    parent = getattr(node, "parent", None)
    while parent is not None:
        if getattr(parent, "name", "").lower() in SEGMENT_BLOCK_TAGS:
            return parent
        parent = parent.parent
    return None


def _block_text_nodes(element: Tag) -> List[NavigableString]:
    nodes: List[NavigableString] = []
    for node in element.find_all(string=True):
        if not _is_translatable_node(node):
            continue
        if _nearest_segment_container(node) is not element:
            continue
        nodes.append(node)
    return nodes


def prepare_document(item: Any) -> DocumentPlan:
    raw_html = content_to_text(item.get_content())
    soup = BeautifulSoup(raw_html, "html.parser")
    blocks: List[SegmentBlock] = []
    segments: List[Dict[str, str]] = []

    for element in soup.find_all(list(SEGMENT_BLOCK_TAGS)):
        if not isinstance(element, Tag):
            continue
        text_nodes = _block_text_nodes(element)
        if not text_nodes:
            continue
        source_text = "".join(str(node) for node in text_nodes)
        cleaned_text = source_text.strip()
        if not cleaned_text:
            continue
        segment_id = f"seg_{len(blocks) + 1:04d}"
        blocks.append(SegmentBlock(segment_id=segment_id, element=element, source_text=source_text))
        segments.append({"id": segment_id, "text": cleaned_text})

    item_id = getattr(item, "id", None) or getattr(item, "uid", None) or ""
    return DocumentPlan(
        file_name=getattr(item, "file_name", item_id),
        item_id=item_id,
        raw_html=raw_html,
        soup=soup,
        blocks=blocks,
        segments=segments,
        source_hash=compute_sha1(raw_html),
    )


def batch_segments(
    segments: List[Dict[str, str]],
    max_batch_chars: int,
    max_batch_segments: int = 64,
) -> List[List[Dict[str, str]]]:
    if not segments:
        return []

    batches: List[List[Dict[str, str]]] = []
    current: List[Dict[str, str]] = []
    current_size = 0

    for segment in segments:
        segment_size = len(segment["text"])
        if current and (
            current_size + segment_size > max_batch_chars or len(current) >= max_batch_segments
        ):
            batches.append(current)
            current = []
            current_size = 0
        current.append(segment)
        current_size += segment_size

    if current:
        batches.append(current)
    return batches


def _preserve_whitespace(original: str, translated: str) -> str:
    leading = re.match(r"^\s*", original).group(0)
    trailing = re.search(r"\s*$", original).group(0)
    body = translated.strip()
    return f"{leading}{body}{trailing}" if body else original


def _set_block_text(element: Tag, text: str) -> None:
    element.clear()
    lines = text.splitlines()
    if not lines:
        return
    if len(lines) == 1:
        element.append(lines[0])
        return
    temp_soup = BeautifulSoup("", "html.parser")
    for index, line in enumerate(lines):
        if index > 0:
            element.append(temp_soup.new_tag("br"))
        element.append(line)


def extract_document_title(content: Any, fallback: str = "") -> str:
    raw_html = content_to_text(content)
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag_name in TITLE_CANDIDATE_TAGS:
        for element in soup.find_all(tag_name):
            text = _clean_text_value("".join(element.stripped_strings))
            if text:
                return text
    return _clean_text_value(fallback)


def apply_translations(plan: DocumentPlan, translated_map: Dict[str, str]) -> str:
    for block in plan.blocks:
        translated_text = translated_map.get(block.segment_id)
        if not translated_text:
            continue
        _set_block_text(block.element, _preserve_whitespace(block.source_text, translated_text))

    rendered = str(plan.soup)
    xml_decl = re.match(r"^\s*(<\?xml[^>]+\?>\s*)", plan.raw_html, flags=re.IGNORECASE)
    doctype = re.search(r"(<!DOCTYPE[^>]+>\s*)", plan.raw_html, flags=re.IGNORECASE)
    prefix = ""
    if xml_decl and not rendered.lstrip().startswith("<?xml"):
        prefix += xml_decl.group(1)
    if doctype and "<!DOCTYPE" not in rendered[:100].upper():
        prefix += doctype.group(1)
    return prefix + rendered.lstrip()


def set_item_content(item: Any, content: str) -> None:
    content = re.sub(r"^\s*<\?xml[^>]+\?>\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"^\s*<!DOCTYPE[^>]+>\s*", "", content, flags=re.IGNORECASE)
    if hasattr(item, "set_content"):
        item.set_content(content)
        return
    item.content = content.encode("utf-8")
