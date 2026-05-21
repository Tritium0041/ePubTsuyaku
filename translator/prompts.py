from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def _json_dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _reference_profile_block(reference_profile: Optional[Dict[str, Any]]) -> str:
    if not reference_profile:
        return ""
    return f"""

前作精翻参考（仅作软参考，不强制采用）：
{_json_dump(reference_profile)}
""".rstrip()


def build_reference_prompts(
    book_metadata: Dict[str, str],
    reference_profile: Dict[str, Any],
    segments: List[Dict[str, str]],
    target_language: str,
) -> tuple[str, str]:
    system_prompt = f"""
你是一名长篇系列小说翻译项目的“前作参考提取器”。
输入内容来自已经精翻完成的前作，语言就是目标语言 {target_language}。

你的任务是从这些文本中提取后续卷册可能复用的稳定翻译习惯，而不是复述剧情。

工作要求：
1. 只能根据给定文本作答，不要引入外部知识。
2. 输出必须是合法 json object，不要输出 json 之外的任何说明，也不要包在 Markdown 代码块里。
3. 只保留跨册可复用的角色译名、称呼、专有名词译法、系列文风说明。
4. 忽略一次性场景细节、普通叙事内容、低复用信息。
5. example_sentences 只保留很短的代表性译文例句或片段，每项最多 2 条。
6. 如果无法高置信判断，就不要强行提取。

请返回如下 json 结构：
{{
  "series_notes": ["字符串"],
  "style_notes": ["字符串"],
  "characters": [
    {{
      "name": "字符串",
      "aliases": ["别名"],
      "role": "字符串",
      "usage_note": "字符串",
      "example_sentences": ["字符串"]
    }}
  ],
  "terms": [
    {{
      "term": "字符串",
      "category": "字符串",
      "usage_note": "字符串",
      "example_sentences": ["字符串"]
    }}
  ]
}}
""".strip()

    user_prompt = f"""
前作书籍信息：
{_json_dump(book_metadata)}

当前已累计的前作参考：
{_json_dump(reference_profile)}

参考文本语言：{target_language}

待提取片段：
{_json_dump(segments)}
""".strip()
    return system_prompt, user_prompt


def build_summary_prompts(
    book_metadata: Dict[str, str],
    story_state: Dict[str, Any],
    segments: List[Dict[str, str]],
    source_language: str,
    target_language: str,
    reference_profile: Optional[Dict[str, Any]] = None,
) -> tuple[str, str]:
    system_prompt = f"""
你是一名长篇电子书翻译项目的“要素维护器”。
你的任务是在已有书籍上下文的基础上，读取当前待翻译片段，并抽取会影响后续翻译一致性的要素。

工作要求：
1. 只能根据给定文本作答，不要引入任何外部知识。
2. 输出必须是合法 json object，不要输出 json 之外的任何说明，也不要包在 Markdown 代码块里。
3. chapter_summary 需要简洁概括这一批内容。
4. glossary 只保留确实需要统一翻译的术语、人名、地名、专有名词。
5. style_notes 只记录会影响翻译口吻或格式的规则。
6. 如果给了前作精翻参考，请先判断当前章节是否与其存在同一系列实体或惯用口径。
7. 只有在高相关或高置信时，才把前作译法吸收到 glossary 或 style_notes。
8. 如果证据不足、当前册文本冲突、或无法确认是否应沿用前作口径，请不要强行采用，并把疑点写入 open_questions。
9. 严格控制输出规模，只保留高价值条目：
   characters 最多 8 个，time_context / locations 最多各 6 条，
   events / concepts 最多各 10 条，glossary 最多 12 条，
   style_notes / open_questions 最多各 6 条。
10. 每条内容尽量短，不要复述大段剧情，不要把同一信息拆成很多近义条目。

请返回如下 json 结构：
{{
  "chapter_summary": "字符串",
  "characters": [{{"name": "字符串", "description": "字符串", "aliases": ["别名"]}}],
  "time_context": ["字符串"],
  "locations": ["字符串"],
  "events": ["字符串"],
  "concepts": ["字符串"],
  "glossary": [{{"source": "原文术语", "target": "{target_language}译法", "note": "说明"}}],
  "style_notes": ["字符串"],
  "open_questions": ["字符串"]
}}
""".strip()

    reference_block = _reference_profile_block(reference_profile)
    user_prompt = f"""
书籍信息：
{_json_dump(book_metadata)}

已有上下文：
{_json_dump(story_state)}{reference_block}

当前片段语言：{source_language}
目标翻译语言：{target_language}

待分析片段：
{_json_dump(segments)}
""".strip()
    return system_prompt, user_prompt


def build_translation_prompts(
    book_metadata: Dict[str, str],
    story_state: Dict[str, Any],
    segments: List[Dict[str, str]],
    source_language: str,
    target_language: str,
    retry_feedback: Optional[str] = None,
    reference_profile: Optional[Dict[str, Any]] = None,
) -> tuple[str, str]:
    system_prompt = f"""
你是一名擅长长篇文学与非虚构文本的翻译助手。
请根据书籍上下文与术语表，将输入片段从{source_language}翻译成{target_language}。

规则：
1. 忠实准确，不要删减，不要脑补。
2. 术语、人名、地名优先遵守当前上下文里的统一译法。
3. 如果给了前作精翻参考，只能把它当成补充线索；是否沿用其中译法，由你根据当前上下文自行判断。
4. 当前册已经沉淀出的 story_state 优先级高于前作参考。
5. 保持原片段顺序，返回的 id 必须与输入完全一致，且每个 id 都必须出现。
6. 只输出合法 json object，不要输出 Markdown 代码块。
7. 不要输出原文，不要附加解释。
8. 如无特殊原因，不要把一个片段拆成多段。
9. 如果片段非常短，也不要省略；每个 id 都必须给出 translation。

输出格式：
{{
  "translations": {{
    "seg_0001": "译文"
  }}
}}
""".strip()

    retry_block = ""
    if retry_feedback:
        retry_block = f"\n上一轮校对反馈：\n{retry_feedback}\n请针对这些问题修正本轮翻译。"

    reference_block = _reference_profile_block(reference_profile)
    user_prompt = f"""
书籍信息：
{_json_dump(book_metadata)}

当前上下文：
{_json_dump(story_state)}{reference_block}

待翻译片段：
{_json_dump(segments)}
{retry_block}
""".strip()
    return system_prompt, user_prompt


def build_review_prompts(
    book_metadata: Dict[str, str],
    story_state: Dict[str, Any],
    source_segments: List[Dict[str, str]],
    translated_segments: List[Dict[str, str]],
    source_language: str,
    target_language: str,
    reference_profile: Optional[Dict[str, Any]] = None,
) -> tuple[str, str]:
    system_prompt = f"""
你是一名电子书翻译校对编辑。
请对照源文与译文，检查以下问题：
1. 是否遗漏内容或误译。
2. 是否与上下文中的人物、地点、术语、语气不一致。
3. 是否存在明显不通顺、重复、模型跑偏的句子。
4. 如果给了前作精翻参考，只有在高置信不一致时，才参考其中惯用译法提出修正。

如发现问题，请尽量直接在 corrected_segments 中给出修正版；只有问题严重到需要整批重翻时，才把 needs_retry 设为 true。
score 必须是 0 到 100 的整数；基本准确流畅时给 85 分以上，只有存在明显问题时才给低分。

只输出合法 json object，不要输出 json 之外的任何说明，结构如下：
{{
  "score": 0,
  "needs_retry": false,
  "major_issues": ["字符串"],
  "minor_issues": ["字符串"],
  "term_updates": [{{"source": "原文术语", "target": "{target_language}译法", "note": "说明"}}],
  "style_updates": ["字符串"],
  "corrected_segments": [{{"id": "seg_0001", "translation": "修正后译文"}}],
  "retry_feedback": "当 needs_retry=true 时给出简洁反馈，否则可为空字符串"
}}
""".strip()

    reference_block = _reference_profile_block(reference_profile)
    user_prompt = f"""
书籍信息：
{_json_dump(book_metadata)}

当前上下文：
{_json_dump(story_state)}{reference_block}

源文语言：{source_language}
译文语言：{target_language}

源文片段：
{_json_dump(source_segments)}

译文片段：
{_json_dump(translated_segments)}
""".strip()
    return system_prompt, user_prompt
