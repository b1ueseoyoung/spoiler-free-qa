"""wikitext → 본문 텍스트.

틀(머리말·꼬리말 등)과 각주는 지우고, 링크는 보이는 글자만 남긴다. 장 제목(첫 제목줄)은 본문에서 빼 따로 둔다.
한자 병기와 검열 표시(×)는 원문 그대로 둔다(인용 검증의 기준이 원문이다).
마크업이 하나라도 남으면 실패로 본다.
"""

from __future__ import annotations

import html
import json
import re
from collections import Counter
from dataclasses import dataclass, field

import mwparserfromhell
from mwparserfromhell.nodes import Comment, ExternalLink, Heading, HTMLEntity, Tag, Template, Text, Wikilink

from sfqa.paths import CHAPTERS, CLEAN, CLEAN_REPORT, SOURCES, clean_path, raw_path

DROP_TAGS = {"ref", "references"}
SKIP_LINK_PREFIXES = ("분류:", "Category:", "파일:", "File:", "그림:", "Image:")
_MAGIC = re.compile(r"__[A-Z]+__")
_RESIDUE = re.compile(r"\{\{|\}\}|\[\[|\]\]|<ref|</?[a-zA-Z][^>]*>|__[A-Z]+__|^=+.*=+\s*$", re.MULTILINE)


class CleanError(ValueError):
    pass


@dataclass
class Cleaned:
    text: str
    heading: str | None
    removed_templates: Counter = field(default_factory=Counter)


def _render(nodes, out: Cleaned) -> str:
    parts: list[str] = []
    for node in nodes:
        if isinstance(node, Template):
            out.removed_templates[str(node.name).strip()] += 1
        elif isinstance(node, Heading):
            title = _render(node.title.nodes, out).strip()
            if out.heading is None:
                out.heading = title  # 첫 제목줄은 장 제목이다. 본문에서 뺀다.
            else:
                parts.append(f"\n{title}\n")
        elif isinstance(node, Comment):
            continue
        elif isinstance(node, Tag):
            tag = str(node.tag).lower()
            if tag in DROP_TAGS:
                continue
            if tag == "br":
                parts.append("\n")
            elif node.contents is not None:
                parts.append(_render(node.contents.nodes, out))
        elif isinstance(node, Wikilink):
            target = str(node.title).strip()
            if target.startswith(SKIP_LINK_PREFIXES):
                continue
            parts.append(_render(node.text.nodes, out) if node.text else target)
        elif isinstance(node, ExternalLink):
            parts.append(_render(node.title.nodes, out) if node.title else "")
        elif isinstance(node, HTMLEntity):
            parts.append(html.unescape(str(node)))
        elif isinstance(node, Text):
            parts.append(_MAGIC.sub("", str(node)))
        else:
            parts.append(str(node))
    return "".join(parts)


def normalize_paragraphs(text: str) -> str:
    """줄 끝 공백을 지우고, 빈 줄이 여러 개면 하나로 줄인다. 문단 사이는 빈 줄 하나다."""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def clean_wikitext(raw: str) -> Cleaned:
    out = Cleaned(text="", heading=None)
    body = _render(mwparserfromhell.parse(raw).nodes, out)
    out.text = normalize_paragraphs(body)
    residue = _RESIDUE.findall(out.text)
    if residue:
        raise CleanError(f"마크업이 남았다: {sorted(set(residue))[:5]}")
    return out


def clean_all() -> dict:
    sources = json.loads(SOURCES.read_text(encoding="utf-8"))
    toc = {p["chapter"]: p["toc_title"] for p in sources["chapters"]}
    CLEAN.mkdir(parents=True, exist_ok=True)
    report = {"chapters": []}
    for c in CHAPTERS:
        raw = raw_path(c).read_text(encoding="utf-8")
        cleaned = clean_wikitext(raw)
        clean_path(c).write_text(cleaned.text, encoding="utf-8")
        paragraphs = [p for p in cleaned.text.split("\n\n") if p.strip()]
        report["chapters"].append({
            "chapter": c,
            "toc_title": toc[c],
            "heading": cleaned.heading,
            "heading_matches_toc": (cleaned.heading or "").replace(" ", "") == toc[c].replace(" ", ""),
            "chars": len(cleaned.text),
            "paragraphs": len(paragraphs),
            "removed_templates": dict(cleaned.removed_templates),
            "hanja_parens": len(re.findall(r"\([一-鿿]+\)", cleaned.text)),
            "censor_marks": cleaned.text.count("×"),
            "first": cleaned.text[:40],
            "last": paragraphs[-1][-80:] if paragraphs else "",
        })
    last = mwparserfromhell.parse(raw_path(19).read_text(encoding="utf-8"))
    footer = next((t for t in last.filter_templates() if str(t.name).strip() == "꼬리말"), None)
    next_link = str(footer.get("다음").value).strip() if footer and footer.has("다음") else None
    report["completeness"] = {
        "last_chapter_next_link": next_link,
        "last_chapter_has_next": bool(next_link),
        "note": "제19장 꼬리말의 '다음'이 비어 있으면 위키문헌상 마지막 장이다. 결말 문단은 사람이 한 번 읽어 확인한다.",
    }
    CLEAN_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
