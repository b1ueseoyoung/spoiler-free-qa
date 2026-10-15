"""wikitext → 본문 텍스트(작품마다 data/works/<id>/clean).

틀은 허용 목록에 있는 것만 지운다(머리말·꼬리말 등). 모르는 틀이 나오면 실패로 본다. 본문을 감싼 틀(글자 크기 등)을
아무 말 없이 지우면 본문이 사라져도 모르기 때문이다. 머리말 틀의 notes(줄거리 요약)도 틀째 지운다.
요약이 본문에 섞이면 뒤 장 스포일러가 된다(「무정」).
각주는 지우고, 링크는 보이는 글자만 남긴다. 장 제목(첫 제목줄)은 본문에서 빼 따로 둔다.
한자 병기와 검열 표시(×)는 원문 그대로 둔다(인용 검증의 기준이 원문이다). 마크업이 하나라도 남으면 실패로 본다.
묶음 페이지(bundles)는 source.split 표시로 장을 나눈다.
"""

from __future__ import annotations

import html
import json
import re
from collections import Counter
from dataclasses import dataclass, field

import mwparserfromhell
from mwparserfromhell.nodes import Comment, ExternalLink, Heading, HTMLEntity, Tag, Template, Text, Wikilink

from sfqa.works import Work

DROP_TAGS = {"ref", "references"}
# 지워도 되는 틀(장 이동·출전 표시). 작품마다 source.drop_templates로 더할 수 있다.
DROP_TEMPLATES = {"머리말", "꼬리말", "문단 그림"}
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
    unknown_templates: Counter = field(default_factory=Counter)
    drop: frozenset = frozenset(DROP_TEMPLATES)


def _render(nodes, out: Cleaned) -> str:
    parts: list[str] = []
    for node in nodes:
        if isinstance(node, Template):
            name = str(node.name).strip()
            if name in out.drop or name.startswith("PD-"):
                out.removed_templates[name] += 1
            else:
                out.unknown_templates[name] += 1
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
            elif tag == "li" and node.wiki_markup == "*" and not str(node.contents or "").strip():
                # 내용 없는 '*' 한 줄은 목록이 아니라 장면 구분 표시다(「상록수」). 지우면 다시 정제할 때 결과가 달라진다.
                # 다른 줄 머리 표시(':' 들여쓰기, ';', 내용 있는 '*' 목록)는 예전처럼 표시를 버리고 글만 남긴다.
                # ':'를 남기면 여러 줄에 걸친 인용이 검증에서 떨어진다(리뷰에서 확인, 2026-10-14).
                parts.append("*")
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


def clean_wikitext(raw: str, drop_templates: set[str] | frozenset = DROP_TEMPLATES) -> Cleaned:
    out = Cleaned(text="", heading=None, drop=frozenset(drop_templates))
    body = _render(mwparserfromhell.parse(raw).nodes, out)
    if out.unknown_templates:
        raise CleanError(f"허용 목록에 없는 틀이 있다(본문을 감쌌을 수 있다): {dict(out.unknown_templates)}")
    out.text = normalize_paragraphs(body)
    residue = _RESIDUE.findall(out.text)
    if residue:
        raise CleanError(f"마크업이 남았다: {sorted(set(residue))[:5]}")
    return out


def apply_fixups(work: Work, chapter: int, raw: str) -> str:
    """원문의 깨진 마크업을 정제 전에 바로잡는다(source.fixups). 받은 원문(raw/)은 고치지 않는다.
    바로잡을 글이 원문에 정확히 한 번 있어야 한다. 판이 바뀌어 사라지면 실패해서 알 수 있게 한다."""
    for f in work.source.get("fixups", []):
        if f["chapter"] != chapter:
            continue
        if raw.count(f["find"]) != 1:
            raise CleanError(f"{work.id} 제{chapter}장: 바로잡을 글이 {raw.count(f['find'])}번 있다: {f['find']!r}")
        raw = raw.replace(f["find"], f["replace"])
    return raw


def chapter_wikitexts(work: Work) -> dict[int, str]:
    """장 번호 → 그 장의 wikitext. bundles는 묶음 페이지를 source.split 표시로 나눈다."""
    src = work.source
    if src.get("mode", "subpages") == "subpages":
        return {c: apply_fixups(work, c, work.raw_path(c).read_text(encoding="utf-8")) for c in work.chapter_numbers}
    sources = json.loads(work.sources_path.read_text(encoding="utf-8"))
    split = re.compile(src["split"])
    out: dict[int, str] = {}
    for part in sources["parts"]:
        raw = (work.raw_dir / f"part{part['part']:02d}.wiki").read_text(encoding="utf-8")
        pieces = split.split(raw)
        head, chapters = pieces[0], pieces[1:]
        if clean_wikitext(head, work_drop(work)).text.strip():  # 첫 표시 앞에는 머리말 틀만 있어야 한다
            raise CleanError(f"{part['page']}: 첫 장 표시 앞에 본문이 있다")
        expect = part["last"] - part["first"] + 1
        if len(chapters) != expect:
            raise CleanError(f"{part['page']}: 장 표시가 {len(chapters)}개다({expect}개여야 한다)")
        for i, text in enumerate(chapters):
            out[part["first"] + i] = apply_fixups(work, part["first"] + i, text)
    if sorted(out) != list(work.chapter_numbers):
        raise CleanError(f"장 번호가 1~{work.last}가 아니다")
    return out


def work_drop(work: Work) -> set[str]:
    return DROP_TEMPLATES | set(work.source.get("drop_templates", []))


def clean_all(work: Work) -> dict:
    sources = json.loads(work.sources_path.read_text(encoding="utf-8"))
    toc = {p["chapter"]: p.get("toc_title", "") for p in sources["chapters"]}
    work.clean_dir.mkdir(parents=True, exist_ok=True)
    report = {"work": work.id, "chapters": []}
    for c, raw in chapter_wikitexts(work).items():
        try:
            cleaned = clean_wikitext(raw, work_drop(work))
        except CleanError as e:
            raise CleanError(f"{work.id} 제{c}장: {e}") from None
        work.clean_path(c).write_text(cleaned.text, encoding="utf-8")
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
    # 마지막 장 머리말·꼬리말 틀의 '다음'으로 완결을 본다. 그런 틀이 없으면(묶음 페이지 등) '확인 못 함'(None)으로 둔다.
    next_link, seen_in = None, None
    if work.source.get("mode", "subpages") == "subpages":
        last = mwparserfromhell.parse(work.raw_path(work.last).read_text(encoding="utf-8"))
        for t in last.filter_templates():
            if str(t.name).strip() in ("꼬리말", "머리말") and t.has("다음"):
                next_link, seen_in = str(t.get("다음").value).strip(), str(t.name).strip()
                if next_link:
                    break
    report["completeness"] = {
        "last_chapter_next_link": next_link,
        "last_chapter_has_next": None if seen_in is None else bool(next_link),
        "checked_template": seen_in,
        "note": (f"제{work.last}장 {seen_in} 틀의 '다음'이 비어 있으면 위키문헌상 마지막 장이다." if seen_in
                 else "'다음'이 든 틀이 없어 완결 여부를 자동으로 확인하지 못했다.") + " 결말 문단은 사람이 한 번 읽어 확인한다.",
    }
    work.clean_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
