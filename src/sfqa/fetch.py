"""위키문헌에서 「탁류」 원문(wikitext)을 받는다.

판(revid)을 sources.json에 고정해 두고, 다음부터는 같은 판을 받는다(--update면 최신 판으로 바꾼다).
위키미디어 정책에 따라 User-Agent를 붙이고 초당 1회만 요청한다.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone

import httpx

from sfqa.config import Settings
from sfqa.paths import CHAPTERS, RAW, SOURCES, raw_path

API = "https://ko.wikisource.org/w/api.php"
INDEX = "https://ko.wikisource.org/w/index.php"
WORK = "탁류"
_TOC = re.compile(r"^#\s*\[\[/제(\d+)장/\|(.+?)\]\]\s*$", re.MULTILINE)


def page_title(chapter: int) -> str:
    return f"{WORK}/제{chapter}장"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def toc_titles(main_wikitext: str) -> dict[int, str]:
    """작품 첫 페이지 목차에서 장 제목을 읽는다."""
    return {int(n): title.strip() for n, title in _TOC.findall(main_wikitext)}


class Wikisource:
    def __init__(self, user_agent: str, min_interval_s: float = 1.0):
        self._http = httpx.Client(headers={"User-Agent": user_agent}, timeout=30, follow_redirects=True)
        self._min = min_interval_s
        self._last = 0.0

    def _get(self, url: str, params: dict) -> httpx.Response:
        wait = self._min - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        resp = self._http.get(url, params=params)
        self._last = time.monotonic()
        resp.raise_for_status()
        return resp

    def latest_revisions(self, titles: list[str]) -> dict[str, dict]:
        data = self._get(API, {
            "action": "query", "prop": "revisions", "rvprop": "ids|timestamp",
            "titles": "|".join(titles), "format": "json", "formatversion": "2",
        }).json()
        out = {}
        for page in data["query"]["pages"]:
            if page.get("missing"):
                raise RuntimeError(f"위키문헌에 페이지가 없다: {page['title']}")
            rev = page["revisions"][0]
            out[page["title"]] = {"revid": rev["revid"], "timestamp": rev["timestamp"]}
        return out

    def raw(self, revid: int) -> str:
        return self._get(INDEX, {"oldid": revid, "action": "raw"}).text


def fetch(s: Settings, update: bool = False) -> dict:
    """원문을 받아 data/raw에 쓰고 sources.json을 돌려준다. 고정된 판과 해시가 같은 파일은 다시 받지 않는다."""
    old = json.loads(SOURCES.read_text(encoding="utf-8")) if SOURCES.exists() else None
    wiki = Wikisource(s.wiki_user_agent)
    titles = [WORK] + [page_title(c) for c in CHAPTERS]
    if old and not update:
        revs = {p["page"]: {"revid": p["revid"], "timestamp": p["timestamp"]} for p in [old["work"], *old["chapters"]]}
    else:
        revs = wiki.latest_revisions(titles)

    main = wiki.raw(revs[WORK]["revid"])
    titles_by_ch = toc_titles(main)
    if sorted(titles_by_ch) != list(CHAPTERS):
        raise RuntimeError(f"목차의 장 번호가 1~19가 아니다: {sorted(titles_by_ch)}")

    RAW.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    chapters = []
    for c in CHAPTERS:
        page = page_title(c)
        rev = revs[page]
        path = raw_path(c)
        pinned = next((p for p in (old or {}).get("chapters", []) if p["chapter"] == c), None)
        if pinned and pinned["revid"] == rev["revid"] and path.exists() and sha256(path.read_text(encoding="utf-8")) == pinned["sha256"]:
            chapters.append(pinned)
            continue
        text = wiki.raw(rev["revid"])
        path.write_text(text, encoding="utf-8")
        chapters.append({
            "chapter": c, "page": page, "toc_title": titles_by_ch[c], "revid": rev["revid"],
            "timestamp": rev["timestamp"], "sha256": sha256(text), "chars": len(text), "fetched_at": now,
            "url": f"{INDEX}?oldid={rev['revid']}",
        })
        print(f"제{c}장 받음: revid {rev['revid']}, {len(text)}자")

    license_templates = re.findall(r"\{\{(PD-[^}|]+)", main)
    sources = {
        "work": {"page": WORK, "revid": revs[WORK]["revid"], "timestamp": revs[WORK]["timestamp"],
                 "sha256": sha256(main), "license_templates": license_templates},
        "chapters": chapters,
        "note": "원문은 공유 저작물(작가 채만식 1950년 사망). 위키문헌 편집자가 붙인 틀은 정제 단계에서 지운다.",
    }
    SOURCES.write_text(json.dumps(sources, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return sources
