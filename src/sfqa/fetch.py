"""위키문헌에서 작품 원문(wikitext)을 받는다. 작품마다 data/works/<id>/work.json의 source를 따른다.

수집 방식(source.mode):
- subpages: 목차 페이지에 장별 하위 페이지 링크가 있다(「탁류」, 「태평천하」, 「상록수」, 「운현궁의 봄」).
  raw/chNN.wiki에 장마다 하나씩 쓴다.
- bundles: 여러 장을 묶은 페이지가 있다(「무정」: 20장씩 7페이지). raw/partNN.wiki에 묶음째 쓰고,
  장 나누기는 정제 단계(clean.py)에서 source.split 표시로 한다.

판(revid)을 sources.json에 고정해 두고, 다음부터는 같은 판을 받는다(--update면 최신 판으로 바꾼다).
위키미디어 정책에 따라 User-Agent를 붙이고 초당 1회만 요청한다. 공유 저작물 틀(PD-)이 없으면 받지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone

import httpx

from sfqa.config import Settings
from sfqa.works import Work

API = "https://ko.wikisource.org/w/api.php"
INDEX = "https://ko.wikisource.org/w/index.php"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def toc_titles(main_wikitext: str, pattern: str) -> dict[int, str]:
    """목차에서 장 번호와 제목을 읽는다(subpages). pattern은 (장 번호, 제목) 두 그룹."""
    return {int(n): title.strip() for n, title in re.findall(pattern, main_wikitext, re.MULTILINE)}


def toc_parts(main_wikitext: str, pattern: str) -> list[tuple[str, int, int]]:
    """목차에서 묶음 페이지를 읽는다(bundles). pattern은 (페이지 이름, 첫 장, 끝 장) 세 그룹."""
    return [(title.strip(), int(a), int(b)) for title, a, b in re.findall(pattern, main_wikitext, re.MULTILINE)]


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
        out = {}
        for i in range(0, len(titles), 50):  # API는 한 번에 50개까지 받는다
            data = self._get(API, {
                "action": "query", "prop": "revisions", "rvprop": "ids|timestamp",
                "titles": "|".join(titles[i:i + 50]), "format": "json", "formatversion": "2",
            }).json()
            for page in data["query"]["pages"]:
                if page.get("missing"):
                    raise RuntimeError(f"위키문헌에 페이지가 없다: {page['title']}")
                rev = page["revisions"][0]
                out[page["title"]] = {"revid": rev["revid"], "timestamp": rev["timestamp"]}
        return out

    def raw(self, revid: int) -> str:
        return self._get(INDEX, {"oldid": revid, "action": "raw"}).text


def _pinned_revs(old: dict | None) -> dict[str, dict]:
    if not old:
        return {}
    pages = [old["work"], *old.get("chapters", []), *old.get("parts", [])]
    return {p["page"]: {"revid": p["revid"], "timestamp": p["timestamp"]} for p in pages if "revid" in p}


def _download(wiki: Wikisource, path, page: str, rev: dict, pinned: dict | None, now: str, extra: dict) -> dict:
    """고정된 판과 해시가 같은 파일은 다시 받지 않는다."""
    if pinned and pinned.get("revid") == rev["revid"] and path.exists() \
            and sha256(path.read_text(encoding="utf-8")) == pinned.get("sha256"):
        return pinned
    text = wiki.raw(rev["revid"])
    path.write_text(text, encoding="utf-8")
    print(f"{page} 받음: revid {rev['revid']}, {len(text)}자")
    return {**extra, "page": page, "revid": rev["revid"], "timestamp": rev["timestamp"], "sha256": sha256(text),
            "chars": len(text), "fetched_at": now, "url": f"{INDEX}?oldid={rev['revid']}"}


def fetch(s: Settings, work: Work, update: bool = False) -> dict:
    """원문을 받아 work.raw_dir에 쓰고 sources.json을 돌려준다."""
    src = work.source
    mode = src.get("mode", "subpages")
    page = src["page"]
    old = json.loads(work.sources_path.read_text(encoding="utf-8")) if work.sources_path.exists() else None
    wiki = Wikisource(s.wiki_user_agent)
    pinned = {} if update else _pinned_revs(old)
    main_rev = pinned.get(page) or wiki.latest_revisions([page])[page]
    main = wiki.raw(main_rev["revid"])
    license_templates = re.findall(r"\{\{(PD-[^}|]+)", main)
    if not license_templates:
        raise RuntimeError(f"{page} 목차에 공유 저작물 틀(PD-)이 없다. 저작권을 확인한다")

    work.raw_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # 묶음 작품의 chapters 항목에는 page가 없다(장 번호와 제목만). 받은 페이지 기록만 모은다.
    old_by_page = {p["page"]: p for p in [*(old or {}).get("chapters", []), *(old or {}).get("parts", [])] if "page" in p}
    sources: dict = {"work": {"page": page, "revid": main_rev["revid"], "timestamp": main_rev["timestamp"],
                              "sha256": sha256(main), "license_templates": license_templates}}

    if mode == "subpages":
        titles_by_ch = toc_titles(main, src["toc"])
        if sorted(titles_by_ch) != list(work.chapter_numbers):
            raise RuntimeError(f"목차의 장 번호가 1~{work.last}가 아니다: {sorted(titles_by_ch)}")
        pages = {c: src["subpage"].format(page=page, n=c) for c in work.chapter_numbers}
        revs = {p: pinned[p] for p in pages.values() if p in pinned}
        missing = [p for p in pages.values() if p not in revs]
        if missing:
            revs |= wiki.latest_revisions(missing)
        sources["chapters"] = [
            _download(wiki, work.raw_path(c), pages[c], revs[pages[c]], old_by_page.get(pages[c]), now,
                      {"chapter": c, "toc_title": titles_by_ch[c]})
            for c in work.chapter_numbers
        ]
    elif mode == "bundles":
        parts = toc_parts(main, src["toc"])
        covered = [c for _, a, b in parts for c in range(a, b + 1)]
        if covered != list(work.chapter_numbers):
            raise RuntimeError(f"묶음 페이지가 1~{work.last}장을 빈틈없이 덮지 않는다: {parts}")
        revs = {p: pinned[p] for p, _, _ in parts if p in pinned}
        missing = [p for p, _, _ in parts if p not in revs]
        if missing:
            revs |= wiki.latest_revisions(missing)
        sources["parts"] = [
            _download(wiki, work.raw_dir / f"part{i:02d}.wiki", p, revs[p], old_by_page.get(p), now,
                      {"part": i, "first": a, "last": b})
            for i, (p, a, b) in enumerate(parts, 1)
        ]
        sources["chapters"] = [{"chapter": c, "toc_title": ""} for c in work.chapter_numbers]
    else:
        raise ValueError(f"모르는 수집 방식이다: {mode}")

    death = f" {work.author_death_year}년 사망" if work.author_death_year else ""
    sources["note"] = f"원문은 공유 저작물(작가 {work.author}{death}). 위키문헌 편집자가 붙인 틀은 정제 단계에서 지운다."
    work.sources_path.write_text(json.dumps(sources, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return sources
