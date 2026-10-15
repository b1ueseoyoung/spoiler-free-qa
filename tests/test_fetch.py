"""수집기: 위키문헌 없이 가짜 클라이언트로 두 번 받기(판 고정, 묶음 페이지)를 확인한다."""

import json

import pytest

from sfqa import fetch as fetch_mod
from sfqa.config import load_settings
from sfqa.works import Work

MAIN = "{{PD-old-50}}\n* [[책/1장~2장|1장~2장]]\n* [[책/3장~3장|3장~3장]]\n"
PAGES = {"책/1장~2장": "{{머리말|notes=요약}}\n{{문단 그림}}\n첫 장\n{{문단 그림}}\n둘째 장\n",
         "책/3장~3장": "{{머리말}}\n{{문단 그림}}\n셋째 장\n"}


class FakeWiki:
    calls: list = []

    def __init__(self, user_agent):
        self.revs = {"책": 1, **{p: 10 + i for i, p in enumerate(PAGES)}}

    def latest_revisions(self, titles):
        FakeWiki.calls.append(("latest", tuple(titles)))
        return {t: {"revid": self.revs[t], "timestamp": "2026-10-14T00:00:00Z"} for t in titles}

    def raw(self, revid):
        FakeWiki.calls.append(("raw", revid))
        page = {v: k for k, v in self.revs.items()}[revid]
        return MAIN if page == "책" else PAGES[page]


@pytest.fixture
def bundle_work(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_mod, "Wikisource", FakeWiki)
    FakeWiki.calls = []
    return Work(id="book", title="책", author="작가", label="책", chapters=3, dir=tmp_path / "book",
                author_death_year=1950,
                source={"page": "책", "mode": "bundles", "toc": r"^\*\s*\[\[(책/(\d+)장~(\d+)장)\|",
                        "split": r"\{\{문단 그림\}\}"})


def test_bundles_fetch_twice_reuses_pins(bundle_work):
    s = load_settings({})
    first = fetch_mod.fetch(s, bundle_work)
    assert [p["page"] for p in first["parts"]] == list(PAGES) and len(first["chapters"]) == 3
    FakeWiki.calls = []
    again = fetch_mod.fetch(s, bundle_work)  # 리뷰에서 KeyError('page')가 나던 경로
    assert again["parts"] == first["parts"]
    assert ("latest",) not in [c[:1] for c in FakeWiki.calls]  # 고정된 판을 쓰고 최신 판을 묻지 않는다
    assert [c for c in FakeWiki.calls if c[0] == "raw"] == [("raw", 1)]  # 목차만 다시 받고 묶음은 해시가 같아 건너뛴다
    updated = fetch_mod.fetch(s, bundle_work, update=True)
    assert json.loads(bundle_work.sources_path.read_text())["parts"] == updated["parts"]


def test_bundles_split_into_chapters_and_drop_header_notes(bundle_work):
    from sfqa.clean import clean_all

    fetch_mod.fetch(load_settings({}), bundle_work)
    report = clean_all(bundle_work)
    texts = [bundle_work.clean_path(c).read_text() for c in (1, 2, 3)]
    assert texts == ["첫 장\n", "둘째 장\n", "셋째 장\n"]
    assert "요약" not in "".join(texts)  # 머리말 notes는 본문에 섞이지 않는다
    assert report["completeness"]["last_chapter_has_next"] is None  # 묶음은 자동 확인을 못 한다
