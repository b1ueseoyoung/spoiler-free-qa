import json

import pytest

from sfqa.clean import CleanError, clean_wikitext, normalize_paragraphs
from sfqa.fetch import sha256, toc_titles
from sfqa.clean import chapter_wikitexts
from sfqa.works import gold_work, list_works

CHAPTER = """__NOTOC__
{{머리말
|제목 = [[../]]
|지은이 = [[저자:채만식|채만식]]
|부제 = 제1장: 인간기념물
|설명 = 줄거리: 초봉이는 나중에 ○○을 한다
}}
==인간기념물 ==

금강(錦江)…….

고서방이란 이 집에 하숙을 하고 있는 ××은행의 [[태수|고태수]] 말이다.<ref>각주 내용</ref><!-- 주석 -->
줄바꿈&nbsp;있음<br />다음 줄


[[분류:장편소설]]
{{꼬리말
|이전 =
|다음 = [[../제2장|제2장]]
}}"""


def test_templates_and_their_plot_summary_are_removed():
    out = clean_wikitext(CHAPTER)
    assert "줄거리" not in out.text and "○○" not in out.text
    assert out.removed_templates == {"머리말": 1, "꼬리말": 1}


def test_heading_is_taken_out_of_body():
    out = clean_wikitext(CHAPTER)
    assert out.heading == "인간기념물"
    assert "인간기념물" not in out.text


def test_links_keep_display_text_and_footnotes_vanish():
    out = clean_wikitext(CHAPTER)
    assert "고태수 말이다." in out.text
    assert "각주" not in out.text and "주석" not in out.text and "분류" not in out.text
    assert "줄바꿈 있음\n다음 줄" in out.text


def test_hanja_and_censor_marks_are_kept():
    out = clean_wikitext(CHAPTER)
    assert "금강(錦江)……." in out.text and "××은행" in out.text


def test_no_markup_left_and_paragraphs_normalized():
    out = clean_wikitext(CHAPTER)
    assert "__NOTOC__" not in out.text and "{{" not in out.text and "[[" not in out.text
    assert "\n\n\n" not in out.text and out.text.endswith("\n") and not out.text.startswith("\n")


def test_residue_fails_loudly():
    with pytest.raises(CleanError):
        clean_wikitext("본문 <span>남은 태그")  # 닫히지 않아 파서가 텍스트로 남기는 경우


def test_normalize_paragraphs():
    assert normalize_paragraphs("a  \n\n\n\nb\r\n") == "a\n\nb\n"


def test_toc_titles():
    main = "== 목차 ==\n# [[/제1장/|인간기념물]]\n# [[/제19장/|서곡(序曲)]]\n"
    assert toc_titles(main, gold_work().source["toc"]) == {1: "인간기념물", 19: "서곡(序曲)"}


# 아래는 실제로 받은 데이터를 본다. 아직 받지 않았으면 건너뛴다.
READY = list_works(ready_only=True)
needs_data = pytest.mark.skipif(not READY, reason="sfqa fetch && sfqa clean을 먼저 실행")
per_work = pytest.mark.parametrize("work", READY, ids=[w.id for w in READY])


@needs_data
@per_work
def test_sources_pin_every_page_and_hashes_match(work):
    sources = json.loads(work.sources_path.read_text(encoding="utf-8"))
    assert [p["chapter"] for p in sources["chapters"]] == list(work.chapter_numbers)
    if "parts" in sources:  # 묶음 페이지
        for p in sources["parts"]:
            assert sha256((work.raw_dir / f"part{p['part']:02d}.wiki").read_text(encoding="utf-8")) == p["sha256"]
    else:
        for p in sources["chapters"]:
            assert sha256(work.raw_path(p["chapter"]).read_text(encoding="utf-8")) == p["sha256"], p["chapter"]
    assert sources["work"]["license_templates"], "공유 저작물 틀(PD-)이 있어야 한다"


@needs_data
@per_work
def test_cleaned_chapters_have_no_markup(work):
    for c in work.chapter_numbers:
        text = work.clean_path(c).read_text(encoding="utf-8")
        assert len(text) > 1000, c
        assert clean_wikitext(text).text == text, c  # 정제한 글을 다시 정제해도 그대로다


@needs_data
@per_work
def test_header_synopsis_never_reaches_chapter_text(work):
    """머리말 틀의 notes(줄거리 요약)가 본문에 섞이면 뒤 장 스포일러가 된다(「무정」)."""
    import mwparserfromhell

    raws = chapter_wikitexts(work) if work.source.get("mode") == "bundles" else {}
    pages = [p.read_text(encoding="utf-8") for p in work.raw_dir.glob("*.wiki")]
    notes = []
    for raw in pages:
        for t in mwparserfromhell.parse(raw).filter_templates():
            for key in ("notes", "설명"):
                if t.has(key) and len(str(t.get(key).value).strip()) > 20:
                    notes.append(str(t.get(key).value).strip())
    body = "".join(work.clean_path(c).read_text(encoding="utf-8") for c in work.chapter_numbers)
    for note in notes:
        assert note[:30] not in body
    assert raws is not None


def test_gold_work_headings_match_toc_and_last_chapter_is_last():
    work = gold_work()
    if not work.ready:
        pytest.skip("탁류 정제 본문이 없다")
    report = json.loads(work.clean_report_path.read_text(encoding="utf-8"))
    assert all(ch["heading_matches_toc"] for ch in report["chapters"])
    assert report["completeness"]["last_chapter_has_next"] is False
