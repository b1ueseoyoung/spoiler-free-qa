import json

import pytest

from sfqa.clean import CleanError, clean_wikitext, normalize_paragraphs
from sfqa.fetch import sha256, toc_titles
from sfqa.paths import CHAPTERS, CLEAN_REPORT, SOURCES, clean_path, raw_path

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
    assert toc_titles(main) == {1: "인간기념물", 19: "서곡(序曲)"}


# 아래는 실제로 받은 데이터를 본다. 아직 받지 않았으면 건너뛴다.
needs_data = pytest.mark.skipif(not CLEAN_REPORT.exists(), reason="sfqa fetch && sfqa clean을 먼저 실행")


@needs_data
def test_sources_pin_every_chapter_and_hashes_match():
    sources = json.loads(SOURCES.read_text(encoding="utf-8"))
    assert [p["chapter"] for p in sources["chapters"]] == list(CHAPTERS)
    for p in sources["chapters"]:
        assert sha256(raw_path(p["chapter"]).read_text(encoding="utf-8")) == p["sha256"], p["chapter"]


@needs_data
def test_cleaned_chapters_have_no_markup():
    for c in CHAPTERS:
        text = clean_path(c).read_text(encoding="utf-8")
        assert len(text) > 3000, c
        assert clean_wikitext(text).text == text, c  # 정제한 글을 다시 정제해도 그대로다


@needs_data
def test_report_headings_match_toc_and_last_chapter_is_last():
    report = json.loads(CLEAN_REPORT.read_text(encoding="utf-8"))
    assert all(ch["heading_matches_toc"] for ch in report["chapters"])
    assert report["completeness"]["last_chapter_has_next"] is False
