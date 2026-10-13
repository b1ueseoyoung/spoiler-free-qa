import pytest

from sfqa.chunk import MIN_OVERLAP, SIZE, SNAP, chunk_chapter, chunk_corpus
from sfqa.gold import QUESTIONS, load_questions, load_texts
from sfqa.paths import CLEAN_REPORT
from sfqa.textnorm import search_text


def test_search_text_drops_hanja_parens_and_censor_marks():
    assert search_text("금강(錦江)…… 갈재〔蘆嶺〕와 ××은행") == "금강…… 갈재와 은행"
    assert search_text("그는 (웃으며) 말했다") == "그는 (웃으며) 말했다"  # 한글 괄호는 남긴다
    assert search_text("a \t  b") == "a b"


def make_text(n: int) -> str:
    words = ["초봉이는", "제중당에서", "일한다.", "정주사는", "미두장에", "간다.", "승재는", "왕진을", "다닌다."]
    return " ".join(words[i % len(words)] for i in range(n))


@pytest.mark.parametrize("length_words", [1, 50, 200, 2000])
def test_chunks_are_exact_slices_and_cover_the_chapter(length_words):
    text = make_text(length_words)
    chunks = chunk_chapter(3, text)
    assert chunks[0].start == 0 and chunks[-1].end == len(text)
    for c in chunks:
        assert c.text == text[c.start:c.end] and c.chapter == 3
        assert len(c.text) <= SIZE + SNAP
    for a, b in zip(chunks, chunks[1:]):
        assert a.end - b.start >= MIN_OVERLAP


def test_any_short_span_fits_in_one_chunk():
    text = make_text(2000)
    chunks = chunk_chapter(1, text)
    for start in range(0, len(text) - 150, 37):
        assert any(c.start <= start and start + 150 <= c.end for c in chunks), start


def test_corpus_is_sorted_and_never_spans_chapters():
    texts = {2: make_text(300), 1: make_text(300), 3: "짧다."}
    chunks = chunk_corpus(texts)
    assert [c.chapter for c in chunks] == sorted(c.chapter for c in chunks)
    assert all(c.text in texts[c.chapter] for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


needs_data = pytest.mark.skipif(not (CLEAN_REPORT.exists() and QUESTIONS.exists()), reason="정제 본문과 정답지가 필요하다")


@needs_data
def test_real_chunks_are_exact_and_every_gold_quote_fits():
    texts = load_texts()
    chunks = chunk_corpus(texts)
    for c in chunks:
        assert c.text == texts[c.chapter][c.start:c.end]
    by_ch = {}
    for c in chunks:
        by_ch.setdefault(c.chapter, []).append(c)
    for q in load_questions():
        for ev in q.evidence:
            pos = texts[ev.chapter].find(ev.quote)
            assert any(c.start <= pos and pos + len(ev.quote) <= c.end for c in by_ch[ev.chapter]), (q.id, ev.chapter)
