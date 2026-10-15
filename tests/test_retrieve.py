import json

import numpy as np
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from sfqa import index as index_mod
from sfqa.chunk import chunk_corpus
from sfqa.gold import Evidence
from sfqa.retrieval_eval import content_chapter, covers
from sfqa.retrieve import ChapterIndex, UptoRetriever

TEXTS = {ch: " ".join(f"제{ch}장의 {i}번째 문장이다." for i in range(120)) for ch in range(1, 20)}
EMB = DeterministicFakeEmbedding(size=16)


@pytest.fixture(scope="module")
def idx() -> ChapterIndex:
    chunks = chunk_corpus(TEXTS)
    vectors = index_mod.normalize_rows(np.asarray(EMB.embed_documents([c.text for c in chunks])))
    return ChapterIndex(chunks, vectors)


@pytest.mark.parametrize("n", range(1, 20))
def test_no_chunk_after_n_ever(idx, n):
    q = np.asarray(EMB.embed_query("질문"))
    hits = idx.search(q, n, k=1000)  # k를 크게 해도 N장 뒤는 나오지 않는다
    assert hits and max(c.chapter for c, _ in hits) <= n
    assert len(hits) == idx.prefix_len(n)


@pytest.mark.parametrize("k", [5, 12])
def test_boundary_chapter_is_included_at_n_and_excluded_before(idx, k):
    target = next(c for c in idx.chunks if c.chapter == k)
    q = idx.vectors[idx.chunks.index(target)]  # 그 청크와 똑같은 벡터로 찾는다
    assert idx.search(q, k, 1)[0][0] == target
    assert all(c.chapter < k for c, _ in idx.search(q, k - 1, 1000))


def test_unsorted_chunks_rejected(idx):
    with pytest.raises(ValueError):
        ChapterIndex(list(reversed(idx.chunks)), idx.vectors[::-1])


def test_retriever_returns_documents_with_chapter(idx):
    docs = UptoRetriever(index=idx, embeddings=EMB, n=4, k=6).invoke("제3장 이야기")
    assert len(docs) == 6 and all(d.metadata["chapter"] <= 4 for d in docs)
    assert all(d.page_content == TEXTS[d.metadata["chapter"]][d.metadata["start"]:d.metadata["end"]] for d in docs)


def test_covers_and_content_chapter(idx):
    c = next(c for c in idx.chunks if c.chapter == 7)
    quote = c.text[10:60]
    assert covers(c, Evidence(chapter=7, quote=quote), TEXTS)
    assert not covers(c, Evidence(chapter=8, quote=quote), TEXTS)
    assert content_chapter(c.text, TEXTS) == 7


def test_index_refuses_other_model_or_text(tmp_path):
    from sfqa.works import Work

    work = Work(id="w", title="w", author="a", label="w", chapters=2, dir=tmp_path / "w")
    other = Work(id="v", title="v", author="a", label="v", chapters=2, dir=tmp_path / "v")
    texts = {1: TEXTS[1], 2: TEXTS[2]}
    meta = index_mod.build_index(work, "org/model-a", EMB, texts)
    assert json.loads((tmp_path / "w" / "index" / "org__model-a" / "meta.json").read_text())["n_chunks"] == meta["n_chunks"]
    chunks, vectors, _ = index_mod.load_index(work, "org/model-a", texts)
    assert len(chunks) == vectors.shape[0]
    with pytest.raises(index_mod.IndexMismatch):
        index_mod.load_index(work, "org/model-b", texts)  # 다른 모델의 색인은 따로 있어야 한다
    with pytest.raises(index_mod.IndexMismatch):
        index_mod.load_index(work, "org/model-a", {1: TEXTS[1], 2: TEXTS[2] + " 고친 문장."})
    with pytest.raises(index_mod.IndexMismatch):
        index_mod.load_index(other, "org/model-a", texts)  # 작품마다 색인이 따로 있다


from sfqa.bm25 import PrefixBM25, tokenize_many  # noqa: E402
from sfqa.retrieve import rrf  # noqa: E402


@pytest.mark.parametrize("method", ["bm25", "hybrid"])
@pytest.mark.parametrize("n", [1, 7, 19])
def test_bm25_and_hybrid_never_return_after_n(idx, method, n):
    q = np.asarray(EMB.embed_query("제19장의 3번째 문장이다."))
    terms = tokenize_many(["제19장의 3번째 문장이다."])[0]
    hits = idx.search(q, n, 50, method, terms)
    assert hits and all(c.chapter <= n for c, _ in hits)


def test_bm25_idf_uses_only_docs_up_to_n():
    docs = [["초봉", "약국"], ["초봉", "서울"], ["형보", "서울"], ["서울", "서울"]]
    bm = PrefixBM25(docs)
    # '서울'은 앞 2개 문서에서 1번, 4개 문서 전체에서는 3번 나온다. 앞 2개만 볼 때 IDF가 더 커야 한다.
    two = bm.scores(["서울"], 2)
    four = bm.scores(["서울"], 4)
    assert len(two) == 2 and two[1] > 0 and two[1] > four[1]
    assert bm.scores(["형보"], 2).sum() == 0  # 3번째 문서에만 있는 낱말은 N 이하에서 점수가 없다


def test_bm25_finds_name_match():
    docs = tokenize_many(["초봉이는 제중당 약국에 다닌다.", "형보가 서울로 올라간다.", "승재가 개복동에 왕진을 간다."])
    bm = PrefixBM25(docs)
    assert int(np.argmax(bm.scores(tokenize_many(["초봉이가 일하는 약국은?"])[0], 3))) == 0
    assert int(np.argmax(bm.scores(tokenize_many(["형보는 어디로 가나?"])[0], 3))) == 1


def test_rrf_prefers_agreement_and_breaks_ties_stably():
    out = rrf([[1, 2, 3], [2, 1, 3]], k=3)
    assert [i for i, _ in out] == [1, 2, 3]  # 1과 2가 동점이면 앞 목록 순위가 높은 1
    assert rrf([[5], [5]], k=1)[0][0] == 5


def test_gate_k_can_be_raised():
    from sfqa.index import normalize_rows
    from sfqa.retrieval_eval import evaluate

    chunks = chunk_corpus(TEXTS)
    idx = ChapterIndex(chunks, normalize_rows(np.asarray(EMB.embed_documents([c.text for c in chunks]))))
    rep = evaluate([], None, None, idx, TEXTS, "dense", gate_k=20)
    assert rep["gate"]["k"] == 20 and "recall@20" in rep["summary"]["all"] and "recall@18" in rep["summary"]["all"]
