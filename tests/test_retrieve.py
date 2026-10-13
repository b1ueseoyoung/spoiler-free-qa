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


def test_index_refuses_other_model_or_text(tmp_path, monkeypatch):
    for name in ("INDEX_DIR", "VECTORS", "CHUNKS", "META"):
        monkeypatch.setattr(index_mod, name, tmp_path / getattr(index_mod, name).name)
    texts = {1: TEXTS[1], 2: TEXTS[2]}
    meta = index_mod.build_index("model-a", EMB, texts)
    assert json.loads((tmp_path / "meta.json").read_text())["n_chunks"] == meta["n_chunks"]
    chunks, vectors, _ = index_mod.load_index("model-a", texts)
    assert len(chunks) == vectors.shape[0]
    with pytest.raises(index_mod.IndexMismatch):
        index_mod.load_index("model-b", texts)
    with pytest.raises(index_mod.IndexMismatch):
        index_mod.load_index("model-a", {1: TEXTS[1], 2: TEXTS[2] + " 고친 문장."})
