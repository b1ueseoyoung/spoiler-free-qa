"""N장 이하에서만 찾는 검색: dense(코사인), bm25(Kiwi 형태소), hybrid(둘을 RRF로 합침).

청크가 (장, 위치) 순서라 N장 이하 청크는 앞쪽 연속 구간이다. N마다 그 구간만 잘라 찾는다.
검색 뒤에 거르는 필터가 아니므로, N이 작아도 k개를 채우고 필터 인자를 빠뜨려 새는 일이 없다.
결과마다 장 ≤ N을 한 번 더 확인한다(2차 방어선).
"""

from __future__ import annotations

from bisect import bisect_right

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from sfqa.bm25 import PrefixBM25, tokenize_many
from sfqa.chunk import Chunk
from sfqa.textnorm import search_text

METHODS = ("dense", "bm25", "hybrid")
RRF_K = 60
RRF_DEPTH = 100  # RRF에 넣을 각 검색기의 상위 순위 수


def ranked(scores: np.ndarray) -> list[int]:
    """점수 내림차순, 같으면 앞 청크 먼저."""
    return [int(i) for i in np.lexsort((np.arange(len(scores)), -scores))]


def rrf(rankings: list[list[int]], k: int, rrf_k: int = RRF_K) -> list[tuple[int, float]]:
    """여러 순위 목록을 RRF로 합친다. 같으면 앞 목록의 순위, 그다음 청크 번호 순으로 정한다."""
    fused: dict[int, float] = {}
    ranks: dict[int, list[float]] = {}
    for j, ranking in enumerate(rankings):
        for rank, i in enumerate(ranking[:RRF_DEPTH], start=1):
            fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + rank)
            ranks.setdefault(i, [float("inf")] * len(rankings))[j] = rank
    best = sorted(fused, key=lambda i: (-fused[i], *ranks[i], i))[:k]
    return [(i, fused[i]) for i in best]


class ChapterIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, bm25: PrefixBM25 | None = None):
        order = [(c.chapter, c.start) for c in chunks]
        if order != sorted(order):
            raise ValueError("청크는 (장, 위치) 순서여야 한다")
        self.chunks = chunks
        self.vectors = vectors
        self._chapters = [c.chapter for c in chunks]
        self._bm25 = bm25

    @property
    def bm25(self) -> PrefixBM25:
        if self._bm25 is None:
            self._bm25 = PrefixBM25(tokenize_many([c.text for c in self.chunks]))
        return self._bm25

    def prefix_len(self, n: int) -> int:
        """N장 이하 청크의 개수(= 앞쪽 구간 길이)."""
        return bisect_right(self._chapters, n)

    def search(self, query_vec: np.ndarray | None, n: int, k: int, method: str = "dense",
               query_terms: list[str] | None = None) -> list[tuple[Chunk, float]]:
        upto = self.prefix_len(n)
        if upto == 0:
            return []
        rankings, scores = [], None
        if method in ("bm25", "hybrid"):
            assert query_terms is not None, "bm25에는 질의 형태소가 필요하다"
            scores = self.bm25.scores(query_terms, upto)
            rankings.append(ranked(scores))
        if method in ("dense", "hybrid"):
            assert query_vec is not None, "dense에는 질의 벡터가 필요하다"
            q = query_vec / (np.linalg.norm(query_vec) or 1.0)
            scores = self.vectors[:upto] @ q.astype(np.float32)
            rankings.append(ranked(scores))
        if method not in METHODS:
            raise ValueError(f"없는 검색 방식이다: {method}")
        top = rrf(rankings, k) if method == "hybrid" else [(i, float(scores[i])) for i in rankings[0][:k]]
        hits = [(self.chunks[i], s) for i, s in top]
        assert all(c.chapter <= n for c, _ in hits), "N장 뒤 청크가 검색됐다"
        return hits


class UptoRetriever(BaseRetriever):
    """LangChain 검색기. n장 이하에서 k개를 찾는다."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    index: ChapterIndex
    embeddings: Embeddings
    n: int
    k: int = 18  # 검색 관문 기준과 같다(docs/eval-spec.md)
    method: str = "dense"

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        qvec = np.asarray(self.embeddings.embed_query(search_text(query))) if self.method != "bm25" else None
        terms = tokenize_many([query])[0] if self.method != "dense" else None
        return [
            Document(page_content=c.text, metadata={"chapter": c.chapter, "chunk_id": c.chunk_id,
                                                    "start": c.start, "end": c.end, "score": score})
            for c, score in self.index.search(qvec, self.n, self.k, self.method, terms)
        ]
