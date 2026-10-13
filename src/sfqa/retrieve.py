"""N장 이하에서만 찾는 검색.

청크가 (장, 위치) 순서라 N장 이하 청크는 앞쪽 연속 구간이다. N마다 그 구간의 행렬만 잘라 정확한 코사인으로 찾는다.
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

from sfqa.chunk import Chunk
from sfqa.textnorm import search_text


class ChapterIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        order = [(c.chapter, c.start) for c in chunks]
        if order != sorted(order):
            raise ValueError("청크는 (장, 위치) 순서여야 한다")
        self.chunks = chunks
        self.vectors = vectors
        self._chapters = [c.chapter for c in chunks]

    def prefix_len(self, n: int) -> int:
        """N장 이하 청크의 개수(= 앞쪽 구간 길이)."""
        return bisect_right(self._chapters, n)

    def search(self, query_vec: np.ndarray, n: int, k: int) -> list[tuple[Chunk, float]]:
        upto = self.prefix_len(n)
        if upto == 0:
            return []
        q = query_vec / (np.linalg.norm(query_vec) or 1.0)
        sims = self.vectors[:upto] @ q.astype(np.float32)
        top = np.argsort(-sims, kind="stable")[:k]
        hits = [(self.chunks[i], float(sims[i])) for i in top]
        assert all(c.chapter <= n for c, _ in hits), "N장 뒤 청크가 검색됐다"
        return hits


class UptoRetriever(BaseRetriever):
    """LangChain 검색기. n장 이하에서 k개를 찾는다."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    index: ChapterIndex
    embeddings: Embeddings
    n: int
    k: int = 6

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        qvec = np.asarray(self.embeddings.embed_query(search_text(query)))
        return [
            Document(page_content=c.text, metadata={"chapter": c.chapter, "chunk_id": c.chunk_id,
                                                    "start": c.start, "end": c.end, "score": score})
            for c, score in self.index.search(qvec, self.n, self.k)
        ]
