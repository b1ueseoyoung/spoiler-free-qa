"""임베딩 색인: 청크 임베딩을 한 번 계산해 data/index에 저장한다.

meta.json에 임베딩 모델, 청크 설정, 본문 지문을 적고, 셋 중 하나라도 지금과 다르면 불러오지 않는다
(한 색인에 모델이나 본문이 섞이지 않게).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone

import numpy as np
from langchain_core.embeddings import Embeddings

from sfqa import chunk as chunking
from sfqa.chunk import Chunk, chunk_corpus
from sfqa.paths import DATA
from sfqa.textnorm import search_text

INDEX_DIR = DATA / "index"
VECTORS = INDEX_DIR / "vectors.npy"
CHUNKS = INDEX_DIR / "chunks.jsonl"
META = INDEX_DIR / "meta.json"


class IndexMismatch(RuntimeError):
    pass


def corpus_fingerprint(texts: Mapping[int, str]) -> str:
    h = hashlib.sha256()
    for ch in sorted(texts):
        h.update(f"{ch}\n".encode())
        h.update(texts[ch].encode("utf-8"))
    return h.hexdigest()


def expected_meta(embed_model: str, texts: Mapping[int, str]) -> dict:
    return {
        "embed_model": embed_model,
        "chunking": {"size": chunking.SIZE, "stride": chunking.STRIDE, "snap": chunking.SNAP},
        "corpus_sha256": corpus_fingerprint(texts),
    }


def normalize_rows(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.where(norms == 0, 1, norms)).astype(np.float32)


def build_index(embed_model: str, embeddings: Embeddings, texts: Mapping[int, str]) -> dict:
    chunks = chunk_corpus(texts)
    vectors = normalize_rows(np.asarray(embeddings.embed_documents([search_text(c.text) for c in chunks])))
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(VECTORS, vectors)
    with CHUNKS.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")
    meta = {**expected_meta(embed_model, texts), "dim": int(vectors.shape[1]), "n_chunks": len(chunks),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def load_index(embed_model: str, texts: Mapping[int, str]) -> tuple[list[Chunk], np.ndarray, dict]:
    if not META.exists():
        raise IndexMismatch("색인이 없다. `uv run sfqa index`로 먼저 만든다")
    meta = json.loads(META.read_text(encoding="utf-8"))
    want = expected_meta(embed_model, texts)
    diff = [k for k in want if meta.get(k) != want[k]]
    if diff:
        raise IndexMismatch(f"색인이 지금 설정과 다르다({', '.join(diff)}). `uv run sfqa index`로 다시 만든다")
    chunks = [Chunk(**json.loads(line)) for line in CHUNKS.read_text(encoding="utf-8").splitlines() if line]
    vectors = np.load(VECTORS)
    if len(chunks) != vectors.shape[0]:
        raise IndexMismatch("청크 수와 벡터 수가 다르다. 다시 만든다")
    return chunks, vectors, meta
