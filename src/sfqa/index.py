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

INDEX_ROOT = DATA / "index"


def index_dir(embed_model: str, provider: str = "lmstudio"):
    """모델과 실행 방식마다 색인 폴더를 따로 둔다. 같은 모델이라도 런타임(GGUF, fp32)이 다르면 벡터가 조금 다르다.

    LM Studio는 data/index/<모델 이름>(기존 경로), 그 밖은 data/index/<런타임>__<모델 이름>.
    """
    name = embed_model.replace("/", "__")
    return INDEX_ROOT / (name if provider == "lmstudio" else f"{provider}__{name}")


class IndexMismatch(RuntimeError):
    pass


def corpus_fingerprint(texts: Mapping[int, str]) -> str:
    h = hashlib.sha256()
    for ch in sorted(texts):
        h.update(f"{ch}\n".encode())
        h.update(texts[ch].encode("utf-8"))
    return h.hexdigest()


def expected_meta(embed_model: str, texts: Mapping[int, str], provider: str = "lmstudio") -> dict:
    # LM Studio 색인은 런타임을 적지 않는다(기존 색인과 호환).
    runtime = {} if provider == "lmstudio" else {"embed_provider": provider}
    return {
        "embed_model": embed_model,
        **runtime,
        "chunking": {"size": chunking.SIZE, "stride": chunking.STRIDE, "snap": chunking.SNAP},
        "corpus_sha256": corpus_fingerprint(texts),
    }


def normalize_rows(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.where(norms == 0, 1, norms)).astype(np.float32)


def build_index(embed_model: str, embeddings: Embeddings, texts: Mapping[int, str], provider: str = "lmstudio") -> dict:
    chunks = chunk_corpus(texts)
    vectors = normalize_rows(np.asarray(embeddings.embed_documents([search_text(c.text) for c in chunks])))
    d = index_dir(embed_model, provider)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "vectors.npy", vectors)
    with (d / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")
    meta = {**expected_meta(embed_model, texts, provider), "dim": int(vectors.shape[1]), "n_chunks": len(chunks),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def load_index(embed_model: str, texts: Mapping[int, str], provider: str = "lmstudio") -> tuple[list[Chunk], np.ndarray, dict]:
    d = index_dir(embed_model, provider)
    if not (d / "meta.json").exists():
        raise IndexMismatch(f"{embed_model} 색인이 없다. `uv run sfqa index`로 먼저 만든다")
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    want = expected_meta(embed_model, texts, provider)
    diff = [k for k in want if meta.get(k) != want[k]]
    if diff:
        raise IndexMismatch(f"색인이 지금 설정과 다르다({', '.join(diff)}). `uv run sfqa index`로 다시 만든다")
    chunks = [Chunk(**json.loads(line)) for line in (d / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line]
    vectors = np.load(d / "vectors.npy")
    if len(chunks) != vectors.shape[0]:
        raise IndexMismatch("청크 수와 벡터 수가 다르다. 다시 만든다")
    return chunks, vectors, meta
