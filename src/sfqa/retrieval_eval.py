"""검색만 따로 잰다(생성 모델은 부르지 않는다. 질문 임베딩만 설정한 임베딩 공급자로 계산한다).

- A 문항: 정답 근거 인용이 검색된 청크 안에 통째로 들어가는지(recall@k). 근거가 여럿이면 '하나라도'와 '전부'를 따로 센다.
- 모든 문항: L4 누출. 검색된 청크의 장을 메타데이터가 아니라 본문 내용으로 다시 찾아 N보다 크면 누출이다.
결과는 runs/retrieval-<시각>-<방식>.json에 남긴다.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime

import numpy as np

from sfqa.chunk import Chunk
from sfqa.gold import Evidence, Question
from sfqa.paths import ROOT
from sfqa.bm25 import tokenize_many
from sfqa.retrieve import METHODS, ChapterIndex
from sfqa.textnorm import search_text

KS = (1, 3, 6, 10, 18)
GATE_K = 18  # 2026-10-13 6에서 18로 변경(docs/eval-spec.md 변경 기록)
GATE = 0.8  # docs/eval-spec.md 목표값


def positions(quote: str, text: str) -> list[int]:
    out, i = [], text.find(quote)
    while i != -1:
        out.append(i)
        i = text.find(quote, i + 1)
    return out


def covers(chunk: Chunk, ev: Evidence, texts: Mapping[int, str]) -> bool:
    return chunk.chapter == ev.chapter and any(
        chunk.start <= p and p + len(ev.quote) <= chunk.end for p in positions(ev.quote, texts[ev.chapter])
    )


def content_chapter(chunk_text: str, texts: Mapping[int, str]) -> int | None:
    """청크 글이 글자 그대로 들어 있는 가장 앞 장. 메타데이터를 믿지 않고 본문으로 다시 찾는다."""
    return next((c for c in sorted(texts) if chunk_text in texts[c]), None)


def evaluate(questions: list[Question], query_vecs: np.ndarray | None, query_terms: list[list[str]] | None,
             index: ChapterIndex, texts: Mapping[int, str], method: str = "dense", gate_k: int = GATE_K) -> dict:
    ks = tuple(sorted(set(KS) | {gate_k}))
    kmax = max(ks)
    rows = []
    for i, q in enumerate(questions):
        qv = None if query_vecs is None else query_vecs[i]
        terms = None if query_terms is None else query_terms[i]
        hits = index.search(qv, q.n, kmax, method, terms)
        leaks = [c.chunk_id for c, _ in hits[:gate_k] if (content_chapter(c.text, texts) or 99) > q.n]
        row = {"id": q.id, "split": q.split, "type": q.type, "n": q.n, "l4_leaks": leaks,
               "top": [c.chunk_id for c, _ in hits[:gate_k]]}
        if q.type == "A":
            any_rank = next((r for r, (c, _) in enumerate(hits, 1) if any(covers(c, ev, texts) for ev in q.evidence)), None)
            all_rank = None
            covered = [next((r for r, (c, _) in enumerate(hits, 1) if covers(c, ev, texts)), None) for ev in q.evidence]
            if all(r is not None for r in covered):
                all_rank = max(covered)
            row |= {"any_rank": any_rank, "all_rank": all_rank, "question": q.question,
                    "evidence_chapters": [ev.chapter for ev in q.evidence]}
        rows.append(row)

    def recall(sel, key, k):
        a = [r for r in rows if r["type"] == "A" and sel(r)]
        return sum(1 for r in a if r[key] is not None and r[key] <= k), len(a)

    summary = {}
    for name, sel in [("all", lambda r: True), ("dev", lambda r: r["split"] == "dev"), ("test", lambda r: r["split"] == "test")]:
        summary[name] = {f"recall@{k}": recall(sel, "any_rank", k) for k in ks}
        summary[name][f"recall_all@{gate_k}"] = recall(sel, "all_rank", gate_k)
    hit, total = summary["all"][f"recall@{gate_k}"]
    return {
        "summary": summary,
        "l4_leak_questions": sum(1 for r in rows if r["l4_leaks"]),
        "gate": {"k": gate_k, "threshold": GATE, "hit": hit, "total": total, "pass": total > 0 and hit / total >= GATE},
        "rows": rows,
    }


def run(method: str, questions: list[Question], index: ChapterIndex, embed_query_batch, texts: Mapping[int, str],
        meta: dict, gate_k: int = GATE_K) -> tuple[dict, str]:
    if method not in METHODS:
        raise ValueError(f"없는 검색 방식이다: {method} (가능: {', '.join(METHODS)})")
    vecs = None if method == "bm25" else np.asarray(embed_query_batch([search_text(q.question) for q in questions]))
    terms = None if method == "dense" else tokenize_many([q.question for q in questions])
    report = evaluate(questions, vecs, terms, index, texts, method, gate_k)
    report["method"] = method
    report["index_meta"] = meta
    out_dir = ROOT / "runs"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"retrieval-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{method}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report, str(path)
