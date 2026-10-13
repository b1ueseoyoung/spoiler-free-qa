"""채점기: 실행 기록(records.jsonl)을 모델 없이 채점해 summary.md / summary.json / review.csv를 만든다.

기준은 docs/eval-spec.md다. 결과는 유형 × 조건별 개수로만 낸다(하나의 정확도로 합치지 않는다).
누출은 독자가 실제로 보는 것(최종 답 + 보여 주는 인용)으로 판정한다.
- C4: 검증을 통과한 인용만 보여 준다. 답하지 않으면 고정 문구만 보인다.
- C0~C3: 모델 답과 인용을 그대로 보여 준다.
사람 판정(review.csv의 human_outcome, human_leak)은 자동 판정과 따로 센다(--with-review).
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from sfqa.chunk import chunk_corpus
from sfqa.gold import Question, entity_index, load_entities, load_questions, load_texts
from sfqa.retrieval_eval import content_chapter
from sfqa.textnorm import has_key

KEPT = ("verified", "chapter_fixed")
OUTCOMES = {
    "A": ["correct", "wrong", "missed", "format_fail"],
    "B": ["abstained", "answered", "format_fail"],
    "C": ["abstained", "ungrounded", "format_fail"],
}
LABELS = {"correct": "정답", "wrong": "오답", "missed": "놓침", "format_fail": "형식 실패", "abstained": "보류",
          "answered": "답함", "ungrounded": "근거 없는 답"}
REVIEW_COLS = ["priority", "qid", "condition", "type", "n", "question", "gold_answer", "status", "answer",
               "shown_quotes", "auto_outcome", "auto_leaks", "reasons", "human_outcome", "human_leak", "memo"]


def shown(result: dict) -> tuple[str, list[dict]]:
    """독자에게 보이는 (답, 인용)."""
    if result["status"] != "answered":
        return result["answer"], []
    quotes = result["quotes"]
    if result["condition"] == "C4":
        quotes = [q for q in quotes if q["status"] in KEPT]
    return result["answer"], quotes


def judge(rec: dict, q: Question, entity_first: dict, chunk_text: dict, texts: dict) -> dict:
    r = rec["result"]
    n = rec["n"]
    answer, quotes = shown(r)
    answered = r["status"] == "answered"
    visible = " ".join([answer, *(x["text"] for x in quotes)]) if answered else ""
    fmt_fail = r["invalid"] or r["truncated"]

    if q.type == "A":
        if fmt_fail:
            outcome = "format_fail"
        elif not answered:
            outcome = "missed"
        else:
            outcome = "correct" if all(has_key(answer, k) for k in q.answer_keys) else "wrong"
    elif fmt_fail:
        outcome = "format_fail"
    elif q.type == "B":
        outcome = "answered" if answered else "abstained"
    else:
        outcome = "ungrounded" if answered else "abstained"

    kept = [x for x in r["quotes"] if x["status"] in KEPT]
    evidence_chapters = {ev.chapter for ev in q.evidence}
    leaks = {
        "L1": answered and any(x["status"] == "future_only" for x in quotes),
        "L2": answered and any(ff.chapter > n and all(has_key(visible, k) for k in ff.keys) for ff in q.future_facts),
        "L3": answered and any(first is not None and first > n and name in visible for name, first in entity_first.items()),
        "L4": any((content_chapter(chunk_text[d["chunk_id"]], texts) or 99) > n for d in r["retrieved"]),
    }
    return {
        "qid": rec["qid"], "condition": rec["condition"], "type": q.type, "split": q.split, "n": n,
        "outcome": outcome, "leaks": leaks, "leak_shown": leaks["L1"] or leaks["L2"] or leaks["L3"],
        "kept_quotes": len(kept), "quote_chapter_match": any(x["found_chapter"] in evidence_chapters for x in kept),
        "answered": answered, "visible": visible, "invalid": r["invalid"], "truncated": r["truncated"],
        "latency_s": r["response"].get("latency_s", 0.0), "input_tokens": r["response"].get("input_tokens", 0),
        "output_tokens": r["response"].get("output_tokens", 0), "reasons": r["reasons"],
        "answer": answer, "quotes": quotes, "status": r["status"],
    }


def priority(j: dict) -> int | None:
    if any(j["leaks"].values()) or (j["type"] in "BC" and j["answered"]) or (j["answered"] and j["kept_quotes"] == 0):
        return 1
    if j["type"] == "A" and j["outcome"] in ("wrong", "missed"):
        return 2
    return None


def read_review(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig") as fh:
        return {(row["qid"], row["condition"]): row for row in csv.DictReader(fh)}


def apply_human(judged: list[dict], review: dict) -> list[dict]:
    """사람이 채운 칸만 덮어쓴다. human_outcome은 outcome 이름이나 한글 이름, human_leak은 예/아니오."""
    by_label = {v: k for k, v in LABELS.items()}
    out = []
    for j in judged:
        row = review.get((j["qid"], j["condition"]), {})
        j = dict(j)
        ho = (row.get("human_outcome") or "").strip()
        if ho:
            j["outcome"] = by_label.get(ho, ho)
        hl = (row.get("human_leak") or "").strip()
        if hl:
            j["leak_shown"] = hl in ("예", "yes", "y", "1", "true")
        out.append(j)
    return out


def aggregate(judged: list[dict]) -> dict:
    table: dict = defaultdict(lambda: defaultdict(Counter))
    for j in judged:
        c = table[j["condition"]][j["type"]]
        c[j["outcome"]] += 1
        c["total"] += 1
        for name, hit in j["leaks"].items():
            c[name] += hit
        c["leak_shown"] += j["leak_shown"]
        if j["type"] == "A" and j["answered"]:
            c["answered"] += 1
            c["kept_quote"] += j["kept_quotes"] > 0
            c["chapter_match"] += j["quote_chapter_match"]
        c["latency_s"] += j["latency_s"]
        c["input_tokens"] += j["input_tokens"]
        c["output_tokens"] += j["output_tokens"]
    return {cond: {t: dict(cnt) for t, cnt in types.items()} for cond, types in table.items()}


def targets(agg: dict) -> list[str]:
    """docs/eval-spec.md 목표값과 비교(있는 조건만)."""
    lines = []
    for cond in ("C2", "C3", "C4"):
        if cond in agg:
            l4 = sum(t.get("L4", 0) for t in agg[cond].values())
            lines.append(f"{cond} L4 누출 문항 {l4}개 (목표 0) → {'달성' if l4 == 0 else '미달'}")
    if "C4" in agg and "B" in agg["C4"]:
        b = agg["C4"]["B"]
        lines.append(f"C4 B 누출(L1~L3) {b.get('leak_shown', 0)}/{b['total']} (목표 0) → {'달성' if not b.get('leak_shown') else '미달'}")
    if "C4" in agg and "C" in agg["C4"]:
        c = agg["C4"]["C"]
        ok = c.get("abstained", 0) / c["total"] >= 0.9
        lines.append(f"C4 C 보류 {c.get('abstained', 0)}/{c['total']} (목표 90% 이상) → {'달성' if ok else '미달'}")
    return lines


def render_md(run_dir: Path, info: dict, agg: dict, gold_status: Counter, label: str) -> str:
    conds = [c for c in ("C0", "C1", "C2", "C3", "C4") if c in agg]
    lines = [f"# 평가 결과 — {run_dir.name} ({label})", ""]
    reviewed = gold_status.get("human-reviewed", 0)
    lines += [f"- 정답지: {sum(gold_status.values())}문항 중 사람 검토 {reviewed}개"
              + (" — **사람 검토 전 정답지로 낸 결과다**" if reviewed == 0 else ""),
              f"- 설정: {info.get('model', {}).get('llm')} / 임베딩 {info.get('model', {}).get('embed')} / k={info.get('k')}"
              f" / 프롬프트 {info.get('prompt_hash')} / git {info.get('git', {}).get('head')}"
              + (" (커밋 안 된 변경 있음)" if info.get("git", {}).get("dirty") else ""),
              f"- 실행: 기록 {info.get('records')}건, 모델 호출 {info.get('model_calls')}회, "
              f"{'완료' if info.get('complete') else '중간에 멈춤: ' + str(info.get('stopped'))}", ""]
    for t, title in (("A", "A 답 있음"), ("B", "B 스포일러 미끼"), ("C", "C 작품에 없음")):
        if not any(t in agg[cond] for cond in conds):
            continue
        cols = OUTCOMES[t]
        head = ["조건", *(LABELS[c] for c in cols), "누출 L1", "L2", "L3", "보인 누출(L1~L3)"]
        if t == "A":
            head += ["답한 것 중 검증 인용 있음", "인용 장 일치"]
        lines += [f"## {title}", "", "| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
        for cond in conds:
            c = agg[cond].get(t)
            if not c:
                continue
            row = [cond, *(f"{c.get(o, 0)}/{c['total']}" for o in cols),
                   str(c.get("L1", 0)), str(c.get("L2", 0)), str(c.get("L3", 0)), f"{c.get('leak_shown', 0)}/{c['total']}"]
            if t == "A":
                row += [f"{c.get('kept_quote', 0)}/{c.get('answered', 0)}", f"{c.get('chapter_match', 0)}/{c.get('answered', 0)}"]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    lines += ["## 검색 누출 L4 (본문 내용으로 다시 찾은 장 > N인 청크가 검색된 문항)", "", "| 조건 | 문항 |", "| --- | --- |"]
    for cond in conds:
        if cond == "C0":
            continue
        lines.append(f"| {cond} | {sum(t.get('L4', 0) for t in agg[cond].values())}/{sum(t['total'] for t in agg[cond].values())} |")
    lines += ["", "## 목표값 비교", "", *(f"- {x}" for x in targets(agg)), "", "## 호출", "",
              "| 조건 | 문항 | 평균 지연(초) | 입력 토큰 합 | 출력 토큰 합 |", "| --- | --- | --- | --- | --- |"]
    for cond in conds:
        tot = sum(t["total"] for t in agg[cond].values())
        lat = sum(t.get("latency_s", 0) for t in agg[cond].values())
        lines.append(f"| {cond} | {tot} | {lat / tot:.1f} | {sum(t.get('input_tokens', 0) for t in agg[cond].values())} | "
                     f"{sum(t.get('output_tokens', 0) for t in agg[cond].values())} |")
    lines += ["", "C4는 C3 응답을 다시 쓰므로 지연과 토큰이 C3와 같다(새 호출 0회)."]
    return "\n".join(lines) + "\n"


def write_review(path: Path, judged: list[dict], questions: dict[str, Question]) -> int:
    rows = []
    for j in judged:
        p = priority(j)
        if p is None:
            continue
        q = questions[j["qid"]]
        rows.append({
            "priority": p, "qid": j["qid"], "condition": j["condition"], "type": j["type"], "n": j["n"],
            "question": q.question, "gold_answer": q.gold_answer or "", "status": j["status"], "answer": j["answer"],
            "shown_quotes": "\n".join(f"[{x['chapter']}장/{x['status']}] {x['text']}" for x in j["quotes"]),
            "auto_outcome": LABELS[j["outcome"]], "auto_leaks": ", ".join(k for k, v in j["leaks"].items() if v),
            "reasons": "; ".join(j["reasons"]), "human_outcome": "", "human_leak": "", "memo": "",
        })
    rows.sort(key=lambda r: (r["priority"], r["condition"], r["qid"]))
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=REVIEW_COLS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def score_run(run_dir: Path, with_review: bool = False) -> dict:
    info = json.loads((run_dir / "run-info.json").read_text(encoding="utf-8"))
    records = [json.loads(x) for x in (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    texts = load_texts()
    questions = {q.id: q for q in load_questions()}
    entity_first = entity_index(load_entities(), texts)
    chunk_text = {c.chunk_id: c.text for c in chunk_corpus(texts)}
    judged = [judge(r, questions[r["qid"]], entity_first, chunk_text, texts) for r in records]
    gold_status = Counter(questions[qid].review for qid in {r["qid"] for r in records})

    review_path = run_dir / "review.csv"
    existing = read_review(review_path)
    human_filled = any((row.get("human_outcome") or row.get("human_leak") or "").strip() for row in existing.values())
    if not human_filled:
        n_review = write_review(review_path, judged, questions)
    else:
        n_review = len(existing)  # 사람이 채운 표는 덮어쓰지 않는다

    agg = aggregate(judged)
    out = {"run": run_dir.name, "auto": agg, "targets": targets(agg), "review_rows": n_review,
           "gold_review_status": dict(gold_status)}
    md = render_md(run_dir, info, agg, gold_status, "자동 판정")
    if with_review and human_filled:
        hagg = aggregate(apply_human(judged, existing))
        out["with_human_review"] = hagg
        out["targets_with_human_review"] = targets(hagg)
        md += "\n" + render_md(run_dir, info, hagg, gold_status, "사람 판정 반영")
    (run_dir / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(md, encoding="utf-8")
    return out
