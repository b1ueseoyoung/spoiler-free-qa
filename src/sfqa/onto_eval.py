"""추출 온톨로지를 설정집 정답과 비교한다(docs/ontology-extract.md 7절). 판정은 추출과 다른 모델로 Batch API(반값)에서 한다.

1. 후보: 정답 사실마다 추출 사실 후보 k개를 BM25(sfqa.bm25)로 고른다. 뒤 장 사실을 앞 장에서 뽑았는지도 보려고
   장을 가리지 않고 모든 추출 사실에서 찾는다.
2. 판정: 판정 모델이 후보 가운데 정답과 같은 사실을 모두 고른다. 정답 하나가 추출 사실 여러 개에 나뉘어 있을 수 있다.
   같은 사실이 없다고 나온 정답만, 공개 장 앞뒤 1장의 추출 사실을 후보로 한 번 더 묻는다(BM25가 놓친 경우를 잡는다).
   추출 사실은 무작위 표본만 근거 판정(인용이 사실 문장을 뒷받침하는가)을 받는다. 전부 하면 추출 사실이 정답의 약 3배라 비싸다.
3. 수치: 사실 재현율(반전 따로), 공개 장(앞당김·같음·늦음), 아는 인물 정밀도·재현율, 아는 장 앞당김, 공개 뒤에 알게 됨.
   - 아는 인물은 추출 범위 본문에 이름이 나온 정답 인물로 양쪽을 똑같이 제한한다. 정답에 없는 인물은 세지 않는다.
   - 추출 사실이 상식(common)이면 그 인물들이 모두 0장부터 안다고 본다.
   - 아는 장 비교: 정답 하나가 추출 사실 여러 개에 나뉘어 짝지어졌으면 모든 부분을 알게 된 가장 늦은 장을 쓴다. 부분 짝(정답
     일부만 담은 짝)의 앞당김은 따로 센다. 10/24까지는 가장 이른 부분을 써서, 앞 장의 부분 사실(2장 '왼손 장갑')로 뒤 장
     정답(8장 '장갑 속 서약흔')을 일찍 안다고 셌다.
   - 공개 장 앞당김과 아는 장 앞당김은 사례 목록을 따로 낸다(사람이 보고 실제 누출인지 판단한다).
   - 사람이 판정을 확인할 30개 표(review-matches.csv)를 만든다.

실행: uv run python -m sfqa.onto_eval judge --run runs/onto-extract/<결과> --gold wn01-rofan --max-usd 0.6
      uv run python -m sfqa.onto_eval score --run runs/onto-extract/<결과> --gold wn01-rofan
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path

from sfqa.llm import LLMResponse, _load_cassette
from sfqa.onto_extract import PRICES, RUNS, STOP_TO_FINISH, request_key
from sfqa.ontology import SYNTHETIC_ROOT, Fact, Ontology, from_synthetic

JUDGE_MODEL = "claude-sonnet-5-5"
JUDGE_SETTINGS = ({"type": "between_tools"}, "high")  # thinking 끔
JUDGE_MAX_TOKENS = 512
K = 8
SUPPORT_SAMPLE = 100
SUPPORT_PARAGRAPHS = 3  # 근거 판정에 인용 앞뒤로 붙이는 문단 수
REVIEW_SAMPLE = 30
SEED = 20261023

JUDGE_MATCH = """정답 사실 하나와, 추출기가 뽑은 사실 후보 여러 개를 받는다. 정답과 같은 내용을 말하는 후보의 번호를 모두 고른다.
- 같은 사건·관계·정체·속성을 말하면 같은 사실이다. 표현이 달라도 된다.
- 정답의 내용이 여러 후보에 나뉘어 있으면 그 후보들을 모두 고른다.
- 고른 후보들을 합쳐도 정답의 핵심 일부만 담으면 partial을 true로 한다.
- 같은 내용의 후보가 없으면 matches를 빈 배열로 한다.
- reason은 한 문장으로 쓴다."""
JUDGE_SUPPORT = """추출기가 뽑은 사실 문장과, 그 근거로 든 본문 인용문을 받는다. 인용문과 그 앞뒤 본문만으로 사실 문장이 뒷받침되는지 판단한다.
- yes: 인용 부분이 사실 문장을 분명히 보여 준다.
- partial: 일부만 보여 주거나, 문장이 인용보다 더 나아간다.
- no: 인용과 관계없거나, 인용에 없는 내용을 단정한다(작품 밖 기억, 뒤 장 내용).
- reason은 한 문장으로 쓴다."""


def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


_STR, _BOOL = {"type": "string"}, {"type": "boolean"}
MATCH_SCHEMA = _obj({"matches": {"type": "array", "items": {"type": "integer"}}, "partial": _BOOL, "reason": _STR})
SUPPORT_SCHEMA = _obj({"supported": {"type": "string", "enum": ["yes", "partial", "no"]}, "reason": _STR})


# ---------- 인물 잇기와 후보 ----------


def name_score(c, g) -> int:
    """이름이 정확히 같으면 3점, 이름이 정답 별칭과 같으면 2점, 정답 이름 안에 들어 있으면 1점."""
    if c.name == g.name:
        return 3
    if c.name in g.forms or any(a.form == g.name for a in c.aliases):
        return 2
    return 1 if len(c.name) >= 2 and c.name in g.name else 0


def id_map(onto: Ontology, gold: Ontology) -> dict[str, str]:
    """추출 인물마다 점수가 가장 높은 정답 인물로 잇는다(0점이면 잇지 않는다)."""
    out = {}
    for c in onto.characters:
        scores = {g.id: name_score(c, g) for g in gold.characters}
        best = max(scores, key=scores.get)
        if scores[best]:
            out[c.id] = best
    return out


def candidates(gold_facts: list[Fact], facts: list[Fact], k: int = K) -> dict[str, list[str]]:
    """정답 사실 id → 추출 사실 id k개(BM25 순). 점수가 0인 후보는 넣지 않는다."""
    from sfqa.bm25 import PrefixBM25, tokenize_many

    if not facts:
        return {g.id: [] for g in gold_facts}
    bm = PrefixBM25(tokenize_many([f.statement for f in facts]))
    queries = tokenize_many([g.statement for g in gold_facts])
    out = {}
    for g, q in zip(gold_facts, queries):
        s = bm.scores(q, len(facts))
        order = sorted(range(len(facts)), key=lambda i: -s[i])
        out[g.id] = [facts[i].id for i in order[:k] if s[i] > 0]
    return out


# ---------- 판정 요청과 Batch ----------


def _request(system: str, user: str, schema: dict) -> dict:
    thinking, effort = JUDGE_SETTINGS
    return {"model": JUDGE_MODEL, "max_tokens": JUDGE_MAX_TOKENS, "system": system,
            "messages": [{"role": "user", "content": user}], "thinking": thinking,
            "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": schema}}}


def match_request(g: Fact, cands: list[Fact]) -> dict:
    lines = "\n".join(f"{i}. {f.statement}" for i, f in enumerate(cands, 1)) or "(후보 없음)"
    return _request(JUDGE_MATCH, f"정답: {g.statement}\n후보:\n{lines}", MATCH_SCHEMA)


def support_request(f: Fact, texts: Mapping[int, str]) -> dict:
    """인용 줄과 그 앞뒤 문단(빈 줄은 세지 않고 SUPPORT_PARAGRAPHS개씩)을 보낸다.

    10/26까지는 앞뒤 한 줄만 붙였다. 본문은 문단 사이에 빈 줄이 있어 판정 모델이 인용 한 줄만 받았다(18절).
    """
    lines = texts[f.reveal_ch].splitlines()
    near = f.evidence[0].lines if f.evidence else []
    quote = "\n".join(lines[i - 1] for i in near)
    filled = [i for i, line in enumerate(lines, 1) if line.strip()]
    pos = [p for p, i in enumerate(filled) if near and min(near) <= i <= max(near)]
    around = filled[max(0, pos[0] - SUPPORT_PARAGRAPHS):pos[-1] + 1 + SUPPORT_PARAGRAPHS] if pos else []
    context = "\n\n".join(lines[i - 1] for i in around)
    return _request(JUDGE_SUPPORT, f"사실: {f.statement}\n인용: {quote}\n앞뒤 본문:\n{context}", SUPPORT_SCHEMA)


def batch_cost(resp: LLMResponse) -> float:
    i, o, cr, cw = PRICES[resp.model]
    x = resp.extra
    return (resp.input_tokens * i + x.get("cache_write", 0) * cw + x.get("cache_read", 0) * cr
            + resp.output_tokens * o) / 1_000_000 / 2  # Batch는 반값


def _guess(req: dict) -> float:
    """보내기 전 어림: 글자 수 × 1.1 토큰 + 스키마 600 토큰, 출력은 상한의 절반."""
    chars = len(req["system"]) + sum(len(m["content"]) for m in req["messages"])
    i, o, _, _ = PRICES[req["model"]]
    return ((chars * 1.1 + 600) * i + req["max_tokens"] / 2 * o) / 1_000_000 / 2


class Budget(Exception):
    pass


def run_batch(items: dict[str, dict], cassette: Path | None, *, sdk=None, max_usd: float = 0.6, poll_s: float = 30,
              log: Callable[[str], None] = print) -> tuple[dict[str, LLMResponse], float]:
    """custom_id → 요청을 Batch로 한 번에 보낸다. 카세트에 있는 것은 보내지 않는다. (응답, 새로 쓴 돈)을 돌려준다."""
    cache = _load_cassette(cassette)
    keys = {cid: request_key(req, 0) for cid, req in items.items()}
    out = {cid: cache[keys[cid]] for cid in items if keys[cid] in cache}
    todo = {cid: req for cid, req in items.items() if cid not in out}
    if not todo:
        return out, 0.0
    guess = sum(_guess(req) for req in todo.values())
    if guess > max_usd:
        raise Budget(f"판정 어림 ${guess:.2f}가 상한 ${max_usd:.2f}를 넘는다({len(todo)}건)")
    if sdk is None:
        from sfqa.claude import KEY_ENV, load_key

        if not load_key():
            raise RuntimeError(f"{KEY_ENV}가 없다. 프로젝트 .env에 넣거나 환경 변수로 준다")
        import anthropic

        sdk = anthropic.Anthropic(max_retries=3)
    batch = sdk.messages.batches.create(requests=[{"custom_id": cid, "params": req} for cid, req in todo.items()])
    log(f"Batch {batch.id}: {len(todo)}건, 어림 ${guess:.2f}")
    while (status := sdk.messages.batches.retrieve(batch.id)).processing_status != "ended":
        log(f"  처리 중 {status.request_counts}")
        time.sleep(poll_s)
    spent = 0.0
    for r in sdk.messages.batches.results(batch.id):
        req = todo[r.custom_id]
        if r.result.type != "succeeded":  # 실패·만료는 비용이 없고, 카세트에 남기지 않아 다음에 다시 보낸다
            out[r.custom_id] = LLMResponse(raw_text=None, finish_reason=r.result.type, model=req["model"],
                                           extra={"batch_id": batch.id})
            continue
        msg, usage = r.result.message, r.result.message.usage
        resp = LLMResponse(
            raw_text=None if msg.stop_reason == "refusal" else next((b.text for b in msg.content if b.type == "text"), None),
            finish_reason=STOP_TO_FINISH.get(msg.stop_reason, msg.stop_reason), input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens, model=req["model"],
            extra={"batch_id": batch.id, "stop_reason": msg.stop_reason,
                   "cache_read": getattr(usage, "cache_read_input_tokens", 0) or 0,
                   "cache_write": getattr(usage, "cache_creation_input_tokens", 0) or 0})
        spent += batch_cost(resp)
        if cassette:
            cassette.parent.mkdir(parents=True, exist_ok=True)
            with cassette.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": keys[r.custom_id], "response": asdict(resp)}, ensure_ascii=False) + "\n")
        out[r.custom_id] = resp
    return out, spent


def _parsed(resp: LLMResponse | None) -> dict | None:
    if resp is None or resp.finish_reason != "stop" or not resp.raw_text:
        return None
    try:
        return json.loads(resp.raw_text)
    except json.JSONDecodeError:
        return None


def judge(run_dir: Path, gold_id: str, *, root: Path = SYNTHETIC_ROOT, cassette: Path | None = RUNS / "judge-cassette.jsonl",
          k: int = K, support_sample: int = SUPPORT_SAMPLE, max_usd: float = 0.6, sdk=None, poll_s: float = 30,
          log: Callable[[str], None] = print, reuse: bool = True, redo_support: bool = False) -> dict:
    """정답 사실마다 같은 사실 판정, 추출 사실 표본마다 근거 판정을 하고 run_dir/judgments.json에 쓴다.

    reuse이면 이전 판정(judgments.json)에서 같은 사실을 찾은 정답과 근거 표본은 그대로 쓰고, 새로 공개된 정답과
    이전에 놓친 정답만 판정한다. 뒤 장을 더 뽑아도 앞 장에서 찾은 짝의 공개 장은 바뀌지 않는다. 아는 인물은 score가
    지금 온톨로지로 다시 센다. 후보는 새 추출 사실로 다시 고르므로, 다시 판정하면 이전과 다른 요청이 된다.
    redo_support이면 이전 근거 표본과 같은 사실들을 지금 요청(support_request)으로 다시 판정한다.
    """
    from sfqa.gold import load_texts
    from sfqa.works import load_work

    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    upto = report["done"]
    gold = from_synthetic(gold_id, root)
    onto = Ontology.model_validate_json((run_dir / "ontology.json").read_text(encoding="utf-8"))
    texts = load_texts(load_work(gold_id, root))
    facts = {f.id: f for f in onto.facts}
    previous = run_dir / "judgments.json"
    prev = json.loads(previous.read_text(encoding="utf-8")) if reuse and previous.exists() else {}
    if prev.get("gold") != gold_id:
        prev = {}
    kept = {gid: v for gid, v in prev.get("match", {}).items() if v["ids"] and all(i in facts for i in v["ids"])}
    gfacts = [g for g in gold.facts if g.reveal_ch <= upto and g.id not in kept]
    cands = candidates(gfacts, onto.facts, k)
    items = {f"m-{g.id}": match_request(g, [facts[i] for i in cands[g.id]]) for g in gfacts}
    if prev.get("support") and not redo_support:
        sample = []
    elif prev.get("support"):
        sample = sorted(fid for fid in prev["support"] if fid in facts)
    else:
        pool = sorted(f.id for f in onto.facts if f.evidence)
        sample = sorted(random.Random(SEED).sample(pool, min(support_sample, len(pool))))
    items |= {f"s-{fid}": support_request(facts[fid], texts) for fid in sample}
    responses, spent = run_batch(items, cassette, sdk=sdk, max_usd=max_usd, poll_s=poll_s, log=log)

    def picked(cid: str, cand: list[str]) -> tuple[dict | None, list[str]]:
        p = _parsed(responses.get(cid))
        return p, [cand[i - 1] for i in p["matches"] if 1 <= i <= len(cand)] if p else []

    # 둘째 판정: 첫 판정에서 같은 사실이 없던 정답만, 공개 장 앞뒤 1장의 추출 사실을 모두 후보로 다시 묻는다(BM25가 놓친 경우).
    # 개수를 자르지 않는다. 앞에서부터 자르면 앞 장 사실만 남는다(10/23 첫 실행에서 29장 사실이 잘렸다).
    near = {}
    for g in gfacts:
        p, ids = picked(f"m-{g.id}", cands[g.id])
        if p is not None and not ids:
            near[g.id] = [f.id for f in onto.facts if abs(f.reveal_ch - g.reveal_ch) <= 1 and f.id not in cands[g.id]]
    retry = {f"m2-{gid}": match_request(next(g for g in gfacts if g.id == gid), [facts[i] for i in ids])
             for gid, ids in near.items() if ids}
    if retry:
        more, extra = run_batch(retry, cassette, sdk=sdk, max_usd=max_usd - spent, poll_s=poll_s, log=log)
        responses |= more
        spent += extra

    out = {"gold": gold_id, "upto": upto, "judge_model": JUDGE_MODEL, "spent_usd": round(spent, 4),
           "match": dict(kept), "support": {} if redo_support else dict(prev.get("support", {})), "failed": [],
           "reused": len(kept)}
    for g in gfacts:
        p, ids = picked(f"m-{g.id}", cands[g.id])
        if p is None:
            out["failed"].append(f"m-{g.id}")
            continue
        entry = {"ids": ids, "partial": bool(p["partial"]) and bool(ids), "reason": p["reason"], "candidates": cands[g.id],
                 "pass": 1}
        if not ids and f"m2-{g.id}" in retry:
            p2, ids2 = picked(f"m2-{g.id}", near[g.id])
            if p2 is None:
                out["failed"].append(f"m2-{g.id}")
            elif ids2:
                entry = {"ids": ids2, "partial": bool(p2["partial"]), "reason": p2["reason"],
                         "candidates": cands[g.id] + near[g.id], "pass": 2}
        out["match"][g.id] = entry
    for fid in sample:
        p = _parsed(responses.get(f"s-{fid}"))
        if p is None:
            out["failed"].append(f"s-{fid}")
            continue
        out["support"][fid] = {"supported": p["supported"], "reason": p["reason"]}
    (run_dir / "judgments.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


# ---------- 수치 ----------


def score(gold: Ontology, onto: Ontology, upto: int, judgments: dict, texts: Mapping[int, str]) -> dict:
    m, facts = id_map(onto, gold), {f.id: f for f in onto.facts}
    seen = "".join(texts[c] for c in range(1, upto + 1))
    comparable = {c.id for c in gold.characters if any(f in seen for f in c.forms)}
    out = {"upto": upto, "gold_facts": 0, "full": 0, "partial": 0, "missed": 0, "by_category": {}, "twists": Counter(),
           "reveal": Counter(), "reveal_early": [], "tp": 0, "fp": 0, "fn": 0, "since_early": [], "since_early_partial": [],
           "since_late": 0, "after_reveal": 0, "after_reveal_hit": 0, "comparable": sorted(comparable)}
    cats: dict[str, Counter] = {}
    for g in (f for f in gold.facts if f.reveal_ch <= upto):
        out["gold_facts"] += 1
        j = judgments["match"].get(g.id)
        kind = "missed" if not j or not j["ids"] else "partial" if j["partial"] else "full"
        out[kind] += 1
        cats.setdefault(g.category, Counter())[kind] += 1
        if g.is_twist:
            out["twists"][kind] += 1
        if kind == "missed":
            continue
        first = min(facts[e].reveal_ch for e in j["ids"])
        out["reveal"]["early" if first < g.reveal_ch else "same" if first == g.reveal_ch else "late"] += 1
        if first < g.reveal_ch:
            e = min(j["ids"], key=lambda x: facts[x].reveal_ch)
            out["reveal_early"].append({"gold": g.id, "gold_ch": g.reveal_ch, "gold_statement": g.statement,
                                        "extracted": e, "extracted_ch": first, "extracted_statement": facts[e].statement})
        gk = {k.character: k.from_ch for k in g.known_by if k.from_ch <= upto and k.character in comparable}
        parts = []  # 짝지어진 추출 사실마다 인물 → 아는 장
        for e in j["ids"]:
            known = ({c: 0 for c in comparable} if facts[e].common
                     else {m.get(k.character): k.from_ch for k in facts[e].known_by})
            parts.append({who: s for who, s in known.items() if who in comparable})
        ek: dict[str, int] = {}  # 어느 부분이든 아는 인물(정밀도·재현율, 10/23 보고와 같은 계산)
        for part in parts:
            for who, s in part.items():
                ek[who] = min(ek.get(who, s), s)
        for who in ek:
            out["tp" if who in gk else "fp"] += 1
        out["fn"] += sum(1 for who in gk if who not in ek)
        # 아는 장 비교: 정답 사실 전체를 안 장 = 모든 부분을 알게 된 가장 늦은 장. 부분 짝은 따로 센다(정답 일부만 담았다).
        whole = {who: max(p[who] for p in parts) for who in set.intersection(*(set(p) for p in parts))}
        for who, s in whole.items():
            if who in gk and s < gk[who]:
                out["since_early_partial" if kind == "partial" else "since_early"].append(
                    {"gold": g.id, "character": who, "extracted": s, "gold_since": gk[who], "statement": g.statement})
            elif who in gk and s > gk[who] and kind == "full":
                out["since_late"] += 1
        after = [who for who, s in gk.items() if s > g.reveal_ch]
        out["after_reveal"] += len(after)
        out["after_reveal_hit"] += sum(1 for who in after if who in ek)
    out["by_category"] = {c: dict(v) for c, v in sorted(cats.items())}
    out["twists"], out["reveal"] = dict(out["twists"]), dict(out["reveal"])
    out["precision"] = round(out["tp"] / max(1, out["tp"] + out["fp"]), 3)
    out["recall"] = round(out["tp"] / max(1, out["tp"] + out["fn"]), 3)
    out["support"] = dict(Counter(s["supported"] for s in judgments["support"].values()))
    return out


def review_sheet(path: Path, gold: Ontology, onto: Ontology, judgments: dict) -> None:
    """판정 확인용 표: 정답 사실 무작위 30개. 사람이 human_ok(예·아니오)와 human_note를 채운다."""
    facts = {f.id: f for f in onto.facts}
    golds = {g.id: g for g in gold.facts}
    ids = sorted(judgments["match"])
    pick = sorted(random.Random(SEED).sample(ids, min(REVIEW_SAMPLE, len(ids))))
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["gold_id", "gold_ch", "gold_statement", "judged", "partial", "matched", "candidates", "reason",
                    "human_ok", "human_note"])
        for gid in pick:
            j = judgments["match"][gid]
            w.writerow([gid, golds[gid].reveal_ch, golds[gid].statement, "같음" if j["ids"] else "없음", j["partial"],
                        " / ".join(f"{e}[{facts[e].reveal_ch}장] {facts[e].statement}" for e in j["ids"]),
                        " / ".join(f"{e} {facts[e].statement}" for e in j["candidates"]), j["reason"], "", ""])


def _summary(s: dict) -> str:
    r = s["reveal"]
    return (f"사실 {s['gold_facts']}개: 완전 {s['full']}·일부 {s['partial']}·놓침 {s['missed']} | 반전 {s['twists']} | "
            f"공개 장 같음 {r.get('same', 0)}·앞당김 {r.get('early', 0)}·늦음 {r.get('late', 0)} | "
            f"아는 인물 정밀도 {s['precision']} 재현율 {s['recall']} | 공개 뒤에 알게 됨 {s['after_reveal_hit']}/{s['after_reveal']} | "
            f"아는 장 앞당김 {len(s['since_early'])}(부분 짝 {len(s['since_early_partial'])} 따로)·늦음 {s['since_late']} | "
            f"근거 표본 {s['support']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sfqa.onto_eval", description="추출 온톨로지를 정답과 비교한다")
    sub = parser.add_subparsers(dest="cmd", required=True)
    pj = sub.add_parser("judge", help="판정 모델로 같은 사실·근거를 판정한다(Batch, 유료)")
    pj.add_argument("--run", required=True)
    pj.add_argument("--gold", required=True)
    pj.add_argument("--max-usd", type=float, required=True)
    pj.add_argument("--k", type=int, default=K)
    pj.add_argument("--support-sample", type=int, default=SUPPORT_SAMPLE)
    pj.add_argument("--redo-support", action="store_true", help="이전 근거 표본과 같은 사실들을 지금 요청으로 다시 판정한다")
    ps = sub.add_parser("score", help="판정 결과로 수치를 낸다(모델 호출 없음)")
    ps.add_argument("--run", required=True)
    ps.add_argument("--gold", required=True)
    args = parser.parse_args(argv)

    run_dir = Path(args.run)
    if args.cmd == "judge":
        out = judge(run_dir, args.gold, k=args.k, support_sample=args.support_sample, max_usd=args.max_usd,
                    redo_support=args.redo_support)
        print(f"판정: 같은 사실 {len(out['match'])}건, 근거 {len(out['support'])}건, 실패 {len(out['failed'])}건, "
              f"새로 쓴 돈 ${out['spent_usd']}")
        return 1 if out["failed"] else 0
    from sfqa.gold import load_texts
    from sfqa.works import load_work

    judgments = json.loads((run_dir / "judgments.json").read_text(encoding="utf-8"))
    gold = from_synthetic(args.gold)
    onto = Ontology.model_validate_json((run_dir / "ontology.json").read_text(encoding="utf-8"))
    s = score(gold, onto, judgments["upto"], judgments, load_texts(load_work(args.gold, SYNTHETIC_ROOT)))
    (run_dir / "evaluation.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    sheet = run_dir / "review-matches.csv"
    if sheet.exists():  # 사람이 채우고 있을 수 있어 덮어쓰지 않는다
        print(f"확인표는 그대로 둔다: {sheet}")
    else:
        review_sheet(sheet, gold, onto, judgments)
    print(_summary(s))
    print(f"결과: {run_dir / 'evaluation.json'}, 확인표: {sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
