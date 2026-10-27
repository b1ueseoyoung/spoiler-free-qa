"""같은 범위를 다른 규칙으로 뽑은 두 실행을 쌍 단위로 비교한다. 모델을 부르지 않는다(docs/ontology-extract.md 14절).

두 실행 모두 onto_eval judge와 score를 마친 뒤에 쓴다(judgments.json, evaluation.json).
- 아는 인물 쌍: 한쪽에서만 맞힌 정답 쌍, 한쪽에서만 틀린 쌍을 보인다. score처럼 놓친 정답 사실의 쌍은 세지 않는다.
- 채팅 기본값 보기: 인물 시점을 독자 위치 N과 같게 두고, N장마다 공개된 정답 사실 × N장까지 이름이 나온 인물로 칸을
  만든다. 추출 쪽이 더 아는 칸(정답에서는 뒤에 알게 됨 / 범위 끝까지 모름)과 모르는 칸을 센다.
  추출 쪽 '안다'는 두 기준으로 센다(상식이면 0장부터, 놓친 정답은 아무도 모른다고 본다).
  - 모두: 짝지어진 추출 사실을 N장까지 모두 안 경우. 판정이 좁은 사실을 하나 더 짝지으면 나머지 인물이 모두 모름이 된다(15절 f002).
  - 하나라도: 짝지어진 추출 사실 가운데 하나라도 N장까지 안 경우. 아는 인물 정밀도·재현율과 같은 기준이다.

실행(worktree 루트): PYTHONPATH=src uv run --no-sync python docs/ontology-extract/compare_runs.py <기준 실행> <비교 실행>
"""

import json
import sys
from collections import Counter
from pathlib import Path

from sfqa.gold import load_texts
from sfqa.onto_eval import id_map
from sfqa.ontology import SYNTHETIC_ROOT, Ontology, from_synthetic
from sfqa.works import load_work

NEVER = 10**6


def load(run: str, gold: Ontology) -> tuple[str, dict, dict[str, list[dict[str, int]]]]:
    """(이름, evaluation, 정답 사실 id → 짝지어진 추출 사실마다 {정답 인물: 아는 장})"""
    d = Path(run)
    onto = Ontology.model_validate_json((d / "ontology.json").read_text(encoding="utf-8"))
    judgments = json.loads((d / "judgments.json").read_text(encoding="utf-8"))
    ev = json.loads((d / "evaluation.json").read_text(encoding="utf-8"))
    m, facts = id_map(onto, gold), {f.id: f for f in onto.facts}
    parts = {gid: [{c.id: 0 for c in gold.characters} if facts[e].common
                   else {m[k.character]: k.from_ch for k in facts[e].known_by if k.character in m} for e in j["ids"]]
             for gid, j in judgments["match"].items()}
    return d.name, ev, parts


def pair_sets(gold: Ontology, parts: dict, upto: int, comparable: set[str]) -> tuple[dict, set, set, set]:
    """score와 같은 계산: 어느 부분이든 아는 인물(가장 이른 장)로 맞힘·틀림·놓침 쌍을 만든다."""
    early, tp, fp, fn = {}, set(), set(), set()
    for g in (f for f in gold.facts if f.reveal_ch <= upto and parts.get(f.id)):
        gk = {k.character for k in g.known_by if k.from_ch <= upto and k.character in comparable}
        ek: dict[str, int] = {}
        for p in parts[g.id]:
            for who, s in p.items():
                if who in comparable:
                    ek[who] = min(ek.get(who, s), s)
        early.update({(g.id, who): s for who, s in ek.items()})
        tp |= {(g.id, w) for w in ek if w in gk}
        fp |= {(g.id, w) for w in ek if w not in gk}
        fn |= {(g.id, w) for w in gk if w not in ek}
    return early, tp, fp, fn


def view_cells(gold: Ontology, parts: dict, upto: int, first: dict[str, int], whole: bool = True) -> Counter:
    cells = Counter()
    for g in (f for f in gold.facts if f.reveal_ch <= upto):
        gs = {k.character: k.from_ch for k in g.known_by}
        ps = parts.get(g.id) or []
        if whole:
            es = {w: max(p[w] for p in ps) for w in set.intersection(*(set(p) for p in ps))} if ps else {}
        else:
            es = {w: min(p[w] for p in ps if w in p) for w in set().union(*ps)}
        for n in range(g.reveal_ch, upto + 1):
            for who in (c for c, fc in first.items() if fc <= n):
                gk, ek = gs.get(who, NEVER) <= n, es.get(who, NEVER) <= n
                if gk and ek:
                    cells["맞게 앎"] += 1
                elif ek:
                    cells["더 앎(정답은 뒤에 앎)" if gs.get(who, NEVER) <= upto else "더 앎(정답은 범위 끝까지 모름)"] += 1
                elif gk:
                    cells["모름"] += 1
    return cells


def main(run_a: str, run_b: str) -> None:
    gold = from_synthetic("wn01-rofan")
    golds = {g.id: g for g in gold.facts}
    (na, ea, pa), (nb, eb, pb) = load(run_a, gold), load(run_b, gold)
    upto, comparable = ea["upto"], set(ea["comparable"])
    assert eb["upto"] == upto and set(eb["comparable"]) == comparable
    texts = load_texts(load_work("wn01-rofan", SYNTHETIC_ROOT))
    first = {c.id: next(n for n in range(1, upto + 1) if any(f in texts[n] for f in c.forms))
             for c in gold.characters if c.id in comparable}
    (sa, ta, fa, ma), (sb, tb, fb, mb) = pair_sets(gold, pa, upto, comparable), pair_sets(gold, pb, upto, comparable)
    for name, ev, t, f, m in ((na, ea, ta, fa, ma), (nb, eb, tb, fb, mb)):
        assert (len(t), len(f), len(m)) == (ev["tp"], ev["fp"], ev["fn"]), f"{name}: score와 다르다"
    print(f"기준 {na} / 비교 {nb} ({upto}장까지, 정답 사실 {ea['gold_facts']}개)")
    print(f"아는 인물 쌍 맞힘 {len(ta)} → {len(tb)}, 틀림 {len(fa)} → {len(fb)}, 놓침 {len(ma)} → {len(mb)}")

    def show(title: str, pairs: set) -> None:
        print(f"{title} {len(pairs)}쌍")
        for gid, who in sorted(pairs):
            gs = next((k.from_ch for k in golds[gid].known_by if k.character == who), None)
            print(f"  {gid}[{golds[gid].reveal_ch}장] {who} 정답 {gs}장, 기준 {sa.get((gid, who))}장, 비교 {sb.get((gid, who))}장:"
                  f" {golds[gid].statement[:50]}")

    show("비교에서 잃은 정답 쌍", ta - tb)
    show("비교에서 새로 맞힌 쌍", tb - ta)
    show("비교에서만 틀린 쌍", fb - fa)
    print(f"기준에서만 틀린 쌍 {len(fa - fb)}쌍, 둘 다 틀린 쌍 {len(fa & fb)}쌍")
    print("채팅 기본값 보기(인물 시점 = 독자 위치, 1장부터 칸 합)")
    for whole, label in ((True, "모두"), (False, "하나라도")):
        for name, parts in ((na, pa), (nb, pb)):
            print(f"  [{label}] {name}: {dict(view_cells(gold, parts, upto, first, whole))}")


if __name__ == "__main__":
    main(*sys.argv[1:])
