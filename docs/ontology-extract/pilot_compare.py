"""시험 실행(wn01 1~4장) 결과를 설정집 정답과 나란히 놓는다. 모델을 부르지 않는다. 판정은 사람이 읽고 한다.

- 정답 사실마다 추출 사실 후보 3개(글자 두 개씩 겹침 비율 순)를 모델별로 보인다.
- 추출 인물 id를 정답 인물 id로 잇는다(이름이 같거나 정답 별칭과 같으면). 아는 인물 비교에 쓴다.
- 추출 사실이 정답에서 뒤 장(5장 이후)에 공개되는 사실과 더 닮았으면 '앞당김 의심'으로 따로 보인다.
- --matches에 판정 파일(pilot-matches.json)을 주면 실행마다 수치를 낸다.
  - 아는 인물은 1~4장 본문에 이름이 나온 정답 인물로 양쪽을 똑같이 제한한다. 정답에 없는 인물은 세지 않는다.
  - 추출 사실이 상식(common)이면 그 인물들이 모두 0장부터 안다고 본다.
  - '공개 뒤에 알게 됨'은 정답에서 아는 장이 공개 장보다 늦은 경우다(learned가 잡아야 하는 것).

실행(worktree 루트): PYTHONPATH=src uv run --no-sync python docs/ontology-extract/pilot_compare.py <결과 폴더>... [--matches 파일] [--upto N]
"""

import argparse
import json
from pathlib import Path

from sfqa.gold import load_texts
from sfqa.ontology import SYNTHETIC_ROOT, Ontology, from_synthetic
from sfqa.verify import compact
from sfqa.works import load_work

UPTO = 4


def grams(text: str) -> set[str]:
    t = compact(text)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def sim(a: str, b: str) -> float:
    x, y = grams(a), grams(b)
    return len(x & y) / max(1, len(x | y))


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


def side_by_side(gold: Ontology, runs: dict[str, Ontology], maps: dict[str, dict]) -> None:
    gfacts = [f for f in gold.facts if f.reveal_ch <= UPTO]
    print(f"정답 사실(1~{UPTO}장 공개): {len(gfacts)}개 | " + ", ".join(f"{n} 추출 {len(o.facts)}개" for n, o in runs.items()))
    for name, o in runs.items():
        unmapped = [c.id for c in o.characters if c.id not in maps[name]]
        print(f"[{name}] 인물 {len(o.characters)}명, 정답에 이어지지 않은 id: {unmapped}")
    print("\n정답 인물(1~4장 첫 등장):", [(c.id, c.first_ch) for c in gold.characters if c.first_ch <= UPTO])
    for name, o in runs.items():
        print(f"  {name}:", [(maps[name].get(c.id, c.id), c.first_ch, c.mention_ch) for c in o.characters])

    for g in gfacts:
        gk = {k.character: k.from_ch for k in g.known_by if k.from_ch <= UPTO}
        print(f"\n■ {g.id} [{g.reveal_ch}장] {g.statement}\n   정답 앎: {gk}")
        for name, o in runs.items():
            for s, f in sorted(((sim(g.statement, f.statement), f) for f in o.facts), key=lambda x: -x[0])[:3]:
                fk = "상식" if f.common else {maps[name].get(k.character, k.character): k.from_ch for k in f.known_by}
                print(f"   {name} {s:.2f} {f.id} [{f.reveal_ch}장] {f.statement}\n        앎: {fk}")

    print("\n\n앞당김 의심(추출 사실이 자기 장보다 뒤에 공개되는 정답 사실을 더 닮음, 겹침 0.35 이상):")
    for name, o in runs.items():
        for f in o.facts:
            now = [g for g in gold.facts if g.reveal_ch <= f.reveal_ch]
            later = [g for g in gold.facts if g.reveal_ch > f.reveal_ch]
            best_now = max((sim(f.statement, g.statement) for g in now), default=0)
            s, g = max(((sim(f.statement, g.statement), g) for g in later), key=lambda x: x[0])
            if s >= 0.35 and s > best_now:
                print(f"  {name} {f.id} [{f.reveal_ch}장] {f.statement}\n     ↔ {g.id} [{g.reveal_ch}장 공개] {g.statement} ({s:.2f})")


def scores(gold: Ontology, onto: Ontology, run: dict, comparable: set[str]) -> dict:
    m, facts = id_map(onto, gold), {f.id: f for f in onto.facts}
    out = {"full": 0, "partial": 0, "missed": 0, "reveal_early": 0, "reveal_same": 0, "reveal_late": 0,
           "tp": 0, "fp": 0, "fn": 0, "since_early": [], "since_early_partial": [], "since_late": 0, "after_reveal": 0,
           "after_reveal_hit": 0}
    for g in (f for f in gold.facts if f.reveal_ch <= UPTO):
        eids = run["matches"].get(g.id, [])
        if not eids:
            out["missed"] += 1
            continue
        partial = g.id in run["partial"]
        out["partial" if partial else "full"] += 1
        first = min(facts[e].reveal_ch for e in eids)
        out["reveal_early" if first < g.reveal_ch else "reveal_same" if first == g.reveal_ch else "reveal_late"] += 1
        gk = {k.character: k.from_ch for k in g.known_by if k.from_ch <= UPTO and k.character in comparable}
        parts = []
        for e in eids:
            known = ({c: 0 for c in comparable} if facts[e].common
                     else {m.get(k.character): k.from_ch for k in facts[e].known_by})
            parts.append({who: s for who, s in known.items() if who in comparable and s <= UPTO})  # 범위 뒤 앎은 뺀다
        ek: dict[str, int] = {}  # 어느 부분이든 아는 인물(정밀도·재현율)
        for part in parts:
            for who, s in part.items():
                ek[who] = min(ek.get(who, s), s)
        for who in ek:
            out["tp" if who in gk else "fp"] += 1
        out["fn"] += sum(1 for who in gk if who not in ek)
        # 아는 장: 모든 부분을 알게 된 가장 늦은 장(sfqa.onto_eval.score와 같은 규칙). 부분 짝은 따로 센다.
        whole = {who: max(p[who] for p in parts) for who in set.intersection(*(set(p) for p in parts))}
        for who, s in whole.items():
            if who in gk and s < gk[who]:
                out["since_early_partial" if partial else "since_early"].append(f"{g.id}:{who} {s}<{gk[who]}")
            elif who in gk and s > gk[who] and not partial:
                out["since_late"] += 1
        after = [who for who, s in gk.items() if s > g.reveal_ch]
        out["after_reveal"] += len(after)
        out["after_reveal_hit"] += sum(1 for who in after if who in ek)
    return out


def score_table(gold: Ontology, runs: dict[str, Ontology], matches: dict) -> None:
    texts = load_texts(load_work("wn01-rofan", SYNTHETIC_ROOT))
    seen = "".join(texts[c] for c in range(1, UPTO + 1))
    comparable = {c.id for c in gold.characters if any(f in seen for f in c.forms)}
    print(f"\n아는 인물 비교 범위(1~{UPTO}장 본문에 이름이 나온 정답 인물): {sorted(comparable)}")
    for name, o in runs.items():
        if name not in matches["runs"]:
            continue
        s = scores(gold, o, matches["runs"][name], comparable)
        p, r = s["tp"] / max(1, s["tp"] + s["fp"]), s["tp"] / max(1, s["tp"] + s["fn"])
        print(f"[{name}] 사실 완전 {s['full']}·일부 {s['partial']}·놓침 {s['missed']} | 공개 장 같음 {s['reveal_same']}·"
              f"앞당김 {s['reveal_early']}·늦음 {s['reveal_late']} | 아는 인물 정밀도 {p:.2f}({s['tp']}/{s['tp'] + s['fp']}) "
              f"재현율 {r:.2f}({s['tp']}/{s['tp'] + s['fn']}) | 공개 뒤에 알게 됨 {s['after_reveal_hit']}/{s['after_reveal']} | "
              f"아는 장 앞당김 {len(s['since_early'])} {s['since_early']}(부분 짝 {len(s['since_early_partial'])} 따로)·"
              f"늦음 {s['since_late']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+")
    parser.add_argument("--matches", help="판정 파일(pilot-matches.json). 주면 수치만 낸다")
    parser.add_argument("--upto", type=int, default=UPTO, help="비교할 마지막 장(기본 4)")
    args = parser.parse_args()
    globals()["UPTO"] = args.upto
    gold = from_synthetic("wn01-rofan")
    runs = {Path(d).name: Ontology.model_validate_json((Path(d) / "ontology.json").read_text(encoding="utf-8")) for d in args.dirs}
    if args.matches:
        score_table(gold, runs, json.loads(Path(args.matches).read_text(encoding="utf-8")))
    else:
        side_by_side(gold, runs, {name: id_map(o, gold) for name, o in runs.items()})


if __name__ == "__main__":
    main()
