"""본문 추출기 비용 추정(docs/ontology-extract.md). 추출기의 지시문·스키마(sfqa.onto_extract)와, 설정집 정답으로 만든
'이상적 출력'을 쓴다. tokens-20261023.json은 모듈을 만들기 전 초안 지시문으로 잰 값이다(다시 재면 조금 달라진다).

- count: count_tokens(무료)로 토큰을 세어 tokens-<날짜>.json에 쓴다. 키는 프로젝트 .env에서 읽고 출력하지 않는다.
- cost: 그 JSON으로 모델·캐시 방식별 비용을 계산한다(모델 호출 없음).

실행(worktree 루트): PYTHONPATH=src uv run --no-sync python docs/ontology-extract/estimate_cost.py count|cost
"""

import json
import sys
from pathlib import Path

from sfqa.onto_eval import JUDGE_MATCH, JUDGE_SUPPORT, MATCH_SCHEMA, SUPPORT_SCHEMA
from sfqa.onto_extract import CATEGORIES, PRICES, SCHEMA, SYSTEM

HERE = Path(__file__).parent
TOKENS = HERE / "tokens-20261023.json"
MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
WRAP = 30  # 장마다 본문을 감싸는 말(추정)
# 평가 판정 프롬프트는 sfqa.onto_eval에 있다. 판정 출력 토큰은 가정이다.
JUDGE_OUT = 60  # 판정 출력 JSON 토큰(가정)


def ideal(onto, texts, c: int) -> tuple[str, str]:
    """정답으로 만든 c장 출력(JSON)과, 그 출력을 기록에 덧붙일 블록(텍스트). learned의 quote는 60자로 어림한다."""
    lines = texts[c].splitlines()
    chars = [ch for ch in onto.characters if ch.first_ch == c]
    new_alias = [(ch.id, a.form) for ch in onto.characters for a in ch.aliases if a.from_ch == c and c > ch.first_ch]
    facts = [f for f in onto.facts if f.reveal_ch == c]
    learned = [(f, k.character) for f in onto.facts for k in f.known_by if k.from_ch == c and f.reveal_ch < c]
    scene = next(s for s in onto.scenes if s.chapter == c)

    def quote(f):
        return lines[f.evidence[0].lines[0] - 1][:150] if f.evidence else f.statement[:150]

    out = {
        "characters": [{"id": ch.id, "name": ch.name, "appeared": True,
                        "aliases": [a.form for a in ch.aliases if a.from_ch == ch.first_ch]} for ch in chars],
        "aliases": [{"character": cid, "alias": form} for cid, form in new_alias],
        "merges": [],
        "facts": [{"statement": f.statement, "category": f.category if f.category in CATEGORIES else "event",
                   "quote": quote(f), "knowers": [{"character": k.character, "since": k.from_ch}
                                                  for k in f.known_by if k.from_ch <= c]} for f in facts],
        "learned": [{"fact": f.id, "character": who, "quote": texts[c][:60]} for f, who in learned],
        "scene": {"pov": scene.pov or None, "location": scene.location, "present": scene.present},
    }
    block = [f"[{c}장]"]
    block += [f"새 인물: {ch.id} {ch.name}" + (f" (별칭: {', '.join(x['aliases'])})" if x["aliases"] else "")
              for ch, x in zip(chars, out["characters"])]
    block += [f"새 별칭: {cid} = {form}" for cid, form in new_alias]
    block += [f"새 사실: {f.id} {f.statement} | 앎: " + " ".join(f"{k.character}({k.from_ch})" for k in f.known_by if k.from_ch <= c)
              for f in facts]
    block += [f"새로 앎: {f.id} <- {who}" for f, who in learned]
    block.append(f"장면: 시점 {scene.pov}, {scene.location}, 등장 {' '.join(scene.present)}")
    return json.dumps(out, ensure_ascii=False), "\n".join(block)


def count() -> None:
    from sfqa.claude import load_key
    from sfqa.gold import load_texts
    from sfqa.ontology import SYNTHETIC_ROOT, from_synthetic, synthetic_ids
    from sfqa.works import load_work

    if not load_key():
        sys.exit("ANTHROPIC_API_KEY가 없다")
    import anthropic

    client = anthropic.Anthropic(max_retries=3)

    def n(model: str, text: str, system: str | None = None, fmt: bool | dict = False) -> int:
        kw = {"system": system} if system else {}
        if fmt:
            kw["output_config"] = {"format": {"type": "json_schema", "schema": SCHEMA if fmt is True else fmt}}
        return client.messages.count_tokens(model=model, messages=[{"role": "user", "content": text}], **kw).input_tokens

    # 판정 예시: wn01 f034(정답)와 후보 8개, 인용문과 앞뒤 본문
    pilot = from_synthetic("wn01-rofan")
    f034 = next(f for f in pilot.facts if f.id == "f034")
    others = [f.statement for f in pilot.facts if f.reveal_ch in (9, 10, 11) and f.id != "f034"][:7]
    match_user = f"정답: {f034.statement}\n후보:\n" + "\n".join(f"{i}. {s}" for i, s in enumerate([f034.statement, *others], 1))
    text10 = load_texts(load_work("wn01-rofan", SYNTHETIC_ROOT))[10]
    support_user = f"사실: {f034.statement}\n인용: {text10.splitlines()[166]}\n앞뒤 본문:\n{text10[5800:6200] if len(text10) > 6200 else text10[-400:]}"

    report = {"models": {}, "works": {}}
    for m in MODELS:
        base = n(m, "가")
        report["models"][m] = {"system": n(m, "가", system=SYSTEM) - base,
                               "schema_overhead": n(m, "가", system=SYSTEM, fmt=True) - n(m, "가", system=SYSTEM), "base": base,
                               "judge_match": n(m, match_user, system=JUDGE_MATCH, fmt=MATCH_SCHEMA),
                               "judge_support": n(m, support_user, system=JUDGE_SUPPORT, fmt=SUPPORT_SCHEMA)}
    for wid in synthetic_ids():
        onto = from_synthetic(wid)
        texts = load_texts(load_work(wid, SYNTHETIC_ROOT))
        outs, blocks = zip(*(ideal(onto, texts, c) for c in range(1, onto.chapters + 1)))
        body = [texts[c] for c in range(1, onto.chapters + 1)]
        report["works"][wid] = {
            "chars": {"chapter": [len(t) for t in body], "output": [len(o) for o in outs], "block": [len(b) for b in blocks]},
            "tokens": {m: {"chapters_total": n(m, "\n\n".join(body)), "outputs_total": n(m, "\n".join(outs)),
                           "blocks_total": n(m, "\n".join(blocks))} for m in MODELS},
        }
    TOKENS.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"썼다: {TOKENS}")


def profile(r: dict, wid: str, model: str, upto: int | None = None) -> dict:
    """장 순서대로 부를 때의 토큰. 기록 끝에 캐시 지점을 두면 앞 호출의 접두부를 읽고, 새 블록만 쓴다."""
    m, w = r["models"][model], r["works"][wid]
    t, ch = w["tokens"][model], w["chars"]
    static = m["system"] + m["schema_overhead"] + m["base"]
    chapters = len(ch["chapter"]) if upto is None else upto
    x = [t["chapters_total"] * c / sum(ch["chapter"]) + WRAP for c in ch["chapter"]]
    o = [t["outputs_total"] * c / sum(ch["output"]) for c in ch["output"]]
    b = [t["blocks_total"] * c / sum(ch["block"]) for c in ch["block"]]
    p = {"read": 0.0, "write": 0.0, "plain": 0.0, "out": 0.0, "nocache": 0.0, "calls": chapters}
    record = 0.0  # 앞 장까지 쌓인 기록
    for i in range(chapters):
        if i == 0:
            p["write"] += static
        else:
            p["read"] += static + record - b[i - 1]
            p["write"] += b[i - 1]
        p["plain"] += x[i]
        p["out"] += o[i]
        p["nocache"] += static + record + x[i]
        record += b[i]
    return p


def cost(p: dict, model: str, thinking: float = 0.0, cache: bool = True) -> tuple[float, float, float]:
    """(합계, 입력, 출력) 달러. thinking은 출력 JSON 대비 thinking 토큰 비율(출력 요금)."""
    i, o, cr, cw = PRICES[model]
    inp = (p["read"] * cr + p["write"] * cw + p["plain"] * i) if cache else p["nocache"] * i
    out = p["out"] * (1 + thinking) * o
    return (inp + out) / 1e6, inp / 1e6, out / 1e6


def report() -> None:
    r = json.loads(TOKENS.read_text(encoding="utf-8"))
    works = list(r["works"])
    settings = [("Opus 5.5, thinking 0.5배", "claude-opus-5-5", 0.5), ("Opus 5.5, thinking 1배", "claude-opus-5-5", 1.0),
                ("Opus 5.5, thinking 2배", "claude-opus-5-5", 2.0), ("Sonnet 5.5, thinking 끔", "claude-sonnet-5-5", 0.0),
                ("Sonnet 5.5, thinking 1배", "claude-sonnet-5-5", 1.0), ("Haiku 5.5, thinking 끔", "claude-haiku-5-5", 0.0)]
    for label, sel in (("wn01 1~5장", [("wn01-rofan", 5)]), ("wn01 50장", [("wn01-rofan", None)]),
                       ("4편 200장", [(w, None) for w in works])):
        print(f"\n[{label}]")
        for name, model, think in settings:
            ps = [profile(r, w, model, u) for w, u in sel]
            tot = {k: sum(p[k] for p in ps) for k in ps[0]}
            total, inp, out = cost(tot, model, think)
            print(f"  {name}: ${total:.2f} (입력 {inp:.2f} + 출력 {out:.2f})")
        ps = [profile(r, w, "claude-opus-5-5", u) for w, u in sel]
        tot = {k: sum(p[k] for p in ps) for k in ps[0]}
        print(f"  참고: Opus 5.5 thinking 1배를 캐시 없이: ${cost(tot, 'claude-opus-5-5', 1.0, cache=False)[0]:.2f}")
        print(f"  토큰: 호출 {tot['calls']}, 캐시 읽기 {tot['read']/1e3:.0f}K, 쓰기 {tot['write']/1e3:.0f}K, "
              f"일반 입력 {tot['plain']/1e3:.0f}K, 출력 JSON {tot['out']/1e3:.0f}K")
    # 판정: 정답 사실마다 같은 사실 고르기 1번, 추출 사실마다 근거 판정 1번(추출 사실 수는 정답 수와 같다고 둔다). Batch는 반값.
    gold = {"wn01-rofan": 189, "wn02-hyeonpan": 233, "wn03-muhyeop": 243, "wn04-romance": 220}

    def judge(model: str, facts: int) -> float:
        m, (i, o, _, _) = r["models"][model], PRICES[model]
        return (facts * (m["judge_match"] + m["judge_support"]) * i + facts * 2 * JUDGE_OUT * o) / 1e6 / 2

    def extract(model: str, thinking: float, sel: list) -> float:
        ps = [profile(r, w, model, u) for w, u in sel]
        return cost({k: sum(p[k] for p in ps) for k in ps[0]}, model, thinking)[0]

    print("\n[평가 판정, Batch 반값, thinking 끔(Opus 5.5는 끌 수 없어 thinking 토큰이 빠진 값)]")
    for label, facts in (("wn01", gold["wn01-rofan"]), ("4편", sum(gold.values()))):
        for model in MODELS:
            print(f"  {label} {model}: ${judge(model, facts):.2f} (판정 {facts * 2}번)")
    # 단계 합계(20% 여유). 판정 모델은 추출과 다르게: Opus 추출 → Sonnet 판정, Sonnet 추출 → Opus 판정
    opus, sonnet = "claude-opus-5-5", "claude-sonnet-5-5"
    pilot, rest = [("wn01-rofan", 5)], [(w, None) for w in works if w != "wn01-rofan"]
    print("\n[단계 합계, 20% 여유]")
    print(f"  시험(wn01 1~5장, Opus와 Sonnet 한 번씩): ${(extract(opus, .5, pilot) + extract(sonnet, 0, pilot)) * 1.2:.2f}"
          f"~{(extract(opus, 2, pilot) + extract(sonnet, 0, pilot)) * 1.2:.2f}")
    for label, sel, facts in (("wn01 50장과 평가", [("wn01-rofan", None)], gold["wn01-rofan"]),
                              ("나머지 3편과 평가", rest, sum(gold[w] for w, _ in rest))):
        low = (extract(sonnet, 0, sel) + judge(opus, facts)) * 1.2
        high = [(extract(opus, t, sel) + judge(sonnet, facts)) * 1.2 for t in (.5, 2)]
        print(f"  {label}: Sonnet 추출 ${low:.2f} / Opus 추출 ${high[0]:.2f}~{high[1]:.2f}")


if __name__ == "__main__":
    {"count": count, "cost": report}[sys.argv[1] if len(sys.argv) > 1 else "cost"]()
