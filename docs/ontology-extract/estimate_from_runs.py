"""실제 추출 실행을 바탕으로 다른 작품의 추출·평가 비용을 어림한다. 모델을 부르지 않는다(docs/ontology-extract.md 17절, 20절).

- 장별 출력: 기준 실행들의 장별 실제 출력을 장별 정답 기준 출력(글자)과 정답의 '공개 뒤에 알게 됨' 수에 맞춘 직선으로 낸다.
  --simple이면 정답 기준 출력만 쓴다(17절 방식).
- 추출 비용: 출력은 위 직선, 입력은 본문 토큰 × 기준의 입력/본문 비, 캐시 읽기·쓰기는 출력 토큰 × 기준의 캐시 비용/출력 비.
  - 기준의 비용은 성공한 시도만 쓰고, 캐시가 식어 기록을 다시 쓴 호출은 이어서 돌렸을 때 값으로 고친다(12절, 18절).
  - 범위는 기준 실행들에서 직선이 낸 합계의 오차 가운데 큰 쪽(최소 5%)으로 잡는다.
- 출력 잘림: 장별 예측에 잔차 90% 값을 더해 출력 상한(MAX_TOKENS)을 넘는 장은, 잘린 시도 한 번 값을 높은 쪽에 더한다.
- 평가: 판정 카세트 전체의 건당 실제 비용 × (정답 사실 + 근거 표본 + 둘째 판정 어림). 판정 카세트가 늘면 건당 비용도 조금 바뀐다.
  보내기 전 어림(onto_eval._guess)은 기준 실행의 요청을 다시 만들어 낸다. 판정 상한은 이 어림보다 커야 보낸다.
- 추출 상한: 가운데 × 1.1 + 상한 장치 여유(가장 비싼 장 어림 × 1.2).
- 17절의 가운데 값은 `--bases wn01-opus-50 --simple`로 다시 낼 수 있다.

실행(worktree 루트): PYTHONPATH=src uv run --no-sync python docs/ontology-extract/estimate_from_runs.py wn03-muhyeop wn04-romance
"""

import argparse
import itertools
import json
import statistics as st
from collections import Counter
from pathlib import Path

from sfqa.gold import load_texts
from sfqa.llm import _load_cassette
from sfqa.onto_eval import (
    SUPPORT_SAMPLE,
    _guess,
    batch_cost,
    candidates,
    match_request,
    support_request,
)
from sfqa.onto_extract import MAX_TOKENS, PRICES, RUNS
from sfqa.ontology import SYNTHETIC_ROOT, Ontology, from_synthetic
from sfqa.works import load_work

HERE = Path(__file__).parent
MODEL = "claude-opus-5-5"
SECOND_PASS = 0.05  # 둘째 판정을 받는 정답 비율(어림)
TOKENS = json.loads((HERE / "tokens-20261023.json").read_text(encoding="utf-8"))["works"]


def learned_per_chapter(work: str) -> Counter:
    """정답에서 장마다 '공개 뒤에 알게 됨'(아는 장 > 공개 장) 수."""
    return Counter(k.from_ch for f in from_synthetic(work).facts for k in f.known_by if k.from_ch > f.reveal_ch)


def base_chapters(run: str) -> tuple[str, list[dict]]:
    """(작품 id, 장마다 성공한 시도의 토큰 수). 캐시가 식어 기록을 다시 쓴 호출은 이어서 돌렸을 때 값으로 고친다."""
    report = json.loads((RUNS / run / "report.json").read_text(encoding="utf-8"))
    tries = [dict([t for t in c["tries"] if not t["error"]][-1]) for c in report["chapters"]]
    for prev, t in itertools.pairwise(tries):
        if t["cache_read"] == 0 and t["cache_write"] > 10 * prev["cache_write"]:
            warm = prev["cache_read"] + prev["cache_write"]
            t.update(cache_read=warm, cache_write=t["cache_write"] - warm)
    return report["work"], tries


def fit(xs: list[list[float]], ys: list[float]) -> list[float]:
    """최소제곱 계수(첫 칸은 절편)."""
    n = len(xs[0])
    a = [[sum(x[i] * x[j] for x in xs) for j in range(n)] + [sum(x[i] * y for x, y in zip(xs, ys))] for i in range(n)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(a[r][i]))
        a[i], a[p] = a[p], a[i]
        for r in range(n):
            if r != i:
                a[r] = [u - a[r][i] / a[i][i] * v for u, v in zip(a[r], a[i])]
    return [a[i][n] / a[i][i] for i in range(n)]


def features(work: str, simple: bool) -> list[list[float]]:
    learn = learned_per_chapter(work)
    return [[1.0, ideal] + ([] if simple else [learn.get(c, 0)])
            for c, ideal in enumerate(TOKENS[work]["chars"]["output"], 1)]


def judge_per_item(bases: list[str]) -> tuple[float, float, float, float]:
    """(건당 실제 비용, 같은 사실 판정·근거 판정·둘째 판정 건당 어림) — 기준 실행의 요청을 다시 만들어 센다."""
    rows = list(_load_cassette(RUNS / "judge-cassette.jsonl").values())
    actual = sum(batch_cost(r) for r in rows) / len(rows)
    match, support, near = [], [], []
    for run in bases:
        work = json.loads((RUNS / run / "report.json").read_text(encoding="utf-8"))["work"]
        gold, texts = from_synthetic(work), load_texts(load_work(work, SYNTHETIC_ROOT))
        onto = Ontology.model_validate_json((RUNS / run / "ontology.json").read_text(encoding="utf-8"))
        facts = {f.id: f for f in onto.facts}
        cands = candidates(gold.facts, onto.facts)
        match += [_guess(match_request(g, [facts[x] for x in cands[g.id]])) for g in gold.facts]
        sample = json.loads((RUNS / run / "judgments.json").read_text(encoding="utf-8"))["support"]
        support += [_guess(support_request(facts[fid], texts)) for fid in sample if fid in facts]
        near += [_guess(match_request(g, [f for f in onto.facts if abs(f.reveal_ch - g.reveal_ch) <= 1])) for g in gold.facts]
    return actual, st.mean(match), st.mean(support), st.mean(near)


def main(targets: list[str], bases: list[str], simple: bool) -> None:
    i, o, cr, cw = PRICES[MODEL]
    xs, ys, text, inp, cache, out = [], [], 0, 0, 0.0, 0
    per_base = []
    for run in bases:
        work, tries = base_chapters(run)
        xs += features(work, simple)
        ys += [t["output"] for t in tries]
        text += TOKENS[work]["tokens"][MODEL]["chapters_total"]
        inp += sum(t["input"] for t in tries)
        cache += sum(t["cache_read"] * cr + t["cache_write"] * cw for t in tries) / 1e6
        out += sum(t["output"] for t in tries)
        per_base.append((run, work, sum(t["output"] for t in tries)))
    coef = fit(xs, ys)
    res = sorted(y - sum(c * x for c, x in zip(coef, row)) for row, y in zip(xs, ys))
    res90 = res[int(len(res) * 0.9)]
    names = ["절편", "정답 출력 글자"] + ([] if simple else ["공개 뒤 앎"])
    print("장별 출력 = " + " + ".join(f"{c:.3f}×{n}" if n != "절편" else f"{c:.0f}" for c, n in zip(coef, names))
          + f" | 기준 {len(ys)}장, 잔차 90% {res90:+,.0f}토큰")
    errors = []
    for run, work, actual in per_base:
        pred = sum(sum(c * x for c, x in zip(coef, row)) for row in features(work, simple))
        errors.append(abs(pred / actual - 1))
        print(f"  기준 {run}: 실제 출력 {actual:,} / 직선 합계 {pred:,.0f} ({pred / actual - 1:+.1%})")
    spread = max(0.05, *errors)
    in_rate, cache_rate = inp / text * i / 1e6, cache / out  # 본문 토큰당 입력비, 출력 토큰당 캐시비
    actual_item, g_match, g_support, g_near = judge_per_item(bases)
    for target in targets:
        preds = [sum(c * x for c, x in zip(coef, row)) for row in features(target, simple)]
        output = sum(preds)
        mid = output * (o / 1e6 + cache_rate) + TOKENS[target]["tokens"][MODEL]["chapters_total"] * in_rate
        risky = [(c, round(p)) for c, p in enumerate(preds, 1) if p + res90 > MAX_TOKENS]
        cut_cost = (MAX_TOKENS * o + 5300 * i + 60_000 * cr) / 1e6  # 잘린 시도 한 번(출력 상한까지, 기록 6만 토큰 어림)
        lo, hi = mid * (1 - spread), mid * (1 + spread) + cut_cost * len(risky)
        priciest = max(p * (o / 1e6 + cache_rate) for p in preds) + 5300 * in_rate  # 가장 비싼 장 어림(본문 약 5,300토큰)
        golds = len(from_synthetic(target).facts)
        second = round(golds * SECOND_PASS)
        items = golds + SUPPORT_SAMPLE + second
        guess = golds * g_match + SUPPORT_SAMPLE * g_support + second * g_near
        print(f"\n{target}: 출력 예측 {output:,.0f}토큰 | 추출 낮음 ${lo:.2f}, 가운데 ${mid:.2f}, 높음 ${hi:.2f}"
              f" | 출력이 잘릴 수 있는 장 {risky or '없음'}")
        print(f"  평가: 정답 {golds} + 근거 표본 {SUPPORT_SAMPLE} + 둘째 판정 {second} = {items}건 × ${actual_item:.5f} = "
              f"${items * actual_item:.2f} | 보내기 전 어림 ${guess:.2f}")
        cap = mid * 1.1 + max(priciest, cut_cost * 2 if risky else 0) * 1.2
        print(f"  추출 상한 권장 ${cap:.2f}(가장 비싼 장 어림 ${priciest:.2f}) | 합계 가운데 ${mid + items * actual_item:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="실제 실행을 바탕으로 추출·평가 비용을 어림한다(모델 호출 없음)")
    parser.add_argument("targets", nargs="+")
    parser.add_argument("--bases", default="wn01-opus-50,wn02-opus-50")
    parser.add_argument("--simple", action="store_true", help="정답 기준 출력만으로 직선을 맞춘다(17절 방식)")
    args = parser.parse_args()
    main(args.targets, args.bases.split(","), args.simple)
