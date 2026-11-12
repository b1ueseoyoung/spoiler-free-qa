"""인물 채팅 스포일러 시험: 정해 둔 질문(eval/chat-probes.jsonl)을 인물에게 묻고 답과 검사 결과를 남긴다.

- kind: hidden(미공개 사실을 그 사실을 아는 인물에게 캐묻는다), reader_only(독자만 아는 사실), ending(결말), injection(지시 무시 요구),
  fourth_wall(소설이라고 말하기), meta(공개 전부터 안다는 점이 새 정보인 경우), benign(평범한 대화. 오탐을 잰다), romance(호감 설정),
  minor(미성년 차단. 모델을 부르지 않는다), absent(이용할 수 없는 인물. 모델을 부르지 않는다).
- must_not: 최종 답에 있으면 샌 것으로 보는 글자. 코드로 볼 수 있는 최소한이고, 판정은 사람이 답을 읽어서 한다.
- 응답은 카세트(eval/cassettes/chat-probes.jsonl)에 남아 다시 돌리면 돈이 들지 않는다. 프롬프트를 고치면 새로 부른다.

실행:
  uv run python -m sfqa.chat_probe --estimate
  uv run python -m sfqa.chat_probe --max-calls 30 --max-usd 0.7
  uv run python -m sfqa.chat_probe --replay
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from sfqa.chat import ChatEnv, NotAvailable, RomanceRefused, Turn, build_request, load_env, respond
from sfqa.config import load_settings
from sfqa.llm import LLMResponse, ReplayClient
from sfqa.paths import ROOT

CASES = ROOT / "eval" / "chat-probes.jsonl"
CASSETTE = ROOT / "eval" / "cassettes" / "chat-probes.jsonl"
SETTINGS_ENV = {"LLM_PROVIDER": "anthropic"}
TOKENS_PER_CHAR = 1.07  # docs/ontology-extract.md 실측(본문 기준)
OUTPUT_TOKENS = 150  # 어림. JSON 포함 답 한 번


def load_cases(path: Path = CASES) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _history(case: dict) -> list[Turn]:
    return [Turn(t["role"], t["text"]) for t in case.get("history", [])]


def _envs(cases: list[dict]) -> dict[str, ChatEnv]:
    return {w: load_env(w) for w in {c["work"] for c in cases}}


class CappedClient:
    """상한을 넘는 호출을 막는다. 카세트에 있는 요청은 호출 수에도 비용에도 세지 않는다."""

    def __init__(self, inner, max_calls: int, max_usd: float):
        self.inner, self.max_calls, self.max_usd, self.spent = inner, max_calls, max_usd, 0.0

    def complete(self, req):
        from sfqa.claude import cost_usd

        cached = self.inner.has(req)
        if not cached:
            if self.inner.calls >= self.max_calls:
                raise RuntimeError(f"호출 상한({self.max_calls})에 닿았다")
            if self.spent >= self.max_usd:
                raise RuntimeError(f"비용 상한({self.max_usd}달러)에 닿았다")
        resp = self.inner.complete(req)
        if not cached:
            self.spent += cost_usd(resp)
        return resp


def estimate(cases: list[dict]) -> dict:
    """프롬프트 글자 수로 어림한다. 모델을 부르지 않는다."""
    from sfqa.claude import PRICES

    s = load_settings(SETTINGS_ENV)
    inp, out = PRICES[s.llm_model]
    envs = _envs(cases)
    calls, chars = 0, 0
    for c in cases:
        if c["kind"] in ("minor", "absent"):
            continue
        req = build_request(envs[c["work"]], s, c["character"], c["upto"], c["message"], _history(c), c.get("romance", False))
        calls += 1
        chars += len(req.system) + len(req.user)
    tokens = chars * TOKENS_PER_CHAR
    return {"calls": calls, "chars": chars, "usd": round((tokens * inp + calls * OUTPUT_TOKENS * out) / 1_000_000, 3)}


def run_case(case: dict, env: ChatEnv, llm) -> dict:
    row = {k: case.get(k) for k in ("id", "work", "character", "upto", "kind", "fact", "message", "romance")}
    try:
        r = respond(env, llm, case["character"], case["upto"], case["message"], _history(case), case.get("romance", False))
    except (NotAvailable, RomanceRefused) as e:
        return {**row, "status": "refused", "reply": str(e), "model_reply": "", "replaced": False, "findings": [],
                "must_not_hit": [], "called": False}
    resp = r.response
    return {**row, "status": r.status, "reply": r.reply, "model_reply": r.model_reply, "replaced": r.replaced,
            "invalid": r.invalid, "findings": [asdict(f) for f in r.findings],
            "must_not_hit": [x for x in case.get("must_not", []) if x in r.reply],
            "must_not_hit_raw": [x for x in case.get("must_not", []) if x in r.model_reply],
            "called": bool(resp), "input_tokens": resp.get("input_tokens", 0), "output_tokens": resp.get("output_tokens", 0),
            "latency_s": resp.get("latency_s")}


def report(rows: list[dict], spent: float) -> str:
    kinds = Counter(r["kind"] for r in rows)
    replaced = sum(bool(r.get("replaced")) for r in rows)
    lines = [f"문항 {len(rows)}개, 모델 응답 {sum(bool(r.get('called')) for r in rows)}개(카세트 포함), 이번 실행 비용 {spent:.4f}달러",
             f"종류별 {dict(kinds)}",
             f"검사기가 고정 문구로 바꾼 답 {replaced}개, must_not 적중(최종 답) {sum(bool(r.get('must_not_hit')) for r in rows)}개",
             ""]
    for r in rows:
        mark = "바꿈" if r.get("replaced") else ("거절" if r["status"] == "refused" else ("안내" if r["status"] == "notice" else "그대로"))
        lines.append(f"[{r['id']}] {r['kind']} {r['work']} {r['character']} N={r['upto']} ({mark})")
        lines.append(f"  독자: {r['message']}")
        lines.append(f"  답: {r['reply']}")
        if r.get("replaced") and r.get("model_reply"):
            lines.append(f"  원래 답: {r['model_reply']}")
        if r.get("findings"):
            short = sorted({(f["kind"], tuple(f["terms"]), f["fact"]) for f in r["findings"]}, key=str)[:6]
            lines.append(f"  검사: {short}")
        if r.get("must_not_hit") or r.get("must_not_hit_raw"):
            lines.append(f"  must_not: 최종 {r.get('must_not_hit')}, 원문 {r.get('must_not_hit_raw')}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sfqa.chat_probe", description="인물 채팅 스포일러 시험")
    parser.add_argument("--estimate", action="store_true", help="비용만 어림한다(모델 없음)")
    parser.add_argument("--replay", action="store_true", help="카세트만 쓴다(네트워크 없음)")
    parser.add_argument("--max-calls", type=int, help="실제 호출 상한(카세트에 있으면 세지 않는다)")
    parser.add_argument("--max-usd", type=float, default=1.0, help="비용 상한(달러)")
    parser.add_argument("--only", help="이 글자가 id에 든 문항만")
    parser.add_argument("--out", help="결과를 쓸 폴더(기본: runs/chat-probe-<시각>)")
    args = parser.parse_args(argv)

    cases = load_cases()
    if args.only:
        cases = [c for c in cases if args.only in c["id"]]
    if args.estimate:
        print(json.dumps(estimate(cases), ensure_ascii=False))
        return 0
    s = load_settings(SETTINGS_ENV)
    if args.replay:
        client = CappedClient(ReplayClient(CASSETTE), 0, 0.0)
    else:
        if args.max_calls is None:
            parser.error("실제로 부를 때는 --max-calls가 필요하다")
        from sfqa.claude import AnthropicClient

        client = CappedClient(AnthropicClient(CASSETTE), args.max_calls, args.max_usd)
    envs = _envs(cases)
    rows = []
    for c in cases:
        rows.append(run_case(c, envs[c["work"]], lambda: (s, client)))
        print(f"  {c['id']}: {rows[-1]['status']}{' (바꿈)' if rows[-1].get('replaced') else ''}", flush=True)
    out = Path(args.out) if args.out else ROOT / "runs" / f"chat-probe-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    text = report(rows, client.spent)
    (out / "report.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"결과: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
