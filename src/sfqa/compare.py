"""모델 비교 실행기와 비교표(docs/model-eval-spec.md, 2026-10-14 확정).

모델마다 순서대로 돈다.
1. (LM Studio) 다른 모델을 모두 내리고 이 모델만 올린다. 올리는 시간과 lms ps가 보고한 크기를 남긴다.
2. 점검: 짧은 지문 질문 1회로 JSON이 content로 오는지 본다(thinking이 켜져 있으면 비어 온다). 실패하면 이 모델은 건너뛴다.
3. dev 12문항 C3로 응답 형식을 본다(기록만 한다).
4. KLUE-MRC 240문항
5. test 44문항 C0·C2·C3·C4(C4는 C3 응답을 다시 쓴다) → sfqa.score로 자동 채점

- 응답은 모델별 카세트(eval/cassettes/)에 남는다. 같은 --out으로 다시 돌리면 카세트에 있는 것은 부르지 않는다.
- 질문 임베딩은 기존 카세트(dev·test)를 replay로 쓴다. 임베딩 모델을 올려 둘 필요가 없다.
- 순위는 로컬 모델만 매긴다. Claude는 기준선이다.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

from sfqa import klue
from sfqa.answer import Env
from sfqa.config import Settings
from sfqa.evaluate import CASSETTES, RUNS, BudgetClient, BudgetExceeded, run_eval
from sfqa.llm import LiveClient, LLMRequest, make_chat

TEST_CONDITIONS = ["C0", "C2", "C3", "C4"]
CONTEXT_LENGTH = 32768  # 지금 qwen3.6을 올려 둔 값과 같게
RESTORE = ("qwen/qwen3.6-35b-a3b", "text-embedding-kure-v1")  # 끝나면 웹 데모가 쓰던 상태로 되돌린다

# 고르는 규칙(실행 전 확정)
GATE1_C_ABSTAIN_MIN = 5  # test C4 C 보류 5/6 이상
GATE1_FORMAT_MAX = 2  # test C4 형식 실패 2/44 이하
GATE2_ABSTAIN_MIN = 60  # KLUE 답 없음 80문항 중 보류 60 이상
GATE3_FORMAT_MAX = 5  # KLUE 형식 실패 5/240 이하
TIE_MARGIN = 8  # KLUE 답 있음 160문항 EM 차이가 8문항(5%p) 이내면 같은 것으로 본다


@dataclass(frozen=True)
class Profile:
    slug: str
    model: str
    reasoning_effort: str  # LM Studio에 그대로 보낸다. 빈 값이면 보내지 않는다
    provider: str = "lmstudio"
    reuse: bool = False  # 지금 모델은 기존 dev·test 카세트를 다시 쓴다
    baseline: bool = False

    def cassette(self, stage: str) -> str:
        return stage if self.reuse and stage in ("dev", "test") else f"{stage}-{self.slug}"


PROFILES = {p.slug: p for p in [
    Profile("qwen3.6-35b-a3b", "qwen/qwen3.6-35b-a3b", "none", reuse=True),
    Profile("qwen3.8-27b", "qwen/qwen3.8-27b", "none"),
    Profile("gemma-4-26b-a4b", "google/gemma-4-26b-a4b", ""),
    Profile("gpt-oss-20b", "openai/gpt-oss-20b", "low"),
    # GGUF 판은 기본으로 thinking이 켜져 있다(숨은 출력 약 300토큰). none으로 꺼진다. low·템플릿 옵션으로는 안 꺼졌다.
    Profile("gemma-4-31b-qat", "google/gemma-4-31b-qat", "none"),
    Profile("claude-sonnet-5-5", "claude-sonnet-5-5", "between_tools:high", provider="anthropic", baseline=True),
]}

# 점검 답은 20~30토큰이다. 이보다 훨씬 많으면 보이지 않는 thinking이 돈 것이다.
# (LangChain은 LM Studio의 reasoning_content를 넘겨주지 않아 reasoning_chars로는 잡히지 않는다.)
SMOKE_MAX_OUTPUT_TOKENS = 150

SMOKE_ITEM = klue.Item(
    guid="smoke", question_type=1, is_impossible=False, title="군산",
    context="군산은 전라북도 서북쪽, 금강 하구에 있는 항구 도시로 서해에 접해 있다.",
    question="군산은 어느 바다에 접해 있나?", answers=("서해",),
)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def lms(*args: str, timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(["lms", *args], capture_output=True, text=True, timeout=timeout)


def load_model(model: str, context_length: int = CONTEXT_LENGTH) -> dict:
    lms("unload", "--all")
    started = time.monotonic()
    r = lms("load", model, "-c", str(context_length), "-y", "--identifier", model)
    load_s = time.monotonic() - started
    if r.returncode != 0:
        raise RuntimeError(f"lms load {model} 실패: {(r.stderr or r.stdout).strip()[-300:]}")
    entry = next((e for e in json.loads(lms("ps", "--json").stdout or "[]") if e.get("identifier") == model), {})
    return {"load_s": round(load_s, 1), "size_bytes": entry.get("sizeBytes"), "format": entry.get("format"),
            "context_length": entry.get("contextLength"), "quantization": (entry.get("quantization") or {}).get("name")}


def restore() -> None:
    lms("unload", "--all")
    for model in RESTORE:
        args = ["load", model, "-y", "--identifier", model]
        if not model.startswith("text-embedding"):
            args += ["-c", str(CONTEXT_LENGTH)]
        lms(*args)


def settings_for(p: Profile, base: Settings) -> Settings:
    s = replace(base, llm_model=p.model, reasoning_effort=p.reasoning_effort)
    return replace(s, base_url="https://api.anthropic.com") if p.provider == "anthropic" else s


def make_client(p: Profile, s: Settings, cassette: Path):
    if p.provider == "anthropic":
        from sfqa.claude import AnthropicClient

        return AnthropicClient(cassette)
    return LiveClient(make_chat(s), cassette)


def make_env(s: Settings, client, emb_name: str, k: int = 18) -> Env:
    from sfqa.gold import entity_index, load_entities, load_texts
    from sfqa.index import load_index
    from sfqa.llm import CachedEmbeddings
    from sfqa.retrieve import ChapterIndex
    from sfqa.verify import Verifier
    from sfqa.works import gold_work

    work = gold_work()  # 모델 비교는 「탁류」 정답지로만 한다
    texts = load_texts(work)
    chunks, vectors, _ = load_index(work, s.embed_model, texts, s.embed_provider)
    emb = CachedEmbeddings(None, s.embed_model, CASSETTES / f"{emb_name}-query-embeddings.jsonl", replay=True)
    return Env(s, work, ChapterIndex(chunks, vectors), emb, client,
               Verifier(texts, entity_index(load_entities(work), texts)), k)


def latency_stats(responses: list[dict]) -> dict:
    lat = sorted(r["latency_s"] for r in responses if r.get("latency_s"))
    if not lat:
        return {}
    tps = [r["output_tokens"] / r["latency_s"] for r in responses if r.get("latency_s") and r.get("output_tokens")]
    return {"n": len(lat), "p50_s": round(statistics.median(lat), 2),
            "p95_s": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 2),
            "out_tok_per_s": round(statistics.median(tps), 1) if tps else None,
            "input_tokens": sum(r.get("input_tokens", 0) for r in responses),
            "output_tokens": sum(r.get("output_tokens", 0) for r in responses)}


def smoke(client, s: Settings) -> dict:
    req = klue.build_request(SMOKE_ITEM, s.llm_model, s.max_tokens, s.seed, s.reasoning_effort)
    resp = client.complete(req)
    rec = klue.read_response(resp)
    # 형식과 숨은 출력만 본다(thinking이 켜져 있으면 content가 비거나 출력 토큰이 많다). 정답 여부는 기록만 한다.
    ok = (not rec["invalid"] and not rec["truncated"] and rec["status"] is not None
          and resp.output_tokens <= SMOKE_MAX_OUTPUT_TOKENS)
    return {"ok": ok, "correct": rec["status"] == "answered" and "서해" in rec["answer"], "status": rec["status"], "answer": rec["answer"], "finish_reason": resp.finish_reason,
            "output_tokens": resp.output_tokens, "latency_s": resp.latency_s,
            "raw_head": (resp.raw_text or "")[:120]}


def run_klue(client, s: Settings, items: list[klue.Item], path: Path) -> list[dict]:
    records = []
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            req: LLMRequest = klue.build_request(item, s.llm_model, s.max_tokens, s.seed, s.reasoning_effort)
            resp = client.complete(req)
            rec = {"guid": item.guid, "question_type": item.question_type, **klue.read_response(resp),
                   "response": asdict(resp)}
            rec |= klue.score_item(item, rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            records.append(rec)
    return records


def run_model(p: Profile, base: Settings, out: Path, max_calls: int, check_only: bool = False) -> dict:
    """한 모델을 끝까지 돌리고 model-info.json을 쓴다. 상한에 닿으면 status=stopped로 남긴다."""
    from sfqa.score import score_run

    mdir = out / p.slug
    mdir.mkdir(parents=True, exist_ok=True)
    s = settings_for(p, base)
    info: dict = {"profile": asdict(p), "settings": {"model": s.llm_model, "reasoning_effort": s.reasoning_effort,
                  "max_tokens": s.max_tokens, "seed": s.seed}, "started_at": datetime.now().isoformat(timespec="seconds")}

    def save(status: str) -> dict:
        info.update(status=status, finished_at=datetime.now().isoformat(timespec="seconds"))
        (mdir / "model-info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return info

    try:
        if p.provider == "lmstudio":
            log(f"{p.slug}: 올리는 중")
            info["load"] = load_model(p.model)
            log(f"{p.slug}: 올림 {info['load']['load_s']}초")
        klue_client = BudgetClient(make_client(p, s, CASSETTES / f"{p.cassette('klue')}.jsonl"), max_calls)
        info["smoke"] = smoke(klue_client, s)
        log(f"{p.slug}: 점검 {'통과' if info['smoke']['ok'] else '실패'} {info['smoke']['raw_head']!r}")
        if not info["smoke"]["ok"]:
            return save("smoke_failed")

        used = klue_client.calls
        dev_client = make_client(p, s, CASSETTES / f"{p.cassette('dev')}.jsonl")
        dev_dir, dev = run_eval(s, "dev", ["C3"], max_calls - used, cassette_name=p.cassette("dev"),
                                out_root=mdir, env=make_env(s, dev_client, "dev"))
        used += dev["model_calls"]
        dev_recs = [json.loads(x) for x in (dev_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()]
        info["dev"] = {"run_dir": dev_dir.name, "records": len(dev_recs),
                       "format_fail": sum(r["result"]["invalid"] or r["result"]["truncated"] for r in dev_recs)}
        log(f"{p.slug}: dev C3 형식 실패 {info['dev']['format_fail']}/{len(dev_recs)}")
        if not dev["complete"]:
            return save("stopped")
        if check_only:
            return save("checked")

        items = klue.load_sample()
        klue_client.max_calls = max_calls - dev["model_calls"]  # 점검 호출은 klue_client.calls에 이미 들어 있다
        log(f"{p.slug}: KLUE {len(items)}문항 시작")
        recs = run_klue(klue_client, s, items, mdir / "klue.jsonl")
        by_type = {str(t): v for t, v in klue.aggregate(items, recs).items()}
        info["klue"] = {"by_type": by_type, "latency": latency_stats([r["response"] for r in recs])}
        used = klue_client.calls + dev["model_calls"]
        log(f"{p.slug}: KLUE 끝 {info['klue']['by_type']}")

        test_client = make_client(p, s, CASSETTES / f"{p.cassette('test')}.jsonl")
        test_dir, t = run_eval(s, "test", TEST_CONDITIONS, max_calls - used, cassette_name=p.cassette("test"),
                               out_root=mdir, env=make_env(s, test_client, "test"))
        scored = score_run(test_dir)
        recs_t = [json.loads(x) for x in (test_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()]
        info["test"] = {"run_dir": test_dir.name, "complete": t["complete"], "auto": scored["auto"],
                        "latency_c4": latency_stats([r["result"]["response"] for r in recs_t if r["condition"] == "C4"])}
        info["model_calls"] = used + t["model_calls"]
        log(f"{p.slug}: test 끝 (모델 호출 합계 {info['model_calls']}회)")
        return save("done" if t["complete"] else "stopped")
    except BudgetExceeded as e:
        info["stopped"] = str(e)
        return save("stopped")
    except Exception as e:  # 한 모델이 실패해도 다음 모델로 넘어간다
        info["error"] = f"{type(e).__name__}: {e}"
        log(f"{p.slug}: 오류 {info['error']}")
        return save("error")


def run_compare(base: Settings, slugs: list[str], max_calls: int, out: Path | None = None,
                check_only: bool = False) -> Path:
    bad = [x for x in slugs if x not in PROFILES]
    if bad:
        raise ValueError(f"없는 모델이다: {bad}. 가능한 값: {list(PROFILES)}")
    out = out or RUNS / f"compare-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    local = False
    try:
        for slug in slugs:
            p = PROFILES[slug]
            local |= p.provider == "lmstudio"
            run_model(p, base, out, max_calls, check_only)
            write_report(out)
    finally:
        if local:
            log("웹 데모 상태로 되돌리는 중(qwen3.6 + kure)")
            restore()
    return out


# --- 비교표 ---

def _test_cell(info: dict, cond: str, typ: str, key: str) -> int | None:
    return ((info.get("test") or {}).get("auto", {}).get(cond, {}).get(typ, {}) or {}).get(key)


def gates(info: dict) -> dict:
    """관문 1~3 결과. 값이 없으면 None(측정 못 함)."""
    k = (info.get("klue") or {}).get("by_type")
    auto = (info.get("test") or {}).get("auto")
    out: dict = {"gate1": None, "gate2": None, "gate3": None}
    if auto and "C4" in auto:
        c4 = auto["C4"]
        b, c = c4.get("B", {}), c4.get("C", {})
        fmt = sum(t.get("format_fail", 0) for t in c4.values())
        out["gate1"] = b.get("leak_shown", 0) == 0 and c.get("abstained", 0) >= GATE1_C_ABSTAIN_MIN and fmt <= GATE1_FORMAT_MAX
    if k:
        out["gate2"] = k["3"].get("abstained", 0) >= GATE2_ABSTAIN_MIN
        out["gate3"] = sum(t.get("format_fail", 0) for t in k.values()) <= GATE3_FORMAT_MAX
    return out


def answerable_em(info: dict) -> int:
    k = info["klue"]["by_type"]
    return k["1"].get("exact", 0) + k["2"].get("exact", 0)


def rank(infos: list[dict]) -> dict:
    """관문을 모두 넘은 로컬 모델을 EM 순으로 세우고, 1위와 TIE_MARGIN 이내면 test C4 응답 시간 중앙값으로 고른다."""
    passed = [i for i in infos if not i["profile"]["baseline"] and all(gates(i).values())]
    if not passed:
        return {"order": [], "chosen": None, "tied": []}
    order = sorted(passed, key=lambda i: -answerable_em(i))
    top = answerable_em(order[0])
    tied = [i for i in order if top - answerable_em(i) <= TIE_MARGIN]
    chosen = min(tied, key=lambda i: (i["test"]["latency_c4"].get("p50_s", float("inf")), -answerable_em(i)))
    return {"order": [i["profile"]["slug"] for i in order], "chosen": chosen["profile"]["slug"],
            "tied": [i["profile"]["slug"] for i in tied]}


def load_infos(out: Path) -> list[dict]:
    infos = []
    for slug in PROFILES:
        path = out / slug / "model-info.json"
        if path.exists():
            info = json.loads(path.read_text(encoding="utf-8"))
            if info.get("klue"):  # JSON에 쓰면서 유형 키가 문자열이 된다. 메모리에서 바로 쓸 때도 맞춘다
                info["klue"]["by_type"] = {str(t): v for t, v in info["klue"]["by_type"].items()}
            infos.append(info)
    return infos


def cassette_cost(slug: str, model: str, root: Path = CASSETTES) -> float:
    """이 모델 카세트(dev·klue·test)에 기록된 응답 전체의 비용. 실제로 부른 호출마다 한 줄씩이다."""
    from sfqa.claude import PRICES

    inp, outp = PRICES[model]
    total = 0.0
    for path in root.glob(f"*-{slug}.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)["response"]
                total += (r["input_tokens"] * inp + r["output_tokens"] * outp) / 1e6
    return total


def _fmt(v, suffix: str = "") -> str:
    return "—" if v is None else f"{v}{suffix}"


def render(out: Path, infos: list[dict]) -> str:
    def name(i):
        return i["profile"]["slug"] + (" (기준선)" if i["profile"]["baseline"] else "")

    L = [f"# 모델 비교 — {out.name}", "",
         "- 기준: `docs/model-eval-spec.md`(2026-10-14 확정). 사람 판정·AI 판정 없음. 모델마다 1회.",
         "- test 정답지는 사람 검토 전(pre-human-review)이다. A 정답(키워드)은 참고로만 쓴다.", ""]
    L += ["## 상태", "", "| 모델 | 상태 | 점검 | dev C3 형식 실패 | 모델 호출 |", "| --- | --- | --- | --- | --- |"]
    for i in infos:
        sm, dev = i.get("smoke") or {}, i.get("dev") or {}
        L.append(f"| {name(i)} | {i['status']}{' — ' + i['error'] if i.get('error') else ''} | "
                 f"{'통과' if sm.get('ok') else '실패' if sm else '—'} | "
                 f"{_fmt(dev.get('format_fail'))}/{_fmt(dev.get('records'))} | {_fmt(i.get('model_calls'))} |")

    L += ["", "## 1. KLUE-MRC 240문항 (지문을 주고 묻기)", "",
          "| 모델 | 유형1 EM /80 | 유형1 F1 | 유형2 EM /80 | 유형2 F1 | **답 있음 EM /160** | 놓침(1·2) | "
          "유형3 보류 /80 | 함정 적중 | 형식 실패 /240 | 안전 거절 |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for i in infos:
        k = (i.get("klue") or {}).get("by_type")
        if not k:
            continue
        t1, t2, t3 = k["1"], k["2"], k["3"]
        fmt = sum(t.get("format_fail", 0) for t in k.values())
        L.append(f"| {name(i)} | {t1.get('exact', 0)} | {t1.get('f1_mean')} | {t2.get('exact', 0)} | {t2.get('f1_mean')} | "
                 f"**{answerable_em(i)}** | {t1.get('missed', 0) + t2.get('missed', 0)} | {t3.get('abstained', 0)} | "
                 f"{t3.get('trap_hit', 0)} | {fmt} | {sum(t.get('refused', 0) for t in k.values())} |")

    L += ["", "## 2. test 44문항 (스포일러 관문, 자동 판정)", "",
          "B 누출 = 독자에게 보이는 답·인용에서 L1~L3 중 하나라도 걸린 B 문항 수(/16).", "",
          "| 모델 | C0 B 누출 | C2 B 누출 | C3 B 누출 | **C4 B 누출** | C4 C 보류 /6 | C4 형식 실패 /44 | C3 L4 | "
          "C4 A 정답(참고) /22 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for i in infos:
        auto = (i.get("test") or {}).get("auto")
        if not auto:
            continue
        c4fmt = sum(t.get("format_fail", 0) for t in auto.get("C4", {}).values())
        c3l4 = sum(t.get("L4", 0) for t in auto.get("C3", {}).values())
        L.append(f"| {name(i)} | " + " | ".join(_fmt(_test_cell(i, c, "B", "leak_shown")) for c in ("C0", "C2", "C3"))
                 + f" | **{_fmt(_test_cell(i, 'C4', 'B', 'leak_shown'))}** | {_fmt(_test_cell(i, 'C4', 'C', 'abstained') or 0)} | "
                 f"{c4fmt} | {c3l4} | {_fmt(_test_cell(i, 'C4', 'A', 'correct') or 0)} |")

    L += ["", "### 근거 인용 (test A 22문항, 자동)", "",
          "검증된 인용 = 정규화 뒤 N장 이하 원문에 글자 그대로 있는 인용. 근거 장 일치 = 그 인용의 장이 정답지 근거 장과 같음.", "",
          "| 모델 | C3 답함 | C3 검증된 인용 있음 | C3 근거 장 일치 | C4 답함(검증 통과) | C4 근거 장 일치 |",
          "| --- | --- | --- | --- | --- | --- |"]
    for i in infos:
        if not (i.get("test") or {}).get("auto"):
            continue
        cell = lambda c, k: _fmt(_test_cell(i, c, "A", k) or 0)  # noqa: E731
        L.append(f"| {name(i)} | {cell('C3', 'answered')} | {cell('C3', 'kept_quote')} | {cell('C3', 'chapter_match')} | "
                 f"{cell('C4', 'answered')} | {cell('C4', 'chapter_match')} |")

    L += ["", "## 3. 속도와 자원", "",
          "응답 시간은 카세트에 기록된 첫 호출 기준이다. C4는 C3 응답을 다시 쓰므로 C3 호출 시간과 같다.", "",
          "| 모델 | 올리기(초) | 크기 | KLUE p50 / p95(초) | test C4 p50 / p95(초) | 출력 토큰/초(KLUE 중앙값) | 비용 |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for i in infos:
        ld, kl = i.get("load") or {}, (i.get("klue") or {}).get("latency") or {}
        tl = (i.get("test") or {}).get("latency_c4") or {}
        size = f"{ld['size_bytes'] / 1e9:.1f}GB {ld.get('format') or ''} {ld.get('quantization') or ''}".strip() \
            if ld.get("size_bytes") else "—"
        cost = "—"
        if i["profile"]["provider"] == "anthropic":
            cost = f"${cassette_cost(i['profile']['slug'], i['profile']['model']):.2f}(카세트 전체: 점검·dev·KLUE·test)"
        L.append(f"| {name(i)} | {_fmt(ld.get('load_s'))} | {size} | {_fmt(kl.get('p50_s'))} / {_fmt(kl.get('p95_s'))} | "
                 f"{_fmt(tl.get('p50_s'))} / {_fmt(tl.get('p95_s'))} | {_fmt(kl.get('out_tok_per_s'))} | {cost} |")

    mark = {True: "통과", False: "**미달**", None: "—"}
    L += ["", "## 4. 관문과 순위", "",
          f"- 관문 1: test C4 B 누출 0/16, C 보류 {GATE1_C_ABSTAIN_MIN}/6 이상, 형식 실패 {GATE1_FORMAT_MAX}/44 이하",
          f"- 관문 2: KLUE 답 없음 보류 {GATE2_ABSTAIN_MIN}/80 이상 · 관문 3: KLUE 형식 실패 {GATE3_FORMAT_MAX}/240 이하",
          f"- 순위: 관문을 모두 넘은 로컬 모델을 답 있음 EM 순으로. 1위와 {TIE_MARGIN}문항 이내면 test C4 응답 시간 중앙값이 짧은 쪽", "",
          "| 모델 | 관문 1 | 관문 2 | 관문 3 | 답 있음 EM |", "| --- | --- | --- | --- | --- |"]
    for i in infos:
        g = gates(i)
        em = answerable_em(i) if (i.get("klue") or {}).get("by_type") else None
        L.append(f"| {name(i)} | {mark[g['gate1']]} | {mark[g['gate2']]} | {mark[g['gate3']]} | {_fmt(em)} |")
    done = [i for i in infos if i.get("klue") and i.get("test")]
    r = rank(done)
    L.append("")
    if r["chosen"]:
        L += [f"- 관문 통과 순서(EM): {' > '.join(r['order'])}",
              f"- 동률 묶음(1위와 {TIE_MARGIN}문항 이내): {', '.join(r['tied'])}",
              f"- **고른 모델: {r['chosen']}**"]
    else:
        L.append("- 관문을 모두 넘은 로컬 모델이 없다(또는 아직 다 돌지 않았다).")
    return "\n".join(L) + "\n"


def write_report(out: Path) -> Path:
    infos = load_infos(out)
    (out / "compare.md").write_text(render(out, infos), encoding="utf-8")
    done = [i for i in infos if i.get("klue") and i.get("test")]
    summary = {"models": {i["profile"]["slug"]: {"status": i["status"], "gates": gates(i)} for i in infos},
               "rank": rank(done)}
    (out / "compare.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out / "compare.md"
