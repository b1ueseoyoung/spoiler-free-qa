import json

import pytest

from sfqa import compare
from sfqa.compare import PROFILES, gates, latency_stats, rank, render, run_klue, settings_for, write_report
from sfqa.config import load_settings
from sfqa.evaluate import BudgetClient, BudgetExceeded
from sfqa.klue import Item
from sfqa.llm import LLMResponse
from tests.test_answer import FakeClient, run


def info(slug, em1=60, em2=50, abstain3=70, fmt=0, c4_leak=0, c4_c=6, c4_fmt=0, p50=5.0, baseline=False):
    return {
        "profile": {**vars(PROFILES.get(slug, PROFILES["qwen3.8-27b"])), "slug": slug, "baseline": baseline},
        "status": "done",
        "klue": {"by_type": {"1": {"total": 80, "exact": em1, "f1_mean": 0.8, "format_fail": fmt},
                             "2": {"total": 80, "exact": em2, "f1_mean": 0.7},
                             "3": {"total": 80, "abstained": abstain3, "trap_hit": 3}},
                 "latency": {"p50_s": 2.0, "p95_s": 4.0, "out_tok_per_s": 40.0, "input_tokens": 1, "output_tokens": 1}},
        "test": {"auto": {"C4": {"A": {"total": 22, "correct": 10, "answered": 12, "kept_quote": 12, "chapter_match": 9},
                                 "B": {"total": 16, "leak_shown": c4_leak, "abstained": 16},
                                 "C": {"total": 6, "abstained": c4_c, "format_fail": c4_fmt}}},
                 "latency_c4": {"p50_s": p50, "p95_s": p50 * 2, "input_tokens": 1, "output_tokens": 1}},
    }


def test_profiles_cover_spec_and_reuse_current_cassettes():
    assert len([p for p in PROFILES.values() if not p.baseline]) == 5
    assert PROFILES["qwen3.6-35b-a3b"].cassette("test") == "test"
    assert PROFILES["qwen3.6-35b-a3b"].cassette("klue") == "klue-qwen3.6-35b-a3b"
    assert PROFILES["gpt-oss-20b"].cassette("test") == "test-gpt-oss-20b"
    assert PROFILES["gpt-oss-20b"].reasoning_effort == "low" and PROFILES["gemma-4-26b-a4b"].reasoning_effort == ""
    assert PROFILES["gemma-4-31b-qat"].reasoning_effort == "none"  # GGUF 판은 기본 thinking


def test_settings_for_claude_does_not_point_at_lm_studio():
    s = settings_for(PROFILES["claude-sonnet-5-5"], load_settings({}))
    assert s.llm_model == "claude-sonnet-5-5" and "anthropic" in s.base_url
    assert settings_for(PROFILES["gpt-oss-20b"], load_settings({})).reasoning_effort == "low"


@pytest.mark.parametrize("kw,failed", [
    ({}, None), ({"c4_leak": 1}, "gate1"), ({"c4_c": 4}, "gate1"), ({"c4_fmt": 3}, "gate1"),
    ({"abstain3": 59}, "gate2"), ({"fmt": 6}, "gate3"),
])
def test_gates_use_spec_thresholds(kw, failed):
    g = gates(info("m", **kw))
    assert [k for k, v in g.items() if not v] == ([failed] if failed else [])


def test_rank_breaks_ties_within_eight_by_latency_and_skips_baseline_and_failures():
    infos = [info("a", em1=60, em2=50, p50=9.0), info("b", em1=58, em2=45, p50=3.0),  # a 110, b 103 (차이 7)
             info("c", em1=50, em2=40, p50=1.0),  # 90: 동률 밖
             info("d", em1=80, em2=80, c4_leak=1),  # 관문 1 미달
             info("claude", em1=80, em2=80, baseline=True)]
    r = rank(infos)
    assert r["order"] == ["a", "b", "c"] and r["tied"] == ["a", "b"] and r["chosen"] == "b"
    assert rank([info("a", em1=60), info("b", em1=51, p50=1.0)])["chosen"] == "a"  # 차이 9는 동률 아님


def test_report_renders_tables_from_saved_infos(tmp_path):
    for slug in ("qwen3.8-27b", "claude-sonnet-5-5"):
        (tmp_path / slug).mkdir()
        i = info(slug, baseline=slug.startswith("claude"))
        (tmp_path / slug / "model-info.json").write_text(json.dumps(i, ensure_ascii=False))
    md = write_report(tmp_path).read_text(encoding="utf-8")
    assert "claude-sonnet-5-5 (기준선)" in md and "**고른 모델: qwen3.8-27b**" in md
    assert "| **110** |" in md
    assert "### 근거 인용" in md and "| 12 | 9 |" in md
    assert json.loads((tmp_path / "compare.json").read_text())["rank"]["chosen"] == "qwen3.8-27b"


def test_render_handles_missing_stages(tmp_path):
    md = render(tmp_path, [{"profile": vars(PROFILES["gpt-oss-20b"]), "status": "smoke_failed",
                            "smoke": {"ok": False}}])
    assert "smoke_failed" in md and "관문을 모두 넘은 로컬 모델이 없다" in md


def test_run_klue_writes_scored_records(tmp_path):
    items = [Item("a", 1, False, "t", "c", "q", ("서해",)), Item("b", 3, True, "t", "c", "q", ("동해",))]
    client = FakeClient('{"status": "answered", "answer": "서해"}')
    recs = run_klue(client, load_settings({}), items, tmp_path / "klue.jsonl")
    assert [r["outcome"] for r in recs] == ["exact", "ungrounded"]
    assert len((tmp_path / "klue.jsonl").read_text().splitlines()) == 2


class CountingClient:
    """has/calls가 있는 실제 호출 클라이언트 대역(AnthropicClient와 같은 모양)."""

    def __init__(self):
        self.calls = 0

    def has(self, req):
        return False

    def complete(self, req):
        self.calls += 1
        return LLMResponse(raw_text="{}", finish_reason="stop")


def test_budget_applies_to_any_live_client():
    b = BudgetClient(CountingClient(), max_calls=1)
    b.complete(object())
    with pytest.raises(BudgetExceeded):
        b.complete(object())


def test_latency_stats():
    st = latency_stats([{"latency_s": x, "output_tokens": 10, "input_tokens": 1} for x in (1.0, 2.0, 3.0, 10.0)])
    assert st["p50_s"] == 2.5 and st["p95_s"] == 10.0 and st["n"] == 4 and st["output_tokens"] == 40


def test_answer_chain_counts_claude_refusal_as_abstain_not_format_fail():
    result, _ = run("C4", raw="", finish="refusal")
    assert result.status == "not_found" and not result.invalid and result.model_status == "not_found"


def test_compare_rejects_unknown_model(tmp_path):
    with pytest.raises(ValueError):
        compare.run_compare(load_settings({}), ["nope"], 1, tmp_path)


@pytest.mark.parametrize("raw,status", [
    ('{"status": "not_found", "answer": ""}', "checked"),  # 형식만 맞으면 점검 통과(정답 여부는 기록만)
    ("<think>생각 중", "smoke_failed"),
])
def test_run_model_saves_info_for_smoke_result(tmp_path, monkeypatch, raw, status):
    from sfqa.compare import Profile, run_model

    class Live(FakeClient):  # LiveClient 모양(has, calls)
        calls = 0

        def has(self, req):
            return False

    monkeypatch.setattr(compare, "make_client", lambda p, s, c: Live(raw))
    monkeypatch.setattr(compare, "run_eval", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("dev 없음")))
    info = run_model(Profile("fake", "m", "none", provider="fake"), load_settings({}), tmp_path, 5, check_only=True)
    saved = json.loads((tmp_path / "fake" / "model-info.json").read_text())
    expected = "error" if status == "checked" else status  # 점검을 넘으면 dev 단계(여기선 일부러 실패)로 간다
    assert info["status"] == saved["status"] == expected
    assert saved["smoke"]["ok"] is (status == "checked")


def test_smoke_fails_on_hidden_thinking_tokens():
    from sfqa.compare import smoke

    class Thinking(FakeClient):
        def complete(self, req):
            return LLMResponse(raw_text='{"status": "answered", "answer": "서해"}', finish_reason="stop", output_tokens=297)

    r = smoke(Thinking(""), load_settings({}))
    assert not r["ok"] and r["correct"] and r["output_tokens"] == 297


def test_cassette_cost_sums_every_recorded_call(tmp_path):
    from sfqa.compare import cassette_cost

    row = json.dumps({"key": "k", "response": {"input_tokens": 1_000_000, "output_tokens": 100_000}})
    (tmp_path / "klue-claude-sonnet-5-5.jsonl").write_text(row + "\n" + row + "\n")
    (tmp_path / "test-claude-sonnet-5-5.jsonl").write_text(row + "\n")
    (tmp_path / "test-query-embeddings.jsonl").write_text("{}\n")  # 다른 파일은 세지 않는다
    assert cassette_cost("claude-sonnet-5-5", "claude-sonnet-5-5", tmp_path) == 3 * (2.0 + 1.0)
