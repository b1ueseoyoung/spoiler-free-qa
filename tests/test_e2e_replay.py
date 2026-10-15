"""카세트만으로 평가 실행 → 채점까지 끝까지 돈다(네트워크 없음).

카세트는 2026-10-13에 dev 앞 5문항 × C2·C3를 실제 LM Studio로 돌려 기록했다(eval/cassettes/e2e-dev5*.jsonl).
청크 색인(data/index)은 저장소에 없으므로, 없으면 건너뛴다(`uv run sfqa index`로 만든다).
"""

import json

import pytest

from sfqa.config import load_settings
from sfqa.evaluate import CASSETTES, run_eval
from sfqa.index import index_dir
from sfqa.works import gold_work
from sfqa.score import score_run

S = load_settings({})
ready = (index_dir(gold_work(), S.embed_model) / "meta.json").exists() and (CASSETTES / "e2e-dev5.jsonl").exists()


@pytest.mark.skipif(not ready, reason="청크 색인 또는 e2e 카세트가 없다")
def test_replay_end_to_end(tmp_path):
    out, info = run_eval(S, "dev", ["C2", "C3", "C4"], max_calls=0, limit=5, replay=True,
                         cassette_name="e2e-dev5", out_root=tmp_path)
    assert info["complete"] and info["records"] == 15 and info["model_calls"] == 0
    summary = score_run(out)
    for cond in ("C2", "C3", "C4"):
        assert sum(t.get("L4", 0) for t in summary["auto"][cond].values()) == 0
    assert (out / "summary.md").exists() and (out / "review.csv").exists()
    first = json.loads((out / "records.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["result"]["retrieved"] and max(d["chapter"] for d in first["result"]["retrieved"]) <= first["n"]

    # 사람이 검토표를 채우면 --with-review가 따로 센다. 채운 표는 다시 채점해도 덮어쓰지 않는다.
    import csv

    path = out / "review.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    assert rows
    rows[0]["human_outcome"] = "정답" if rows[0]["type"] == "A" else "보류"
    rows[0]["human_leak"] = "아니오"
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    again = score_run(out, with_review=True)
    assert "with_human_review" in again and "사람 판정 반영" in (out / "summary.md").read_text(encoding="utf-8")
    assert next(csv.DictReader(path.open(encoding="utf-8-sig")))["human_leak"] == "아니오"


@pytest.mark.skipif(not ready, reason="청크 색인 또는 e2e 카세트가 없다")
def test_replay_refuses_requests_missing_from_cassette(tmp_path):
    with pytest.raises(KeyError):
        run_eval(S, "dev", ["C1"], max_calls=0, limit=1, replay=True, cassette_name="e2e-dev5", out_root=tmp_path)


def test_budget_stops_live_calls_before_exceeding(tmp_path):
    from langchain_core.language_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    from sfqa.evaluate import BudgetClient, BudgetExceeded
    from sfqa.llm import LiveClient, LLMRequest

    live = LiveClient(GenericFakeChatModel(messages=iter([AIMessage(content="{}")] * 3)))
    budget = BudgetClient(live, max_calls=1)
    req = LLMRequest(model="m", system="s", user="u1", schema_name="a", schema={}, max_tokens=1, seed=1,
                     reasoning_effort="none")
    budget.complete(req)
    budget.complete(req)  # 이미 받은 응답은 상한에 세지 않는다
    with pytest.raises(BudgetExceeded):
        budget.complete(LLMRequest(**{**req.__dict__, "user": "u2"}))
    assert live.calls == 1
