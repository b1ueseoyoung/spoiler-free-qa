import json

import pytest

from sfqa.klue import (SAMPLE, Item, aggregate, build_request, char_f1, exact_match, load_sample, read_response,
                       sample, score_item)
from sfqa.llm import LLMResponse


def row(guid, t, answers=("서해",)):
    return {"guid": guid, "question_type": t, "is_impossible": t == 3, "title": "A &amp; B", "context": "지문",
            "question": "질문?", "answers": {"text": list(answers), "answer_start": [0] * len(answers)}}


ROWS = [row(f"g{t}-{i:03}", t) for t in (1, 2, 3) for i in range(120)]


def item(t=1, answers=("서해",)):
    return Item("g", t, t == 3, "제목", "지문", "질문?", answers)


def resp(status="answered", answer="서해", finish="stop", raw=None):
    text = raw if raw is not None else json.dumps({"status": status, "answer": answer}, ensure_ascii=False)
    return LLMResponse(raw_text=text, finish_reason=finish)


def test_sample_is_balanced_deterministic_and_order_free():
    a = sample(ROWS, per_type=80, seed=20261014)
    b = sample(list(reversed(ROWS)), per_type=80, seed=20261014)
    assert [i.guid for i in a] == [i.guid for i in b]
    assert [sum(i.question_type == t for i in a) for t in (1, 2, 3)] == [80, 80, 80]
    assert a[0].title == "A & B" and a[0].answers == ("서해",)
    assert [i.guid for i in sample(ROWS, 80, seed=1)] != [i.guid for i in a]


def test_sample_rejects_type_and_impossible_mismatch():
    bad = [dict(r, is_impossible=False) if r["question_type"] == 3 else r for r in ROWS]
    with pytest.raises(ValueError):
        sample(bad, per_type=5)


def test_committed_sample_matches_spec():
    items = load_sample(SAMPLE)
    assert len(items) == 240 and len({i.guid for i in items}) == 240
    assert [sum(i.question_type == t for i in items) for t in (1, 2, 3)] == [80, 80, 80]
    assert all(i.is_impossible == (i.question_type == 3) for i in items)
    assert all(i.answers for i in items)


@pytest.mark.parametrize("pred,gold,em", [
    ("서해", "서해", True), ("서해입니다", "서해", True), ("'서해'", "서해", True), ("1,000명", "1000명", True),
    ("경기도", "경기", False), ("9.9%", "99%", False), ("미국(USA)", "미국", True), ("동해", "서해", False),
])
def test_exact_match_ported_from_munjigi(pred, gold, em):
    assert exact_match(pred, gold) is em


def test_char_f1():
    assert char_f1("서해", "서해") == 1.0 and char_f1("동해", "서해") == 0.5 and char_f1("가나", "다라") == 0.0


def test_read_response_formats():
    assert read_response(resp())["status"] == "answered"
    bad = read_response(resp(raw="산문 답"))
    assert bad["invalid"] and bad["status"] is None
    cut = read_response(resp(raw='{"status": "ans', finish="length"))
    assert cut["truncated"] and not cut["invalid"]
    refused = read_response(LLMResponse(raw_text=None, finish_reason="refusal"))
    assert refused["refused"] and refused["status"] == "not_found" and not refused["invalid"]
    wrong_enum = read_response(resp(status="maybe"))
    assert wrong_enum["invalid"]


@pytest.mark.parametrize("t,r,outcome", [
    (1, resp(), "exact"), (1, resp(answer="동해"), "inexact"), (2, resp("not_found", ""), "missed"),
    (2, resp(answer=""), "missed"), (1, resp(raw="x"), "format_fail"),
    (3, resp("not_found", ""), "abstained"), (3, resp(answer="동해"), "ungrounded"),
    (3, LLMResponse(raw_text=None, finish_reason="refusal"), "abstained"),
])
def test_score_item(t, r, outcome):
    assert score_item(item(t), read_response(r))["outcome"] == outcome


def test_trap_hit_only_on_unanswerable():
    assert score_item(item(3), read_response(resp(answer="서해는")))["trap_hit"]
    assert not score_item(item(1), read_response(resp()))["trap_hit"]


def test_aggregate_counts_by_type_and_f1_over_all_answerable():
    items = [Item("a", 1, False, "", "", "", ("서해",)), Item("b", 1, False, "", "", "", ("서해",)),
             Item("c", 3, True, "", "", "", ("동해",))]
    recs = [{"guid": "a", **read_response(resp())}, {"guid": "b", **read_response(resp("not_found", ""))},
            {"guid": "c", **read_response(resp(answer="동해"))}]
    agg = aggregate(items, recs)
    assert agg[1]["exact"] == 1 and agg[1]["missed"] == 1 and agg[1]["f1_mean"] == 0.5
    assert agg[3]["ungrounded"] == 1 and agg[3]["trap_hit"] == 1 and "f1_mean" not in agg[3]


def test_request_carries_schema_and_passage():
    req = build_request(item(), "m", 800, 1, "none")
    assert req.schema["required"] == ["status", "answer"] and "[지문] 제목\n지문" in req.user
    assert "status를 answered로" in req.system and "status를 not_found로" in req.system
