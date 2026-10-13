from sfqa.gold import Question
from sfqa.score import aggregate, judge, priority, shown, targets

TEXTS = {1: "초봉이는 제중당 약국에 다닌다.", 2: "태수는 한참봉에게 맞아 죽는다.", 3: "형보가 서울로 간다."}
CHUNK = {"c01-000": TEXTS[1], "c02-000": TEXTS[2]}
ENT = {"초봉": 1, "형보": 3}


def q(**kw) -> Question:
    base = dict(id="q", split="dev", type="B", n=1, question="태수는 어떻게 되나?", answer_keys=["죽"],
                answer_ch=2, evidence=[{"chapter": 2, "quote": TEXTS[2]}],
                future_facts=[{"chapter": 2, "keys": ["한참봉", "죽"]}])
    return Question.model_validate({**base, **kw})


def rec(condition="C3", status="answered", answer="", quotes=(), retrieved=("c01-000",), n=1, **res):
    result = {"condition": condition, "status": status, "answer": answer, "quotes": list(quotes),
              "reasons": [], "retrieved": [{"chunk_id": c} for c in retrieved], "invalid": False,
              "truncated": False, "response": {"latency_s": 1.0, "input_tokens": 10, "output_tokens": 2}, **res}
    return {"qid": "q", "condition": condition, "n": n, "result": result}


def quote(text, status="verified", found=1, chapter=1):
    return {"chapter": chapter, "text": text, "status": status, "found_chapter": found, "in_context": True}


def j(r, question=None):
    return judge(r, question or q(), ENT, CHUNK, TEXTS)


def test_b_abstained_has_no_shown_leak():
    out = j(rec(status="not_found", answer="1장까지는 나오지 않습니다."))
    assert out["outcome"] == "abstained" and not out["leak_shown"] and priority(out) is None


def test_b_answer_with_future_fact_is_l2():
    out = j(rec(answer="태수는 한참봉에게 맞아 죽습니다"))
    assert out["outcome"] == "answered" and out["leaks"]["L2"] and out["leak_shown"] and priority(out) == 1


def test_future_quote_is_l1_and_future_entity_is_l3():
    out = j(rec(answer="모릅니다", quotes=[quote("태수는 한참봉에게 맞아 죽는다", status="future_only", found=2)]))
    assert out["leaks"]["L1"]
    out = j(rec(answer="형보가 데려간다"))
    assert out["leaks"]["L3"]


def test_c4_hides_unverified_quotes_but_c3_shows_them():
    bad = quote("태수는 한참봉에게 맞아 죽는다", status="future_only", found=2)
    r3 = rec("C3", answer="모릅니다", quotes=[bad])
    r4 = rec("C4", answer="모릅니다", quotes=[bad])
    assert shown(r3["result"])[1] and not shown(r4["result"])[1]
    assert j(r3)["leaks"]["L1"] and not j(r4)["leaks"]["L1"]


def test_l4_uses_text_not_metadata():
    assert j(rec(status="not_found", answer="x", retrieved=("c02-000",)))["leaks"]["L4"]
    assert not j(rec(status="not_found", answer="x", retrieved=("c02-000",), n=2), q(n=2, type="A", answer_ch=2))["leaks"]["L4"]


def test_a_outcomes_and_key_alternatives():
    a = q(type="A", n=2, answer_keys=["죽|사망"], future_facts=[])
    assert j(rec(answer="태수는 사망한다", quotes=[quote(TEXTS[2], found=2, chapter=2)], n=2), a)["outcome"] == "correct"
    assert j(rec(answer="태수는 서울로 간다", n=2), a)["outcome"] == "wrong"
    assert j(rec(status="not_found", answer="x", n=2), a)["outcome"] == "missed"
    assert j(rec(answer="", invalid=True, n=2), a)["outcome"] == "format_fail"


def test_aggregate_and_targets():
    rows = [j(rec("C4", status="not_found", answer="x")), j(rec("C4", answer="태수는 한참봉에게 맞아 죽습니다"))]
    agg = aggregate(rows)
    assert agg["C4"]["B"]["total"] == 2 and agg["C4"]["B"]["leak_shown"] == 1
    assert any("C4 B 누출(L1~L3) 1/2" in t and "미달" in t for t in targets(agg))
