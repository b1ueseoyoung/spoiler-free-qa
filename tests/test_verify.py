import pytest

from sfqa.verify import Verifier, compact, refusal

TEXTS = {
    1: "초봉이는 정주사의 맏딸이다. 금강(錦江)은 군산으로 흘러 바다에 닿는다.",
    2: "초봉이는 “제중당” 약국에 다닌다고 동네에 소문이 났다.",
    3: "형보가 서울로 올라간다는 소문이 그날 돌았다.",
}
ENT = {"초봉": 1, "형보": 3, "송희": None}


@pytest.fixture
def v() -> Verifier:
    return Verifier(TEXTS, ENT)


def test_compact_folds_quotes_hanja_and_spaces():
    assert compact("“제중당”  약국") == compact("'제중당' 약국") == "'제중당'약국"
    assert compact("금강(錦江)은") == compact("금강은")
    assert compact("그…") == "그..."


def test_quote_in_claimed_chapter(v):
    q = v.check_quote(2, "초봉이는 \"제중당\" 약국에 다닌다고", n=2, context=TEXTS[2])
    assert q.status == "verified" and q.found_chapter == 2 and q.in_context


def test_wrong_chapter_is_fixed_if_within_n(v):
    q = v.check_quote(2, "금강은 군산으로 흘러", n=2, context="")
    assert q.status == "chapter_fixed" and q.found_chapter == 1 and not q.in_context


def test_quote_only_after_n_is_future(v):
    q = v.check_quote(2, "형보가 서울로 올라간다는", n=2, context="")
    assert q.status == "future_only" and q.found_chapter == 3 and not q.kept


def test_made_up_and_short_quotes_are_dropped(v):
    assert v.check_quote(1, "초봉이는 서울에서 태어났다", n=3, context="").status == "not_found"
    assert v.check_quote(1, "초봉이는", n=3, context="").status == "too_short"


def test_future_entities(v):
    assert v.future_entities("형보가 초봉이를 데려간다", n=2) == ["형보"]
    assert v.future_entities("형보가 초봉이를 데려간다", n=3) == []


def test_refusal_text():
    assert refusal(5) == "5장까지는 나오지 않아요."
    assert refusal(19) == "작품에 나오지 않아요."


def test_guard_answers_only_with_verified_quote(v):
    ok = v.verdict("answered", "제중당 약국", [(2, "초봉이는 “제중당” 약국에 다닌다고")], 2, TEXTS[2], apply_guard=True)
    assert ok.status == "answered" and ok.answer == "제중당 약국" and len(ok.quotes) == 1
    no_quote = v.verdict("answered", "제중당 약국", [], 2, "", apply_guard=True)
    assert no_quote.status == "not_found" and no_quote.answer == "2장까지는 나오지 않아요."
    assert "검증된 인용 없음" in no_quote.reasons


def test_guard_blocks_future_entity_and_future_quote(v):
    ent = v.verdict("answered", "형보와 함께 간다", [(2, "초봉이는 “제중당” 약국에 다닌다고")], 2, "", apply_guard=True)
    assert ent.status == "not_found" and ent.future_entities == ["형보"]
    fut = v.verdict("answered", "서울로 간다", [(3, "형보가 서울로 올라간다는")], 2, "", apply_guard=True)
    assert fut.status == "not_found" and fut.l1_future_quote


def test_without_guard_model_output_is_kept_but_checked(v):
    r = v.verdict("answered", "형보와 함께 간다", [], 2, "", apply_guard=False)
    assert r.status == "answered" and r.answer == "형보와 함께 간다"
    assert r.future_entities == ["형보"] and "검증된 인용 없음" in r.reasons
    r = v.verdict(None, "", [], 19, "", apply_guard=False)
    assert r.status == "not_found" and r.answer == "작품에 나오지 않아요."
