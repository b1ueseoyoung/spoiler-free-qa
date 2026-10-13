import pytest

from sfqa.gold import Entity, Question, check, first_chapter

TEXTS = {
    1: "초봉이는 정주사의 맏딸이다. 태수는 은행에 다닌다.",
    2: "초봉이는 제중당 약국에 다닌다.",
    3: "태수는 초봉이와 혼인한다.",
    4: "송희가 태어났다.",
}
ENTITIES = [Entity(name="초봉", aliases=["초봉이"]), Entity(name="태수"), Entity(name="송희")]


def q(**kw) -> Question:
    base = dict(id="q", split="test", type="A", n=3, question="초봉이는 누구와 혼인하나?",
                gold_answer="태수", answer_keys=["태수"], answer_ch=3,
                evidence=[{"chapter": 3, "quote": "태수는 초봉이와 혼인한다."}], question_entities=["초봉"])
    return Question.model_validate({**base, **kw})


def errors_of(*questions) -> list[str]:
    return check(list(questions), TEXTS, ENTITIES)[0]


def test_valid_pair_has_no_errors():
    b = q(id="b", pair_id="p", type="B", n=2, answer_keys=["혼인"], question="초봉이는 누구와 맺어지나?",
          future_facts=[{"chapter": 3, "keys": ["태수", "혼인"]}])
    a = q(id="a", pair_id="p", n=3, answer_keys=["혼인"], question="초봉이는 누구와 맺어지나?")
    assert errors_of(a, b) == []


def test_quote_must_be_verbatim_in_its_chapter():
    errs = errors_of(q(evidence=[{"chapter": 2, "quote": "태수는 초봉이와 혼인한다."}]))
    assert any("글자 그대로 없다" in e and "[3]" in e for e in errs)
    errs = errors_of(q(evidence=[{"chapter": 3, "quote": "태수는 초봉이와  혼인한다."}]))  # 띄어쓰기 다름
    assert any("글자 그대로 없다" in e for e in errs)


def test_quote_length_limit():
    long_text = {**TEXTS, 3: "가" * 200}
    bad = q(evidence=[{"chapter": 3, "quote": "가" * 151}])
    assert any("151자" in e for e in check([bad], long_text, ENTITIES)[0])


@pytest.mark.parametrize("n", [1, 2])
def test_a_needs_answer_within_n(n):
    assert any("A인데" in e for e in errors_of(q(n=n)))


def test_b_must_be_before_answer_and_not_leak_in_question():
    b = q(type="B", n=3, future_facts=[{"chapter": 3, "keys": ["태수"]}])
    errs = errors_of(b)
    assert any("B인데" in e for e in errs)
    b = q(type="B", n=2, question="초봉이는 태수와 어떻게 되나?", future_facts=[{"chapter": 3, "keys": ["혼인"]}])
    assert any("질문 문장에 정답 키워드" in e for e in errors_of(b))
    b = q(type="B", n=2, answer_keys=["혼인"])
    assert any("future_facts가 없다" in e for e in errors_of(b))


def test_c_keys_must_be_absent_from_text():
    c = q(type="C", n=4, answer_ch=None, evidence=[], answer_keys=["조선일보"], question_entities=[])
    assert errors_of(c) == []
    c = q(type="C", n=4, answer_ch=None, evidence=[], answer_keys=["제중당"], question_entities=[])
    assert any("C의 키워드가 본문에 있다" in e for e in errors_of(c))


def test_question_entity_must_appear_by_n():
    b = q(type="B", n=3, question="송희는 누구의 딸인가?", answer_ch=4, answer_keys=["초봉"],
          evidence=[{"chapter": 4, "quote": "송희가 태어났다."}], question_entities=["송희"],
          future_facts=[{"chapter": 4, "keys": ["송희"]}])
    assert any("'송희'이 제4장에 처음" in e for e in errors_of(b))


def test_pair_must_share_question_and_answer_ch():
    a = q(id="a", pair_id="p")
    b = q(id="b", pair_id="p", type="B", n=2, question="다른 질문?", answer_keys=["혼인"],
          future_facts=[{"chapter": 3, "keys": ["혼인"]}])
    assert any("질문과 answer_ch가 같아야" in e for e in errors_of(a, b))


def test_duplicate_ids():
    assert any("id가 2번" in e for e in errors_of(q(id="x"), q(id="x")))


def test_future_fact_cooccurring_early_is_warned():
    b = q(type="B", n=2, answer_keys=["혼인"], future_facts=[{"chapter": 3, "keys": ["초봉이", "정주사"]}])
    warnings = check([b], TEXTS, ENTITIES)[1]
    assert any("누출 오탐 위험" in w for w in warnings)


def test_first_chapter_uses_aliases():
    assert first_chapter(["초봉"], TEXTS) == 1
    assert first_chapter(["송희"], TEXTS) == 4
    assert first_chapter(["없는이름"], TEXTS) is None


# 실제 정답지를 본다. 아직 만들지 않았으면 건너뛴다.
from sfqa.gold import ENTITIES as REAL_ENTITIES  # noqa: E402
from sfqa.gold import QUESTIONS, load_entities, load_questions, load_texts  # noqa: E402


@pytest.mark.skipif(not QUESTIONS.exists(), reason="eval/questions.jsonl이 아직 없다")
def test_real_gold_has_no_errors():
    errors, _ = check(load_questions(), load_texts(), load_entities())
    assert errors == []


@pytest.mark.skipif(not REAL_ENTITIES.exists(), reason="data/entities.json이 아직 없다")
def test_every_entity_appears_in_text():
    texts = load_texts()
    assert [e.name for e in load_entities() if first_chapter(e.forms, texts) is None] == []
