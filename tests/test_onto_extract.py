import json
from types import SimpleNamespace

import pytest

from sfqa.onto_extract import (
    SCHEMA,
    ChapterOut,
    build_request,
    locate,
    run,
)
from sfqa.ontology import Ontology, character_view, check, split_facts

TEXTS = {
    1: "갑순은 은종탑의 필경사다. 그녀는 매일 서약서를 베낀다.\n\n을동이 찾아와 빚 이야기를 꺼냈다.",
    2: "가면 쓴 사내가 밤에 서고에 숨어들었다.\n\n을동은 갑순에게 아버지 소식을 전했다. “아버지는 북쪽에 계셔.”",
    3: "가면을 벗자 병철의 얼굴이 드러났다. 가면 쓴 사내는 병철이었다.\n\n갑순은 그 사실을 을동에게 말했다.",
}


def out(characters=(), aliases=(), merges=(), facts=(), learned=(), pov=None, present=()) -> dict:
    return {"characters": list(characters), "aliases": list(aliases), "merges": list(merges), "facts": list(facts),
            "learned": list(learned), "scene": {"pov": pov, "location": "은종탑", "present": list(present)}}


def fact(statement, quote, knowers, category="event", common=False) -> dict:
    return {"statement": statement, "category": category, "quote": quote, "common": common,
            "knowers": [{"character": c, "since": s} for c, s in knowers]}


REPLIES = {
    1: out(characters=[{"id": "gapsun", "name": "갑순", "appeared": True, "aliases": []},
                       {"id": "euldong", "name": "을동", "appeared": True, "aliases": []}],
           facts=[fact("갑순은 은종탑의 필경사다.", "갑순은 은종탑의 필경사다.", [("gapsun", 0), ("euldong", 1), ("ghost", 1)],
                       "identity"),
                  fact("갑순은 왕족이다.", "본문에 없는 문장을 지어낸 인용", [("gapsun", 0)])],
           pov="gapsun", present=["gapsun", "euldong"]),
    2: out(characters=[{"id": "masked", "name": "가면 쓴 사내", "appeared": True, "aliases": []}],
           facts=[fact("가면 쓴 사내가 밤에 서고에 숨어들었다.", "가면 쓴 사내가 밤에 서고에 숨어들었다.", [("masked", 2)]),
                  fact("갑순의 아버지는 북쪽에 있다.", "\"아버지는 북쪽에 계셔.\"", [("euldong", 0), ("gapsun", 5)])],
           pov="gapsun", present=["masked", "gapsun", "euldong"]),
    3: out(characters=[{"id": "byeongcheol", "name": "병철", "appeared": True, "aliases": []}],
           merges=[{"temp": "masked", "into": "byeongcheol"}],
           facts=[fact("가면 쓴 사내는 병철이었다.", "가면 쓴 사내는 병철이었다.", [("gapsun", 3), ("byeongcheol", 0)], "identity")],
           learned=[{"fact": "c2-1", "character": "euldong", "since": 3, "quote": "갑순은 그 사실을 을동에게 말했다."}],
           pov="gapsun", present=["byeongcheol", "gapsun"]),
}


class FakeStream:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.msg


class FakeSDK:
    """요청을 기록하고, 준비한 응답을 차례로 돌려준다. 응답이 모자라면 실패한다(카세트 재생 확인용)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.messages = self

    def stream(self, **req):
        self.requests.append(req)
        return FakeStream(self.replies.pop(0))


def msg(payload, stop="end_turn", read=0, write=0) -> SimpleNamespace:
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    usage = SimpleNamespace(input_tokens=1000, output_tokens=500, cache_read_input_tokens=read, cache_creation_input_tokens=write)
    return SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
                           stop_reason=stop, usage=usage, _request_id="req_test")


@pytest.fixture
def root(tmp_path):
    d = tmp_path / "tw"
    (d / "clean").mkdir(parents=True)
    (d / "work.json").write_text(json.dumps({"id": "tw", "title": "시험작", "author": "AI", "label": "시험", "chapters": 3},
                                            ensure_ascii=False), encoding="utf-8")
    for c, text in TEXTS.items():
        (d / "clean" / f"ch{c:02d}.txt").write_text(text, encoding="utf-8")
    return tmp_path


def go(root, tmp_path, replies, **kw):
    sdk = FakeSDK(replies)
    kw = {"cassette": tmp_path / "cassette.jsonl", "max_usd": 1.0, **kw}
    report = run("tw", "claude-opus-5-5", 3, tmp_path / "out", root=root, sdk=sdk, **kw)
    return report, sdk


def test_schema_matches_the_parsed_model():
    def keys(schema):
        return set(schema["properties"]) if schema.get("type") == "object" else set()

    assert keys(SCHEMA) == set(ChapterOut.model_fields)
    for name, model in (("characters", "NewCharacter"), ("facts", "FactOut"), ("learned", "Learned")):
        item = SCHEMA["properties"][name]["items"]
        assert set(item["required"]) == set(item["properties"]), name
    ChapterOut.model_validate(REPLIES[3])


def test_locate_ignores_spacing_and_quote_marks_but_not_words():
    assert locate("\"아버지는 북쪽에  계셔.\"", TEXTS[2]) == [3]
    assert locate("본문에 없는 문장", TEXTS[2]) is None
    assert locate("갑순", TEXTS[1]) is None  # 8자 미만은 근거로 보지 않는다
    assert locate("서약서를 베낀다. 을동이 찾아와", TEXTS[1]) == [1, 2, 3]


def test_request_keeps_the_record_append_only_with_two_cache_points():
    first = build_request("claude-opus-5-5", [], 1, TEXTS[1])
    third = build_request("claude-opus-5-5", ["[1장]\n…", "[2장]\n…"], 3, TEXTS[3])
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert [b.get("cache_control") for b in third["messages"][0]["content"]] == [None, {"type": "ephemeral"}, None]
    assert third["messages"][0]["content"][0]["text"] == "[1장]\n…"
    assert third["thinking"] == {"type": "adaptive"} and third["output_config"]["effort"] == "medium"
    sonnet = build_request("claude-sonnet-5-5", [], 1, TEXTS[1])
    assert sonnet["thinking"] == {"type": "between_tools"} and sonnet["output_config"]["effort"] == "high"


def test_run_builds_an_ontology_from_chapter_outputs(root, tmp_path):
    report, sdk = go(root, tmp_path, [msg(REPLIES[c]) for c in (1, 2, 3)])
    assert report["done"] == 3 and report["stopped"] is None and report["calls"] == 3
    assert report["problems"] == []
    # 1장: 인용이 본문에 없는 사실은 버리고, 없는 인물은 적어 둔다
    assert [r["statement"] for r in report["rejected"]] == ["갑순은 왕족이다."]
    assert "1장: 아는 인물이 없음: ghost" in report["notes"]
    assert "2장: 아는 장이 이번 장보다 뒤: gapsun 5" in report["notes"]
    # 기록은 장마다 한 블록씩 덧붙고, 다음 요청의 앞부분이 된다
    assert [len(r["messages"][0]["content"]) for r in sdk.requests] == [1, 2, 3]
    blocks = [[b["text"] for b in r["messages"][0]["content"][:-1]] for r in sdk.requests]
    assert blocks[2][:1] == blocks[1]  # 캐시 표시는 마지막 블록으로 옮겨 가도 앞 블록 본문은 그대로다
    record = (tmp_path / "out" / "record.txt").read_text(encoding="utf-8")
    assert "새 사실: c1-1 갑순은 은종탑의 필경사다. | 앎: gapsun(0) euldong(1)" in record
    assert "왕족" not in record

    onto = Ontology.model_validate_json((tmp_path / "out" / "ontology.json").read_text(encoding="utf-8"))
    assert check(onto) == []
    facts = {f.id: f for f in onto.facts}
    assert {k.character: k.from_ch for k in facts["c2-2"].known_by} == {"euldong": 0, "gapsun": 2}
    assert facts["c1-1"].evidence[0].lines == [1]
    # 3장: 을동이 가면 사내 이야기를 듣는다(learned)
    assert {k.character: k.from_ch for k in facts["c2-1"].known_by}["euldong"] == 3


def test_merged_identity_waits_for_its_reveal(root, tmp_path):
    go(root, tmp_path, [msg(REPLIES[c]) for c in (1, 2, 3)])
    onto = Ontology.model_validate_json((tmp_path / "out" / "ontology.json").read_text(encoding="utf-8"))
    chars = {c.id: c for c in onto.characters}
    assert "masked" not in chars  # 임시 인물은 원래 인물에 합친다
    assert [(a.form, a.from_ch) for a in chars["byeongcheol"].aliases] == [("가면 쓴 사내", 3)]
    known = {k.character: k.from_ch for k in next(f for f in onto.facts if f.id == "c2-1").known_by}
    assert known["byeongcheol"] == 3  # 2장부터 알았지만 정체가 드러난 3장부터로 늦춘다
    assert "c2-1" in [f.id for f in split_facts(onto, "byeongcheol", 3)[0]]
    assert character_view(onto, "byeongcheol", 3).name == "병철"


def test_common_facts_and_learned_since(root, tmp_path):
    people = [{"id": "gapsun", "name": "갑순", "appeared": True, "aliases": []},
              {"id": "euldong", "name": "을동", "appeared": True, "aliases": []}]
    first = out(characters=people, pov="gapsun", present=["gapsun", "euldong"], facts=[
        fact("은종탑의 필경사는 매일 서약서를 베낀다.", "그녀는 매일 서약서를 베낀다.", [], "world", common=True),
        fact("갑순은 은종탑의 필경사다.", "갑순은 은종탑의 필경사다.", [("gapsun", 0)], "identity")])
    told = "을동은 갑순에게 아버지 소식을 전했다."
    second = out(characters=[{"id": "byeongcheol", "name": "병철", "appeared": False, "aliases": []}],
                 learned=[{"fact": "c1-2", "character": "byeongcheol", "since": 0, "quote": told},  # 원래 알던 것
                          {"fact": "c1-2", "character": "euldong", "since": 9, "quote": told}],  # 뒤 장은 이번 장으로 자른다
                 pov="gapsun", present=["gapsun", "euldong"])
    report, _ = go(root, tmp_path, [msg(first), msg(second), msg(out(pov="gapsun", present=["gapsun"]))])
    assert report["done"] == 3 and "2장: learned의 아는 장이 이번 장보다 뒤: c1-2 euldong 9" in report["notes"]
    record = (tmp_path / "out" / "record.txt").read_text(encoding="utf-8")
    assert "새 사실: c1-1 은종탑의 필경사는 매일 서약서를 베낀다. | 상식" in record
    assert "새로 앎: c1-2 <- byeongcheol(0)" in record and "새로 앎: c1-2 <- euldong(2)" in record
    onto = Ontology.model_validate_json((tmp_path / "out" / "ontology.json").read_text(encoding="utf-8"))
    facts = {f.id: f for f in onto.facts}
    assert facts["c1-1"].common and facts["c1-1"].known_by == []
    assert {k.character: k.from_ch for k in facts["c1-2"].known_by} == {"gapsun": 0, "byeongcheol": 0, "euldong": 2}
    assert {f.id for f in split_facts(onto, "byeongcheol", 3)[0]} == {"c1-1", "c1-2"}


def test_cassette_replays_without_calling_again(root, tmp_path):
    first, _ = go(root, tmp_path, [msg(REPLIES[c]) for c in (1, 2, 3)])
    again, sdk = go(root, tmp_path, [])
    assert again["calls"] == 0 and again["spent_usd"] == 0 and sdk.requests == []
    assert again["done"] == 3 and again["chapters"][2]["output"] == first["chapters"][2]["output"]


def test_budget_and_call_limits_stop_before_spending(root, tmp_path):
    report, sdk = go(root, tmp_path, [msg(REPLIES[1])], max_usd=0.0001)
    assert report["done"] == 0 and "비용 상한" in report["stopped"] and sdk.requests == []
    assert (tmp_path / "out" / "report.json").exists()
    report, _ = go(root, tmp_path, [msg(REPLIES[c]) for c in (1, 2, 3)], cassette=None, max_calls=1)
    assert report["done"] == 1 and "호출 상한" in report["stopped"]


def test_first_call_guess_counts_a_cold_cache_rewrite():
    from sfqa.onto_extract import Budget, Extractor, first_call_guess

    long_record = ["[1장]\n" + "가" * 80_000]  # 47장째처럼 기록이 길다
    req = build_request("claude-opus-5-5", long_record, 48, TEXTS[1])
    assert first_call_guess(req) > 0.75  # 캐시 쓰기 약 0.44달러 + 출력 상한 0.32달러
    ex = Extractor(None, sdk=FakeSDK([]), max_usd=0.68)
    with pytest.raises(Budget, match="비용 상한"):
        ex.complete(req)  # 10/24에는 이 어림이 없어 0.605달러짜리 호출이 지나갔다


def test_truncated_reply_is_retried_with_a_larger_output_limit(root, tmp_path):
    from sfqa.onto_extract import LONG_MAX_TOKENS, MAX_TOKENS

    cut = msg("{\"characters\": [", stop="max_tokens")
    report, sdk = go(root, tmp_path, [cut, msg(REPLIES[1]), msg(REPLIES[2]), msg(REPLIES[3])])
    assert report["done"] == 3 and [t["error"] for t in report["chapters"][0]["tries"]] == ["finish_reason=length", None]
    assert [r["max_tokens"] for r in sdk.requests] == [MAX_TOKENS, LONG_MAX_TOKENS, MAX_TOKENS, MAX_TOKENS]
    again, sdk = go(root, tmp_path, [])  # 첫 요청은 그대로라 다시 돌리면 잘린 응답부터 카세트에서 재생한다
    assert again["done"] == 3 and sdk.requests == []


def test_refusal_is_retried_once_then_stops(root, tmp_path):
    report, _ = go(root, tmp_path, [msg("", stop="refusal"), msg(REPLIES[1]), msg(REPLIES[2]), msg(REPLIES[3])])
    assert report["done"] == 3 and [t["error"] for t in report["chapters"][0]["tries"]] == ["finish_reason=refusal", None]
    assert report["calls"] == 4  # 거절된 호출도 비용이 들었으므로 센다
    report, _ = go(root, tmp_path, [msg("{\"broken\": true}"), msg("", stop="max_tokens")], cassette=None)
    assert report["done"] == 0 and report["stopped"] == "1장 두 번 실패"
    assert report["chapters"][0]["tries"][0]["error"].startswith("형식 오류")
