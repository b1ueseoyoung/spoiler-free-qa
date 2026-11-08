import pytest
from fastapi.testclient import TestClient

from sfqa.web import create_app, serve
from tests.test_answer import WORK, FakeClient, answer_json, env


def client(raw: str | None = None) -> tuple[TestClient, FakeClient]:
    fake = FakeClient(raw if raw is not None else answer_json())
    return TestClient(create_app(lambda w: env(fake), works=[WORK])), fake


def test_index_and_chapters():
    c, _ = client()
    assert "스포일러 없는" in c.get("/").text
    chapters = c.get("/api/chapters").json()
    assert [x["chapter"] for x in chapters] == list(range(1, 20))


def test_ask_c4_returns_verified_quotes_only():
    c, fake = client()
    r = c.post("/api/ask", json={"question": "초봉이는 어디에 다니나?", "upto": 3}).json()
    assert r["status"] == "answered" and r["condition"] == "C4"
    assert r["quotes"] and all(q["verified"] for q in r["quotes"])
    assert max(r["retrieved_chapters"]) <= 3
    assert "제3장까지 읽었다" in fake.requests[0].system


def test_ask_c4_refuses_without_verified_quote():
    c, _ = client(answer_json(quotes=()))
    r = c.post("/api/ask", json={"question": "초봉이는 어디에 다니나?", "upto": 3}).json()
    assert r["status"] == "not_found" and r["answer"] == "3장까지는 나오지 않아요." and r["quotes"] == []


def test_ask_c1_compare_shows_raw_answer():
    c, fake = client(answer_json(answer="형보와 산다", quotes=()))
    r = c.post("/api/ask", json={"question": "초봉이는?", "upto": 3, "condition": "C1"}).json()
    assert r["status"] == "answered" and r["answer"] == "형보와 산다" and r["future_entities"] == ["형보"]
    assert "독자는 제" not in fake.requests[0].system


@pytest.mark.parametrize("body", [
    {"question": "", "upto": 3}, {"question": "x", "upto": 0}, {"question": "x", "upto": 20},
    {"question": "x" * 301, "upto": 3}, {"question": "x", "upto": 3, "condition": "C2"},
])
def test_bad_input_is_rejected(body):
    c, fake = client()
    assert c.post("/api/ask", json=body).status_code == 422
    assert fake.requests == []


def test_model_failure_is_reported_as_503():
    def broken(work):
        raise RuntimeError("LM Studio가 꺼져 있다")

    c = TestClient(create_app(broken, works=[WORK]))
    r = c.post("/api/ask", json={"question": "x", "upto": 3})
    assert r.status_code == 503 and "LM Studio" in r.json()["detail"]


def test_serve_refuses_non_loopback_host():
    with pytest.raises(ValueError):
        serve(host="0.0.0.0")


def test_page_has_no_external_resources():
    r = TestClient(create_app(lambda w: None, works=[WORK])).get("/")
    assert r.status_code == 200 and "/api/ask" in r.text
    assert "http://" not in r.text and "https://" not in r.text  # 글꼴·CDN 없이 로컬에서만 돈다


def test_pretendard_is_served_locally_with_license():
    c = TestClient(create_app(lambda w: None, works=[WORK]))
    font = c.get("/fonts/PretendardVariable.woff2")
    assert font.status_code == 200 and len(font.content) > 1_000_000
    assert "SIL OPEN FONT LICENSE" in c.get("/fonts/OFL.txt").text.upper()
    assert "/fonts/PretendardVariable.woff2" in c.get("/").text


# --- 공개 모드(docs/deploy-plan.md 4단계) ---
import json as _json  # noqa: E402

from sfqa.limits import Limits, MemoryCounter  # noqa: E402


def public_client(raw=None, limits=Limits(daily=100, ip_daily=100, ip_minute=2), env_factory=None):
    fake = FakeClient(raw if raw is not None else answer_json())
    counter = MemoryCounter(limits)
    app = create_app(env_factory or (lambda w: env(fake)), works=[WORK], counter=counter, public=True)
    return TestClient(app), fake, counter


CTX = {"x-amzn-request-context": _json.dumps({"http": {"sourceIp": "203.0.113.7"}})}


def test_public_mode_requires_counter():
    with pytest.raises(ValueError):
        create_app(lambda w: None, works=[WORK], public=True)


def test_config_reports_mode():
    assert TestClient(create_app(lambda w: None, works=[WORK])).get("/api/config").json() == {"public": False}
    c, _, _ = public_client()
    assert c.get("/api/config").json() == {"public": True}


def test_public_limit_blocks_before_model_call():
    c, fake, counter = public_client()
    body = {"question": "초봉이는 어디에 다니나?", "upto": 3}
    assert c.post("/api/ask", json=body, headers=CTX).status_code == 200
    assert c.post("/api/ask", json=body, headers=CTX).status_code == 200
    r = c.post("/api/ask", json=body, headers=CTX)
    assert r.status_code == 429 and "1분" in r.json()["detail"]
    assert len(fake.requests) == 2  # 막힌 요청은 모델을 부르지 않는다
    other = {"x-amzn-request-context": _json.dumps({"http": {"sourceIp": "198.51.100.1"}})}
    assert c.post("/api/ask", json=body, headers=other).status_code == 200  # IP별로 센다


def test_invalid_request_is_not_counted():
    c, _, counter = public_client()
    assert c.post("/api/ask", json={"question": "", "upto": 3}, headers=CTX).status_code == 422
    assert counter.counts == {}


def test_counter_failure_fails_closed():
    class Broken:
        def hit(self, ip):
            raise RuntimeError("dynamodb down")

    fake = FakeClient(answer_json())
    c = TestClient(create_app(lambda w: env(fake), works=[WORK], counter=Broken(), public=True))
    r = c.post("/api/ask", json={"question": "x", "upto": 3})
    assert r.status_code == 503 and "dynamodb" not in r.json()["detail"] and fake.requests == []


class ApiError(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


@pytest.mark.parametrize("err,status,text", [
    (ApiError(429, "Error code: 429 - enforced_spend_limit_reached"), 503, "사용 한도"),
    (ApiError(400, "Error code: 400 - workspace spend limit reached"), 503, "사용 한도"),
    (ApiError(429, "Error code: 429 - rate_limit_error"), 429, "몰려"),
    (RuntimeError("index missing at /secret/path"), 503, "잠시 뒤"),
])
def test_public_errors_are_mapped_without_internals(err, status, text):
    class Raising(FakeClient):
        def complete(self, req):
            raise err

    c, _, _ = public_client(env_factory=lambda w: env(Raising("")))
    r = c.post("/api/ask", json={"question": "x", "upto": 3}, headers=CTX)
    assert r.status_code == status and text in r.json()["detail"] and "/secret" not in r.json()["detail"]


def test_serve_refuses_public_mode_without_limits_table():
    with pytest.raises(ValueError, match="LIMITS_TABLE"):
        serve(env={"PUBLIC_DEMO": "1"})


def test_logs_do_not_contain_question(caplog):
    c, _, _ = public_client()
    with caplog.at_level("INFO", logger="sfqa.web"):
        c.post("/api/ask", json={"question": "아주비밀스러운질문", "upto": 3}, headers=CTX)
    assert caplog.records and all("아주비밀스러운질문" not in r.getMessage() for r in caplog.records)


def test_page_switches_notice_in_public_mode():
    html = TestClient(create_app(lambda w: None, works=[WORK])).get("/").text
    assert 'id="notice"' in html and "/api/config" in html and "회원가입 없이 무료" in html


def test_serve_public_with_memory_table_for_local_check(monkeypatch):
    seen = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: seen.update(app=app, **kw))
    serve(port=8766, env={"PUBLIC_DEMO": "1", "LIMITS_TABLE": "memory", "IP_MINUTE_LIMIT": "1"})
    assert seen["host"] == "127.0.0.1" and seen["port"] == 8766  # 공개 모드도 앱은 loopback에서 듣는다
    assert TestClient(seen["app"]).get("/api/config").json() == {"public": True}


def test_serve_turns_off_access_log_only_in_public_mode(monkeypatch):
    seen = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: seen.update(kw))
    serve(env={"PUBLIC_DEMO": "1", "LIMITS_TABLE": "memory"})
    assert seen["access_log"] is False  # 방문자 IP를 로그에 남기지 않는다
    serve(env={})
    assert seen["access_log"] is True


def test_warm_loads_env_without_model_call_or_counting():
    c, fake, counter = public_client()
    r = c.post("/api/warm")
    assert r.status_code == 200 and r.json() == {"ready": True}
    assert fake.requests == [] and counter.counts == {}


def test_works_api_lists_examples_with_their_first_visible_chapter():
    c, _ = client()
    cards = c.get("/api/works").json()
    assert cards[0]["id"] == "takryu" and cards[0]["examples"] == [{"question": "초봉이는?", "from": 1}]


# 실제 작품 데이터: 첫 화면 문구가 앞 장 범위를 넘지 않는지(리뷰에서 확인, 2026-10-14)
from sfqa.gold import entity_index, load_entities, load_texts  # noqa: E402
from sfqa.web import example_from  # noqa: E402
from sfqa.works import list_works  # noqa: E402

READY = [w for w in list_works(ready_only=True) if w.entities_path.exists()]


@pytest.mark.parametrize("work", READY, ids=[w.id for w in READY])
def test_examples_show_by_default_and_blurb_uses_only_chapter_one_names(work):
    examples = example_from(work)
    assert examples and all(e["from"] <= work.default_upto for e in examples), examples  # 기본 화면에서 다 보인다
    first = entity_index(load_entities(work), load_texts(work))
    late = [n for n, c in first.items() if c is not None and c > 1 and n in work.blurb]
    assert late == [], f"첫 화면 소개에 1장 뒤 인물: {late}"


@pytest.mark.parametrize("work", READY, ids=[w.id for w in READY])
def test_extra_entities_made_it_into_the_table(work):
    names = {f for e in load_entities(work) for f in (e.name, *e.aliases)}
    assert set(work.source.get("extra_entities", [])) <= names


def test_hidden_attribute_always_wins():
    """첫 화면에서 대화 화면(.grid, hidden)이 같이 보이던 문제. CSS display 지정이 hidden을 덮으면 안 된다."""
    html = TestClient(create_app(lambda w: None, works=[WORK])).get("/").text
    assert "[hidden] { display: none !important; }" in html


# --- 인물 채팅(docs/character-chat.md) ---
from sfqa.chat import FALLBACK, MINOR_NOTICE, ChatService  # noqa: E402
from sfqa.config import load_settings  # noqa: E402
from sfqa.llm import LLMResponse  # noqa: E402

CHAT_SERVICE = {}


class ChatFake:
    def __init__(self, reply="숫자부터 말하시오."):
        self.reply, self.requests = reply, []

    def complete(self, req):
        self.requests.append(req)
        return LLMResponse(raw_text=_json.dumps({"reply": self.reply}, ensure_ascii=False), finish_reason="stop",
                           input_tokens=10, output_tokens=5, model=req.model)


def chat_client(reply="숫자부터 말하시오."):
    fake = ChatFake(reply)
    service = ChatService(lambda: (load_settings({"LLM_PROVIDER": "anthropic"}), fake))
    # 온톨로지와 검사기는 테스트 사이에 같이 쓴다(본문을 읽는 데 시간이 든다)
    service._envs = CHAT_SERVICE
    return TestClient(create_app(lambda w: None, works=[WORK], chat=service)), fake


CHAT_BODY = {"work": "wn01-rofan", "character": "cassiel", "upto": 10, "message": "빚이 얼마나 돼요?"}


def test_chat_works_are_the_synthetic_ones_and_qa_list_is_untouched():
    c, _ = chat_client()
    assert [w["id"] for w in c.get("/api/chat/works").json()] == ["wn01-rofan", "wn02-hyeonpan", "wn03-muhyeop", "wn04-romance"]
    assert [w["id"] for w in c.get("/api/works").json()] == ["takryu"]


def test_chat_characters_follow_the_chapter():
    c, _ = chat_client()
    ids = [p["id"] for p in c.get("/api/chat/characters", params={"work": "wn01-rofan", "upto": 1}).json()]
    assert "cecilia" in ids and "noel" not in ids and "ondre" not in ids
    people = c.get("/api/chat/characters", params={"work": "wn01-rofan", "upto": 6}).json()
    assert next(p for p in people if p["id"] == "noel")["romance"] is False
    assert next(p for p in people if p["id"] == "cassiel")["romance"] is True
    assert set(people[0]) == {"id", "name", "aliases", "romance"}  # 역할·나이·첫 등장 장은 내보내지 않는다
    for params in ({"work": "wn01-rofan", "upto": 0}, {"work": "wn01-rofan", "upto": 51}, {"work": "takryu", "upto": 3}):
        assert c.get("/api/chat/characters", params=params).status_code == 422


def test_chat_reply_uses_history_and_says_nothing_about_checks():
    c, fake = chat_client()
    body = {**CHAT_BODY, "history": [{"role": "reader", "text": "안녕하세요"}, {"role": "character", "text": "무슨 일이오."}]}
    r = c.post("/api/chat", json=body)
    assert r.status_code == 200 and r.json()["reply"] == "숫자부터 말하시오." and r.json()["status"] == "ok"
    assert set(r.json()) == {"status", "reply", "upto", "latency_s"}
    assert "상대: 안녕하세요" in fake.requests[0].user and "나: 무슨 일이오." in fake.requests[0].user
    assert "카시엘 로스벤" in fake.requests[0].system


def test_chat_replaced_reply_looks_like_any_other():
    c, fake = chat_client("회색 깃털은 왕세자 테오도르요.")
    r = c.post("/api/chat", json={**CHAT_BODY, "character": "cecilia", "upto": 36})
    assert r.json()["reply"] == FALLBACK and r.json()["status"] == "ok" and set(r.json()) == {"status", "reply", "upto", "latency_s"}
    assert fake.requests


@pytest.mark.parametrize("body", [
    {**CHAT_BODY, "character": "noel", "upto": 4},  # 아직 등장하지 않음
    {**CHAT_BODY, "character": "ondre", "upto": 44},
    {**CHAT_BODY, "character": "rowan", "upto": 35},  # 실종 중
    {**CHAT_BODY, "character": "nobody"},
    {**CHAT_BODY, "work": "takryu"},
    {**CHAT_BODY, "upto": 51},
    {**CHAT_BODY, "upto": 0},
    {**CHAT_BODY, "message": ""},
    {**CHAT_BODY, "message": "   "},
    {**CHAT_BODY, "message": "x" * 301},
    {**CHAT_BODY, "history": [{"role": "reader", "text": "x"}] * 21},
    {**CHAT_BODY, "history": [{"role": "system", "text": "x"}]},
    {**CHAT_BODY, "character": "noel", "upto": 6, "romance": True},  # 미성년 인물의 로맨스 설정
])
def test_chat_bad_input_is_rejected_before_the_model(body):
    c, fake = chat_client()
    assert c.post("/api/chat", json=body).status_code == 422
    assert fake.requests == []


def test_chat_unavailable_reason_is_not_told_apart():
    c, _ = chat_client()
    details = {c.post("/api/chat", json={**CHAT_BODY, "character": who, "upto": n}).json()["detail"]
               for who, n in (("noel", 4), ("ondre", 44), ("rowan", 35), ("nobody", 10))}
    assert len(details) == 1  # 까닭을 구별해 주면 그 자체로 힌트가 된다


def test_chat_minor_notice_has_no_model_call():
    c, fake = chat_client()
    r = c.post("/api/chat", json={**CHAT_BODY, "character": "noel", "upto": 6, "message": "나랑 사귈래?"})
    assert r.status_code == 200 and r.json()["status"] == "notice" and r.json()["reply"] == MINOR_NOTICE and fake.requests == []


def test_chat_model_failure_is_reported():
    def no_key():
        raise RuntimeError("ANTHROPIC_API_KEY가 없다")

    service = ChatService(no_key)
    service._envs = CHAT_SERVICE
    c = TestClient(create_app(lambda w: None, works=[WORK], chat=service))
    r = c.post("/api/chat", json=CHAT_BODY)
    assert r.status_code == 503 and "ANTHROPIC_API_KEY" in r.json()["detail"]
    assert c.get("/api/chat/characters", params={"work": "wn01-rofan", "upto": 5}).status_code == 200  # 목록은 키 없이 열린다


def test_chat_logs_have_no_conversation_text(caplog):
    c, _ = chat_client()
    with caplog.at_level("INFO", logger="sfqa.web"):
        c.post("/api/chat", json={**CHAT_BODY, "message": "아주비밀스러운말"})
    assert caplog.records and all("아주비밀스러운말" not in r.getMessage() and "숫자부터" not in r.getMessage() for r in caplog.records)


def test_chat_is_closed_in_public_mode():
    with pytest.raises(ValueError):
        create_app(lambda w: None, works=[WORK], counter=MemoryCounter(Limits()), public=True, chat=ChatService(lambda: None))
    c, _, _ = public_client()
    assert c.get("/api/chat/works").status_code == 404
    assert c.post("/api/chat", json=CHAT_BODY, headers=CTX).status_code == 404


def test_serve_opens_chat_locally_and_keeps_it_closed_in_public_mode(monkeypatch):
    seen = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: seen.update(app=app))
    serve(env={})
    assert "/api/chat" in [r.path for r in seen["app"].routes]
    serve(env={"PUBLIC_DEMO": "1", "LIMITS_TABLE": "memory"})
    assert "/api/chat" not in [r.path for r in seen["app"].routes]


def test_page_has_talk_screen_and_hides_shelf_without_chat():
    html = TestClient(create_app(lambda w: None, works=[WORK])).get("/").text
    for needle in ('id="talk"', "/api/chat/characters", "/api/chat/works", 'id="people"', 'id="romance"', "AI가 연기하는 인물"):
        assert needle in html
    assert 'id="talk-shelf" hidden' in html  # 채팅이 꺼진 공개 모드에서는 칸이 안 보인다
