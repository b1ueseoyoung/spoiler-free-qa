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
