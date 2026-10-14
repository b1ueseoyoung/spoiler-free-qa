import pytest
from fastapi.testclient import TestClient

from sfqa.web import create_app, serve
from tests.test_answer import FakeClient, answer_json, env


def client(raw: str | None = None) -> tuple[TestClient, FakeClient]:
    fake = FakeClient(raw if raw is not None else answer_json())
    return TestClient(create_app(lambda: env(fake))), fake


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
    def broken():
        raise RuntimeError("LM Studio가 꺼져 있다")

    c = TestClient(create_app(broken))
    r = c.post("/api/ask", json={"question": "x", "upto": 3})
    assert r.status_code == 503 and "LM Studio" in r.json()["detail"]


def test_serve_refuses_non_loopback_host():
    with pytest.raises(ValueError):
        serve(host="0.0.0.0")


def test_page_has_no_external_resources():
    r = TestClient(create_app(lambda: None)).get("/")
    assert r.status_code == 200 and "/api/ask" in r.text
    assert "http://" not in r.text and "https://" not in r.text  # 글꼴·CDN 없이 로컬에서만 돈다


def test_pretendard_is_served_locally_with_license():
    c = TestClient(create_app(lambda: None))
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
    app = create_app(env_factory or (lambda: env(fake)), counter=counter, public=True)
    return TestClient(app), fake, counter


CTX = {"x-amzn-request-context": _json.dumps({"http": {"sourceIp": "203.0.113.7"}})}


def test_public_mode_requires_counter():
    with pytest.raises(ValueError):
        create_app(lambda: None, public=True)


def test_config_reports_mode():
    assert TestClient(create_app(lambda: None)).get("/api/config").json() == {"public": False}
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
    c = TestClient(create_app(lambda: env(fake), counter=Broken(), public=True))
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

    c, _, _ = public_client(env_factory=lambda: env(Raising("")))
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
    html = TestClient(create_app(lambda: None)).get("/").text
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
