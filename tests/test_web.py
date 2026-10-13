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
    assert r["status"] == "not_found" and r["answer"] == "3장까지는 나오지 않습니다." and r["quotes"] == []


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
