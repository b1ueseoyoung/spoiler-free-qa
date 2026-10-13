"""로컬 웹 데모: 장을 고르고 질문하면 답과 근거 인용(장 번호)을 보여 준다.

- 이 맥(127.0.0.1)에서만 연다. 모델 서버도 loopback만 받는다(config.py).
- 모델과 색인은 첫 질문 때 한 번 올린다. LM Studio가 로컬이라 질문은 한 번에 하나씩 처리한다.
- condition: C4(최종 시스템)가 기본이다. C1(19장 전체 검색, 가드 없음)은 필터가 없을 때 스포일러가 새는 모습을
  비교해 보여 주려는 용도다.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from sfqa.answer import AnswerResult, Env, build_chain
from sfqa.config import check_loopback
from sfqa.paths import CHAPTERS, CLEAN_REPORT
from sfqa.score import KEPT

STATIC = Path(__file__).parent / "web_static"
QUESTION_MAX = 300


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=QUESTION_MAX)
    upto: int = Field(ge=min(CHAPTERS), le=max(CHAPTERS))
    condition: Literal["C4", "C1"] = "C4"


def chapters() -> list[dict]:
    if not CLEAN_REPORT.exists():
        return [{"chapter": c, "title": None} for c in CHAPTERS]
    report = json.loads(CLEAN_REPORT.read_text(encoding="utf-8"))
    return [{"chapter": ch["chapter"], "title": ch["toc_title"]} for ch in report["chapters"]]


def view(r: AnswerResult) -> dict:
    """화면에 보여 줄 것만. C4는 검증을 통과한 인용만, C1은 모델이 낸 인용을 검증 결과와 함께 보여 준다."""
    answered = r.status == "answered"
    quotes = []
    if answered:
        for q in r.quotes:
            if r.condition == "C4" and q["status"] not in KEPT:
                continue
            quotes.append({
                "chapter": q["found_chapter"] or q["chapter"],
                "text": q["text"],
                "verified": q["status"] in KEPT,
                "future": q["status"] == "future_only",
            })
    return {
        "status": r.status,
        "answer": r.answer,
        "upto": r.n,
        "condition": r.condition,
        "quotes": quotes,
        "retrieved_chapters": sorted({d["chapter"] for d in r.retrieved}),
        "future_entities": r.future_entities if r.condition == "C1" else [],
        "latency_s": r.response.get("latency_s"),
        "format_problem": r.invalid or r.truncated,
    }


def create_app(env_factory: Callable[[], Env]) -> FastAPI:
    app = FastAPI(title="스포일러 없는 「탁류」 Q&A", docs_url=None, redoc_url=None, openapi_url=None)
    # Pretendard(OFL, fonts/OFL.txt)를 앱이 직접 준다. 외부 글꼴 서버를 쓰지 않는다.
    app.mount("/fonts", StaticFiles(directory=STATIC / "fonts"), name="fonts")
    lock = threading.Lock()
    state: dict[str, Env | None] = {"env": None}

    def env() -> Env:
        if state["env"] is None:
            state["env"] = env_factory()
        return state["env"]

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/b")
    def variant_b() -> FileResponse:
        """디자인 B안(Botanical 레퍼런스). A안과 같은 API를 쓴다."""
        return FileResponse(STATIC / "b.html")

    @app.get("/api/chapters")
    def list_chapters() -> list[dict]:
        return chapters()

    @app.post("/api/ask")
    def ask(body: AskIn) -> dict:
        with lock:
            try:
                result = build_chain(env(), body.condition).invoke({"question": body.question.strip(), "n": body.upto})
            except Exception as e:  # 색인 없음, LM Studio 꺼짐 등은 화면에 알린다
                raise HTTPException(status_code=503, detail=f"답을 만들지 못했다: {type(e).__name__}: {e}") from e
        return view(result)

    return app


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    check_loopback(f"http://{host}:{port}")  # 데모는 이 맥에서만 연다
    import uvicorn

    from sfqa.answer import make_env
    from sfqa.config import load_settings

    uvicorn.run(create_app(lambda: make_env(load_settings())), host=host, port=port, log_level="info")
