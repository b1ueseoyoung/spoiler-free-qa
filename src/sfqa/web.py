"""로컬 웹 데모: 장을 고르고 질문하면 답과 근거 인용(장 번호)을 보여 준다.

- 이 맥(127.0.0.1)에서만 연다. 모델 서버도 loopback만 받는다(config.py).
- 모델과 색인은 첫 질문 때 한 번 올린다. LM Studio가 로컬이라 질문은 한 번에 하나씩 처리한다.
- condition: C4(최종 시스템)가 기본이다. C1(19장 전체 검색, 가드 없음)은 필터가 없을 때 스포일러가 새는 모습을
  비교해 보여 주려는 용도다.
- 공개 모드(PUBLIC_DEMO=1, docs/deploy-plan.md 4단계): Lambda Web Adapter 뒤에서 돈다(앱은 여전히 127.0.0.1에서 듣는다).
  - 모델을 부르기 전에 호출 상한(limits.py, DynamoDB)을 센다. 상한 표(LIMITS_TABLE)가 없으면 시작하지 않는다.
  - Anthropic 지출 한도·속도 제한은 안내 문구로 바꾸고, 그 밖의 내부 오류 내용은 화면에 보이지 않는다.
  - 로그에는 질문 본문을 남기지 않는다(시각·상태·토큰 수만).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from sfqa.answer import AnswerResult, Env, build_chain
from sfqa.config import check_loopback
from sfqa.paths import CHAPTERS, CLEAN_REPORT
from sfqa.score import KEPT

STATIC = Path(__file__).parent / "web_static"
QUESTION_MAX = 300
log = logging.getLogger("sfqa.web")
SPEND_LIMIT_MESSAGE = "이번 달 데모 사용 한도에 닿았습니다. 다음 달에 다시 열립니다."
BUSY_MESSAGE = "요청이 몰려 잠시 답할 수 없습니다. 잠시 뒤 다시 물어 주세요."
PUBLIC_ERROR_MESSAGE = "답을 만들지 못했습니다. 잠시 뒤 다시 시도해 주세요."


def model_error(e: Exception, public: bool) -> tuple[int, str]:
    """모델 호출 실패를 (HTTP 상태, 화면 문구)로. Anthropic 지출 한도는 400(직접 정한 한도)이나
    429 enforced_spend_limit_reached(티어 상한)로 온다."""
    status = getattr(e, "status_code", None)
    text = str(e).lower()
    if "spend_limit" in text or "spend limit" in text:
        return 503, SPEND_LIMIT_MESSAGE
    if status == 429:
        return 429, BUSY_MESSAGE
    if public:
        return 503, PUBLIC_ERROR_MESSAGE
    return 503, f"답을 만들지 못했다: {type(e).__name__}: {e}"


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


def create_app(env_factory: Callable[[], Env], counter=None, public: bool = False) -> FastAPI:
    """counter: limits.MemoryCounter나 DynamoCounter. 공개 모드에서는 반드시 있어야 한다."""
    if public and counter is None:
        raise ValueError("공개 모드에는 호출 상한(counter)이 있어야 한다")
    app = FastAPI(title="스포일러 없는 「탁류」 Q&A", docs_url=None, redoc_url=None, openapi_url=None)
    # Pretendard(OFL, fonts/OFL.txt)를 앱이 직접 준다. 외부 글꼴 서버를 쓰지 않는다.
    app.mount("/fonts", StaticFiles(directory=STATIC / "fonts"), name="fonts")
    lock = threading.Lock()
    state: dict[str, Env | None] = {"env": None}

    def env() -> Env:
        if state["env"] is None:
            started = time.monotonic()
            e = env_factory()
            if e is not None and hasattr(e.embeddings, "embed_query"):
                e.embeddings.embed_query("준비")  # 임베딩 모델을 여기서 올린다(준비 시간을 한 번에 잰다)
            log.info(json.dumps({"event": "env_ready", "seconds": round(time.monotonic() - started, 2)}))
            state["env"] = e
        return state["env"]

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/chapters")
    def list_chapters() -> list[dict]:
        return chapters()

    @app.get("/api/config")
    def config() -> dict:
        return {"public": public}

    @app.post("/api/warm")
    def warm() -> dict:
        """모델·색인만 올린다(모델 호출 없음, 상한에 세지 않음). 배포 직후 scripts/deploy.sh가 한 번 부른다.
        Lambda는 배포 직후 첫 실행에서 이미지 파일을 처음 내려받느라 1분 넘게 걸린다(2026-10-14 측정 94초)."""
        with lock:
            try:
                env()
            except Exception as e:
                log.error(json.dumps({"event": "warm_error", "error": type(e).__name__}))
                raise HTTPException(status_code=503, detail=PUBLIC_ERROR_MESSAGE if public else str(e)) from None
        return {"ready": True}

    @app.post("/api/ask")
    def ask(body: AskIn, request: Request) -> dict:
        if counter is not None:
            from sfqa.limits import LimitExceeded, client_ip

            ip = client_ip(request.headers, request.client.host if request.client else None, trust_lambda=public)
            try:
                counter.hit(ip)
            except LimitExceeded as e:
                log.info(json.dumps({"event": "limited", "scope": e.scope}))
                raise HTTPException(status_code=429, detail=e.message) from None
            except Exception as e:  # 상한을 확인하지 못하면 부르지 않는다(비용 쪽으로 안전하게)
                log.error(json.dumps({"event": "limit_error", "error": type(e).__name__}))
                raise HTTPException(status_code=503, detail=PUBLIC_ERROR_MESSAGE) from None
        with lock:
            try:
                result = build_chain(env(), body.condition).invoke({"question": body.question.strip(), "n": body.upto})
            except Exception as e:  # 색인 없음, 모델 서버 꺼짐, 지출 한도 등은 화면에 알린다
                status, detail = model_error(e, public)
                log.error(json.dumps({"event": "ask_error", "error": type(e).__name__, "status": status}))
                raise HTTPException(status_code=status, detail=detail) from e
        resp = result.response
        log.info(json.dumps({"event": "ask", "condition": result.condition, "status": result.status,
                             "latency_s": resp.get("latency_s"), "input_tokens": resp.get("input_tokens"),
                             "output_tokens": resp.get("output_tokens")}))
        return view(result)

    return app


def serve(host: str = "127.0.0.1", port: int = 8765, env: dict | None = None) -> None:
    """로컬은 이 맥에서만 연다. 공개 모드(PUBLIC_DEMO=1)도 앱은 127.0.0.1에서 듣고, Lambda Web Adapter가 넘겨 준다."""
    env = os.environ if env is None else env
    check_loopback(f"http://{host}:{port}")
    public = env.get("PUBLIC_DEMO") == "1"
    counter = None
    if public:
        from sfqa.limits import DynamoCounter, MemoryCounter, load_limits

        table = env.get("LIMITS_TABLE")
        if not table:
            raise ValueError("PUBLIC_DEMO=1이면 LIMITS_TABLE이 있어야 한다(호출 상한 없이 공개하지 않는다)")
        if table == "memory":  # 이 맥에서 공개 모드 화면을 확인할 때만. Lambda에서는 실행 환경마다 따로 세서 쓰면 안 된다
            counter = MemoryCounter(load_limits(env), env.get("IP_HASH_SALT", ""))
        else:
            counter = DynamoCounter(table, load_limits(env), env.get("IP_HASH_SALT", ""))
    import uvicorn

    from sfqa.answer import make_env
    from sfqa.config import load_settings

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app = create_app(lambda: make_env(load_settings()), counter=counter, public=public)
    uvicorn.run(app, host=host, port=port, log_level="info")
