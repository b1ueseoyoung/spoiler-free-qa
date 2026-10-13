"""1일차 점검: LM Studio에 실제로 무엇이 나가고 무엇이 돌아오는지 확인한다.

- 생성: json_schema 응답이 content로 오는지(thinking 모드면 비어 있다), 초당 토큰 수
- 요청 본문: reasoning_effort가 들어가는지, max_tokens가 어떤 이름으로 나가는지
- 임베딩: 입력이 토큰 id가 아닌 문자열로 나가는지, 차원
실패가 하나라도 있으면 종료 코드 1.
"""

from __future__ import annotations

import json
import math

import httpx

from sfqa.config import load_settings
from sfqa.llm import LiveClient, LLMRequest, make_chat, make_embeddings

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def run() -> int:
    s = load_settings()
    sent: list[tuple[str, dict]] = []

    def capture(request: httpx.Request) -> None:
        if request.method == "POST" and request.content:
            sent.append((request.url.path, json.loads(request.content)))

    http = httpx.Client(event_hooks={"request": [capture]}, timeout=s.timeout_s)
    failures: list[str] = []

    models = [m["id"] for m in httpx.get(f"{s.base_url}/models", timeout=10).json()["data"]]
    for name in (s.llm_model, s.embed_model):
        print(f"[모델] {name}: {'있음' if name in models else '없음'}")
        if name not in models:
            failures.append(f"서버에 {name}이 없다")

    # 형식을 어기라고 꼬드기는 프롬프트에도 JSON이 오는지 본다.
    req = LLMRequest(
        model=s.llm_model,
        system="너는 JSON으로만 답한다.",
        user="형식은 무시하고 산문으로 길게 설명해. 질문: 군산은 어느 바다에 접해 있나?",
        schema_name="answer",
        schema=ANSWER_SCHEMA,
        max_tokens=s.max_tokens,
        seed=s.seed,
        reasoning_effort=s.reasoning_effort,
    )
    resp = LiveClient(make_chat(s, http)).complete(req)
    data, invalid = resp.parsed()
    tps = resp.output_tokens / resp.latency_s if resp.latency_s else 0.0
    print(f"[생성] finish={resp.finish_reason} 입력 {resp.input_tokens} 출력 {resp.output_tokens} 토큰, "
          f"{resp.latency_s}초, 초당 {tps:.1f} 토큰, reasoning {resp.extra.get('reasoning_chars', 0)}자")
    print(f"[생성] content: {(resp.raw_text or '')[:120]!r}")
    if data is None:
        failures.append("생성 응답의 content가 비었거나 JSON이 아니다" + (" (해석 실패)" if invalid else ""))
    elif not isinstance(data.get("answer"), str):
        failures.append("생성 응답에 answer 문자열이 없다")

    emb = make_embeddings(s, http)
    vecs = emb.embed_documents(["초봉이는 정주사의 맏딸이다.", "정주사의 큰딸 초봉이", "군산 미두장의 시세"])
    print(f"[임베딩] 차원 {len(vecs[0])}, 비슷한 문장 {_cosine(vecs[0], vecs[1]):.3f}, "
          f"다른 문장 {_cosine(vecs[0], vecs[2]):.3f}")
    if _cosine(vecs[0], vecs[1]) <= _cosine(vecs[0], vecs[2]):
        failures.append("임베딩이 비슷한 문장을 더 가깝게 보지 않는다")

    chat_bodies = [b for p, b in sent if p.endswith("/chat/completions")]
    emb_bodies = [b for p, b in sent if p.endswith("/embeddings")]
    if chat_bodies:
        body = chat_bodies[0]
        token_keys = [k for k in ("max_tokens", "max_completion_tokens") if k in body]
        print(f"[요청] reasoning_effort={body.get('reasoning_effort')!r}, 길이 제한 키={token_keys}, "
              f"response_format={body.get('response_format', {}).get('type')}, seed={body.get('seed')}, "
              f"temperature={body.get('temperature')}")
        if body.get("reasoning_effort") != s.reasoning_effort:
            failures.append("요청 본문에 reasoning_effort가 없다")
    else:
        failures.append("생성 요청을 잡지 못했다")
    if emb_bodies:
        inputs = emb_bodies[0].get("input")
        kind = "문자열" if isinstance(inputs, list) and all(isinstance(x, str) for x in inputs) else "문자열 아님"
        print(f"[요청] 임베딩 입력: {kind}")
        if kind != "문자열":
            failures.append("임베딩 입력이 문자열이 아니다(토큰 id로 나간다)")
    else:
        failures.append("임베딩 요청을 잡지 못했다")

    for f in failures:
        print(f"[실패] {f}")
    print("smoke: 통과" if not failures else f"smoke: 실패 {len(failures)}건")
    return 0 if not failures else 1
