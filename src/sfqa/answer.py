"""답변 체인(LCEL): 검색 → 프롬프트 → 모델(카세트 경유) → 해석 → 검증.

조건(docs/eval-spec.md):
- C0 검색 없음 + 가드 프롬프트
- C1 19장 전체 검색 + 기본 프롬프트
- C2 N장 이하 검색 + 기본 프롬프트
- C3 N장 이하 검색 + 가드 프롬프트
- C4 C3와 같은 요청(카세트를 다시 씀) + 코드 검증(verify.py)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable, RunnableLambda

from sfqa.config import Settings
from sfqa.llm import LiveClient, LLMRequest, LLMResponse, ReplayClient
from sfqa.prompts import system_prompt, user_prompt
from sfqa.retrieve import ChapterIndex, UptoRetriever
from sfqa.schema import ANSWER_SCHEMA, parse_answer
from sfqa.verify import Verifier

LAST_CHAPTER = 19
CONDITIONS = ("C0", "C1", "C2", "C3", "C4")


@dataclass
class Env:
    settings: Settings
    index: ChapterIndex
    embeddings: Embeddings
    client: LiveClient | ReplayClient
    verifier: Verifier
    k: int = 18


@dataclass
class AnswerResult:
    question: str
    n: int
    condition: str
    status: str  # 최종(가드 적용 뒤)
    answer: str
    model_status: str | None  # answered | not_found | None(해석 실패)
    model_answer: str
    quotes: list[dict] = field(default_factory=list)
    future_entities: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)  # [{chunk_id, chapter}]
    invalid: bool = False
    truncated: bool = False
    request_key: str = ""
    response: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def make_env(s: Settings, k: int = 18, client: LiveClient | ReplayClient | None = None) -> Env:
    from sfqa.gold import entity_index, load_entities, load_texts
    from sfqa.index import load_index
    from sfqa.llm import make_client, make_embeddings

    texts = load_texts()
    chunks, vectors, _ = load_index(s.embed_model, texts)
    verifier = Verifier(texts, entity_index(load_entities(), texts))
    return Env(s, ChapterIndex(chunks, vectors), make_embeddings(s), client or make_client(s), verifier, k)


def format_context(docs: list[Document]) -> str:
    """(장, 위치) 순서로 정렬해 [제7장 #12] 표지를 붙인다."""
    docs = sorted(docs, key=lambda d: (d.metadata["chapter"], d.metadata["start"]))
    return "\n\n".join(
        f"[제{d.metadata['chapter']}장 #{d.metadata['chunk_id'].split('-')[1]}]\n{d.page_content.strip()}" for d in docs
    )


def build_request(env: Env, condition: str, question: str, n: int, context: str | None) -> LLMRequest:
    s = env.settings
    return LLMRequest(
        model=s.llm_model, system=system_prompt(condition, n), user=user_prompt(question, context),
        schema_name="answer", schema=ANSWER_SCHEMA, max_tokens=s.max_tokens, seed=s.seed,
        reasoning_effort=s.reasoning_effort,
    )


def build_chain(env: Env, condition: str) -> Runnable:
    if condition not in CONDITIONS:
        raise ValueError(f"없는 조건이다: {condition}")

    def retrieve(x: dict) -> dict:
        if condition == "C0":
            return {**x, "docs": [], "context": None}
        upto = LAST_CHAPTER if condition == "C1" else x["n"]
        docs = UptoRetriever(index=env.index, embeddings=env.embeddings, n=upto, k=env.k).invoke(x["question"])
        return {**x, "docs": docs, "context": format_context(docs)}

    def call(x: dict) -> dict:
        req = build_request(env, condition, x["question"], x["n"], x["context"])
        return {**x, "req": req, "resp": env.client.complete(req)}

    def finish(x: dict) -> AnswerResult:
        resp: LLMResponse = x["resp"]
        data, _ = resp.parsed()
        parsed = parse_answer(data)
        refused = resp.finish_reason == "refusal"  # Claude 안전 거절: 형식 실패가 아니라 보류로 센다
        model_status = parsed.status if parsed else ("not_found" if refused else None)
        model_answer = parsed.answer if parsed else ""
        quotes = [(q.chapter, q.text) for q in parsed.quotes] if parsed else []
        v = env.verifier.verdict(model_status, model_answer, quotes, x["n"], x["context"] or "",
                                 apply_guard=condition == "C4")
        return AnswerResult(
            question=x["question"], n=x["n"], condition=condition, status=v.status, answer=v.answer,
            model_status=model_status, model_answer=model_answer, quotes=[asdict(q) for q in v.quotes],
            future_entities=v.future_entities, reasons=v.reasons,
            retrieved=[{"chunk_id": d.metadata["chunk_id"], "chapter": d.metadata["chapter"]} for d in x["docs"]],
            # 형식 실패: 잘린 것은 따로 세고, 그 밖에 답 형식으로 읽지 못한 것(빈 content 포함)은 모두 invalid다.
            invalid=parsed is None and not resp.truncated and not refused,
            truncated=resp.truncated, request_key=x["req"].key(), response=asdict(resp),
        )

    return RunnableLambda(retrieve) | RunnableLambda(call) | RunnableLambda(finish)
