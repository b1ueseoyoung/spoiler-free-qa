"""LM Studio(OpenAI 호환) 연결과 카세트.

카세트는 모델 응답을 JSONL로 기록해 두고 같은 요청이면 다시 쓴다(munjigi/llm.py에서 옮김).
테스트는 replay 모드로 돌려 모델 없이 통과한다.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
import openai
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai.types.chat import ChatCompletion

from sfqa.config import Settings, check_loopback

KEY_VERSION = 1
LM_STUDIO_KEY = "lm-studio"  # LM Studio는 키를 검사하지 않는다. 비밀 값이 아니다.
_THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def strip_think(text: str) -> str:
    """앞에 붙은 <think>…</think> 블록을 지운다(thinking을 꺼도 붙어 오는 경우가 있다)."""
    return _THINK.sub("", text, count=1)


def parse_structured(raw_text: str | None) -> tuple[dict | None, bool]:
    """(data, 해석 실패 여부). 빈 응답은 실패가 아니라 data 없음으로 본다."""
    if not raw_text or not raw_text.strip():
        return None, False
    try:
        data = json.loads(strip_think(raw_text))
    except json.JSONDecodeError:
        return None, True
    return (data, False) if isinstance(data, dict) else (None, True)


@dataclass(frozen=True)
class LLMRequest:
    model: str
    system: str
    user: str
    schema_name: str
    schema: dict
    max_tokens: int
    seed: int
    reasoning_effort: str
    temperature: float = 0.0

    def key(self) -> str:
        """실제로 보내는 설정 전체와 키 버전으로 만든 카세트 키."""
        body = {**asdict(self), "v": KEY_VERSION}
        return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


@dataclass
class LLMResponse:
    raw_text: str | None
    finish_reason: str | None
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    latency_s: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"

    def parsed(self) -> tuple[dict | None, bool]:
        return parse_structured(self.raw_text)


def _load_cassette(path: Path | None) -> dict[str, LLMResponse]:
    responses: dict[str, LLMResponse] = {}
    if path is None or not path.exists():
        return responses
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        resp = LLMResponse(**row["response"])
        if row["key"] in responses and asdict(responses[row["key"]]) != asdict(resp):
            raise ValueError(f"카세트에 같은 키의 다른 응답이 있다: {row['key'][:12]}")
        responses[row["key"]] = resp
    return responses


def _from_completion(completion: ChatCompletion, req: LLMRequest, elapsed: float) -> LLMResponse:
    choice = completion.choices[0]
    usage = completion.usage
    return LLMResponse(
        raw_text=choice.message.content,
        finish_reason=choice.finish_reason,
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
        model=completion.model or req.model,
        latency_s=round(elapsed, 3),
    )


def _messages(req: LLMRequest) -> list:
    return [SystemMessage(content=req.system), HumanMessage(content=req.user)]


def _response_format(req: LLMRequest) -> dict:
    return {"type": "json_schema", "json_schema": {"name": req.schema_name, "strict": True, "schema": req.schema}}


class LiveClient:
    """LangChain 채팅 모델을 부른다. 기록할 카세트가 있으면 해석보다 기록을 먼저 한다."""

    def __init__(self, chat: BaseChatModel, cassette: Path | None = None):
        self.chat = chat
        self.cassette = cassette
        self._cache = _load_cassette(cassette)
        self.calls = 0  # 실제로 모델을 부른 횟수

    def has(self, req: LLMRequest) -> bool:
        """카세트나 이번 실행에서 이미 받은 응답이 있는가(있으면 모델을 부르지 않는다)."""
        return req.key() in self._cache

    def complete(self, req: LLMRequest) -> LLMResponse:
        key = req.key()
        if key in self._cache:
            return self._cache[key]
        started = time.monotonic()
        try:
            msg = self.chat.bind(response_format=_response_format(req)).invoke(_messages(req))
        except (openai.LengthFinishReasonError, openai.ContentFilterFinishReasonError) as e:
            # json_schema 경로에서는 잘린 응답이 예외로 온다. 잘린 것도 기록하고 따로 센다.
            self.calls += 1
            resp = _from_completion(e.completion, req, time.monotonic() - started)
        else:
            self.calls += 1
            usage = msg.usage_metadata or {}
            meta = msg.response_metadata or {}
            reasoning = (msg.additional_kwargs or {}).get("reasoning_content")
            resp = LLMResponse(
                raw_text=msg.content if isinstance(msg.content, str) else json.dumps(msg.content, ensure_ascii=False),
                finish_reason=meta.get("finish_reason"),
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                model=meta.get("model_name", req.model),
                latency_s=round(time.monotonic() - started, 3),
                extra={"reasoning_chars": len(reasoning) if reasoning else 0},
            )
        if self.cassette:
            self.cassette.parent.mkdir(parents=True, exist_ok=True)
            with self.cassette.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "response": asdict(resp)}, ensure_ascii=False) + "\n")
        self._cache[key] = resp
        return resp


class ReplayClient:
    """카세트에 있는 응답만 돌려준다. 없으면 네트워크로 가지 않고 실패한다."""

    def __init__(self, cassette: Path):
        self._responses = _load_cassette(cassette)
        self.calls = 0

    def has(self, req: LLMRequest) -> bool:
        return req.key() in self._responses

    def complete(self, req: LLMRequest) -> LLMResponse:
        if req.key() not in self._responses:
            raise KeyError(f"카세트에 없는 요청이다: {req.key()[:12]}")
        return self._responses[req.key()]


def make_chat(s: Settings, http_client: httpx.Client | None = None) -> ChatOpenAI:
    check_loopback(s.base_url)
    return ChatOpenAI(
        model=s.llm_model,
        base_url=s.base_url,
        api_key=LM_STUDIO_KEY,
        temperature=0,
        seed=s.seed,
        max_tokens=s.max_tokens,
        timeout=s.timeout_s,
        max_retries=1,
        use_responses_api=False,
        # LangChain의 모델명별 처리를 거치지 않고 그대로 보낸다(gpt-oss는 none을 받지 않아 low를 쓴다).
        # 빈 값이면 보내지 않는다(thinking이 없는 Gemma).
        extra_body={"reasoning_effort": s.reasoning_effort} if s.reasoning_effort else None,
        http_client=http_client,
    )


# llama.cpp의 pooling 종류(enum llama_pooling_type). KURE-v1은 CLS다.
LLAMA_POOLING = {0: "none", 1: "mean", 2: "cls", 3: "last", 4: "rank"}


def resolve_gguf(model: str) -> Path:
    """'<저장소>/<파일>.gguf' 이름을 실제 파일로. LM Studio 모델 폴더(GGUF_MODELS_DIR)에 있으면 그것을 쓰고,
    없으면 Hugging Face에서 받는다(huggingface-hub 캐시). 배포 이미지에는 GGUF_MODELS_DIR에 미리 넣어 둔다."""
    root = Path(os.environ.get("GGUF_MODELS_DIR", Path.home() / ".lmstudio" / "models"))
    if (root / model).exists():
        return root / model
    repo, _, filename = model.rpartition("/")
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo, filename=filename))


class LocalEmbeddings(Embeddings):
    """GGUF 임베딩 모델을 llama.cpp(llama-cpp-python)로 이 프로세스 안에서 실행한다(EMBED_PROVIDER=local, 공개 데모용).

    LM Studio가 쓰는 것과 같은 파일(mykor/KURE-v1-gguf의 Q8_0)이다. 모델은 처음 쓸 때 올린다.
    pooling이 expect_pooling과 다르면 색인과 질의 벡터가 어긋나므로 바로 실패한다.
    """

    def __init__(self, model_name: str, expect_pooling: str | None = "cls", model=None, n_ctx: int = 2048):
        self.model_name, self.expect_pooling, self.n_ctx = model_name, expect_pooling, n_ctx
        self._model = model

    @property
    def model(self):
        if self._model is None:
            from llama_cpp import Llama  # 컴파일된 확장이라 쓸 때만 불러온다

            # mmap(기본)을 쓴다. Lambda에서 use_mmap=False로 한 번에 읽으면 첫 준비가 44초에서 80초로 늘고
            # 메모리도 1.9GB까지 올라갔다(2026-10-14).
            self._model = Llama(model_path=str(resolve_gguf(self.model_name)), embedding=True, n_ctx=self.n_ctx,
                                n_batch=self.n_ctx, n_ubatch=self.n_ctx, verbose=False)
        if self.expect_pooling:
            mode = LLAMA_POOLING.get(self._model.pooling_type())
            if mode != self.expect_pooling:
                raise ValueError(f"{self.model_name}의 pooling이 {mode}다({self.expect_pooling}여야 한다)")
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, self.model.embed(t))) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def make_embeddings(s: Settings, http_client: httpx.Client | None = None) -> Embeddings:
    if s.embed_provider == "local":
        return LocalEmbeddings(s.embed_model)
    check_loopback(s.base_url)
    return OpenAIEmbeddings(
        model=s.embed_model,
        base_url=s.base_url,
        api_key=LM_STUDIO_KEY,
        # 켜 두면 tiktoken 토큰 id를 보낸다. LM Studio는 문자열을 받아야 한다.
        check_embedding_ctx_length=False,
        chunk_size=32,
        http_client=http_client,
    )


def live_client(s: Settings, cassette: Path | None = None, chat: BaseChatModel | None = None):
    """실제로 모델을 부르는 클라이언트. LLM_PROVIDER에 따라 LM Studio 또는 Claude API."""
    if s.llm_provider == "anthropic":
        from sfqa.claude import AnthropicClient

        return AnthropicClient(cassette)
    return LiveClient(chat or make_chat(s), cassette)


def make_client(s: Settings, chat: BaseChatModel | None = None):
    if s.cassette_mode == "replay":
        assert s.cassette_path is not None
        return ReplayClient(s.cassette_path)
    cassette = s.cassette_path if s.cassette_mode == "record" else None
    return live_client(s, cassette, chat)


class CachedEmbeddings(Embeddings):
    """질문 임베딩을 JSONL에 기록해 두고 다시 쓴다. replay면 기록에 없을 때 실패한다(네트워크로 가지 않는다).

    키는 (임베딩 모델, 글)이다. 청크 임베딩은 data/index에 따로 저장되므로 여기서는 질문만 다룬다.
    """

    def __init__(self, inner: Embeddings | None, model: str, path: Path, replay: bool = False):
        self.inner, self.model, self.path, self.replay = inner, model, path, replay
        self._cache: dict[str, list[float]] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._cache[row["key"]] = row["vector"]

    def _key(self, text: str) -> str:
        return hashlib.sha256(json.dumps([self.model, text], ensure_ascii=False).encode()).hexdigest()

    def embed_query(self, text: str) -> list[float]:
        key = self._key(text)
        if key in self._cache:
            return self._cache[key]
        if self.replay or self.inner is None:
            raise KeyError(f"임베딩 카세트에 없는 질문이다: {text[:30]}")
        vec = [round(float(x), 7) for x in self.inner.embed_query(text)]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "vector": vec}) + "\n")
        self._cache[key] = vec
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]
