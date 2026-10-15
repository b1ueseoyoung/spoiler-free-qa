"""환경 변수 설정.

- 기본은 로컬이다: 생성·임베딩 모두 LM Studio(loopback만 허용).
- 공개 데모용 선택지(docs/deploy-plan.md):
  - LLM_PROVIDER=anthropic: 생성은 Claude API(sfqa.claude). 기본 모델 claude-sonnet-5-5, thinking 끔
  - EMBED_PROVIDER=local: 질문 임베딩을 이 프로세스 안에서 llama.cpp로 만든다(LM Studio와 같은 KURE-v1 GGUF)
- RETRIEVE_K: 질문마다 가져올 청크 수(기본 18). 공개 데모는 20(docs/deploy-plan.md 2단계).
- LM Studio를 하나라도 쓰면 LLM_BASE_URL은 loopback이어야 한다.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
CASSETTE_MODES = {"off", "record", "replay"}
LLM_PROVIDERS = {"lmstudio", "anthropic"}
EMBED_PROVIDERS = {"lmstudio", "local"}
# sfqa.claude.THINKING_OFF와 같아야 한다(테스트로 확인). claude를 여기서 import하면 순환 import가 된다.
ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-5-5"
ANTHROPIC_DEFAULT_EFFORT = "between_tools:high"
LOCAL_EMBED_DEFAULT_MODEL = "mykor/KURE-v1-gguf/KURE-v1-Q8_0.gguf"  # LM Studio와 같은 파일


def check_loopback(url: str) -> None:
    """원문과 질문이 맥 밖으로 나가지 않게, loopback이 아닌 주소는 거부한다."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in LOOPBACK_HOSTS:
        raise ValueError(f"LLM_BASE_URL은 이 맥의 주소(127.0.0.1, localhost, ::1)만 쓸 수 있다: {url}")


@dataclass(frozen=True)
class Settings:
    base_url: str = "http://127.0.0.1:1234/v1"
    llm_model: str = "qwen/qwen3.6-35b-a3b"
    embed_model: str = "text-embedding-kure-v1"
    timeout_s: float = 180.0
    max_tokens: int = 800
    seed: int = 20261013
    # Qwen3.6은 기본이 thinking 모드라 json_schema 응답이 reasoning_content로 간다. none이면 content로 온다.
    reasoning_effort: str = "none"
    llm_provider: str = "lmstudio"
    embed_provider: str = "lmstudio"
    retrieve_k: int = 18
    cassette_mode: str = "off"
    cassette_path: Path | None = None
    wiki_user_agent: str = "sfqa/0.1 (personal research project)"


def load_settings(env: Mapping[str, str] = os.environ) -> Settings:
    d = Settings()
    llm_provider = env.get("LLM_PROVIDER", d.llm_provider)
    embed_provider = env.get("EMBED_PROVIDER", d.embed_provider)
    if llm_provider not in LLM_PROVIDERS:
        raise ValueError(f"LLM_PROVIDER는 {sorted(LLM_PROVIDERS)} 중 하나다: {llm_provider}")
    if embed_provider not in EMBED_PROVIDERS:
        raise ValueError(f"EMBED_PROVIDER는 {sorted(EMBED_PROVIDERS)} 중 하나다: {embed_provider}")
    anthropic = llm_provider == "anthropic"
    s = Settings(
        base_url=env.get("LLM_BASE_URL", d.base_url),
        llm_model=env.get("LLM_MODEL", ANTHROPIC_DEFAULT_MODEL if anthropic else d.llm_model),
        embed_model=env.get("EMBED_MODEL", LOCAL_EMBED_DEFAULT_MODEL if embed_provider == "local" else d.embed_model),
        timeout_s=float(env.get("LLM_TIMEOUT_S", d.timeout_s)),
        max_tokens=int(env.get("LLM_MAX_TOKENS", d.max_tokens)),
        seed=int(env.get("LLM_SEED", d.seed)),
        reasoning_effort=env.get("LLM_REASONING_EFFORT", ANTHROPIC_DEFAULT_EFFORT if anthropic else d.reasoning_effort),
        retrieve_k=int(env.get("RETRIEVE_K", d.retrieve_k)),
        llm_provider=llm_provider,
        embed_provider=embed_provider,
        cassette_mode=env.get("CASSETTE_MODE", d.cassette_mode),
        cassette_path=Path(env["CASSETTE_PATH"]) if env.get("CASSETTE_PATH") else None,
        wiki_user_agent=env.get("WIKI_USER_AGENT", d.wiki_user_agent),
    )
    if "lmstudio" in (s.llm_provider, s.embed_provider):
        check_loopback(s.base_url)
    if s.cassette_mode not in CASSETTE_MODES:
        raise ValueError(f"CASSETTE_MODE는 {sorted(CASSETTE_MODES)} 중 하나다: {s.cassette_mode}")
    if s.cassette_mode != "off" and s.cassette_path is None:
        raise ValueError("CASSETTE_MODE가 record나 replay면 CASSETTE_PATH가 필요하다")
    return s
