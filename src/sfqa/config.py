"""환경 변수 설정. 모델 서버는 이 맥 안(loopback)만 허용한다."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
CASSETTE_MODES = {"off", "record", "replay"}


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
    cassette_mode: str = "off"
    cassette_path: Path | None = None
    wiki_user_agent: str = "sfqa/0.1 (personal research project)"


def load_settings(env: Mapping[str, str] = os.environ) -> Settings:
    d = Settings()
    s = Settings(
        base_url=env.get("LLM_BASE_URL", d.base_url),
        llm_model=env.get("LLM_MODEL", d.llm_model),
        embed_model=env.get("EMBED_MODEL", d.embed_model),
        timeout_s=float(env.get("LLM_TIMEOUT_S", d.timeout_s)),
        max_tokens=int(env.get("LLM_MAX_TOKENS", d.max_tokens)),
        seed=int(env.get("LLM_SEED", d.seed)),
        reasoning_effort=env.get("LLM_REASONING_EFFORT", d.reasoning_effort),
        cassette_mode=env.get("CASSETTE_MODE", d.cassette_mode),
        cassette_path=Path(env["CASSETTE_PATH"]) if env.get("CASSETTE_PATH") else None,
        wiki_user_agent=env.get("WIKI_USER_AGENT", d.wiki_user_agent),
    )
    check_loopback(s.base_url)
    if s.cassette_mode not in CASSETTE_MODES:
        raise ValueError(f"CASSETTE_MODE는 {sorted(CASSETTE_MODES)} 중 하나다: {s.cassette_mode}")
    if s.cassette_mode != "off" and s.cassette_path is None:
        raise ValueError("CASSETTE_MODE가 record나 replay면 CASSETTE_PATH가 필요하다")
    return s
