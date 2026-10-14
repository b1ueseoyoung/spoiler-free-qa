"""Claude API 기준선(docs/model-eval-spec.md). 평가에서만 쓴다. 서비스 경로(web, ask)는 계속 로컬 모델이다.

- 공식 anthropic SDK로 부르고, 응답 형식은 output_config.format(json_schema)으로 강제한다.
- thinking을 끈다. Sonnet 5.5는 disabled를 받지 않아 between_tools(effort high 이하)를 쓴다.
- 서버 쪽 폴백은 쓰지 않는다. 안전 거절(stop_reason refusal)은 finish_reason "refusal"로 남겨 보류로 센다.
- seed와 temperature는 보내지 않는다(LLMRequest 카세트 키에는 그대로 들어간다).
- 키는 ANTHROPIC_API_KEY 환경 변수로만 받는다. 없으면 차례로 찾아 환경 변수에 넣는다. 값은 출력하지 않는다.
  - 배포(Lambda): ANTHROPIC_API_KEY_SSM_PARAM에 적힌 SSM Parameter Store 값(SecureString)
  - 이 맥: 프로젝트 .env의 ANTHROPIC_API_KEY 한 줄
- 카세트·응답 형식은 llm.LiveClient와 같다(LLMResponse). 그래서 평가 실행기와 채점기를 그대로 쓴다.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from sfqa.llm import LLMRequest, LLMResponse, _load_cassette
from sfqa.paths import ROOT

KEY_ENV = "ANTHROPIC_API_KEY"
SSM_PARAM_ENV = "ANTHROPIC_API_KEY_SSM_PARAM"
# 모델 → (thinking, effort). LLMRequest.reasoning_effort에는 "between_tools:high"처럼 적어 카세트 키에 남긴다.
THINKING_OFF = {"claude-sonnet-5-5": ({"type": "between_tools"}, "high")}
# $/백만 토큰(입력, 출력). claude-api 스킬 2026-10-06 캐시 기준.
PRICES = {"claude-sonnet-5-5": (2.00, 10.00)}
STOP_TO_FINISH = {"end_turn": "stop", "max_tokens": "length", "refusal": "refusal"}


def effort_label(model: str) -> str:
    thinking, effort = THINKING_OFF[model]
    return f"{thinking['type']}:{effort}"


def load_key(env_file: Path = ROOT / ".env", ssm=None) -> bool:
    """키가 있으면 True. .env에서는 ANTHROPIC_API_KEY 한 줄만 읽는다(다른 값은 건드리지 않는다)."""
    if os.environ.get(KEY_ENV):
        return True
    param = os.environ.get(SSM_PARAM_ENV)
    if param:
        if ssm is None:
            import boto3  # 배포 이미지에만 있다(aws extra)

            ssm = boto3.client("ssm")
        value = ssm.get_parameter(Name=param, WithDecryption=True)["Parameter"]["Value"]
        if value:
            os.environ[KEY_ENV] = value
            return True
        return False
    if not env_file.exists():
        return False
    for line in env_file.read_text(encoding="utf-8").splitlines():
        name, sep, value = line.strip().removeprefix("export ").partition("=")
        if sep and name.strip() == KEY_ENV:
            value = value.strip().strip('"').strip("'")
            if value:
                os.environ[KEY_ENV] = value
                return True
    return False


def cost_usd(resp: LLMResponse) -> float:
    inp, out = PRICES.get(resp.model, (0.0, 0.0))
    return (resp.input_tokens * inp + resp.output_tokens * out) / 1_000_000


class AnthropicClient:
    """llm.LiveClient와 같은 모양(has, complete, calls). 카세트에 있으면 API를 부르지 않는다."""

    def __init__(self, cassette: Path | None = None, sdk=None):
        if sdk is None:
            if not load_key():
                raise RuntimeError(f"{KEY_ENV}가 없다. 프로젝트 .env에 넣거나 환경 변수로 준다")
            import anthropic

            sdk = anthropic.Anthropic(max_retries=3)
        self._sdk = sdk
        self.cassette = cassette
        self._cache = _load_cassette(cassette)
        self.calls = 0

    def has(self, req: LLMRequest) -> bool:
        return req.key() in self._cache

    def complete(self, req: LLMRequest) -> LLMResponse:
        key = req.key()
        if key in self._cache:
            return self._cache[key]
        thinking, effort = THINKING_OFF[req.model]
        if req.reasoning_effort != effort_label(req.model):
            raise ValueError(f"reasoning_effort는 {effort_label(req.model)}여야 한다: {req.reasoning_effort}")
        started = time.monotonic()
        msg = self._sdk.messages.create(
            model=req.model,
            max_tokens=req.max_tokens,
            system=req.system,
            messages=[{"role": "user", "content": req.user}],
            thinking=thinking,
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": req.schema}},
        )
        elapsed = time.monotonic() - started
        category = None
        if msg.stop_reason == "refusal" and getattr(msg, "stop_details", None) is not None:
            category = getattr(msg.stop_details, "category", None)
        text = next((b.text for b in msg.content if b.type == "text"), None)
        resp = LLMResponse(
            # 안전 거절이면 출력이 스키마를 따르지 않을 수 있어 본문을 쓰지 않는다.
            raw_text=None if msg.stop_reason == "refusal" else text,
            finish_reason=STOP_TO_FINISH.get(msg.stop_reason, msg.stop_reason),
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
            model=req.model,
            latency_s=round(elapsed, 3),
            extra={"stop_reason": msg.stop_reason, "refusal_category": category,
                   "request_id": getattr(msg, "_request_id", None)},
        )
        # 해석보다 기록을 먼저 한다. 잘리거나 거절된 응답도 비용이 들었으므로 남긴다.
        self.calls += 1
        if self.cassette:
            self.cassette.parent.mkdir(parents=True, exist_ok=True)
            with self.cassette.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "response": asdict(resp)}, ensure_ascii=False) + "\n")
        self._cache[key] = resp
        return resp
