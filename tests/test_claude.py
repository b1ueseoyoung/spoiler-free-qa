import json
from types import SimpleNamespace

import pytest

from sfqa.claude import AnthropicClient, effort_label, load_key
from sfqa.llm import LLMRequest
from sfqa.klue import KLUE_SCHEMA


class FakeSDK:
    """anthropic.Anthropic 대역. messages.create 인자를 기록한다."""

    def __init__(self, stop="end_turn", text='{"status": "answered", "answer": "서해"}'):
        self.calls, self.stop, self.text = [], stop, text
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        content = [SimpleNamespace(type="text", text=self.text)]
        return SimpleNamespace(stop_reason=self.stop, stop_details=SimpleNamespace(category="cyber") if self.stop == "refusal" else None,
                               content=content, usage=SimpleNamespace(input_tokens=100, output_tokens=10),
                               _request_id="req_1")


def req(effort=None):
    return LLMRequest(model="claude-sonnet-5-5", system="s", user="u", schema_name="klue_answer", schema=KLUE_SCHEMA,
                      max_tokens=800, seed=1, reasoning_effort=effort or effort_label("claude-sonnet-5-5"))


def test_sends_thinking_off_and_structured_output_without_sampling_params(tmp_path):
    sdk = FakeSDK()
    r = AnthropicClient(tmp_path / "c.jsonl", sdk=sdk).complete(req())
    kw = sdk.calls[0]
    assert kw["thinking"] == {"type": "between_tools"}
    assert kw["output_config"] == {"effort": "high", "format": {"type": "json_schema", "schema": KLUE_SCHEMA}}
    assert "temperature" not in kw and "seed" not in kw and "fallbacks" not in kw
    assert r.parsed()[0]["answer"] == "서해" and r.finish_reason == "stop" and r.input_tokens == 100


def test_cassette_records_and_skips_repeat_calls(tmp_path):
    sdk, path = FakeSDK(), tmp_path / "c.jsonl"
    c = AnthropicClient(path, sdk=sdk)
    c.complete(req()), c.complete(req())
    assert len(sdk.calls) == 1 and c.calls == 1
    again = AnthropicClient(path, sdk=FakeSDK())
    assert again.has(req()) and again.complete(req()).raw_text and again.calls == 0
    assert json.loads(path.read_text().splitlines()[0])["response"]["extra"]["request_id"] == "req_1"


def test_refusal_and_truncation_are_recorded(tmp_path):
    r = AnthropicClient(tmp_path / "a.jsonl", sdk=FakeSDK(stop="refusal")).complete(req())
    assert r.finish_reason == "refusal" and r.raw_text is None and r.extra["refusal_category"] == "cyber"
    t = AnthropicClient(tmp_path / "b.jsonl", sdk=FakeSDK(stop="max_tokens", text='{"sta')).complete(req())
    assert t.truncated


def test_rejects_mismatched_effort_label(tmp_path):
    with pytest.raises(ValueError):
        AnthropicClient(None, sdk=FakeSDK()).complete(req(effort="none"))


def test_load_key_reads_only_that_line(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OTHER", raising=False)
    env = tmp_path / ".env"
    assert not load_key(env)
    env.write_text('OTHER=1\nexport ANTHROPIC_API_KEY="test-value"\n')
    assert load_key(env)
    import os
    assert os.environ["ANTHROPIC_API_KEY"] == "test-value" and "OTHER" not in os.environ


def test_load_key_from_ssm_parameter(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY_SSM_PARAM", "/sfqa/anthropic-api-key")
    calls = []

    class FakeSSM:
        def get_parameter(self, Name, WithDecryption):
            calls.append((Name, WithDecryption))
            return {"Parameter": {"Value": "ssm-test-value"}}

    assert load_key(tmp_path / "없는.env", ssm=FakeSSM())
    import os
    assert os.environ["ANTHROPIC_API_KEY"] == "ssm-test-value" and calls == [("/sfqa/anthropic-api-key", True)]
