import json

import openai
import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from openai.types.chat import ChatCompletion

from sfqa.config import load_settings
from sfqa.llm import LiveClient, LLMRequest, ReplayClient, make_client, parse_structured, strip_think

SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}


def make_req(**kw) -> LLMRequest:
    base = dict(model="m", system="s", user="u", schema_name="answer", schema=SCHEMA,
                max_tokens=100, seed=1, reasoning_effort="none")
    return LLMRequest(**{**base, **kw})


def fake_chat(*contents: str) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter([AIMessage(content=c) for c in contents]))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('<think>\n음\n</think>\n{"a": 1}', '{"a": 1}'),
        ('  <think></think>{"a": 1}', '{"a": 1}'),
        ('{"a": "<think>x</think>"}', '{"a": "<think>x</think>"}'),  # 앞에 붙은 것만 지운다
    ],
)
def test_strip_think(raw, expected):
    assert strip_think(raw) == expected


def test_parse_structured():
    assert parse_structured('<think>a</think>{"answer": "x"}') == ({"answer": "x"}, False)
    assert parse_structured("") == (None, False)  # thinking 모드에서 content가 빈 경우
    assert parse_structured("산문입니다") == (None, True)
    assert parse_structured("[1, 2]") == (None, True)


def test_key_changes_with_any_sent_setting():
    base = make_req()
    assert base.key() == make_req().key()
    for change in [dict(user="u2"), dict(model="m2"), dict(schema={**SCHEMA, "x": 1}),
                   dict(reasoning_effort="low"), dict(seed=2), dict(max_tokens=200)]:
        assert make_req(**change).key() != base.key(), change


def test_live_records_before_parse_and_caches(tmp_path):
    tape = tmp_path / "tape.jsonl"
    client = LiveClient(fake_chat("깨진 응답"), cassette=tape)
    first = client.complete(make_req())
    again = client.complete(make_req())
    assert client.calls == 1 and again is first
    assert first.parsed() == (None, True)  # 해석 실패한 응답도 기록은 남는다
    rows = [json.loads(line) for line in tape.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["key"] == make_req().key()


def test_replay_returns_recorded_and_refuses_unknown(tmp_path):
    tape = tmp_path / "tape.jsonl"
    LiveClient(fake_chat('{"answer": "서해"}'), cassette=tape).complete(make_req())
    replay = ReplayClient(tape)
    assert replay.complete(make_req()).parsed() == ({"answer": "서해"}, False)
    with pytest.raises(KeyError):
        replay.complete(make_req(user="다른 질문"))


def test_conflicting_cassette_rows_rejected(tmp_path):
    tape = tmp_path / "tape.jsonl"
    LiveClient(fake_chat('{"answer": "a"}'), cassette=tape).complete(make_req())
    row = json.loads(tape.read_text())
    row["response"]["raw_text"] = '{"answer": "b"}'
    with tape.open("a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with pytest.raises(ValueError):
        ReplayClient(tape)


class TruncatingChat(GenericFakeChatModel):
    """json_schema 경로에서 잘린 응답이 예외로 오는 LM Studio 동작을 흉내 낸다."""

    def _generate(self, *args, **kwargs):
        completion = ChatCompletion.model_validate({
            "id": "x", "object": "chat.completion", "created": 0, "model": "qwen",
            "choices": [{"index": 0, "finish_reason": "length",
                         "message": {"role": "assistant", "content": '{"answer": "군산'}}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 7, "total_tokens": 47},
        })
        raise openai.LengthFinishReasonError(completion=completion)


def test_truncated_response_is_recorded_not_raised(tmp_path):
    tape = tmp_path / "tape.jsonl"
    client = LiveClient(TruncatingChat(messages=iter([])), cassette=tape)
    resp = client.complete(make_req())
    assert resp.truncated and resp.output_tokens == 7 and client.calls == 1
    assert resp.parsed() == (None, True)
    assert ReplayClient(tape).complete(make_req()).truncated


def test_make_client_replay_never_builds_live_model(tmp_path):
    tape = tmp_path / "tape.jsonl"
    tape.write_text("")
    s = load_settings({"CASSETTE_MODE": "replay", "CASSETTE_PATH": str(tape)})
    assert isinstance(make_client(s), ReplayClient)
