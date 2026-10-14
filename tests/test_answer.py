import json

import numpy as np
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from sfqa.answer import Env, build_chain
from sfqa.chunk import chunk_corpus
from sfqa.config import load_settings
from sfqa.index import normalize_rows
from sfqa.llm import LLMRequest, LLMResponse
from sfqa.retrieve import ChapterIndex
from sfqa.verify import Verifier

TEXTS = {ch: f"제{ch}장 이야기. " + " ".join(f"{ch}장의 {i}번째 문장이 여기 있다." for i in range(60)) for ch in range(1, 20)}
TEXTS[2] = "초봉이는 제중당 약국에 다닌다. " + TEXTS[2]
EMB = DeterministicFakeEmbedding(size=16)


class FakeClient:
    """요청을 기록하고 정해 둔 응답을 돌려준다(모델을 부르지 않는다)."""

    def __init__(self, raw: str, finish: str = "stop"):
        self.raw, self.finish, self.requests = raw, finish, []

    def complete(self, req: LLMRequest) -> LLMResponse:
        self.requests.append(req)
        return LLMResponse(raw_text=self.raw, finish_reason=self.finish, input_tokens=10, output_tokens=5)


def answer_json(status="answered", answer="제중당 약국", quotes=((2, "초봉이는 제중당 약국에 다닌다."),)) -> str:
    return json.dumps({"status": status, "answer": answer,
                       "quotes": [{"chapter": c, "text": t} for c, t in quotes]}, ensure_ascii=False)


def env(client, k=6) -> Env:
    chunks = chunk_corpus(TEXTS)
    idx = ChapterIndex(chunks, normalize_rows(np.asarray(EMB.embed_documents([c.text for c in chunks]))))
    return Env(load_settings({}), idx, EMB, client, Verifier(TEXTS, {"형보": 5}), k)


def run(condition, raw=None, n=3, question="초봉이는 어디에 다니나?", finish="stop"):
    client = FakeClient(raw if raw is not None else answer_json(), finish)
    result = build_chain(env(client), condition).invoke({"question": question, "n": n})
    return result, client.requests


@pytest.mark.parametrize("condition", ["C2", "C3", "C4"])
def test_filtered_conditions_show_only_chapters_up_to_n(condition):
    result, reqs = run(condition, n=3)
    assert result.retrieved and max(d["chapter"] for d in result.retrieved) <= 3
    labels = [int(x.split("장")[0]) for x in reqs[0].user.split("[제")[1:]]
    assert labels and max(labels) <= 3


def test_c1_searches_all_chapters_and_uses_base_prompt():
    result, reqs = run("C1", n=1, question="19장의 30번째 문장")
    assert max(d["chapter"] for d in result.retrieved) > 1
    assert "독자는 제" not in reqs[0].system


def test_c0_has_no_context_and_guard_prompt():
    result, reqs = run("C0", n=4)
    assert result.retrieved == [] and reqs[0].user == "질문: 초봉이는 어디에 다니나?"
    assert "제4장까지 읽었다" in reqs[0].system


def test_c3_and_c4_send_identical_requests():
    _, r3 = run("C3", n=3)
    _, r4 = run("C4", n=3)
    assert r3[0].key() == r4[0].key()
    assert "제3장까지 읽었다" in r3[0].system


def test_c4_answers_with_verified_quote_and_c3_keeps_raw_answer():
    r4, _ = run("C4", n=3)
    assert r4.status == "answered" and r4.answer == "제중당 약국"
    unsupported = answer_json(quotes=())
    r4, _ = run("C4", raw=unsupported, n=3)
    assert r4.status == "not_found" and r4.answer == "3장까지는 나오지 않아요."
    r3, _ = run("C3", raw=unsupported, n=3)
    assert r3.status == "answered" and "검증된 인용 없음" in r3.reasons


def test_c4_blocks_future_entity():
    r, _ = run("C4", raw=answer_json(answer="형보와 함께 산다"), n=3)
    assert r.status == "not_found" and r.future_entities == ["형보"]


def test_invalid_and_truncated_are_recorded_not_raised():
    r, _ = run("C4", raw="산문으로 답합니다", n=3)
    assert r.invalid and r.status == "not_found" and r.model_status is None
    r, _ = run("C4", raw='{"status": "answered", "answ', n=3, finish="length")
    assert r.truncated and not r.invalid and r.status == "not_found"
    r, _ = run("C4", raw="<think>음</think>" + answer_json(), n=3)
    assert r.status == "answered"
