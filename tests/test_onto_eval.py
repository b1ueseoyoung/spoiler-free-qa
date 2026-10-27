import csv
import json
from types import SimpleNamespace

import pytest

from sfqa.onto_eval import (
    Budget,
    candidates,
    id_map,
    judge,
    match_request,
    review_sheet,
    run_batch,
    score,
)
from sfqa.ontology import Alias, Character, Fact, Knowing, Ontology, synthetic_ids


def k(character: str, from_ch: int) -> Knowing:
    return Knowing(character=character, from_ch=from_ch)


GOLD = Ontology(
    work="g", title="정답", chapters=5, source="test",
    characters=[Character(id="cecilia", name="세실리아 온드레", first_ch=1),
                Character(id="cassiel", name="카시엘 로스벤", first_ch=2, aliases=[Alias(form="겨울 공작", from_ch=2)])],
    facts=[Fact(id="f1", statement="세실리아는 서약청의 필경사다.", reveal_ch=1, known_by=[k("cecilia", 0), k("cassiel", 2)]),
           Fact(id="f2", statement="카시엘은 겨울 공작이라 불린다.", reveal_ch=2, known_by=[k("cecilia", 0), k("cassiel", 0)]),
           Fact(id="f3", statement="세실리아에게는 서약을 푸는 해약의 힘이 있다.", reveal_ch=4, known_by=[k("cecilia", 4)],
                is_twist=True)],
)
EXTRACTED = Ontology(
    work="g", title="정답", chapters=5, source="extract:test",
    characters=[Character(id="cec", name="세실리아 온드레", first_ch=1), Character(id="duke", name="카시엘 로스벤", first_ch=2)],
    facts=[Fact(id="c1-1", statement="세실리아 온드레는 서약청의 하급 필경사다.", reveal_ch=1, known_by=[k("cec", 0)]),
           Fact(id="c2-1", statement="카시엘은 '겨울 공작'이라는 별명으로 불린다.", reveal_ch=2, common=True),
           Fact(id="c3-1", statement="세실리아는 서약을 푸는 해약의 힘을 지녔다.", reveal_ch=3, known_by=[k("cec", 3)])],
)
TEXTS = {1: "세실리아 온드레는 필경사다.", 2: "카시엘 로스벤이 왔다.", 3: "은실이 풀렸다.", 4: "해약.", 5: "끝."}
JUDGED = {"match": {"f1": {"ids": ["c1-1"], "partial": False, "reason": "", "candidates": ["c1-1"]},
                    "f2": {"ids": ["c2-1"], "partial": False, "reason": "", "candidates": ["c2-1"]},
                    "f3": {"ids": ["c3-1"], "partial": False, "reason": "", "candidates": ["c3-1"]}},
          "support": {"c1-1": {"supported": "yes", "reason": ""}}}


class FakeBatches:
    """만든 요청을 기억하고, reply(custom_id)가 돌려준 값으로 결과를 만든다(None이면 실패)."""

    def __init__(self, reply):
        self.reply = reply
        self.created = []

    def create(self, requests):
        self.created.append(requests)
        return SimpleNamespace(id=f"batch_{len(self.created)}")

    def retrieve(self, batch_id):
        return SimpleNamespace(processing_status="ended", request_counts={})

    def results(self, batch_id):
        for r in self.created[-1]:
            payload = self.reply(r["custom_id"])
            if payload is None:
                yield SimpleNamespace(custom_id=r["custom_id"], result=SimpleNamespace(type="errored"))
                continue
            usage = SimpleNamespace(input_tokens=900, output_tokens=60, cache_read_input_tokens=0, cache_creation_input_tokens=0)
            msg = SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(payload, ensure_ascii=False))],
                                  stop_reason="end_turn", usage=usage)
            yield SimpleNamespace(custom_id=r["custom_id"], result=SimpleNamespace(type="succeeded", message=msg))


def fake_sdk(reply) -> SimpleNamespace:
    return SimpleNamespace(messages=SimpleNamespace(batches=FakeBatches(reply)))


def test_names_link_extracted_characters_to_gold():
    assert id_map(EXTRACTED, GOLD) == {"cec": "cecilia", "duke": "cassiel"}


def test_bm25_candidates_find_the_same_fact_across_chapters():
    cands = candidates(GOLD.facts, EXTRACTED.facts, k=2)
    assert cands["f1"][0] == "c1-1" and cands["f3"][0] == "c3-1"


def test_score_counts_facts_reveal_and_knowers():
    s = score(GOLD, EXTRACTED, 5, JUDGED, TEXTS)
    assert (s["full"], s["partial"], s["missed"]) == (3, 0, 0) and s["twists"] == {"full": 1}
    assert s["reveal"] == {"same": 2, "early": 1}  # 해약의 힘을 정답보다 한 장 일찍 뽑았다
    assert s["reveal_early"][0]["gold"] == "f3" and s["reveal_early"][0]["extracted_ch"] == 3
    # f1: 카시엘이 2장에 알게 된 것을 놓침, f2: 상식이라 둘 다 0장부터, f3: 3장으로 한 장 앞당김
    assert (s["tp"], s["fp"], s["fn"]) == (4, 0, 1) and s["precision"] == 1.0 and s["recall"] == 0.8
    assert [(e["gold"], e["extracted"], e["gold_since"]) for e in s["since_early"]] == [("f3", 3, 4)]
    assert (s["after_reveal"], s["after_reveal_hit"]) == (1, 0)
    assert s["support"] == {"yes": 1}


def test_since_uses_the_latest_part_and_sets_partial_matches_aside():
    ext = EXTRACTED.model_copy(update={"facts": [
        *EXTRACTED.facts,
        Fact(id="c1-2", statement="카시엘은 왼손에만 장갑을 낀다.", reveal_ch=1, known_by=[k("cec", 1)]),
        Fact(id="c3-2", statement="세실리아는 해약의 대가를 안다.", reveal_ch=3, known_by=[k("cec", 3)]),
    ]})
    judged = {"match": {"f3": {"ids": ["c1-2", "c3-2"], "partial": False}}, "support": {}}  # 두 부분을 모두 알아야 정답 전체를 안다
    s = score(GOLD, ext, 5, judged, TEXTS)
    assert [(e["gold"], e["extracted"]) for e in s["since_early"]] == [("f3", 3)]  # 1장이 아니라 3장
    judged["match"]["f3"]["partial"] = True  # 정답 일부만 담은 짝은 따로 센다
    s = score(GOLD, ext, 5, judged, TEXTS)
    assert s["since_early"] == [] and [e["gold"] for e in s["since_early_partial"]] == ["f3"]


def test_run_batch_reuses_the_cassette_and_resends_failures(tmp_path):
    items = {"m-f1": match_request(GOLD.facts[0], EXTRACTED.facts[:1]), "m-f2": match_request(GOLD.facts[1], EXTRACTED.facts[1:2])}
    sdk = fake_sdk(lambda cid: None if cid == "m-f2" else {"matches": [1], "partial": False, "reason": "같다"})
    out, spent = run_batch(items, tmp_path / "c.jsonl", sdk=sdk, max_usd=1.0, poll_s=0, log=lambda _: None)
    assert out["m-f1"].raw_text and out["m-f2"].finish_reason == "errored" and spent > 0
    again = fake_sdk(lambda cid: {"matches": [], "partial": False, "reason": "없다"})
    out, _ = run_batch(items, tmp_path / "c.jsonl", sdk=again, max_usd=1.0, poll_s=0, log=lambda _: None)
    assert [r["custom_id"] for r in again.messages.batches.created[0]] == ["m-f2"]  # 성공한 것은 다시 보내지 않는다
    assert json.loads(out["m-f1"].raw_text)["matches"] == [1]
    with pytest.raises(Budget):
        run_batch({"m-x": items["m-f1"]}, None, sdk=again, max_usd=0.00001, poll_s=0, log=lambda _: None)


@pytest.mark.skipif("wn01-rofan" not in synthetic_ids(), reason="파일럿(wn01-rofan)이 없다")
def test_judge_asks_again_with_nearby_chapters_when_bm25_misses(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "report.json").write_text(json.dumps({"done": 1}), encoding="utf-8")
    ext = Ontology(work="wn01-rofan", title="t", chapters=50, source="extract:test", characters=[],
                   facts=[Fact(id="c1-1", statement="필사실 창밖으로 비가 내렸다.", reveal_ch=1)])  # 글자가 겹치지 않아 BM25 후보가 없다
    (run / "ontology.json").write_text(ext.model_dump_json(), encoding="utf-8")
    sdk = fake_sdk(lambda cid: {"matches": [1], "partial": True, "reason": "둘째"} if cid.startswith("m2-")
                   else {"matches": [], "partial": False, "reason": "없다"})
    out = judge(run, "wn01-rofan", cassette=None, support_sample=0, max_usd=1.0, sdk=sdk, poll_s=0, log=lambda _: None)
    assert out["match"]["f001"] == {"ids": ["c1-1"], "partial": True, "reason": "둘째", "candidates": ["c1-1"], "pass": 2}
    assert len(sdk.messages.batches.created) == 2  # 첫 판정 묶음과 둘째 판정 묶음


@pytest.mark.skipif("wn01-rofan" not in synthetic_ids(), reason="파일럿(wn01-rofan)이 없다")
def test_judge_reuses_found_matches_and_support_when_more_chapters_arrive(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "report.json").write_text(json.dumps({"done": 2}), encoding="utf-8")
    ext = Ontology(work="wn01-rofan", title="t", chapters=50, source="extract:test", characters=[],
                   facts=[Fact(id="c1-1", statement="세실리아 온드레는 왕립 서약청의 하급 필경사로 3년째 일한다.", reveal_ch=1)])
    (run / "ontology.json").write_text(ext.model_dump_json(), encoding="utf-8")
    found = {"ids": ["c1-1"], "partial": False, "reason": "이전", "candidates": ["c1-1"], "pass": 1}
    missed = {"ids": [], "partial": False, "reason": "없음", "candidates": [], "pass": 1}
    (run / "judgments.json").write_text(json.dumps({"gold": "wn01-rofan", "match": {"f001": found, "f002": missed},
                                                    "support": {"c1-1": {"supported": "yes", "reason": "이전"}}}),
                                        encoding="utf-8")
    sdk = fake_sdk(lambda cid: {"matches": [], "partial": False, "reason": "새로"})
    out = judge(run, "wn01-rofan", cassette=None, max_usd=1.0, sdk=sdk, poll_s=0, log=lambda _: None)
    sent = {r["custom_id"] for batch in sdk.messages.batches.created for r in batch}
    assert "m-f001" not in sent and "m-f002" in sent  # 찾은 짝은 그대로, 놓친 것은 다시 묻는다
    assert not any(cid.startswith("s-") for cid in sent)  # 근거 표본도 그대로
    assert out["match"]["f001"] == found and out["reused"] == 1 and out["support"]["c1-1"]["reason"] == "이전"


def test_support_request_sends_whole_paragraphs_around_the_quote():
    from sfqa.onto_eval import SUPPORT_PARAGRAPHS, support_request

    paras = [f"문단{i}" for i in range(1, 10)]
    f = Fact(id="c1-1", statement="문단5의 사실.", reveal_ch=1, evidence=[{"chapter": 1, "lines": [9]}])  # 9줄 = 문단5
    user = support_request(f, {1: "\n\n".join(paras)})["messages"][0]["content"]  # 본문처럼 문단 사이에 빈 줄
    assert "인용: 문단5" in user
    assert SUPPORT_PARAGRAPHS == 3 and user.split("앞뒤 본문:\n")[1].split("\n\n") == paras[1:8]  # 빈 줄은 세지 않는다


@pytest.mark.skipif("wn01-rofan" not in synthetic_ids(), reason="파일럿(wn01-rofan)이 없다")
def test_judge_can_redo_the_same_support_sample(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "report.json").write_text(json.dumps({"done": 2}), encoding="utf-8")
    ev = [{"chapter": 1, "lines": [1]}]
    ext = Ontology(work="wn01-rofan", title="t", chapters=50, source="extract:test", characters=[],
                   facts=[Fact(id="c1-1", statement="세실리아 온드레는 왕립 서약청의 하급 필경사로 3년째 일한다.", reveal_ch=1,
                               evidence=ev),
                          Fact(id="c1-2", statement="필사실 창밖으로 비가 내렸다.", reveal_ch=1, evidence=ev)])
    (run / "ontology.json").write_text(ext.model_dump_json(), encoding="utf-8")
    found = {"ids": ["c1-1"], "partial": False, "reason": "이전", "candidates": ["c1-1"], "pass": 1}
    (run / "judgments.json").write_text(json.dumps({"gold": "wn01-rofan", "match": {"f001": found},
                                                    "support": {"c1-1": {"supported": "no", "reason": "한 줄뿐"}}}),
                                        encoding="utf-8")
    sdk = fake_sdk(lambda cid: {"supported": "yes", "reason": "앞뒤 문단에 보인다"} if cid.startswith("s-")
                   else {"matches": [], "partial": False, "reason": "없음"})
    out = judge(run, "wn01-rofan", cassette=None, max_usd=1.0, sdk=sdk, poll_s=0, log=lambda _: None, redo_support=True)
    sent = {r["custom_id"] for batch in sdk.messages.batches.created for r in batch}
    assert {cid for cid in sent if cid.startswith("s-")} == {"s-c1-1"}  # 이전 표본만 다시 묻고 새로 뽑지 않는다
    assert "m-f001" not in sent and out["match"]["f001"] == found
    assert out["support"] == {"c1-1": {"supported": "yes", "reason": "앞뒤 문단에 보인다"}}


@pytest.mark.skipif("wn01-rofan" not in synthetic_ids(), reason="파일럿(wn01-rofan)이 없다")
def test_judge_writes_judgments_and_a_review_sheet(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "report.json").write_text(json.dumps({"done": 2}), encoding="utf-8")
    ext = Ontology(work="wn01-rofan", title="t", chapters=50, source="extract:test",
                   characters=[Character(id="cec", name="세실리아 온드레", first_ch=1)],
                   facts=[Fact(id="c1-1", statement="세실리아 온드레는 왕립 서약청의 하급 필경사로 3년째 일한다.", reveal_ch=1,
                               known_by=[k("cec", 0)], evidence=[{"chapter": 1, "lines": [1]}])])
    (run / "ontology.json").write_text(ext.model_dump_json(), encoding="utf-8")
    sdk = fake_sdk(lambda cid: {"supported": "yes", "reason": "보인다"} if cid.startswith("s-")
                   else {"matches": [1], "partial": False, "reason": "같다"})
    out = judge(run, "wn01-rofan", cassette=None, max_usd=1.0, sdk=sdk, poll_s=0, log=lambda _: None)
    assert out["failed"] == [] and out["support"] == {"c1-1": {"supported": "yes", "reason": "보인다"}}
    assert out["match"]["f001"]["ids"] == ["c1-1"]  # 후보가 하나뿐이라 1번이 c1-1이다
    saved = json.loads((run / "judgments.json").read_text(encoding="utf-8"))
    assert saved["upto"] == 2 and set(saved["match"]) == {g for g in out["match"]}
    from sfqa.ontology import from_synthetic

    review_sheet(run / "review.csv", from_synthetic("wn01-rofan"), ext, saved)
    rows = list(csv.reader((run / "review.csv").open(encoding="utf-8-sig")))
    assert rows[0][-2:] == ["human_ok", "human_note"] and len(rows) > 1
