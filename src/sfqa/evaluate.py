"""평가 실행기: 정답지 문항을 조건별로 돌려 runs/<시각>-<split>/records.jsonl에 남긴다.

- 모델 응답과 질문 임베딩은 카세트(eval/cassettes/)에 기록한다. 같은 요청은 다시 부르지 않으므로 중간에 멈춰도 이어서 돈다.
- --max-calls: 실제 모델 호출 수의 상한. 넘기 직전에 멈추고 종료 코드 3으로 끝낸다(그때까지의 기록은 남는다).
- C4는 C3와 같은 요청이라 카세트에서 응답을 다시 쓴다(모델 호출 0회).
- run-info.json에 git 커밋, 프롬프트·스키마·정답지 해시, 모델 설정을 남긴다(side-project/scripts/eval/run.mts 구조).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from sfqa.answer import CONDITIONS, Env, build_chain
from sfqa.config import Settings
from sfqa.gold import ENTITIES, QUESTIONS, Question, entity_index, load_entities, load_questions, load_texts
from sfqa.index import load_index
from sfqa.llm import CachedEmbeddings, LiveClient, ReplayClient, make_chat, make_embeddings
from sfqa.paths import ROOT
from sfqa.prompts import prompt_hash
from sfqa.retrieve import ChapterIndex
from sfqa.schema import ANSWER_SCHEMA
from sfqa.verify import Verifier

CASSETTES = ROOT / "eval" / "cassettes"
RUNS = ROOT / "runs"


class BudgetExceeded(RuntimeError):
    pass


class BudgetClient:
    """새로 부를 요청이 상한을 넘으면 부르기 전에 멈춘다."""

    def __init__(self, inner: LiveClient | ReplayClient, max_calls: int):
        self.inner, self.max_calls = inner, max_calls

    @property
    def calls(self) -> int:
        return self.inner.calls

    def complete(self, req):
        # 재생 모드는 모델을 부르지 않는다. 카세트에 없으면 ReplayClient가 KeyError로 바로 알린다.
        # 실제로 부르는 클라이언트(LM Studio의 LiveClient, Claude의 AnthropicClient)에만 상한을 건다.
        if not isinstance(self.inner, ReplayClient) and not self.inner.has(req) and self.inner.calls >= self.max_calls:
            raise BudgetExceeded(f"모델 호출 상한 {self.max_calls}회에 닿았다")
        return self.inner.complete(req)


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else ""


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        except OSError:
            return ""
    return {"head": run("rev-parse", "--short", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def select(questions: list[Question], split: str, limit: int | None) -> list[Question]:
    qs = [q for q in questions if split == "all" or q.split == split]
    return qs[:limit] if limit else qs


def make_eval_env(s: Settings, cassette: Path, emb_cassette: Path, replay: bool, k: int) -> Env:
    texts = load_texts()
    chunks, vectors, _ = load_index(s.embed_model, texts)
    if replay:
        client = ReplayClient(cassette)
        emb = CachedEmbeddings(None, s.embed_model, emb_cassette, replay=True)
    else:
        client = LiveClient(make_chat(s), cassette)
        emb = CachedEmbeddings(make_embeddings(s), s.embed_model, emb_cassette)
    verifier = Verifier(texts, entity_index(load_entities(), texts))
    return Env(s, ChapterIndex(chunks, vectors), emb, client, verifier, k)


def run_eval(s: Settings, split: str, conditions: list[str], max_calls: int, k: int = 18,
             limit: int | None = None, replay: bool = False, cassette_name: str | None = None,
             out_root: Path = RUNS, env: Env | None = None) -> tuple[Path, dict]:
    bad = [c for c in conditions if c not in CONDITIONS]
    if bad:
        raise ValueError(f"없는 조건이다: {bad}")
    name = cassette_name or split
    cassette = CASSETTES / f"{name}.jsonl"
    emb_cassette = CASSETTES / f"{name}-query-embeddings.jsonl"
    env = env or make_eval_env(s, cassette, emb_cassette, replay, k)
    budget = BudgetClient(env.client, max_calls)
    env = replace(env, client=budget)
    questions = select(load_questions(), split, limit)

    out = out_root / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{split}"
    out.mkdir(parents=True, exist_ok=True)
    info = {
        "split": split, "conditions": conditions, "limit": limit, "k": k, "replay": replay,
        "questions": len(questions), "max_calls": max_calls,
        "git": git_state(), "prompt_hash": prompt_hash(),
        "schema_sha": hashlib.sha256(json.dumps(ANSWER_SCHEMA, sort_keys=True).encode()).hexdigest()[:16],
        "gold_sha": sha_file(QUESTIONS), "entities_sha": sha_file(ENTITIES),
        "model": {"llm": s.llm_model, "embed": s.embed_model, "seed": s.seed, "temperature": 0,
                  "max_tokens": s.max_tokens, "reasoning_effort": s.reasoning_effort, "base_url": s.base_url},
        "cassette": str(cassette.relative_to(ROOT)), "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    done, complete = 0, True
    with (out / "records.jsonl").open("w", encoding="utf-8") as fh:
        try:
            for condition in conditions:
                chain = build_chain(env, condition)
                for q in questions:
                    result = chain.invoke({"question": q.question, "n": q.n})
                    fh.write(json.dumps({"qid": q.id, "type": q.type, "split": q.split, "n": q.n,
                                         "condition": condition, "result": result.to_dict()},
                                        ensure_ascii=False) + "\n")
                    fh.flush()
                    done += 1
        except BudgetExceeded as e:
            complete = False
            info["stopped"] = str(e)
    info |= {"records": done, "complete": complete, "model_calls": budget.calls,
             "finished_at": datetime.now().isoformat(timespec="seconds")}
    (out / "run-info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out, info

