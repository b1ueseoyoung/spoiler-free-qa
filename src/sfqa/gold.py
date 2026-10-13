"""정답지(eval/questions.jsonl)와 인물표(data/entities.json)를 읽고, 모델 없이 검사한다.

기준은 docs/eval-spec.md의 "정답지 검사"다. 오류가 하나라도 있으면 gold-check는 종료 코드 1이다.
인물·지명의 첫 등장 장은 저장하지 않고 본문에서 매번 계산한다(별칭만 사람이 관리한다).
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from sfqa.paths import CHAPTERS, DATA, ROOT, clean_path

QUESTIONS = ROOT / "eval" / "questions.jsonl"
ENTITIES = DATA / "entities.json"
QUOTE_MAX = 150
LAST = max(CHAPTERS)
# eval-spec.md의 문항 구성: (split, 묶음) → 개수
PLAN = {("dev", "pair"): 8, ("dev", "A"): 2, ("dev", "C"): 2, ("test", "pair"): 32, ("test", "A"): 6, ("test", "C"): 6}
BOUNDARY_EACH = 5
_SENTENCE = re.compile(r"(?<=[.?!])\s+|\n+")

Chapter = Field(ge=1, le=LAST)


class Evidence(BaseModel):
    chapter: int = Chapter
    quote: str


class FutureFact(BaseModel):
    chapter: int = Chapter
    keys: list[str] = Field(min_length=1)


class Question(BaseModel):
    id: str
    pair_id: str | None = None
    split: Literal["dev", "test"]
    type: Literal["A", "B", "C"]
    n: int = Chapter
    question: str
    gold_answer: str | None = None
    answer_keys: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    answer_ch: int | None = None
    future_facts: list[FutureFact] = Field(default_factory=list)
    boundary: bool = False
    question_entities: list[str] = Field(default_factory=list)
    review: Literal["pre-human-review", "human-reviewed"] = "pre-human-review"
    note: str | None = None


class Entity(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    kind: Literal["person", "place", "org", "other"] = "person"
    review: Literal["pre-human-review", "human-reviewed"] = "pre-human-review"
    note: str | None = None

    @property
    def forms(self) -> list[str]:
        return [self.name, *self.aliases]


def load_texts() -> dict[int, str]:
    return {c: clean_path(c).read_text(encoding="utf-8") for c in CHAPTERS}


def load_questions(path: Path = QUESTIONS) -> list[Question]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Question.model_validate_json(line) for line in lines if line.strip()]


def load_entities(path: Path = ENTITIES) -> list[Entity]:
    return [Entity.model_validate(e) for e in json.loads(path.read_text(encoding="utf-8"))]


def first_chapter(forms: list[str], texts: Mapping[int, str]) -> int | None:
    """이름이나 별칭 가운데 하나라도 처음 나오는 장."""
    return next((c for c in sorted(texts) if any(f in texts[c] for f in forms)), None)


def entity_index(entities: list[Entity], texts: Mapping[int, str]) -> dict[str, int | None]:
    """이름·별칭 → 첫 등장 장."""
    out: dict[str, int | None] = {}
    for e in entities:
        first = first_chapter(e.forms, texts)
        for form in e.forms:
            out[form] = first
    return out


def _cooccur_before(keys: list[str], texts: Mapping[int, str], upto: int) -> int | None:
    """키가 모두 한 문장 안에 함께 나오는 첫 장(upto 이하). 누출 탐지 오탐의 위험 신호다."""
    for c in sorted(texts):
        if c > upto:
            break
        if any(all(k in s for k in keys) for s in _SENTENCE.split(texts[c])):
            return c
    return None


def check(questions: list[Question], texts: Mapping[int, str],
          entities: list[Entity] | None = None) -> tuple[list[str], list[str]]:
    """(오류, 경고). 오류는 정답지가 틀린 것, 경고는 사람이 한 번 볼 것."""
    errors: list[str] = []
    warnings: list[str] = []
    ent = entity_index(entities or [], texts)

    ids = Counter(q.id for q in questions)
    errors += [f"{i}: id가 {k}번 나온다" for i, k in ids.items() if k > 1]

    for q in questions:
        tag = f"{q.id}({q.type}, N={q.n})"
        for ev in q.evidence:
            if len(ev.quote) > QUOTE_MAX:
                errors.append(f"{tag}: 인용이 {len(ev.quote)}자다(최대 {QUOTE_MAX})")
            if ev.quote not in texts[ev.chapter]:
                where = [c for c in sorted(texts) if ev.quote in texts[c]]
                errors.append(f"{tag}: 인용이 제{ev.chapter}장에 글자 그대로 없다" + (f"(실제 {where}장)" if where else ""))

        if q.type == "A":
            if q.answer_ch is None or q.answer_ch > q.n:
                errors.append(f"{tag}: A인데 answer_ch({q.answer_ch})가 N보다 크거나 없다")
            if not q.evidence or any(ev.chapter > q.n for ev in q.evidence):
                errors.append(f"{tag}: A의 근거는 1~N장에 있어야 한다")
            if not q.answer_keys:
                errors.append(f"{tag}: A에 answer_keys가 없다")
        elif q.type == "B":
            if q.answer_ch is None or q.answer_ch <= q.n:
                errors.append(f"{tag}: B인데 answer_ch({q.answer_ch})가 N 이하다")
            if not any(ev.chapter == q.answer_ch for ev in q.evidence):
                errors.append(f"{tag}: B에 answer_ch 장의 근거가 없다")
            leaked = [k for k in q.answer_keys if k in q.question]
            if leaked:
                errors.append(f"{tag}: B 질문 문장에 정답 키워드가 들어 있다: {leaked}")
            if not q.future_facts:
                errors.append(f"{tag}: B에 future_facts가 없다(누출 탐지에 필요)")
        else:  # C
            if q.answer_ch is not None or q.evidence:
                errors.append(f"{tag}: C에는 answer_ch와 근거가 없어야 한다")
            # answer_keys는 본문 밖의 실제 답(연재 신문 등)이 있을 때만 둔다. 답 자체가 없는 세부 사항은 비워 둔다.
            found = {k: [c for c in sorted(texts) if k in texts[c]] for k in q.answer_keys}
            found = {k: v for k, v in found.items() if v}
            if found:
                errors.append(f"{tag}: C의 키워드가 본문에 있다: {found}")

        for ff in q.future_facts:
            if ff.chapter <= q.n:
                errors.append(f"{tag}: future_facts의 장({ff.chapter})이 N 이하다")
            hit = _cooccur_before(ff.keys, texts, q.n)
            if hit:
                warnings.append(f"{tag}: future_facts 키 {ff.keys}가 제{hit}장(N 이하)의 한 문장에 함께 나온다. 누출 오탐 위험")

        for name in q.question_entities:
            if name not in ent:
                warnings.append(f"{tag}: 질문 인물 '{name}'이 인물표에 없다")
            elif ent[name] is None or ent[name] > q.n:
                errors.append(f"{tag}: 질문 인물 '{name}'이 제{ent[name]}장에 처음 나와 N보다 늦다")

    pairs: dict[str, list[Question]] = defaultdict(list)
    for q in questions:
        if q.pair_id:
            pairs[q.pair_id].append(q)
    for pid, qs in pairs.items():
        types = sorted(q.type for q in qs)
        if types != ["A", "B"]:
            errors.append(f"쌍 {pid}: A 하나와 B 하나여야 한다({types})")
        elif len({q.question for q in qs}) != 1 or len({q.answer_ch for q in qs}) != 1:
            errors.append(f"쌍 {pid}: 두 문항의 질문과 answer_ch가 같아야 한다")
        if len({q.split for q in qs}) != 1:
            errors.append(f"쌍 {pid}: 두 문항이 같은 split에 있어야 한다")

    groups = Counter((q.split, "pair" if q.pair_id else q.type) for q in questions)
    for key, want in PLAN.items():
        if groups.get(key, 0) != want:
            warnings.append(f"문항 구성 {key}: {groups.get(key, 0)}개(계획 {want}개)")
    edge_a = sum(1 for q in questions if q.boundary and q.type == "A" and q.n == q.answer_ch)
    edge_b = sum(1 for q in questions if q.boundary and q.type == "B" and q.answer_ch == q.n + 1)
    if (edge_a, edge_b) != (BOUNDARY_EACH, BOUNDARY_EACH):
        warnings.append(f"경계 문항: A(N=answer_ch) {edge_a}개, B(N=answer_ch-1) {edge_b}개(계획 각 {BOUNDARY_EACH}개)")
    return errors, warnings


SHEET = ROOT / "eval" / "gold-review.csv"


def write_sheet(questions: list[Question], path: Path = SHEET) -> Path:
    """사람 검토용 표. 엑셀·Numbers에서 한글이 깨지지 않게 BOM을 붙인다. 판정 칸은 비워 둔다."""
    import csv

    cols = ["id", "split", "type", "n", "answer_ch", "boundary", "question", "gold_answer", "answer_keys",
            "evidence", "future_facts", "question_entities", "note", "review", "판정(ok/수정/삭제)", "메모"]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for q in questions:
            w.writerow([
                q.id, q.split, q.type, q.n, q.answer_ch or "", "예" if q.boundary else "", q.question,
                q.gold_answer or "", ", ".join(q.answer_keys),
                "\n".join(f"[{ev.chapter}장] {ev.quote}" for ev in q.evidence),
                "\n".join(f"[{ff.chapter}장] {' + '.join(ff.keys)}" for ff in q.future_facts),
                ", ".join(q.question_entities), q.note or "", q.review, "", "",
            ])
    return path


def summary(questions: list[Question]) -> str:
    by = Counter((q.split, q.type) for q in questions)
    reviewed = sum(q.review == "human-reviewed" for q in questions)
    chapters = sorted(Counter(q.answer_ch for q in questions if q.answer_ch).items())
    return (f"문항 {len(questions)}개 " + ", ".join(f"{s}-{t} {k}" for (s, t), k in sorted(by.items()))
            + f" / 사람 검토 {reviewed}개 / answer_ch 분포 {dict(chapters)}")
