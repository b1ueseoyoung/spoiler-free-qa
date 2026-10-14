"""KLUE-MRC 독해 평가(docs/model-eval-spec.md 1번).

- 표본: validation에서 유형별 80문항, seed 20261014. eval/klue/에 내보낸다(CC BY-SA 4.0, eval/klue/README.md).
- 지문을 주고 묻는다(검색 없음). 모델은 {status, answer} JSON으로 답한다.
- 채점은 사람이 만든 정답과 비교만 한다(모델 판정 없음).
  정규화·EM·문자 F1·함정 적중은 munjigi(normalize.py, scoring.py)에서 옮겼다.
- 답 없는 문항(유형 3)의 answers는 정답이 아니라 지문 속 가짜 답(함정)이다.
"""

from __future__ import annotations

import html
import json
import random
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from sfqa.llm import LLMRequest, LLMResponse
from sfqa.paths import ROOT

SAMPLE = ROOT / "eval" / "klue" / "klue-mrc-val-240.jsonl"
PER_TYPE = 80
SEED = 20261014
TYPES = (1, 2, 3)  # 1 바꿔 쓴 표현, 2 여러 문장 추론, 3 지문에 답 없음

KLUE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["answered", "not_found"]},
        "answer": {"type": "string"},
    },
    "required": ["status", "answer"],
    "additionalProperties": False,
}

# 두 status 값을 모두 설명한다. not_found 조건만 적었더니 qwen 두 모델이 쉬운 문항까지 모두 not_found로 답했다
# (2026-10-14 점검, docs/model-eval-spec.md 변경 기록). test 프롬프트(prompts.py)와 같은 방식이다.
SYSTEM = """너는 주어진 지문만 근거로 질문에 답한다.
- 지문에서 답을 찾으면 status를 answered로 하고, answer에 질문이 묻는 짧은 구(이름, 숫자, 날짜, 명사구)를 지문에 나온 표현 그대로 쓴다. 문장으로 풀어 쓰지 않는다.
- 지문에서 답을 찾을 수 없으면 status를 not_found로, answer를 빈 문자열로 한다. 지문 밖 지식으로 답하지 않는다.
- JSON 하나로만 답한다."""


@dataclass(frozen=True)
class Item:
    guid: str
    question_type: int
    is_impossible: bool
    title: str
    context: str
    question: str
    answers: tuple[str, ...]


def to_item(row: dict) -> Item:
    """parquet 행을 Item으로. 제목과 정답 표기의 HTML 엔티티('&amp;')는 푼다(지문에는 없다)."""
    answers = row["answers"]["text"] if isinstance(row["answers"], dict) else row["answers"]
    return Item(
        guid=row["guid"], question_type=int(row["question_type"]), is_impossible=bool(row["is_impossible"]),
        title=html.unescape(row["title"]), context=row["context"], question=row["question"],
        answers=tuple(html.unescape(a) for a in answers),
    )


def sample(rows: list[dict], per_type: int = PER_TYPE, seed: int = SEED) -> list[Item]:
    """유형마다 guid 순으로 정렬한 뒤 같은 난수열로 뽑는다(파일 행 순서와 무관하게 같은 표본)."""
    rng = random.Random(seed)
    picked: list[Item] = []
    for t in TYPES:
        pool = sorted((r for r in rows if int(r["question_type"]) == t), key=lambda r: r["guid"])
        picked += [to_item(r) for r in rng.sample(pool, per_type)]
    bad = [i.guid for i in picked if i.is_impossible != (i.question_type == 3)]
    if bad:
        raise ValueError(f"유형 3과 is_impossible이 어긋난다: {bad[:3]}")
    return picked


def write_sample(items: list[Item], path: Path = SAMPLE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(asdict(i), ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
    return path


def load_sample(path: Path = SAMPLE) -> list[Item]:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [Item(**{**r, "answers": tuple(r["answers"])}) for r in rows]


def user_prompt(item: Item) -> str:
    return f"[지문] {item.title}\n{item.context}\n\n[질문] {item.question}"


def build_request(item: Item, model: str, max_tokens: int, seed: int, reasoning_effort: str) -> LLMRequest:
    return LLMRequest(model=model, system=SYSTEM, user=user_prompt(item), schema_name="klue_answer",
                      schema=KLUE_SCHEMA, max_tokens=max_tokens, seed=seed, reasoning_effort=reasoning_effort)


# --- 정규화(munjigi/normalize.py에서 옮김) ---
# - 정답 표기는 주석자가 고른 명사구라 조사를 떼지 않는다. 괄호 병기를 뺀 형태만 더한다.
# - 모델 답은 끝에 붙은 조사·'이다'를 정해진 표면형 목록으로만 떼어 본다(형태소 분석은 이름 끝 음절을 자른다).
# - 숫자 사이의 '.', '~', ':', '/'는 지우지 않는다('9.9%'와 '99%'가 같아지지 않게).
_THOUSANDS_COMMA = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_NON_WORD = re.compile(r"(?:(?!(?<=\d)[.~:/](?=\d))[\W_])+")
_PAREN = re.compile(r"\s*[(\[（［][^)\]）］]*[)\]）］]")
# '도', '만'은 '경기도', '3만'을 망가뜨려서 뺐다.
TRAILING_PARTICLES = (
    "입니다", "이었다", "이에요", "에서", "에게", "으로", "이다", "였다", "이며",
    "은", "는", "이", "가", "을", "를", "의", "에", "로", "와", "과",
)
MIN_STRIPPED_LEN = 2


def normalize_text(text: str) -> str:
    """NFKC·소문자, 천 단위 쉼표 제거, 문장부호·기호·공백 제거(숫자 사이 구분자는 유지)."""
    text = _THOUSANDS_COMMA.sub("", unicodedata.normalize("NFKC", text).lower())
    return _NON_WORD.sub("", text)


@lru_cache(maxsize=50_000)
def gold_forms(text: str) -> frozenset[str]:
    """정답 표기의 비교 형태: 원형과 괄호 병기를 뺀 형태. 기호만이면 공백만 지운 원문."""
    forms = {normalize_text(text), normalize_text(_PAREN.sub("", text))} - {""}
    return frozenset(forms or {re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))})


@lru_cache(maxsize=50_000)
def pred_forms(text: str) -> frozenset[str]:
    """모델 답의 비교 형태: gold_forms에 끝 조사를 뗀 형태를 더한다."""
    base = gold_forms(text)
    stripped = {f[: -len(p)] for f in base for p in TRAILING_PARTICLES
                if f.endswith(p) and len(f) - len(p) >= MIN_STRIPPED_LEN}
    return base | frozenset(stripped)


def f1_score(p: str, g: str) -> float:
    """문자 단위 F1(KorQuAD 방식)."""
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> bool:
    return bool(pred_forms(pred) & gold_forms(gold))


def char_f1(pred: str, gold: str) -> float:
    return max(f1_score(p, g) for p in pred_forms(pred) for g in gold_forms(gold))


# --- 채점 ---
OUTCOMES = {
    "answerable": ["exact", "inexact", "missed", "format_fail"],
    "unanswerable": ["abstained", "ungrounded", "format_fail"],
}
LABELS = {"exact": "EM", "inexact": "EM 아님", "missed": "놓침", "format_fail": "형식 실패",
          "abstained": "보류", "ungrounded": "근거 없는 답"}


def read_response(resp: LLMResponse) -> dict:
    """응답에서 (status, answer)와 형식 문제를 읽는다. 안전 거절(Claude refusal)은 형식 실패가 아니라 보류로 센다."""
    refused = resp.finish_reason == "refusal"
    data, _ = resp.parsed()
    ok = (isinstance(data, dict) and data.get("status") in ("answered", "not_found")
          and isinstance(data.get("answer"), str))
    return {
        "status": data["status"] if ok else ("not_found" if refused else None),
        "answer": data["answer"] if ok else "",
        "truncated": resp.truncated,
        "invalid": not ok and not resp.truncated and not refused,
        "refused": refused,
    }


def score_item(item: Item, rec: dict) -> dict:
    """rec: read_response 결과. 빈 답은 보류로 본다."""
    answer = rec["answer"] or ""
    abstained = rec["status"] != "answered" or not answer.strip()
    em, f1, trap = False, 0.0, False
    if rec["invalid"] or rec["truncated"]:
        outcome = "format_fail"
    elif item.is_impossible:
        outcome = "abstained" if abstained else "ungrounded"
        trap = not abstained and any(pred_forms(answer) & gold_forms(a) for a in item.answers)
    elif abstained:
        outcome = "missed"
    else:
        em = any(exact_match(answer, g) for g in item.answers)
        f1 = max(char_f1(answer, g) for g in item.answers)
        outcome = "exact" if em else "inexact"
    return {"outcome": outcome, "em": em, "f1": round(f1, 4), "trap_hit": trap}


def aggregate(items: list[Item], records: list[dict]) -> dict:
    """유형별 개수. F1 평균은 답 있는 문항 전체(놓침·형식 실패는 0)로 낸다."""
    by_guid = {i.guid: i for i in items}
    table: dict[int, Counter] = {t: Counter() for t in TYPES}
    f1_sum: Counter = Counter()
    for r in records:
        item = by_guid[r["guid"]]
        s = score_item(item, r)
        c = table[item.question_type]
        c["total"] += 1
        c[s["outcome"]] += 1
        c["trap_hit"] += s["trap_hit"]
        c["refused"] += r["refused"]
        f1_sum[item.question_type] += s["f1"]
    out = {}
    for t, c in table.items():
        row = dict(c)
        if t != 3 and c["total"]:
            row["f1_mean"] = round(f1_sum[t] / c["total"], 4)
        out[t] = row
    return out
