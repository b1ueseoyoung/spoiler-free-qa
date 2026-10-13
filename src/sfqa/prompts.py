"""조건별 프롬프트(docs/eval-spec.md의 실험 조건).

- 기본(C1, C2): 발췌만 근거로 답하라는 지시만 있다. 독자가 몇 장까지 읽었는지는 알려 주지 않는다.
- 가드(C3, C4): 독자가 N장까지 읽었다고 알리고, 작품을 알아도 발췌 밖 지식을 쓰지 말고 이후 전개를 암시하지 말라고 한다.
- 검색 없음(C0): 발췌 없이 모델의 기억만으로 답하게 하되, 가드 문장은 넣는다.
프롬프트를 고치면 해시가 바뀌어 카세트가 새로 기록된다. 프롬프트는 dev 문항으로만 고친다.
"""

from __future__ import annotations

import hashlib
import json

WORK = "채만식의 장편소설 「탁류」"

FORMAT_RULES = (
    "답은 JSON으로만 낸다.\n"
    "- 답을 찾으면 status는 answered, answer에는 한국어로 짧게 답한다.\n"
    "- quotes에는 답의 근거가 되는 문장을 발췌에서 글자 그대로 옮긴다. 장 번호와 함께 1~3개, 각 150자 이하.\n"
    "- 답할 수 없으면 status는 not_found, answer는 빈 문자열, quotes는 빈 배열로 둔다."
)

BASE_SYSTEM = (
    f"너는 {WORK}을 읽는 독자의 질문에 답한다.\n"
    "아래 [발췌]에 있는 내용만 근거로 답한다. 발췌로 답할 수 없으면 답하지 않는다.\n" + FORMAT_RULES
)

GUARD_SYSTEM = (
    f"너는 {WORK}을 읽는 독자의 질문에 답한다. 독자는 제{{n}}장까지 읽었다.\n"
    "아래 [발췌]는 제1장~제{n}장에서 찾은 것이다. 발췌에 있는 내용만 근거로 답한다.\n"
    "- 네가 이 작품을 알고 있더라도 발췌에 없는 내용은 쓰지 않는다.\n"
    "- 이후 전개를 암시하지 않는다. '나중에', '아직은', '곧' 같은 말로 뒷이야기가 있음을 내비치지 않는다.\n"
    "- 발췌로 답할 수 없으면 답하지 않는다.\n" + FORMAT_RULES
)

MEMORY_SYSTEM = (
    f"너는 {WORK}을 읽는 독자의 질문에 답한다. 독자는 제{{n}}장까지 읽었다. 발췌는 주지 않는다.\n"
    "네가 이 작품에 대해 아는 것으로 답하되, 제{n}장 뒤의 내용은 말하지 않고 이후 전개를 암시하지 않는다.\n"
    "근거 문장을 정확히 기억하면 quotes에 장 번호와 함께 적고, 아니면 빈 배열로 둔다. 모르면 답하지 않는다.\n"
    + FORMAT_RULES.replace("발췌에서 글자 그대로 옮긴다", "원문에서 글자 그대로 옮긴다")
)


def system_prompt(condition: str, n: int) -> str:
    if condition == "C0":
        return MEMORY_SYSTEM.format(n=n)
    if condition in ("C3", "C4"):
        return GUARD_SYSTEM.format(n=n)
    return BASE_SYSTEM


def user_prompt(question: str, context: str | None) -> str:
    if context is None:
        return f"질문: {question}"
    return f"[발췌]\n{context}\n\n질문: {question}"


def prompt_hash() -> str:
    body = json.dumps([BASE_SYSTEM, GUARD_SYSTEM, MEMORY_SYSTEM], ensure_ascii=False)
    return hashlib.sha256(body.encode()).hexdigest()[:16]
