"""답변 뒤 코드 검증(C4의 코드 가드). 모델을 다시 부르지 않는다.

1. 인용 검증: 정규화(따옴표 접기, NFKC, 한자 괄호·× 제거, 공백 무시) 뒤 1~N장 원문에 있는 인용만 남긴다.
   - 적힌 장에 있으면 그대로, 다른 N장 이하 장에 있으면 장 번호를 고쳐 남긴다.
   - N장 뒤에만 있으면 버리고 '원문 암기 누출(L1)'로 표시한다.
   - 검색된 발췌에는 없지만 N장 이하 원문에 있으면 남기되 '기억에서 옮김'으로 표시한다.
   - 8자 미만은 근거로 보지 않는다("네." 같은 짧은 글은 어디에나 있다).
2. 인물 가드: N장 뒤에 처음 나오는 인물·지명이 답에 있으면 답하지 않는다.
3. 최종: 모델이 답했고, 검증된 인용이 하나 이상이고, 인물 가드에 걸리지 않을 때만 답한다.
   아니면 모델 문장을 버리고 고정 문구를 낸다(뒷이야기가 있는지조차 흘리지 않게).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field

from sfqa.textnorm import search_text

LAST_CHAPTER = 19
MIN_QUOTE_CHARS = 8
_WS = re.compile(r"\s+")
# 따옴표와 가운뎃점 계열을 하나로 접는다(munjigi/verify.py에서 옮김).
_FOLD = str.maketrans({c: "'" for c in "‘’‚‛“”„‟«»「」『』\""} | {c: "·" for c in "ㆍᆞ・‧∙•"})


def compact(text: str | None) -> str:
    """인용 비교용: 한자 괄호·× 제거, 따옴표 접기, NFKC(…→...), 소문자, 공백 제거."""
    folded = unicodedata.normalize("NFKC", search_text(text or "").translate(_FOLD)).translate(_FOLD)
    return _WS.sub("", folded.lower())


def refusal(n: int) -> str:
    return "작품에 나오지 않습니다." if n >= LAST_CHAPTER else f"{n}장까지는 나오지 않습니다."


@dataclass
class QuoteCheck:
    chapter: int  # 모델이 적은 장
    text: str
    status: str  # verified | chapter_fixed | future_only | not_found | too_short
    found_chapter: int | None = None
    in_context: bool = False

    @property
    def kept(self) -> bool:
        return self.status in ("verified", "chapter_fixed")


@dataclass
class Verdict:
    status: str  # answered | not_found
    answer: str
    quotes: list[QuoteCheck] = field(default_factory=list)
    future_entities: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def l1_future_quote(self) -> bool:
        return any(q.status == "future_only" for q in self.quotes)


class Verifier:
    def __init__(self, texts: Mapping[int, str], entity_first: Mapping[str, int | None]):
        self._compact = {c: compact(t) for c, t in texts.items()}
        self._entity_first = {k: v for k, v in entity_first.items() if v is not None}

    def check_quote(self, chapter: int, text: str, n: int, context: str) -> QuoteCheck:
        key = compact(text)
        q = QuoteCheck(chapter=chapter, text=text, status="not_found")
        if len(key) < MIN_QUOTE_CHARS:
            q.status = "too_short"
            return q
        q.in_context = key in compact(context)
        if 1 <= chapter <= n and key in self._compact.get(chapter, ""):
            q.status, q.found_chapter = "verified", chapter
            return q
        earlier = next((c for c in sorted(self._compact) if c <= n and key in self._compact[c]), None)
        if earlier is not None:
            q.status, q.found_chapter = "chapter_fixed", earlier
            return q
        later = next((c for c in sorted(self._compact) if c > n and key in self._compact[c]), None)
        if later is not None:
            q.status, q.found_chapter = "future_only", later
        return q

    def future_entities(self, answer: str, n: int) -> list[str]:
        return sorted(name for name, first in self._entity_first.items() if first > n and name in answer)

    def verdict(self, model_status: str | None, answer: str, quotes: list[tuple[int, str]], n: int,
                context: str, apply_guard: bool) -> Verdict:
        checks = [self.check_quote(ch, text, n, context) for ch, text in quotes]
        future = self.future_entities(answer, n)
        reasons = []
        if model_status != "answered":
            reasons.append("모델이 답하지 않음" if model_status == "not_found" else "모델 응답을 해석하지 못함")
        if model_status == "answered" and not any(c.kept for c in checks):
            reasons.append("검증된 인용 없음")
        if future:
            reasons.append(f"N장 뒤 인물·지명: {', '.join(future)}")
        if any(c.status == "future_only" for c in checks):
            reasons.append("N장 뒤 원문 인용(L1)")
        if not apply_guard:
            # 가드 없는 조건(C0~C3)은 모델 출력을 그대로 둔다. 검증 결과는 채점용으로만 남긴다.
            status = "answered" if model_status == "answered" else "not_found"
            return Verdict(status, answer if status == "answered" else refusal(n), checks, future, reasons)
        ok = model_status == "answered" and any(c.kept for c in checks) and not future
        if ok:
            return Verdict("answered", answer, [c for c in checks if c.kept], future, reasons)
        return Verdict("not_found", refusal(n), checks, future, reasons)
