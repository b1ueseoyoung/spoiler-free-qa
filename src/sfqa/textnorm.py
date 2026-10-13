"""검색용 텍스트 만들기.

원문(한자 병기와 × 그대로)은 모델에 보여 주고 인용을 검증하는 기준이다.
검색용 텍스트는 임베딩과 어휘 검색에만 쓴다: 한자만 든 괄호와 ×를 지우고 공백을 줄인다.
"""

from __future__ import annotations

import re
import unicodedata

# 금강(錦江), 갈재〔蘆嶺〕처럼 한자만 든 괄호. (웃으며)처럼 한글이 든 괄호는 남긴다.
_HANJA_PAREN = re.compile(r"[(〔\[][㐀-䶿一-鿿豈-﫿·,\s]+[)〕\]]")
_SPACES = re.compile(r"[ \t ]+")


def search_text(text: str) -> str:
    text = _HANJA_PAREN.sub("", text)
    text = text.replace("×", "")
    return _SPACES.sub(" ", text).strip()


# 정답 키워드 비교용(munjigi/normalize.py에서 옮김): NFKC·소문자, 천 단위 쉼표 제거, 문장부호·공백 제거.
# 숫자 사이의 '.', '~', ':', '/'는 남긴다(지우면 '9.9'와 '99'가 같아진다).
_THOUSANDS_COMMA = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_NON_WORD = re.compile(r"(?:(?!(?<=\d)[.~:/](?=\d))[\W_])+")


def match_form(text: str) -> str:
    text = _THOUSANDS_COMMA.sub("", unicodedata.normalize("NFKC", search_text(text or "")).lower())
    return _NON_WORD.sub("", text)


def has_key(text: str, key: str) -> bool:
    """키워드가 글에 있는가. 'A|B'는 둘 중 하나라도 있으면 된다(예: '자현|자수')."""
    form = match_form(text)
    return any(match_form(alt) and match_form(alt) in form for alt in key.split("|"))
