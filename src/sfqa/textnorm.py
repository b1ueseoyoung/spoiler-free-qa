"""검색용 텍스트 만들기.

원문(한자 병기와 × 그대로)은 모델에 보여 주고 인용을 검증하는 기준이다.
검색용 텍스트는 임베딩과 어휘 검색에만 쓴다: 한자만 든 괄호와 ×를 지우고 공백을 줄인다.
"""

from __future__ import annotations

import re

# 금강(錦江), 갈재〔蘆嶺〕처럼 한자만 든 괄호. (웃으며)처럼 한글이 든 괄호는 남긴다.
_HANJA_PAREN = re.compile(r"[(〔\[][㐀-䶿一-鿿豈-﫿·,\s]+[)〕\]]")
_SPACES = re.compile(r"[ \t ]+")


def search_text(text: str) -> str:
    text = _HANJA_PAREN.sub("", text)
    text = text.replace("×", "")
    return _SPACES.sub(" ", text).strip()
