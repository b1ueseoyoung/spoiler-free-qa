"""장마다 따로 고정 창으로 자른다. 장을 넘는 청크는 생길 수 없다.

창 700자, 보폭 520자, 경계는 뒤쪽 공백으로 최대 30자까지 밀어 낱말 중간을 피한다.
이웃한 두 청크는 언제나 700 - 520 - 30 = 150자 이상 겹친다. 그래서 150자 이하의 인용은
어느 청크 하나에 통째로 들어간다(tests/test_chunk.py에서 정답지 인용으로 확인).
LangChain의 RecursiveCharacterTextSplitter는 문단 단위로 합쳐서 겹침이 150자보다 짧아질 수 있어 쓰지 않는다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

SIZE = 700
STRIDE = 520
SNAP = 30
MIN_OVERLAP = SIZE - STRIDE - SNAP  # 150


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    chapter: int
    start: int
    end: int
    text: str  # 원문 그대로 = 장 본문[start:end]


def _snap(text: str, pos: int) -> int:
    """pos부터 최대 SNAP자 안에서 공백 바로 뒤로 민다. 공백이 없으면 그대로 둔다."""
    if pos <= 0 or pos >= len(text):
        return max(0, min(pos, len(text)))
    for i in range(pos, min(pos + SNAP, len(text))):
        if text[i - 1].isspace():
            return i
    return pos


def chunk_chapter(chapter: int, text: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    i = 0
    while True:
        start = _snap(text, i * STRIDE)
        end = len(text) if start + SIZE >= len(text) else _snap(text, start + SIZE)
        chunks.append(Chunk(f"c{chapter:02d}-{i:03d}", chapter, start, end, text[start:end]))
        if end >= len(text):
            return chunks
        i += 1


def chunk_corpus(texts: Mapping[int, str]) -> list[Chunk]:
    """(장, 위치) 순서. 그래서 N장 이하 청크는 언제나 앞쪽 연속 구간(접두부)이다."""
    return [c for ch in sorted(texts) for c in chunk_chapter(ch, texts[ch])]
