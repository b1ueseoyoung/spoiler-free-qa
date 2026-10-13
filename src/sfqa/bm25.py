"""Kiwi 형태소 BM25(munjigi/retrieval.py에서 옮김).

N장 이하에서 찾을 때 IDF와 평균 문서 길이도 N장 이하 청크로만 센다. 전체 통계를 쓰면 뒷장의 낱말 빈도가
순위에 섞이기 때문이다(뒷장 글이 나오지는 않지만, 엄격하게 막는다).
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections import Counter, defaultdict
from collections.abc import Sequence
from functools import lru_cache

import numpy as np

from sfqa.textnorm import search_text

K1, B = 1.2, 0.75
# 체언(NN*, NR, NP), 용언 어간(VV, VA), 어근(XR), 외국어·숫자·한자(SL, SN, SH), 일반 부사(MAG)
CONTENT_TAG_PREFIXES = ("NN", "NR", "NP", "VV", "VA", "XR", "SL", "SN", "SH", "MAG")


@lru_cache(maxsize=1)
def kiwi():
    from kiwipiepy import Kiwi

    return Kiwi()


def tokenize_many(texts: Sequence[str]) -> list[list[str]]:
    return [
        [t.form.lower() for t in tokens if t.tag.startswith(CONTENT_TAG_PREFIXES)]
        for tokens in kiwi().tokenize([search_text(t) for t in texts])
    ]


class PrefixBM25:
    """문서는 (장, 위치) 순서로 들어온다. scores(..., upto)는 앞쪽 upto개 문서만 본다."""

    def __init__(self, docs: list[list[str]]):
        self.tf = [Counter(d) for d in docs]
        self.length = np.array([len(d) for d in docs], dtype=np.float64)
        self.cum_length = np.cumsum(self.length)
        postings: dict[str, list[int]] = defaultdict(list)
        for i, counts in enumerate(self.tf):
            for term in counts:
                postings[term].append(i)
        self.postings = dict(postings)

    def scores(self, query_terms: list[str], upto: int) -> np.ndarray:
        out = np.zeros(upto)
        if upto == 0:
            return out
        avgdl = self.cum_length[upto - 1] / upto or 1.0
        for term, qtf in Counter(query_terms).items():
            post = self.postings.get(term)
            if not post:
                continue
            m = bisect_left(post, upto)  # upto보다 앞 문서에 든 개수 = N장 이하 df
            if m == 0:
                continue
            idf = math.log1p((upto - m + 0.5) / (m + 0.5))
            for i in post[:m]:
                tf = self.tf[i][term]
                out[i] += qtf * idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * self.length[i] / avgdl))
        return out
