"""인물 채팅 응답 검사: 미공개 사실이나 독자만 아는 사실이 답에 섞였는지 코드로 본다. 모델을 부르지 않는다.

verify.py의 인물 가드(N장 뒤에 처음 나오는 이름)와 평가의 L2(미래 사실 키워드 조합)를 채팅에 맞게 옮겼다.
채팅에는 근거 인용이 없어 인용 검증을 쓸 수 없다. 대신 사실 문장에서 뽑은 낱말로 본다.

- 낱말: 사실의 키워드, 문장 속 고유명사와 인물·세계관 이름(강한 낱말), 일반 명사(약한 낱말).
  - 반전 사실은 동사·형용사도 약한 낱말로 본다. '장헌이 곽태허를 죽였다'는 이름이 모두 읽은 말이고 새로운 것은 동사뿐이다.
  - 강한 낱말은 글자 그대로 찾고, 약한 낱말은 답을 형태소로 나눈 낱말과 맞춘다('세실리아' 속 '실리'를 잡지 않게).
- 미공개 사실(reveal_ch > N):
  - 읽은 장 본문에 한 번도 나오지 않은 강한 낱말이 답에 있으면 잡는다.
  - 낱말 둘이 답의 한 문장에 같이 있는데 읽은 장 본문의 한 문장에 같이 나온 적이 없으면 잡는다. 둘 중 하나는 강한 낱말이어야 한다.
    '회색 깃털'과 '테오도르'는 따로는 읽은 말이지만 같이 나오면 정체가 샌다.
  - 일반 사실에서는 읽은 인물 이름 둘을 쌍으로 보지 않는다. 두 사람을 한 문장에 말했다고 드러나는 것이 없다(시험 답의 오탐, docs/character-chat.md).
    반전 사실은 본다. 관계가 반전이면 이름 둘이 곧 누출이다('양규천의 딸 설단영').
- 독자만 아는 사실(독자는 읽었지만 인물은 모른다): 같은 규칙을 본문 대신 인물 보기의 글에 댄다.
  독자에게 스포일러가 아니고 일반 낱말에 잘못 걸리기도 해서(사실 키워드 '손끝') 답을 막지 않고 기록만 한다(Finding.blocks).
- 아직 나오지 않은 이름, 숨은 별칭과 그 인물의 이름이 같이 나오는 경우도 잡는다.
- 프롬프트에 들어간 글(인물 보기)은 독자 축에서 안전하다. 본문보다 세계관 설명에 먼저 나오는 말이 있어 따로 뺀다.
- 독자가 먼저 꺼낸 말은 샌 것이 아니다. 낱말은 독자가 말했으면 빼고, 쌍은 둘 다 독자가 말했을 때만 뺀다.

한계:
- 이름이나 키워드 없이 풀어 쓴 누출("그는 살아 있소")과 독자의 추측에 "그렇소"라고만 답하는 경우는 잡지 못한다.
- 한 문장만 보므로 나눠 말한 누출("누가 죽였냐고? 장헌이오.")도 잡지 못한다. 숨은 별칭은 문장을 가리지 않는다.
- 프롬프트에 없는 사실을 모델이 말하는 일은 드물다. 이 검사는 보기 쪽 실수와 모델의 즉흥을 잡는 둘째 겹이다.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from itertools import combinations

from sfqa.ontology import Character, CharacterView, Fact, Ontology, character_view, split_facts
from sfqa.verify import compact

Words = Callable[[str], list[tuple[str, str]]]  # (낱말, proper | noun | verb)
SENTENCE = re.compile(r"(?<=[.?!…])\s+|\n+")
_HANGUL = re.compile(r"^[가-힣]+$")
# 어느 사실에나 나올 수 있어 쌍으로 봐도 뜻이 없는 일반 명사
GENERIC_NOUNS = frozenset(
    "사실 정체 이유 사람 이야기 때문 자신 생각 시절 무렵 사이 처음 지금 당시 마지막 이름 마음 얼굴 자리 문제 상태 관계 결과 내용 "
    "하나 모두 누구 무엇 그녀 그날 오늘 내일 어제 이번 다음 동안 정도 경우 모습 소리 부분 전부 혼자 함께 서로 직접 실제 일부 "
    "아무 사건 소식 집안 가족 이후 이전 최근 예전 나중 앞날 뒷일".split()
)
# 어느 문장에나 나오는 동사·형용사
GENERIC_VERBS = frozenset("하 있 되 없 보 가 오 알 주 받 말 같 않 이 아니 그렇 이렇 나오 들 만들 쓰 두 놓 나 많 크 작".split())


@dataclass(frozen=True)
class Finding:
    kind: str  # future_name | hidden_alias | hidden_word | hidden_pair | reader_only_word | reader_only_pair
    terms: tuple[str, ...]
    fact: str | None = None

    @property
    def blocks(self) -> bool:
        """답을 막는가. 독자만 아는 사실은 기록만 한다."""
        return not self.kind.startswith("reader_only")


@dataclass(frozen=True)
class Term:
    key: str  # compact로 정규화한 낱말
    strong: bool


@dataclass(frozen=True)
class _Scope:
    reader_only: list[Fact]
    hidden: list[Fact]
    lines: list[str]  # 인물 보기의 글(정규화)
    known: str
    visible: frozenset[str]  # 독자가 이 장까지 읽은 인물의 이름(정규화)


def name_forms(c: Character, upto: int | None = None) -> list[str]:
    """인물을 가리키는 이름. upto를 주면 그 장까지 공개된 별칭만. 띄어쓰기 없는 세 글자 한국 이름은 이름만 부르는 꼴(윤도하 → 도하)도 넣는다."""
    aliases = [a.form for a in c.aliases if upto is None or a.from_ch <= upto]
    forms = [c.name, c.name.split()[0], *aliases]
    if len(c.name) == 3 and _HANGUL.match(c.name):
        forms.append(c.name[1:])
    return [f for f in dict.fromkeys(forms) if len(f) >= 2]


_KIWI = None
_TAGS = {"NNP": "proper", "NNG": "noun", "VV": "verb", "VA": "verb"}


def korean_words(text: str) -> list[tuple[str, str]]:
    """형태소로 나눈 낱말. 명사는 두 글자 이상만. kiwipiepy는 처음 쓸 때 불러온다(사전을 읽느라 1초쯤 걸린다)."""
    global _KIWI
    if _KIWI is None:
        from kiwipiepy import Kiwi

        _KIWI = Kiwi()
    return [(t.form, _TAGS[t.tag]) for t in _KIWI.tokenize(text)
            if t.tag in _TAGS and (t.tag in ("VV", "VA") or len(t.form) >= 2)]


def fact_terms(f: Fact, names: Iterable[str], words: Words = korean_words) -> list[Term]:
    """사실 하나의 낱말. 같은 낱말이 강약 둘 다로 나오면 강한 쪽을 남긴다."""
    statement = compact(f.statement)
    terms: dict[str, bool] = {}
    for form, kind in words(f.statement):
        key = compact(form)
        if not key or (kind == "noun" and form in GENERIC_NOUNS):
            continue
        if kind == "verb" and (not f.is_twist or form in GENERIC_VERBS):
            continue
        terms[key] = terms.get(key, False) or kind == "proper"
    for k in f.keywords:
        key = compact(k)
        if len(key) >= 2:
            terms[key] = True
    for n in names:
        key = compact(n)
        if len(key) >= 2 and key in statement:
            terms[key] = True
    return [Term(k, s) for k, s in terms.items()]


def view_lines(view: CharacterView, onto: Ontology) -> list[str]:
    """인물 보기의 글을 한 줄씩. 프롬프트에 들어가는 글과 같은 범위다."""
    names = {c.id: c.name for c in onto.characters}
    lines = [view.name]
    lines += [f.statement for f in view.facts]
    lines += [f"{names[r.a]} {names[r.b]} {s.status}" for r in view.relations for s in r.states]
    lines += [f"{c.name} {c.form}" for c in view.calls]
    lines += [f"{w.name} {w.desc}" for w in view.world]
    return lines


def _pairs(terms: list[Term]):
    for a, b in combinations(terms, 2):
        if (a.strong or b.strong) and a.key not in b.key and b.key not in a.key:
            yield a, b


def _has(t: Term, key: str, tokens: set[str]) -> bool:
    return t.key in key if t.strong else t.key in tokens


class ChatChecker:
    """작품 하나의 검사기. 본문 문장 색인은 처음 쓰는 낱말마다 만들어 둔다."""

    def __init__(self, onto: Ontology, texts: Mapping[int, str], words: Words = korean_words):
        self.onto = onto
        self._words = words
        self._sents = [(ch, s) for ch in sorted(texts) for s in (compact(x) for x in SENTENCE.split(texts[ch])) if s]
        self._post: dict[str, frozenset[int]] = {}
        self._scopes: dict[tuple[str, int], _Scope] = {}
        names = {n for c in onto.characters for n in name_forms(c)} | {w.name for w in onto.world if w.kind != "summary"}
        self._terms = {f.id: fact_terms(f, names, words) for f in onto.facts}
        self._name_keys = sorted({compact(n) for n in names if len(compact(n)) >= 2})

    def _postings(self, key: str) -> frozenset[int]:
        if key not in self._post:
            self._post[key] = frozenset(i for i, (_, s) in enumerate(self._sents) if key in s)
        return self._post[key]

    def _seen_by(self, key: str, upto: int) -> bool:
        hits = self._postings(key)
        return bool(hits) and self._sents[min(hits)][0] <= upto

    def _together_by(self, a: str, b: str, upto: int) -> bool:
        hits = self._postings(a) & self._postings(b)
        return bool(hits) and self._sents[min(hits)][0] <= upto

    def _scope(self, character: str, upto: int) -> _Scope:
        key = (character, upto)
        if key not in self._scopes:
            if len(self._scopes) >= 256:
                self._scopes.clear()
            _, reader_only, hidden = split_facts(self.onto, character, upto)
            lines = [compact(x) for x in view_lines(character_view(self.onto, character, upto), self.onto)]
            visible = frozenset(compact(n) for c in self.onto.characters if c.first_ch is not None and c.first_ch <= upto
                                for n in name_forms(c, upto))
            self._scopes[key] = _Scope(reader_only, hidden, lines, "\n".join(lines), visible)
        return self._scopes[key]

    def check(self, reply: str, character: str, upto: int, user_text: str = "") -> list[Finding]:
        """비어 있으면 통과다. user_text: 이번 대화에서 독자가 한 말 전부. 등장하지 않은 인물은 ValueError."""
        key, said = compact(reply), compact(user_text)
        scope = self._scope(character, upto)
        tokens = {compact(form) for form, _ in self._words(reply)}
        parts = [(compact(s), {compact(form) for form, _ in self._words(s)}) for s in SENTENCE.split(reply) if s.strip()]
        out = [Finding("future_name", (n,)) for n in self._name_keys
               if n in key and n not in said and n not in scope.known and not self._seen_by(n, upto) and self._postings(n)]
        out += self._hidden_aliases(key, said, character, upto, scope)
        for f in scope.hidden:
            terms = self._present(f, key, tokens)
            out += [Finding("hidden_word", (t.key,), f.id) for t in terms
                    if t.strong and t.key not in said and t.key not in scope.known and not self._seen_by(t.key, upto)]
            out += [Finding("hidden_pair", (a.key, b.key), f.id) for a, b in _pairs(terms)
                    if (f.is_twist or not (a.key in scope.visible and b.key in scope.visible))
                    and any(_has(a, *p) and _has(b, *p) for p in parts)
                    and not (a.key in said and b.key in said) and not self._together_by(a.key, b.key, upto)
                    and not any(a.key in s and b.key in s for s in scope.lines)]
        for f in scope.reader_only:
            terms = self._present(f, key, tokens)
            out += [Finding("reader_only_word", (t.key,), f.id) for t in terms
                    if t.strong and t.key not in said and t.key not in scope.known]
            out += [Finding("reader_only_pair", (a.key, b.key), f.id) for a, b in _pairs(terms)
                    if not (a.key in said and b.key in said) and a.key in scope.known and b.key in scope.known
                    and not any(a.key in s and b.key in s for s in scope.lines)]
        return out

    def _present(self, f: Fact, key: str, tokens: set[str]) -> list[Term]:
        return [t for t in self._terms[f.id] if _has(t, key, tokens)]

    def _hidden_aliases(self, key: str, said: str, character: str, upto: int, scope: _Scope) -> list[Finding]:
        """아직 그 인물의 별칭으로 공개되지 않은 이름. 다른 인물의 공개된 별칭과 같으면 보지 않는다('취죽문 문주').

        본문에서 이미 함께 나온 이름은 숨은 것이 아니다. 45장에 등장하는 인물의 별칭 '온드레 남작'은 1장부터 본문에 나온다.
        """
        out = []
        for c in self.onto.characters:
            names = [compact(x) for x in name_forms(c, upto)]
            for h in (compact(x) for x in c.hidden_forms(upto)):
                if h not in key or h in scope.visible:
                    continue
                if any(self._together_by(h, n, upto) or any(h in s and n in s for s in scope.lines) for n in names if n != h):
                    continue
                if c.id == character and h not in said and h not in scope.known:  # 본인이 먼저 그 이름을 꺼냈다
                    out.append(Finding("hidden_alias", (h,)))
                out += [Finding("hidden_alias", (h, n)) for n in names
                        if n in key and n != h and not (h in said and n in said)][:1]
        return out
