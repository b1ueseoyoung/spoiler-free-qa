"""장 시간축 온톨로지: 인물·사실·관계에 '독자가 몇 장에서 아는가'와 '인물이 몇 장부터 아는가'를 단다.

등장인물 채팅 확장안 2번이다(docs/ontology.md). 차단은 두 겹이다.
- 독자 축: 독자가 upto장까지 읽었으면 reveal_ch가 upto보다 큰 것은 누구의 보기에도 넣지 않는다.
- 인물 축: 독자가 읽은 사실이라도 인물이 at장(기본은 upto)까지 알게 되지 않았으면 그 인물의 보기에 넣지 않는다.

보기(reader_view, character_view)에는 프롬프트에 그대로 넣어도 되는 것만 담는다. 인물이 모르는 사실과
아직 공개되지 않은 사실의 문장은 넣지 않는다("모르는 척하라"는 지시는 샌다). 평가는 split_facts로 나눈다.
아직 등장하지 않은 인물은 이름·관계·호칭을 보기에 넣지 않는다. 그 인물 이야기는 공개된 사실 문장으로만 들어간다.
인물이 아는 사실이라도 그 인물의 아직 공개되지 않은 별칭이 들어 있으면 그 인물의 보기에서 뺀다(숨은 정체 누출).

정답 온톨로지는 합성 웹소설 설정집에서 만든다(from_synthetic). 본문에서 뽑는 추출기는 다음 단계다.
설정집의 성격·말투·뒷이야기·결말은 화 표시가 없고 뒤 화 내용이 섞여 있어(예: '(24화 공개)') 옮기지 않는다.
판단 근거 메모(note)는 옮기되 보기에는 넣지 않는다. 작가용 메모라 뒤 화를 가리키는 말이 섞인다.

실행: uv run python -m sfqa.ontology build | show (sfqa 명령에 붙이는 것은 성능 개선 작업과 합친 뒤에 한다)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path

from pydantic import BaseModel, Field

from sfqa.paths import DATA

SYNTHETIC_ROOT = DATA / "synthetic"
WORLD_KINDS = {"places": "place", "institutions": "institution", "magic": "magic", "festivals": "festival"}


class Knowing(BaseModel):
    """인물이 그 사실(관계 상태)을 알게 되는 장. 0은 이야기 시작 전부터 안다."""

    character: str
    from_ch: int = Field(ge=0)
    note: str = ""


class Evidence(BaseModel):
    chapter: int
    lines: list[int]  # 정제 본문의 줄 번호(1부터)


class Fact(BaseModel):
    id: str
    statement: str
    category: str = ""
    reveal_ch: int = Field(ge=1)  # 독자가 처음 알게 되는 장
    known_by: list[Knowing] = Field(default_factory=list)
    is_twist: bool = False
    inferable_ch: int | None = None  # 공개 전에 독자가 짐작할 수 있게 되는 가장 이른 장(평가용)
    keywords: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    common: bool = False  # 이야기 속 세상 누구나 아는 상식. 뒤에 나오는 인물까지 모두 0장부터 안다(추출기가 표시)

    def known_from(self, character: str) -> int | None:
        if self.common:
            return 0
        return min((k.from_ch for k in self.known_by if k.character == character), default=None)


class Alias(BaseModel):
    form: str
    from_ch: int
    to_ch: int | None = None  # 없으면 끝까지


class Calling(BaseModel):
    """by가 이 인물을 부르는 호칭."""

    by: str
    form: str
    from_ch: int
    to_ch: int | None = None
    context: str = ""  # 공석·사석 같은 조건


class Character(BaseModel):
    id: str
    name: str
    first_ch: int | None = Field(ge=1)  # 처음 등장하는 장. 언급만 된 장이 아니다. None은 등장하지 않음(채팅 불가)
    mention_ch: int | None = None  # 처음 언급된 장(추출기가 채운다)
    aliases: list[Alias] = Field(default_factory=list)
    called: list[Calling] = Field(default_factory=list)
    gender: str = ""
    age: int | None = None
    is_minor: bool = False  # 설정 검사용. 독자 진행도와 상관없이 처음부터 쓴다

    @property
    def forms(self) -> list[str]:
        """본문에서 찾을 이름(두 글자 이상). 서양식 이름은 띄어쓰기 앞(이름)도 넣는다."""
        forms = [self.name, self.name.split()[0], *(a.form for a in self.aliases)]
        return [f for f in dict.fromkeys(forms) if len(f) >= 2]

    def hidden_forms(self, upto: int) -> list[str]:
        """독자가 upto장까지 읽었을 때 아직 이 인물의 별칭으로 공개되지 않은 이름(예: 37장 전의 '회색 깃털').

        이미 공개된 이름에 들어 있는 짧은 이름('테오도르' 안의 '테오')은 빼서, 본인 이야기를 통째로 막지 않는다.
        """
        seen = [self.name, self.name.split()[0], *(a.form for a in self.aliases if a.from_ch <= upto)]
        return [a.form for a in self.aliases
                if a.from_ch > upto and len(a.form) >= 2 and not any(a.form in s for s in seen)]


class RelationState(BaseModel):
    from_ch: int = Field(ge=0)  # 설정집 값. 0은 이야기 시작 전부터의 배경 상태
    reveal_ch: int = Field(ge=1)  # 독자가 이 상태를 처음 알 수 있는 장(배경 상태는 추정, from_synthetic 참고)
    status: str
    public: bool = False  # 이야기 속 세상에 알려진 상태인가
    known_by: list[Knowing] = Field(default_factory=list)
    note: str = ""


class Relation(BaseModel):
    id: str
    a: str
    b: str
    canon_romance: bool = False
    shipping_candidate: bool = False  # 2차 창작에서 엮을 만한 쌍(설정집 표시)
    states: list[RelationState] = Field(default_factory=list)  # 이야기 순서


class Scene(BaseModel):
    """장마다 시점 인물·장소·등장인물. 줄거리 기준이라 한 장 안의 여러 장면을 하나로 묶는다. 보기에는 아직 쓰지 않는다."""

    chapter: int
    pov: str = ""
    location: str = ""
    present: list[str] = Field(default_factory=list)


class WorldItem(BaseModel):
    kind: str  # summary, place, institution, magic, festival
    name: str
    reveal_ch: int = Field(ge=1)
    desc: str = ""  # reveal_ch까지 공개되는 내용만(설정집 규칙)


class Ontology(BaseModel):
    work: str
    title: str
    chapters: int
    source: str  # gold:synthetic(설정집) 또는 뒤에 붙일 추출기 이름
    characters: list[Character]
    facts: list[Fact]
    relations: list[Relation] = Field(default_factory=list)
    scenes: list[Scene] = Field(default_factory=list)
    world: list[WorldItem] = Field(default_factory=list)


# ---------- 보기: 프롬프트에 넣어도 되는 것만 ----------


class FactItem(BaseModel):
    id: str
    statement: str
    chapter: int  # 독자에게 공개된 장
    since: int | None = None  # 인물 보기에서 그 인물이 알게 된 장(0은 처음부터)
    evidence: list[Evidence] = Field(default_factory=list)


class StateItem(BaseModel):
    since: int  # 이야기에서 이 상태가 시작되는 장(0은 이야기 전부터)
    status: str
    public: bool


class RelationItem(BaseModel):
    id: str
    a: str
    b: str
    states: list[StateItem]  # 이야기 순서. 마지막이 지금 상태


class CharacterItem(BaseModel):
    id: str
    name: str
    first_ch: int
    aliases: list[str]


class CallItem(BaseModel):
    target: str
    name: str
    form: str
    context: str = ""


class ReaderView(BaseModel):
    work: str
    upto: int
    characters: list[CharacterItem]
    facts: list[FactItem]
    relations: list[RelationItem]
    world: list[WorldItem]
    hidden: int  # 아직 공개되지 않은 사실 수(문장은 넣지 않는다)


class CharacterView(BaseModel):
    work: str
    upto: int
    at: int
    character: str
    name: str
    facts: list[FactItem]
    relations: list[RelationItem]
    calls: list[CallItem]  # 이 인물이 at장에 다른 인물을 부르는 호칭
    world: list[WorldItem]  # 세계관 항목은 인물별 앎 표시가 없어 독자 기준으로만 자른다


def _chapters(onto: Ontology, upto: int, at: int | None = None) -> int:
    if not 1 <= upto <= onto.chapters:
        raise ValueError(f"upto는 1~{onto.chapters}장이어야 한다: {upto}")
    at = upto if at is None else at
    if not 1 <= at <= upto:
        raise ValueError(f"at은 1~upto({upto})장이어야 한다: {at}")
    return at


def _knows(known_by: list[Knowing], character: str, at: int) -> bool:
    return any(k.character == character and k.from_ch <= at for k in known_by)


def _appeared(onto: Ontology, upto: int) -> dict[str, Character]:
    return {c.id: c for c in onto.characters if c.first_ch is not None and c.first_ch <= upto}


def _relations(onto: Ontology, upto: int, keep: Callable[[RelationState], bool]) -> list[RelationItem]:
    seen = _appeared(onto, upto)
    out = []
    for r in onto.relations:
        if r.a not in seen or r.b not in seen:
            continue
        states = [StateItem(since=s.from_ch, status=s.status, public=s.public)
                  for s in r.states if s.reveal_ch <= upto and keep(s)]
        if states:
            out.append(RelationItem(id=r.id, a=r.a, b=r.b, states=states))
    return out


def split_facts(onto: Ontology, character: str, upto: int, at: int | None = None) -> tuple[list[Fact], list[Fact], list[Fact]]:
    """(인물이 말할 수 있는 것, 독자만 아는 것, 아직 공개되지 않은 것). 평가용이다. 프롬프트에는 첫째만 넣는다.

    인물이 알아도 그 인물의 아직 공개되지 않은 별칭이 든 사실은 둘째로 보낸다. 37장 전의 테오도르가
    '회색 깃털' 편지를 안다고 말하면 정체가 샌다.
    """
    at = _chapters(onto, upto, at)
    me = next((c for c in onto.characters if c.id == character), None)
    if me is None:
        raise ValueError(f"없는 인물: {character}")
    secret = me.hidden_forms(upto)
    known, reader_only, hidden = [], [], []
    for f in onto.facts:
        if f.reveal_ch > upto:
            hidden.append(f)
        elif (f.common or _knows(f.known_by, character, at)) and not any(s in f.statement for s in secret):
            known.append(f)
        else:
            reader_only.append(f)
    return known, reader_only, hidden


def reader_view(onto: Ontology, upto: int) -> ReaderView:
    _chapters(onto, upto)
    chars = [CharacterItem(id=c.id, name=c.name, first_ch=c.first_ch,
                           aliases=list(dict.fromkeys(a.form for a in c.aliases if a.from_ch <= upto)))
             for c in _appeared(onto, upto).values()]
    facts = [FactItem(id=f.id, statement=f.statement, chapter=f.reveal_ch, evidence=f.evidence)
             for f in onto.facts if f.reveal_ch <= upto]
    return ReaderView(work=onto.work, upto=upto, characters=chars, facts=facts,
                      relations=_relations(onto, upto, lambda s: True),
                      world=[w for w in onto.world if w.reveal_ch <= upto],
                      hidden=sum(f.reveal_ch > upto for f in onto.facts))


def character_view(onto: Ontology, character: str, upto: int, at: int | None = None) -> CharacterView:
    """독자가 upto장까지 읽었을 때, 인물이 at장 시점에 아는 것. 아직 등장하지 않은 인물과는 채팅할 수 없다."""
    at = _chapters(onto, upto, at)
    seen = _appeared(onto, upto)
    if character not in seen:
        raise ValueError(f"{character}는 없는 인물이거나 {upto}장까지 등장하지 않았다")
    known, _, _ = split_facts(onto, character, upto, at)
    facts = [FactItem(id=f.id, statement=f.statement, chapter=f.reveal_ch, since=f.known_from(character), evidence=f.evidence)
             for f in known]
    calls = [CallItem(target=c.id, name=c.name, form=k.form, context=k.context)
             for c in seen.values() for k in c.called
             if k.by == character and k.from_ch <= at <= (k.to_ch or onto.chapters)]
    return CharacterView(work=onto.work, upto=upto, at=at, character=character, name=seen[character].name, facts=facts,
                         relations=_relations(onto, upto, lambda s: _knows(s.known_by, character, at)),
                         calls=calls, world=[w for w in onto.world if w.reveal_ch <= upto])


# ---------- 검사 ----------


def check(onto: Ontology) -> list[str]:
    """id 중복, 없는 인물, 장 범위. 문제 문장 목록을 돌려준다(없으면 빈 목록)."""
    n, errs = onto.chapters, []
    ids = [c.id for c in onto.characters]
    for kind, seq in (("인물", ids), ("사실", [f.id for f in onto.facts]), ("관계", [r.id for r in onto.relations])):
        errs += [f"{kind} id 중복: {i}" for i, k in Counter(seq).items() if k > 1]
    people = set(ids)

    def who(where: str, cid: str) -> None:
        if cid not in people:
            errs.append(f"{where}: 없는 인물 {cid}")

    def span(where: str, lo: int, hi: int | None = None, low: int = 1) -> None:
        if not low <= lo <= n or (hi is not None and not lo <= hi <= n):
            errs.append(f"{where}: 장 범위 밖({lo}~{hi if hi is not None else ''}, 작품은 {n}장)")

    for c in onto.characters:
        if c.first_ch is not None:
            span(f"인물 {c.id} 첫 등장", c.first_ch)
        if c.mention_ch is not None:
            span(f"인물 {c.id} 첫 언급", c.mention_ch)
        for a in c.aliases:
            span(f"인물 {c.id} 별칭 '{a.form}'", a.from_ch, a.to_ch)
        for k in c.called:
            who(f"인물 {c.id} 호칭 by", k.by)
            span(f"인물 {c.id} 호칭 '{k.form}'", k.from_ch, k.to_ch)
    for f in onto.facts:
        span(f"사실 {f.id} 공개", f.reveal_ch)
        if f.inferable_ch is not None and not 1 <= f.inferable_ch <= f.reveal_ch:
            errs.append(f"사실 {f.id}: 짐작 가능 장({f.inferable_ch})이 공개 장({f.reveal_ch})보다 뒤다")
        for k in f.known_by:
            who(f"사실 {f.id} known_by", k.character)
            span(f"사실 {f.id} known_by {k.character}", k.from_ch, low=0)
    for r in onto.relations:
        who(f"관계 {r.id} a", r.a)
        who(f"관계 {r.id} b", r.b)
        for s in r.states:
            span(f"관계 {r.id} '{s.status[:12]}' 시작", s.from_ch, low=0)
            span(f"관계 {r.id} '{s.status[:12]}' 공개", s.reveal_ch)
            for k in s.known_by:
                who(f"관계 {r.id} known_by", k.character)
                span(f"관계 {r.id} known_by {k.character}", k.from_ch, low=0)
    for s in onto.scenes:
        span(f"장면 {s.chapter}", s.chapter)
        for cid in ([s.pov] if s.pov else []) + s.present:
            who(f"장면 {s.chapter}", cid)
    for w in onto.world:
        span(f"세계관 '{w.name}'", w.reveal_ch)
    return errs


# ---------- 합성 웹소설 설정집 → 정답 온톨로지 ----------


def _lines_with(text: str, keywords: list[str]) -> list[int]:
    """키워드(두 글자 이상) 가운데 하나라도 든 줄 번호. 기계 검사(check_synthetic.py)와 같은 기준이다."""
    kws = [k for k in keywords if len(k) >= 2]
    return [i for i, line in enumerate(text.splitlines(), 1) if any(k in line for k in kws)]


def _knowing(k: dict) -> Knowing:
    return Knowing(character=k["character"], from_ch=int(k["from_ep"]), note=k.get("note", ""))


def _character(c: dict) -> Character:
    first = int(c["first_appearance_ep"])
    aliases = [Alias(form=a, from_ch=first) for a in c.get("aliases", [])]  # 정해진 별칭은 첫 등장부터 끝까지
    aliases += [Alias(form=a["alias"], from_ch=int(a["from_ep"]), to_ch=a.get("to_ep")) for a in c.get("aliases_timed", [])]
    called = [Calling(by=h["by"], form=h.get("form") or h["call"], from_ch=int(h["from_ep"]), to_ch=h.get("to_ep"),
                      context=h.get("context", ""))  # 파일럿(wn01)은 form, 나머지는 call
              for h in c.get("how_called", [])]
    return Character(id=c["id"], name=c["name"], first_ch=first, aliases=aliases, called=called,
                     gender=c.get("gender", ""), age=c.get("age"), is_minor=bool(c.get("is_minor", False)))


def _first_together(a: Character, b: Character, texts: Mapping[int, str]) -> int | None:
    return next((ch for ch in sorted(texts)
                 if any(f in texts[ch] for f in a.forms) and any(f in texts[ch] for f in b.forms)), None)


def _relation(r: dict, chars: Mapping[str, Character], texts: Mapping[int, str]) -> Relation:
    a, b = chars.get(r["a"]), chars.get(r["b"])  # 없는 인물은 check가 잡는다
    states = []
    for s in r.get("timeline", []):
        reveal = int(s["from_ep"])
        if reveal == 0:  # 배경 상태: 규칙은 '두 인물이 처음 함께 언급되는 화'에 공개. 그 화가 설정집에 없어 늦게 잡는다
            firsts = [x.first_ch for x in (a, b) if x]
            together = _first_together(a, b, texts) if a and b else None
            reveal = max([1, *firsts, together or 1])
        states.append(RelationState(from_ch=int(s["from_ep"]), reveal_ch=reveal, status=s["status"],
                                    public=bool(s.get("public", False)),
                                    known_by=[_knowing(k) for k in s.get("known_by", [])], note=s.get("note", "")))
    return Relation(id=r["id"], a=r["a"], b=r["b"], canon_romance=bool(r.get("canon_romance", False)),
                    shipping_candidate=bool(r.get("shipping_candidate", False)), states=states)


def _fact(f: dict, texts: Mapping[int, str]) -> Fact:
    kws = f.get("evidence_keywords", [])
    lines = _lines_with(texts.get(int(f["reveal_ep"]), ""), kws)
    return Fact(id=f["id"], statement=f["statement"], category=f.get("category", ""), reveal_ch=int(f["reveal_ep"]),
                known_by=[_knowing(k) for k in f.get("known_by", [])], is_twist=bool(f.get("is_twist", False)),
                inferable_ch=f.get("inferable_from_ep"), keywords=kws,
                evidence=[Evidence(chapter=int(f["reveal_ep"]), lines=lines)] if lines else [])


def _world(world: dict) -> list[WorldItem]:
    items = []
    if world.get("summary"):
        items.append(WorldItem(kind="summary", name="세계 요약", reveal_ch=int(world.get("summary_first_reveal_ep", 1)),
                               desc=world["summary"]))
    for key, kind in WORLD_KINDS.items():
        items += [WorldItem(kind=kind, name=it["name"], reveal_ch=int(it["first_reveal_ep"]), desc=it.get("desc", ""))
                  for it in world.get(key, [])]
    return items


def synthetic_ids(root: Path = SYNTHETIC_ROOT) -> list[str]:
    return sorted(p.parent.name for p in root.glob("*/bible.json"))


def from_synthetic(work_id: str, root: Path = SYNTHETIC_ROOT) -> Ontology:
    """설정집(bible.json)·사실 장부(facts.jsonl)·줄거리(outline.json)를 정답 온톨로지로 옮긴다. 모델을 부르지 않는다.

    본문(clean/)은 사실의 근거 줄 번호와 배경 관계의 공개 장 추정에만 쓴다.
    """
    from sfqa.gold import load_texts
    from sfqa.works import load_work

    work = load_work(work_id, root)
    bible = json.loads((work.dir / "bible.json").read_text(encoding="utf-8"))
    outline = json.loads((work.dir / "outline.json").read_text(encoding="utf-8"))
    raw = [json.loads(line) for line in (work.dir / "facts.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    texts = load_texts(work)
    chars = [_character(c) for c in bible["characters"]]
    by_id = {c.id: c for c in chars}
    return Ontology(
        work=work.id, title=work.title, chapters=work.chapters, source="gold:synthetic", characters=chars,
        facts=[_fact(f, texts) for f in raw],
        relations=[_relation(r, by_id, texts) for r in bible.get("relationships", [])],
        scenes=[Scene(chapter=int(e["ep"]), pov=e.get("pov", ""), location=e.get("location", ""),
                      present=e.get("characters_present", [])) for e in outline],
        world=_world(bible.get("world", {})),
    )


# ---------- 명령 ----------


def _summary(onto: Ontology, problems: list[str]) -> str:
    states = [s for r in onto.relations for s in r.states]
    return (f"{onto.work} 「{onto.title}」: 인물 {len(onto.characters)}, "
            f"사실 {len(onto.facts)}(반전 {sum(f.is_twist for f in onto.facts)}, "
            f"짐작 가능 {sum(f.inferable_ch is not None for f in onto.facts)}, "
            f"근거 줄 없음 {sum(not f.evidence for f in onto.facts)}), "
            f"관계 {len(onto.relations)}(상태 {len(states)}, 배경 {sum(s.from_ch == 0 for s in states)}), "
            f"장면 {len(onto.scenes)}, 세계관 {len(onto.world)}, 문제 {len(problems)}")


def _print_character(onto: Ontology, v: CharacterView) -> None:
    _, reader_only, hidden = split_facts(onto, v.character, v.upto, v.at)
    print(f"「{onto.title}」 독자 {v.upto}장, {v.name}({v.character})의 {v.at}장 시점")
    print(f"- 아는 사실 {len(v.facts)}개 / 독자만 아는 사실 {len(reader_only)}개 / 아직 공개 안 된 사실 {len(hidden)}개")
    print("[아는 사실]")
    for f in v.facts:
        print(f"  {f.id} ({f.chapter}장 공개, {'처음부터' if f.since == 0 else f'{f.since}장부터'} 앎) {f.statement}")
    print("[관계]")
    names = {c.id: c.name for c in onto.characters}
    for r in v.relations:
        print(f"  {names[r.a]}–{names[r.b]}: "
              + " → ".join(f"{s.status}({'이야기 전부터' if s.since == 0 else f'{s.since}장'})" for s in r.states))
    print("[호칭]")
    for c in v.calls:
        print(f"  {c.name}: {c.form}" + (f" ({c.context})" if c.context else ""))


def _print_reader(onto: Ontology, v: ReaderView) -> None:
    print(f"「{onto.title}」 독자 {v.upto}장까지: 인물 {len(v.characters)}, 공개 사실 {len(v.facts)}, "
          f"관계 {len(v.relations)}, 세계관 {len(v.world)}, 아직 공개 안 된 사실 {v.hidden}")
    for c in v.characters:
        print(f"  {c.id} {c.name}({c.first_ch}장 등장)" + (f" 별칭: {', '.join(c.aliases)}" if c.aliases else ""))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sfqa.ontology", description="장 시간축 온톨로지(합성 웹소설 정답)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_build = sub.add_parser("build", help="설정집에서 정답 온톨로지를 만들고 검사한다(모델 없음)")
    p_build.add_argument("--work", action="append", help="작품 id(여러 번 쓸 수 있다). 없으면 data/synthetic 전부")
    p_build.add_argument("--out", help="온톨로지 JSON(<작품>.json)을 쓸 폴더")
    p_show = sub.add_parser("show", help="독자 보기나 인물 보기를 보여 준다")
    p_show.add_argument("--work", required=True)
    p_show.add_argument("--upto", type=int, required=True, help="독자가 읽은 마지막 장")
    p_show.add_argument("--character", help="인물 id. 없으면 독자 보기")
    p_show.add_argument("--at", type=int, help="인물의 시점(장). 기본은 upto")
    p_show.add_argument("--json", action="store_true", help="보기를 JSON으로 낸다")
    args = parser.parse_args(argv)

    if args.cmd == "build":
        bad = 0
        for wid in args.work or synthetic_ids():
            onto = from_synthetic(wid)
            problems = check(onto)
            bad += bool(problems)
            print(_summary(onto, problems))
            for p in problems:
                print(f"  - {p}")
            if args.out:
                out = Path(args.out)
                out.mkdir(parents=True, exist_ok=True)
                (out / f"{wid}.json").write_text(onto.model_dump_json(indent=1), encoding="utf-8")
        return 1 if bad else 0
    onto = from_synthetic(args.work)
    view = character_view(onto, args.character, args.upto, args.at) if args.character else reader_view(onto, args.upto)
    if args.json:
        print(view.model_dump_json(indent=1))
    elif args.character:
        _print_character(onto, view)
    else:
        _print_reader(onto, view)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
