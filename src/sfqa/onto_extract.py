"""본문 추출기: 장을 순서대로 한 장씩 읽혀 장 시간축 온톨로지를 쌓는다(docs/ontology-extract.md).

- 호출마다 고정 지시문 + 지금까지의 기록 + 이번 장 본문을 보내고, 이번 장에서 새로 생긴 것만 JSON 스키마로 받는다.
- 기록은 장마다 블록 하나를 뒤에 덧붙인다. 앞 블록이 그대로라 다음 호출이 앞 호출이 쓴 캐시를 읽는다.
  캐시 지점은 지시문 끝과 마지막 기록 블록 끝 두 곳이다.
- 추출기도 c장까지만 본다. 사실과 learned마다 그 장 본문 인용을 받아 글자 그대로 있는지 대조한다(verify.compact).
  대조에 실패하면 기록과 온톨로지에 넣지 않고 결과 파일에만 남긴다. 유명한 작품에서 모델 기억이 섞이는 것을 막는다.
- 정체가 밝혀진 임시 인물(가면 쓴 사내 등)은 원래 인물에 합친다. 임시 인물의 이름은 그 장부터 원래 인물의 별칭이 되고,
  임시 인물의 앎은 원래 인물에게 옮기되 아는 장을 정체가 드러난 장보다 앞당기지 않는다(정체가 새지 않게).
- 응답은 카세트(JSONL)에 남긴다. 같은 요청은 다시 부르지 않는다. 호출 수와 비용에 상한을 둔다.
- 서버 쪽 폴백은 쓰지 않는다(평가에서 모델이 섞이지 않게). 거절·잘림·형식 오류는 한 번 다시 부르고, 또 실패하면
  멈춘다. 장을 건너뛰면 기록이 끊긴다.

실행: uv run python -m sfqa.onto_extract run --work wn01-rofan --model claude-opus-5-5 --upto 5 --max-usd 0.8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

from sfqa.llm import LLMResponse, _load_cassette
from sfqa.ontology import (
    SYNTHETIC_ROOT,
    Alias,
    Character,
    Evidence,
    Fact,
    Knowing,
    Ontology,
    Scene,
    check,
)
from sfqa.paths import ROOT
from sfqa.verify import MIN_QUOTE_CHARS, compact

KEY_VERSION = 1
MAX_TOKENS = 16000
LONG_MAX_TOKENS = 48000  # 출력이 잘린 장만 이 상한으로 다시 묻는다(10/26 wn02 46장, 인물 일곱이 모여 learned가 수백 건)
RUNS = ROOT / "runs" / "onto-extract"
# 모델 → (thinking, effort)
SETTINGS = {
    "claude-opus-5-5": ({"type": "adaptive"}, "medium"),  # thinking을 끌 수 없다. effort 기본값도 medium
    "claude-sonnet-5-5": ({"type": "between_tools"}, "high"),  # thinking 끔. effort high 이하에서만 받는다
    "claude-haiku-5-5": ({"type": "disabled"}, "medium"),
}
# $/백만 토큰: (입력, 출력, 캐시 읽기, 캐시 쓰기 5분). claude-api 스킬 2026-10-06 캐시 기준.
# 캐시 쓰기는 입력의 1.25배. Haiku 5.5 캐시 읽기는 표에 없어 일반 규칙(입력의 0.1배)으로 둔다.
PRICES = {"claude-opus-5-5": (4.0, 20.0, 0.20, 5.0), "claude-sonnet-5-5": (2.0, 10.0, 0.20, 2.5),
          "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125)}
STOP_TO_FINISH = {"end_turn": "stop", "max_tokens": "length", "refusal": "refusal"}
CATEGORIES = ["identity", "event", "relationship", "secret", "motive", "ability", "location", "death", "world"]

SYSTEM = """너는 한국어 장편소설을 한 장씩 읽으며 '장 시간축 온톨로지'를 쌓는 추출기다.
사용자 메시지의 앞부분은 지금까지 쌓은 기록(장마다 한 블록)이고, 마지막은 이번 장 본문이다. 이번 장에서 새로 생긴 것만 낸다.

목적: 독자가 N장까지 읽었을 때 공개된 것만, 그리고 인물마다 그 시점까지 알게 된 것만 꺼내 쓰려는 것이다. 그래서 '언제 처음 드러났는가'와 '누가 언제부터 아는가'가 가장 중요하다.

규칙
1. 이번 장 본문과 기록만 쓴다. 이 작품을 따로 알고 있더라도 그 기억은 쓰지 않는다.
2. 새 사실(facts)은 이번 장에서 독자가 처음 알게 되는 것만 적는다.
   - 본문에 분명히 쓰였거나 장면으로 분명히 보인 것만 적는다. 암시, 복선, 인물의 짐작은 사실이 아니다.
   - 기록에 이미 있는 사실을 다시 말하는 장면은 새 사실이 아니다. 그 장면에서 누가 새로 알게 되면 learned에 적는다.
   - 사실 하나에는 내용 하나만 담는다. 인물마다 아는 범위가 다를 수 있는 내용은 나눈다.
   - 줄거리 전개뿐 아니라 인물의 신분, 나이, 관계, 능력, 비밀, 장소와 제도도 사실이다.
   - statement는 인물 이름을 넣은 완결된 한 문장으로 쓴다.
   - quote는 이번 장 본문에서 그 사실을 보여 주는 부분을 한 문단 안에서 글자 그대로 옮긴다(150자 이내). 코드로 대조해서 다르면 버린다.
3. knowers에는 이번 장 끝에서 그 사실을 아는 인물을 빠짐없이 넣는다.
   - 그 일을 직접 겪었거나, 그 장면에 있었거나, 들은 인물은 since를 이번 장으로 한다.
   - 본문이 '전부터 알고 있었다'고 밝히거나, 자기 신분이나 과거처럼 처음부터 알 수밖에 없는 것이면 그 장이나 0(이야기 시작 전)으로 한다.
   - 짐작은 앎이 아니다. 단, 서술이 그 인물의 결론을 단정하고 그 인물이 그에 따라 움직이면 안다고 본다.
4. common은 이야기 속 세상 사람이라면 누구나 아는 상식일 때 true다(나라의 제도와 법, 지리, 관습, 공개된 행사).
   - common이면 아직 나오지 않은 인물까지 모두 처음부터 안다고 보므로 knowers는 비워도 된다.
   - 특정 인물에 관한 사실(이름, 나이, 신분, 지위, 관계, 과거, 능력, 비밀)은 common으로 하지 않는다. 그 인물을 직접 알거나 듣게 된 인물만 knowers나 learned로 넣는다.
   - 예외로, 세상에 널리 퍼진 그 인물의 평판이나 별명(모두가 그렇게 부르는 별명)은 common으로 할 수 있다.
   - 한 사실에 상식인 부분과 아닌 부분이 섞이면 두 사실로 나눈다.
   - 비밀, 소문의 진상, 일부 사람만 아는 일은 common이 아니다.
5. learned는 기록에 있는 이전 사실을 어떤 인물이 새로 알게 된 것이다. 이번 장을 읽고 다음 경우를 빠짐없이 찾는다.
   - 처음 만난 인물끼리 서로의 이름, 신분, 직업처럼 만나면 알게 되는 것을 알게 된다(예: 새로 온 하인이 주인을 처음 만나면 주인의 이름과 지위를 알게 된다).
   - 누군가 이전 사실을 말해 주거나, 편지·소문·기록으로 전해 듣거나, 엿듣거나, 목격한다. 본문이 장면 밖에서 알게 됐다고 밝힌 경우도 넣는다.
   - 이번 장에 처음 나온 인물이 기록의 사실을 원래부터 알고 있었다는 것이 본문에 드러난다(예: 처음 나온 늙은 집사가 주인 집안의 옛일을 이미 안다).
   - since는 알게 된 장이다. 이번 장에 알게 됐으면 이번 장, 원래 알고 있었으면 그보다 이른 장이나 0이다.
   - fact에는 기록의 사실 키(예: c3-2)를 그대로 쓰고, quote에는 그 앎이 드러나는 부분을 글자 그대로 옮긴다.
6. characters는 이번 장에 처음 등장하거나 처음 이름이 나오는 인물이다.
   - id는 로마자 소문자와 밑줄로 짓는다. 기록에 있는 인물은 기록의 id를 그대로 쓴다.
   - appeared는 이번 장에 직접 등장했으면 true, 이름만 언급되면 false다.
   - 정체를 모르는 인물(예: 가면 쓴 사내)은 그 호칭으로 임시 인물을 만든다.
7. aliases는 기존 인물에게 이번 장부터 붙는 새 별칭이다.
8. merges는 이번 장에서 정체가 밝혀져 임시 인물이 기존 인물과 같은 사람으로 드러난 경우다. temp는 임시 인물 id, into는 원래 인물 id다.
9. scene은 이번 장의 시점 인물(pov), 주된 장소, 직접 등장한 인물 id다. 이번 장에 처음 나온 인물도 id로 쓴다.
10. 이번 장에 새로 생긴 것이 없으면 빈 배열로 낸다."""


def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


_STR, _INT, _BOOL = {"type": "string"}, {"type": "integer"}, {"type": "boolean"}
SCHEMA = _obj({
    "characters": {"type": "array", "items": _obj({"id": _STR, "name": _STR, "appeared": _BOOL,
                                                     "aliases": {"type": "array", "items": _STR}})},
    "aliases": {"type": "array", "items": _obj({"character": _STR, "alias": _STR})},
    "merges": {"type": "array", "items": _obj({"temp": _STR, "into": _STR})},
    "facts": {"type": "array", "items": _obj({
        "statement": _STR, "category": {"type": "string", "enum": CATEGORIES}, "quote": _STR, "common": _BOOL,
        "knowers": {"type": "array", "items": _obj({"character": _STR, "since": _INT})}})},
    "learned": {"type": "array", "items": _obj({"fact": _STR, "character": _STR, "since": _INT, "quote": _STR})},
    "scene": _obj({"pov": {"anyOf": [_STR, {"type": "null"}]}, "location": _STR, "present": {"type": "array", "items": _STR}}),
})


class NewCharacter(BaseModel):
    id: str
    name: str
    appeared: bool
    aliases: list[str]


class NewAlias(BaseModel):
    character: str
    alias: str


class Merge(BaseModel):
    temp: str
    into: str


class Knower(BaseModel):
    character: str
    since: int


class FactOut(BaseModel):
    statement: str
    category: str
    quote: str
    common: bool
    knowers: list[Knower]


class Learned(BaseModel):
    fact: str
    character: str
    since: int
    quote: str


class SceneOut(BaseModel):
    pov: str | None
    location: str
    present: list[str]


class ChapterOut(BaseModel):
    characters: list[NewCharacter]
    aliases: list[NewAlias]
    merges: list[Merge]
    facts: list[FactOut]
    learned: list[Learned]
    scene: SceneOut


# ---------- 인용 대조 ----------


def locate(quote: str, text: str) -> list[int] | None:
    """인용이 본문에 글자 그대로 있으면 그 줄 번호(1부터), 없거나 너무 짧으면 None. 공백·따옴표 차이는 무시한다."""
    q = compact(quote)
    if len(q) < MIN_QUOTE_CHARS or q not in compact(text):
        return None
    lines = [compact(line) for line in text.splitlines()]
    for size in range(1, 6):  # 여러 줄에 걸친 인용은 가장 짧은 연속 구간
        for i in range(len(lines) - size + 1):
            if q in "".join(lines[i:i + size]):
                return list(range(i + 1, i + size + 1))
    return []


# ---------- 기록(상태) ----------


@dataclass
class State:
    work: str
    title: str
    chapters: int
    characters: dict[str, dict] = field(default_factory=dict)  # id → name, first_ch, mention_ch, aliases, merged
    facts: dict[str, dict] = field(default_factory=dict)  # 키(c3-2) → statement, category, reveal_ch, quote, lines, knowers
    scenes: list[dict] = field(default_factory=list)
    blocks: list[str] = field(default_factory=list)  # 장마다 기록 블록. 덧붙이기만 한다
    rejected: list[dict] = field(default_factory=list)  # 인용 대조에 실패한 사실·learned
    notes: list[str] = field(default_factory=list)  # 형식 문제(없는 인물, 없는 사실 키, 뒤로 간 장 등)

    def _alias(self, cid: str, form: str, chapter: int) -> None:
        entry = self.characters[cid]
        if form and form != entry["name"] and form not in [a for a, _ in entry["aliases"]]:
            entry["aliases"].append([form, chapter])

    def apply(self, c: int, out: ChapterOut, text: str) -> dict:
        """c장 출력을 기록에 반영하고, 기록 블록을 덧붙인다. 이 장의 집계를 돌려준다."""
        notes: list[str] = []
        new_chars, new_aliases, merges = [], [], []
        for ch in out.characters:
            cid = ch.id.strip().lower()
            if cid in self.characters:
                notes.append(f"이미 있는 인물을 새 인물로 냄: {cid}")
                if ch.appeared and self.characters[cid]["first_ch"] is None:
                    self.characters[cid]["first_ch"] = c
            else:
                self.characters[cid] = {"name": ch.name, "first_ch": c if ch.appeared else None, "mention_ch": c,
                                        "aliases": [], "merged": None}
                new_chars.append(cid)
            for form in ch.aliases:
                self._alias(cid, form, c)
        for a in out.aliases:
            if a.character not in self.characters:
                notes.append(f"별칭의 인물이 없음: {a.character}")
                continue
            self._alias(a.character, a.alias, c)
            new_aliases.append((a.character, a.alias))
        for m in out.merges:
            if m.temp not in self.characters or m.into not in self.characters or m.temp == m.into:
                notes.append(f"합칠 수 없음: {m.temp} → {m.into}")
                continue
            temp = self.characters[m.temp]
            temp["merged"] = [m.into, c]
            for form in [temp["name"], *(a for a, _ in temp["aliases"])]:
                self._alias(m.into, form, c)
            merges.append((m.temp, m.into))
        present = []
        for cid in out.scene.present:
            if cid not in self.characters:
                notes.append(f"장면의 인물이 없음: {cid}")
                continue
            present.append(cid)
            if self.characters[cid]["first_ch"] is None:  # 장면에 직접 나왔다
                self.characters[cid]["first_ch"] = c
        pov = out.scene.pov if out.scene.pov in self.characters else None
        if out.scene.pov and pov is None:
            notes.append(f"시점 인물이 없음: {out.scene.pov}")
        self.scenes.append({"chapter": c, "pov": pov, "location": out.scene.location, "present": present})

        accepted = []
        for f in out.facts:
            lines = locate(f.quote, text)
            if lines is None:
                self.rejected.append({"chapter": c, "kind": "fact", "statement": f.statement, "quote": f.quote})
                continue
            knowers: dict[str, int] = {}
            for k in f.knowers:
                if k.character not in self.characters:
                    notes.append(f"아는 인물이 없음: {k.character}")
                    continue
                since = min(max(k.since, 0), c)
                if k.since > c:
                    notes.append(f"아는 장이 이번 장보다 뒤: {k.character} {k.since}")
                knowers[k.character] = min(knowers.get(k.character, since), since)
            key = f"c{c}-{len(accepted) + 1}"
            self.facts[key] = {"statement": f.statement, "category": f.category, "reveal_ch": c, "quote": f.quote,
                               "lines": lines, "knowers": knowers, "common": f.common}
            accepted.append(key)
        learned = []
        for item in out.learned:
            fact = self.facts.get(item.fact)
            if fact is None or item.character not in self.characters:
                notes.append(f"learned를 반영할 수 없음: {item.fact} {item.character}")
                continue
            if locate(item.quote, text) is None:
                self.rejected.append({"chapter": c, "kind": "learned", "fact": item.fact, "character": item.character,
                                      "quote": item.quote})
                continue
            if item.since > c:
                notes.append(f"learned의 아는 장이 이번 장보다 뒤: {item.fact} {item.character} {item.since}")
            since = min(max(item.since, 0), c)  # 원래 알던 것이면 이번 장보다 이를 수 있다
            if fact["knowers"].get(item.character, c + 1) > since:
                fact["knowers"][item.character] = since
                learned.append((item.fact, item.character, since))

        block = [f"[{c}장]"]
        for cid in new_chars:
            e = self.characters[cid]
            aliases = [a for a, _ in e["aliases"]]
            block.append(f"새 인물: {cid} {e['name']}" + (f" (별칭: {', '.join(aliases)})" if aliases else "")
                         + ("" if e["first_ch"] == c else " [언급만]"))
        block += [f"새 별칭: {cid} = {form}" for cid, form in new_aliases]
        block += [f"합침: {temp} → {into}" for temp, into in merges]
        for key in accepted:
            f = self.facts[key]
            known = "상식" if f["common"] else "앎: " + " ".join(f"{k}({s})" for k, s in f["knowers"].items())
            block.append(f"새 사실: {key} {f['statement']} | {known}")
        block += [f"새로 앎: {key} <- {cid}({since})" for key, cid, since in learned]
        block.append(f"장면: 시점 {pov or '-'}, {out.scene.location}, 등장 {' '.join(present) or '-'}")
        self.blocks.append("\n".join(block))
        self.notes += [f"{c}장: {n}" for n in notes]
        return {"facts": len(accepted), "rejected": sum(1 for r in self.rejected if r["chapter"] == c),
                "learned": len(learned), "new_characters": len(new_chars), "notes": notes}

    def ontology(self, source: str) -> Ontology:
        """합쳐진 임시 인물은 빼고, 그 앎은 원래 인물에게 정체 공개 장 이후로 옮긴다."""
        merged = {cid: e["merged"] for cid, e in self.characters.items() if e["merged"]}

        def resolve(cid: str, since: int) -> tuple[str, int]:
            while cid in merged:
                cid, at = merged[cid]
                since = max(since, at)
            return cid, since

        chars = [Character(id=cid, name=e["name"], first_ch=e["first_ch"], mention_ch=e["mention_ch"],
                           aliases=[Alias(form=a, from_ch=ch) for a, ch in e["aliases"]])
                 for cid, e in self.characters.items() if cid not in merged]
        facts = []
        for key, f in self.facts.items():
            known: dict[str, int] = {}
            for cid, since in f["knowers"].items():
                who, at = resolve(cid, since)
                known[who] = min(known.get(who, at), at)
            facts.append(Fact(id=key, statement=f["statement"], category=f["category"], reveal_ch=f["reveal_ch"],
                              known_by=[Knowing(character=k, from_ch=s) for k, s in known.items()],
                              evidence=[Evidence(chapter=f["reveal_ch"], lines=f["lines"])] if f["lines"] else [],
                              common=f.get("common", False)))
        scenes = [Scene(chapter=s["chapter"], pov=resolve(s["pov"], 0)[0] if s["pov"] else "", location=s["location"],
                        present=list(dict.fromkeys(resolve(p, 0)[0] for p in s["present"]))) for s in self.scenes]
        return Ontology(work=self.work, title=self.title, chapters=self.chapters, source=source,
                        characters=chars, facts=facts, scenes=scenes)


# ---------- 요청과 호출 ----------


def build_request(model: str, blocks: list[str], chapter: int, text: str) -> dict:
    thinking, effort = SETTINGS[model]
    content = [{"type": "text", "text": b} for b in blocks]
    if content:
        content[-1]["cache_control"] = {"type": "ephemeral"}
    lead = "앞의 블록들이 지금까지의 기록이다." if blocks else "아직 기록이 없다. 이번이 첫 장이다."
    content.append({"type": "text", "text": f"{lead}\n\n<본문 장=\"{chapter}\">\n{text}\n</본문>\n\n"
                                            f"이번 장은 {chapter}장이다. 이번 장에서 새로 생긴 것만 스키마에 맞게 내라."})
    return {"model": model, "max_tokens": MAX_TOKENS,
            "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": content}], "thinking": thinking,
            "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}}}


def request_key(req: dict, attempt: int) -> str:
    body = {"req": req, "attempt": attempt, "v": KEY_VERSION}
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def cost_usd(resp: LLMResponse) -> float:
    i, o, cr, cw = PRICES.get(resp.model, (0.0, 0.0, 0.0, 0.0))
    x = resp.extra
    return (resp.input_tokens * i + x.get("cache_write", 0) * cw + x.get("cache_read", 0) * cr
            + resp.output_tokens * o) / 1_000_000


def first_call_guess(req: dict) -> float:
    """실행의 첫 호출 어림. 캐시가 비었다고 보고 입력 전체를 캐시 쓰기 값으로, 출력은 상한까지 센다.

    전날 실행을 이어 가면 캐시(5분)가 비어 쌓인 기록 전체를 다시 쓴다. 10/24 48장은 기록 8.7만 토큰을 다시 써서
    0.605달러가 들었다(출력만 센 어림은 0.32달러였다). 글자당 1.1토큰, 출력 스키마 1,200토큰(10/23 실측 1,170)으로 센다.
    """
    _, o, _, cw = PRICES[req["model"]]
    chars = sum(len(b["text"]) for b in req["system"]) + sum(len(b["text"]) for m in req["messages"] for b in m["content"])
    return ((chars * 1.1 + 1200) * cw + req["max_tokens"] * o) / 1_000_000


def parse(resp: LLMResponse) -> tuple[ChapterOut | None, str | None]:
    if resp.finish_reason != "stop" or not resp.raw_text:
        return None, f"finish_reason={resp.finish_reason}"
    try:
        return ChapterOut.model_validate_json(resp.raw_text), None
    except ValidationError as e:
        return None, f"형식 오류: {e.errors()[0]['msg']}"


class Budget(Exception):
    pass


class Extractor:
    """카세트에 있으면 API를 부르지 않는다. 새로 부른 호출만 calls와 spent에 센다."""

    def __init__(self, cassette: Path | None, sdk=None, max_calls: int = 60, max_usd: float = 1.0):
        self._sdk = sdk
        self.cassette = cassette
        self._cache = _load_cassette(cassette)
        self.calls, self.spent, self.max_calls, self.max_usd = 0, 0.0, max_calls, max_usd
        self._max_cost = 0.0  # 지금까지 가장 비쌌던 호출

    def _client(self):
        if self._sdk is None:
            from sfqa.claude import KEY_ENV, load_key

            if not load_key():
                raise RuntimeError(f"{KEY_ENV}가 없다. 프로젝트 .env에 넣거나 환경 변수로 준다")
            import anthropic

            self._sdk = anthropic.Anthropic(max_retries=3)
        return self._sdk

    def complete(self, req: dict, attempt: int = 0) -> tuple[LLMResponse, bool]:
        """(응답, 새로 불렀는가)."""
        key = request_key(req, attempt)
        if key in self._cache:
            return self._cache[key], False
        # 다음 호출 비용은 지금까지 가장 비쌌던 호출의 1.2배로 어림한다. 첫 호출은 캐시가 빈 경우로 본다(first_call_guess).
        guess = self._max_cost * 1.2 if self._max_cost else first_call_guess(req)
        if self.calls >= self.max_calls:
            raise Budget(f"호출 상한 {self.max_calls}회")
        if self.spent + guess > self.max_usd:
            raise Budget(f"비용 상한 ${self.max_usd:.2f}(쓴 돈 ${self.spent:.2f}, 다음 호출 어림 ${guess:.2f})")
        started = time.monotonic()
        with self._client().messages.stream(**req) as stream:
            msg = stream.get_final_message()
        usage = msg.usage
        text = next((b.text for b in msg.content if b.type == "text"), None)
        resp = LLMResponse(
            # 거절이면 출력이 스키마를 따르지 않을 수 있어 본문을 쓰지 않는다.
            raw_text=None if msg.stop_reason == "refusal" else text,
            finish_reason=STOP_TO_FINISH.get(msg.stop_reason, msg.stop_reason),
            input_tokens=usage.input_tokens, output_tokens=usage.output_tokens, model=req["model"],
            latency_s=round(time.monotonic() - started, 3),
            extra={"stop_reason": msg.stop_reason, "cache_read": getattr(usage, "cache_read_input_tokens", 0) or 0,
                   "cache_write": getattr(usage, "cache_creation_input_tokens", 0) or 0,
                   "request_id": getattr(msg, "_request_id", None)})
        # 해석보다 기록을 먼저 한다. 잘리거나 거절된 응답도 비용이 들었다.
        self.calls += 1
        cost = cost_usd(resp)
        self._max_cost = max(self._max_cost, cost)
        self.spent += cost
        if self.cassette:
            self.cassette.parent.mkdir(parents=True, exist_ok=True)
            with self.cassette.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "response": asdict(resp)}, ensure_ascii=False) + "\n")
        self._cache[key] = resp
        return resp, True


def run(work_id: str, model: str, upto: int, out_dir: Path, *, root: Path = SYNTHETIC_ROOT,
        cassette: Path | None = RUNS / "cassette.jsonl", max_usd: float = 1.0, max_calls: int = 60, sdk=None) -> dict:
    """1장부터 upto장까지 차례로 추출한다. 상한에 걸리거나 두 번 실패하면 멈추고, 그때까지의 결과를 남긴다."""
    from sfqa.gold import load_texts
    from sfqa.works import load_work

    if model not in SETTINGS:
        raise ValueError(f"모르는 모델: {model}")
    work = load_work(work_id, root)
    texts = load_texts(work)
    state = State(work=work.id, title=work.title, chapters=work.chapters)
    ex = Extractor(cassette, sdk=sdk, max_calls=max_calls, max_usd=max_usd)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"work": work.id, "model": model, "settings": SETTINGS[model], "upto": upto, "done": 0, "stopped": None,
              "chapters": []}
    for c in range(1, min(upto, work.chapters) + 1):
        req = build_request(model, state.blocks, c, texts[c])
        tries = []
        out = None
        try:
            for attempt in (0, 1):
                # 출력이 잘렸으면 상한을 늘려 다시 묻는다. 첫 요청은 그대로라 이미 돌린 장의 카세트 키가 바뀌지 않는다.
                cut = bool(tries) and tries[-1]["error"] == "finish_reason=length"
                resp, fresh = ex.complete({**req, "max_tokens": LONG_MAX_TOKENS} if cut else req, attempt)
                out, err = parse(resp)
                tries.append({"attempt": attempt, "fresh": fresh, "error": err, "cost": round(cost_usd(resp), 5),
                              "input": resp.input_tokens, "cache_read": resp.extra.get("cache_read", 0),
                              "cache_write": resp.extra.get("cache_write", 0), "output": resp.output_tokens,
                              "latency_s": resp.latency_s, "raw": resp.raw_text if err else None})
                if out is not None:
                    break
        except Budget as e:
            report["stopped"] = str(e)
            report["chapters"].append({"chapter": c, "tries": tries})
            break
        if out is None:
            report["stopped"] = f"{c}장 두 번 실패"
            report["chapters"].append({"chapter": c, "tries": tries})
            break
        summary = state.apply(c, out, texts[c])
        report["chapters"].append({"chapter": c, "tries": tries, **summary, "output": out.model_dump()})
        report["done"] = c
    onto = state.ontology(source=f"extract:{model}")
    report["problems"] = check(onto)
    report["calls"], report["spent_usd"] = ex.calls, round(ex.spent, 4)
    report["rejected"], report["notes"] = state.rejected, state.notes
    (out_dir / "ontology.json").write_text(onto.model_dump_json(indent=1), encoding="utf-8")
    (out_dir / "record.txt").write_text("\n\n".join(state.blocks), encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def _print(report: dict) -> None:
    print(f"{report['work']} {report['model']}: {report['done']}/{report['upto']}장, 새 호출 {report['calls']}회, "
          f"${report['spent_usd']:.4f}" + (f", 멈춤: {report['stopped']}" if report["stopped"] else ""))
    for ch in report["chapters"]:
        t = ch["tries"][-1] if ch["tries"] else {}
        print(f"  {ch['chapter']}장: 사실 {ch.get('facts', '-')}(버림 {ch.get('rejected', '-')}), learned {ch.get('learned', '-')}, "
              f"새 인물 {ch.get('new_characters', '-')} | 입력 {t.get('input', 0)} 캐시읽기 {t.get('cache_read', 0)} "
              f"캐시쓰기 {t.get('cache_write', 0)} 출력 {t.get('output', 0)} | ${t.get('cost', 0):.4f} {t.get('latency_s', 0):.0f}초"
              + (f" | 시도 {len(ch['tries'])}회" if len(ch["tries"]) > 1 else ""))
    if report["problems"]:
        print("  온톨로지 문제:", *report["problems"][:10], sep="\n    ")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sfqa.onto_extract", description="본문에서 장 시간축 온톨로지를 뽑는다")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="1장부터 차례로 추출한다(Claude API 유료 호출)")
    p.add_argument("--work", required=True)
    p.add_argument("--model", required=True, choices=list(SETTINGS))
    p.add_argument("--upto", type=int, required=True)
    p.add_argument("--max-usd", type=float, required=True, help="이 실행에서 새로 쓸 수 있는 최대 금액")
    p.add_argument("--max-calls", type=int, default=60)
    p.add_argument("--classic", action="store_true", help="data/works의 작품(기본은 data/synthetic)")
    p.add_argument("--out", help="결과 폴더(기본 runs/onto-extract/<작품>-<모델>-<시각>)")
    args = parser.parse_args(argv)

    from sfqa.works import WORKS_ROOT

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else RUNS / f"{args.work}-{args.model.removeprefix('claude-')}-{stamp}"
    report = run(args.work, args.model, args.upto, out, root=WORKS_ROOT if args.classic else SYNTHETIC_ROOT,
                 max_usd=args.max_usd, max_calls=args.max_calls)
    _print(report)
    print(f"결과: {out}")
    return 1 if report["stopped"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
