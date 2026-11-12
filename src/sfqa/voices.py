"""인물 말투: 본문 대사마다 화자를 한 번 표시해 파일에 두고, 읽은 장까지의 대사만 말투 예시로 쓴다.

설정집의 말투·예시 대사에는 뒤 화 내용이 섞여 있어 쓰지 않는다(docs/ontology.md 4절). 대사는 장 번호가 있어 코드로 자를 수 있다.

- 표시는 장마다 따로 한다. 모델에게는 그 장 본문과 그 장까지 공개된 인물·별칭만 준다.
  가면을 쓰고 말한 대사는 정체를 모르는 채 표시되므로 숨은 정체가 대사 파일로 새지 않는다.
- 코드만으로 붙이면 설정집 예시 대사 기준 붙인 줄의 37%가 틀렸다(125줄 중 46줄, docs/character-chat.md).
- 결과는 data/synthetic/<작품>/lines.jsonl. 응답은 카세트(eval/cassettes/voices-<작품>-<모델>.jsonl)에 남긴다.
- 설정집의 sample_lines는 정확도를 잴 때만 쓴다. 프롬프트에는 넣지 않는다.

실행:
  uv run python -m sfqa.voices extract --work wn01-rofan --model claude-haiku-5-5 --max-calls 50 --max-usd 0.5
  uv run python -m sfqa.voices check --work wn01-rofan
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from sfqa.config import load_settings
from sfqa.gold import load_texts
from sfqa.llm import LLMRequest
from sfqa.ontology import SYNTHETIC_ROOT, Ontology, from_synthetic
from sfqa.paths import ROOT
from sfqa.works import Work, load_work

LINES_FILE = "lines.jsonl"
UNKNOWN = "unknown"
QUOTE = re.compile(r'^["“](.+?)["”]\s*$')
INNER_QUOTE = re.compile(r'["“”]')
VOICE_LINES = 8  # 프롬프트에 넣는 예시 수
VOICE_MIN, VOICE_MAX = 4, 80  # 예시 대사 길이(글자)
SPEAKER_SCHEMA = {
    "type": "object",
    "properties": {
        "speakers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"n": {"type": "integer"}, "speaker": {"type": "string"}},
                "required": ["n", "speaker"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["speakers"],
    "additionalProperties": False,
}
SPEAKER_SYSTEM = (
    "너는 소설 본문을 읽고 대사마다 말한 인물을 표시한다.\n"
    "- [본문]에서 [번호]가 붙은 줄이 대사다. 번호가 붙은 대사 전부에 답한다.\n"
    "- speaker에는 [인물] 목록의 id를 쓴다. 대사 앞뒤의 서술과 대화 흐름으로 판단한다. 대사 뒤에 나오는 인물이 말한 사람이 아니라 듣고 반응하는 사람일 수 있다.\n"
    "- 목록에 없는 사람이 말했거나, 말한 사람을 본문만으로 정할 수 없거나, 편지·방송·기억 속 말이면 unknown으로 쓴다."
)


@dataclass(frozen=True)
class VoiceLine:
    chapter: int
    line: int  # 정제 본문의 줄 번호(1부터)
    speaker: str
    text: str


def quotes_of(text: str) -> list[tuple[int, str]]:
    """(줄 번호, 대사). 한 줄 전체가 따옴표로 둘러싸인 줄만 본다."""
    return [(i, m.group(1)) for i, line in enumerate(text.splitlines(), 1) if (m := QUOTE.match(line.strip()))]


def roster(onto: Ontology, chapter: int) -> dict[str, str]:
    """id → '이름 (별칭들)'. 그 장까지 등장한 인물과 그 장까지 공개된 별칭만."""
    out = {}
    for c in onto.characters:
        if c.first_ch is not None and c.first_ch <= chapter:
            aliases = list(dict.fromkeys(a.form for a in c.aliases if a.from_ch <= chapter))
            out[c.id] = c.name + (f" ({', '.join(aliases)})" if aliases else "")
    return out


def speaker_request(onto: Ontology, chapter: int, text: str, model: str, effort: str, seed: int) -> LLMRequest:
    numbers = {line: i for i, (line, _) in enumerate(quotes_of(text), 1)}
    marked = "\n".join(f"[{numbers[i]}] {line}" if i in numbers else line for i, line in enumerate(text.splitlines(), 1))
    people = "\n".join(f"{cid}: {desc}" for cid, desc in roster(onto, chapter).items())
    return LLMRequest(model=model, system=SPEAKER_SYSTEM, user=f"[인물]\n{people}\n\n[본문]\n{marked}",
                      schema_name="speakers", schema=SPEAKER_SCHEMA, max_tokens=max(1000, 40 * len(numbers)), seed=seed,
                      reasoning_effort=effort)


def apply_speakers(chapter: int, text: str, data: dict | None, allowed: set[str]) -> list[VoiceLine]:
    """모델 응답을 줄에 붙인다. 없는 번호와 목록에 없는 id는 unknown으로 둔다."""
    named = {}
    for item in (data or {}).get("speakers", []):
        if isinstance(item, dict) and isinstance(item.get("n"), int) and isinstance(item.get("speaker"), str):
            named[item["n"]] = item["speaker"]
    return [VoiceLine(chapter, line, s if (s := named.get(i, UNKNOWN)) in allowed else UNKNOWN, quote)
            for i, (line, quote) in enumerate(quotes_of(text), 1)]


def lines_path(work: Work) -> Path:
    return work.dir / LINES_FILE


def load_lines(work: Work) -> list[VoiceLine]:
    path = lines_path(work)
    if not path.exists():
        return []
    return [VoiceLine(**json.loads(x)) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def pick_lines(lines: list[VoiceLine], speaker: str, upto: int, keep: Callable[[str], bool],
               k: int = VOICE_LINES) -> list[str]:
    """읽은 장까지 그 인물이 한 대사에서 예시를 고른다. 최근 장부터 장마다 가장 긴 줄을 돌아가며 뽑는다.

    말투는 관계에 따라 바뀌므로(온드레 양 → 세실리아) 최근 장을 먼저 본다. keep은 검사기로 거르는 함수다.
    """
    by_chapter: dict[int, list[str]] = {}
    for x in lines:
        # 안쪽에 큰따옴표가 있으면 대사 사이에 서술이 낀 줄이다('…하지." 모르데인이 말했다. "…')
        if x.speaker == speaker and x.chapter <= upto and VOICE_MIN <= len(x.text) <= VOICE_MAX and not INNER_QUOTE.search(x.text):
            by_chapter.setdefault(x.chapter, []).append(x.text)
    for texts in by_chapter.values():
        texts.sort(key=len, reverse=True)
    out: list[str] = []
    chapters = sorted(by_chapter, reverse=True)
    while chapters and len(out) < k:
        for ch in list(chapters):
            if len(out) >= k:
                break
            if not by_chapter[ch]:
                chapters.remove(ch)
                continue
            text = by_chapter[ch].pop(0)
            if text not in out and keep(text):
                out.append(text)
    return out


def extract(work_id: str, client, model: str, effort: str, seed: int, max_calls: int, max_usd: float,
            root: Path = SYNTHETIC_ROOT) -> tuple[list[VoiceLine], float]:
    """장마다 화자를 표시해 lines.jsonl에 쓴다. 상한에 닿으면 멈춘다(이미 받은 응답은 카세트에 남고 다음 실행에서 세지 않는다)."""
    from sfqa.claude import cost_usd

    work = load_work(work_id, root)
    onto = from_synthetic(work_id, root)
    texts = load_texts(work)
    out: list[VoiceLine] = []
    spent = 0.0
    for ch in sorted(texts):
        req = speaker_request(onto, ch, texts[ch], model, effort, seed)
        cached = client.has(req)
        if not cached:
            if client.calls >= max_calls:
                raise RuntimeError(f"호출 상한({max_calls})에 닿았다. {ch}장부터 남았다")
            if spent >= max_usd:
                raise RuntimeError(f"비용 상한({max_usd}달러)에 닿았다. {ch}장부터 남았다")
        resp = client.complete(req)
        if not cached:
            spent += cost_usd(resp)
        data, bad = resp.parsed()
        if bad or resp.truncated:
            print(f"  {ch}장 응답을 읽지 못함(finish={resp.finish_reason}). 이 장은 unknown으로 둔다")
        out += apply_speakers(ch, texts[ch], data, {*roster(onto, ch), UNKNOWN})
        print(f"  {ch}장 대사 {len(quotes_of(texts[ch]))}줄, 누적 {spent:.4f}달러", flush=True)
    lines_path(work).write_text("".join(json.dumps(asdict(x), ensure_ascii=False) + "\n" for x in out), encoding="utf-8")
    return out, spent


def accuracy(work: Work, lines: list[VoiceLine]) -> dict:
    """설정집 예시 대사가 본문에 그대로 있으면 그 줄의 정답으로 본다. 정답지가 아니라 어림이다."""
    bible = json.loads((work.dir / "bible.json").read_text(encoding="utf-8"))
    truth = {s.strip('"“”'): c["id"] for c in bible["characters"] for s in c.get("sample_lines", [])}
    right = wrong = unknown = 0
    for x in lines:
        if x.text in truth:
            if x.speaker == UNKNOWN:
                unknown += 1
            elif x.speaker == truth[x.text]:
                right += 1
            else:
                wrong += 1
    named = right + wrong
    return {"right": right, "wrong": wrong, "unknown": unknown, "precision": right / named if named else None,
            "lines": len(lines), "named": sum(x.speaker != UNKNOWN for x in lines)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sfqa.voices", description="본문 대사 화자 표시와 정확도 어림")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_ex = sub.add_parser("extract", help="장마다 화자를 표시한다(Claude API, 유료)")
    p_ex.add_argument("--work", required=True)
    p_ex.add_argument("--model", default="claude-haiku-5-5")
    p_ex.add_argument("--max-calls", type=int, required=True, help="실제 호출 상한(카세트에 있으면 세지 않는다)")
    p_ex.add_argument("--max-usd", type=float, required=True, help="비용 상한(달러)")
    p_check = sub.add_parser("check", help="저장된 표시의 정확도를 어림한다(모델 없음)")
    p_check.add_argument("--work", required=True)
    args = parser.parse_args(argv)

    work = load_work(args.work, SYNTHETIC_ROOT)
    if args.cmd == "extract":
        from sfqa.claude import AnthropicClient, effort_label

        cassette = ROOT / "eval" / "cassettes" / f"voices-{args.work}-{args.model}.jsonl"
        client = AnthropicClient(cassette)
        lines, spent = extract(args.work, client, args.model, effort_label(args.model), load_settings({}).seed,
                               args.max_calls, args.max_usd)
        print(f"{args.work}: 대사 {len(lines)}줄, 실제 호출 {client.calls}회, 이번 실행 비용 {spent:.4f}달러")
    else:
        lines = load_lines(work)
    print(json.dumps(accuracy(work, lines), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
