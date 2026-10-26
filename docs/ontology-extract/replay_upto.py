"""추출 실행이 남긴 장별 출력(report.json)을 N장까지만 다시 적용해, 그 시점의 온톨로지를 만든다. 모델을 부르지 않는다.

뒤 장의 learned가 앞 사실의 아는 인물을 늘리므로, 50장 실행의 온톨로지를 그대로 N장에서 자르면 N장 시점과 다르다.
같은 범위를 다른 규칙으로 돌린 실행과 비교할 때 기준선으로 쓴다(docs/ontology-extract.md 14절).

실행(worktree 루트): PYTHONPATH=src uv run --no-sync python docs/ontology-extract/replay_upto.py <실행 폴더> <N> <새 폴더>
"""

import json
import sys
from pathlib import Path

from sfqa.gold import load_texts
from sfqa.onto_extract import ChapterOut, State
from sfqa.ontology import SYNTHETIC_ROOT, check
from sfqa.works import load_work


def main(run: str, upto: str, out: str) -> None:
    src, n, dst = Path(run), int(upto), Path(out)
    report = json.loads((src / "report.json").read_text(encoding="utf-8"))
    work = load_work(report["work"], SYNTHETIC_ROOT)
    texts = load_texts(work)
    state = State(work=work.id, title=work.title, chapters=work.chapters)
    chapters = [c for c in report["chapters"] if "output" in c and c["chapter"] <= n]
    for c in chapters:
        state.apply(c["chapter"], ChapterOut.model_validate(c["output"]), texts[c["chapter"]])
    onto = state.ontology(source=f"{report['model']} 출력 재적용({src.name}, {n}장까지)")
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "ontology.json").write_text(onto.model_dump_json(indent=1), encoding="utf-8")
    (dst / "record.txt").write_text("\n\n".join(state.blocks), encoding="utf-8")
    (dst / "report.json").write_text(json.dumps(
        {"work": work.id, "model": report["model"], "upto": n, "done": len(chapters), "stopped": None, "calls": 0,
         "spent_usd": 0, "replayed_from": str(src), "chapters": [], "problems": check(onto)}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"{dst}: {len(chapters)}장, 사실 {len(onto.facts)}, 인물 {len(onto.characters)}, 문제 {len(check(onto))}")


if __name__ == "__main__":
    main(*sys.argv[1:])
