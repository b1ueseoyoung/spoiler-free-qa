"""작품 설정과 경로(data/works/<id>/work.json). 책갈피는 여러 작품을 다룬다.

- 장 수는 work.json의 chapters가 기준이다. 장 번호는 1부터 chapters까지다.
- 정답지와 평가(eval/)는 「탁류」 전용이다(GOLD_WORK). 다른 작품은 평가하지 않는다(evaluated=false).
- label은 프롬프트에 들어가는 작품 이름이다. 「탁류」는 기존 문자열과 한 글자도 다르지 않아야 한다
  (카세트 키가 프롬프트 전문으로 만들어진다).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from sfqa.paths import DATA

WORKS_ROOT = DATA / "works"
GOLD_WORK = "takryu"


@dataclass(frozen=True)
class Work:
    id: str
    title: str
    author: str
    label: str
    chapters: int
    dir: Path
    order: int = 99
    author_death_year: int | None = None
    published: str = ""
    blurb: str = ""
    examples: tuple[tuple[str, int], ...] = ()  # (질문, 답이 있는 장)
    default_upto: int = 1
    evaluated: bool = False
    source: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def last(self) -> int:
        return self.chapters

    @property
    def chapter_numbers(self) -> range:
        return range(1, self.chapters + 1)

    @property
    def raw_dir(self) -> Path:
        return self.dir / "raw"

    @property
    def clean_dir(self) -> Path:
        return self.dir / "clean"

    @property
    def sources_path(self) -> Path:
        return self.dir / "sources.json"

    @property
    def clean_report_path(self) -> Path:
        return self.clean_dir / "clean-report.json"

    @property
    def entities_path(self) -> Path:
        return self.dir / "entities.json"

    @property
    def index_root(self) -> Path:
        return self.dir / "index"

    def raw_path(self, chapter: int) -> Path:
        return self.raw_dir / f"ch{chapter:02d}.wiki"

    def clean_path(self, chapter: int) -> Path:
        return self.clean_dir / f"ch{chapter:02d}.txt"

    @property
    def ready(self) -> bool:
        """정제 본문이 있어 서비스할 수 있는가."""
        return self.clean_report_path.exists()


def load_work(work_id: str, root: Path = WORKS_ROOT) -> Work:
    d = root / work_id
    data = json.loads((d / "work.json").read_text(encoding="utf-8"))
    if data["id"] != work_id:
        raise ValueError(f"work.json의 id({data['id']})가 폴더 이름({work_id})과 다르다")
    return Work(
        id=data["id"], title=data["title"], author=data["author"], label=data["label"],
        chapters=int(data["chapters"]), dir=d, order=int(data.get("order", 99)),
        author_death_year=data.get("author_death_year"), published=data.get("published", ""),
        blurb=data.get("blurb", ""),
        examples=tuple((e, 1) if isinstance(e, str) else (e["question"], int(e["chapter"])) for e in data.get("examples", [])),
        default_upto=int(data.get("default_upto", 1)), evaluated=bool(data.get("evaluated", False)),
        source=data.get("source", {}),
    )


def list_works(root: Path = WORKS_ROOT, ready_only: bool = False) -> list[Work]:
    works = [load_work(p.parent.name, root) for p in sorted(root.glob("*/work.json"))]
    works = [w for w in works if w.ready] if ready_only else works
    return sorted(works, key=lambda w: (w.order, w.id))


def gold_work() -> Work:
    return load_work(GOLD_WORK)
