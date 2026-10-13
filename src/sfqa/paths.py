"""프로젝트 안의 경로."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
CLEAN = DATA / "clean"
SOURCES = RAW / "sources.json"
CLEAN_REPORT = CLEAN / "clean-report.json"
CHAPTERS = range(1, 20)  # 「탁류」 제1장~제19장


def raw_path(chapter: int) -> Path:
    return RAW / f"ch{chapter:02d}.wiki"


def clean_path(chapter: int) -> Path:
    return CLEAN / f"ch{chapter:02d}.txt"
