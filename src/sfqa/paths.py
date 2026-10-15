"""프로젝트 안의 경로. 작품별 경로는 sfqa.works.Work가 갖는다(data/works/<id>/)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
