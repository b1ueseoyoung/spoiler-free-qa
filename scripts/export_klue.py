"""KLUE-MRC validation에서 평가 표본(유형별 80문항)을 뽑아 eval/klue/에 쓴다.

parquet은 munjigi가 Hugging Face(klue/klue)에서 받아 둔 것을 읽는다. pyarrow는 이 스크립트에만 필요하다.
    uv run --with pyarrow python scripts/export_klue.py [parquet 경로]
"""

import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from sfqa.klue import sample, write_sample

DEFAULT = Path.home() / "Projects/dev/munjigi/data/raw/mrc/validation-00000-of-00001.parquet"

src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
rows = pq.read_table(src).to_pylist()
items = sample(rows)
out = write_sample(items)
print(f"원본 {len(rows)}문항 {dict(sorted(Counter(r['question_type'] for r in rows).items()))}")
print(f"표본 {len(items)}문항 {dict(sorted(Counter(i.question_type for i in items).items()))} → {out}")
print(f"지문 길이 중앙값 {sorted(len(i.context) for i in items)[len(items) // 2]}자, 서로 다른 지문 "
      f"{len({i.context for i in items})}개")
