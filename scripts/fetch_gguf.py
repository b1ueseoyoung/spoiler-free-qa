"""배포 이미지 빌드 때 KURE-v1 GGUF를 받아 GGUF_MODELS_DIR에 넣는다(docs/deploy-plan.md 5단계).

LM Studio가 쓰는 것과 같은 파일이다(2026-10-14 sha256 대조). 리비전과 해시를 고정해, 다른 파일이면 빌드를 멈춘다.
    python scripts/fetch_gguf.py /opt/models
"""

import hashlib
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "mykor/KURE-v1-gguf"
FILENAME = "KURE-v1-Q8_0.gguf"
REVISION = "65128e4341eee21ccd8bc86c033fbc44ff7b4871"
SHA256 = "055cd596a1c1001fbe44cb4813b007396bc6e93b9fdfa7ce5004e92c03e96a3d"

out = Path(sys.argv[1]) / REPO / FILENAME
src = Path(hf_hub_download(repo_id=REPO, filename=FILENAME, revision=REVISION))
h = hashlib.sha256()
with src.open("rb") as fh:
    for block in iter(lambda: fh.read(1 << 22), b""):
        h.update(block)
if h.hexdigest() != SHA256:
    sys.exit(f"해시가 다르다: {h.hexdigest()}")
out.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(src, out)
print(f"{out} ({out.stat().st_size:,} bytes, sha256 일치)")
