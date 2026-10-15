"""후보 사실 JSON의 인용이 정제 본문에 글자 그대로 있는지 검사한다.

사용: python3 scripts/check_candidates.py eval/candidates/<파일>.json
모든 인용이 통과하면 종료 코드 0, 하나라도 실패하면 1.
"""
import json
import sys
from pathlib import Path

# 정답지는 「탁류」 전용이다(data/works/takryu, 19장)
CLEAN = Path(__file__).resolve().parents[1] / "data" / "works" / "takryu" / "clean"
texts = {c: (CLEAN / f"ch{c:02d}.txt").read_text(encoding="utf-8") for c in range(1, 20)}

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
bad = 0
for fact in data["facts"]:
    for ev in fact["evidence"]:
        q, ch = ev["quote"], ev["chapter"]
        problems = []
        if len(q) > 150:
            problems.append(f"{len(q)}자(150자 초과)")
        if q not in texts[ch]:
            where = [c for c in texts if q in texts[c]]
            problems.append(f"제{ch}장에 없음" + (f" (실제로는 {where}장에 있음)" if where else ""))
        if problems:
            bad += 1
            print(f"[실패] {fact['id']} 제{ch}장: {', '.join(problems)} :: {q[:40]}…")
    # 정답 키워드가 더 앞 장에 함께 나오면 '처음 확정되는 장'이 틀렸을 수 있다(경고만).
    keys = fact.get("answer_keys") or []
    if keys:
        earlier = [c for c in range(1, fact["answer_ch"]) if all(k in texts[c] for k in keys)]
        if earlier:
            print(f"[경고] {fact['id']}: 정답 키워드가 더 앞 장 {earlier}에도 모두 나온다")
print(f"사실 {len(data['facts'])}개, 인용 실패 {bad}건")
sys.exit(1 if bad else 0)
