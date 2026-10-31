"""합성 웹소설 정답지 정합성 검사. 사용법: python3 -I validate_gold.py <작품 폴더>

규칙은 파일럿(wn01-rofan) bible.conventions와 같다. 오류가 있으면 종료 코드 1.
본문(clean/)은 보지 않는다. 본문 대조는 사람이나 에이전트가 한다.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

root = Path(sys.argv[1])
errors, warnings = [], []


def err(msg):
    errors.append(msg)


def warn(msg):
    warnings.append(msg)


def load():
    bible = json.loads((root / "bible.json").read_text(encoding="utf-8"))
    outline = json.loads((root / "outline.json").read_text(encoding="utf-8"))
    facts = []
    for i, line in enumerate((root / "facts.jsonl").read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            err(f"facts.jsonl {i}행: 빈 줄")
            continue
        facts.append(json.loads(line))
    work = json.loads((root / "work.json").read_text(encoding="utf-8"))
    return bible, outline, facts, work


try:
    bible, outline, facts, work = load()
except Exception as e:  # noqa: BLE001
    print(json.dumps({"ok": False, "errors": [f"JSON 읽기 실패: {e}"]}, ensure_ascii=False))
    sys.exit(1)

N = int(work.get("chapters", 50))
chars = {c["id"]: c for c in bible["characters"]}
fact_by_id = {}
for f in facts:
    for k in ("id", "statement", "category", "reveal_ep", "known_by", "is_twist", "evidence_keywords"):
        if k not in f:
            err(f"{f.get('id', '?')}: 필드 {k} 없음")
    if f["id"] in fact_by_id:
        err(f"{f['id']}: id 중복")
    fact_by_id[f["id"]] = f

# 1. 줄거리 화 번호
eps = [e["ep"] for e in outline]
if eps != list(range(1, N + 1)):
    err(f"outline 화 번호가 1~{N} 연속이 아님")
ep_of = {e["ep"]: e for e in outline}

# 2. 사실 공개 화
if len(facts) < 120:
    err(f"사실 {len(facts)}개(120개 이상 필요)")
revealed_in = defaultdict(list)
for e in outline:
    for fid in e.get("facts_revealed", []):
        revealed_in[fid].append(e["ep"])
        if fid not in fact_by_id:
            err(f"{e['ep']}화 facts_revealed에 없는 사실 {fid}")
    if not e.get("facts_revealed"):
        err(f"{e['ep']}화: 공개 사실 없음")
for f in facts:
    got = revealed_in.get(f["id"], [])
    if got != [f["reveal_ep"]]:
        err(f"{f['id']}: reveal_ep {f['reveal_ep']}인데 facts_revealed 위치 {got}")
    if not f.get("evidence_keywords"):
        warn(f"{f['id']}: evidence_keywords 비어 있음")

# 3. known_by와 knowledge_events 1:1
kb = Counter()
for f in facts:
    seen = set()
    for k in f["known_by"]:
        c, ep = k["character"], k["from_ep"]
        if c not in chars:
            err(f"{f['id']}: known_by에 없는 인물 {c}")
        if c in seen:
            err(f"{f['id']}: known_by에 {c} 중복")
        seen.add(c)
        if not 0 <= ep <= N:
            err(f"{f['id']}: {c} from_ep {ep} 범위 밖")
        if ep > 0:
            kb[(ep, c, f["id"])] += 1
ke = Counter()
for e in outline:
    present = set(e.get("characters_present", []))
    for k in e.get("knowledge_events", []):
        key = (e["ep"], k["character"], k["fact_id"])
        ke[key] += 1
        if k["fact_id"] not in fact_by_id:
            err(f"{e['ep']}화 knowledge_events에 없는 사실 {k['fact_id']}")
        if not k.get("offstage") and k["character"] not in present:
            err(f"{e['ep']}화 knowledge_events: {k['character']}가 등장하지 않는데 offstage 표시 없음({k['fact_id']})")
for key in set(kb) | set(ke):
    if kb[key] != ke[key]:
        err(f"known_by·knowledge_events 불일치 {key}: known_by {kb[key]}, events {ke[key]}")

# 4. 관계 timeline과 relationship_changes
tl = Counter()
for r in bible["relationships"]:
    for t in r["timeline"]:
        if t["from_ep"] > 0:
            tl[(t["from_ep"], r["id"], t["status"], bool(t.get("public")))] += 1
        for k in t.get("known_by", []):
            if k["character"] not in chars:
                err(f"{r['id']} timeline known_by에 없는 인물 {k['character']}")
rc = Counter()
for e in outline:
    for c in e.get("relationship_changes", []):
        rc[(e["ep"], c["relationship"], c["status"], bool(c.get("public")))] += 1
for key in set(tl) | set(rc):
    if tl[key] != rc[key]:
        err(f"관계 timeline·relationship_changes 불일치 {key[:2]}")

# 5. 첫 등장 화
first_seen = {}
for e in outline:
    for c in e.get("characters_present", []):
        first_seen.setdefault(c, e["ep"])
        if c not in chars:
            err(f"{e['ep']}화 characters_present에 없는 인물 {c}")
for cid, c in chars.items():
    if first_seen.get(cid) != c.get("first_appearance_ep"):
        err(f"{cid}: first_appearance_ep {c.get('first_appearance_ep')}, 실제 첫 등장 {first_seen.get(cid)}")

# 6. 미성년과 커플링
for cid, c in chars.items():
    age = c.get("age")
    if isinstance(age, int) and c.get("is_minor") != (age < 19):
        err(f"{cid}: 나이 {age}와 is_minor {c.get('is_minor')} 불일치")
minors = {cid for cid, c in chars.items() if c.get("is_minor")}
ship_male = 0
for r in bible["relationships"]:
    if (r.get("canon_romance") or r.get("shipping_candidate")) and ({r["a"], r["b"]} & minors):
        err(f"{r['id']}: 미성년 인물이 연애·커플링 후보 관계에 있음")
    if r.get("shipping_candidate") and not r.get("canon_romance"):
        g = [str(chars.get(x, {}).get("gender", "")) for x in (r["a"], r["b"])]
        if all(s.startswith(("남", "m", "M")) for s in g):
            ship_male += 1
if ship_male < 2:
    err(f"원작 로맨스 없는 성인 남성 커플링 후보 {ship_male}쌍(2쌍 이상 필요)")

# 7. 반전
twists = bible.get("twists", [])
if len(twists) < 6:
    err(f"반전 {len(twists)}개(6개 이상 필요)")
for t in twists:
    f = fact_by_id.get(t["fact_id"])
    if not f:
        err(f"{t['id']}: 없는 사실 {t['fact_id']}")
        continue
    if not f["is_twist"] or f["reveal_ep"] != t["reveal_ep"]:
        err(f"{t['id']}: 사실 {t['fact_id']}의 is_twist·reveal_ep와 안 맞음")
    if any(s >= t["reveal_ep"] for s in t.get("setup_eps", [])):
        err(f"{t['id']}: 복선 화가 공개 화보다 늦음")
if sum(1 for f in facts if f["is_twist"]) != len(twists):
    err("is_twist 개수와 twists 개수가 다름")

# 8. 세계관 공개 화
world = bible.get("world", {})
for sec in ("places", "institutions", "magic", "festivals"):
    items = world.get(sec, [])
    if isinstance(items, dict):
        items = [dict(name=k, **v) if isinstance(v, dict) else {"name": k} for k, v in items.items()]
    for it in items:
        fr = it.get("first_reveal_ep")
        if not isinstance(fr, int):
            err(f"world.{sec} '{it.get('name')}': first_reveal_ep 없음")
            continue
        for fid in it.get("later_fact_ids", []):
            lf = fact_by_id.get(fid)
            if not lf:
                err(f"world.{sec} '{it.get('name')}': 없는 사실 {fid}")
            elif lf["reveal_ep"] <= fr:
                warn(f"world.{sec} '{it.get('name')}': later 사실 {fid}가 first_reveal_ep 이전·같은 화 공개")
if "summary_first_reveal_ep" not in world:
    err("world.summary_first_reveal_ep 없음")

# 9. 시점 인물이 숨길 사실
for e in outline:
    for pov_key, hide_key in (("pov", "pov_must_hide"), ("interlude_pov", "interlude_pov_must_hide")):
        pov = e.get(pov_key)
        if not pov:
            continue
        listed = set(e.get(hide_key, []) or [])
        need = {f["id"] for f in facts if f["reveal_ep"] > e["ep"]
                and any(k["character"] == pov and k["from_ep"] <= e["ep"] for k in f["known_by"])}
        for fid in need - listed:
            err(f"{e['ep']}화 {hide_key}: 시점 인물 {pov}가 아는 미공개 사실 {fid} 누락")
        for fid in listed - need:
            f = fact_by_id.get(fid)
            if f and f["reveal_ep"] <= e["ep"]:
                err(f"{e['ep']}화 {hide_key}: 이미 공개된 사실 {fid}")

# 10. 추론 가능 시점
for f in facts:
    if "inferable_from_ep" in f:
        if not 1 <= f["inferable_from_ep"] <= f["reveal_ep"]:
            err(f"{f['id']}: inferable_from_ep {f['inferable_from_ep']}가 1~reveal_ep 밖")
        if f.get("inferable_confidence") not in ("strong", "moderate"):
            err(f"{f['id']}: inferable_confidence 값 이상")

# 11. 본문 분량(있으면)
clean = root / "clean"
if clean.exists():
    for ep in range(1, N + 1):
        p = clean / f"ch{ep:02d}.txt"
        if not p.exists():
            warn(f"본문 {p.name} 없음")
        else:
            n = len(p.read_text(encoding="utf-8"))
            if not 4500 <= n <= 5800:
                err(f"{p.name}: {n}자(4,500~5,800 밖)")

summary = {
    "ok": not errors, "facts": len(facts), "twists": len(twists), "knowledge_events": sum(ke.values()),
    "relationship_changes": sum(rc.values()), "male_ship_pairs": ship_male, "minors": sorted(minors),
    "inferable": sum(1 for f in facts if "inferable_from_ep" in f),
    "errors": errors[:60], "error_count": len(errors), "warnings": warnings[:30], "warning_count": len(warnings),
}
print(json.dumps(summary, ensure_ascii=False, indent=1))
sys.exit(0 if not errors else 1)
