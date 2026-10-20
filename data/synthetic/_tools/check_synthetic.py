"""합성 웹소설 기계 검사. 사용법: python3 -I check_synthetic.py <작품 폴더>

모델 판정 없이 문자열로만 본다. 걸린 항목은 '의심'이고, 사람이 읽어 확정한다.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

root = Path(sys.argv[1])
bible = json.loads((root / "bible.json").read_text(encoding="utf-8"))
outline = json.loads((root / "outline.json").read_text(encoding="utf-8"))
facts = [json.loads(l) for l in (root / "facts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
work = json.loads((root / "work.json").read_text(encoding="utf-8"))
n_eps = int(work["chapters"])

texts = {}
missing_files = []
for ep in range(1, n_eps + 1):
    p = root / "clean" / f"ch{ep:02d}.txt"
    if p.exists():
        texts[ep] = p.read_text(encoding="utf-8")
    else:
        missing_files.append(ep)

out = {"missing_files": missing_files}

# 1. 분량
lens = {ep: len(t) for ep, t in texts.items()}
out["length"] = {
    "min": min(lens.values()) if lens else None,
    "max": max(lens.values()) if lens else None,
    "total": sum(lens.values()),
    "out_of_range": {ep: n for ep, n in lens.items() if not 4500 <= n <= 5800},
}

# 2. 형식
fmt_patterns = {
    "markdown": r"^\s*(#|\*\*|---|>)",
    "ep_number": r"^\s*(제?\s*\d+\s*화|EP\.?\s*\d+|Chapter)",
    "meta": r"(다음 화|작가의 ?말|To be continued|계속\)|\(계속)",
    "foreshadow_narrator": r"(그때는 몰랐다|나중에야 알게|훗날|그것이 마지막)",
}
fmt = defaultdict(list)
for ep, t in texts.items():
    for name, pat in fmt_patterns.items():
        for line in t.splitlines():
            if re.search(pat, line):
                fmt[name].append((ep, line.strip()[:60]))
out["format"] = {k: v[:20] for k, v in fmt.items()}

# 3. 사실 특징 표현의 첫 등장 화 vs 공개 화
early, absent = [], []
for f in facts:
    kws = [k for k in f.get("evidence_keywords", []) if len(k) >= 2]
    if not kws:
        continue
    first = {}
    for kw in kws:
        eps = [ep for ep in sorted(texts) if kw in texts[ep]]
        first[kw] = eps[0] if eps else None
    rev = f["reveal_ep"]
    hits_before = {kw: e for kw, e in first.items() if e is not None and e < rev}
    if hits_before:
        early.append({"id": f["id"], "reveal_ep": rev, "twist": f.get("is_twist", False),
                      "statement": f["statement"][:80], "early_keywords": hits_before})
    if rev in texts and not any(kw in texts[rev] for kw in kws):
        absent.append({"id": f["id"], "reveal_ep": rev, "keywords": kws, "statement": f["statement"][:80]})
out["keyword_before_reveal"] = early
out["keyword_absent_in_reveal_ep"] = absent

# 4. 인물 이름이 첫 등장 화보다 먼저 나오는가(언급일 수 있으니 의심만)
name_early = []
for c in bible["characters"]:
    names = [c["name"]] + [a for a in c.get("aliases", []) if len(a) >= 2]
    given = c["name"].split()[0]
    if len(given) >= 2:
        names.append(given)
    fa = int(c.get("first_appearance_ep", 1))
    for ep in sorted(texts):
        if ep >= fa:
            break
        hit = [n for n in set(names) if n in texts[ep]]
        if hit:
            name_early.append({"character": c["name"], "first_appearance_ep": fa, "ep": ep, "names": hit})
out["name_before_first_appearance"] = name_early

# 5. 미성년 인물 근처의 연애 표현(같은 문단)
romance = r"(입맞춤|키스|입술을 겹|연모|사랑해|설레|품에 안|약혼자로서|신부로|혼인하자)"
minor_flags = []
for c in bible["characters"]:
    if not c.get("is_minor"):
        continue
    names = [c["name"], c["name"].split()[0]] + c.get("aliases", [])
    for ep, t in texts.items():
        for para in t.split("\n\n"):
            if any(n in para for n in names if len(n) >= 2) and re.search(romance, para):
                minor_flags.append({"ep": ep, "character": c["name"], "para": para.strip()[:120]})
out["minor_near_romance"] = minor_flags

# 6. 화 사이 반복 문단(40자 이상 같은 문단)
seen = defaultdict(list)
for ep, t in texts.items():
    for para in t.split("\n\n"):
        s = para.strip()
        if len(s) >= 40:
            seen[s].append(ep)
out["repeated_paragraphs"] = [{"eps": eps, "para": s[:80]} for s, eps in seen.items() if len(set(eps)) > 1][:30]

# 7. 화별 공개 사실 수와 대사 비중
dialog = {}
for ep, t in texts.items():
    lines = [l for l in t.splitlines() if l.strip()]
    dialog[ep] = round(sum(1 for l in lines if l.strip().startswith(("\"", "“", "「", "'"))) / max(1, len(lines)), 2)
out["dialogue_ratio"] = {"min": min(dialog.values()) if dialog else None, "max": max(dialog.values()) if dialog else None,
                         "mean": round(sum(dialog.values()) / len(dialog), 2) if dialog else None}

summary = {
    "episodes_present": len(texts), "missing_files": len(missing_files),
    "length_min_max_total": (out["length"]["min"], out["length"]["max"], out["length"]["total"]),
    "out_of_range": len(out["length"]["out_of_range"]),
    "format_hits": {k: len(v) for k, v in fmt.items()},
    "keyword_before_reveal": len(early), "keyword_before_reveal_twists": sum(1 for e in early if e["twist"]),
    "keyword_absent_in_reveal_ep": len(absent), "facts_total": len(facts),
    "name_before_first_appearance": len(name_early), "minor_near_romance": len(minor_flags),
    "repeated_paragraphs": len(out["repeated_paragraphs"]), "dialogue_ratio": out["dialogue_ratio"],
}
(root / "checks").mkdir(exist_ok=True)
(root / "checks" / "mechanical.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=1))
