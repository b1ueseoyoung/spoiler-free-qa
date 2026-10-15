"""새 작품의 인물·지명표를 형태소 분석기(kiwipiepy)로 자동으로 만든다(사용자 결정, 2026-10-14).

C4의 인물 가드는 "N장 뒤에 처음 나오는 이름이 답에 있으면 답하지 않는다"이다. 그래서 이름 목록이 필요하다.
- 고유명사(NNP) 가운데 두 글자 이상, 작품 전체에서 min_count번 이상 나온 것만 남긴다.
- '선형이'처럼 끝에 '이'가 붙은 형태는 '선형'의 별칭으로 합친다.
- 성이 붙은 세 글자 이름은 성을 뗀 이름을 별칭으로 둔다(김정근 → 정근, 강기천 → 기천). 본문은 주로 성 없이 부른다.
- 형태소 분석기가 놓친 이름은 work.json의 extra_entities에 적는다(예: 「무정」의 병욱은 '병/욱'으로 쪼개져 빠졌다).
- 첫 등장 장은 저장하지 않는다. 쓸 때 본문에서 계산한다(gold.entity_index).
- 사람이 검토하지 않은 목록이다(review: pre-human-review). 일반 명사가 섞이거나 이름이 빠질 수 있다.
  「탁류」 인물표(사람이 손본 88개)는 이 방식으로 다시 만들지 않는다.
"""

from __future__ import annotations

import json
from collections import Counter

from sfqa.gold import first_chapter, load_texts
from sfqa.works import Work

# 흔한 한 글자 성. 세 글자 이름에서 이 성을 떼어 별칭을 만든다.
SURNAMES = set("김이박최정강조윤장임한오서신권황안송류전홍고문양손배백허유남심노하곽성차주우구민진나지엄채원천방공현함변염여추도소석선설마길연위표명기반왕금옥육인맹제모탁국어은편용예봉경")


def extract(work: Work, min_count: int = 3) -> list[dict]:
    from kiwipiepy import Kiwi

    texts = load_texts(work)
    kiwi = Kiwi()
    counts: Counter = Counter()
    for text in texts.values():
        for tok in kiwi.tokenize(text):
            if tok.tag == "NNP" and len(tok.form) >= 2:
                counts[tok.form] += 1
    body = "".join(texts.values())
    names = {f for f, n in counts.items() if n >= min_count}
    for extra in work.source.get("extra_entities", []):  # 사람이 더한 이름(형태소 분석기가 놓친 것)
        names.add(extra)
        counts[extra] = max(counts[extra], body.count(extra))
    aliases: dict[str, list[str]] = {}
    for f in sorted(names):
        if f.endswith("이") and f[:-1] in names:  # '선형이' → '선형'
            aliases.setdefault(f[:-1], []).append(f)
    for f in sorted(names):
        given = f[1:]
        if len(f) == 3 and f[0] in SURNAMES and body.count(given) >= min_count:  # '김정근' → '정근'
            aliases.setdefault(given, []).append(f)
            names.add(given)
            counts[given] = max(counts[given], body.count(given))
    alias_of = {a for al in aliases.values() for a in al}
    out = []
    for f in sorted(names, key=lambda x: (-counts[x], x)):
        if f in alias_of:
            continue
        forms = [f, *aliases.get(f, [])]
        if first_chapter(forms, texts) is None:  # 형태소 분석기가 원문과 다른 꼴로 돌려준 경우
            continue
        how = "work.json extra_entities" if f in work.source.get("extra_entities", []) else "kiwipiepy 고유명사"
        out.append({"name": f, "aliases": aliases.get(f, []), "kind": "other", "review": "pre-human-review",
                    "note": f"{how}, {counts[f]}회"})
    return out


def write(work: Work, min_count: int = 3) -> int:
    entities = extract(work, min_count)
    work.entities_path.write_text(json.dumps(entities, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(entities)
