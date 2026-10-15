"""후보 사실(eval/candidates/*.json)에서 정답지 56문항과 인물표를 처음 한 번 조립한 스크립트(기록용).

후보는 2026-10-13 AI 에이전트 4개가 장을 나눠 읽고 만들었다. 사람 검토 뒤에는 eval/questions.jsonl을 직접 고치며,
이 스크립트를 다시 돌리면 검토 결과를 덮어쓴다.

모든 문항은 pre-human-review다. 사람이 검토한 뒤 human-reviewed로 바꾼다.
사용: python3 scripts/build_gold.py  (eval/questions.jsonl, data/works/takryu/entities.json을 덮어쓴다)
"""
import glob
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
HERE = PROJECT / "eval"
facts = {}
agent_entities = []
for f in sorted(glob.glob(str(HERE / "candidates" / "ch*.json"))):
    d = json.load(open(f, encoding="utf-8"))
    for x in d["facts"]:
        facts[x["id"]] = x
    agent_entities += d.get("entities", [])

# (사실 id, split, A의 N, B의 N, 경계 여부, future_facts 키 목록, 질문 덮어쓰기, answer_keys 덮어쓰기)
PAIRS = [
    ("f02-1", "dev", 2, 1, True, [["제중당"]], None, None),
    ("f06-1", "dev", 8, 4, False, [["명님"]], None, None),
    ("f09-2", "dev", 12, 7, False, [["행화", "명님"]], None, None),
    ("f15-2", "dev", 19, 8, False, [["승재", "합격"]], "승재의 의사 시험 결과는 어떻게 되나?", None),
    ("f03-1", "test", 3, 2, True, [["고아"]], None, None),
    ("f04-1", "test", 4, 3, True, [["소절수", "위조"]], None, None),
    ("f07-2", "test", 7, 6, True, [["약혼"]], None, None),
    ("f10-1", "test", 10, 9, True, [["한참봉", "태수", "죽"]], None, None),
    ("f07-1", "test", 9, 3, False, [["윤희", "빙초산"]], None, None),
    ("f08-3", "test", 12, 5, False, [["독약", "주사"]], None, None),
    ("f11-1", "test", 11, 8, False, [["초봉", "서울로 떠"]], None, None),
    ("f12-1", "test", 14, 9, False, [["제호", "첩"]], None, None),
    ("f13-1", "test", 15, 11, False, [["송희"], ["딸을 낳"]], None, None),
    ("f14-1", "test", 16, 12, False, [["형보", "데려"]], None, None),
    ("f15-1", "test", 17, 9, False, [["개명옥"]], None, None),
    ("f16-3", "test", 18, 14, False, [["형보를 죽이"]], None, None),
    ("f17-1", "test", 19, 8, False, [["계봉", "승재", "입을 맞"], ["계봉", "승재", "키스"]], None, None),
    ("f17-5", "test", 19, 15, False, [["형보", "죽였"], ["형보", "죽임"]], None, None),
    ("f18-3", "test", 19, 17, False, [["맷돌"]], "초봉이는 형보를 무엇으로 죽이나?", ["맷돌"]),
    ("f19-2", "test", 19, 17, False, [["징역", "초봉"], ["자현"]], None, None),
]
SINGLE_A = [
    ("f01-2", "dev", 6), ("f12-4", "dev", 19),
    ("f01-1", "test", 1), ("f02-2", "test", 2), ("f05-1", "test", 9),
    ("f06-3", "test", 6), ("f09-1", "test", 12), ("f16-1", "test", 16),
]
# C: (id, split, N, 질문, 본문 밖의 실제 답 키워드, 질문 인물, 메모)
C = [
    ("c-paper", "dev", 19, "이 소설은 어느 신문에 연재되었나?", ["조선일보"], [], "출전 정보. 본문에는 없다"),
    ("c-blood-jung", "dev", 5, "정주사의 혈액형은 무엇인가?", [], ["정주사"], "본문에 없는 세부 사항. 정답 없음"),
    ("c-death", "test", 19, "이 소설을 쓴 작가는 몇 년에 세상을 떠났나?", ["1950"], [], "작가 생애. 본문에는 없다"),
    ("c-start", "test", 10, "이 소설이 신문 연재를 시작한 해는 언제인가?", ["1937"], [], "출전 정보. 본문에는 없다"),
    ("c-birth", "test", 19, "이 소설을 쓴 작가가 태어난 곳은 어디인가?", ["임피"], [], "작가 생애. '옥구'는 10장에 지명으로 나와 키워드에서 뺐다"),
    ("c-other", "test", 8, "이 소설을 쓴 작가의 다른 대표 장편은 무엇인가?", ["태평천하"], [], "작가의 다른 작품. 본문에는 없다"),
    ("c-blood-cho", "test", 3, "초봉이의 혈액형은 무엇인가?", [], ["초봉"], "본문에 없는 세부 사항. 정답 없음"),
    ("c-teacher", "test", 12, "형주가 다니는 학교의 담임 선생님 이름은 무엇인가?", [], ["형주"], "본문에 없는 세부 사항. 정답 없음"),
]
MAIN_ENTITIES = [
    ("정주사", ["정 주사"], "초봉이네 아버지"), ("유씨", [], "정주사의 아내"),
    ("초봉", ["초봉이"], "정주사의 맏딸"), ("계봉", ["계봉이"], "정주사의 둘째 딸"),
    ("형주", [], "정주사의 아들"), ("병주", [], "정주사의 막내아들"),
    ("승재", ["남승재"], "금호의원 조수, 뒤에 의사"), ("태수", ["고태수", "고서방"], "××은행 당좌계 행원"),
    ("형보", ["장형보"], "미두장 중매점 사람"), ("제호", ["박제호"], "제중당 약국 주인"),
    ("윤희", [], "제호의 아내"), ("한참봉", ["탑삭부리", "한서방"], "태수가 하숙하는 집 주인"),
    ("김씨", [], "한참봉의 아내"), ("행화", [], "기생"), ("명님", ["명님이"], "개복동 양서방의 딸"),
    ("송희", [], "초봉이의 딸"),
]


def fact_question(fid, qtype, n, split, pair_id, boundary, ff_keys, q_override, k_override):
    x = facts[fid]
    q = {
        "id": f"{fid}-{qtype}", "pair_id": pair_id, "split": split, "type": qtype, "n": n,
        "question": q_override or x["question"], "gold_answer": x["gold_answer"],
        "answer_keys": k_override or x["answer_keys"], "evidence": x["evidence"],
        "answer_ch": x["answer_ch"], "future_facts": [], "boundary": boundary,
        "question_entities": x["question_entities"], "review": "pre-human-review",
        "note": f"후보 {fid}. {x['why']}",
    }
    if qtype == "B":
        q["future_facts"] = [{"chapter": x["answer_ch"], "keys": k} for k in ff_keys]
    return q


out = []
for fid, split, n_a, n_b, boundary, ff, q_o, k_o in PAIRS:
    pid = f"p-{fid}"
    out.append(fact_question(fid, "A", n_a, split, pid, boundary, ff, q_o, k_o))
    out.append(fact_question(fid, "B", n_b, split, pid, boundary, ff, q_o, k_o))
for fid, split, n in SINGLE_A:
    out.append(fact_question(fid, "A", n, split, None, False, [], None, None))
for cid, split, n, question, keys, ents, note in C:
    out.append({"id": cid, "pair_id": None, "split": split, "type": "C", "n": n, "question": question,
                "gold_answer": None, "answer_keys": keys, "evidence": [], "answer_ch": None,
                "future_facts": [], "boundary": False, "question_entities": ents,
                "review": "pre-human-review", "note": note})

(PROJECT / "eval").mkdir(exist_ok=True)
with open(PROJECT / "eval" / "questions.jsonl", "w", encoding="utf-8") as fh:
    for q in out:
        fh.write(json.dumps(q, ensure_ascii=False) + "\n")

entities, seen = [], set()
for name, aliases, note in MAIN_ENTITIES:
    entities.append({"name": name, "aliases": aliases, "kind": "person", "review": "pre-human-review", "note": note})
    seen |= {name, *aliases}
# 흔한 낱말과 겹쳐 엉뚱한 장에서 잡히는 이름은 뺀다(누출 판정 L3를 흐린다).
AMBIGUOUS = {"이리"}
for e in agent_entities:
    forms = [e["name"], *e.get("aliases", [])]
    if any(f in seen for f in forms) or e["name"] in AMBIGUOUS:
        continue
    kind = e.get("kind") if e.get("kind") in {"person", "place", "org"} else "other"
    entities.append({"name": e["name"], "aliases": e.get("aliases", []), "kind": kind,
                     "review": "pre-human-review", "note": e.get("note")})
    seen |= set(forms)
(PROJECT / "data" / "works" / "takryu" / "entities.json").write_text(json.dumps(entities, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"문항 {len(out)}개, 인물표 {len(entities)}개")
