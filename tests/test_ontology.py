import json

import pytest

from sfqa.ontology import (
    SYNTHETIC_ROOT,
    Alias,
    Calling,
    Character,
    Fact,
    Knowing,
    Ontology,
    Relation,
    RelationState,
    character_view,
    check,
    from_synthetic,
    main,
    reader_view,
    split_facts,
    synthetic_ids,
)


def k(character: str, from_ch: int) -> Knowing:
    return Knowing(character=character, from_ch=from_ch)


# 10장짜리 작은 작품. 갑순(a)은 1장, 을동(b)은 2장, 병철(c)은 7장에 처음 나온다.
ONTO = Ontology(
    work="t", title="시험", chapters=10, source="test",
    characters=[
        Character(id="a", name="갑순", first_ch=1, aliases=[Alias(form="회색 깃털", from_ch=6)],
                  called=[Calling(by="b", form="갑순 씨", from_ch=2, to_ch=4), Calling(by="b", form="갑순", from_ch=5)]),
        Character(id="b", name="을동", first_ch=2),
        Character(id="c", name="병철", first_ch=7),
    ],
    facts=[
        Fact(id="f1", statement="갑순은 필경사다.", reveal_ch=1, known_by=[k("a", 0), k("b", 2)]),
        Fact(id="f2", statement="갑순이 회색 깃털이다.", reveal_ch=6, known_by=[k("a", 0), k("b", 8)], is_twist=True),
        Fact(id="f3", statement="병철은 갑순의 오빠다.", reveal_ch=7, known_by=[k("a", 0), k("c", 0)]),
        Fact(id="f4", statement="을동은 빚이 있다.", reveal_ch=3, known_by=[k("b", 0)]),
    ],
    relations=[
        Relation(id="r1", a="a", b="b", states=[
            RelationState(from_ch=2, reveal_ch=2, status="처음 만남", known_by=[k("a", 2), k("b", 2)]),
            RelationState(from_ch=5, reveal_ch=5, status="비밀 연인", known_by=[k("a", 5), k("b", 5)]),
        ]),
        Relation(id="r2", a="a", b="c", states=[
            RelationState(from_ch=0, reveal_ch=7, status="남매", known_by=[k("a", 0), k("c", 0)]),
        ]),
    ],
)


def ids(facts) -> list[str]:
    return [f.id for f in facts]


def test_reader_axis_hides_what_the_reader_has_not_read_even_if_the_character_knows():
    known, reader_only, hidden = split_facts(ONTO, "a", upto=5)
    assert ids(known) == ["f1"]
    assert ids(reader_only) == ["f4"]  # 갑순은 을동의 빚을 모른다
    assert ids(hidden) == ["f2", "f3"]  # 갑순은 처음부터 알지만 독자는 6·7장에 안다


def test_character_axis_keeps_reader_facts_the_character_has_not_learned():
    known, reader_only, _ = split_facts(ONTO, "b", upto=6)
    assert ids(known) == ["f1", "f4"]
    assert ids(reader_only) == ["f2"]  # 독자는 6장에 알지만 을동은 8장에 안다


def test_at_moves_the_character_back_in_time():
    assert "f2" in ids(split_facts(ONTO, "b", upto=9, at=7)[1])
    assert "f2" in ids(split_facts(ONTO, "b", upto=9, at=8)[0])


def test_views_never_hold_unread_or_unknown_statements():
    for upto in range(1, ONTO.chapters + 1):
        rv = reader_view(ONTO, upto).model_dump_json()
        assert all(f.statement not in rv for f in ONTO.facts if f.reveal_ch > upto)
        for c in ONTO.characters:
            if c.first_ch > upto:
                continue
            for at in range(1, upto + 1):
                cv = character_view(ONTO, c.id, upto, at).model_dump_json()
                _, reader_only, hidden = split_facts(ONTO, c.id, upto, at)
                assert all(f.statement not in cv for f in reader_only + hidden)


def test_characters_not_yet_on_stage_are_left_out():
    v = reader_view(ONTO, 6)
    assert [c.id for c in v.characters] == ["a", "b"]
    assert [r.id for r in v.relations] == ["r1"]  # 병철이 아직 안 나와 남매 관계도 뺀다
    assert v.characters[0].aliases == ["회색 깃털"]
    assert reader_view(ONTO, 5).characters[0].aliases == []  # 별칭도 6장부터
    assert [r.id for r in reader_view(ONTO, 7).relations] == ["r1", "r2"]


def test_relation_states_pass_both_axes():
    assert [s.status for s in reader_view(ONTO, 4).relations[0].states] == ["처음 만남"]
    assert [s.status for s in reader_view(ONTO, 5).relations[0].states] == ["처음 만남", "비밀 연인"]
    assert [r.id for r in character_view(ONTO, "c", 7).relations] == ["r2"]  # 병철은 갑순·을동 관계를 모른다


def test_calls_follow_chapter_ranges():
    def forms(upto: int) -> list[str]:
        return [c.form for c in character_view(ONTO, "b", upto).calls]

    assert forms(4) == ["갑순 씨"]  # to_ch는 그 장을 포함한다
    assert forms(5) == ["갑순"]
    assert [c.form for c in character_view(ONTO, "b", 9, at=3).calls] == ["갑순 씨"]


def test_character_view_rejects_bad_requests():
    with pytest.raises(ValueError, match="등장하지 않았다"):
        character_view(ONTO, "c", 6)
    with pytest.raises(ValueError, match="at은"):
        character_view(ONTO, "a", 5, at=6)
    with pytest.raises(ValueError, match="upto는"):
        reader_view(ONTO, 11)
    with pytest.raises(ValueError, match="없는 인물"):
        split_facts(ONTO, "z", 5)


def test_hidden_alias_keeps_a_secret_identity_out_of_its_owner_view():
    onto = ONTO.model_copy(update={"facts": [
        *ONTO.facts,
        Fact(id="f5", statement="회색 깃털이 을동에게 편지를 보냈다.", reveal_ch=3, known_by=[k("a", 0), k("b", 3)]),
    ]})
    assert "f5" in ids(split_facts(onto, "a", 5)[1])  # 갑순이 회색 깃털이라는 것은 6장에 공개된다
    assert "f5" in ids(split_facts(onto, "b", 5)[0])  # 받은 사람은 말해도 된다
    assert "f5" in ids(split_facts(onto, "a", 6)[0])


def test_hidden_alias_skips_names_inside_known_names():
    teo = Character(id="t", name="테오도르", first_ch=1, aliases=[Alias(form="테오", from_ch=6)])
    onto = Ontology(work="t", title="시험", chapters=10, source="test", characters=[teo],
                    facts=[Fact(id="f1", statement="테오도르는 왕세자다.", reveal_ch=1, known_by=[k("t", 0)])])
    assert teo.hidden_forms(3) == []
    assert ids(split_facts(onto, "t", 3)[0]) == ["f1"]


def test_common_knowledge_is_known_by_everyone_from_the_start_but_still_waits_for_the_reader():
    onto = ONTO.model_copy(update={"facts": [
        *ONTO.facts,
        Fact(id="f6", statement="은종은 서약이 봉인될 때마다 운다.", reveal_ch=3, common=True),
    ]})
    for who, upto in (("a", 3), ("b", 3), ("c", 7)):  # 병철은 7장에 처음 나오지만 원래 아는 상식이다
        assert "f6" in ids(split_facts(onto, who, upto)[0])
    assert "f6" in ids(split_facts(onto, "a", 2)[2])  # 독자가 아직 안 읽었으면 상식이라도 숨긴다
    assert next(f for f in character_view(onto, "c", 7).facts if f.id == "f6").since == 0


def test_characters_never_on_stage_stay_out_of_views():
    onto = ONTO.model_copy(update={"characters": [*ONTO.characters, Character(id="d", name="정숙", first_ch=None, mention_ch=2)]})
    assert check(onto) == []
    assert "d" not in [c.id for c in reader_view(onto, 10).characters]
    with pytest.raises(ValueError, match="등장하지 않았다"):
        character_view(onto, "d", 10)


def test_check_reports_broken_references_and_ranges():
    broken = ONTO.model_copy(update={"facts": [
        *ONTO.facts,
        Fact(id="f1", statement="중복", reveal_ch=2),
        Fact(id="f9", statement="범위 밖", reveal_ch=11, known_by=[k("z", 3)], inferable_ch=12),
    ]})
    errs = check(broken)
    assert check(ONTO) == []
    assert "사실 id 중복: f1" in errs
    assert any("없는 인물 z" in e for e in errs)
    assert any("f9 공개: 장 범위 밖" in e for e in errs)
    assert any("짐작 가능 장(12)" in e for e in errs)


# ---------- 합성 웹소설 설정집(data/synthetic) ----------

WORKS = synthetic_ids()
needs_synthetic = pytest.mark.skipif(not WORKS, reason="합성 웹소설(data/synthetic)이 없다")
needs_pilot = pytest.mark.skipif("wn01-rofan" not in WORKS, reason="파일럿(wn01-rofan)이 없다")


@pytest.fixture(scope="module")
def gold() -> dict[str, Ontology]:
    return {w: from_synthetic(w) for w in WORKS}


@needs_synthetic
def test_gold_conversion_is_clean_and_complete(gold):
    for wid, onto in gold.items():
        assert check(onto) == [], wid
        lines = (SYNTHETIC_ROOT / wid / "facts.jsonl").read_text(encoding="utf-8").splitlines()
        bible = json.loads((SYNTHETIC_ROOT / wid / "bible.json").read_text(encoding="utf-8"))
        assert len(onto.facts) == sum(1 for line in lines if line.strip())
        assert len(onto.characters) == len(bible["characters"])
        assert len(onto.relations) == len(bible["relationships"])
        assert [s.chapter for s in onto.scenes] == list(range(1, onto.chapters + 1))


@needs_synthetic
def test_gold_buckets_partition_and_knowledge_only_grows(gold):
    for wid, onto in gold.items():
        everything = sorted(f.id for f in onto.facts)
        for c in onto.characters:
            before: set[str] = set()
            for n in range(1, onto.chapters + 1):
                known, reader_only, hidden = split_facts(onto, c.id, n)
                assert sorted(ids(known + reader_only + hidden)) == everything, (wid, c.id, n)  # 빠짐도 겹침도 없다
                assert all(f.reveal_ch <= n for f in known + reader_only) and all(f.reveal_ch > n for f in hidden)
                now = set(ids(known))
                assert before <= now, (wid, c.id, n)  # 한 번 안 것은 잊지 않는다
                before = now


@needs_synthetic
def test_gold_views_hold_no_unread_or_unknown_fact(gold):
    for wid, onto in gold.items():
        for n in range(1, onto.chapters + 1):
            rv = reader_view(onto, n)
            assert all(f.chapter <= n for f in rv.facts) and rv.hidden == sum(f.reveal_ch > n for f in onto.facts)
            assert all(c.first_ch <= n for c in rv.characters)
            for c in onto.characters:
                if c.first_ch > n:
                    continue
                cv = character_view(onto, c.id, n)
                _, reader_only, hidden = split_facts(onto, c.id, n)
                assert not {f.id for f in cv.facts} & set(ids(reader_only + hidden)), (wid, c.id, n)
                assert not {f.statement for f in cv.facts} & {f.statement for f in reader_only + hidden}, (wid, c.id, n)


@needs_pilot
def test_pilot_twists_follow_both_axes(gold):
    onto = gold["wn01-rofan"]

    def bucket(character: str, upto: int, fid: str, at: int | None = None) -> str:
        parts = dict(zip(("known", "reader_only", "hidden"), split_facts(onto, character, upto, at)))
        return next(name for name, facts in parts.items() if fid in ids(facts))

    # f034 해약의 힘: 10장 공개. 세실리아·카시엘·힐다는 10장, 노엘은 40장에 안다
    assert bucket("cecilia", 9, "f034") == "hidden"
    assert bucket("cassiel", 10, "f034") == "known"
    assert bucket("noel", 10, "f034") == "reader_only"
    assert bucket("noel", 45, "f034", at=30) == "reader_only"
    assert bucket("noel", 45, "f034") == "known"
    # f114 회색 깃털의 정체: 테오도르는 처음부터 알지만 독자는 37장에 안다. 로완은 43장에 안다
    assert bucket("theodore", 36, "f114") == "hidden"
    assert bucket("theodore", 37, "f114") == "known"
    assert bucket("rowan", 37, "f114") == "reader_only"
    # f024 회색 깃털 편지(6장 공개): 테오도르는 6장부터 알지만, 정체가 드러나는 37장 전에는 말하면 안 된다
    assert bucket("theodore", 36, "f024") == "reader_only"
    assert bucket("theodore", 37, "f024") == "known"
    assert bucket("cecilia", 10, "f024") == "known"


@needs_pilot
def test_pilot_calls_background_relations_and_evidence(gold):
    onto = gold["wn01-rofan"]

    def calls(upto: int) -> set[str]:
        return {c.form for c in character_view(onto, "cassiel", upto).calls if c.target == "cecilia"}

    assert calls(15) == {"온드레 양"} and calls(16) == {"세실리아"}
    # 세실리아–레온 '부녀'(배경 상태)는 레온이 등장하는 45장부터 보인다(늦게 막는 쪽으로 추정)
    assert "r12" not in [r.id for r in character_view(onto, "cecilia", 44).relations]
    assert "r12" in [r.id for r in character_view(onto, "cecilia", 45).relations]
    f034 = next(f for f in onto.facts if f.id == "f034")
    assert f034.evidence[0].chapter == 10 and 167 in f034.evidence[0].lines  # 10장 167줄 '해약.'


@needs_pilot
def test_cli_build_and_show(capsys, tmp_path):
    assert main(["build", "--work", "wn01-rofan", "--out", str(tmp_path)]) == 0
    assert "wn01-rofan 「은종탑의 필경사는 계약 약혼 중」: 인물 12, 사실 189" in capsys.readouterr().out
    saved = Ontology.model_validate_json((tmp_path / "wn01-rofan.json").read_text(encoding="utf-8"))
    assert len(saved.facts) == 189
    assert main(["show", "--work", "wn01-rofan", "--upto", "10", "--character", "cassiel"]) == 0
    out = capsys.readouterr().out
    assert "카시엘 로스벤(cassiel)의 10장 시점" in out and "f034" in out
    assert main(["show", "--work", "wn01-rofan", "--upto", "3", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["upto"] == 3
