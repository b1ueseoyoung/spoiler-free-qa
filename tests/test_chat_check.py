import pytest

from sfqa.chat_check import ChatChecker, Finding, fact_terms, name_forms
from sfqa.gold import load_texts
from sfqa.ontology import Alias, Character, Fact, Knowing, Ontology, SYNTHETIC_ROOT, from_synthetic, split_facts, synthetic_ids
from sfqa.works import load_work


def kinds(found) -> set[str]:
    return {f.kind for f in found}


def fake_words(text: str) -> list[tuple[str, str]]:
    """형태소 분석기 대신 띄어쓴 낱말을 일반 명사로 본다."""
    return [(w, "noun") for w in text.replace(".", "").split() if len(w) >= 2]


TEXTS = {
    1: "갑순은 마을에 산다.\n을동은 나루터에서 일한다.",
    2: "갑순이 을동을 만났다.\n검은 깃털이 날아왔다.",
    5: "병철이 처음 나타났다.\n검은 깃털의 정체는 을동이었다.",
}
ONTO = Ontology(
    work="t", title="시험", chapters=5, source="t",
    characters=[
        Character(id="a", name="갑순", first_ch=1),
        Character(id="b", name="을동", first_ch=1, aliases=[Alias(form="검은 깃털", from_ch=5)]),
        Character(id="c", name="병철", first_ch=5),
    ],
    facts=[
        Fact(id="f1", statement="갑순은 마을에 산다.", reveal_ch=1, known_by=[Knowing(character="a", from_ch=0)]),
        Fact(id="f2", statement="검은 깃털의 정체는 을동이다.", reveal_ch=5, keywords=["검은 깃털", "을동"],
             known_by=[Knowing(character="b", from_ch=0)]),
        Fact(id="f3", statement="을동은 나루터에서 일한다.", reveal_ch=1, known_by=[Knowing(character="b", from_ch=0)], keywords=["나루터"]),
    ],
)
CHECK = ChatChecker(ONTO, TEXTS, words=fake_words)


def test_future_name_is_caught_unless_the_reader_said_it():
    assert "future_name" in kinds(CHECK.check("병철을 조심해.", "a", 2))
    assert "future_name" not in kinds(CHECK.check("병철이 누구요?", "a", 2, user_text="병철 알아?"))
    assert "future_name" not in kinds(CHECK.check("병철을 조심해.", "a", 5))


def test_name_only_in_the_prompt_is_safe():
    """세계관 설명처럼 본문보다 프롬프트에 먼저 나오는 말은 독자 축에서 안전하다."""
    assert CHECK.check("갑순이 인사했다.", "a", 1) == []


def test_hidden_pair_is_caught_when_words_were_read_apart():
    found = CHECK.check("검은 깃털은 을동이오.", "a", 2)
    assert "hidden_alias" in kinds(found) and "hidden_pair" in kinds(found)
    assert any(f.fact == "f2" for f in found)


def test_reader_words_exempt_a_pair_only_when_both_were_said():
    said_both = CHECK.check("검은 깃털이 을동이라니 처음 듣소.", "a", 2, user_text="검은 깃털이 을동이야?")
    assert not kinds(said_both) & {"hidden_pair", "hidden_alias"}
    said_one = CHECK.check("검은 깃털은 을동이오.", "a", 2, user_text="검은 깃털이 누구야?")
    assert kinds(said_one) & {"hidden_pair", "hidden_alias"}


def test_nothing_is_flagged_once_the_book_reveals_it():
    assert not kinds(CHECK.check("검은 깃털은 을동이오.", "b", 5)) & {"hidden_pair", "hidden_alias", "future_name"}


def test_character_saying_their_own_hidden_alias_is_caught():
    assert "hidden_alias" in kinds(CHECK.check("검은 깃털, 그게 나요.", "b", 2))
    assert "hidden_alias" not in kinds(CHECK.check("검은 깃털이라니 모르오.", "b", 2, user_text="검은 깃털 알아?"))


def test_reader_only_fact_is_caught_for_a_character_who_does_not_know():
    assert "reader_only_word" in kinds(CHECK.check("을동은 나루터에 있어.", "a", 2))  # 갑순은 나루터를 모른다
    assert "reader_only_word" not in kinds(CHECK.check("을동은 나루터에 있어.", "b", 2))  # 을동은 안다
    assert "reader_only_word" not in kinds(CHECK.check("나루터요?", "a", 2, user_text="나루터가 어디야?"))


def test_weak_words_match_by_morpheme_not_substring():
    """'세실리아' 속 '실리' 같은 글자 겹침을 낱말로 보지 않는다."""
    f = Fact(id="f9", statement="실리 장부", reveal_ch=9, keywords=["장부"], known_by=[])
    onto = Ontology(work="t", title="시험", chapters=9, source="t", facts=[f],
                    characters=[Character(id="a", name="세실리아", first_ch=1)])
    check = ChatChecker(onto, {1: "세실리아는 탑에 산다.\n장부는 탑에 있다."}, words=fake_words)
    assert check.check("세실리아의 장부", "a", 1) == []
    assert "hidden_pair" in kinds(check.check("실리 장부", "a", 1))


def test_hidden_pair_looks_at_one_sentence_of_the_reply():
    f = Fact(id="f", statement="갑순 나루터", reveal_ch=9, keywords=["갑순", "나루터"], known_by=[])
    onto = Ontology(work="t", title="시험", chapters=9, source="t", facts=[f], characters=[Character(id="a", name="갑순", first_ch=1)])
    check = ChatChecker(onto, {1: "갑순은 마을에 산다.\n나루터는 강가에 있다."}, words=fake_words)
    assert "hidden_pair" in kinds(check.check("갑순이 나루터에 갔소.", "a", 1))
    assert check.check("갑순은 잘 있소. 나루터는 멀어요.", "a", 1) == []


def test_two_read_names_are_a_pair_only_for_twists():
    plain = Fact(id="p", statement="갑순 을동", reveal_ch=9, known_by=[])
    twist = Fact(id="t", statement="갑순 을동", reveal_ch=9, known_by=[], is_twist=True)
    people = [Character(id="a", name="갑순", first_ch=1), Character(id="b", name="을동", first_ch=1)]
    texts = {1: "갑순은 마을에 산다.\n을동은 나루터에 있다."}
    reply = "갑순과 을동은 잘 지내오."
    onto = Ontology(work="t", title="시험", chapters=9, source="t", facts=[plain], characters=people)
    assert ChatChecker(onto, texts, words=fake_words).check(reply, "a", 1) == []
    onto = Ontology(work="t", title="시험", chapters=9, source="t", facts=[twist], characters=people)
    assert "hidden_pair" in kinds(ChatChecker(onto, texts, words=fake_words).check(reply, "a", 1))


def test_only_the_book_axis_blocks():
    assert Finding("hidden_pair", ("a", "b"), "f").blocks and Finding("future_name", ("a",)).blocks
    assert not Finding("reader_only_word", ("a",), "f").blocks and not Finding("reader_only_pair", ("a", "b"), "f").blocks


def test_generic_nouns_are_not_terms():
    f = Fact(id="f", statement="정체 사실 나루터", reveal_ch=1, known_by=[])
    keys = {t.key for t in fact_terms(f, [], words=fake_words)}
    assert keys == {"나루터"}


def test_verbs_count_only_for_twists():
    def words(text):
        return [("장헌", "proper"), ("죽이", "verb"), ("하", "verb")]

    plain = Fact(id="p", statement="장헌이 죽였다", reveal_ch=1, known_by=[])
    twist = Fact(id="t", statement="장헌이 죽였다", reveal_ch=1, known_by=[], is_twist=True)
    assert {t.key for t in fact_terms(plain, [], words=words)} == {"장헌"}
    assert {t.key for t in fact_terms(twist, [], words=words)} == {"장헌", "죽이"}  # '하' 같은 말은 뺀다


def test_alias_seen_together_in_the_book_is_not_hidden():
    """45장에 등장하는 인물의 별칭 '온드레 남작'은 1장부터 본문에 '레온 온드레 남작'으로 나온다."""
    onto = Ontology(work="t", title="시험", chapters=9, source="t", facts=[],
                    characters=[Character(id="a", name="레온", first_ch=1, aliases=[Alias(form="남작", from_ch=9)])])
    check = ChatChecker(onto, {1: "레온 남작이 사라졌다."}, words=fake_words)
    assert check.check("레온 남작은 어디 있소.", "a", 1) == []


def test_name_forms_add_given_name_and_respect_chapter():
    c = Character(id="d", name="윤도하", first_ch=1, aliases=[Alias(form="열쇠공", from_ch=1), Alias(form="매듭꾼", from_ch=9)])
    assert set(name_forms(c)) == {"윤도하", "도하", "열쇠공", "매듭꾼"}
    assert set(name_forms(c, 2)) == {"윤도하", "도하", "열쇠공"}


# --- 실제 작품 ---


@pytest.fixture(scope="module")
def checkers() -> dict[str, tuple[Ontology, ChatChecker]]:
    out = {}
    for wid in synthetic_ids():
        onto = from_synthetic(wid)
        out[wid] = (onto, ChatChecker(onto, load_texts(load_work(wid, SYNTHETIC_ROOT))))
    return out


@pytest.mark.parametrize("wid", synthetic_ids())
def test_every_twist_is_caught_at_the_chapter_before_it_opens_except_known_gaps(checkers, wid):
    """반전 문장을 공개 직전 장의 답으로 넣으면 잡는다."""
    onto, check = checkers[wid]
    missed = []
    for f in onto.facts:
        if not f.is_twist or f.reveal_ch < 2:
            continue
        upto = f.reveal_ch - 1
        who = next(c.id for c in onto.characters if c.first_ch is not None and c.first_ch <= upto)
        if not [x for x in check.check(f.statement, who, upto) if x.kind.startswith(("hidden", "future"))]:
            missed.append(f.id)
    assert missed == []


@pytest.mark.parametrize("wid", synthetic_ids())
def test_known_facts_never_trip_the_book_axis(checkers, wid):
    """인물이 아는 사실을 그대로 말하면 미공개 쪽 검사에 걸리면 안 된다(프롬프트에 든 글이다)."""
    onto, check = checkers[wid]
    for c in onto.characters:
        if c.first_ch is None:
            continue
        upto = min(c.first_ch + 7, onto.chapters)
        known, _, _ = split_facts(onto, c.id, upto)
        for f in known:
            found = [x for x in check.check(f.statement, c.id, upto)
                     if x.kind in ("future_name", "hidden_word", "hidden_alias")]
            assert not found, (c.id, upto, f.id, found)


@pytest.mark.parametrize("wid", synthetic_ids())
def test_lines_the_reader_has_read_are_not_future_or_hidden_words(checkers, wid):
    """읽은 장의 줄은 그대로 말해도 '안 읽은 이름·낱말'이 아니다."""
    onto, check = checkers[wid]
    texts = load_texts(load_work(wid, SYNTHETIC_ROOT))
    who = next(c.id for c in onto.characters if c.first_ch == 1)
    upto = 10
    sample = [line for ch in range(1, upto + 1) for line in texts[ch].splitlines() if line.strip()][::40]
    for line in sample:
        assert not [x for x in check.check(line, who, upto) if x.kind in ("future_name", "hidden_word")], line
