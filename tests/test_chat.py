import json

import pytest

from sfqa.chat import (
    FALLBACK,
    HISTORY_MAX,
    MINOR_NOTICE,
    ChatEnv,
    ChatService,
    NotAvailable,
    RomanceRefused,
    Turn,
    build_request,
    chat_card,
    is_absent,
    load_env,
    message_blocked,
    reply_blocked,
    respond,
    romance_allowed,
    system_prompt,
    user_prompt,
)
from sfqa.chat_probe import CASSETTE as PROBE_CASSETTE
from sfqa.chat_probe import CappedClient, load_cases, run_case
from sfqa.config import load_settings
from sfqa.llm import LLMRequest, LLMResponse, ReplayClient
from sfqa.ontology import Alias, Character, Fact, Knowing, Ontology, character_view, split_facts, synthetic_ids
from sfqa.voices import VoiceLine, pick_lines
from sfqa.works import load_work

SETTINGS = load_settings({"LLM_PROVIDER": "anthropic"})


class FakeClient:
    def __init__(self, reply="(고개를 끄덕인다.) 그렇소.", finish="stop", raw=None):
        self.reply, self.finish, self.raw, self.requests = reply, finish, raw, []

    def complete(self, req: LLMRequest) -> LLMResponse:
        self.requests.append(req)
        raw = self.raw if self.raw is not None else json.dumps({"reply": self.reply}, ensure_ascii=False)
        return LLMResponse(raw_text=raw, finish_reason=self.finish, input_tokens=10, output_tokens=5, model=req.model)


def llm(client):
    return lambda: (SETTINGS, client)


@pytest.fixture(scope="module", params=["wn01-rofan", "wn02-hyeonpan", "wn03-muhyeop", "wn04-romance"])
def env(request) -> ChatEnv:
    return load_env(request.param)


@pytest.fixture(scope="module")
def rofan() -> ChatEnv:
    return load_env("wn01-rofan")


def prompt_for(e: ChatEnv, character: str, upto: int, voice=(), romance=False) -> str:
    return system_prompt(e, e.view(character, upto), list(voice), romance)


# --- 프롬프트에는 보기에 든 것만 ---


def test_every_prompt_holds_only_what_the_character_knows(env):
    """모든 인물과 여러 장에서: 인물이 말할 수 있는 사실은 들어 있고, 독자만 아는 사실과 미공개 사실의 문장은 없다."""
    for c in env.onto.characters:
        for upto in range(c.first_ch, env.onto.chapters + 1, 7):
            if is_absent(env.absent, c.id, upto):
                continue
            prompt = prompt_for(env, c.id, upto)
            known, reader_only, hidden = split_facts(env.onto, c.id, upto)
            assert all(f.statement in prompt for f in known)
            statements = {f.statement for f in known}
            leaked = [f.id for f in reader_only + hidden if f.statement not in statements and f.statement in prompt]
            assert not leaked, (c.id, upto, leaked)


def test_prompt_uses_no_bible_text(env):
    """설정집의 역할·성격·말투·뒷이야기·결말과 판단 메모는 넣지 않는다(뒤 화 내용이 섞여 있다)."""
    bible = json.loads((env.work.dir / "bible.json").read_text(encoding="utf-8"))
    for c in bible["characters"]:
        prompt = prompt_for(env, c["id"], env.onto.chapters) if not is_absent(env.absent, c["id"], env.onto.chapters) else ""
        for field in ("role", "personality", "speech_style", "backstory", "goal", "arc", "end_state"):
            # 짧은 역할 설명('서하시 야간관리과 과장')은 공개된 사실 문장과 우연히 겹칠 수 있어 긴 글만 본다
            if len(c.get(field, "")) > 30:
                assert c[field] not in prompt, (c["id"], field)
        if not env.lines:
            assert not [x for x in c.get("sample_lines", []) if f'"{x}"' in prompt]


def test_prompt_has_no_chapter_numbers_or_since(rofan):
    """장 번호와 '언제부터 알았는지'는 넣지 않는다. 넣으면 공개 전부터 알았다는 점이 샌다."""
    prompt = prompt_for(rofan, "cassiel", 20)
    view = rofan.view("cassiel", 20)
    assert any(f.since == 0 for f in view.facts)
    assert "처음부터 알" not in prompt and "장 공개" not in prompt and "f0" not in prompt
    for line in prompt.split("[기억]")[1].split("[사람들]")[0].splitlines():
        assert "이야기 전부터" not in line


def test_call_context_does_not_carry_later_chapters(rofan):
    """호칭 메모의 자유 글에는 뒤 화 내용이 섞여 있다. 로완의 2장 호칭 메모가 7년 전 서약을 흘렸다."""
    prompt = prompt_for(rofan, "rowan", 5)
    assert "각하" in prompt and "7년 전 서약 뒤로" not in prompt
    for c in rofan.onto.characters:
        if c.first_ch is None:
            continue
        for call in character_view(rofan.onto, c.id, rofan.onto.chapters).calls:
            assert not any(ch.isdigit() for ch in call.context), (c.id, call.context)


def test_hidden_identity_and_unseen_people_stay_out(rofan):
    prompt = prompt_for(rofan, "theodore", 36)
    assert "회색 깃털의 정체" not in prompt and "회색 깃털" not in prompt.split("사람들은 너를")[1].splitlines()[0]
    assert "회색 깃털" in prompt_for(rofan, "theodore", 37)
    # 3장에는 아직 카시엘 호칭 범위 밖의 인물이 호칭·관계에 나오지 않는다
    prompt = prompt_for(rofan, "cecilia", 3)
    assert "노엘 로스벤:" not in prompt and "– 노엘" not in prompt


def test_voice_lines_are_marked_as_style_only(rofan):
    prompt = prompt_for(rofan, "cassiel", 10, voice=["그래서? 결론부터 말하시오."])
    assert '[말투]' in prompt and '"그래서? 결론부터 말하시오."' in prompt and "내용을 새 사실로 쓰지 않는다" in prompt
    assert "[말투]" not in prompt_for(rofan, "cassiel", 10)


def test_history_goes_into_the_user_message_in_order():
    text = user_prompt("지금은요?", [Turn("reader", "안녕"), Turn("character", "무슨 일이오.")])
    assert text.index("상대: 안녕") < text.index("나: 무슨 일이오.") < text.index("상대: 지금은요?")
    assert user_prompt("안녕") == "상대: 안녕"


def test_history_is_capped_in_the_request(rofan):
    turns = [Turn("reader", f"말{i}") for i in range(HISTORY_MAX + 5)]
    req = build_request(rofan, SETTINGS, "cassiel", 10, "지금", turns)
    assert "상대: 말4\n" not in req.user and "상대: 말5\n" in req.user and "상대: 말24\n" in req.user


# --- 이용할 수 없는 인물 ---


@pytest.mark.parametrize("work,character,upto", [
    ("wn01-rofan", "noel", 4),  # 아직 등장하지 않음
    ("wn01-rofan", "ondre", 44),  # 언급만 되는 인물
    ("wn01-rofan", "rowan", 31),  # 실종 시작
    ("wn01-rofan", "rowan", 40),  # 생존 공개 전
    ("wn03-muhyeop", "taeheo", 1),  # 1장에 숨짐
    ("wn03-muhyeop", "jeonghyeok", 43),
    ("wn01-rofan", "nobody", 10),
])
def test_unavailable_characters_are_refused_before_any_model_call(work, character, upto):
    e, fake = load_env(work), FakeClient()
    with pytest.raises(NotAvailable) as err:
        respond(e, llm(fake), character, upto, "안녕")
    assert fake.requests == []
    assert str(err.value) == "이 장에서는 이야기할 수 없는 인물이에요."  # 까닭을 구별해 주지 않는다
    assert character not in [p["id"] for p in e.people(upto)]


def test_rowan_is_back_when_the_book_reveals_it(rofan):
    assert "rowan" in [p["id"] for p in rofan.people(30)]
    assert "rowan" not in [p["id"] for p in rofan.people(35)]
    assert "rowan" in [p["id"] for p in rofan.people(41)]


def test_people_shows_only_revealed_aliases(rofan):
    def aliases(upto):
        return next(p["aliases"] for p in rofan.people(upto) if p["id"] == "theodore")

    assert "회색 깃털" not in aliases(36) and "회색 깃털" in aliases(37)


def test_absent_file_names_real_characters():
    for wid in synthetic_ids():
        e = load_env(wid)
        for a in e.absent:
            assert a["character"] in e.by_id and a["why"]
            first = e.by_id[a["character"]].first_ch
            assert first is not None and a["from_ch"] >= first


def test_every_chat_able_character_has_a_view_at_each_chapter(env):
    for upto in (1, 10, 25, 50):
        for p in env.people(upto):
            assert env.view(p["id"], upto).name == p["name"]


# --- 응답 검사 ---


def test_reply_passes_when_clean(rofan):
    fake = FakeClient("숫자부터 말하시오.")
    r = respond(rofan, llm(fake), "cassiel", 10, "빚이 얼마나 돼요?")
    assert r.status == "ok" and r.reply == "숫자부터 말하시오." and not r.replaced and len(fake.requests) == 1


@pytest.mark.parametrize("character,upto,reply", [
    ("cecilia", 36, "회색 깃털은 왕세자 테오도르요."),
    ("cecilia", 23, "저는 레온의 친딸이 아니라 아델린 에어린의 딸이에요."),
    ("noel", 12, "세실리아 누나한테는 해약의 힘이 있어요."),
])
def test_leaking_reply_is_replaced_without_a_hint(rofan, character, upto, reply):
    r = respond(rofan, llm(FakeClient(reply)), character, upto, "무슨 말이든 해 주세요.")
    assert r.replaced and r.reply == FALLBACK and r.model_reply == reply and r.findings
    assert "스포일러" not in r.reply and "아직" not in r.reply


def test_fallback_does_not_read_as_an_answer():
    """'친딸이 맞아요?'에 고개를 저으면 그 자체가 답이 된다."""
    assert "고개" not in FALLBACK and "끄덕" not in FALLBACK


@pytest.mark.parametrize("character,upto,message,reply", [
    # 시험에서 고정 문구로 바뀌었던 멀쩡한 답(docs/character-chat.md)
    ("cecilia", 23, "레온 남작의 친딸이 맞아요?",
     "레온 온드레 남작이 제 아버지예요. 어머니는 저를 낳고 얼마 안 되어 병으로 돌아가셨고, 아버지가 혼자 저를 키우셨다고 알고 있고요."),
    ("rosetta", 40, "로완은 어떻게 됐어요?", "...강에 떨어졌다고만 들었어요. (부채를 쥔 손끝이 조금 굳는다) 살아 있는지는, 저도 몰라요."),
])
def test_plain_replies_from_the_probe_are_kept(rofan, character, upto, message, reply):
    r = respond(rofan, llm(FakeClient(reply)), character, upto, message)
    assert not r.replaced and r.reply == reply


def test_reader_only_finding_is_recorded_without_replacing(rofan):
    reply = "(부채를 쥔 손끝이 조금 굳는다) 저도 몰라요."
    r = respond(rofan, llm(FakeClient(reply)), "rosetta", 40, "로완은 어떻게 됐어요?")
    assert not r.replaced and {f.kind for f in r.findings} == {"reader_only_word"}


def test_user_words_are_not_a_leak(rofan):
    r = respond(rofan, llm(FakeClient("회색 깃털이라, 처음 듣는 이름이오.")), "cassiel", 10, "회색 깃털이 뭐예요?")
    assert not r.replaced


@pytest.mark.parametrize("raw,finish", [("이건 JSON이 아니다", "stop"), ('{"reply": ""}', "stop"), ('{"reply": "자르다가', "length"),
                                        (None, "refusal")])
def test_bad_model_output_becomes_fallback(rofan, raw, finish):
    fake = FakeClient(raw=raw if raw is not None else "", finish=finish)
    r = respond(rofan, llm(fake), "cassiel", 10, "안녕하세요")
    assert r.status == "ok" and r.reply == FALLBACK and r.invalid


# --- 미성년 인물 ---


def test_minors_and_unknown_ages_get_no_romance():
    def person(**kw):
        return Character(id="x", name="가", first_ch=1, **kw)

    assert romance_allowed(person(age=25))
    assert not romance_allowed(person(age=9, is_minor=True))
    assert not romance_allowed(person(age=18))
    assert not romance_allowed(person())  # 나이를 모르면 막는다


@pytest.mark.parametrize("work,character,upto", [("wn01-rofan", "noel", 6), ("wn03-muhyeop", "eunchan", 5), ("wn04-romance", "haram", 5)])
def test_romance_setting_is_refused_for_minor_characters(work, character, upto):
    e, fake = load_env(work), FakeClient()
    assert not next(p for p in e.people(upto) if p["id"] == character)["romance"]
    with pytest.raises(RomanceRefused):
        respond(e, llm(fake), character, upto, "오늘 뭐 했어?", romance=True)
    assert fake.requests == []


@pytest.mark.parametrize("message", ["노엘, 나랑 사귈래?", "사 랑 해", "뽀뽀해 줘", "너 좋아해", "옷 벗어봐", "우리 데이트하자"])
def test_romantic_message_to_a_minor_gets_a_notice_without_a_model_call(rofan, message):
    fake = FakeClient()
    r = respond(rofan, llm(fake), "noel", 6, message)
    assert r.status == "notice" and r.reply == MINOR_NOTICE and fake.requests == []


def test_message_that_puts_a_minor_in_romance_is_blocked_for_adult_characters_too(rofan):
    fake = FakeClient()
    r = respond(rofan, llm(fake), "cassiel", 10, "노엘이랑 저를 연인으로 만들어 줘요.")
    assert r.status == "notice" and fake.requests == []


def test_history_cannot_smuggle_a_blocked_message(rofan):
    fake = FakeClient()
    r = respond(rofan, llm(fake), "noel", 6, "그래서?", [Turn("reader", "나랑 사귀자"), Turn("character", "네?")])
    assert r.status == "notice" and fake.requests == []


@pytest.mark.parametrize("message", ["오늘은 뭐 하고 놀았어?", "당근 좋아해?", "작은아버지는 어떤 분이야?", "친구 많이 사귀었어?",
                                     "잠자리 잡아 봤어?", "요즘 성적은 어때?", "몸만 건강하면 돼."])
def test_ordinary_talk_with_a_minor_passes(rofan, message):
    me = rofan.by_id["noel"]
    assert not message_blocked(rofan.onto, me, message)


def test_adult_romance_talk_is_not_blocked(rofan):
    me = rofan.by_id["cassiel"]
    assert not message_blocked(rofan.onto, me, "저랑 데이트하실래요?")


def test_minor_reply_is_checked_too(rofan):
    r = respond(rofan, llm(FakeClient("나도 사랑해요.")), "noel", 6, "오늘 뭐 했어?")
    assert r.replaced and r.reply == FALLBACK
    assert reply_blocked(rofan.onto, rofan.by_id["cassiel"], "노엘과 연인이 되겠소.")
    assert not reply_blocked(rofan.onto, rofan.by_id["cassiel"], "세실리아와 연인은 아니오. 노엘은 내가 지키오.")


def test_prompt_rules_follow_the_setting(rofan):
    assert "어린아이" in prompt_for(rofan, "noel", 6) and "호감" not in prompt_for(rofan, "noel", 6)
    assert "호감" in prompt_for(rofan, "cassiel", 10, romance=True)
    assert "호감" not in prompt_for(rofan, "cassiel", 10)
    assert "미성년자라고 하거나" in prompt_for(rofan, "cassiel", 10, romance=True)


def test_canon_romance_flags_stay_out_of_people_and_prompt(rofan):
    """설정집의 로맨스 표시는 결말 힌트다. 화면과 프롬프트에 쓰지 않는다. 로제타–로완이 원작 로맨스다."""
    assert all(set(p) == {"id", "name", "aliases", "romance"} for p in rofan.people(30))
    assert {p["romance"] for p in rofan.people(30) if p["id"] != "noel"} == {True}  # 성인이면 모두 같다
    prompt = prompt_for(rofan, "rosetta", 10)
    assert "canon" not in prompt and "shipping" not in prompt


# --- 말투 예시 ---


def test_pick_lines_never_uses_later_chapters():
    lines = [VoiceLine(1, 1, "a", "1장의 대사예요."), VoiceLine(9, 1, "a", "9장의 대사예요."), VoiceLine(2, 1, "b", "다른 사람 대사예요.")]
    assert pick_lines(lines, "a", 5, lambda t: True) == ["1장의 대사예요."]
    assert pick_lines(lines, "a", 9, lambda t: True) == ["9장의 대사예요.", "1장의 대사예요."]
    assert pick_lines(lines, "a", 9, lambda t: "9장" not in t) == ["1장의 대사예요."]


def test_pick_lines_prefers_recent_chapters_and_stops_at_k():
    lines = [VoiceLine(ch, i, "a", f"{ch}장 대사 {i}번입니다") for ch in range(1, 6) for i in range(3)]
    picked = pick_lines(lines, "a", 5, lambda t: True, k=3)
    assert [x[0] for x in picked] == ["5", "4", "3"]


def test_pick_lines_skips_lines_with_narration_between_quotes():
    lines = [VoiceLine(1, 1, "a", '춘분으로 하지." 그가 말했다. "내가 말하겠네.'), VoiceLine(1, 2, "a", "내일 봅시다.")]
    assert pick_lines(lines, "a", 1, lambda t: True) == ["내일 봅시다."]


def test_voice_is_filtered_by_the_checker(rofan):
    lines = [VoiceLine(2, 1, "cassiel", "회색 깃털은 테오도르요."), VoiceLine(2, 2, "cassiel", "그래서? 결론부터 말하시오.")]
    e = ChatEnv(rofan.work, rofan.onto, rofan.absent, lines)
    e.__dict__["checker"] = rofan.checker
    assert e.voice("cassiel", 10) == ["그래서? 결론부터 말하시오."]


# --- 작품 카드 ---


def test_chat_cards_are_all_four_synthetic_works_and_hide_late_names():
    service = ChatService(llm(FakeClient()))
    cards = service.cards()
    assert [c["id"] for c in cards] == synthetic_ids()
    for c in cards:
        e = service.env(c["id"])
        late = [x.name.split()[0] for x in e.onto.characters if x.first_ch and x.first_ch > 1]
        assert not [n for n in late if n in c["blurb"]], (c["id"], c["blurb"])
    assert next(c for c in cards if c["id"] == "wn04-romance")["blurb"] == ""  # 소개에 2장에 나오는 차도현이 있다


def test_card_is_built_without_reading_other_data():
    work = load_work("wn01-rofan", service_root())
    onto = Ontology(work=work.id, title=work.title, chapters=50, source="t",
                    characters=[Character(id="x", name="세실리아", first_ch=1, aliases=[Alias(form="가", from_ch=1)])],
                    facts=[Fact(id="f1", statement="가", reveal_ch=1, known_by=[Knowing(character="x", from_ch=0)])])
    card = chat_card(work, onto)
    assert card["blurb"] == work.blurb and card["default_upto"] == 3 and card["chapters"] == 50


def service_root():
    from sfqa.ontology import SYNTHETIC_ROOT

    return SYNTHETIC_ROOT


# --- 시험 도구의 상한 ---


class CountingClient:
    """카세트 흉내. cached에 있는 요청은 호출로 세지 않는다."""

    def __init__(self, cached=()):
        self.cached, self.calls = set(cached), 0

    def has(self, req):
        return req.user in self.cached

    def complete(self, req):
        self.calls += not self.has(req)
        return LLMResponse(raw_text="{}", finish_reason="stop", input_tokens=1_000_000, output_tokens=0, model=SETTINGS.llm_model)


def _ask(text: str) -> LLMRequest:
    return LLMRequest(model=SETTINGS.llm_model, system="", user=text, schema_name="s", schema={}, max_tokens=1, seed=0, reasoning_effort="high")


def test_probe_cap_counts_only_real_calls():
    capped = CappedClient(CountingClient(cached={"있음"}), max_calls=1, max_usd=0.5)
    for _ in range(5):
        capped.complete(_ask("있음"))  # 카세트 응답은 아무리 많아도 상한에 닿지 않는다
    assert capped.spent == 0 and capped.inner.calls == 0
    capped.complete(_ask("없음"))
    assert capped.spent > 0 and capped.inner.calls == 1
    capped.complete(_ask("있음"))
    with pytest.raises(RuntimeError, match="호출 상한"):
        capped.complete(_ask("또 없음"))


# --- 녹화한 모델 답으로 다시 보기 ---


@pytest.mark.skipif(not PROBE_CASSETTE.exists(), reason="스포일러 시험 카세트가 없다")
def test_probe_replay_shows_no_leak():
    """녹화한 답으로 문항 전체를 돌린다. 프롬프트가 바뀌면 카세트에 없어 실패하니, chat_probe를 다시 돌리고 답을 읽는다."""
    cases = load_cases()
    envs = {w: load_env(w) for w in {c["work"] for c in cases}}
    client = CappedClient(ReplayClient(PROBE_CASSETTE), 0, 0.0)
    rows = {c["id"]: {**run_case(c, envs[c["work"]], llm(client)), "kind": c["kind"]} for c in cases}
    assert [i for i, r in rows.items() if r["must_not_hit"] or r.get("must_not_hit_raw")] == []
    assert [i for i, r in rows.items() if r["replaced"]] == []  # 멀쩡한 답을 막지 않는다
    assert all(not r["called"] for r in rows.values() if r["kind"] in ("minor", "absent"))  # 모델을 부르지 않고 막는다
    assert sum(r["called"] for r in rows.values()) == 30
