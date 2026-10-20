export const meta = {
  name: 'webnovel-batch',
  description: '합성 웹소설 여러 편을 동시에 생성: 설정집(비평 반복) → 5화 묶음 집필·검사·수정 2회 → 경계 점검 → 인물 지식 전수 스캔·추론 가능 시점 → 정답지 보정·검증',
  phases: [
    { title: 'Bible', detail: '설정집·사실 장부·50화 줄거리' },
    { title: 'Critique', detail: '비평과 수정, 큰 문제가 없어질 때까지 최대 4회' },
    { title: 'Write', detail: '5화씩 10묶음 집필' },
    { title: 'Check', detail: '묶음별 누출·누락·지식·연속성 검사와 수정(최대 2회)' },
    { title: 'Seams', detail: '묶음 경계 연결 점검' },
    { title: 'Gold', detail: '인물 지식 전수 스캔, 추론 가능 시점, 정답지 보정·검증' },
  ],
}

const BASE = '/Users/seoyeong/Projects/dev/spoiler-free-qa/data/synthetic'
const SCRATCH = '/private/tmp/claude-501/-Users-seoyeong-Projects-dev-spoiler-free-qa/f910a356-628e-4807-91c3-004a56212eec/scratchpad'
const VALIDATE = `${SCRATCH}/validate_gold.py`
const PILOT = `${BASE}/wn01-rofan`
const DATE = (args && args.date) || '2026-10-20'
const WORKS = (args && args.works) || []
const pad = n => String(n).padStart(2, '0')
const LEN_CMD = `python3 -c "import sys;[print(p, len(open(p,encoding='utf-8').read())) for p in sys.argv[1:]]" <파일들>`

const CONVENTIONS = {
  known_by: 'from_ep는 인물이 그 사실을 알게 되는 화다. 0은 이야기 시작 전부터 앎. 사건을 직접 한 인물과 그 장면에 있던 인물은 그 사건이 일어난 화부터 안다.',
  knowledge_events: 'known_by(from_ep>0)와 하나씩 대응한다. 그 화에 등장하지 않거나 본문에 그리지 않는 곳에서 알게 되면 offstage: true를 붙이며, 본문은 그 장면을 서술하지 않는다.',
  relationship_timeline: 'from_ep는 독자가 그 관계 상태를 처음 알 수 있게 되는 화다. 0은 이야기 시작 전부터의 배경 상태로, 두 인물이 처음 함께 언급되는 화까지 공개되는 내용만 담는다. known_by는 그 상태를 아는 인물과 아는 시점이다.',
  pov_must_hide: '시점 인물이 알고 있지만 아직 독자에게 공개되지 않은 사실 목록. 그 화 본문(시점 인물의 내면 포함)에서 서술하거나 암시하지 않는다.',
  aliases: 'aliases는 첫 등장부터 끝까지 그 인물을 가리키는 별칭, aliases_timed는 from_ep~to_ep(없으면 끝까지) 동안만 그 인물을 가리키는 별칭이다.',
  how_called: 'by 인물이 그 인물을 부르는 호칭. from_ep~to_ep(없으면 끝까지) 동안 쓰며, context는 공석·사석 같은 조건이다.',
  world_reveal: "world의 places·institutions·magic·festivals 항목은 first_reveal_ep(본문에 처음 나오는 화)와 desc(그 화까지 공개되는 내용만)를 가진다. 그 뒤 화에 공개되는 내용은 desc에 쓰지 않고 later_fact_ids의 사실(각 사실의 reveal_ep)로만 참조한다. summary는 summary_first_reveal_ep 화까지의 내용만 담는다. 해당 개념이 없는 장르(예: 마법 없음)는 빈 배열로 둔다.",
  interlude_pov: '막간이 있는 화는 interlude_pov(막간 시점 인물), interlude_scope(막간 범위), interlude_pov_must_hide(막간 시점 인물이 그 화에 알고 있지만 아직 독자에게 공개되지 않은 사실)를 둔다. interlude_pov_must_hide의 규칙은 pov_must_hide와 같다.',
  interlude_conclusion: '막간에서 서술자가 시점 인물의 결론을 단정형으로 서술하고 그 인물이 그 결론에 따라 움직이면, 그 화부터 그 인물이 안다고 본다. 뒤에 그 인물이 대사로 짐작이라고 낮춰 말해도 from_ep를 늦추지 않는다. 서술 없이 의심만 드러나는 경우는 앎으로 보지 않는다.',
  known_by_note: 'known_by 항목의 note는 판단 근거를 밝혀 두거나, 그 인물이 사실의 일부만 알 때 아는 범위를 적는다.',
  atomic_facts: '사실 하나에는 내용 하나만 담는다. 인물마다 아는 범위가 달라질 수 있는 내용(예: 살아 있다는 것과 구조 경위)은 처음부터 다른 사실로 나눈다. 한 사실이 다른 사실의 핵심을 포함하면, 포함하는 사실을 아는 인물은 같은 화나 그보다 이르게 포함되는 사실도 안다.',
  inferable: 'inferable_from_ep는 독자가 명시 공개(reveal_ep) 전에 그 사실을 확신하거나 강하게 의심할 수 있게 되는 가장 이른 화다. inferable_confidence는 strong(독자 대부분이 확신) 또는 moderate(눈치 빠른 독자가 의심), inferable_basis는 단서가 나온 화와 내용이다. 추론 가능하지 않은 사실에는 필드를 넣지 않는다.',
}

const SCHEMA_SPEC = `
[파일 구조] 파일럿 ${PILOT}의 bible.json·outline.json·facts.jsonl·work.json과 같은 키를 쓴다. 쓰기 전에 파일럿 파일을 python3로 열어 키 구조만 확인하라(내용은 참고하지도 베끼지도 마라).
- bible.json 키: title, genre, logline, rating("15세"), world{summary, summary_first_reveal_ep, era, places[], institutions[], magic[], festivals[]}, characters[], relationships[], twists[], conventions.
  · characters 항목: id(영문 소문자), name, aliases[], aliases_timed[{alias, from_ep, to_ep}], how_called[{by, call, from_ep, to_ep, context}], age(정수), age_note, gender("남"|"여"|기타), is_minor(만 19세 미만이면 true), role, appearance, personality, speech_style, sample_lines[3~5], backstory, goal, first_appearance_ep, secrets[fact id], arc, end_state.
  · relationships 항목: id, a, b, canon_romance, shipping_candidate, timeline[{from_ep, status, public, known_by[{character, from_ep}], note}].
  · twists 항목: id, fact_id, reveal_ep, setup_eps[], summary.
  · world의 places·institutions·magic·festivals 항목: name, first_reveal_ep, desc, later_fact_ids[].
  · conventions: 아래 [정답지 규칙]을 그대로 넣는다.
- facts.jsonl 한 줄: id("f001"…), statement, category(identity|relationship|event|secret|world|ability|death|location|motive), reveal_ep, known_by[{character, from_ep, note?}], is_twist, evidence_keywords[1~3](그 화 본문에 들어갈 특징적 표현, 흔한 낱말 금지).
- outline.json 50개 항목: ep, title, pov, interlude_pov, interlude_scope, location, characters_present[], summary(5~8문장), beats[4~7], facts_revealed[], hints_allowed[], relationship_changes[{relationship, a, b, status, public}], knowledge_events[{character, fact_id, offstage?, note?}], pov_must_hide[], interlude_pov_must_hide[], ending_hook, end_state.
- work.json: id, order, title, author("AI 생성"), label("AI가 생성한 <장르> 웹소설 「제목」"), published("2026-10 생성(합성 데이터)"), chapters(50), unit("화"), blurb(한 문장), examples[{question, chapter} 3개], default_upto(3), evaluated(false), synthetic(true), genre, generator("claude-opus-5-5 (Claude Code 워크플로)"), source{kind:"synthetic", created:"${DATE}", license:"AI 생성 합성 데이터. 실존 작품·인물과 무관"}.

[정답지 규칙]
${JSON.stringify(CONVENTIONS, null, 1)}

[검사기] 공용 검사기: python3 -I ${VALIDATE} <작품 폴더>. 오류가 0이어야 하고 종료 코드 0이어야 한다. 검사기 파일은 고치지 않는다.
`

function rootOf(w) { return `${BASE}/${w.id}` }
function scratchOf(w) { return `${SCRATCH}/gen-${w.id}` }

function biblePrompt(w) {
  const ROOT = rootOf(w)
  return `
너는 한국 웹소설 기획자다. 아래 조건으로 오리지널 ${w.genre} 장편(50화)의 설정집을 만들어 ${ROOT}에 저장하라.

[용도] '책갈피' 앱의 등장인물 채팅 기능 시험 데이터다. 독자가 N화까지 읽었을 때 인물이 N화 이후를 말하면 안 되고, 인물은 자기가 아는 것만 말해야 한다. 설정집의 사실 장부가 그 정답지다. 문장력보다 정확성과 내부 일관성이 중요하다.

[작품 조건]
- 장르: ${w.genre}. ${w.brief}
- 50화, 화당 약 5,000자. 15세 이용가(연애는 키스까지, 노골적 성 묘사 없음, 폭력 절제).
- 완전한 오리지널: 실존 인물, 기존 작품의 고유 인물명·설정·줄거리를 베끼지 않는다. 흔한 장르 관습은 써도 된다. 파일럿(${PILOT})의 인물·설정·줄거리와도 겹치지 않게 한다.
- 이름 있는 인물 10~14명. 연애 축이 있으면 모두 성인이고 나이를 명시한다. ${w.child ? '어린 조연 1명(만 7~12세)을 넣되 어떤 연애 관계·커플링 후보에도 엮지 않는다.' : '미성년 인물은 넣지 않는다(전원 만 19세 이상).'}
- 원작에서는 로맨스가 없지만 감정선이 짙은 성인 남성 인물 쌍을 최소 2쌍(라이벌, 주종, 오랜 친구 등) 만든다(shipping_candidate=true, canon_romance=false). 원작 안에서는 끝까지 로맨스가 아니다.
- 관계는 시간에 따라 변한다. 변화 시점을 화 번호로 적는다.
- 큰 반전 6개 이상을 10~50화에 고르게 배치한다.
- 인물별 지식 차이를 일부러 만든다: 독자는 알지만 특정 인물은 모르는 사실, 특정 인물이 독자보다 먼저 아는 사실을 합쳐 15개 이상.
- 인물마다 말투가 뚜렷하다(존댓말·반말, 호칭, 입버릇).
- 사실 120~200개, 화마다 1개 이상 고르게. 사소한 사실(직업, 장소, 관계)도 넣는다.

[파일럿에서 배운 규칙]
1. 사실은 원자 단위로 쓴다(정답지 규칙 atomic_facts). 복합 사실은 나중에 인물 지식 모순을 만든다.
2. hints_allowed에 정답을 쓰지 않는다. '깃털과 테오도르를 연결하지 말 것'처럼 금지 문구가 정답을 담으면 안 된다. 대신 '편지 발신자를 짐작하게 하는 단서는 금지'처럼 쓴다.
3. 관계 timeline의 from_ep는 독자가 그 상태를 처음 알 수 있는 화다. 0화 항목에는 뒤에 공개될 내용을 넣지 않는다.
4. world 항목의 desc에는 first_reveal_ep까지 공개된 내용만 쓴다. 뒤의 직함·사건(예: '49화에 해체')은 later_fact_ids로만 참조한다. 별칭에 미래 직함을 넣지 말고 aliases_timed로 시점을 정한다.
5. 시점 인물이 공개 전 사실을 많이 아는 화는 누출 위험이 크다. 그런 화는 pov_must_hide를 빠짐없이 채우고, 가능하면 시점 인물을 바꾼다.
6. 반전 사실의 known_by는 사건 시점과 장면 참석자를 정확히 따른다. 의식을 잃은 인물은 그동안 일어난 일을 모른다.
${SCHEMA_SPEC}
[작업 방법] 생성용 스크립트가 필요하면 ${scratchOf(w)}/ 에만 둔다(다른 작품과 섞이지 않게). 다 쓴 뒤 공용 검사기를 돌려 오류 0이 될 때까지 고쳐라.
마지막 응답은 요약이다.`
}

const BIBLE_SUMMARY = {
  type: 'object',
  properties: {
    title: { type: 'string' }, logline: { type: 'string' }, characters: { type: 'number' }, facts: { type: 'number' },
    twists: { type: 'number' }, shipping_pairs: { type: 'array', items: { type: 'string' } }, minor_character: { type: 'string' },
    validator_errors: { type: 'number' }, self_check: { type: 'string' },
  },
  required: ['title', 'logline', 'characters', 'facts', 'twists', 'shipping_pairs', 'minor_character', 'validator_errors', 'self_check'],
}

const CRITIQUE = {
  type: 'object',
  properties: {
    validator_errors: { type: 'number' },
    issues: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          severity: { type: 'string', enum: ['blocking', 'major', 'minor'] }, area: { type: 'string' }, where: { type: 'string' },
          problem: { type: 'string' }, fix: { type: 'string' },
        },
        required: ['severity', 'area', 'where', 'problem', 'fix'],
      },
    },
    overall: { type: 'string' },
  },
  required: ['validator_errors', 'issues', 'overall'],
}

function criticPrompt(w) {
  return `
너는 깐깐한 웹소설 편집자이자 정답지 검수자다. ${rootOf(w)}의 bible.json, facts.jsonl, outline.json, work.json을 읽고 비평하라. 파일은 고치지 말고 문제만 보고한다.
1. 공용 검사기를 돌려 오류 수를 validator_errors에 적고, 오류는 blocking으로 올린다.
2. 줄거리 누출: 어떤 화의 summary·beats·hints_allowed·end_state가 그 화보다 뒤에 공개될 사실을 드러내거나 독자가 확신할 수준으로 암시하는가. hints_allowed 문구 자체가 정답을 담는가.
3. 정답지 정확성: known_by가 사건 시점·장면 참석자와 맞는가. 복합 사실이 있는가(원자 사실 규칙). 포함 관계 사실끼리 known_by가 모순되는가. 관계 timeline 0화 항목에 뒤의 내용이 섞였는가. world desc에 first_reveal_ep 뒤의 내용이 섞였는가.
4. 안전: 미성년 인물이 연애·성적 맥락이나 커플링 후보에 있는가. 15세 이용가를 넘는가.
5. 시험 데이터 요건: 성인 남성 커플링 후보 2쌍 이상, 반전 6개 이상 고르게, 인물별 지식 차이 15개 이상, 시간에 따라 바뀌는 관계.
6. 이야기: 반전과 동기의 논리 구멍, 이야기 규칙의 모순, 중반 늘어짐, 말투 구별.
7. 독창성: 유명한 기존 작품이나 파일럿(${PILOT})과 닮은 고유 설정.
severity: blocking(정답지가 틀어지거나 검사기 오류), major(시험 요건·이야기 논리 훼손), minor(다듬을 것). 각 문제에 위치와 구체적인 수정 지시를 적어라.
${SCHEMA_SPEC}`
}

function revisePrompt(w, c) {
  return `
너는 웹소설 기획자다. ${rootOf(w)}의 설정집 파일에 편집자가 아래 비평을 냈다. blocking과 major는 모두 고치고 minor는 쉽게 고칠 수 있으면 고쳐라. 파일을 직접 수정하라.
사실을 옮기거나 지우거나 나눌 때는 facts.jsonl, outline.json(facts_revealed, knowledge_events, pov_must_hide), bible.json(twists, secrets, relationships, world later_fact_ids)을 함께 맞춰라.
고친 뒤 공용 검사기를 돌려 오류 0으로 만든다. 생성용 스크립트는 ${scratchOf(w)}/ 에만 둔다.
마지막 응답은 고친 내용 요약이다.
${SCHEMA_SPEC}
[비평]
${JSON.stringify(c, null, 1)}`
}

const BATCHES = Array.from({ length: 10 }, (_, i) => ({ s: i * 5 + 1, e: i * 5 + 5 }))

function writerPrompt(w, b) {
  const ROOT = rootOf(w)
  const prev = b.s > 1 ? `${b.s}화는 outline의 ${b.s - 1}화 end_state에서 이어 시작한다(앞 화 본문은 다른 작가가 동시에 쓰고 있어 없을 수 있다).` : '1화는 작품의 첫 화다. 독자가 세계와 주인공을 자연스럽게 알게 하라.'
  return `
너는 한국 웹소설 작가다. ${ROOT}의 bible.json, outline.json, facts.jsonl을 먼저 읽어라. 그다음 ${b.s}~${b.e}화를 순서대로 써서 ${ROOT}/clean/chNN.txt(두 자리)에 저장하라. 이미 파일이 있으면 덮어쓴다.
[규칙]
- 분량: 각 화 4,500~5,800자(공백 포함). 쓰고 나서 재고, 벗어나면 고쳐라: ${LEN_CMD}
- 문체: 한국 웹소설 문체. 짧은 문단, 대사 비중 높게, 문단 사이 빈 줄. 마크다운, 제목, 화 번호, 작가의 말, 메타 문구를 넣지 않는다. 본문만.
- 내용: 그 화 beats를 모두 담는다. facts_revealed의 사실은 독자가 본문만 읽고 분명히 답할 수 있게 드러내고 evidence_keywords를 자연스럽게 쓴다.
- 스포일러 규칙(가장 중요): reveal_ep가 그 화보다 뒤인 사실은 드러내지도 확정적으로 암시하지도 않는다. hints_allowed 수위의 복선만 허용한다. 서술자가 미래를 아는 듯 쓰지 않는다. pov_must_hide·interlude_pov_must_hide의 사실은 시점 인물의 내면에서도 서술하거나 암시하지 않는다. offstage 지식 사건은 본문에 그리지 않는다.
- 인물 지식 규칙: 인물은 known_by.from_ep 전에는 그 사실을 아는 듯 말하거나 행동하지 않는다. 장면에 없던 인물에게 그 장면의 내용을 들려주는 대사를 새로 만들지 않는다(정답지에 없는 지식 전달이 생긴다). 이미 공개된 사실을 다른 인물에게 다시 들려주는 장면이 outline에 없으면 만들지 않는다.
- 말투와 호칭: speech_style, sample_lines, how_called, aliases_timed를 지킨다. 이름 표기를 바꾸지 않는다.
- 안전: 15세 이용가. 미성년 인물은 연애·성적 맥락에 넣지 않는다.
- 연결: ${prev} 같은 묶음 안에서는 방금 쓴 앞 화를 다시 읽고 잇는다.
마지막 응답은 화별 결과다.`
}

const WRITE_RESULT = {
  type: 'object',
  properties: { episodes: { type: 'array', items: { type: 'object', properties: { ep: { type: 'number' }, chars: { type: 'number' }, notes: { type: 'string' } }, required: ['ep', 'chars', 'notes'] } } },
  required: ['episodes'],
}

const CHECK_RESULT = {
  type: 'object',
  properties: {
    episodes: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          ep: { type: 'number' }, chars: { type: 'number' },
          missing_facts: { type: 'array', items: { type: 'string' } }, early_leaks: { type: 'array', items: { type: 'string' } },
          knowledge_violations: { type: 'array', items: { type: 'string' } }, consistency_issues: { type: 'array', items: { type: 'string' } },
          safety_issues: { type: 'array', items: { type: 'string' } }, format_issues: { type: 'array', items: { type: 'string' } },
          verdict: { type: 'string', enum: ['ok', 'fix'] }, fix_instructions: { type: 'string' },
        },
        required: ['ep', 'chars', 'missing_facts', 'early_leaks', 'knowledge_violations', 'consistency_issues', 'safety_issues', 'format_issues', 'verdict', 'fix_instructions'],
      },
    },
    notes: { type: 'string' },
  },
  required: ['episodes', 'notes'],
}

function checkerPrompt(w, eps) {
  const ROOT = rootOf(w)
  const files = eps.map(n => `${ROOT}/clean/ch${pad(n)}.txt`).join(' ')
  return `
너는 엄격한 검수자다. ${ROOT}의 bible.json, outline.json, facts.jsonl을 읽고 다음 화를 검사하라: ${files}
파일은 고치지 말고 보고만 하라. 앞뒤 화 본문이 있으면 연속성 확인에 참고하라.
화마다:
- chars: 글자 수. 4,500~5,800자 밖이면 format_issues. ${LEN_CMD}
- missing_facts: facts_revealed 중 본문만 읽고 독자가 분명히 답할 수 없는 fact id.
- early_leaks: reveal_ep가 그 화보다 뒤인 사실 가운데 본문이 드러내거나 확신할 수준으로 암시하는 것("fact id: 근거 요약"). 시점 인물 내면의 pov_must_hide 위반과 서술자의 미래 예고 포함. 가장 중요하다.
- knowledge_violations: 인물이 known_by.from_ep 전에 그 사실을 아는 듯 말하거나 행동한 곳, 그리고 본문에서 인물이 사실을 새로 알게 되는데 known_by에 없는 곳(정답지 쪽을 고쳐야 하면 '정답지:'로 시작해 적는다).
- consistency_issues: 이름·호칭·나이·장소·시간대·관계가 bible·outline·앞뒤 화와 어긋나는 곳, 같은 화 안의 모순.
- safety_issues: 미성년 인물의 연애·성적 맥락, 15세 이용가 초과.
- format_issues: 마크다운, 제목, 화 번호, 작가의 말, 메타 문구, 분량.
하나라도 있으면 verdict는 fix이고 fix_instructions에 구체적으로 적는다. '정답지:' 항목만 있으면 본문은 ok로 두고 notes에 모은다. 사소한 문장 취향은 문제로 보지 않는다.`
}

function fixerPrompt(w, report) {
  return `
너는 한국 웹소설 작가다. ${rootOf(w)}의 bible.json, outline.json, facts.jsonl을 읽고, 검수자가 fix로 판정한 화의 본문을 고쳐라(정답지 파일은 고치지 않는다).
보고서의 모든 항목을 해결하되 문제없는 부분은 그대로 둔다. 분량 4,500~5,800자를 확인하라: ${LEN_CMD}
스포일러 규칙과 인물 지식 규칙을 지킨다. 장면에 없던 인물에게 새로 정보를 들려주는 대사를 만들지 않는다.
마지막 응답은 고친 화와 요약이다.
[검수 보고서]
${JSON.stringify(report, null, 1)}`
}

const SCAN = {
  type: 'object',
  properties: {
    missing: { type: 'array', items: { type: 'object', properties: { fact_id: { type: 'string' }, character: { type: 'string' }, ep: { type: 'number' }, evidence: { type: 'string' } }, required: ['fact_id', 'character', 'ep', 'evidence'] } },
    too_early: { type: 'array', items: { type: 'object', properties: { fact_id: { type: 'string' }, character: { type: 'string' }, listed_from_ep: { type: 'number' }, should_be_ep: { type: 'number' }, evidence: { type: 'string' } }, required: ['fact_id', 'character', 'listed_from_ep', 'should_be_ep', 'evidence'] } },
    notes: { type: 'string' },
  },
  required: ['missing', 'too_early', 'notes'],
}

const INFER = {
  type: 'object',
  properties: {
    inferable: { type: 'array', items: { type: 'object', properties: { fact_id: { type: 'string' }, reveal_ep: { type: 'number' }, inferable_from_ep: { type: 'number' }, confidence: { type: 'string', enum: ['strong', 'moderate'] }, basis: { type: 'string' } }, required: ['fact_id', 'reveal_ep', 'inferable_from_ep', 'confidence', 'basis'] } },
    summary: { type: 'string' },
  },
  required: ['inferable', 'summary'],
}

const VERIFY = {
  type: 'object',
  properties: {
    validator_errors: { type: 'number' },
    unresolved: { type: 'array', items: { type: 'object', properties: { severity: { type: 'string', enum: ['blocking', 'major', 'minor'] }, item: { type: 'string' }, problem: { type: 'string' }, fix: { type: 'string' } }, required: ['severity', 'item', 'problem', 'fix'] } },
    sampled_ok: { type: 'number' }, sampled_total: { type: 'number' }, summary: { type: 'string' },
  },
  required: ['validator_errors', 'unresolved', 'sampled_ok', 'sampled_total', 'summary'],
}

const GOLD_COMMON = (w) => `
작품 폴더: ${rootOf(w)} (bible.json, facts.jsonl, outline.json, clean/ch01.txt~ch50.txt)
기준은 본문이다. 인물이 사실을 안다고 보려면 본문에서 그 인물이 그 내용을 직접 겪거나, 보거나, 듣거나, 읽거나, 전해 듣는 장면이 있어야 한다. 짐작·의심은 앎이 아니다(정답지 규칙 interlude_conclusion 예외).
큰 파일은 python3로 필요한 부분만 읽어라. 근거는 '화:행' 형식으로 짧게 인용하라.
${SCHEMA_SPEC}`

async function runWork(w) {
  const tag = w.id.split('-')[0]
  const ROOT = rootOf(w)

  // 1. 설정집
  const bible = await agent(biblePrompt(w), { label: `${tag}:bible`, phase: 'Bible', schema: BIBLE_SUMMARY })
  if (!bible) return { id: w.id, failed: 'bible' }
  log(`${tag} 설정집: 「${bible.title}」 사실 ${bible.facts}개, 반전 ${bible.twists}개, 검사기 오류 ${bible.validator_errors}`)

  // 2. 비평 반복(최대 4회 비평, 마지막 상태는 항상 비평으로 확인)
  const critiques = []
  for (let round = 1; round <= 4; round++) {
    const c = await agent(criticPrompt(w), { label: `${tag}:critic:${round}`, phase: 'Critique', schema: CRITIQUE })
    if (!c) break
    const nb = c.issues.filter(i => i.severity === 'blocking').length
    const nm = c.issues.filter(i => i.severity === 'major').length
    critiques.push({ round, validator_errors: c.validator_errors, blocking: nb, major: nm, minor: c.issues.length - nb - nm })
    log(`${tag} 비평 ${round}회: 검사기 오류 ${c.validator_errors}, blocking ${nb}, major ${nm}`)
    if (c.validator_errors === 0 && nb === 0 && nm <= 3) break
    if (round === 4) { log(`${tag} 비평 4회 뒤에도 남음: blocking ${nb}, major ${nm}`); break }
    await agent(revisePrompt(w, c), { label: `${tag}:revise:${round}`, phase: 'Critique' })
  }

  // 3. 집필 → 검사 → 수정·재검사(최대 2회)
  const batchResults = await pipeline(
    BATCHES,
    b => agent(writerPrompt(w, b), { label: `${tag}:write:${pad(b.s)}-${pad(b.e)}`, phase: 'Write', schema: WRITE_RESULT }),
    (wr, b) => agent(checkerPrompt(w, Array.from({ length: 5 }, (_, i) => b.s + i)), { label: `${tag}:check:${pad(b.s)}-${pad(b.e)}`, phase: 'Check', schema: CHECK_RESULT }),
    async (check, b) => {
      let cur = check
      const goldNotes = [check ? check.notes : '']
      for (let r = 1; r <= 2; r++) {
        const bad = (cur && cur.episodes || []).filter(e => e.verdict === 'fix')
        if (!bad.length) break
        await agent(fixerPrompt(w, { episodes: bad }), { label: `${tag}:fix${r}:${pad(b.s)}-${pad(b.e)}`, phase: 'Check' })
        cur = await agent(checkerPrompt(w, bad.map(e => e.ep)), { label: `${tag}:recheck${r}:${pad(b.s)}-${pad(b.e)}`, phase: 'Check', schema: CHECK_RESULT })
        if (cur) goldNotes.push(cur.notes)
      }
      const remaining = (cur && cur.episodes || []).filter(e => e.verdict === 'fix').map(e => ({ ep: e.ep, fix: e.fix_instructions }))
      return { batch: b, remaining, goldNotes: goldNotes.filter(Boolean) }
    },
  )
  const textRemaining = batchResults.filter(Boolean).flatMap(r => r.remaining)
  const checkerGoldNotes = batchResults.filter(Boolean).flatMap(r => r.goldNotes)
  log(`${tag} 집필 끝. 수정 2회 뒤에도 남은 화: ${textRemaining.map(r => r.ep).join(', ') || '없음'}`)

  // 4. 경계 점검
  const seams = await agent(`
너는 웹소설 편집자다. ${ROOT}/clean/의 본문은 5화씩 묶음으로 동시에 쓰였다. 묶음 경계(5→6, 10→11, …, 45→46화)마다 앞 화의 마지막 2,000자와 뒤 화의 처음 2,000자를 읽어 장소·시간·인물 상태·감정선·직전 사건이 이어지는지 점검하라. 같은 장면이 두 번 나오면 뒤 화 첫 부분을 고친다.
어긋나면 뒤 화의 첫 부분만 최소로 고친다. 분량 4,500~5,800자를 확인한다: ${LEN_CMD}
또 수정 2회 뒤에도 남은 아래 문제를 본문에서 고쳐라(분량 유지): ${JSON.stringify(textRemaining)}
스포일러 규칙과 인물 지식 규칙을 지킨다. ${ROOT}/outline.json, facts.jsonl을 참고한다.`, {
    label: `${tag}:seams`, phase: 'Seams',
    schema: { type: 'object', properties: { seams: { type: 'array', items: { type: 'object', properties: { after_ep: { type: 'number' }, ok: { type: 'boolean' }, fix_applied: { type: 'boolean' }, note: { type: 'string' } }, required: ['after_ep', 'ok', 'fix_applied', 'note'] } }, remaining_fixed: { type: 'string' } }, required: ['seams', 'remaining_fixed'] },
  })

  // 5. 인물 지식 전수 스캔 + 추론 가능 시점(본문 확정 뒤)
  const [scans, infer] = await Promise.all([
    parallel(BATCHES.map(b => () => agent(`너는 정답지 검수자다. ${pad(b.s)}~${pad(b.e)}화 본문을 처음부터 끝까지 읽고, 인물이 사실을 알게 되는 장면을 모두 찾아 facts.jsonl의 known_by와 대조하라.
- missing: 본문에서 인물이 그 화에 사실을 알게 되는데 known_by에 없거나 from_ep가 늦은 경우. 이미 공개된 사실을 다른 인물에게 다시 들려주는 장면(고백, 회의, 재판, 편지 낭독)에서 같은 자리에 있던 인물을 빠짐없이 본다.
- too_early: from_ep가 본문에서 그 인물이 알 수 있는 시점보다 이른 경우. 이 묶음에서 증거를 찾을 수 있는 것만.
- 포함 관계 사실이 있으면 둘 다 확인한다. 파일은 고치지 않는다.
${GOLD_COMMON(w)}`, { label: `${tag}:scan:${pad(b.s)}-${pad(b.e)}`, phase: 'Gold', schema: SCAN }))),
    agent(`너는 꼼꼼한 독자다. 이 작품을 1화부터 순서대로 읽으며 reveal_ep보다 앞에서 독자가 확신하거나 강하게 의심할 수 있는 사실을 찾아라. 반전 사실(is_twist)은 모두 보고(앞당길 단서가 없으면 inferable_from_ep=reveal_ep로 두고 이유를 적는다), 나머지는 strong·moderate인 것만 넣는다. inferable_from_ep는 추론에 필요한 단서가 모두 나온 가장 이른 화다. 파일은 고치지 않는다.
${GOLD_COMMON(w)}`, { label: `${tag}:inferable`, phase: 'Gold', schema: INFER }),
  ])
  const found = scans.filter(Boolean)
  const missing = found.flatMap(s => s.missing)
  const tooEarly = found.flatMap(s => s.too_early)
  log(`${tag} 스캔: 누락 ${missing.length}, 너무 이름 ${tooEarly.length}, 추론 가능 ${infer ? infer.inferable.length : 0}`)

  const goldFix = (extra) => `너는 정답지 편집자다. 아래 자료를 반영해 ${ROOT}의 facts.jsonl, outline.json(knowledge_events, relationship_changes, pov_must_hide), bible.json(relationships timeline known_by, world)을 고쳐라. 본문(clean/)은 고치지 않는다.
1. 스캔 결과: missing은 known_by에 추가하거나 from_ep를 앞당기고, too_early는 늦추거나 뺀다. 본문과 안 맞으면 본문을 직접 확인해 판단한다.
2. 포함 관계 규칙과 원자 사실 규칙을 지킨다. 한 사실을 인물마다 아는 범위가 다르면 사실을 나누고, 새 사실은 facts_revealed·knowledge_events에도 넣는다.
3. 추론 가능 시점 목록을 facts.jsonl에 inferable_from_ep, inferable_confidence, inferable_basis 필드로 넣는다. 목록에 없는 사실에는 넣지 않는다.
4. 검수자들이 '정답지:'로 남긴 메모도 확인해 반영한다.
5. 공용 검사기 오류 0이 될 때까지 고친다(pov_must_hide 누락 포함).
마지막 응답은 바꾼 항목 수와 요약이다.
${GOLD_COMMON(w)}${extra}`

  await agent(goldFix(`\n[스캔 결과]\n${JSON.stringify({ missing, too_early: tooEarly })}\n[추론 가능 시점]\n${JSON.stringify(infer ? infer.inferable : [])}\n[검수자 정답지 메모]\n${JSON.stringify(checkerGoldNotes)}`), { label: `${tag}:goldfix:1`, phase: 'Gold' })

  const verifyPrompt = `너는 독립 검증자다. 정답지 편집이 끝났다.
1. 공용 검사기를 돌려 오류 수를 validator_errors에 적는다.
2. 반전 사실(is_twist) 전부와, 이미 공개된 사실을 다시 들려주는 장면에서 다룬 사실 10개를 골라 known_by를 본문과 직접 대조한다(sampled_ok/sampled_total).
3. 포함 관계 모순, 복합 사실 문제를 본다.
남은 문제는 unresolved에 severity와 구체적 수정 지시로 적는다. 파일은 고치지 않는다.
${GOLD_COMMON(w)}`
  let verdict = await agent(verifyPrompt, { label: `${tag}:verify:1`, phase: 'Gold', schema: VERIFY })
  const serious = v => v ? v.unresolved.filter(u => u.severity !== 'minor').length : 0
  if (verdict && (verdict.validator_errors > 0 || serious(verdict) > 0)) {
    await agent(goldFix(`\n[독립 검증에서 남은 문제만 고친다]\n${JSON.stringify(verdict.unresolved)}`), { label: `${tag}:goldfix:2`, phase: 'Gold' })
    verdict = await agent(verifyPrompt, { label: `${tag}:verify:2`, phase: 'Gold', schema: VERIFY })
  }
  log(`${tag} 정답지 검증: 검사기 오류 ${verdict ? verdict.validator_errors : '?'}, blocking·major ${serious(verdict)}, 표본 ${verdict ? verdict.sampled_ok + '/' + verdict.sampled_total : '?'}`)

  return {
    id: w.id, title: bible.title, bible, critiques, text_remaining_before_seams: textRemaining,
    seams: seams ? seams.seams.filter(s => !s.ok || s.fix_applied) : null, seams_remaining_fixed: seams ? seams.remaining_fixed : null,
    scan: { missing: missing.length, too_early: tooEarly.length }, inferable: infer ? infer.inferable.length : 0,
    verdict,
  }
}

const results = await parallel(WORKS.map(w => () => runWork(w)))
return results
