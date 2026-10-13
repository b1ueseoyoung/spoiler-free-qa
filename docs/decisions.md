# 결정과 확인한 사실

## 2026-10-13 (D1) LM Studio 연동

환경: LM Studio 0.4.25, `qwen/qwen3.6-35b-a3b`(MLX 4-bit), `text-embedding-kure-v1`, langchain-core 1.6.10, langchain-openai 1.7.1, openai 3.29.0, Python 3.12.14. 확인 명령은 `uv run sfqa smoke`.

- **thinking 끄기:** 요청 본문에 `reasoning_effort: "none"`을 `extra_body`로 넣으면 json_schema 응답이 content로 온다. reasoning은 0자였다.
- **길이 제한:** langchain-openai 1.7.1은 `max_tokens`를 `max_completion_tokens`라는 이름으로 보낸다. LM Studio는 이 이름을 지킨다(8로 두자 7토큰에서 `length`로 멈췄다).
- **잘린 응답:** json_schema를 붙이면 OpenAI SDK의 구조화 출력 경로를 타서, 잘린 응답이 `LengthFinishReasonError`로 온다. `LiveClient`가 이 예외를 잡아 `truncated`로 기록한다(`tests/test_llm.py`).
- **임베딩:** `check_embedding_ctx_length=False`면 입력이 문자열로 나간다. 차원은 1024. 비슷한 문장 쌍 0.871, 다른 문장 0.296.
- **속도:** 출력 약 300토큰에서 초당 약 90토큰. 호출마다 고정 지연이 약 2.5초 있다. 입력이 긴 실제 질문의 속도는 D6에 다시 잰다.
- **구조화 출력 방식(D1):** `with_structured_output` 대신 `response_format`(json_schema, strict)을 직접 붙이고, 앞에 붙은 `<think>…</think>`를 지운 뒤 직접 해석한다. 해석 실패와 잘림을 예외가 아닌 기록으로 남기기 위해서다.

## 2026-10-13 (D2) 원문 받기와 정제

확인 명령: `uv run sfqa fetch && uv run sfqa clean`, `uv run pytest tests/test_clean.py`.

- **출처:**
  - 위키문헌 `탁류`와 `탁류/제1장`~`탁류/제19장`. 각 장의 판(revid)과 sha256을 `data/raw/sources.json`에 고정했다.
  - 작품 페이지의 라이선스 틀은 `PD-old-50`이다.
  - 출전은 조선일보(1937.10.12~1938.5.17)로 적혀 있다.
- **User-Agent:** 기본값 `sfqa/0.1 (personal research project)`로 받았다. 위키미디어는 연락처를 넣으라고 권한다. 다시 받을 때는 `WIKI_USER_AGENT`에 연락처를 넣는다.
- **페이지 구조:**
  - 장 페이지는 `머리말` 틀, 제목줄 하나, 본문, `꼬리말` 틀로 되어 있다.
  - 「무정」과 달리 머리말에 줄거리 요약이 없다. 그래도 틀은 모두 지운다(장마다 머리말 1, 꼬리말 1).
- **정제 결과:**
  - 19개 장, 합계 416,937자. 장 길이는 4,688자(11장)~40,254자(10장)다.
  - 첫 제목줄이 19개 장 모두 목차의 장 제목과 같다.
  - 남은 마크업은 0이다. 중괄호 두 쌍(`{심청전}`, `{성층권(成層圈)의 연구(硏究)}`)은 원문의 책 제목 표시라 그대로 둔다.
- **완결 여부:** 제19장 꼬리말의 '다음'이 비어 있고, 마지막 문단이 장면을 맺으며 끝난다. 위키문헌 기준으로 완결본으로 본다.
- **×의 쓰임:** 검열 삭제만이 아니다. `××은행`처럼 이름을 가린 곳도 있다. 원문에는 그대로 두고, 검색용 텍스트에서만 지운다(계획대로).
- **청크 수 추정 수정:** 700자에 겹침 150자로 나누면 약 760개다. 계획의 약 300개보다 많지만, N별 색인을 메모리에서 만드는 방식은 그대로 쓸 수 있다.
