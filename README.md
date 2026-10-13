# 회차 스포일러 없는 작품 Q&A

채만식 「탁류」(공유 저작물)를 읽는 독자가 "N장까지 읽었다"고 하면, 1~N장 원문에서만 찾아 답하고 근거 인용과 장 번호를 붙인다. 원문에 없으면 답하지 않는다.

- 사용 도구: LangChain(Python), LM Studio 로컬 모델(`qwen/qwen3.6-35b-a3b`, 임베딩 `text-embedding-kure-v1`)
- 원문과 질문은 이 맥 밖으로 나가지 않는다.
- 배경: 공저 논문 PA-RAG(진행도 인식 검색 증강 생성을 활용한 조건부 스포일러 탐지)를 작게 구현했다.

```text
질문 + N
  → N장 이하 청크에서만 검색(k=18)
  → 가드 프롬프트("독자는 N장까지 읽었다, 발췌 밖 지식 금지")
  → LM Studio(json_schema)
  → 코드 검증
      · 인용이 1~N장 원문에 글자 그대로 있는가
      · N장 뒤에 처음 나오는 인물이 답에 있는가
  → 답 + 인용(장 번호), 또는 "N장까지는 나오지 않습니다."
```

## 문서

- [평가 기준](docs/eval-spec.md): 만들기 전에 정했다. 바꾼 기록도 여기에 있다.
- [결정과 확인한 사실](docs/decisions.md): 날짜별 기록
- [한계](docs/limitations.md)
- [평가 결과](#평가-결과): 아래

## 실행

```bash
uv run pytest                                   # 모델 없이 도는 테스트(카세트 e2e 포함)
uv run sfqa smoke                               # LM Studio(127.0.0.1:1234) 연결 점검
uv run sfqa fetch && uv run sfqa clean          # 위키문헌 원문 받기(판 고정) → 정제
uv run sfqa gold-check                          # 정답지를 모델 없이 검사
uv run sfqa index                               # 청크 임베딩 색인(data/index, 저장소 밖)
uv run sfqa retrieval-eval --method dense       # 검색만 따로 재기(recall@k, L4)
uv run sfqa ask --upto 5 "초봉이가 일하는 곳은 어디인가?"
uv run sfqa eval-run --split test --max-calls 200
uv run sfqa eval-score runs/<실행 폴더>
```

- LM Studio에서 위 두 모델을 올리고 서버를 켜 둔다.
- 모델 서버 주소는 loopback(127.0.0.1, localhost, ::1)만 받는다.
- 평가 실행은 모델 응답과 질문 임베딩을 `eval/cassettes/`에 기록한다. 같은 요청은 다시 부르지 않고, `--replay`면 네트워크 없이 기록만 쓴다.

## 사람이 할 일

AI가 만든 정답지와 자동 판정은 아직 사람이 검토하지 않았다. 아래 두 단계를 마치면 결과를 "사람 검토 후" 값으로 다시 낸다.

1. **정답지 검토:**
   - `eval/gold-review.csv`의 판정 칸에 ok, 수정, 삭제를 적는다.
   - 고칠 것은 `eval/questions.jsonl`에서 고치고 `review`를 `human-reviewed`로 바꾼다.
   - 정답 키워드는 `A|B`로 대안을 줄 수 있다.
   - 고친 뒤 `uv run sfqa gold-check`
2. **누출 의심 검토:**
   - test 실행 폴더의 `review.csv`에서 1순위 행(누출 플래그, B·C인데 답함, 인용 실패)의 `human_outcome`과 `human_leak`(예/아니오)를 채운다.
   - 그다음 `uv run sfqa eval-score runs/<test 실행 폴더> --with-review`

## 평가 결과

(D10에 채운다)
