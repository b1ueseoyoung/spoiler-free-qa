# 회차 스포일러 없는 작품 Q&A

채만식 「탁류」(공유 저작물)를 읽는 독자가 "N장까지 읽었다"고 하면, 1~N장 원문에서만 찾아 답하고 근거 인용과 장 번호를 붙인다. 원문에 없으면 답하지 않는다. LangChain(Python)과 LM Studio 로컬 모델로 돌아가며, 원문과 질문은 이 맥 밖으로 나가지 않는다.

- 평가 기준: [docs/eval-spec.md](docs/eval-spec.md)
- 결정과 확인한 사실: [docs/decisions.md](docs/decisions.md)

## 실행

```bash
uv run pytest            # 모델 없이 도는 테스트
uv run sfqa smoke        # LM Studio(127.0.0.1:1234) 연결 점검
uv run sfqa fetch        # 위키문헌 원문 받기(sources.json의 판 고정)
uv run sfqa clean        # 틀·각주 제거 → data/clean
```
