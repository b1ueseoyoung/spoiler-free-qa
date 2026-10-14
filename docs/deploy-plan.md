# 공개 데모 배포 계획 (Claude API + AWS Lambda, 2026-10-14)

## 목표와 완료 기준

- **목표:** 웹 데모를 공개 주소로 연다.
  - 생성 모델은 Claude API를 쓰고, 앱은 Docker 이미지로 AWS Lambda에 올린다.
  - 이 맥의 로컬 경로(LM Studio + qwen3.6)는 기본값으로 그대로 둔다.
- **완료 기준:**
  - 공개 주소에서 질문하면 답, 인용, 장 번호가 나온다.
  - 바뀐 임베딩 런타임으로 두 관문을 다시 통과한다.
    - 검색 관문: recall@18 ≥ 23/28
    - test C4 관문: B 누출 0/16, C 보류 5/6 이상
  - 비용 상한이 세 겹으로 걸려 있다: Anthropic 워크스페이스 한도, 앱 일일 상한, IP별 제한.

## 정한 것

| 항목 | 결정 |
| --- | --- |
| 생성 모델 | Claude Sonnet 5.5(평가 완료, 질문당 약 3.4센트) |
| 질문 임베딩 | 서버 안에서 KURE-v1 GGUF(Q8_0, LM Studio와 같은 파일)를 llama.cpp로 실행. torch 없음 |
| 검색 청크 수 | 공개 데모만 k=20(`RETRIEVE_K=20`). 로컬은 18 그대로 |
| 실행 환경 | AWS Lambda, 컨테이너 이미지(arm64), 서울 리전. 공개 주소는 Lambda 함수 URL |
| Anthropic 한도 | 데모 전용 워크스페이스 월 10달러, 자동 충전 끔(조직 API 크레딧 200달러에서 차감) |

## 구성

```
브라우저 ─HTTPS─▶ Lambda 함수 URL ─▶ Lambda(Docker 이미지, arm64, 메모리 2GB부터)
                                      ├ Lambda Web Adapter → FastAPI(sfqa web)
                                      ├ KURE-v1 GGUF + llama.cpp(질문 임베딩) + 색인 파일(이미지 안)
                                      ├ DynamoDB: 일일·IP별 호출 수
                                      ├ SSM Parameter Store: Anthropic 키(SecureString)
                                      └─▶ Claude API(Sonnet 5.5)
```

- **Lambda Web Adapter:** AWS가 공개한 확장 프로그램이다. 이미지의 `/opt/extensions`에 복사하면 Lambda 요청을 일반 HTTP 요청으로 바꿔 FastAPI에 넘긴다. 그래서 앱 코드는 거의 바뀌지 않는다.
- **이미지:** 로컬에서 쓰는 Docker로 만들어 ECR(AWS의 이미지 저장소)에 올린다. 이 맥이 arm64라 그대로 빌드된다.
- **인프라:** CloudFormation(SAM 문법) 템플릿 하나로 정의하고 `aws cloudformation deploy`로 올린다. 대상은 Lambda, 함수 URL, IAM 역할, DynamoDB 표, 로그다.

## 바뀌는 설계 원칙

- 지금 원칙은 "원문과 질문이 이 맥 밖으로 나가지 않는다"다. 공개 데모에서는 질문이 Anthropic과 AWS로 간다. 원문은 공유 저작물이다.
- `decisions.md`에 결정으로 남긴다. 화면에도 "질문은 Claude API로 전송됩니다"라고 알린다.
- 로컬 실행(`sfqa web`)은 지금처럼 loopback만 연다. 공개 모드는 설정으로만 켠다.

## 단계

### 1. 공급자 설정 (코드)

- `LLM_PROVIDER=lmstudio|anthropic`(기본 lmstudio)
  - anthropic이면 평가에 쓴 `claude.AnthropicClient`를 그대로 쓴다(thinking 끔).
  - loopback 검사는 LM Studio 주소에만 건다.
- `EMBED_PROVIDER=lmstudio|local`(기본 lmstudio)
  - local이면 llama-cpp-python으로 KURE-v1 GGUF를 실행한다(CLS pooling 확인). 파일은 `GGUF_MODELS_DIR`(기본 LM Studio 모델 폴더)에서 찾고, 없으면 Hugging Face에서 받는다.
- `RETRIEVE_K`(기본 18): 질문마다 가져올 청크 수
- 색인 폴더 이름에 임베딩 런타임을 넣는다. 다른 런타임으로 만든 색인은 불러오지 않게 한다.
- **확인:** `uv run pytest`(기존 178개 + 공급자 전환 테스트)

### 2. 임베딩을 다시 만들고 검색 재평가

- local 런타임으로 청크 805개를 다시 임베딩한다. LM Studio의 GGUF 판과 벡터가 조금 다를 수 있다.
- **확인:** `EMBED_PROVIDER=local uv run --extra gguf-embed sfqa retrieval-eval --method dense`
  - recall@18 23/28 이상, L4 누출 0
- **결과(2026-10-14):** 관문 미달. 결정은 k=20, 22/28을 한계로 기록하는 것이다.
  - 같은 글의 벡터는 LM Studio와 거의 같다(코사인 0.9995 이상). fp32(sentence-transformers)와 GGUF(llama.cpp) 모두 recall@18이 21/28이었다.
  - LM Studio(23/28)와의 차이는 17~21위 경계에 걸린 두 문항(f12-1-A, f02-2-A)뿐이다.
  - 사용자 결정: 공개 데모는 k=20(GGUF 22/28, f02-2-A가 21위). 관문을 넘기려고 k를 더 올리지 않는다(test 문항에 맞추는 일이 되므로). 서버 임베딩은 이미지가 작은 GGUF + llama.cpp로 한다.

### 3. Claude로 test 다시 돌리기

- 새 색인으로 test C0·C2·C3·C4를 돌린다(카세트 기록, 약 3달러).
- **확인:** C4 B 누출 0/16, C 보류 5/6 이상, 형식 실패 2/44 이하. 결과는 `runs/`에 남긴다.
- **결과(2026-10-14, `runs/20261014-154619-test`, 약 3.1달러):** 관문 통과
  - C4: B 누출 0/16, C 보류 6/6, 형식 실패 0/44, L4 0
  - A 22문항: 답함 21, 검증된 인용 21, 근거 장 일치 18, 키워드 정답 15(참고)
  - 이전(LM Studio 임베딩, k=18)과 비교: 근거 장 일치 19 → 18, 키워드 정답 16 → 15. 1문항 차이로 흔들림 범위다.
  - 응답 시간 중앙값 3.5초, 질문당 입력 약 1만 6천 토큰(약 3.4센트)
  - 사람 확인이 남은 것: C4에서 답한 B 9문항(이름 없이 풀어 쓴 누출이 있는지)

### 4. 공개 서비스 방어

- **호출 수 세기는 DynamoDB에 한다.** Lambda는 요청이 몰리면 실행 환경을 여러 개 띄워서, 메모리에 세면 환경마다 따로 센다.
  - 앱 전체 하루 상한: 300질문. 넘으면 "오늘 데모 한도에 닿았습니다"를 보여 준다.
  - IP별 하루 30질문. 항목은 TTL로 다음 날 지운다.
- **동시 실행 상한:** reserved concurrency를 2로 건다.
  - 계정의 남는 동시 실행이 100 이하면 걸 수 없다(신규 계정). 이때는 위의 DynamoDB 상한에만 기댄다.
- **Anthropic 한도에 닿을 때:** 지출 한도(400, 429 `enforced_spend_limit_reached`)와 속도 제한(429)에 걸리면 친절한 문구로 바꾸고 재시도하지 않는다.
- **입력과 로그:** 질문 길이는 이미 300자로 제한돼 있다. CloudWatch 로그에는 시각, 상태, 토큰 수만 남기고 질문 본문은 남기지 않는다.
- **IP 확인:** 함수 URL 뒤에서 실제 사용자 IP를 어느 값(`X-Forwarded-For` 등)으로 받는지 문서로 확인하지 못했다. 배포 뒤 로그로 확인하고 맞춘다.
- **확인:** 상한, 제한, 한도 오류의 응답을 모델 없이 테스트로 확인한다(DynamoDB는 대역으로).

### 5. 이미지와 배포

- **Dockerfile:**
  - python 3.12 slim(arm64)에 uv로 설치한다. llama-cpp-python은 빌드 단계에서 컴파일하고, 실행 이미지에는 결과만 넣는다.
  - KURE GGUF(634MB)와 색인을 이미지 안(`GGUF_MODELS_DIR`)에 넣는다. 실행 중 다운로드를 없앤다.
  - Lambda Web Adapter를 `/opt/extensions`에 복사하고 `PORT`를 맞춘다.
  - Lambda는 `/tmp` 말고는 읽기 전용이다. 캐시 경로를 `/tmp`로 둔다.
- **Lambda 설정:**
  - 메모리 2,048MB로 시작(1,769MB가 vCPU 1개분), 제한 시간 60초
  - 초기화가 10초를 넘으면 첫 호출로 넘어간다. GGUF는 이 맥에서 3초에 올라갔다.
- **키:** Anthropic 데모 워크스페이스에서 새 키를 발급해 SSM Parameter Store(SecureString)에 넣는다. 시작할 때 한 번 읽는다. 지금 `.env`의 키는 쓰지 않는다.
- **IAM 권한:** 해당 파라미터 읽기, 해당 DynamoDB 표, 로그 쓰기만 준다.
- **사용자가 할 일:**
  - AWS 계정, 결제, 로그인 설정(`aws configure` 또는 SSO)
  - Anthropic 데모 워크스페이스와 키 발급
  - AWS Budgets로 월 5달러 예산 알림 만들기
- **확인:**
  - 공개 주소에서 B·C 문항을 각 2개씩 손으로 물어 본다.
  - 콜드 스타트 시간, 응답 시간 중앙값, 실제 메모리 사용량(CloudWatch)을 잰다.

### 6. 마무리

- README에 공개 주소, 구성도, 비용 방어를 적는다. 포트폴리오에서 링크한다.
- `decisions.md`에 실행 환경, 모델, 임베딩, 한도 결정을 기록한다.

## 비용 (데모 규모 추정)

| 항목 | 추정 |
| --- | --- |
| Claude | 질문당 약 3.4센트(k=20, 입력 약 1만 6천 토큰). 월 10달러 한도에서 약 290질문 |
| Lambda | 무료 한도가 매달 요청 100만 건, 40만 GB-초다. 2GB로 질문당 약 10GB-초(Claude를 기다리는 시간 포함)를 쓰면 월 약 4만 질문까지 무료 |
| ECR(이미지 저장) | 이미지 약 1GB(추정), 월 0.1달러 수준 |
| DynamoDB, SSM, 로그 | 데모 규모면 거의 0 |

## 위험과 대응

| 위험 | 대응 |
| --- | --- |
| 콜드 스타트(이미지 약 1GB, GGUF 올리기 약 3초) | 데모라 허용한다. 화면에 "첫 질문은 느릴 수 있음"을 표시한다 |
| 메모리 부족 또는 과다 | 2GB로 시작해 CloudWatch로 재고 조정한다(GGUF라 4GB까지는 필요 없을 것으로 본다) |
| 키 유출, 비용 폭주 | 워크스페이스 한도, 자동 충전 끔, 앱 일일 상한, AWS 예산 알림. 유출이 의심되면 키를 바로 폐기한다 |
| 신규 계정이라 동시 실행 상한을 못 건다 | DynamoDB 일일 상한과 Anthropic 한도로 막는다 |

## 일정

- 1~3단계: 반나절(코드와 재평가)
- 4~5단계: 하루(AWS 계정 준비는 사용자가 한다)
- 커밋과 push는 요청할 때만 한다. 작업 브랜치는 `omo/claude-deploy`

## 근거 (확인한 AWS 문서)

- Lambda 한도(메모리 128~10,240MB, 1,769MB = vCPU 1개, 제한 시간 15분, 응답 6MB): https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html
- 컨테이너 이미지(최대 10GB, 읽기 전용 파일 시스템과 /tmp, 단일 아키텍처): https://docs.aws.amazon.com/lambda/latest/dg/images-create.html
- 무료 한도(매달 요청 100만 건, 40만 GB-초): https://repost.aws/knowledge-center/lambda-reduce-costs
- reserved concurrency(남는 동시 실행 − 100까지): https://docs.aws.amazon.com/lambda/latest/dg/configuration-concurrency.html
- Lambda Web Adapter(이미지의 /opt/extensions, PORT): https://aws.amazon.com/blogs/compute/using-response-streaming-with-aws-lambda-web-adapter-to-optimize-performance/
