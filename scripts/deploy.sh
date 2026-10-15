#!/usr/bin/env bash
# 공개 데모 배포(docs/deploy-plan.md 5단계): 이미지 빌드 → ECR push → CloudFormation(SAM) 스택 갱신.
#   AWS_PROFILE=sfqa scripts/deploy.sh
# 필요한 권한: infra/deploy-policy.json. Anthropic 키는 이 스크립트가 다루지 않는다(SSM에 사용자가 직접 넣는다).
set -euo pipefail
cd "$(dirname "$0")/.."

REGION="${AWS_REGION:-ap-northeast-2}"
STACK=sfqa-demo
REPO=sfqa-demo
# 작품마다 서버용(GGUF) 색인이 있어야 한다. 하나라도 없으면 멈춘다
for work in data/works/*/; do
  [ -f "$work/clean/clean-report.json" ] || continue
  [ -f "$work/index/local__mykor__KURE-v1-gguf__KURE-v1-Q8_0.gguf/meta.json" ] || {
    echo "색인이 없다($work): EMBED_PROVIDER=local uv run --extra gguf-embed sfqa index --work all"; exit 1; }
done

ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGISTRY="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
TAG="$(git rev-parse --short HEAD)$([ -n "$(git status --porcelain)" ] && echo -dirty || true)"

if ! aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" >/dev/null 2>&1; then
  aws ecr create-repository --repository-name "$REPO" --region "$REGION" \
    --image-scanning-configuration scanOnPush=true >/dev/null
  echo "ECR 저장소를 만들었다: $REPO"
fi
# 최근 이미지 3개만 남긴다
aws ecr put-lifecycle-policy --repository-name "$REPO" --region "$REGION" --lifecycle-policy-text \
  '{"rules":[{"rulePriority":1,"description":"keep last 3","selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":3},"action":{"type":"expire"}}]}' >/dev/null

aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY" >/dev/null
# --provenance=false: Lambda는 증명(attestation)이 붙은 다중 매니페스트 이미지를 받지 않는다
docker buildx build --platform linux/arm64 --provenance=false -t "$REGISTRY/$REPO:$TAG" --push .
docker logout "$REGISTRY" >/dev/null  # ~/.docker/config.json에 남은 ECR 토큰(12시간짜리)을 지운다
DIGEST=$(aws ecr describe-images --repository-name "$REPO" --region "$REGION" --image-ids imageTag="$TAG" \
  --query 'imageDetails[0].imageDigest' --output text)

PARAMS=("ImageUri=$REGISTRY/$REPO@$DIGEST")
if ! aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" >/dev/null 2>&1; then
  PARAMS+=("IpHashSalt=$(openssl rand -hex 16)")  # 처음 만들 때만. 이후에는 이전 값을 쓴다
fi
aws cloudformation deploy --template-file infra/template.yaml --stack-name "$STACK" --region "$REGION" \
  --capabilities CAPABILITY_IAM CAPABILITY_AUTO_EXPAND --no-fail-on-empty-changeset \
  --parameter-overrides "${PARAMS[@]}" "$@"

URL=$(aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='FunctionUrl'].OutputValue" --output text)
# 배포 직후 첫 실행은 이미지를 처음 내려받느라 느리다. 사용자 대신 여기서 한 번 겪는다(Claude는 부르지 않는다).
curl -s -o /dev/null -w "준비 요청: %{http_code}, %{time_total}초\n" -X POST "${URL%/}/api/warm" --max-time 150 || true
echo "배포 완료: $TAG ($DIGEST)"
echo "공개 주소: $URL"
