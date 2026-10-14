"""배포 파일 점검(모델·AWS 없이): 이미지에 비밀 값이 들어가지 않고, 템플릿이 계획과 맞는지."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dockerignore_is_an_allowlist_without_secrets():
    lines = [x.strip() for x in (ROOT / ".dockerignore").read_text().splitlines() if x.strip() and not x.startswith("#")]
    assert lines[0] == "**"  # 모두 빼고 필요한 것만 넣는다
    allowed = [x[1:] for x in lines if x.startswith("!")]
    assert not any(".env" in x or "cassettes" in x or x.startswith("runs") or x.startswith("eval") for x in allowed)


def test_dockerfile_keeps_key_out_and_runs_public_mode():
    d = (ROOT / "Dockerfile").read_text()
    assert "ANTHROPIC_API_KEY=" not in d and ".env" not in d.replace("ENV ", "")
    assert "PUBLIC_DEMO=1" in d and "RETRIEVE_K=20" in d and "EMBED_PROVIDER=local" in d
    assert "aws-lambda-adapter:1.1.0" in d and "GGML_NATIVE=OFF" in d


def test_template_wires_limits_key_and_public_url():
    t = (ROOT / "infra" / "template.yaml").read_text()
    for needle in ("LIMITS_TABLE: !Ref LimitsTable", "ANTHROPIC_API_KEY_SSM_PARAM", "AuthType: NONE",
                   "Architectures: [arm64]", "AttributeName: ttl", "dynamodb:UpdateItem", "RetentionInDays: 14"):
        assert needle in t, needle


def test_deploy_policy_is_scoped():
    p = json.loads((ROOT / "infra" / "deploy-policy.json").read_text())
    for s in p["Statement"]:
        res = s["Resource"] if isinstance(s["Resource"], list) else [s["Resource"]]
        actions = s["Action"] if isinstance(s["Action"], list) else [s["Action"]]
        assert not any(a.endswith(":*") for a in actions), s["Sid"]
        if res == ["*"]:  # 자원을 좁힐 수 없는 읽기·로그인 동작만 * 허용
            assert all(a.split(":")[1].startswith(("Get", "Describe", "Validate")) for a in actions), s["Sid"]
