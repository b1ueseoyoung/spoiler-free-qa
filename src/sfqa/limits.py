"""공개 데모의 호출 상한(docs/deploy-plan.md 4단계).

- 세 가지를 센다: 하루 전체, IP별 하루, IP별 1분. 날짜는 한국 시간 기준이다.
- 세 칸을 한 번에 올린다. 하나라도 상한이면 아무것도 세지 않고 LimitExceeded를 낸다.
- 운영: DynamoDB(TransactWriteItems + 조건식). Lambda는 실행 환경을 여러 개 띄우므로 메모리에 세면 맞지 않는다.
- IP는 원문 대신 HMAC 해시로만 저장하고, 항목은 TTL로 지운다(하루 칸 2일, 1분 칸 10분).
- 모델을 부르기 전에 센다. 형식이 틀린 요청(422)은 세지 않는다.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")
SCOPES = ("ip_minute", "ip_daily", "daily")
MESSAGES = {
    "ip_minute": "질문이 너무 빠릅니다. 1분 뒤에 다시 물어 주세요.",
    "ip_daily": "오늘 이 기기에서 물을 수 있는 횟수를 다 썼습니다. 내일 다시 써 주세요.",
    "daily": "오늘 데모 한도에 닿았습니다. 내일 다시 써 주세요.",
}


@dataclass(frozen=True)
class Limits:
    daily: int = 300
    ip_daily: int = 30
    ip_minute: int = 5


def load_limits(env: Mapping[str, str]) -> Limits:
    d = Limits()
    return Limits(daily=int(env.get("DAILY_LIMIT", d.daily)), ip_daily=int(env.get("IP_DAILY_LIMIT", d.ip_daily)),
                  ip_minute=int(env.get("IP_MINUTE_LIMIT", d.ip_minute)))


class LimitExceeded(Exception):
    def __init__(self, scope: str):
        super().__init__(scope)
        self.scope = scope

    @property
    def message(self) -> str:
        return MESSAGES[self.scope]


def ip_hash(ip: str, salt: str) -> str:
    return hmac.new(salt.encode(), ip.encode(), hashlib.sha256).hexdigest()[:24]


def slots(ip: str, now: datetime, limits: Limits, salt: str) -> list[tuple[str, str, int, int]]:
    """(scope, 키, 상한, 만료 시각 epoch). 순서는 SCOPES와 같다."""
    local = now.astimezone(KST)
    day, minute = local.strftime("%Y%m%d"), local.strftime("%Y%m%d%H%M")
    h = ip_hash(ip, salt)
    day_ttl = int((now + timedelta(days=2)).timestamp())
    return [
        ("ip_minute", f"ip#{h}#min#{minute}", limits.ip_minute, int((now + timedelta(minutes=10)).timestamp())),
        ("ip_daily", f"ip#{h}#day#{day}", limits.ip_daily, day_ttl),
        ("daily", f"all#day#{day}", limits.daily, day_ttl),
    ]


def _now(now: datetime | None) -> datetime:
    return now or datetime.now(timezone.utc)


class MemoryCounter:
    """테스트와 로컬 확인용. 프로세스 하나 안에서만 맞다."""

    def __init__(self, limits: Limits = Limits(), salt: str = ""):
        self.limits, self.salt = limits, salt
        self.counts: dict[str, int] = {}

    def hit(self, ip: str, now: datetime | None = None) -> None:
        s = slots(ip, _now(now), self.limits, self.salt)
        for scope, key, limit, _ in s:
            if self.counts.get(key, 0) >= limit:
                raise LimitExceeded(scope)
        for _, key, _, _ in s:
            self.counts[key] = self.counts.get(key, 0) + 1


class DynamoCounter:
    """DynamoDB 표(파티션 키 pk, 문자열. TTL 속성 ttl)에 센다. client는 boto3 DynamoDB 클라이언트."""

    def __init__(self, table: str, limits: Limits = Limits(), salt: str = "", client=None):
        if client is None:
            import boto3  # 배포 이미지에만 있다(aws extra)

            client = boto3.client("dynamodb")
        self.table, self.limits, self.salt, self.client = table, limits, salt, client

    def hit(self, ip: str, now: datetime | None = None) -> None:
        s = slots(ip, _now(now), self.limits, self.salt)
        items = [{"Update": {
            "TableName": self.table,
            "Key": {"pk": {"S": key}},
            "UpdateExpression": "ADD #n :one SET #ttl = if_not_exists(#ttl, :ttl)",
            "ConditionExpression": "attribute_not_exists(#n) OR #n < :limit",
            "ExpressionAttributeNames": {"#n": "n", "#ttl": "ttl"},  # ttl은 DynamoDB 예약어다
            "ExpressionAttributeValues": {":one": {"N": "1"}, ":limit": {"N": str(limit)}, ":ttl": {"N": str(ttl)}},
        }} for _, key, limit, ttl in s]
        try:
            self.client.transact_write_items(TransactItems=items)
        except Exception as e:
            response = getattr(e, "response", None) or {}
            if response.get("Error", {}).get("Code") == "TransactionCanceledException":
                reasons = response.get("CancellationReasons") or []
                for (scope, *_), reason in zip(s, reasons):
                    if (reason or {}).get("Code") == "ConditionalCheckFailed":
                        raise LimitExceeded(scope) from None
            raise


def client_ip(headers: Mapping[str, str], peer: str | None, trust_lambda: bool) -> str:
    """사용자 IP. Lambda Web Adapter 뒤에서는 요청이 127.0.0.1에서 오므로, 어댑터가 넣는 요청 정보
    (x-amzn-request-context의 http.sourceIp)를 쓴다. 이 헤더는 공개 모드(어댑터 뒤)에서만 믿는다.
    X-Forwarded-For는 사용자가 꾸밀 수 있어 쓰지 않는다."""
    if trust_lambda:
        raw = headers.get("x-amzn-request-context")
        if raw:
            try:
                ctx = json.loads(raw)
            except json.JSONDecodeError:
                ctx = {}
            ip = (ctx.get("http") or {}).get("sourceIp") or (ctx.get("identity") or {}).get("sourceIp")
            if ip:
                return ip
    return peer or "unknown"
