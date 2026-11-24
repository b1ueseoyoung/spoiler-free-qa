"""공개 데모의 호출 상한(docs/deploy-plan.md 4단계).

- 세 가지를 센다: 하루 전체, IP별 하루, IP별 1분. 날짜는 한국 시간 기준이다.
- 세 칸을 한 번에 올린다. 하나라도 상한이면 아무것도 세지 않고 LimitExceeded를 낸다.
- 운영: DynamoDB(TransactWriteItems + 조건식). Lambda는 실행 환경을 여러 개 띄우므로 메모리에 세면 맞지 않는다.
- IP는 원문 대신 HMAC 해시로만 저장하고, 항목은 TTL로 지운다(하루 칸 2일, 1분 칸 10분).
- IPv6는 /64 대역으로 접어 센다. 한 사용자가 대역 안에서 주소만 바꿔 IP별 상한을 피하지 못하게 한다(리뷰, 2026-10-14).
- 동시 요청이 같은 '전체 하루' 칸을 갱신하면 DynamoDB가 한쪽을 TransactionConflict로 취소한다. 상한 때문이 아니므로
  잠깐 쉬었다가 몇 번 다시 시도한다(AWS SDK는 이 경우를 다시 시도하지 않는다).
- 모델을 부르기 전에 센다. 형식이 틀린 요청(422)은 세지 않는다.
- 질문(/api/ask)과 인물 채팅 턴(/api/chat)은 따로 센다(Limits.kind). 같은 표에 키 앞머리만 달리 넣고, 넘었을 때 문구도 다르다.
  채팅은 모델을 부르는 턴만 센다(이야기할 수 없는 인물, 미성년 보호 안내는 세지 않는다). 기본값의 근거는 docs/deploy-plan.md 7단계.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import random
import time
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
CHAT_MESSAGES = {
    "ip_minute": "말을 너무 빨리 걸었습니다. 1분 뒤에 다시 이야기해 주세요.",
    "ip_daily": "오늘 이 기기에서 나눌 수 있는 대화를 다 썼습니다. 내일 다시 이야기해 주세요.",
    "daily": "오늘 인물 대화 한도에 닿았습니다. 내일 다시 이야기해 주세요.",
}


@dataclass(frozen=True)
class Limits:
    daily: int = 300
    ip_daily: int = 30
    ip_minute: int = 5
    kind: str = ""  # 세는 것. ""은 질문, "chat"은 인물 채팅 턴. 키 앞머리, 환경 변수 이름, 넘었을 때 문구가 갈린다


# 채팅 한 턴은 보통 1~4센트, 최악 6센트다(뒤 장일수록, 이력이 길수록 크다). IP별 하루 최악 약 1.2달러로 질문(약 1달러)과 맞춘다
CHAT_LIMITS = Limits(daily=100, ip_daily=20, ip_minute=5, kind="chat")


def load_limits(env: Mapping[str, str], base: Limits = Limits()) -> Limits:
    """환경 변수로 기본값을 바꾼다. 질문은 DAILY_LIMIT·IP_DAILY_LIMIT·IP_MINUTE_LIMIT, 채팅은 이름 앞에 CHAT_을 붙인다."""
    p = f"{base.kind.upper()}_" if base.kind else ""
    return Limits(daily=int(env.get(f"{p}DAILY_LIMIT", base.daily)),
                  ip_daily=int(env.get(f"{p}IP_DAILY_LIMIT", base.ip_daily)),
                  ip_minute=int(env.get(f"{p}IP_MINUTE_LIMIT", base.ip_minute)), kind=base.kind)


class LimitExceeded(Exception):
    def __init__(self, scope: str, kind: str = ""):
        super().__init__(scope)
        self.scope, self.kind = scope, kind

    @property
    def message(self) -> str:
        return (CHAT_MESSAGES if self.kind == "chat" else MESSAGES)[self.scope]


def ip_bucket(ip: str) -> str:
    """세는 단위. IPv4는 주소 그대로, IPv6는 /64 대역(보통 한 가입자나 한 서버가 받는 크기)."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return ip
    if addr.version == 4:
        return str(addr)
    if addr.ipv4_mapped is not None:  # ::ffff:1.2.3.4 꼴로 온 IPv4
        return str(addr.ipv4_mapped)
    return str(ipaddress.ip_network(f"{addr}/64", strict=False))


def ip_hash(ip: str, salt: str) -> str:
    return hmac.new(salt.encode(), ip_bucket(ip).encode(), hashlib.sha256).hexdigest()[:24]


def slots(ip: str, now: datetime, limits: Limits, salt: str) -> list[tuple[str, str, int, int]]:
    """(scope, 키, 상한, 만료 시각 epoch). 순서는 SCOPES와 같다. 채팅 키는 'chat#'으로 시작한다(질문 키는 앞머리 없이 그대로)."""
    local = now.astimezone(KST)
    day, minute = local.strftime("%Y%m%d"), local.strftime("%Y%m%d%H%M")
    h = ip_hash(ip, salt)
    p = f"{limits.kind}#" if limits.kind else ""
    day_ttl = int((now + timedelta(days=2)).timestamp())
    return [
        ("ip_minute", f"{p}ip#{h}#min#{minute}", limits.ip_minute, int((now + timedelta(minutes=10)).timestamp())),
        ("ip_daily", f"{p}ip#{h}#day#{day}", limits.ip_daily, day_ttl),
        ("daily", f"{p}all#day#{day}", limits.daily, day_ttl),
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
                raise LimitExceeded(scope, self.limits.kind)
        for _, key, _, _ in s:
            self.counts[key] = self.counts.get(key, 0) + 1


class DynamoCounter:
    """DynamoDB 표(파티션 키 pk, 문자열. TTL 속성 ttl)에 센다. client는 boto3 DynamoDB 클라이언트."""

    RETRIES = 3

    def __init__(self, table: str, limits: Limits = Limits(), salt: str = "", client=None,
                 sleep=time.sleep):
        if client is None:
            import boto3  # 배포 이미지에만 있다(aws extra)

            client = boto3.client("dynamodb")
        self.table, self.limits, self.salt, self.client, self._sleep = table, limits, salt, client, sleep

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
        for attempt in range(self.RETRIES + 1):
            try:
                self.client.transact_write_items(TransactItems=items)
                return
            except Exception as e:
                response = getattr(e, "response", None) or {}
                if response.get("Error", {}).get("Code") != "TransactionCanceledException":
                    raise
                codes = [(r or {}).get("Code") for r in response.get("CancellationReasons") or []]
                for (scope, *_), code in zip(s, codes):
                    if code == "ConditionalCheckFailed":
                        raise LimitExceeded(scope, self.limits.kind) from None
                if "TransactionConflict" not in codes or attempt == self.RETRIES:
                    raise
                self._sleep(random.uniform(0.02, 0.08) * (attempt + 1))  # 다른 요청과 겹쳤다. 잠깐 뒤 다시


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
