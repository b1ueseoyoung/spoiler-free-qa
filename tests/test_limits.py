import json
from datetime import datetime, timedelta, timezone

import pytest

from sfqa.limits import (DynamoCounter, LimitExceeded, Limits, MemoryCounter, client_ip, ip_hash, load_limits,
                         slots)

T0 = datetime(2026, 10, 14, 14, 59, 30, tzinfo=timezone.utc)  # 한국 시간 23:59:30


def test_memory_counter_enforces_each_scope():
    c = MemoryCounter(Limits(daily=100, ip_daily=100, ip_minute=2))
    c.hit("1.1.1.1", T0), c.hit("1.1.1.1", T0)
    with pytest.raises(LimitExceeded) as e:
        c.hit("1.1.1.1", T0)
    assert e.value.scope == "ip_minute" and "1분" in e.value.message
    c.hit("2.2.2.2", T0)  # 다른 IP는 따로 센다
    c.hit("1.1.1.1", T0 + timedelta(minutes=1))  # 다음 분에는 다시 된다

    d = MemoryCounter(Limits(daily=2, ip_daily=100, ip_minute=100))
    d.hit("a", T0), d.hit("b", T0)
    with pytest.raises(LimitExceeded) as e:
        d.hit("c", T0)
    assert e.value.scope == "daily"


def test_exceeded_request_counts_nothing():
    c = MemoryCounter(Limits(daily=1, ip_daily=100, ip_minute=100))
    c.hit("a", T0)
    before = dict(c.counts)
    with pytest.raises(LimitExceeded):
        c.hit("b", T0)
    assert c.counts == before  # 막힌 요청은 IP 칸도 올리지 않는다


def test_day_rolls_over_at_korean_midnight():
    c = MemoryCounter(Limits(daily=1, ip_daily=1, ip_minute=100))
    c.hit("a", T0)
    with pytest.raises(LimitExceeded):
        c.hit("a", T0 + timedelta(seconds=20))  # 아직 23:59:50
    c.hit("a", T0 + timedelta(seconds=40))  # 한국 시간 다음 날 00:00:10


def test_slots_hash_ip_and_set_ttl():
    s = slots("203.0.113.7", T0, Limits(), salt="s")
    assert [x[0] for x in s] == ["ip_minute", "ip_daily", "daily"]
    assert all("203.0.113.7" not in key for _, key, _, _ in s)  # IP 원문은 저장하지 않는다
    assert s[1][1] == f"ip#{ip_hash('203.0.113.7', 's')}#day#20261014"
    assert s[0][3] - int(T0.timestamp()) == 600 and s[2][3] - int(T0.timestamp()) == 2 * 86400
    assert ip_hash("203.0.113.7", "s") != ip_hash("203.0.113.7", "other")


def test_load_limits_from_env():
    assert load_limits({}) == Limits()
    assert load_limits({"DAILY_LIMIT": "10", "IP_DAILY_LIMIT": "3", "IP_MINUTE_LIMIT": "1"}) == Limits(10, 3, 1)


class CanceledError(Exception):
    def __init__(self, codes):
        super().__init__("TransactionCanceledException")
        self.response = {"Error": {"Code": "TransactionCanceledException"},
                         "CancellationReasons": [{"Code": c} for c in codes]}


class FakeDynamo:
    def __init__(self, fail_codes=None, error=None):
        self.calls, self.fail_codes, self.error = [], fail_codes, error

    def transact_write_items(self, TransactItems):
        self.calls.append(TransactItems)
        if self.error:
            raise self.error
        if self.fail_codes:
            raise CanceledError(self.fail_codes)


def test_dynamo_counter_sends_one_conditional_transaction():
    db = FakeDynamo()
    DynamoCounter("limits", Limits(daily=300, ip_daily=30, ip_minute=5), "s", client=db).hit("1.1.1.1", T0)
    items = db.calls[0]
    assert len(items) == 3 and all(x["Update"]["TableName"] == "limits" for x in items)
    u = items[2]["Update"]
    assert u["Key"] == {"pk": {"S": "all#day#20261014"}}
    assert u["ConditionExpression"] == "attribute_not_exists(#n) OR #n < :limit"
    assert u["ExpressionAttributeValues"][":limit"] == {"N": "300"}
    assert u["ExpressionAttributeNames"]["#ttl"] == "ttl"


def test_dynamo_counter_maps_failed_condition_to_scope():
    db = FakeDynamo(fail_codes=["None", "ConditionalCheckFailed", "None"])
    with pytest.raises(LimitExceeded) as e:
        DynamoCounter("limits", client=db).hit("1.1.1.1", T0)
    assert e.value.scope == "ip_daily"


def test_dynamo_counter_reraises_other_errors():
    with pytest.raises(RuntimeError):
        DynamoCounter("limits", client=FakeDynamo(error=RuntimeError("throttled"))).hit("1.1.1.1", T0)


def test_client_ip_trusts_adapter_context_only_in_public_mode():
    ctx = {"x-amzn-request-context": json.dumps({"http": {"sourceIp": "203.0.113.7"}}),
           "x-forwarded-for": "6.6.6.6"}
    assert client_ip(ctx, "127.0.0.1", trust_lambda=True) == "203.0.113.7"
    assert client_ip(ctx, "127.0.0.1", trust_lambda=False) == "127.0.0.1"  # 로컬에서는 헤더를 믿지 않는다
    assert client_ip({"x-forwarded-for": "6.6.6.6"}, "127.0.0.1", trust_lambda=True) == "127.0.0.1"  # XFF는 안 쓴다
    assert client_ip({"x-amzn-request-context": "{깨진"}, None, trust_lambda=True) == "unknown"


def test_ipv6_is_counted_per_64_block():
    c = MemoryCounter(Limits(daily=100, ip_daily=100, ip_minute=2))
    c.hit("2001:db8:1:2::a", T0), c.hit("2001:db8:1:2::b", T0)
    with pytest.raises(LimitExceeded):
        c.hit("2001:db8:1:2:ffff::1", T0)  # 같은 /64 안에서 주소만 바꿔도 같은 칸
    c.hit("2001:db8:1:3::a", T0)  # 다른 /64는 따로 센다
    assert ip_hash("::ffff:203.0.113.7", "s") == ip_hash("203.0.113.7", "s")  # IPv4가 IPv6 꼴로 와도 같다


class ConflictThenOk(FakeDynamo):
    def __init__(self, conflicts):
        super().__init__()
        self.conflicts = conflicts

    def transact_write_items(self, TransactItems):
        self.calls.append(TransactItems)
        if self.conflicts:
            self.conflicts -= 1
            raise CanceledError(["None", "None", "TransactionConflict"])


def test_dynamo_counter_retries_transaction_conflicts():
    db = ConflictThenOk(conflicts=2)
    DynamoCounter("limits", client=db, sleep=lambda s: None).hit("1.1.1.1", T0)
    assert len(db.calls) == 3
    always = ConflictThenOk(conflicts=99)
    with pytest.raises(CanceledError):  # 계속 겹치면 지금처럼 실패(웹은 503, 모델을 부르지 않는다)
        DynamoCounter("limits", client=always, sleep=lambda s: None).hit("1.1.1.1", T0)
    assert len(always.calls) == DynamoCounter.RETRIES + 1
