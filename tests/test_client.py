import json

import pytest

from museclient import BudgetError, Client, ResponseError, UnknownOutcome, usage_cost

RATES = {"input": 0.10, "cached": 0.002, "output": 0.20}
SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string"}},
    "required": ["label"],
    "additionalProperties": False,
}
MESSAGES = [{"role": "user", "content": "say ok"}]


def fake(answer='{"label": "ok"}', status=200, finish="stop", calls=None):
    def transport(body, key):
        if calls is not None:
            calls.append(body)
        response = {
            "model": body["model"],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
            "choices": [{"finish_reason": finish, "message": {"content": answer}}],
        }
        return status, json.dumps(response)

    return transport


def client(tmp_path, cap=1.0, **kw):
    return Client(cap, tmp_path, rates=RATES, key="test", **kw)


def test_answer_is_parsed_and_cost_is_the_reported_usage(tmp_path):
    c = client(tmp_path, transport=fake())
    assert c.complete(MESSAGES, SCHEMA) == {"label": "ok"}
    assert c.spend()["reported_usd"] == pytest.approx((1000 * 0.10 + 500 * 0.20) / 1e6)


def test_a_repeated_request_is_served_from_the_cache(tmp_path):
    calls = []
    c = client(tmp_path, transport=fake(calls=calls))
    c.complete(MESSAGES, SCHEMA)
    c.complete(MESSAGES, SCHEMA)
    assert len(calls) == 1


def test_a_request_past_the_cap_is_refused_before_it_is_sent(tmp_path):
    calls = []
    # max_completion_tokens alone reserves 8000 * 0.20 / 1e6 = $0.0016
    c = client(tmp_path, cap=0.001, transport=fake(calls=calls))
    with pytest.raises(BudgetError):
        c.complete(MESSAGES, SCHEMA)
    assert calls == []


def test_the_cap_counts_every_call_in_the_workdir(tmp_path):
    first = client(tmp_path, cap=0.01, transport=fake())
    first.complete(MESSAGES, SCHEMA)
    second = client(tmp_path, cap=first.exposure() + 1e-6, transport=fake())
    with pytest.raises(BudgetError):
        second.complete([{"role": "user", "content": "another"}], SCHEMA)


def test_an_off_schema_answer_is_rejected(tmp_path):
    c = client(tmp_path, transport=fake(answer='{"other": 1}'))
    with pytest.raises(Exception, match="label"):
        c.complete(MESSAGES, SCHEMA)


def test_a_truncated_answer_is_rejected(tmp_path):
    c = client(tmp_path, transport=fake(finish="length"))
    with pytest.raises(ResponseError):
        c.complete(MESSAGES, SCHEMA)


def test_a_timed_out_call_is_kept_in_the_ledger_and_not_retried(tmp_path):
    def timed_out(body, key):
        raise UnknownOutcome("APITimeoutError")

    c = client(tmp_path, transport=timed_out)
    with pytest.raises(UnknownOutcome):
        c.complete(MESSAGES, SCHEMA)
    reserve = c.exposure()
    assert reserve > 0
    again = client(tmp_path, transport=fake())
    with pytest.raises(UnknownOutcome, match="may have been charged"):
        again.complete(MESSAGES, SCHEMA)


def test_abandoning_a_call_allows_a_deliberate_retry_and_keeps_its_reserve(tmp_path):
    def timed_out(body, key):
        raise UnknownOutcome("APITimeoutError")

    c = client(tmp_path, transport=timed_out)
    with pytest.raises(UnknownOutcome):
        c.complete(MESSAGES, SCHEMA)
    reserve = c.exposure()
    call_id = next(p.name for p in c.calls.iterdir())
    c.abandon(call_id)
    again = client(tmp_path, transport=fake())
    assert again.complete(MESSAGES, SCHEMA) == {"label": "ok"}
    statuses = sorted(
        json.loads(p.read_text())["status"] for p in again.calls.glob("*/accounting.json")
    )
    assert statuses == ["abandoned", "valid"]
    assert again.exposure() > reserve


def test_cached_tokens_are_discounted_but_the_bound_is_not():
    usage = {
        "prompt_tokens": 1000,
        "completion_tokens": 0,
        "prompt_tokens_details": {"cached_tokens": 1000},
    }
    assert usage_cost(usage, RATES) == pytest.approx(1000 * 0.002 / 1e6)
    assert usage_cost(usage, RATES, cached=False) == pytest.approx(1000 * 0.10 / 1e6)
