"""A cached, budget-capped client for Meta's Muse Spark.

Meta ships no client of its own; its documentation points to the OpenAI SDK with
``base_url="https://api.meta.ai/v1"``. This module uses that SDK for transport
and adds what the SDK does not do: a hard spending cap, a cache, and a ledger.

Every request is identified by a hash of its body. Its request, the provider's
raw response and an accounting record are written under ``workdir/calls/<id>/``,
so asking again for the same thing returns the stored answer instead of paying
twice. Before a request is sent, its worst-case cost (input bytes as a token
bound, plus the full ``max_completion_tokens`` at the output rate) is reserved
against the authorized cap in a ledger shared by everything using that workdir.
A request that would take total exposure past the cap is refused, unsent. After
the response, the reservation is replaced by the cost of the usage the provider
reported.

The SDK's own retries are switched off: a retry is a spending decision. A call
that times out or loses its connection may still have been charged, so it is
recorded as ``unknown_outcome`` and not retried. ``Client.abandon(call_id)``
sets it aside, keeping its reserve in the ledger, so the same request can be
sent again on purpose.

Rates come from the provider's model listing (``muse-code`` metadata), read once
per client; the client refuses a model that is not listed or not priced in USD.
Adapted from notnews/rd_images ``judge.py`` and appeler/pranaam
``romanize_ror.py``.
"""

import fcntl
import hashlib
import json
import os
import time
import tomllib
from collections.abc import Callable
from pathlib import Path

import httpx
import jsonschema
import openai

MODEL = "muse-spark-1.3-contributor"
BASE_URL = "https://api.meta.ai/v1"
MODELS_ENDPOINT = "https://api.meta.ai/muse-code/models"
CONTEXT_LIMIT = 1_048_576
# Bytes bound tokens for any script; the margin covers the chat template and
# the rendered schema.
TEMPLATE_MARGIN = 8192
TIMEOUT_SECONDS = 600

CREDENTIALS = (
    Path.home() / ".config/muse/auth.json",
    Path.home() / ".config/local_elections/haryana_ocr.toml",
)


class BudgetError(RuntimeError):
    """The request would exceed the authorized cap."""


class ResponseError(ValueError):
    """The provider's response is incomplete, malformed or off-contract."""


class UnknownOutcomeError(RuntimeError):
    """The call was sent but no response arrived; it may have been charged."""


def canonical(value):
    """Deterministic JSON text: sorted keys, no spaces, Unicode kept."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(text):
    """Hex SHA-256 of a string."""
    return hashlib.sha256(text.encode()).hexdigest()


def atomic_json(path, value):
    """Write ``value`` as JSON via a temporary file, so a crash leaves no half file."""
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(canonical(value) + "\n", encoding="utf-8")
    temp.replace(path)


def api_key():
    """The key from MUSE_API_KEY / MODEL_API_KEY or a known config file."""
    for name in ("MUSE_API_KEY", "MODEL_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    for path in CREDENTIALS:
        if not path.exists():
            continue
        if path.suffix == ".json":
            found = json.loads(path.read_text()).get("providers", {}).get("meta", {})
        else:
            found = tomllib.loads(path.read_text()).get("meta", {})
        if found.get("api_key"):
            return found["api_key"]
    raise RuntimeError(
        "No Muse API key: set MUSE_API_KEY or providers.meta.api_key in "
        f"{CREDENTIALS[0]}"
    )


def sdk_transport(body, key, timeout=TIMEOUT_SECONDS):
    """Send a chat completion through the OpenAI SDK; return (status, raw body).

    Raises UnknownOutcomeError when the request may have reached the provider but no
    response came back.
    """
    sdk = openai.OpenAI(api_key=key, base_url=BASE_URL, max_retries=0, timeout=timeout)
    try:
        raw = sdk.chat.completions.with_raw_response.create(**body)
    except openai.APIStatusError as error:
        return error.status_code, error.response.text
    except (openai.APITimeoutError, openai.APIConnectionError) as error:
        raise UnknownOutcomeError(f"{type(error).__name__}: {error}") from error
    return raw.http_response.status_code, raw.http_response.text


def listed_rates(model, key):
    """USD per million tokens for input, cached input and output, as listed."""
    response = httpx.get(
        MODELS_ENDPOINT, headers={"Authorization": f"Bearer {key}"}, timeout=30
    )
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data", []) if isinstance(payload, dict) else payload
    row = next((r for r in rows if r.get("id") == model), None)
    if row is None:
        raise RuntimeError(f"{model} is not in the provider's model listing")
    cost = row["metadata"]["muse-code"]["cost"]
    if cost.get("currency") != "USD":
        raise RuntimeError(f"{model} is not priced in USD")
    rates = {k: float(cost[k]) for k in ("input", "cached", "output")}
    if any(v < 0 for v in rates.values()):
        raise RuntimeError(f"{model} lists a negative price")
    return rates


def usage_cost(usage, rates, cached=True):
    """USD for a usage record; ``cached=False`` ignores the cache discount."""
    prompt, completion = usage["prompt_tokens"], usage["completion_tokens"]
    cache = (
        (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        if cached
        else 0
    )
    if not (
        isinstance(prompt, int) and isinstance(completion, int) and 0 <= cache <= prompt
    ):
        raise ResponseError(f"invalid usage: {usage}")
    return (
        (prompt - cache) * rates["input"]
        + cache * rates["cached"]
        + completion * rates["output"]
    ) / 1e6


class Client:
    """Structured Muse Spark completions under a hard spending cap."""

    def __init__(
        self,
        cap_usd: float,
        workdir: str | Path,
        model: str = MODEL,
        rates: dict[str, float] | None = None,
        transport: Callable[[dict, str], tuple[int, str]] | None = None,
        key: str | None = None,
        ledger: str | Path | None = None,
    ):
        """Open a ledger and cache in ``workdir`` under an authorized cap in USD.

        Args:
            cap_usd: Total spend allowed across every call in ``workdir``.
            workdir: Directory holding ``calls/`` and the ledger lock.
            model: Model id; must appear in the provider's model listing.
            rates: USD per million tokens (``input``, ``cached``, ``output``);
                read from the provider when not given.
            transport: ``(body, key) -> (status, raw)``; the OpenAI SDK by default.
            key: API key; found in the environment or config when not given.
            ledger: Directory of per-call spend entries counted against the cap.
                Defaults to ``workdir/ledger``; pass one directory to several
                workdirs to hold them all under one cap.

        Raises:
            ValueError: If ``cap_usd`` is not positive.
        """
        if not cap_usd > 0:
            raise ValueError("a positive authorized cap is required")
        self.cap = float(cap_usd)
        self.model = model
        self.transport = transport or sdk_transport
        self.root = Path(workdir)
        self.calls = self.root / "calls"
        self.calls.mkdir(parents=True, exist_ok=True)
        self.ledger = Path(ledger) if ledger else self.root / "ledger"
        self.ledger.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.ledger / ".lock"
        self._import_unledgered()
        self._key = key
        self.rates = rates or listed_rates(model, self.key)

    @property
    def key(self):
        """The API key, looked up on first use."""
        if self._key is None:
            self._key = api_key()
        return self._key

    def exposure(self):
        """Reserved or reported cost of every call recorded in the ledger."""
        return sum(
            json.loads(p.read_text())["charged_bound_usd"]
            for p in self.ledger.glob("*.json")
        )

    def payload(self, messages, schema, name, max_completion_tokens, reasoning_effort):
        """The chat-completions request body for a structured answer."""
        return {
            "model": self.model,
            "store": False,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
            "reasoning_effort": reasoning_effort,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": name, "strict": True, "schema": schema},
            },
        }

    def complete(
        self,
        messages,
        schema,
        name="result",
        max_completion_tokens=8000,
        reasoning_effort="low",
        key_extra=None,
        with_call=False,
    ):
        """The parsed, schema-valid JSON answer to ``messages``; cached by request.

        ``key_extra`` is folded into the cache key and recorded, but not sent:
        use it for deliberate repeats of an identical request (a replicate, an
        explicit retry). With ``with_call=True`` the call's folder is returned
        too, so a caller can keep its own derived files beside the request.
        """
        body = self.payload(
            messages, schema, name, max_completion_tokens, reasoning_effort
        )
        identity = {"request": body, "extra": key_extra} if key_extra else body
        call_id = sha(canonical(identity))[:24]
        folder = self.calls / call_id
        if (folder / "answer.json").exists():
            answer = json.loads((folder / "answer.json").read_text())
            return (answer, folder) if with_call else answer
        with self.lock_path.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (folder / "accounting.json").exists():
                if not (folder / "response.json").exists():
                    raise UnknownOutcomeError(
                        f"call {call_id} was sent with no stored response and "
                        f"may have been charged; Client.abandon('{call_id}') sets "
                        "it aside so the request can be sent again"
                    )
            else:
                self._reserve(folder, body, call_id, max_completion_tokens, key_extra)
        if not (folder / "response.json").exists():
            try:
                status, raw = self.transport(body, self.key)
            except UnknownOutcomeError as error:
                self._update(folder, status="unknown_outcome", error=str(error)[:300])
                raise
            atomic_json(folder / "response.json", {"status": status, "raw": raw})
        answer = self._settle(folder, schema)
        return (answer, folder) if with_call else answer

    def abandon(self, call_id):
        """Set aside a call with no response, keeping its reserve in the ledger."""
        folder = self.calls / call_id
        if (folder / "response.json").exists():
            raise ValueError(f"call {call_id} has a response; nothing to abandon")
        self._update(folder, status="abandoned", abandoned_at=time.time())
        number = 1 + len(list(self.calls.glob(f"{call_id}.abandoned-*")))
        folder.rename(self.calls / f"{call_id}.abandoned-{number}")

    def spend(self):
        """Provider-reported cost of completed calls, and current exposure."""
        reported = sum(
            json.loads(p.read_text()).get("estimated_cost_usd", 0.0)
            for p in self.ledger.glob("*.json")
        )
        return {
            "reported_usd": reported,
            "exposure_usd": self.exposure(),
            "cap_usd": self.cap,
        }

    def _reserve(self, folder, body, call_id, max_completion_tokens, key_extra=None):
        token_bound = len(canonical(body).encode()) + TEMPLATE_MARGIN
        if token_bound + max_completion_tokens > CONTEXT_LIMIT:
            raise ValueError("request exceeds the context limit")
        reserve = usage_cost(
            {"prompt_tokens": token_bound, "completion_tokens": max_completion_tokens},
            self.rates,
            cached=False,
        )
        spent = self.exposure()
        if spent + reserve > self.cap:
            raise BudgetError(
                f"${spent:.4f} spent or reserved + ${reserve:.4f} for this call "
                f"would exceed the ${self.cap:.2f} cap"
            )
        folder.mkdir(parents=True)
        atomic_json(folder / "request.json", body)
        started = time.time()
        accounting = {
            "call_id": call_id,
            "key_extra": key_extra,
            "model": self.model,
            "rates_usd_per_million": self.rates,
            "reserved_usd": reserve,
            "charged_bound_usd": reserve,
            "authorized_cap_usd": self.cap,
            "status": "reserved",
            "started_at": started,
            # unique per call, so an abandoned call and its retry never collide
            "ledger_entry": sha(f"{folder.resolve()}|{started}")[:24],
        }
        atomic_json(folder / "accounting.json", accounting)
        atomic_json(self.ledger / f"{accounting['ledger_entry']}.json", accounting)

    def _import_unledgered(self):
        """Give calls written before the ledger existed an entry, so they count."""
        for path in self.calls.glob("*/accounting.json"):
            accounting = json.loads(path.read_text())
            if "ledger_entry" in accounting:
                continue
            accounting["ledger_entry"] = sha(f"{path.parent.resolve()}|imported")[:24]
            atomic_json(path, accounting)
            atomic_json(self.ledger / f"{accounting['ledger_entry']}.json", accounting)

    def _update(self, folder, **fields):
        path = folder / "accounting.json"
        accounting = json.loads(path.read_text())
        accounting.update(fields)
        atomic_json(path, accounting)
        atomic_json(self.ledger / f"{accounting['ledger_entry']}.json", accounting)
        return accounting

    def _settle(self, folder, schema):
        stored = json.loads((folder / "response.json").read_text())
        if stored["status"] != 200:
            self._update(folder, status="failed", error=f"HTTP {stored['status']}")
            raise ResponseError(
                f"provider returned HTTP {stored['status']}: {stored['raw'][:300]}"
            )
        response = json.loads(stored["raw"])
        usage = response["usage"]
        started = json.loads((folder / "accounting.json").read_text())["started_at"]
        self._update(
            folder,
            status="received",
            usage=usage,
            estimated_cost_usd=usage_cost(usage, self.rates),
            # the bound ignores the cache discount, so the ledger never undercounts
            charged_bound_usd=usage_cost(usage, self.rates, cached=False),
            response_model=response.get("model"),
            elapsed_seconds=time.time() - started,
        )
        choice = response["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise ResponseError(
                f"incomplete response: finish_reason={choice.get('finish_reason')}"
            )
        answer = json.loads(choice["message"]["content"])
        jsonschema.validate(answer, schema)
        atomic_json(folder / "answer.json", answer)
        self._update(folder, status="valid")
        return answer
