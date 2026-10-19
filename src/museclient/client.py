"""A cached, budget-capped client for Meta's Muse Spark chat completions API.

Every request is identified by a hash of its body. Its request, the provider's
raw response and an accounting record are written under ``workdir/calls/<id>/``,
so asking again for the same thing returns the stored answer instead of paying
twice. Before a request is sent, its worst-case cost (input bytes as a token
bound, plus the full ``max_completion_tokens`` at the output rate) is reserved
against the authorized cap in a ledger shared by everything using that workdir;
a request that would take total exposure past the cap is refused, unsent. After
the response, the reservation is replaced by the cost of the usage the provider
reported.

A request whose outcome is unknown (sent, but no response stored) is never
retried automatically: it may have been charged. Delete its folder to retry it
deliberately.

Rates come from the provider's own model listing (``muse-code`` metadata), read
once per client; the client refuses to run if the model is not listed or is not
priced in USD. Adapted from notnews/rd_images ``judge.py`` and appeler/pranaam
``romanize_ror.py``.
"""

import fcntl
import hashlib
import json
import os
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

import jsonschema

MODEL = "muse-spark-1.3-contributor"
ENDPOINT = "https://api.meta.ai/v1/chat/completions"
MODELS_ENDPOINT = "https://api.meta.ai/muse-code/models"
CONTEXT_LIMIT = 1_048_576
# Bytes are a conservative bound on tokens for any script; the margin covers the
# provider's chat template and schema rendering.
TEMPLATE_MARGIN = 8192

CREDENTIALS = (
    Path.home() / ".config/muse/auth.json",
    Path.home() / ".config/local_elections/haryana_ocr.toml",
)


class BudgetError(RuntimeError):
    """The request would exceed the authorized cap."""


class ResponseError(ValueError):
    """The provider's response is incomplete, malformed or off-contract."""


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def atomic_json(path, value):
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
        f"No Muse API key: set MUSE_API_KEY or providers.meta.api_key in {CREDENTIALS[0]}"
    )


def post(url, body, key, timeout=300):
    request = urllib.request.Request(
        url,
        canonical(body).encode(),
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def get(url, key, timeout=30):
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def listed_rates(model, key, fetch=get):
    """USD per million tokens for input, cached input and output, as listed."""
    payload = fetch(MODELS_ENDPOINT, key)
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
    prompt, completion = usage["prompt_tokens"], usage["completion_tokens"]
    cache = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) if cached else 0
    if not (isinstance(prompt, int) and isinstance(completion, int) and 0 <= cache <= prompt):
        raise ResponseError(f"invalid usage: {usage}")
    return (
        (prompt - cache) * rates["input"] + cache * rates["cached"] + completion * rates["output"]
    ) / 1e6


class Client:
    """Structured Muse Spark completions under a hard spending cap."""

    def __init__(
        self,
        cap_usd,
        workdir,
        model=MODEL,
        endpoint=ENDPOINT,
        rates=None,
        transport=post,
        key=None,
    ):
        if not cap_usd > 0:
            raise ValueError("a positive authorized cap is required")
        self.cap = float(cap_usd)
        self.model = model
        self.endpoint = endpoint
        self.transport = transport
        self.root = Path(workdir)
        self.calls = self.root / "calls"
        self.calls.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.root / "ledger.lock"
        self._key = key
        self.rates = rates or listed_rates(model, self.key)

    @property
    def key(self):
        if self._key is None:
            self._key = api_key()
        return self._key

    def exposure(self):
        """Reserved or reported cost of every call recorded under this workdir."""
        total = 0.0
        for path in self.calls.glob("*/accounting.json"):
            total += json.loads(path.read_text())["charged_bound_usd"]
        return total

    def payload(self, messages, schema, name, max_completion_tokens, reasoning_effort):
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
    ):
        """The parsed, schema-valid JSON answer to `messages`; cached by request."""
        body = self.payload(messages, schema, name, max_completion_tokens, reasoning_effort)
        call_id = sha(canonical(body))[:24]
        folder = self.calls / call_id
        if (folder / "answer.json").exists():
            return json.loads((folder / "answer.json").read_text())
        with self.lock_path.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (folder / "accounting.json").exists():
                if not (folder / "response.json").exists():
                    raise RuntimeError(
                        f"call {call_id} was sent with no stored response; it may "
                        "have been charged. Delete its folder to retry deliberately."
                    )
            else:
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
                atomic_json(
                    folder / "accounting.json",
                    {
                        "call_id": call_id,
                        "model": self.model,
                        "rates_usd_per_million": self.rates,
                        "reserved_usd": reserve,
                        "charged_bound_usd": reserve,
                        "authorized_cap_usd": self.cap,
                        "status": "reserved",
                        "started_at": time.time(),
                    },
                )
        if not (folder / "response.json").exists():
            status, raw = self.transport(self.endpoint, body, self.key)
            atomic_json(folder / "response.json", {"status": status, "raw": raw})
        return self._settle(folder, schema)

    def _settle(self, folder, schema):
        accounting = json.loads((folder / "accounting.json").read_text())
        stored = json.loads((folder / "response.json").read_text())
        if stored["status"] != 200:
            accounting.update(status="failed", error=f"HTTP {stored['status']}")
            atomic_json(folder / "accounting.json", accounting)
            raise ResponseError(f"provider returned HTTP {stored['status']}: {stored['raw'][:300]}")
        response = json.loads(stored["raw"])
        usage = response["usage"]
        accounting.update(
            status="received",
            usage=usage,
            estimated_cost_usd=usage_cost(usage, self.rates),
            # the bound ignores the cache discount, so the ledger never undercounts
            charged_bound_usd=usage_cost(usage, self.rates, cached=False),
            response_model=response.get("model"),
            elapsed_seconds=time.time() - accounting["started_at"],
        )
        atomic_json(folder / "accounting.json", accounting)
        choice = response["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise ResponseError(f"incomplete response: finish_reason={choice.get('finish_reason')}")
        answer = json.loads(choice["message"]["content"])
        jsonschema.validate(answer, schema)
        atomic_json(folder / "answer.json", answer)
        accounting["status"] = "valid"
        atomic_json(folder / "accounting.json", accounting)
        return answer

    def spend(self):
        """Provider-reported cost of completed calls, and current exposure."""
        reported = 0.0
        for path in self.calls.glob("*/accounting.json"):
            reported += json.loads(path.read_text()).get("estimated_cost_usd", 0.0)
        return {"reported_usd": reported, "exposure_usd": self.exposure(), "cap_usd": self.cap}
