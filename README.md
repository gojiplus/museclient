# museclient

Cached, budget-capped calls to Meta's Muse Spark (`muse-spark-1.3-contributor` by
default) for structured JSON answers.

## Install

From GitHub (not on PyPI), Python 3.12+:

```sh
uv add "museclient @ git+https://github.com/gojiplus/museclient"
```

Meta ships no client of its own and documents the OpenAI SDK pointed at
`https://api.meta.ai/v1`; museclient sends requests that way and adds the cap,
cache and ledger.

## Use

```python
from museclient import Client

client = Client(cap_usd=2, workdir="data/muse")  # rates read from the provider
answer = client.complete(
    [{"role": "user", "content": "..."}],
    schema={
        "type": "object",
        "properties": {...},
        "required": [...],
        "additionalProperties": False,
    },
    max_completion_tokens=8000,
    reasoning_effort="low",
)
print(client.spend())  # {'reported_usd': ..., 'exposure_usd': ..., 'cap_usd': 2.0}
```

What it guarantees:

- **No double payment.** Each request is keyed by a hash of its body; asking again
  returns the stored answer.
- **A hard cap.** Before sending, the call's worst case (input bytes as a token bound,
  plus all of `max_completion_tokens`) is reserved against the cap, across every call
  in the workdir. A call that would pass the cap is refused unsent.
- **No silent paid retries.** The SDK's retries are off. A call that times out or
  loses its connection may have been charged; it is recorded as `unknown_outcome`
  and refused until `Client.abandon(call_id)` sets it aside, keeping its reserve in
  the ledger.
- **Checked answers.** A truncated response (`finish_reason != "stop"`) or one that
  fails the JSON schema raises, after its usage is recorded.
- **Listed prices.** Rates come from the provider's model listing; the client refuses a
  model that is not listed or not priced in USD.

The key is read from `MUSE_API_KEY` or `MODEL_API_KEY`, else
`~/.config/muse/auth.json` (`providers.meta.api_key`) or
`~/.config/local_elections/haryana_ocr.toml` (`meta.api_key`). It is never written to
the workdir.

Each call leaves `request.json`, `response.json`, `accounting.json` and `answer.json`
under `workdir/calls/<id>/`.

Adapted from the judge service in notnews/rd_images and the pricing check in
appeler/pranaam.
