# Changelog

## Unreleased

- Python 3.11+; only the docs build needs 3.12.
- `key_extra` for deliberate repeats of an identical request (replicates, explicit
  retries): part of the cache key and the record, never sent to the model.
- A ledger directory (`ledger=`), shareable by several workdirs under one cap;
  calls written before the ledger existed are imported so they still count.
- `complete(..., with_call=True)` returns the call folder with the answer, and
  `folder_for(...)` names it in advance, so a caller can record a failure there.

- `Client(cap_usd, workdir).complete(messages, schema=...)`: structured Muse Spark
  completions through the OpenAI SDK, cached by request hash, under a hard cap
  reserved before each call, with rates read from the provider's model listing.
- Timeouts and lost connections are recorded as `unknown_outcome` and not retried;
  `Client.abandon(call_id)` sets such a call aside, keeping its reserve.
