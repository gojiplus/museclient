# Changelog

## Unreleased

- `Client(cap_usd, workdir).complete(messages, schema=...)`: structured Muse Spark
  completions through the OpenAI SDK, cached by request hash, under a hard cap
  reserved before each call, with rates read from the provider's model listing.
- Timeouts and lost connections are recorded as `unknown_outcome` and not retried;
  `Client.abandon(call_id)` sets such a call aside, keeping its reserve.
