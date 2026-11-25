# Public chat demo (Modal)

Lets a visitor ask Minsky questions about its own repository, in a browser.
Minsky's core package stays local-first and untouched - this is an additive
deployment layer, not a rewrite.

**Architecture:** one Modal `cls`, scale-to-zero. The embedding model
(`nomic-embed-text`) runs via Ollama baked into the container image. The
LanceDB index is pre-built from this repo's own source (see
`build_demo_index.py`) and lives on a persistent Modal Volume - no live
re-indexing in prod. The chat answer comes from a hosted Claude call. A daily
request cap and a per-IP rate limit guard the metered API from being run up
by strangers. A shared access token (app-level, not Modal's platform proxy
auth - that gates the page load itself, which doesn't fit "visit a URL and
chat") is embedded in the page's JS and required on `/chat`; it blocks
scripts/bots that hit the API directly without ever loading the page.

## One-time setup

```sh
pip install modal
modal setup                        # opens a browser to authenticate this machine

# Build the demo index locally (Ollama running, nomic-embed-text pulled)
.venv/bin/python deploy/build_demo_index.py

# Create the volume and upload the pre-built index
modal volume create minsky-demo-db
modal volume put minsky-demo-db deploy/build/db /

# Create the secret (placeholder is fine until the real key is ready - the
# /chat endpoint degrades gracefully to "retrieval only" without it)
modal secret create anthropic-api-key ANTHROPIC_API_KEY=placeholder

# Shared access token required on /chat (generate your own; not a third-party key)
modal secret create minsky-chat-token CHAT_ACCESS_TOKEN=$(python3 -c "import secrets; print(secrets.token_urlsafe(24))")
```

## Deploy

```sh
modal deploy deploy/modal_app.py
```

Prints a URL like `https://<workspace>--minsky-chat-demo-minskychat-web.modal.run`.

## Updating the real Anthropic key later

```sh
modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-ant-... --force
```

No redeploy needed - the secret is re-injected on the next cold start.

## Re-indexing after the repo changes

```sh
.venv/bin/python deploy/build_demo_index.py
modal volume put minsky-demo-db deploy/build/db / --force
```

## Known limits (demo, not production)

- Per-IP rate limiting is in-memory per container; under concurrent cold
  starts it's a soft limit, not an exact one. Fine for a low-traffic demo.
- The daily request counter lives on the volume so it survives scale-to-zero,
  but isn't atomic across simultaneous requests - an exact cap would need a
  real datastore, not worth it at this scale.
- Cold start takes a few seconds (container boot + Ollama ready) after an
  idle period; `scaledown_window=300` keeps a container warm for 5 minutes
  after the last request to absorb most of this within a conversation.
