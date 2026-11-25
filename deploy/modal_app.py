"""Public chat demo for Minsky, deployed on Modal.

Architecture: one Modal cls, scale-to-zero. The embedding model
(nomic-embed-text) runs locally inside the container via Ollama, baked into
the image so cold starts don't re-download it. The chat answer itself comes
from a hosted Claude call. The LanceDB index is pre-built (see
build_demo_index.py) and lives on a persistent Modal Volume, not rebuilt on
every cold start.

Deploy with: modal deploy deploy/modal_app.py
"""

import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import modal

APP_NAME = "minsky-chat-demo"
VOLUME_NAME = "minsky-demo-db"
SECRET_NAME = "anthropic-api-key"
DB_MOUNT_PATH = "/data/minsky-db"
# `modal volume put ... deploy/build/db /` uploads the directory under its own
# name, landing the LanceDB files at <mount>/db rather than at <mount> itself.
DB_DATA_PATH = f"{DB_MOUNT_PATH}/db"
VECTOR_DIM = 768
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "claude-haiku-4-5-20251001"
TOP_K = 6

# Daily request cap and per-IP rate limit - a public demo endpoint hitting a
# metered API needs a hard ceiling, not just good intentions.
DAILY_REQUEST_LIMIT = 150
PER_IP_LIMIT = 6
PER_IP_WINDOW_SECONDS = 60

SYSTEM_PROMPT = """You are Minsky, a local-first RAG engine, answering questions about your \
own source code and design for visitors to your public demo.

You are given retrieved chunks from your own repository as context. Answer using only \
that context. If the context doesn't contain the answer, say so plainly instead of \
guessing - don't invent file names, functions, or behavior that isn't in front of you.

Keep answers concise and concrete. When you reference a file, use its real path from \
the context."""

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "zstd")
    .run_commands(
        "curl -fsSL https://ollama.com/install.sh | sh",
        "bash -c 'ollama serve & "
        "for i in $(seq 1 30); do curl -sf http://localhost:11434 >/dev/null && break; sleep 1; done; "
        f"ollama pull {EMBED_MODEL}; "
        "pkill ollama || true'",
    )
    .pip_install(
        "lancedb>=0.13",
        "pylance>=11.0.0",
        "pyarrow>=14",
        "httpx>=0.27",
        "fastapi>=0.110",
        "anthropic>=0.40",
    )
    .add_local_python_source("minsky")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
anthropic_secret = modal.Secret.from_name(SECRET_NAME)


CHAT_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Talk to Minsky</title>
<style>
  :root {
    --bg: #0d1117; --panel: #141b23; --ink: #d8e4ea; --muted: #7792a1;
    --accent: #5ec4e6; --rule: #22303b; --you: #1c3a4a; --mins: #1a2430;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; background: var(--bg); color: var(--ink);
    font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; }
  .wrap { max-width: 720px; margin: 0 auto; height: 100%; display: flex; flex-direction: column; padding: 1.25rem; }
  header { padding-bottom: 0.9rem; border-bottom: 1px solid var(--rule); margin-bottom: 1rem; }
  header h1 { font-size: 1.05rem; margin: 0 0 0.25rem; color: var(--accent); font-weight: 600; }
  header p { margin: 0; font-size: 0.8rem; color: var(--muted); }
  #log { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 0.75rem; padding-bottom: 1rem; }
  .msg { padding: 0.6rem 0.8rem; border-radius: 6px; font-size: 0.88rem; line-height: 1.5; max-width: 88%; white-space: pre-wrap; }
  .msg.you { background: var(--you); align-self: flex-end; }
  .msg.minsky { background: var(--mins); align-self: flex-start; }
  .sources { margin-top: 0.5rem; font-size: 0.72rem; color: var(--muted); }
  .sources code { color: var(--accent); }
  form { display: flex; gap: 0.5rem; border-top: 1px solid var(--rule); padding-top: 1rem; }
  input { flex: 1; background: var(--panel); border: 1px solid var(--rule); color: var(--ink);
    padding: 0.65rem 0.8rem; border-radius: 6px; font: inherit; }
  input:focus { outline: 1px solid var(--accent); }
  button { background: var(--accent); color: #06202c; border: none; border-radius: 6px;
    padding: 0.65rem 1.1rem; font: inherit; font-weight: 600; cursor: pointer; }
  button:disabled { opacity: 0.5; cursor: default; }
  .hint { color: var(--muted); font-size: 0.8rem; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>&gt; talk to minsky_</h1>
    <p>A local-first RAG engine, answering questions about its own repository. Ask about chunking, retrieval, eval, or the delegate pipeline.</p>
  </header>
  <div id="log"></div>
  <form id="f">
    <input id="q" autocomplete="off" placeholder="e.g. how does multi-query fusion work?" />
    <button id="send" type="submit">Ask</button>
  </form>
</div>
<script>
  const log = document.getElementById('log');
  const form = document.getElementById('f');
  const input = document.getElementById('q');
  const send = document.getElementById('send');

  function addMsg(text, who) {
    const div = document.createElement('div');
    div.className = 'msg ' + who;
    div.textContent = text;
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
    return div;
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const q = input.value.trim();
    if (!q) return;
    addMsg(q, 'you');
    input.value = '';
    send.disabled = true;
    const thinking = addMsg('...', 'minsky');
    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: q }),
      });
      const data = await res.json();
      if (!res.ok) {
        thinking.textContent = data.detail || 'Something went wrong.';
      } else {
        thinking.textContent = data.answer;
        if (data.sources && data.sources.length) {
          const src = document.createElement('div');
          src.className = 'sources';
          src.innerHTML = 'from: ' + data.sources.map(s => '<code>' + s + '</code>').join(', ');
          thinking.appendChild(src);
        }
      }
    } catch (err) {
      thinking.textContent = 'Network error - try again.';
    } finally {
      send.disabled = false;
    }
  });
</script>
</body>
</html>
"""


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


@app.cls(
    image=image,
    volumes={DB_MOUNT_PATH: volume},
    secrets=[anthropic_secret],
    scaledown_window=300,
    min_containers=0,
)
class MinskyChat:
    @modal.enter()
    def start(self) -> None:
        from minsky.ollama_client import OllamaClient
        from minsky.store import Store

        subprocess.Popen(["ollama", "serve"])
        self.ollama = OllamaClient("http://localhost:11434", EMBED_MODEL)
        for _ in range(30):
            if self.ollama.ping():
                break
            time.sleep(1)
        else:
            raise RuntimeError("ollama did not come up inside the container")

        self.store = Store(Path(DB_DATA_PATH), VECTOR_DIM)
        self._ip_hits: dict[str, list[float]] = {}

    def _usage_path(self) -> Path:
        return Path(DB_MOUNT_PATH) / "_usage.json"

    def _check_daily_budget(self) -> bool:
        path = self._usage_path()
        today = _today()
        record = {"date": today, "count": 0}
        if path.exists():
            try:
                record = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                pass
        if record.get("date") != today:
            record = {"date": today, "count": 0}
        if record["count"] >= DAILY_REQUEST_LIMIT:
            return False
        record["count"] += 1
        path.write_text(json.dumps(record))
        volume.commit()
        return True

    def _check_ip_rate_limit(self, ip: str) -> bool:
        now = time.time()
        hits = [t for t in self._ip_hits.get(ip, []) if now - t < PER_IP_WINDOW_SECONDS]
        if len(hits) >= PER_IP_LIMIT:
            self._ip_hits[ip] = hits
            return False
        hits.append(now)
        self._ip_hits[ip] = hits
        return True

    @modal.asgi_app()
    def web(self):
        import anthropic
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import HTMLResponse
        from pydantic import BaseModel

        web_app = FastAPI()
        claude = anthropic.Anthropic()

        class ChatRequest(BaseModel):
            message: str

        @web_app.get("/", response_class=HTMLResponse)
        def index() -> str:
            return CHAT_HTML

        @web_app.post("/chat")
        def chat(req: ChatRequest, request: Request) -> dict:
            message = req.message.strip()
            if not message:
                raise HTTPException(400, "empty message")
            if len(message) > 1000:
                raise HTTPException(400, "message too long")

            ip = request.headers.get("x-forwarded-for", request.client.host).split(",")[0].strip()
            if not self._check_ip_rate_limit(ip):
                raise HTTPException(429, "slow down - a few questions per minute, please")
            if not self._check_daily_budget():
                raise HTTPException(429, "demo has hit its daily question limit - try again tomorrow")

            vector = self.ollama.embed(message)
            hits = self.store.search(vector, top_k=TOP_K)
            if not hits:
                return {
                    "answer": "Nothing in the index looks related to that - try asking "
                    "about chunking, retrieval, eval, or the delegate pipeline.",
                    "sources": [],
                }

            context = "\n\n".join(
                f"### {h['path']}:{h['start_line']}\n{h['content']}" for h in hits
            )
            sources = sorted({h["path"] for h in hits})

            try:
                resp = claude.messages.create(
                    model=CHAT_MODEL,
                    max_tokens=600,
                    system=SYSTEM_PROMPT,
                    messages=[
                        {
                            "role": "user",
                            "content": f"Context:\n{context}\n\nQuestion: {message}",
                        }
                    ],
                )
                answer = resp.content[0].text
            except Exception:
                answer = (
                    "The chat model isn't configured yet, but here's what retrieval "
                    "found for that question:\n\n"
                    + "\n".join(f"- {s}" for s in sources)
                )

            return {"answer": answer, "sources": sources}

        return web_app
