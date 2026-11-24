from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from .ollama_client import OllamaClient

DEFAULT_SYSTEM_PROMPT = (
    "You are a local coding worker. You produce boilerplate, scaffolding, "
    "and other bulk/mechanical output exactly as asked. Do not explain your "
    "output or add commentary outside of code comments where appropriate. "
    "Output only the requested content."
)


def log_dir(db_path: Path) -> Path:
    d = db_path.parent / "delegate_logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def delegate(
    ollama: OllamaClient,
    model: str,
    prompt: str,
    *,
    system: str | None = DEFAULT_SYSTEM_PROMPT,
    context: str | None = None,
    db_path: Path,
    show: bool = False,
) -> str:
    full_prompt = f"{context}\n\n---\n\n{prompt}" if context else prompt

    start = time.time()
    if show:
        print(f"\033[1m--> {model}\033[0m", file=sys.stderr)
        if system:
            print(f"\033[2m[system] {system}\033[0m", file=sys.stderr)
        print(f"\033[2m[prompt]\033[0m {prompt}", file=sys.stderr)
        if context:
            print(f"\033[2m[context: {len(context)} chars attached]\033[0m", file=sys.stderr)
        print(f"\033[1m<-- {model}\033[0m", file=sys.stderr)
        chunks: list[str] = []
        for chunk in ollama.chat_stream(model, full_prompt, system=system):
            print(chunk, end="", flush=True, file=sys.stderr)
            chunks.append(chunk)
        print(file=sys.stderr)
        response = "".join(chunks)
    else:
        response = ollama.chat(model, full_prompt, system=system)
    elapsed = time.time() - start

    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model,
        "system": system,
        "prompt": prompt,
        "context": context,
        "response": response,
        "elapsed_seconds": round(elapsed, 1),
    }
    out_dir = log_dir(db_path)
    out_path = out_dir / f"{int(start)}.json"
    out_path.write_text(json.dumps(record, indent=2))

    return response


def format_transcript(record: dict) -> str:
    lines = [
        f"[{record['timestamp']}] {record['model']} ({record['elapsed_seconds']}s)",
        "",
    ]
    if record.get("system"):
        lines += ["--- system ---", record["system"], ""]
    if record.get("context"):
        lines += ["--- context ---", record["context"], ""]
    lines += ["--- prompt ---", record["prompt"], "", "--- response ---", record["response"]]
    return "\n".join(lines)


def list_logs(db_path: Path) -> list[Path]:
    return sorted(log_dir(db_path).glob("*.json"))
