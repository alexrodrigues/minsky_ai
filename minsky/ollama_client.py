from __future__ import annotations

import json
from collections.abc import Iterator

import httpx


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=timeout)

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["embeddings"]

    def ping(self) -> bool:
        try:
            resp = self._client.get(f"{self.base_url}/api/version", timeout=3.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def chat(
        self, model: str, prompt: str, system: str | None = None, timeout: float = 300.0
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.post(
            f"{self.base_url}/api/chat",
            json={"model": model, "messages": messages, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def chat_stream(
        self, model: str, prompt: str, system: str | None = None, timeout: float = 300.0
    ) -> Iterator[str]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        with self._client.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json={"model": model, "messages": messages, "stream": True},
            timeout=timeout,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
                if chunk.get("done"):
                    break
