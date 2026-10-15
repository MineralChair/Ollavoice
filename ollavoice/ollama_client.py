"""Minimal, dependency-free client for a local Ollama server.

Uses only :mod:`urllib` from the standard library so the package works in a
venv with nothing else installed (and so ``ollavoice doctor`` can talk to
Ollama even before the speech extras are present).

Endpoints used: ``/api/version``, ``/api/tags``, ``/api/ps``, ``/api/chat``,
``/api/generate``, ``/api/pull``. See https://docs.ollama.com/api
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator


class OllamaError(RuntimeError):
    pass


class OllamaNotRunning(OllamaError):
    pass


class ModelNotFound(OllamaError):
    pass


@dataclass
class ChatMessage:
    role: str            # system | user | assistant | tool
    content: str
    images: list[str] = field(default_factory=list)   # base64, unused by voice flows

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.images:
            out["images"] = self.images
        return out

    @classmethod
    def user(cls, text: str) -> "ChatMessage":
        return cls("user", text)

    @classmethod
    def assistant(cls, text: str) -> "ChatMessage":
        return cls("assistant", text)

    @classmethod
    def system(cls, text: str) -> "ChatMessage":
        return cls("system", text)


ProgressFn = Callable[[dict[str, Any]], None]


class OllamaClient:
    def __init__(self, host: str = "http://127.0.0.1:11434", api_key: str = "",
                 timeout: float = 600.0) -> None:
        # Keep only scheme://host:port so a stray path in the config cannot
        # silently turn "/api/chat" into something else.
        parsed = urllib.parse.urlsplit(host.strip() or "http://127.0.0.1:11434")
        if parsed.scheme and parsed.netloc:
            self.host = f"{parsed.scheme}://{parsed.netloc}"
        else:
            self.host = host.rstrip("/")
        self.api_key = api_key or os.environ.get("OLLAMA_API_KEY", "")
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    def _request(self, path: str, payload: dict[str, Any] | None = None, *,
                 stream: bool = False, timeout: float | None = None) -> Any:
        url = self.host + path
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, data=data, headers=headers,
                                     method="POST" if data else "GET")
        try:
            resp = urllib.request.urlopen(req, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:800]
            try:
                msg = json.loads(body).get("error", body)
            except json.JSONDecodeError:
                msg = body
            if exc.code == 404 and "not found" in str(msg).lower():
                raise ModelNotFound(str(msg)) from exc
            raise OllamaError(f"Ollama returned HTTP {exc.code}: {msg}") from exc
        except urllib.error.URLError as exc:
            raise OllamaNotRunning(
                f"Cannot reach Ollama at {self.host} ({exc.reason}).\n"
                "Start it with `ollama serve`, or launch Ollama from the Start menu."
            ) from exc

        if stream:
            return resp
        with resp:
            raw = resp.read().decode("utf-8", "replace")
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise OllamaError(f"Ollama sent invalid JSON from {path}: {exc}") from exc

    # ------------------------------------------------------------------ #
    def version(self, timeout: float = 5.0) -> str:
        try:
            data = self._request("/api/version", timeout=timeout)
            return str(data.get("version", "unknown"))
        except OllamaError:
            return ""

    def is_alive(self, timeout: float = 3.0) -> bool:
        try:
            with urllib.request.urlopen(self.host + "/api/version", timeout=timeout) as resp:
                return resp.status == 200
        except Exception:
            return False

    def list_models(self) -> list[dict[str, Any]]:
        data = self._request("/api/tags")
        return data.get("models", []) or []

    def model_names(self) -> list[str]:
        return [m.get("name", "") for m in self.list_models() if m.get("name")]

    def has_model(self, name: str) -> bool:
        base = name.split(":")[0]
        for m in self.list_models():
            if m.get("name") == name or m.get("name", "").split(":")[0] == base:
                return True
        return False

    def ps(self) -> list[dict[str, Any]]:
        return self._request("/api/ps").get("models", []) or []

    # ------------------------------------------------------------------ #
    def pull(self, model: str, *, progress: ProgressFn | None = None,
             stream: bool = True) -> None:
        """Pull a model, reporting progress. ``progress`` gets the raw status dict."""
        if not stream:
            self._request("/api/pull", {"model": model, "stream": False},
                          timeout=self.timeout)
            return
        resp = self._request("/api/pull", {"model": model, "stream": True}, stream=True)
        try:
            for line in resp:  # type: ignore[union-attr]
                line = line.decode("utf-8", "replace").strip() if isinstance(line, bytes) else str(line).strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if evt.get("error"):
                    raise OllamaError(f"pull failed: {evt['error']}")
                if progress:
                    progress(evt)
                if evt.get("status") == "success":
                    return
        finally:
            try:
                resp.close()  # type: ignore[union-attr]
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    def chat(self, messages: list[ChatMessage | dict[str, Any]], model: str, *,
             options: dict[str, Any] | None = None, stream: bool = False,
             keep_alive: str = "10m") -> Iterator[dict[str, Any]] | dict[str, Any]:
        """Call ``/api/chat``. With ``stream=True`` returns an iterator of events."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.to_dict() if isinstance(m, ChatMessage) else m for m in messages],
            "stream": stream,
            "keep_alive": keep_alive,
        }
        if options:
            payload["options"] = options

        if not stream:
            return self._request("/api/chat", payload)

        def gen() -> Iterator[dict[str, Any]]:
            resp = self._request("/api/chat", payload, stream=True)
            try:
                for raw in resp:  # type: ignore[union-attr]
                    line = raw.decode("utf-8", "replace").strip() if isinstance(raw, bytes) else str(raw).strip()
                    if not line:
                        continue
                    try:
                        evt = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if evt.get("error"):
                        raise OllamaError(str(evt["error"]))
                    yield evt
                    if evt.get("done"):
                        return
            finally:
                try:
                    resp.close()  # type: ignore[union-attr]
                except Exception:
                    pass

        return gen()

    def chat_once(self, messages: list[ChatMessage | dict[str, Any]], model: str, *,
                  options: dict[str, Any] | None = None, keep_alive: str = "10m") -> str:
        data = self.chat(messages, model, options=options, stream=False, keep_alive=keep_alive)
        assert isinstance(data, dict)
        return (data.get("message") or {}).get("content", "")

    def generate(self, prompt: str, model: str, *, system: str = "",
                 options: dict[str, Any] | None = None, keep_alive: str = "10m") -> Iterator[dict[str, Any]]:
        payload: dict[str, Any] = {"model": model, "prompt": prompt, "stream": True,
                                   "keep_alive": keep_alive}
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options
        resp = self._request("/api/generate", payload, stream=True)

        def gen() -> Iterator[dict[str, Any]]:
            try:
                for raw in resp:  # type: ignore[union-attr]
                    line = raw.decode("utf-8", "replace").strip() if isinstance(raw, bytes) else str(raw).strip()
                    if not line:
                        continue
                    try:
                        evt = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if evt.get("error"):
                        raise OllamaError(str(evt["error"]))
                    yield evt
                    if evt.get("done"):
                        return
            finally:
                try:
                    resp.close()  # type: ignore[union-attr]
                except Exception:
                    pass

        return gen()

    def unload(self, model: str) -> None:
        """Free VRAM by setting keep_alive=0 for a loaded model."""
        try:
            self._request("/api/generate", {"model": model, "prompt": "", "keep_alive": 0},
                          timeout=30)
        except OllamaError:
            pass


@dataclass
class ChatHistory:
    """Sliding-window conversation memory (kept in RAM; never leaves the box)."""

    turns: int = 20
    system: str = ""
    messages: list[ChatMessage] = field(default_factory=list)

    def add(self, message: ChatMessage) -> None:
        self.messages.append(message)
        self._trim()

    def add_user(self, text: str) -> None:
        self.add(ChatMessage.user(text))

    def add_assistant(self, text: str) -> None:
        self.add(ChatMessage.assistant(text))

    def _trim(self) -> None:
        if self.turns <= 0:
            self.messages = []
            return
        keep = self.turns * 2
        if len(self.messages) > keep:
            self.messages = self.messages[-keep:]

    def for_api(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if self.system:
            out.append(ChatMessage.system(self.system).to_dict())
        out.extend(m.to_dict() for m in self.messages)
        return out

    def clear(self) -> None:
        self.messages.clear()

    def as_transcript(self) -> str:
        lines = []
        for m in self.messages:
            who = {"user": "You", "assistant": "Ollama", "system": "System"}.get(m.role, m.role)
            lines.append(f"{who}: {m.content}")
        return "\n".join(lines)


def wait_for_server(host: str = "http://127.0.0.1:11434", timeout_s: float = 30.0,
                    poll_s: float = 0.5) -> bool:
    """Block until Ollama answers, or ``timeout_s`` elapses."""
    client = OllamaClient(host, timeout=3)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if client.is_alive():
            return True
        time.sleep(poll_s)
    return False
