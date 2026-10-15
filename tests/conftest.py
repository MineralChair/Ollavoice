"""Shared fixtures, including a fake Ollama server so the pipeline can be
exercised end to end without Ollama installed."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from ollavoice.config import Config, apply_tier
from ollavoice.wav import AudioClip


@pytest.fixture
def tmp_home(tmp_path, monkeypatch) -> Path:
    """Redirect every OllaVoice path into a temp dir so tests never touch the
    real %APPDATA% / ~/.cache."""
    monkeypatch.setenv("OLLAVOICE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("OLLAVOICE_OFFLINE", "1")
    (tmp_path / "home").mkdir(parents=True, exist_ok=True)
    return tmp_path / "home"


class FakeOllama(BaseHTTPRequestHandler):
    """Speaks just enough of the Ollama HTTP API for the pipeline."""

    answer = "Hello. I am a local model. Everything stayed on this machine."
    delay = 0.0
    requests: list[dict] = []

    def log_message(self, *args) -> None:  # silence
        pass

    def _json(self, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/api/version"):
            return self._json({"version": "0.0.0-fake"})
        if self.path.startswith("/api/tags"):
            return self._json({"models": [
                {"name": "testmodel:latest", "size": 1000, "modified_at": 0},
                {"name": "qwen3:4b", "size": 2000, "modified_at": 0},
            ]})
        if self.path.startswith("/api/ps"):
            return self._json({"models": []})
        self.send_error(404, "unknown endpoint")

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            payload = {}
        type(self).requests.append(payload)

        if self.path.startswith("/api/chat"):
            if not payload.get("stream"):
                return self._json({"model": payload.get("model"), "done": True,
                                   "message": {"role": "assistant", "content": self.answer}})
            words = self.answer.split(" ")
            body = b""
            for i, word in enumerate(words):
                chunk = word if i == 0 else " " + word
                body += json.dumps({"model": payload.get("model"), "done": False,
                                    "message": {"role": "assistant", "content": chunk}}
                                   ).encode() + b"\n"
                if self.delay:
                    time.sleep(self.delay)
            body += json.dumps({"model": payload.get("model"), "done": True,
                                "message": {"role": "assistant", "content": ""},
                                "total_duration": 1}).encode() + b"\n"
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/api/pull"):
            return self._json({"status": "success"})
        self.send_error(404, "unknown endpoint")


@pytest.fixture
def fake_ollama():
    """Run the fake server on an ephemeral port; yields its base URL."""
    handler = type("Bound", (FakeOllama,), {"requests": []})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    try:
        yield f"http://{host}:{port}", handler
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


@pytest.fixture
def cfg(tmp_home, fake_ollama) -> Config:
    """A config that points at the fake Ollama and uses silent/synthetic engines."""
    config = Config()
    apply_tier(config, "low")
    config.ollama.host = fake_ollama[0]
    config.ollama.model = "testmodel:latest"
    config.ollama.keep_alive = "0"
    config.gui.save_transcripts = False
    config.tts.engine = "none"
    config.stt.engine = "faster-whisper"
    return config


def tone_clip(duration_s: float = 1.0, freq: float = 440.0, rate: int = 16000,
              amplitude: float = 0.4) -> AudioClip:
    """A sine wave - handy for VAD/level tests that need deterministic audio."""
    t = np.arange(int(duration_s * rate)) / float(rate)
    samples = (amplitude * 32767 * np.sin(2 * np.pi * freq * t)).astype(np.int16)
    return AudioClip(samples, rate)


def silence_clip(duration_s: float = 1.0, rate: int = 16000) -> AudioClip:
    return AudioClip(np.zeros(int(duration_s * rate), dtype=np.int16), rate)
