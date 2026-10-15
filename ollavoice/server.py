"""A local HTTP API that puts speech in front of Ollama.

Standard library only (``http.server``), bound to loopback by default, no
authentication - treat it like any other local Ollama endpoint and do **not**
expose it to your network.

OpenAI-compatible routes, so existing tooling can point at it::

    POST /v1/audio/transcriptions   multipart: file, model?, language?   -> JSON/text
    POST /v1/audio/speech           JSON: input, voice?, model?, speed?  -> audio/wav

OllaVoice-native routes::

    GET  /health
    GET  /config
    GET  /voices
    POST /api/transcribe            JSON {audio_b64} or raw bytes       -> {"text": ...}
    POST /api/speak                 JSON {text, voice?, engine?}        -> audio/wav
    POST /api/chat                  JSON {text} or {audio_b64, speak}   -> {"text", "audio_b64"}
    GET  /                          the browser UI (mic capture + playback)

``python -m ollavoice serve`` starts it standalone; the desktop app can host it
too (Settings -> Advanced -> local speech API).
"""

from __future__ import annotations

import base64
import json
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import __version__
from .config import Config
from .pipeline import PipelineError, VoicePipeline

STATIC_DIR = Path(__file__).parent / "server" / "static"


# --------------------------------------------------------------------------- #
# Multipart parsing (the stdlib dropped cgi.parse_multipart, so do it ourselves)
# --------------------------------------------------------------------------- #


def parse_multipart(body: bytes, content_type: str) -> dict[str, Any]:
    """Return ``{field_name: str}`` and ``{field_name: (filename, bytes)}`` merged."""
    out: dict[str, Any] = {}
    if "boundary=" not in content_type:
        return out
    boundary = content_type.split("boundary=", 1)[1].split(";")[0].strip().strip('"')
    if not boundary:
        return out
    delim = ("--" + boundary).encode()
    for part in body.split(delim):
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        if b"\r\n\r\n" not in part:
            continue
        head, _, payload = part.partition(b"\r\n\r\n")
        headers = {}
        for line in head.decode("utf-8", "replace").splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()
        disposition = headers.get("content-disposition", "")
        name = ""
        filename = None
        for token in disposition.split(";"):
            token = token.strip()
            if token.startswith("name="):
                name = token[5:].strip('"')
            elif token.startswith("filename="):
                filename = token[9:].strip('"')
        if not name:
            continue
        if filename is not None:
            out[name] = (filename, payload)
        else:
            out[name] = payload.decode("utf-8", "replace")
    return out


# --------------------------------------------------------------------------- #
# Request handler
# --------------------------------------------------------------------------- #


class Handler(BaseHTTPRequestHandler):
    server_version = f"OllaVoice/{__version__}"
    protocol_version = "HTTP/1.1"
    pipeline: VoicePipeline | None = None      # injected by serve()
    lock: threading.Lock = threading.Lock()    # one turn at a time

    # ------------------------------------------------------------------ #
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 - stdlib API
        print(f"[speech-api] {self.address_string()} {fmt % args}")

    # ------------------------------------------------------------------ #
    def _send(self, code: int, body: bytes, content_type: str,
              extra: dict[str, str] | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _json(self, code: int, payload: Any) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _error(self, code: int, message: str) -> None:
        self._json(code, {"error": {"message": message, "type": "ollavoice_error",
                                    "code": code}})

    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    # ------------------------------------------------------------------ #
    def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib API
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - stdlib API
        route = urllib.parse.urlparse(self.path).path.rstrip("/") or "/"
        if route in ("/", "/index.html"):
            return self._serve_static("index.html", "text/html; charset=utf-8")
        if route == "/app.js":
            return self._serve_static("app.js", "application/javascript; charset=utf-8")
        if route == "/health":
            pipe = self.pipeline
            assert pipe is not None
            return self._json(200, {
                "status": "ok", "app": "OllaVoice", "version": __version__,
                "ollama": pipe.ollama.is_alive(timeout=2),
                "model": pipe.config.ollama.model,
                "stt": {"engine": pipe.config.stt.engine,
                        "model": pipe.config.stt.model,
                        "loaded_engine": pipe.stt.name,
                        "loaded_model": getattr(pipe.stt, "model_name", "")},
                "tts": {"engine": pipe.config.tts.engine,
                        "loaded_engine": pipe.tts.name,
                        "voice": getattr(pipe.tts, "voice_name", "")},
                "busy": pipe.is_busy,
            })
        if route == "/config":
            assert self.pipeline is not None
            cfg = self.pipeline.config
            return self._json(200, {
                "ollama": {"host": cfg.ollama.host, "model": cfg.ollama.model},
                "stt": {"engine": cfg.stt.engine, "model": cfg.stt.model,
                        "language": cfg.stt.language},
                "tts": {"engine": cfg.tts.engine, "piper_voice": cfg.tts.piper_voice,
                        "kokoro_voice": cfg.tts.kokoro_voice, "rate": cfg.tts.rate},
                "tier": cfg.tier, "language": cfg.language,
            })
        if route == "/voices":
            from .tts_engines import installed_voices

            return self._json(200, installed_voices())
        return self._error(404, f"no such route: {route}")

    def _serve_static(self, name: str, content_type: str) -> None:
        path = STATIC_DIR / name
        if not path.exists():
            return self._error(404, f"missing UI asset: {name}")
        return self._send(200, path.read_bytes(), content_type)

    # ------------------------------------------------------------------ #
    def do_POST(self) -> None:  # noqa: N802 - stdlib API
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path.rstrip("/")
        query = dict(urllib.parse.parse_qsl(parsed.query))
        content_type = self.headers.get("Content-Type", "")
        body = self._body()
        try:
            if route in ("/v1/audio/transcriptions", "/api/transcriptions"):
                return self._transcriptions(content_type, body, query)
            if route in ("/v1/audio/speech", "/api/speech", "/api/speak"):
                return self._speech(content_type, body, query)
            if route in ("/api/transcribe",):
                return self._transcribe_json(content_type, body)
            if route in ("/api/chat", "/v1/chat"):
                return self._chat(content_type, body)
            return self._error(404, f"no such route: {route}")
        except PipelineError as exc:
            return self._error(400, str(exc))
        except Exception as exc:  # keep the server alive on bad input
            return self._error(500, f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ #
    def _transcriptions(self, content_type: str, body: bytes,
                        query: dict[str, str]) -> None:
        audio, fields = self._extract_audio(content_type, body, query)
        if audio is None:
            return self._error(400, "no audio provided (send a multipart 'file' field, "
                                    "raw audio bytes, or JSON {\"audio_b64\": ...})")
        pipe = self.pipeline
        assert pipe is not None
        language = fields.get("language") or None
        model = fields.get("model")
        if model:
            pipe.config.stt.model = model
            pipe.reload_engines()
        try:
            result = pipe.stt.transcribe(audio, language=language)
        except Exception as exc:
            return self._error(500, f"transcription failed: {exc}")
        fmt = fields.get("response_format", "json")
        if fmt in ("text", "srt", "vtt"):
            if fmt == "text":
                return self._send(200, result.text.encode("utf-8"), "text/plain; charset=utf-8")
            text = _subtitle_format(result, fmt)
            ctype = "text/srt" if fmt == "srt" else "text/vtt"
            return self._send(200, text.encode("utf-8"), f"{ctype}; charset=utf-8")
        return self._json(200, {
            "text": result.text, "task": "transcribe",
            "language": result.language, "duration": round(result.duration_s, 3),
            "segments": [{"start": round(s.start, 3), "end": round(s.end, 3),
                          "text": s.text} for s in result.segments],
            "engine": result.engine, "model": result.model,
        })

    def _transcribe_json(self, content_type: str, body: bytes) -> None:
        audio, _ = self._extract_audio(content_type, body, {})
        if audio is None:
            return self._error(400, "expected JSON with 'audio_b64' or raw audio bytes")
        pipe = self.pipeline
        assert pipe is not None
        result = pipe.stt.transcribe(audio)
        return self._json(200, {"text": result.text, "language": result.language,
                                "duration_s": round(result.duration_s, 3),
                                "decode_s": round(result.decode_s, 3),
                                "engine": result.engine, "model": result.model})

    def _extract_audio(self, content_type: str, body: bytes,
                       query: dict[str, str]) -> tuple[Any, dict[str, str]]:
        """Accept multipart, raw bytes, base64 JSON, or ?url= (local file only)."""
        fields: dict[str, str] = {k: v for k, v in query.items()}
        if content_type.startswith("multipart/"):
            parts = parse_multipart(body, content_type)
            audio: Any = None
            for key, value in parts.items():
                if isinstance(value, tuple):
                    audio = value[1]
                elif isinstance(value, str):
                    fields[key] = value
            return audio, fields
        if content_type.startswith("application/json"):
            try:
                payload = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                return None, fields
            for key in ("audio_b64", "audio", "data"):
                if isinstance(payload.get(key), str):
                    try:
                        return base64.b64decode(payload[key]), {**fields, **payload}
                    except Exception:
                        return None, fields
            if isinstance(payload.get("file"), str) and Path(payload["file"]).exists():
                return Path(payload["file"]), {**fields, **payload}
            return None, {**fields, **payload}
        if content_type.startswith("audio/") or content_type.startswith("application/octet-stream"):
            return body, fields
        if body:
            return body, fields
        return None, fields

    # ------------------------------------------------------------------ #
    def _speech(self, content_type: str, body: bytes, query: dict[str, str]) -> None:
        pipe = self.pipeline
        assert pipe is not None
        text = ""
        voice = query.get("voice", "")
        engine = query.get("engine", "")
        speed = query.get("speed", "")
        if content_type.startswith("application/json") or not content_type:
            try:
                payload = json.loads(body.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                payload = {}
            text = str(payload.get("input") or payload.get("text") or "")
            voice = str(payload.get("voice") or voice)
            engine = str(payload.get("engine") or engine)
            speed = str(payload.get("speed") or payload.get("rate") or speed)
        else:
            text = body.decode("utf-8", "replace")
        if not text:
            return self._error(400, "'input' (the text to speak) is required")

        t0 = time.time()
        try:
            clip = self._synthesise(pipe, text, engine=engine, voice=voice, speed=speed)
        except Exception as exc:
            return self._error(500, f"synthesis failed: {exc}")
        if not len(clip.samples):
            return self._error(
                500, "the TTS engine produced no audio - check that the voice is "
                     "installed (`ollavoice voices`) and that tts.engine is not 'none'")
        wav = clip.to_wav_bytes()
        return self._send(200, wav, "audio/wav",
                          {"X-Synth-Seconds": f"{time.time() - t0:.3f}",
                           "X-Audio-Seconds": f"{clip.duration_s:.3f}",
                           "X-Tts-Engine": engine or pipe.config.tts.engine})

    @staticmethod
    def _synthesise(pipe, text: str, *, engine: str = "", voice: str = "",
                    speed: str = ""):
        """Synthesise with per-request overrides, without touching global config.

        Building a one-off engine from a *copy* of the settings is important: the
        earlier approach mutated ``pipe.config`` and called ``reload_engines()``,
        which threw away the loaded model on every request and raced with any turn
        already in flight.
        """
        from copy import deepcopy

        from .tts_engines import create_engine as create_tts

        settings = deepcopy(pipe.config.tts)
        if engine:
            settings.engine = engine
        if voice:
            if (engine or settings.engine) == "kokoro":
                settings.kokoro_voice = voice
            elif (engine or settings.engine) == "sapi":
                settings.sapi_voice = voice
            else:
                settings.piper_voice = voice
        try:
            if speed:
                settings.rate = max(0.5, min(2.0, float(speed)))
        except ValueError:
            pass

        # A request with no overrides reuses the warm engine; anything else gets a
        # temporary one so the conversation's voice is left alone.
        if engine or voice or speed:
            tts = create_tts(settings, language=pipe.config.language)
            with pipe._tts_lock:
                pieces = list(tts.synthesize_stream(text))
        else:
            pipe.config.tts = settings
            pieces = list(pipe._synth_pieces(text))

        if not pieces:
            import numpy as np

            from .wav import AudioClip

            return AudioClip(np.zeros(0, dtype=np.int16), 22050)
        clip = pieces[0]
        for extra in pieces[1:]:
            clip = clip.concat(extra)
        return clip

    # ------------------------------------------------------------------ #
    def _chat(self, content_type: str, body: bytes) -> None:
        """Speech and/or text in; text and/or speech out."""
        pipe = self.pipeline
        assert pipe is not None
        payload: dict[str, Any] = {}
        audio: Any = None
        if content_type.startswith("multipart/"):
            parts = parse_multipart(body, content_type)
            for key, value in parts.items():
                if isinstance(value, tuple):
                    audio = value[1]
                else:
                    payload[key] = value
        elif content_type.startswith("application/json"):
            try:
                payload = json.loads(body.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                return self._error(400, "invalid JSON body")
            if payload.get("audio_b64"):
                audio = base64.b64decode(str(payload["audio_b64"]))
        elif body:
            audio = body

        want_audio_out = str(payload.get("speak", "true")).lower() in ("1", "true", "yes", "on")
        if str(payload.get("response_audio", "false")).lower() in ("0", "false", "no", "off"):
            want_audio_out = False

        if not self.lock.acquire(blocking=False):
            return self._error(429, "another request is being processed - the speech "
                                    "engines are single-threaded")
        payload_out: dict[str, Any] | None = None
        error_out: tuple[int, str] | None = None
        try:
            text = str(payload.get("text") or payload.get("prompt") or payload.get("input") or "")
            transcript = ""
            if audio is not None:
                result = pipe.transcribe(audio, emit=False)
                transcript = result.text.strip()
                if not text:
                    text = transcript
            if not text:
                error_out = (400, "provide 'text' or an audio file")
            else:
                t0 = time.time()
                answer = pipe.run_turn(text=text, speak=False)
                llm_s = time.time() - t0

                audio_b64 = ""
                audio_seconds = 0.0
                if want_audio_out and answer and pipe.config.tts.engine not in ("none", ""):
                    try:
                        clip = pipe.speak_text(answer, play=False)
                        audio_b64 = base64.b64encode(clip.to_wav_bytes()).decode("ascii")
                        audio_seconds = clip.duration_s
                    except Exception as exc:
                        answer += f"\n\n[TTS failed: {exc}]"

                payload_out = {
                    "text": answer,
                    "transcript": transcript,
                    "audio_b64": audio_b64,
                    "audio_format": "wav" if audio_b64 else None,
                    "audio_seconds": round(audio_seconds, 3),
                    "timings": {"total_s": round(time.time() - t0, 3),
                                "llm_s": round(llm_s, 3),
                                **{k: round(v, 3) for k, v in (pipe.last_stats or {}).items()}},
                    "model": pipe.config.ollama.model,
                    "local": True,
                }
        finally:
            # Release before writing the response: otherwise a client that fires
            # the next request the instant it sees the last byte would race the
            # unlock and get a spurious 429.
            self.lock.release()

        if error_out:
            return self._error(*error_out)
        assert payload_out is not None
        return self._json(200, payload_out)


def _subtitle_format(result: Any, fmt: str) -> str:
    lines: list[str] = []
    if fmt == "vtt":
        lines.append("WEBVTT")
        lines.append("")
    for i, seg in enumerate(result.segments or [], start=1):
        start = _stamp(seg.start, fmt)
        end = _stamp(seg.end, fmt)
        if fmt == "srt":
            lines += [str(i), f"{start} --> {end}", seg.text.strip(), ""]
        else:
            lines += [f"{start} --> {end}", seg.text.strip(), ""]
    return "\n".join(lines)


def _stamp(seconds: float, fmt: str) -> str:
    ms = int(round((seconds - int(seconds)) * 1000))
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    sep = "." if fmt == "vtt" else ","
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


# --------------------------------------------------------------------------- #
# Server lifecycle
# --------------------------------------------------------------------------- #


class SpeechServer:
    """Thin wrapper so the GUI and the CLI can start/stop it identically."""

    def __init__(self, httpd: ThreadingHTTPServer, pipeline: VoicePipeline,
                 thread: threading.Thread) -> None:
        self.httpd = httpd
        self.pipeline = pipeline
        self.thread = thread

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def shutdown(self) -> None:
        try:
            self.httpd.shutdown()
            self.httpd.server_close()
        except OSError:
            pass
        if self.thread.is_alive():
            self.thread.join(timeout=3)


def make_server(cfg: Config, host: str = "127.0.0.1", port: int = 8787,
                pipeline: VoicePipeline | None = None) -> tuple[SpeechServer, ThreadingHTTPServer]:
    pipeline = pipeline or VoicePipeline(cfg, audio_backend=_headless_backend())
    handler = type("BoundHandler", (Handler,), {"pipeline": pipeline,
                                                "lock": threading.Lock()})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    return SpeechServer(httpd, pipeline, threading.Thread()), httpd


def start_background(cfg: Config, host: str = "127.0.0.1",
                     port: int = 8787) -> SpeechServer:
    server, httpd = make_server(cfg, host, port)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True,
                              name="ollavoice-speech-api")
    server.thread = thread
    thread.start()
    return server


def _headless_backend() -> str:
    """Pick an audio backend for the API server, degrading to a silent one.

    A server may legitimately run on a box with no microphone or speakers (a NAS,
    a container, CI): transcription and synthesis still work on bytes, so we fall
    back to the null backend instead of refusing to start.
    """
    from . import audioio

    for backend in ("sounddevice", "qt", "system"):
        try:
            return audioio.choose_backend(backend)
        except audioio.AudioError:
            continue
    return "null"


def serve(cfg: Config, host: str = "127.0.0.1", port: int = 8787,
          open_browser: bool = True) -> int:
    """Blocking serve, used by ``ollavoice serve``."""
    from .cli import bold, dim, green, yellow

    try:
        server, httpd = make_server(cfg, host, port)
    except OSError as exc:
        print(f"Could not bind {host}:{port} - {exc}")
        return 2
    thread = threading.Thread(target=httpd.serve_forever, daemon=True,
                              name="ollavoice-speech-api")
    server.thread = thread
    thread.start()

    url = f"http://{host}:{port}"
    print(bold("OllaVoice local speech API"))
    print(green(f"  listening on {url}"))
    print(dim(f"  model {cfg.ollama.model} | stt {cfg.stt.engine}/{cfg.stt.model} "
              f"| tts {cfg.tts.engine}"))
    print(dim("  endpoints:"))
    print(dim("    GET  /                     browser UI"))
    print(dim("    POST /v1/audio/transcriptions   (OpenAI-compatible)"))
    print(dim("    POST /v1/audio/speech           (OpenAI-compatible)"))
    print(dim("    POST /api/chat                  speech-in / speech-out"))
    print(dim("    GET  /health  /config  /voices"))
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(yellow("  WARNING: bound to a non-loopback address. This API has no "
                     "authentication - do not expose it to an untrusted network."))
    print(dim("  Ctrl+C to stop"))

    if open_browser:
        threading.Timer(0.6, lambda: _open_browser(url)).start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print(dim("\nshutting down"))
        server.shutdown()
        return 0


def _open_browser(url: str) -> None:
    try:
        webbrowser.open(url)
    except Exception:
        pass


if __name__ == "__main__":  # pragma: no cover
    from .config import load as _load

    raise SystemExit(serve(_load()))
