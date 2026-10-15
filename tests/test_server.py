"""The local HTTP API: routing, multipart parsing, and the OpenAI-compatible shape."""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from typing import Any

import pytest

from ollavoice import server as server_mod
from ollavoice.config import Config
from ollavoice.pipeline import VoicePipeline
from ollavoice.server import SpeechServer, parse_multipart

from tests.conftest import tone_clip
from tests.stubs import NoPlayer, StubStt, StubTts


def _make(cfg: Config) -> SpeechServer:
    """Start the server on an ephemeral port with stubbed speech engines."""
    pipeline = VoicePipeline(cfg, audio_backend="system")
    pipeline._stt = StubStt(cfg.stt)
    pipeline._tts = StubTts(cfg.tts, language=cfg.language)
    pipeline._player = NoPlayer()
    cfg.tts.engine = "piper"
    handle, httpd = server_mod.make_server(cfg, "127.0.0.1", 0, pipeline=pipeline)
    import threading

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    handle.thread = thread
    thread.start()
    return handle


@pytest.fixture
def api(cfg):
    handle = _make(cfg)
    yield handle.url, handle.pipeline
    handle.shutdown()


def _get(url: str) -> tuple[int, Any, dict]:
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers)


def _post(url: str, body: bytes, content_type: str) -> tuple[int, Any, dict]:
    req = urllib.request.Request(url, data=body, headers={"Content-Type": content_type})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers)


# --------------------------------------------------------------------------- #
# Multipart parsing
# --------------------------------------------------------------------------- #

def test_parse_multipart_reads_fields_and_files():
    boundary = "X"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="model"\r\n\r\nwhisper\r\n'
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="a.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode() + b"\x00\x01RIFFfake" + f"\r\n--{boundary}--\r\n".encode()

    parts = parse_multipart(body, f"multipart/form-data; boundary={boundary}")
    assert parts["model"] == "whisper"
    name, blob = parts["file"]
    assert name == "a.wav"
    assert blob.endswith(b"RIFFfake")


def test_parse_multipart_without_boundary_is_empty():
    assert parse_multipart(b"anything", "application/json") == {}


# --------------------------------------------------------------------------- #
# Read-only routes
# --------------------------------------------------------------------------- #

def test_health(api):
    base, pipe = api
    code, body, _ = _get(base + "/health")
    assert code == 200
    data = json.loads(body)
    assert data["status"] == "ok"
    assert data["ollama"] is True
    assert data["model"] == "testmodel:latest"
    assert data["stt"]["loaded_engine"] == "stub"


def test_config_route_hides_nothing_sensitive(api, cfg):
    base, _ = api
    code, body, _ = _get(base + "/config")
    data = json.loads(body)
    assert code == 200
    assert data["stt"]["engine"] == cfg.stt.engine     # what is configured
    assert "api_key" not in json.dumps(data)           # and nothing secret leaks


def test_voices_route(api):
    base, _ = api
    code, body, _ = _get(base + "/voices")
    assert code == 200
    assert "piper" in json.loads(body)


def test_unknown_route_404s(api):
    base, _ = api
    code, body, _ = _get(base + "/nope")
    assert code == 404
    assert "error" in json.loads(body)


def test_index_page_is_served(api):
    base, _ = api
    code, body, headers = _get(base + "/")
    assert code == 200
    assert b"OllaVoice" in body
    assert "text/html" in headers["Content-Type"]
    code, body, _ = _get(base + "/app.js")
    assert code == 200 and b"MediaRecorder" in body


def test_cors_header_is_present(api):
    base, _ = api
    _, _, headers = _get(base + "/health")
    assert headers.get("Access-Control-Allow-Origin") == "*"


# --------------------------------------------------------------------------- #
# Transcription
# --------------------------------------------------------------------------- #

def _multipart_body(fields: dict[str, str], filename: str, blob: bytes) -> tuple[bytes, str]:
    boundary = "----ollavoicetest"
    out = bytearray()
    for key, value in fields.items():
        out += (f"--{boundary}\r\nContent-Disposition: form-data; "
                f'name="{key}"\r\n\r\n{value}\r\n').encode()
    out += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f'filename="{filename}"\r\nContent-Type: audio/wav\r\n\r\n').encode()
    out += blob + f"\r\n--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


def test_transcriptions_multipart(api):
    base, _ = api
    wav = tone_clip(0.6).to_wav_bytes()
    body, ctype = _multipart_body({"response_format": "json"}, "rec.wav", wav)
    code, raw, _ = _post(base + "/v1/audio/transcriptions", body, ctype)
    assert code == 200
    data = json.loads(raw)
    assert data["text"] == "What does Ollama do?"
    assert data["engine"] == "stub"
    assert data["duration"] == pytest.approx(0.6, abs=0.05)
    assert "segments" in data


def test_transcriptions_text_format(api):
    base, _ = api
    wav = tone_clip(0.4).to_wav_bytes()
    body, ctype = _multipart_body({"response_format": "text"}, "rec.wav", wav)
    code, raw, headers = _post(base + "/v1/audio/transcriptions", body, ctype)
    assert code == 200
    assert raw.decode() == "What does Ollama do?"
    assert "text/plain" in headers["Content-Type"]


def test_transcriptions_srt_format(api):
    base, pipe = api
    pipe._stt.text = "First line."
    wav = tone_clip(0.4).to_wav_bytes()
    body, ctype = _multipart_body({"response_format": "srt"}, "rec.wav", wav)
    code, raw, headers = _post(base + "/api/transcriptions", body, ctype)
    assert code == 200
    assert "text/srt" in headers["Content-Type"]
    assert b"-->" in raw


def test_transcriptions_accepts_raw_bytes(api):
    base, _ = api
    wav = tone_clip(0.4).to_wav_bytes()
    code, raw, _ = _post(base + "/api/transcribe", wav, "audio/wav")
    assert code == 200
    assert json.loads(raw)["text"] == "What does Ollama do?"


def test_transcriptions_accepts_base64_json(api):
    base, _ = api
    wav = base64.b64encode(tone_clip(0.4).to_wav_bytes()).decode()
    code, raw, _ = _post(base + "/api/transcribe",
                         json.dumps({"audio_b64": wav}).encode(), "application/json")
    assert code == 200
    assert json.loads(raw)["text"]


def test_transcriptions_without_audio_is_a_400(api):
    base, _ = api
    code, raw, _ = _post(base + "/v1/audio/transcriptions", b"{}", "application/json")
    assert code == 400
    assert "no audio" in json.loads(raw)["error"]["message"].lower()


# --------------------------------------------------------------------------- #
# Speech synthesis
# --------------------------------------------------------------------------- #

def test_speech_returns_wav(api):
    base, pipe = api
    code, raw, headers = _post(base + "/v1/audio/speech",
                               json.dumps({"input": "Say this aloud."}).encode(),
                               "application/json")
    assert code == 200
    assert headers["Content-Type"] == "audio/wav"
    assert raw[:4] == b"RIFF"
    assert float(headers["X-Audio-Seconds"]) > 0
    assert pipe._tts.spoken, "the text should have reached the TTS engine"


def test_speech_requires_input(api):
    base, _ = api
    code, raw, _ = _post(base + "/v1/audio/speech", json.dumps({}).encode(),
                         "application/json")
    assert code == 400
    assert "input" in json.loads(raw)["error"]["message"]


def test_speech_accepts_plain_text_body(api):
    base, _ = api
    code, raw, headers = _post(base + "/api/speak", b"plain text body", "text/plain")
    assert code == 200 and raw[:4] == b"RIFF"


def test_speech_query_overrides_do_not_leak_into_the_config(api):
    """A per-request voice must not permanently change the app's voice."""
    base, pipe = api
    before_engine = pipe.config.tts.engine
    before_voice = pipe.config.tts.piper_voice
    before_rate = pipe.config.tts.rate

    # engine=none keeps it fast and download-free; the point is config isolation.
    code, raw, _ = _post(base + "/v1/audio/speech?engine=none&voice=en_GB-alan-medium&speed=1.4",
                         json.dumps({"input": "hello"}).encode(), "application/json")
    assert code == 500, "the silent engine produces no audio, which is reported, not hidden"
    assert "no audio" in json.loads(raw)["error"]["message"]

    assert pipe.config.tts.engine == before_engine
    assert pipe.config.tts.piper_voice == before_voice
    assert pipe.config.tts.rate == before_rate
    # ...and the normal path still works afterwards.
    code, raw, headers = _post(base + "/v1/audio/speech",
                               json.dumps({"input": "back to normal"}).encode(),
                               "application/json")
    assert code == 200 and raw[:4] == b"RIFF"
    assert headers.get("X-Tts-Engine") == before_engine


def test_speech_with_silent_engine_is_a_500(api):
    base, pipe = api
    pipe.config.tts.engine = "none"
    code, raw, _ = _post(base + "/v1/audio/speech",
                         json.dumps({"input": "hello"}).encode(), "application/json")
    assert code == 500
    assert "no audio" in json.loads(raw)["error"]["message"]


# --------------------------------------------------------------------------- #
# Speech-to-speech chat
# --------------------------------------------------------------------------- #

def test_chat_with_text_returns_answer_and_audio(api):
    base, _ = api
    code, raw, _ = _post(base + "/api/chat",
                         json.dumps({"text": "hello there", "response_audio": True}).encode(),
                         "application/json")
    assert code == 200
    data = json.loads(raw)
    assert data["text"].startswith("Hello.")
    assert data["local"] is True
    assert data["audio_b64"]
    audio = base64.b64decode(data["audio_b64"])
    assert audio[:4] == b"RIFF"
    assert data["timings"]["llm_s"] >= 0


def test_chat_with_audio_transcribes_first(api):
    base, pipe = api
    wav = base64.b64encode(tone_clip(0.8).to_wav_bytes()).decode()
    code, raw, _ = _post(base + "/api/chat",
                         json.dumps({"audio_b64": wav, "response_audio": False}).encode(),
                         "application/json")
    assert code == 200
    data = json.loads(raw)
    assert data["transcript"] == "What does Ollama do?"
    assert data["text"].startswith("Hello.")
    assert data["audio_b64"] == ""


def test_chat_conversation_history_persists(api):
    base, pipe = api
    for text in ("first question", "second question"):
        _post(base + "/api/chat", json.dumps({"text": text}).encode(), "application/json")
    roles = [m.role for m in pipe.history.messages]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert pipe.history.messages[2].content == "second question"


def test_chat_without_input_is_a_400(api):
    base, _ = api
    code, raw, _ = _post(base + "/api/chat", b"{}", "application/json")
    assert code == 400
    assert "text" in json.loads(raw)["error"]["message"]


def test_chat_serialises_concurrent_requests(api):
    """Two simultaneous turns must not interleave: the loser gets a clean 429."""
    import threading

    base, _ = api
    body = json.dumps({"text": "hello"}).encode()
    codes: list[int] = []

    def hit() -> None:
        code, _, _ = _post(base + "/api/chat", body, "application/json")
        codes.append(code)

    threads = [threading.Thread(target=hit) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert codes, "both requests should have answered"
    assert all(code in (200, 429) for code in codes), codes
    assert 200 in codes
    assert 500 not in codes


def test_bad_json_body_is_a_400(api):
    base, _ = api
    code, raw, _ = _post(base + "/api/chat", b"{not json", "application/json")
    assert code == 400


# --------------------------------------------------------------------------- #
# Subtitle formatting helper
# --------------------------------------------------------------------------- #

def test_subtitle_formats():
    from ollavoice.stt_engines import Segment, Transcription

    tr = Transcription(text="a b", segments=[Segment(0.0, 1.5, "a"), Segment(1.5, 62.25, "b")])
    srt = server_mod._subtitle_format(tr, "srt")
    assert srt.startswith("1\n00:00:00,000 --> 00:00:01,500\na")
    assert "00:01:02,250" in srt
    vtt = server_mod._subtitle_format(tr, "vtt")
    assert vtt.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.500" in vtt
