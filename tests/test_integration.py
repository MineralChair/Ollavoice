"""Real-model integration tests: actual Piper synthesis and actual Whisper.

These download nothing on their own. They run only when the models are already
on disk (i.e. after ``ollavoice setup``), or when ``OLLAVOICE_IT=1`` is set to
allow fetching them. Otherwise they skip, so CI stays fast and offline.

    pytest -m integration            # run them
    OLLAVOICE_IT=1 pytest            # allow downloads too
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from ollavoice import downloads, paths as _paths
from ollavoice.config import SttSettings, TtsSettings
from ollavoice.pipeline import Event, VoicePipeline
from ollavoice.stt_engines import FasterWhisperStt
from ollavoice.tts_engines import PiperTts
from ollavoice.wav import AudioClip

ALLOW_DOWNLOAD = os.environ.get("OLLAVOICE_IT", "") in {"1", "true", "yes"}
VOICE = "en_US-lessac-medium"
WHISPER = "tiny.en"

pytestmark = pytest.mark.integration


def _piper_available() -> bool:
    try:
        import piper  # noqa: F401
    except ImportError:
        return False
    if (_paths.piper_dir() / f"{VOICE}.onnx").exists():
        return True
    return ALLOW_DOWNLOAD


def _whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    root = _paths.whisper_dir()
    if root.exists() and any(root.rglob("*tiny*")):
        return True
    return ALLOW_DOWNLOAD


requires_piper = pytest.mark.skipif(
    not _piper_available(), reason=f"piper or the {VOICE} voice is not installed")
requires_whisper = pytest.mark.skipif(
    not _whisper_available(), reason=f"faster-whisper or {WHISPER} is not installed")


def _piper_engine() -> PiperTts:
    settings = TtsSettings(engine="piper", piper_voice=VOICE, rate=1.0)
    return PiperTts(settings, language="en", auto_download=True)


def _whisper_engine() -> FasterWhisperStt:
    settings = SttSettings(engine="faster-whisper", model=WHISPER, language="en",
                           device="cpu", compute_type="int8", beam_size=1,
                           cpu_threads=2, vad_filter=False)
    return FasterWhisperStt(settings)


# --------------------------------------------------------------------------- #


@requires_piper
def test_piper_synthesises_speech():
    engine = _piper_engine()
    engine.load()
    result = engine.synthesize("The local voice engine is working correctly.")
    assert result.clip.sample_rate > 0
    assert result.clip.duration_s > 1.0
    assert result.clip.peak_db() > -30, "audio is essentially silence"
    assert result.realtime_factor > 0


@requires_piper
def test_piper_streams_in_pieces():
    engine = _piper_engine()
    engine.load()
    pieces = list(engine.synthesize_stream("First part. Second part. Third part."))
    assert len(pieces) >= 1
    assert all(len(p.samples) for p in pieces)
    assert all(p.sample_rate == pieces[0].sample_rate for p in pieces)


@requires_piper
def test_piper_rate_setting_changes_the_duration():
    engine = _piper_engine()
    engine.load()
    text = "Speaking rate should change how long this sentence takes to say."
    engine.settings.rate = 0.8
    slow = engine.synthesize(text).clip.duration_s
    engine.settings.rate = 1.4
    fast = engine.synthesize(text).clip.duration_s
    assert slow > fast * 1.1


@requires_piper
def test_piper_writes_a_readable_wav(tmp_path):
    engine = _piper_engine()
    engine.load()
    clip = engine.synthesize("Write me to disk.").clip
    path = tmp_path / "voice.wav"
    clip.write_wav(path)
    from ollavoice.wav import read_wav

    back = read_wav(path)
    assert back.duration_s == pytest.approx(clip.duration_s, abs=0.01)


@requires_whisper
def test_whisper_transcribes_a_known_clip(tmp_path):
    from tests.conftest import tone_clip

    engine = _whisper_engine()
    # A sine wave has no words; Whisper must still return cleanly and quickly.
    clip = tone_clip(1.0)
    path = tmp_path / "tone.wav"
    clip.write_wav(path)
    result = engine.transcribe(path)
    assert isinstance(result.text, str)
    assert result.decode_s >= 0
    assert result.engine == "faster-whisper"


@requires_piper
@requires_whisper
def test_full_local_loop_speech_to_text_and_back():
    """The core promise: local voice in, local text out, no network at runtime."""
    sentence = "Everything about this assistant runs on my own computer."
    tts = _piper_engine()
    tts.load()
    clip = tts.synthesize(sentence).clip

    stt = _whisper_engine()
    stt.load()
    result = stt.transcribe(clip)

    heard = " ".join(result.text.lower().split())
    spoken = " ".join(sentence.lower().split())
    words = [w.strip(".,") for w in spoken.split()]
    hits = sum(1 for w in words if w in heard)
    assert hits >= len(words) * 0.6, (
        f"Whisper only recovered {hits}/{len(words)} words.\n"
        f"  spoken: {spoken}\n  heard : {heard}"
    )
    # tiny.en is allowed to mangle proper nouns, but it must be faster than realtime.
    assert result.realtime_factor < 3.0


@requires_piper
def test_pipeline_speaks_through_the_real_voice(cfg):
    """VoicePipeline with a real TTS engine and a silent player."""
    # The cfg fixture forces offline mode; this test needs the real downloader.
    os.environ.pop("OLLAVOICE_OFFLINE", None)
    cfg.tts.engine = "piper"
    cfg.tts.piper_voice = VOICE
    events: list[Event] = []
    pipeline = VoicePipeline(cfg, audio_backend="system", event_fn=events.append)
    pipeline._tts = _piper_engine()

    class _Recorder:
        def __init__(self) -> None:
            self.clips: list[AudioClip] = []
            self.is_playing = False

        def play(self, clip, *, blocking: bool = False) -> None:
            self.clips.append(clip)

        def stop(self) -> None:
            pass

    player = _Recorder()
    pipeline._player = player

    answer = pipeline.run_turn(text="Say one short sentence.", speak=True)
    pipeline.wait_for_speech(timeout_s=120)     # synthesis runs on its own thread
    assert answer
    assert player.clips, "the real voice engine should have produced audio"
    assert sum(c.duration_s for c in player.clips) > 0.5
    assert any(e.kind == "sentence" for e in events)
    assert any(e.kind == "audio" for e in events)


@requires_piper
def test_setup_verify_path_reports_no_problems(cfg, capsys):
    """The setup wizard's verifier should pass with a real voice installed."""
    from ollavoice import setup as setup_mod

    os.environ.pop("OLLAVOICE_OFFLINE", None)

    cfg.tts.engine = "piper"
    cfg.tts.piper_voice = VOICE
    cfg.stt.engine = "faster-whisper"
    cfg.stt.model = WHISPER
    problems = setup_mod._verify(cfg, offline=False)
    out = capsys.readouterr().out
    assert "TTS piper" in out
    # Ollama/Whisper may be missing in CI; only complain about TTS problems.
    assert not [p for p in problems if p.startswith("TTS")], problems


def test_offline_bundle_roundtrip(tmp_home, tmp_path):
    """Models copied into a bundle must be restorable on a 'new machine'."""
    src = tmp_home / "models" / "piper"
    src.mkdir(parents=True, exist_ok=True)
    (src / f"{VOICE}.onnx").write_bytes(b"fake-weights")
    (src / f"{VOICE}.onnx.json").write_bytes(b"{}")

    bundle = downloads.make_offline_bundle(tmp_path / "bundle")
    assert (bundle / "piper" / f"{VOICE}.onnx").exists()

    # Simulate the second machine: empty models dir + the bundle as vendor.
    shutil.rmtree(tmp_home / "models" / "piper")
    vendor = tmp_home / "vendor" / "models"
    vendor.mkdir(parents=True)
    shutil.copy(bundle / "piper" / f"{VOICE}.onnx", vendor / f"{VOICE}.onnx")
    shutil.copy(bundle / "piper" / f"{VOICE}.onnx.json", vendor / f"{VOICE}.onnx.json")

    os.environ["OLLAVOICE_OFFLINE"] = "1"
    try:
        got = downloads.ensure_piper_voice(VOICE, _paths.piper_dir())
    finally:
        os.environ.pop("OLLAVOICE_OFFLINE", None)
    assert Path(got).exists()
    assert Path(got).read_bytes() == b"fake-weights"
