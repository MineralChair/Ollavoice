"""The speech queue: synthesis and playback must never block generation."""

from __future__ import annotations

import threading
import time

import pytest

from ollavoice.pipeline import VoicePipeline
from ollavoice.wav import AudioClip

from tests.conftest import tone_clip
from tests.stubs import STUB_ANSWER, NoPlayer, StubStt, StubTts, make_pipeline


@pytest.fixture
def rig(cfg):
    """(pipeline, events, player) - everything stubbed except the HTTP client."""
    return make_pipeline(cfg)


class SlowTts(StubTts):
    """Pretends to be Kokoro on a small CPU: slower than realtime."""

    def __init__(self, settings, *, language: str = "en", delay: float = 0.25) -> None:
        super().__init__(settings, language=language)
        self.delay = delay
        self.synth_started: list[float] = []

    def synthesize_stream(self, text: str):
        self.synth_started.append(time.monotonic())
        time.sleep(self.delay)
        self.spoken.append(self._prepare_text(text))
        yield tone_clip(0.2, amplitude=0.3)


@pytest.fixture
def slow_rig(cfg):
    events = []
    pipeline = VoicePipeline(cfg, audio_backend="null", event_fn=events.append)
    cfg.tts.engine = "piper"
    pipeline._stt = StubStt(cfg.stt)
    pipeline._tts = SlowTts(cfg.tts, language=cfg.language, delay=0.25)
    player = NoPlayer()
    pipeline._player = player
    return pipeline, events, player


# --------------------------------------------------------------------------- #

def test_generation_is_not_blocked_by_a_slow_voice(slow_rig):
    """The whole point of the queue: tokens keep arriving while TTS catches up."""
    pipe, events, player = slow_rig
    t0 = time.monotonic()
    answer = pipe.run_turn(text="say something", speak=True, blocking_playback=False)
    generation_s = time.monotonic() - t0

    assert answer == STUB_ANSWER
    pipe.wait_for_speech(timeout_s=15)          # synthesis runs on its own thread
    n_sentences = sum(1 for e in events if e.kind == "sentence")
    # Two sentences at 0.25 s each would add >= 0.5 s if synthesis were inline.
    assert generation_s < 0.4, f"generation took {generation_s:.2f}s - was it blocked?"
    assert n_sentences >= 2
    assert len(pipe._tts.spoken) == n_sentences
    assert len(player.played) == n_sentences
    assert " ".join(pipe._tts.spoken).split() == STUB_ANSWER.split()


def test_blocking_playback_waits_for_the_voice(slow_rig):
    pipe, _, player = slow_rig
    t0 = time.monotonic()
    pipe.run_turn(text="say something", speak=True, blocking_playback=True)
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.4, "a blocking turn must not return before the voice is done"
    assert player.played


def test_wait_for_speech_returns_promptly_when_idle(rig):
    pipe, _, _ = rig
    t0 = time.monotonic()
    assert pipe.wait_for_speech(timeout_s=5) is True
    assert time.monotonic() - t0 < 0.5


def test_wait_for_speech_times_out_on_a_stuck_queue(cfg):
    events = []
    pipe = VoicePipeline(cfg, audio_backend="null", event_fn=events.append)
    cfg.tts.engine = "piper"
    pipe._tts = SlowTts(cfg.tts, language=cfg.language, delay=5.0)
    pipe._player = NoPlayer()
    pipe.enqueue_speech("this will take a while")
    time.sleep(0.1)
    assert pipe.wait_for_speech(timeout_s=0.3) is False
    pipe.cancel_speech()
    assert pipe.wait_for_speech(timeout_s=2) is True


def test_cancel_speech_drops_the_queue(cfg):
    events = []
    pipe = VoicePipeline(cfg, audio_backend="null", event_fn=events.append)
    cfg.tts.engine = "piper"
    pipe._tts = SlowTts(cfg.tts, language=cfg.language, delay=0.2)
    player = NoPlayer()
    pipe._player = player

    for i in range(5):
        pipe.enqueue_speech(f"sentence number {i}")
    time.sleep(0.05)
    pipe.cancel_speech()
    pipe.wait_for_speech(timeout_s=2)
    assert len(player.played) < 5, "queued sentences should have been dropped"


def test_speech_queue_thread_is_reused_not_leaked(cfg):
    pipe = VoicePipeline(cfg, audio_backend="null")
    cfg.tts.engine = "piper"
    pipe._tts = StubTts(cfg.tts, language=cfg.language)
    pipe._player = NoPlayer()

    before = threading.active_count()
    for _ in range(5):
        pipe.start_speech_queue()
    assert threading.active_count() <= before + 1

    pipe.enqueue_speech("hello")
    assert pipe.wait_for_speech(timeout_s=5)
    pipe.cancel_speech()


def test_tts_errors_do_not_kill_the_turn(rig, monkeypatch):
    """A voice failure must be reported, not crash the conversation."""
    pipe, events, _ = rig

    def boom(_text):
        raise RuntimeError("voice exploded")

    monkeypatch.setattr(pipe._tts, "synthesize_stream", boom)
    answer = pipe.run_turn(text="say something", speak=True, blocking_playback=False)
    pipe.wait_for_speech(timeout_s=5)
    assert answer == STUB_ANSWER, "the text answer survives a TTS failure"
    assert any("voice exploded" in e.text for e in events if e.kind == "error")


def test_reload_engines_stops_the_speech_thread(slow_rig):
    pipe, _, _ = slow_rig
    pipe.enqueue_speech("a slow sentence")
    time.sleep(0.05)
    thread = pipe._speech_thread
    pipe.reload_engines()
    assert pipe._speech_thread is None
    if thread is not None and thread.is_alive():
        thread.join(timeout=3)
    assert thread is None or not thread.is_alive()


def test_silent_engine_never_starts_a_thread(cfg):
    cfg.tts.engine = "none"
    pipe = VoicePipeline(cfg, audio_backend="null")
    pipe.enqueue_speech("nothing to say")
    assert pipe._speech_thread is None
    assert pipe.wait_for_speech(timeout_s=1) is True


def test_first_audio_timing_is_reported(rig):
    pipe, events, _ = rig
    pipe.run_turn(text="say something", speak=True, blocking_playback=True)
    stats = [e.data for e in events if e.kind == "stats"]
    assert stats and "tts_first_s" in stats[-1]
    assert stats[-1]["tts_first_s"] >= 0


def test_null_audio_backend_records_silence_and_discards_playback():
    from ollavoice import audioio

    rec = audioio.make_recorder("null", sample_rate=16000)
    rec.start()
    time.sleep(0.05)
    clip = rec.stop()
    assert isinstance(clip, AudioClip)
    assert len(clip.samples) == 0

    player = audioio.make_player("null")
    player.play(tone_clip(0.1))
    assert player.clips and not player.is_playing
    assert audioio.choose_backend("null") == "null"
