"""End-to-end pipeline tests using stub engines and the fake Ollama server.

The point is to prove the *orchestration*: audio in -> transcript -> streamed
answer -> per-sentence audio out, with the right event order and no lost text.
Real Whisper/Piper runs live in ``test_integration.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from ollavoice.ollama_client import OllamaError
from ollavoice.pipeline import PipelineError
from ollavoice.wav import AudioClip

from tests.conftest import tone_clip
from tests.stubs import STUB_ANSWER, STUB_TRANSCRIPT, make_pipeline  # noqa: F401


@pytest.fixture
def rig(cfg):
    """(pipeline, events, player) - everything stubbed except the HTTP client."""
    return make_pipeline(cfg)


# --------------------------------------------------------------------------- #
# Typed turns
# --------------------------------------------------------------------------- #

def test_typed_turn_streams_the_answer(rig):
    pipe, events, _ = rig
    answer = pipe.run_turn(text="hello there", speak=False)

    kinds = [e.kind for e in events]
    assert "transcript" in kinds, "typed text is published like a transcript"
    assert "token" in kinds
    assert any(e.kind == "done" and e.data.get("final") for e in events)

    tokens = "".join(e.text for e in events if e.kind == "token")
    assert tokens == answer
    assert answer == STUB_ANSWER
    assert pipe.history.messages[-1].role == "assistant"
    assert pipe.history.messages[-2].role == "user"
    assert pipe.history.messages[-2].content == "hello there"


def test_stats_event_carries_timings(rig):
    pipe, events, _ = rig
    pipe.run_turn(text="hi", speak=False)
    stats = [e.data for e in events if e.kind == "stats"]
    assert stats and "llm_s" in stats[-1]
    assert pipe.last_stats.get("ttft_s", 1) <= pipe.last_stats["llm_s"] + 1e-6


def test_history_grows_and_is_trimmed(rig, cfg):
    pipe, _, _ = rig
    cfg.ollama.history_turns = 2
    pipe.history.turns = 2
    for i in range(5):
        pipe.run_turn(text=f"message {i}", speak=False)
    assert len(pipe.history.messages) == 4
    assert pipe.history.messages[0].content == "message 3"


# --------------------------------------------------------------------------- #
# Spoken turns
# --------------------------------------------------------------------------- #

def test_audio_turn_transcribes_then_answers(rig):
    pipe, events, _ = rig
    answer = pipe.run_turn(clip=tone_clip(1.0), speak=False)

    assert pipe._stt.calls == [pytest.approx(1.0, abs=0.05)]
    transcripts = [e for e in events if e.kind == "transcript"]
    assert transcripts and transcripts[0].text == STUB_TRANSCRIPT
    assert transcripts[0].data["model"] == "stub"
    assert answer == STUB_ANSWER
    assert pipe.history.messages[0].content == STUB_TRANSCRIPT


def test_speaking_happens_sentence_by_sentence(rig):
    pipe, events, player = rig
    pipe.run_turn(text="say something", speak=True)
    pipe.wait_for_speech()          # synthesis runs on its own thread

    spoken = pipe._tts.spoken
    assert spoken, "the voice engine should have been fed sentences"
    # The answer reached the voice engine in pieces rather than one blob at the
    # end; every piece was played and nothing was lost or duplicated.
    assert len(spoken) >= 2, f"expected sentence-level streaming, got {spoken}"
    assert len(player.played) == len(spoken)
    assert " ".join(spoken).split() == STUB_ANSWER.split()
    assert [e.kind for e in events].count("sentence") == len(spoken)
    assert [e.kind for e in events].count("audio") == len(spoken)
    # The first piece is shorter than the whole answer - that is the latency win.
    assert len(spoken[0]) < len(STUB_ANSWER)


def test_speaking_can_be_switched_off(rig):
    pipe, _, player = rig
    pipe.run_turn(text="say something", speak=False)
    assert pipe._tts.spoken == []
    assert player.played == []


def test_mic_is_muted_while_speaking(rig, cfg):
    pipe, _, player = rig
    cfg.audio.mute_mic_while_speaking = True
    pipe.run_turn(text="say something", speak=True)
    pipe.wait_for_speech()
    assert player.played
    assert not pipe._mic_muted.is_set(), "the mute flag must be released afterwards"


# --------------------------------------------------------------------------- #
# Degenerate input
# --------------------------------------------------------------------------- #

def test_silence_is_not_sent_to_the_model(rig):
    pipe, events, _ = rig
    pipe._stt.text = "   "
    assert pipe.run_turn(clip=tone_clip(0.5), speak=False) == ""
    assert pipe.history.messages == []
    assert any("did not catch" in e.text for e in events if e.kind == "info")


def test_whisper_hallucination_is_dropped(rig):
    pipe, _, _ = rig
    pipe._stt.text = "you"
    assert pipe.run_turn(clip=tone_clip(0.5), speak=False) == ""


def test_empty_clip_is_rejected_before_any_work(rig):
    pipe, _, _ = rig
    empty = AudioClip(np.zeros(0, dtype=np.int16), 16000)
    with pytest.raises(PipelineError):
        pipe.run_turn(clip=empty, speak=False)
    assert pipe._stt.calls == []


def test_no_audio_and_no_text_is_an_error(rig):
    pipe, _, _ = rig
    with pytest.raises(PipelineError):
        pipe.run_turn(speak=False)


# --------------------------------------------------------------------------- #
# Failure modes
# --------------------------------------------------------------------------- #

def test_missing_model_gives_an_actionable_error(cfg):
    cfg.ollama.model = ""
    pipe, _, _ = make_pipeline(cfg)
    with pytest.raises(PipelineError) as exc:
        pipe.run_turn(text="hello", speak=False)
    assert "ollavoice config set ollama.model" in str(exc.value)


def test_ollama_failure_rolls_back_the_user_message(rig, monkeypatch):
    pipe, events, _ = rig

    def boom(*args, **kwargs):
        raise OllamaError("model exploded")

    monkeypatch.setattr(pipe.ollama, "chat", boom)
    with pytest.raises(PipelineError):
        pipe.run_turn(text="hello", speak=False)
    assert pipe.history.messages == [], "a failed turn must not leave a dangling user message"
    assert any("exploded" in e.text for e in events if e.kind == "error")


def test_stt_failure_surfaces_as_an_event(rig, monkeypatch):
    pipe, events, _ = rig

    def boom(*args, **kwargs):
        raise RuntimeError("no microphone")

    monkeypatch.setattr(pipe._stt, "transcribe", boom)
    with pytest.raises(PipelineError):
        pipe.run_turn(clip=tone_clip(0.5), speak=False)
    assert any("no microphone" in e.text for e in events if e.kind == "error")


def test_cancel_stops_the_turn(rig, monkeypatch):
    pipe, events, _ = rig
    original = pipe.ollama.chat

    def chat_then_cancel(*args, **kwargs):
        stream = original(*args, **kwargs)
        pipe.cancel()          # the user hit Stop as the stream was created
        return stream

    monkeypatch.setattr(pipe.ollama, "chat", chat_then_cancel)
    answer = pipe.run_turn(text="hello", speak=True)
    assert answer == ""
    assert any("cancel" in e.text for e in events if e.kind == "info")


def test_busy_pipeline_refuses_a_second_turn(rig):
    pipe, _, _ = rig
    pipe._busy.set()
    with pytest.raises(PipelineError):
        pipe.run_turn(text="hello", speak=False)
    pipe._busy.clear()


# --------------------------------------------------------------------------- #
# Engine lifecycle
# --------------------------------------------------------------------------- #

def test_warm_reports_load_times(rig):
    pipe, _, _ = rig
    timings = pipe.warm()
    assert "tts_load_s" in timings and "stt_load_s" in timings
    assert pipe._stt.loaded and pipe._tts.loaded


def test_reload_engines_drops_the_cache(rig, cfg):
    pipe, _, _ = rig
    pipe.warm()
    first = pipe._stt
    pipe.reload_engines()
    assert pipe._stt is None
    assert pipe.stt is not first          # rebuilt from config, not the stub


def test_speak_text_returns_audio_without_a_turn(rig):
    pipe, _, _ = rig
    clip = pipe.speak_text("Read this aloud.", blocking=False)
    assert clip.duration_s > 0
    assert pipe.history.messages == []


def test_silent_tts_engine_short_circuits(rig):
    pipe, events, player = rig
    pipe.config.tts.engine = "none"
    clip = pipe.speak_text("nothing", blocking=False)
    assert len(clip.samples) == 0
    assert player.played == []
    assert any("disabled" in e.text for e in events if e.kind == "info")


def test_status_reports_the_world(rig):
    pipe, _, _ = rig
    status = pipe.status()
    assert status["ollama_alive"] is True
    assert status["model"] == "testmodel:latest"
    assert status["stt"]["engine"] == "stub"
    assert status["tts"]["engine"] == "stub"      # the stub stands in for piper
    assert pipe.config.tts.engine == "piper"      # while the config still says piper


def test_ensure_model_detects_a_missing_one(rig):
    pipe, _, _ = rig
    assert pipe.ensure_model() == "testmodel:latest"
    pipe.config.ollama.model = "not-installed:latest"
    with pytest.raises(PipelineError) as exc:
        pipe.ensure_model()
    assert "ollama pull not-installed" in str(exc.value)
    assert pipe.ensure_model(pull=True) == "not-installed:latest"


def test_readable_error_shortens_tracebacks():
    from ollavoice.pipeline import readable_error

    assert "not running" in readable_error(Exception("Cannot reach Ollama at x"))
    assert len(readable_error(Exception("y" * 900))) < 400
