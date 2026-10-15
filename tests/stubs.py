"""Deterministic stand-ins for the speech engines, shared by the pipeline tests.

The real Whisper/Piper/Kokoro runs are in ``test_integration.py``; these stubs let
the orchestration tests be exact, fast and hardware-free.
"""

from __future__ import annotations

import pathlib
from typing import Iterator

from ollavoice.config import Config
from ollavoice.pipeline import Event, VoicePipeline
from ollavoice.stt_engines import Segment, SttEngine, Transcription
from ollavoice.tts_engines import SynthesisResult, TtsEngine
from ollavoice.wav import AudioClip

from tests.conftest import tone_clip

STUB_TRANSCRIPT = "What does Ollama do?"
STUB_ANSWER = "Hello. I am a local model. Everything stayed on this machine."


class StubStt(SttEngine):
    """Deterministic stand-in for Whisper."""

    name = "stub"

    def __init__(self, settings, *, app_language: str = "") -> None:
        super().__init__(settings, app_language=app_language)
        self.model_name = "stub"
        self.calls: list[float] = []
        self.text = STUB_TRANSCRIPT

    def load(self) -> None:
        self._loaded = True

    def transcribe(self, clip, *, language=None, initial_prompt=None) -> Transcription:
        # Go through the same input-normalising helper the real engines use, so
        # the stub sees AudioClip, paths, WAV bytes and encoded bytes alike.
        from ollavoice.stt_engines import _as_audio_source
        from ollavoice.wav import read_wav

        source, cleanup = _as_audio_source(clip)
        try:
            if isinstance(source, pathlib.Path):
                duration = read_wav(source).duration_s
            elif isinstance(clip, AudioClip):
                duration = clip.duration_s
            else:
                duration = 0.0
        except Exception:
            duration = 0.0
        finally:
            if cleanup and isinstance(source, pathlib.Path):
                source.unlink(missing_ok=True)

        self.calls.append(duration)
        half = max(duration / 2.0, 0.01)
        return Transcription(text=self.text, language="en", language_probability=0.99,
                             duration_s=duration, decode_s=0.05, engine=self.name,
                             model=self.model_name,
                             segments=[Segment(0.0, half, self.text),
                                       Segment(half, duration, self.text)])


class StubTts(TtsEngine):
    """Deterministic stand-in for Piper/Kokoro.

    Records every string it is asked to speak and returns a short tone, so tests
    can assert both on the sentence splitting and on the audio maths.
    """

    name = "stub"

    def __init__(self, settings, *, language: str = "en") -> None:
        super().__init__(settings, language=language)
        self.voice_name = "stub-voice"
        self.spoken: list[str] = []

    def load(self) -> None:
        self._loaded = True

    def synthesize(self, text: str) -> SynthesisResult:
        clip = tone_clip(min(1.5, max(0.2, len(text) / 40.0)), amplitude=0.3)
        return SynthesisResult(clip, 0.01, self.name, self.voice_name)

    def synthesize_stream(self, text: str) -> Iterator[AudioClip]:
        self.spoken.append(self._prepare_text(text))
        yield tone_clip(0.25, amplitude=0.3)


class NoPlayer:
    """Stands in for the audio backend: records what would have been played."""

    backend = "null"

    def __init__(self) -> None:
        self.played: list[AudioClip] = []
        self.is_playing = False

    def play(self, clip, *, blocking: bool = False) -> None:
        self.played.append(clip)

    def stop(self) -> None:
        pass


def make_pipeline(cfg: Config) -> tuple[VoicePipeline, list[Event], NoPlayer]:
    """A pipeline wired to the stubs, plus the event log and the fake player."""
    # The stub voice engine stands in for a real one, so the config has to say
    # "speech output is on" or the pipeline would (correctly) skip it.
    cfg.tts.engine = "piper"
    cfg.tts.auto_speak = True
    events: list[Event] = []
    pipeline = VoicePipeline(cfg, audio_backend="system", event_fn=events.append)
    pipeline._stt = StubStt(cfg.stt, app_language=cfg.language)
    pipeline._tts = StubTts(cfg.tts, language=cfg.language)
    player = NoPlayer()
    pipeline._player = player
    return pipeline, events, player
