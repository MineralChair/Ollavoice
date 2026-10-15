"""Background workers: keep Whisper, TTS and the Ollama stream off the UI thread.

Every worker -> GUI message goes through a Qt signal, so it is queued onto the
GUI thread automatically and the UI is never touched from a worker. GUI -> worker
calls also go through signals (``request_*``), which is the idiomatic and safe way
to invoke a method on an object living in another thread.

The pipeline plays audio non-blocking and the worker waits for the audio queue to
drain at the end of a turn. That means "stop" can interrupt speech mid-sentence
instead of being frozen until the whole answer has been read out.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal

from .. import paths as _paths
from ..config import Config
from ..pipeline import PipelineError, VoicePipeline
from ..wav import AudioClip


class Signals(QObject):
    """One bundle of worker -> GUI signals."""

    log = Signal(str)                    # plain status line
    phase = Signal(str)                  # idle|listening|transcribing|thinking|speaking|error
    level = Signal(float, str)           # dBFS, vad state
    transcript = Signal(str, dict)       # final user text + meta
    token = Signal(str)                  # streamed piece of the answer
    sentence = Signal(str)               # sentence handed to TTS
    audio = Signal(str, dict)            # sentence synthesised, timing meta
    stats = Signal(dict)
    done = Signal(str, bool)             # final answer text, is_final
    error = Signal(str)
    finished = Signal()


class PipelineWorker(QObject):
    """Runs the pipeline (STT / LLM / TTS / playback) on its own thread."""

    # GUI -> worker
    request_warm = Signal()
    request_turn = Signal(object, str, object)      # clip, text, speak(bool|None)
    request_speak = Signal(str)
    request_transcribe = Signal(object)

    def __init__(self, pipeline: VoicePipeline) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.signals = Signals()
        pipeline.event_fn = self._on_event
        self.request_warm.connect(self.warm)
        self.request_turn.connect(self.run_turn)
        self.request_speak.connect(self.speak_text)
        self.request_transcribe.connect(self.transcribe_clip)

    # ------------------------------------------------------------------ #
    def _on_event(self, event) -> None:
        kind, text, data = event.kind, event.text, event.data
        try:
            if kind == "phase":
                self.signals.phase.emit(event.phase or "idle")
            elif kind == "level":
                self.signals.level.emit(float(data.get("db", -120.0)),
                                        str(data.get("state", "")))
            elif kind == "token":
                self.signals.token.emit(text)
            elif kind == "transcript":
                self.signals.transcript.emit(text, data)
            elif kind == "sentence":
                self.signals.sentence.emit(text)
            elif kind == "audio":
                self.signals.audio.emit(text, data)
            elif kind == "stats":
                self.signals.stats.emit(data)
            elif kind == "done":
                self.signals.done.emit(text, bool(data.get("final", False)))
            elif kind == "error":
                self.signals.error.emit(text)
            elif kind == "info":
                self.signals.log.emit(text)
        except RuntimeError:
            pass  # interpreter shutting down

    # ------------------------------------------------------------------ #
    def warm(self) -> None:
        try:
            timings = self.pipeline.warm()
            if timings:
                self.signals.log.emit("engines loaded: " + json.dumps(
                    {k: round(v, 2) for k, v in timings.items()}))
        except Exception as exc:
            self.signals.error.emit(f"warm-up failed: {exc}")
        finally:
            self.signals.finished.emit()

    def run_turn(self, clip: object = None, text: str = "", speak: object = None) -> None:
        answer = ""
        try:
            answer = self.pipeline.run_turn(
                clip if isinstance(clip, AudioClip) else None,
                text=text or "",
                speak=None if speak is None else bool(speak),
                blocking_playback=False,
            )
            # Let the last sentences finish playing before we unlock the UI.
            self.pipeline.wait_for_playback(timeout_s=180.0)
        except PipelineError as exc:
            self.signals.error.emit(str(exc))
        except Exception as exc:
            self.signals.error.emit(f"{type(exc).__name__}: {exc}")
            self.signals.log.emit(traceback.format_exc(limit=8))
        finally:
            self.signals.done.emit(answer, True)
            self.signals.finished.emit()

    def speak_text(self, text: str) -> None:
        try:
            self.pipeline.speak_text(text, blocking=False)
            self.pipeline.wait_for_playback(timeout_s=180.0)
        except Exception as exc:
            self.signals.error.emit(f"speech output failed: {exc}")
        finally:
            self.signals.finished.emit()

    def transcribe_clip(self, clip: object) -> None:
        try:
            if isinstance(clip, AudioClip):
                self.pipeline.transcribe(clip)
        except Exception as exc:
            self.signals.error.emit(f"transcription failed: {exc}")
        finally:
            self.signals.finished.emit()


class RefreshWorker(QObject):
    """Polls Ollama for the model list and gathers engine availability."""

    request = Signal()
    models = Signal(list)
    status = Signal(dict)
    error = Signal(str)
    finished = Signal()

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.request.connect(self.run)

    def run(self) -> None:
        from .. import stt_engines, tts_engines
        from ..ollama_client import OllamaClient, OllamaError

        try:
            client = OllamaClient(self.config.ollama.host, self.config.ollama.api_key,
                                  timeout=6)
            alive = client.is_alive()
            models = client.model_names() if alive else []
            self.models.emit(models)
            self.status.emit({
                "alive": alive,
                "version": client.version() if alive else "",
                "models": models,
                "running": client.ps() if alive else [],
                "stt_engines": stt_engines.available_engines(),
                "tts_engines": tts_engines.available_engines(),
                "voices": tts_engines.installed_voices(),
            })
        except OllamaError as exc:
            self.error.emit(str(exc))
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.finished.emit()


class ThreadHandle:
    """Owns a QThread + worker pair, and tears it down cleanly on exit."""

    def __init__(self, worker: QObject, name: str = "ollavoice-worker") -> None:
        self.thread = QThread()
        self.thread.setObjectName(name)
        self.worker = worker
        self.worker.moveToThread(self.thread)
        self.thread.start()

    @property
    def signals(self) -> Signals:
        return self.worker.signals  # type: ignore[attr-defined]

    def stop(self, timeout_ms: int = 4000) -> None:
        try:
            self.thread.quit()
            if not self.thread.wait(timeout_ms):
                self.thread.terminate()
                self.thread.wait(500)
        except RuntimeError:
            pass
        finally:
            try:
                self.worker.deleteLater()
            except RuntimeError:
                pass


def save_turn_to_transcript(entry: dict[str, Any], session: str = "") -> Path | None:
    """Append one turn to today's JSONL transcript (opt-in, stays on disk)."""
    try:
        directory = _paths.transcripts_dir()
        directory.mkdir(parents=True, exist_ok=True)
        day = time.strftime("%Y-%m-%d")
        path = directory / (f"{day}-{session}.jsonl" if session else f"{day}.jsonl")
        entry = dict(entry)
        entry.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return path
    except OSError:
        return None
