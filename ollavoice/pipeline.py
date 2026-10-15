"""The voice pipeline: microphone -> Whisper -> Ollama -> TTS -> speakers.

This module contains no GUI code. It is driven by the desktop app, the CLI and
the local HTTP server alike, and it reports what it is doing through a single
callback that receives :class:`Event` objects - so a UI just has to render them.

The important design decision here is **sentence-level streaming**: as Ollama
emits tokens we split them into sentences and hand each sentence to the TTS
engine immediately. That turns "wait 12 s, then hear 12 s of audio" into "hear
audio after ~1 s", which is the difference between a toy and something you can
actually talk to.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import audioio
from .config import Config
from .ollama_client import ChatHistory, OllamaClient, OllamaError
from .stt_engines import SttEngine, Transcription, create_engine as create_stt
from .textsplit import SentenceSplitter
from .tts_engines import TtsEngine, create_engine as create_tts
from .vad import GateState, SpeechGate, VadSettingsLite, trim_clip
from .wav import AudioClip


class Phase(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"


@dataclass
class Event:
    kind: str                       # see KINDS below
    text: str = ""
    phase: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return f"Event({self.kind}, text={self.text[:40]!r}, data={self.data})"


KINDS = (
    "phase",        # pipeline moved to a new phase
    "level",        # mic level in dBFS (for the meter)
    "partial",      # interim transcript / interim answer text
    "transcript",   # final transcript of the user's speech
    "token",        # a piece of the assistant answer (streamed)
    "sentence",     # a complete sentence was queued for speech
    "audio",        # a synthesised AudioClip is ready to play
    "done",         # turn finished
    "info",         # status/log line
    "error",        # something failed
    "stats",        # timing info for the status bar
)

EventFn = Callable[[Event], None]


class CancelledError_(RuntimeError):
    pass


class PipelineError(RuntimeError):
    pass


class VoicePipeline:
    """Owns the engines and the conversation, runs one turn at a time."""

    def __init__(self, config: Config, *, audio_backend: str = "auto",
                 event_fn: EventFn | None = None, auto_download: bool = True) -> None:
        self.config = config
        self.event_fn = event_fn or (lambda _e: None)
        self.audio_backend = audio_backend
        self.auto_download = auto_download

        self.ollama = OllamaClient(config.ollama.host, config.ollama.api_key,
                                   config.ollama.timeout_s)
        self.history = ChatHistory(turns=config.ollama.history_turns,
                                   system=config.ollama.system_prompt)

        self._stt: SttEngine | None = None
        self._tts: TtsEngine | None = None
        self._player: Any = None
        self._recorder: audioio.Recorder | None = None
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._busy = threading.Event()
        self._mic_muted = threading.Event()
        self._tts_lock = threading.RLock()      # one synthesis at a time
        self._speech_q: "queue.Queue[str]" = queue.Queue()
        self._speech_stop = threading.Event()
        self._speech_busy = threading.Event()
        self._speech_thread: threading.Thread | None = None
        self.last_transcription: Transcription | None = None
        self.last_stats: dict[str, float] = {}
        self._first_audio_at = 0.0

    # ------------------------------------------------------------------ #
    # Events
    # ------------------------------------------------------------------ #
    def emit(self, kind: str, *, text: str = "", phase: str = "",
             data: dict[str, Any] | None = None) -> None:
        """Publish one pipeline event to the UI.

        A raising callback must never be able to kill a turn, so this swallows
        everything the listener does.
        """
        try:
            self.event_fn(Event(kind=kind, text=text, phase=phase, data=dict(data or {})))
        except Exception:
            pass

    def _set_phase(self, phase: Phase) -> None:
        self.emit("phase", phase=phase.value)

    @property
    def is_busy(self) -> bool:
        return self._busy.is_set()

    def cancel(self) -> None:
        """Stop the current turn: interrupt playback, drop pending TTS work."""
        self._cancel.set()
        self.cancel_speech()
        rec = self._recorder
        if rec is not None:
            try:
                rec.abort()
            except Exception:
                pass

    def _check_cancel(self) -> None:
        if self._cancel.is_set():
            raise CancelledError_("cancelled")

    # ------------------------------------------------------------------ #
    # Engine lifecycle
    # ------------------------------------------------------------------ #
    @property
    def stt(self) -> SttEngine:
        with self._lock:
            if self._stt is None:
                self._stt = create_stt(self.config.stt, ollama_host=self.config.ollama.host,
                                       app_language=self.config.language)
            return self._stt

    @property
    def tts(self) -> TtsEngine:
        with self._lock:
            if self._tts is None:
                self._tts = create_tts(self.config.tts, language=self.config.language,
                                       auto_download=self.auto_download)
            return self._tts

    def _synth_pieces(self, text: str):
        """Yield audio for ``text``, holding the TTS lock.

        ONNX Runtime sessions are not safe to call concurrently, and the speech
        queue runs on its own thread, so every synthesis is serialised here.
        """
        if self.config.tts.engine in ("none", ""):
            return
        with self._tts_lock:
            for clip in self.tts.synthesize_stream(text):
                if self._cancel.is_set() or self._speech_stop.is_set():
                    return
                if len(clip.samples):
                    yield clip

    @property
    def player(self) -> Any:
        with self._lock:
            if self._player is None:
                self._player = audioio.make_player(self.audio_backend,
                                                   device=self.config.audio.output_device)
            return self._player

    def reload_engines(self) -> None:
        """Drop cached engines so new config takes effect on the next turn."""
        self.cancel_speech()
        thread, self._speech_thread = self._speech_thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        with self._lock, self._tts_lock:
            self._stt = None
            self._tts = None
            if self._player is not None:
                try:
                    self._player.stop()
                except Exception:
                    pass
                self._player = None

    def warm(self, *, stt: bool = True, tts: bool = True) -> dict[str, float]:
        """Load model weights up-front (called from a worker thread at startup)."""
        timings: dict[str, float] = {}
        if tts and self.config.tts.engine not in ("none", ""):
            t0 = time.time()
            try:
                self.tts.load()
                timings["tts_load_s"] = time.time() - t0
                self.emit("info", text=f"TTS ready: {self.config.tts.engine} "
                                       f"({getattr(self.tts, 'voice_name', '?')}) "
                                       f"in {timings['tts_load_s']:.1f}s")
            except Exception as exc:
                self.emit("error", text=f"TTS failed to load: {exc}", engine="tts")
        if stt:
            t0 = time.time()
            try:
                self.stt.load()
                timings["stt_load_s"] = time.time() - t0
                self.emit("info", text=f"STT ready: {self.stt.describe()} "
                                       f"in {timings['stt_load_s']:.1f}s")
            except Exception as exc:
                self.emit("error", text=f"STT failed to load: {exc}", engine="stt")
        return timings

    def speak_text(self, text: str, *, blocking: bool = True,
                   play: bool = True) -> AudioClip:
        """Synthesise (and optionally play) arbitrary text.

        ``play=False`` is what the HTTP API uses: it only wants the bytes back.
        Returns the full clip, concatenated from every streamed piece.
        """
        # An explicit request to speak is a fresh intent: a Stop press from the
        # previous turn must not swallow it.
        self._cancel.clear()
        if self.config.tts.engine in ("none", ""):
            self.emit("info", text="Speech output is disabled (tts.engine=none).")
            return AudioClip(np.zeros(0, dtype=np.int16), 22050)
        pieces: list[AudioClip] = []
        for clip in self._synth_pieces(text):
            self._check_cancel()
            pieces.append(clip)
            self.emit("audio", text=text[:80],
                      data={"duration_s": round(clip.duration_s, 2)})
            if play:
                self._play(clip, blocking=blocking)
        rate = pieces[0].sample_rate if pieces else 22050
        full = AudioClip(np.zeros(0, dtype=np.int16), rate)
        for piece in pieces:
            full = full.concat(piece)
        return full

    # ------------------------------------------------------------------ #
    # Speech queue
    # ------------------------------------------------------------------ #
    # Synthesis and playback run on their own thread. Without that, a voice that
    # is slower than realtime (Kokoro on a small CPU) would stall the token loop:
    # the model would stop being read until the previous sentence had been
    # synthesised *and* played. With the queue, generation runs flat out and the
    # voice catches up behind it - which is also exactly what you want when
    # synthesis is faster than realtime.

    def start_speech_queue(self) -> None:
        if self._speech_thread is not None and self._speech_thread.is_alive():
            return
        self._speech_stop.clear()
        self._speech_thread = threading.Thread(target=self._speech_loop,
                                               name="ollavoice-speech", daemon=True)
        self._speech_thread.start()

    def enqueue_speech(self, text: str) -> None:
        """Queue a sentence for synthesis + playback. Never blocks the caller."""
        if self.config.tts.engine in ("none", ""):
            return
        self.start_speech_queue()
        self._speech_q.put(text)

    def wait_for_speech(self, timeout_s: float = 600.0) -> bool:
        """Block until everything queued has been synthesised and played.

        Returns False only if the wait timed out with work still pending; a
        cancelled queue counts as drained rather than as a failure.

        The worker thread deliberately stays alive between turns (it is only torn
        down by a cancel or an engine reload), so "drained" means the queue is
        empty and no synthesis is in flight - not that the thread has exited.
        """
        deadline = time.time() + timeout_s
        while True:
            if self._speech_stop.is_set():
                return True
            if self._speech_q.empty() and not self._speech_busy.is_set():
                return True
            if time.time() >= deadline:
                return False
            time.sleep(0.02)

    def cancel_speech(self) -> None:
        """Drop everything queued and stop the current playback."""
        self._speech_stop.set()
        while not self._speech_q.empty():
            try:
                self._speech_q.get_nowait()
                self._speech_q.task_done()
            except queue.Empty:
                break
        player = self._player
        if player is not None:
            try:
                player.stop()
            except Exception:
                pass

    def _speech_loop(self) -> None:
        while not self._speech_stop.is_set():
            try:
                text = self._speech_q.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                self._speak_now(text)
            except Exception as exc:
                self.emit("error", text=f"Speech output failed: {exc}",
                          data={"source": "tts"})
            finally:
                self._speech_q.task_done()
        # Drain whatever is left so a cancelled queue cannot wedge the counter.
        while not self._speech_q.empty():
            try:
                self._speech_q.get_nowait()
                self._speech_q.task_done()
            except queue.Empty:
                break

    def _speak_now(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self._speech_busy.set()
        try:
            self.emit("sentence", text=text)
            t0 = time.time()
            clips = list(self._synth_pieces(text))
            if not clips:
                return
            full = clips[0]
            for c in clips[1:]:
                full = full.concat(c)
            synth_s = time.time() - t0
            if not self._first_audio_at:
                self._first_audio_at = time.time()
            self.emit("audio", text=text,
                      data={"duration_s": round(full.duration_s, 2),
                            "synth_s": round(synth_s, 2)})
            self._play(full, blocking=True)
        finally:
            self._speech_busy.clear()

    # ------------------------------------------------------------------ #
    # Listening
    # ------------------------------------------------------------------ #
    def start_listening(self) -> None:
        """Begin push-to-talk capture. Call :meth:`stop_listening` to get a turn."""
        if self._recorder is not None:
            return
        rec = audioio.make_recorder(
            self.audio_backend,
            sample_rate=self.config.audio.sample_rate,
            channels=self.config.audio.channels,
            chunk_ms=self.config.audio.chunk_ms,
            device=self.config.audio.input_device,
            on_chunk=self._on_mic_chunk,
            on_state=lambda s: self.emit("info", text=f"mic: {s}"),
        )
        self._recorder = rec
        self._cancel.clear()
        rec.start()
        self._set_phase(Phase.LISTENING)

    def stop_listening(self) -> AudioClip:
        rec, self._recorder = self._recorder, None
        if rec is None:
            return AudioClip(np.zeros(0, dtype=np.int16), self.config.audio.sample_rate)
        clip = rec.stop()
        clip = trim_clip(clip, threshold_db=self.config.vad.threshold_db - 4)
        self.emit("info", text=f"captured {clip.duration_s:.1f}s of audio")
        return clip

    def _on_mic_chunk(self, chunk: np.ndarray) -> None:
        if self._mic_muted.is_set():
            return
        from .vad import level_meter_db  # noqa: PLC0415

        db = level_meter_db(chunk)
        self.emit("level", data={"db": round(db, 1)})

    # ------------------------------------------------------------------ #
    # Hands-free (VAD) listening: returns when the speaker stops talking
    # ------------------------------------------------------------------ #
    def listen_until_silence(self, *, max_s: float | None = None) -> AudioClip:
        vad_cfg = self.config.vad
        gate = SpeechGate(VadSettingsLite(
            silence_ms=vad_cfg.silence_ms,
            min_speech_ms=vad_cfg.min_speech_ms,
            max_record_s=max_s or vad_cfg.max_record_s,
            threshold_db=vad_cfg.threshold_db,
            adaptive=vad_cfg.adaptive,
            adaptive_margin_db=vad_cfg.adaptive_margin_db,
            pre_speech_pad_ms=vad_cfg.pre_speech_pad_ms,
            calibration_frames=vad_cfg.calibration_frames,
        ))

        def on_chunk(chunk: np.ndarray) -> None:
            db = float(20.0 * np.log10(max(1e-7, float(np.sqrt(np.mean(
                np.square(chunk.astype(np.float32) / 32768.0)))))))
            self.emit("level", data={"db": round(db, 1),
                                     "state": gate.state.value,
                                     "threshold": round(gate.threshold_db, 1)})
            for ev in gate.push(chunk, rec.sample_rate):
                if ev.kind == "speech_start":
                    self.emit("info", text="speech detected...")
                elif ev.kind == "speech_end":
                    done.set()
                elif ev.kind == "timeout":
                    done.set()

        done = threading.Event()
        rec = audioio.make_recorder(
            self.audio_backend,
            sample_rate=self.config.audio.sample_rate,
            channels=self.config.audio.channels,
            chunk_ms=self.config.audio.chunk_ms,
            device=self.config.audio.input_device,
            on_chunk=on_chunk,
        )
        self._recorder = rec
        self._cancel.clear()
        self._set_phase(Phase.LISTENING)
        rec.start()
        try:
            while not done.wait(0.05):
                self._check_cancel()
                if rec.elapsed_s > (max_s or vad_cfg.max_record_s) + 2:
                    break
        except CancelledError_:
            rec.abort()
            self._recorder = None
            raise
        clip = rec.stop()
        self._recorder = None
        if gate.state == GateState.IDLE and gate.speech_ms <= 0:
            self.emit("info", text="no speech detected")
            return AudioClip(np.zeros(0, dtype=np.int16), clip.sample_rate)
        # Prefer the gate's own trimmed buffer; fall back to the raw capture.
        gated = gate.clip
        out = gated if len(gated.samples) > 0 else clip
        return trim_clip(out, threshold_db=vad_cfg.threshold_db - 4)

    # ------------------------------------------------------------------ #
    # Transcription
    # ------------------------------------------------------------------ #
    def transcribe(self, clip: AudioClip | Path | str, *, emit: bool = True) -> Transcription:
        self._cancel.clear()
        self._set_phase(Phase.TRANSCRIBING)
        t0 = time.time()
        try:
            result = self.stt.transcribe(clip, language=self.config.stt.language)
        except Exception as exc:
            self._set_phase(Phase.ERROR)
            self.emit("error", text=f"Transcription failed: {exc}", data={"source": "stt"})
            raise PipelineError(str(exc)) from exc
        self.last_transcription = result
        result.decode_s = result.decode_s or (time.time() - t0)
        if emit:
            self.emit("transcript", text=result.text, data={
                "language": result.language,
                "decode_s": round(result.decode_s, 2),
                "audio_s": round(result.duration_s, 2),
                "engine": result.engine,
                "model": result.model,
            })
            self.emit("stats", data={"stt_s": round(result.decode_s, 2),
                                     "audio_s": round(result.duration_s, 2),
                                     "rtf": round(result.realtime_factor, 2)})
        return result

    # ------------------------------------------------------------------ #
    # The main turn
    # ------------------------------------------------------------------ #
    def run_turn(self, clip: AudioClip | None = None, *, text: str = "",
                 speak: bool | None = None, blocking_playback: bool = True) -> str:
        """Run one conversation turn. Returns the assistant's answer text.

        Supply either ``clip`` (audio to transcribe) or ``text`` (already typed).
        """
        if self._busy.is_set():
            raise PipelineError("a turn is already running")
        self._busy.set()
        self._cancel.clear()
        self._first_audio_at = 0.0
        speak = self.config.tts.auto_speak if speak is None else speak
        timings: dict[str, float] = {}
        try:
            user_text = text.strip()
            if user_text:
                # Typed input: publish it the same way a transcript is published, so
                # every UI has exactly one path for "the user said this".
                self.emit("transcript", text=user_text, data={"source": "keyboard"})
            else:
                if clip is None or not len(clip.samples):
                    raise PipelineError("nothing to do: no audio and no text")
                t0 = time.time()
                result = self.transcribe(clip)
                timings["stt_s"] = time.time() - t0
                user_text = result.text.strip()
                if not user_text:
                    self.emit("info", text="I did not catch that - nothing was transcribed.")
                    self._set_phase(Phase.IDLE)
                    return ""
                if user_text.lower() in {"you", "."}:
                    self.emit("info", text="Ignoring a spurious transcription.")
                    self._set_phase(Phase.IDLE)
                    return ""

            self.history.add_user(user_text)
            self.last_stats = timings
            answer = self._stream_answer(speak=speak, blocking=blocking_playback,
                                         timings=timings)
            self.emit("stats", data={k: round(v, 2) for k, v in timings.items()})
            self.emit("done", text=answer, data={"final": True})
            self._set_phase(Phase.IDLE)
            return answer
        except CancelledError_:
            self.emit("info", text="cancelled")
            self._set_phase(Phase.IDLE)
            return ""
        except OllamaError as exc:
            self._set_phase(Phase.ERROR)
            self.emit("error", text=str(exc), data={"source": "ollama"})
            # Roll back the user message so a retry does not duplicate it.
            if self.history.messages and self.history.messages[-1].role == "user":
                self.history.messages.pop()
            raise PipelineError(str(exc)) from exc
        except PipelineError as exc:
            self._set_phase(Phase.ERROR)
            self.emit("error", text=str(exc))
            raise
        except Exception as exc:
            self._set_phase(Phase.ERROR)
            self.emit("error", text=f"{type(exc).__name__}: {exc}")
            raise PipelineError(str(exc)) from exc
        finally:
            self._busy.clear()

    # ------------------------------------------------------------------ #
    def _stream_answer(self, *, speak: bool, blocking: bool,
                       timings: dict[str, float]) -> str:
        self._set_phase(Phase.THINKING)
        model = self._resolve_model()
        options = {
            "temperature": float(self.config.ollama.temperature),
            "num_ctx": int(self.config.ollama.num_ctx),
        }
        if int(self.config.ollama.num_predict) > 0:
            options["num_predict"] = int(self.config.ollama.num_predict)

        splitter = SentenceSplitter(min_chars=int(self.config.tts.min_speak_chars))
        pieces: list[str] = []
        t0 = time.time()
        first_token = True
        spoken_any = False

        for evt in self.ollama.chat(self.history.for_api(), model, options=options,
                                    stream=True, keep_alive=self.config.ollama.keep_alive):
            self._check_cancel()
            delta = (evt.get("message") or {}).get("content", "")
            if delta:
                if first_token:
                    timings["ttft_s"] = time.time() - t0
                    first_token = False
                    self._set_phase(Phase.SPEAKING if speak else Phase.THINKING)
                pieces.append(delta)
                self.emit("token", text=delta)
                if speak:
                    for sentence in splitter.feed(delta):
                        spoken_any = True
                        self.enqueue_speech(sentence)
            if evt.get("done"):
                break

        if speak:
            for sentence in splitter.flush():
                spoken_any = True
                self.enqueue_speech(sentence)

        answer = "".join(pieces).strip()
        timings["llm_s"] = time.time() - t0
        self.history.add_assistant(answer)
        if speak and not spoken_any:
            self.emit("info", text="Nothing to speak (empty answer).")

        if speak and spoken_any:
            timings["tts_first_s"] = self._first_audio_offset(t0)
            # With blocking playback the caller expects the turn to be finished
            # when we return, so wait for the voice to catch up.
            if blocking:
                self.wait_for_speech(timeout_s=600.0)
        return answer

    def _first_audio_offset(self, t0: float) -> float:
        """Seconds from the start of generation until the first audio was ready."""
        return round(self._first_audio_at - t0, 3) if self._first_audio_at else 0.0

    def _play(self, clip: AudioClip, *, blocking: bool = True) -> None:
        if not len(clip.samples):
            return
        if self.config.audio.mute_mic_while_speaking:
            self._mic_muted.set()
        try:
            self.player.play(clip, blocking=blocking)
            if not blocking:
                # Give the sink a moment to actually start before we unmute.
                time.sleep(0.05)
        finally:
            if blocking and self.config.audio.mute_mic_while_speaking:
                self._mic_muted.clear()

    def wait_for_playback(self, timeout_s: float = 120.0) -> None:
        """Block until the speech queue is drained and the sink is idle.

        This is what front-ends call at the end of a turn: generation finishes
        long before a slow voice has read the last sentence, and the UI should not
        unlock until the user has actually heard it.
        """
        self.wait_for_speech(timeout_s=timeout_s)
        player = self._player
        if player is None:
            return
        deadline = time.time() + timeout_s
        while getattr(player, "is_playing", False) and time.time() < deadline:
            time.sleep(0.03)

    # ------------------------------------------------------------------ #
    def _resolve_model(self) -> str:
        model = (self.config.ollama.model or "").strip()
        if not model:
            raise PipelineError(
                "No Ollama model configured. Set one with:\n"
                "  ollavoice config set ollama.model llama3.2:3b\n"
                "or pick one in the GUI toolbar."
            )
        return model

    def ensure_model(self, *, pull: bool = False,
                     progress: Callable[[dict[str, Any]], None] | None = None) -> str:
        """Check the configured model exists; optionally pull it."""
        model = self._resolve_model()
        if self.ollama.has_model(model):
            return model
        if not pull:
            raise PipelineError(
                f"Ollama does not have '{model}' yet.\n"
                f"  ollama pull {model}\n"
                "or run: ollavoice setup --pull"
            )
        self.emit("info", text=f"Pulling {model} ...")
        self.ollama.pull(model, progress=progress)
        return model

    # ------------------------------------------------------------------ #
    def transcribe_file(self, path: str | Path, *, speak_answer: bool = False) -> Transcription:
        """Convenience for ``ollavoice listen --file x.wav``."""
        from .wav import read_wav  # noqa: PLC0415

        clip = read_wav(path)
        return self.transcribe(clip, emit=True)

    def status(self) -> dict[str, Any]:
        return {
            "ollama_alive": self.ollama.is_alive(timeout=2),
            "ollama_version": self.ollama.version(),
            "model": self.config.ollama.model,
            "stt": self._stt.describe() if self._stt else {"engine": self.config.stt.engine,
                                                            "model": self.config.stt.model,
                                                            "loaded": False},
            "tts": self._tts.describe() if self._tts else {"engine": self.config.tts.engine,
                                                           "voice": self.config.tts.piper_voice,
                                                           "loaded": False},
            "history_turns": len(self.history.messages),
            "audio_backend": self.audio_backend,
        }


def readable_error(exc: Exception) -> str:
    """A short, human-friendly message for the GUI status bar."""
    msg = str(exc)
    if "Cannot reach Ollama" in msg:
        return "Ollama is not running - start it and try again."
    if "does not have" in msg:
        return "The selected model is not downloaded yet."
    if len(msg) > 300:
        msg = msg[:300] + "..."
    return msg
