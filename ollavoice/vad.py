"""Energy-based voice activity detection and end-of-speech gating.

Deliberately dependency-free (no silero-vad or webrtcvad wheels to fight with on
Windows): speech is much louder than a quiet room, so a measured noise floor plus
a hangover timer answers "have they finished their sentence?" well enough, and it
costs nothing at runtime.

Design notes
------------
* **Calibration window.** For the first ``calibration_frames`` chunks (~0.45 s)
  the gate only measures the room; it takes a low percentile of those levels as
  the noise floor and emits a ``calibrated`` event. Without this the detector
  cannot tell a noisy room from speech, because it has nothing to compare
  against yet.
* **Hysteresis.** Speech must exceed ``floor + adaptive_margin_db``; the floor
  then keeps tracking *quiet* frames slowly, so a fan switching on mid-conversation
  does not lock the detector at a stale value.
* **Hangover.** After the level drops, we wait ``silence_ms`` before cutting, and
  keep ``pre_speech_pad_ms`` of pre-roll so the first phoneme is not clipped.

Two entry points:

* :class:`SpeechGate` - streaming, for hands-free listening.
* :func:`trim_clip` / :func:`split_on_silence` - offline clean-up of a
  push-to-talk recording.

All levels are dBFS and all timings milliseconds, matching
:class:`ollavoice.config.VadSettings`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List

import numpy as np

from .wav import AudioClip, rms_db


class GateState(str, Enum):
    IDLE = "idle"            # waiting for speech to start
    SPEECH = "speech"        # someone is talking
    TRAILING = "trailing"    # quiet so far, but not long enough to cut yet


@dataclass
class VadSettingsLite:
    """Structural mirror of ``config.VadSettings`` so this module stays standalone."""

    silence_ms: int = 900
    min_speech_ms: int = 350
    max_record_s: float = 120.0
    threshold_db: float = -38.0
    adaptive: bool = True
    adaptive_margin_db: float = 9.0
    pre_speech_pad_ms: int = 300
    #: Chunks of audio used to measure the room before deciding anything
    #: (~0.45 s at 30 ms chunks). ``0`` disables calibration and trusts
    #: ``threshold_db`` alone.
    calibration_frames: int = 15


@dataclass
class GateEvent:
    kind: str          # "speech_start" | "speech_end" | "level" | "timeout" | "calibrated"
    db: float = -120.0
    ms: float = 0.0
    detail: str = ""


class SpeechGate:
    """Streaming VAD state machine.

    ``push(chunk, sample_rate)`` returns a list of events; after ``speech_end``
    the accumulated audio is available via :attr:`clip`.
    """

    #: Which percentile of the calibration window becomes the noise floor. A low
    #: percentile ignores a cough or a keystroke during calibration.
    CALIBRATION_PERCENTILE = 40.0
    #: Step size for the slow background tracking after calibration.
    TRACKING_ALPHA = 0.02
    #: The floor is never allowed above this, however loud the room is.
    FLOOR_CEILING_DB = -6.0

    def __init__(self, settings: VadSettingsLite | None = None) -> None:
        self.s = settings or VadSettingsLite()
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.state = GateState.IDLE
        self._frames: List[np.ndarray] = []
        self._pad: List[np.ndarray] = []      # pre-roll, for a natural attack
        self._calibration: List[float] = []
        self._calibrated = self.s.calibration_frames <= 0
        self._noise_db = float(self.s.threshold_db)
        self._noise_samples = 0
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._trailing_ms = 0.0
        self._total_ms = 0.0
        self._sr = 16000

    @property
    def threshold_db(self) -> float:
        return self._noise_db + (self.s.adaptive_margin_db if self.s.adaptive else 0.0)

    @property
    def noise_floor_db(self) -> float:
        return self._noise_db

    @property
    def calibrated(self) -> bool:
        """False while the gate is still measuring the room."""
        return self._calibrated

    @property
    def speech_ms(self) -> float:
        return self._speech_ms

    @property
    def total_ms(self) -> float:
        return self._total_ms

    @property
    def clip(self) -> AudioClip:
        if not self._frames:
            return AudioClip(np.zeros(0, dtype=np.int16), self._sr)
        return AudioClip(np.concatenate(self._frames).astype(np.int16), self._sr)

    # ------------------------------------------------------------------ #
    def push(self, chunk: np.ndarray, sample_rate: int = 16000) -> list[GateEvent]:
        """Feed one int16 mono chunk; get back zero or more events."""
        events: list[GateEvent] = []
        if chunk.size == 0:
            return events
        self._sr = int(sample_rate)
        chunk = np.asarray(chunk, dtype=np.int16)
        ms = 1000.0 * len(chunk) / float(sample_rate)
        db = rms_db(chunk)
        self._total_ms += ms
        events.append(GateEvent("level", db=db, ms=self._total_ms))

        if not self._calibrated:
            self._calibration.append(db)
            self._push_pad(chunk, sample_rate)
            if len(self._calibration) >= self.s.calibration_frames:
                self._finish_calibration()
                events.append(GateEvent(
                    "calibrated", db=self._noise_db, ms=self._total_ms,
                    detail=f"noise floor {self._noise_db:.1f} dBFS"))
            return events

        threshold = max(self.threshold_db, self.s.threshold_db)
        voiced = db > threshold
        if not voiced:
            # Track the room in every state, on quiet frames only.
            self._update_noise(db)

        if self.state == GateState.IDLE:
            if voiced:
                # Start from the pre-roll so the first phoneme survives.
                self._frames = list(self._pad)
                self._frames.append(chunk)
                self._speech_ms = sum(1000.0 * len(f) / sample_rate for f in self._frames)
                self._silence_ms = 0.0
                self._trailing_ms = 0.0
                self.state = GateState.SPEECH
                events.append(GateEvent("speech_start", db=db, ms=self._total_ms))
            else:
                self._push_pad(chunk, sample_rate)
            return events

        self._frames.append(chunk)
        if voiced:
            self._speech_ms += ms
            self._silence_ms = 0.0
            self._trailing_ms = 0.0
            if self.state == GateState.TRAILING:
                self.state = GateState.SPEECH
        else:
            self._silence_ms += ms
            self._trailing_ms += ms
            self.state = GateState.TRAILING

        if self._silence_ms >= self.s.silence_ms:
            if self._speech_ms >= self.s.min_speech_ms:
                self._trim_trailing(sample_rate)
                events.append(GateEvent("speech_end", db=db, ms=self._total_ms,
                                        detail=f"{self._speech_ms:.0f} ms of speech"))
                self.state = GateState.IDLE
                return events
            events.append(GateEvent("speech_end", db=db, ms=self._total_ms,
                                    detail="ignored: too short"))
            self.reset()
            return events

        if self.s.max_record_s and self._total_ms / 1000.0 >= self.s.max_record_s:
            events.append(GateEvent("timeout", db=db, ms=self._total_ms,
                                    detail=f"hit max_record_s={self.s.max_record_s}"))
        return events

    # ------------------------------------------------------------------ #
    def _push_pad(self, chunk: np.ndarray, sample_rate: int) -> None:
        self._pad.append(chunk.copy())
        budget = max(0.0, self.s.pre_speech_pad_ms) / 1000.0 * sample_rate
        while self._pad and sum(len(c) for c in self._pad) > budget:
            self._pad.pop(0)

    def _finish_calibration(self) -> None:
        """Turn the calibration window into a noise floor and start detecting."""
        levels = [db for db in self._calibration if db > -119.0]
        if levels:
            measured = float(np.percentile(levels, self.CALIBRATION_PERCENTILE))
            self._noise_db = max(min(measured, self.FLOOR_CEILING_DB), self.s.threshold_db)
        self._calibrated = True

    def _update_noise(self, db: float) -> None:
        """Slow background tracking of the ambient floor (quiet frames only)."""
        if not self.s.adaptive or db < -110 or not self._calibrated:
            return
        self._noise_samples += 1
        self._noise_db = ((1 - self.TRACKING_ALPHA) * self._noise_db
                          + self.TRACKING_ALPHA * db)
        self._noise_db = max(min(self._noise_db, self.FLOOR_CEILING_DB), self.s.threshold_db)

    def _trim_trailing(self, sample_rate: int) -> None:
        """Drop the hangover silence recorded while waiting for more speech.

        Only the *trailing* quiet stretch is a candidate - pauses inside the
        utterance are part of natural speech and must survive.
        """
        if not self._frames or self._trailing_ms <= 0:
            return
        trailing = self._trailing_ms / 1000.0 * sample_rate
        keep_silence = min(self._trailing_ms, 250) / 1000.0 * sample_rate
        total = sum(len(f) for f in self._frames)
        drop = int(max(0.0, min(trailing - keep_silence, total - keep_silence)))
        # Walk backwards across whole frames; simpler than slicing mid-frame.
        while drop > 0 and self._frames:
            last = self._frames[-1]
            if len(last) <= drop:
                drop -= len(last)
                self._frames.pop()
            else:
                self._frames[-1] = last[: len(last) - drop]
                drop = 0


# --------------------------------------------------------------------------- #
# Offline helpers (push-to-talk clean-up)
# --------------------------------------------------------------------------- #


def trim_clip(clip: AudioClip, *, threshold_db: float = -40.0, pad_ms: int = 150,
              frame_ms: int = 20) -> AudioClip:
    """Remove leading/trailing silence from a finished recording."""
    if not len(clip.samples):
        return clip
    frame = max(1, int(clip.sample_rate * frame_ms / 1000))
    frames = [clip.samples[i:i + frame] for i in range(0, len(clip.samples), frame)]
    active = [i for i, f in enumerate(frames) if rms_db(f) > threshold_db]
    if not active:
        return clip
    pad = max(1, int(pad_ms / frame_ms))
    start = max(0, active[0] - pad) * frame
    end = min(len(frames), active[-1] + 1 + pad) * frame
    return AudioClip(np.ascontiguousarray(clip.samples[start:end]), clip.sample_rate)


def split_on_silence(clip: AudioClip, *, threshold_db: float = -40.0,
                     min_gap_ms: int = 700, min_part_ms: int = 400) -> list[AudioClip]:
    """Split a long dictation into utterances at gaps longer than ``min_gap_ms``.

    Useful when someone rambles for a minute in hands-free mode: transcribing per
    utterance keeps Whisper inside its 30 s windows and lets the reply start
    sooner.
    """
    if not len(clip.samples):
        return []
    frame_ms = 20
    frame = max(1, int(clip.sample_rate * frame_ms / 1000))
    frames = [clip.samples[i:i + frame] for i in range(0, len(clip.samples), frame)]
    voiced = [rms_db(f) > threshold_db for f in frames]

    min_gap = max(1, int(min_gap_ms / frame_ms))
    min_part = max(1, int(min_part_ms / frame_ms))

    parts: list[AudioClip] = []
    start: int | None = None
    gap = 0
    for i, v in enumerate(voiced):
        if v:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap:
                end = i - gap + 1
                if end - start >= min_part:
                    parts.append(AudioClip(np.ascontiguousarray(
                        clip.samples[start * frame:end * frame]), clip.sample_rate))
                start, gap = None, 0
    if start is not None and len(voiced) - start >= min_part:
        parts.append(AudioClip(np.ascontiguousarray(clip.samples[start * frame:]),
                               clip.sample_rate))
    return parts or [clip]


def level_meter_db(chunk: np.ndarray) -> float:
    """Convenience wrapper used by the GUI's level bar."""
    return rms_db(chunk)


@dataclass
class RecordingStats:
    duration_s: float = 0.0
    peak_db: float = -120.0
    mean_db: float = -120.0
    voiced_ratio: float = 0.0
    clipped: bool = False

    @classmethod
    def from_clip(cls, clip: AudioClip, frame_ms: int = 20) -> "RecordingStats":
        if not len(clip.samples):
            return cls()
        frame = max(1, int(clip.sample_rate * frame_ms / 1000))
        frames = [clip.samples[i:i + frame] for i in range(0, len(clip.samples), frame)]
        dbs = np.array([rms_db(f) for f in frames], dtype=np.float32)
        audible = dbs[dbs > -119.0]
        return cls(
            duration_s=clip.duration_s,
            peak_db=float(clip.peak_db()),
            mean_db=float(np.mean(audible)) if audible.size else -120.0,
            voiced_ratio=float(np.mean(dbs > -40.0)) if dbs.size else 0.0,
            clipped=bool(np.max(np.abs(clip.samples)) >= 32760),
        )
