"""Tiny WAV helpers (no ffmpeg, no external binaries).

All audio inside OllaVoice is normalised to 16-bit signed PCM mono so that every
engine and every playback path can consume the same bytes.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_WIDTH = 2  # 16-bit


@dataclass(frozen=True)
class AudioClip:
    """Mono 16-bit PCM audio plus its sample rate."""

    samples: np.ndarray  # int16, shape (n,)
    sample_rate: int

    @property
    def duration_s(self) -> float:
        return len(self.samples) / float(self.sample_rate or 1)

    @property
    def is_silent(self) -> bool:
        return len(self.samples) == 0 or float(np.max(np.abs(self.samples.astype(np.float32)))) < 1.0

    def to_bytes(self) -> bytes:
        return np.ascontiguousarray(self.samples, dtype=np.int16).tobytes()

    def to_float(self) -> np.ndarray:
        return self.samples.astype(np.float32) / 32768.0

    def to_wav_bytes(self) -> bytes:
        buf = io.BytesIO()
        _write_wav_to(buf, self)
        return buf.getvalue()

    def write_wav(self, path: str | object) -> None:
        target = Path(str(path))
        if target.parent and not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
        # Open the file ourselves: if wave.open() fails halfway, CPython's
        # Wave_write.__del__ raises a confusing AttributeError during GC.
        with open(target, "wb") as fh:
            _write_wav_to(fh, self)

    @staticmethod
    def silence(duration_s: float, sample_rate: int) -> "AudioClip":
        return AudioClip(np.zeros(int(duration_s * sample_rate), dtype=np.int16), sample_rate)

    def concat(self, other: "AudioClip") -> "AudioClip":
        if not len(other.samples):
            return self
        if not len(self.samples):
            return other
        if other.sample_rate != self.sample_rate:
            other = other.resample(self.sample_rate)
        return AudioClip(np.concatenate([self.samples, other.samples]).astype(np.int16),
                         self.sample_rate)

    def resample(self, target_sr: int) -> "AudioClip":
        if target_sr == self.sample_rate or not len(self.samples):
            return self
        # Linear interpolation is plenty for speech going 22.05k/24k -> 16k.
        duration = len(self.samples) / float(self.sample_rate)
        n_out = max(1, int(round(duration * target_sr)))
        x_old = np.linspace(0.0, 1.0, num=len(self.samples), dtype=np.float64)
        x_new = np.linspace(0.0, 1.0, num=n_out, dtype=np.float64)
        y = np.interp(x_new, x_old, self.samples.astype(np.float64))
        return AudioClip(np.clip(y, -32768, 32767).astype(np.int16), target_sr)

    def peak_db(self) -> float:
        if not len(self.samples):
            return -120.0
        peak = float(np.max(np.abs(self.samples.astype(np.float32)))) / 32768.0
        return 20.0 * np.log10(peak) if peak > 0 else -120.0


def float_to_clip(audio: np.ndarray, sample_rate: int) -> AudioClip:
    """Accept float32 [-1, 1] or int16 arrays (any channel count) -> mono int16."""
    arr = np.asarray(audio)
    if arr.ndim > 1:
        arr = arr.mean(axis=-1) if arr.shape[0] > arr.shape[-1] else arr.mean(axis=0)
    if arr.dtype in (np.float32, np.float64, float):
        arr = np.clip(arr, -1.0, 1.0) * 32767.0
    return AudioClip(np.ascontiguousarray(arr, dtype=np.int16), int(sample_rate))


def bytes_to_clip(pcm: bytes, sample_rate: int) -> AudioClip:
    return AudioClip(np.frombuffer(pcm, dtype=np.int16).copy(), int(sample_rate))


def read_wav(path: str | object) -> AudioClip:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n_ch = w.getnchannels()
        width = w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if width != SAMPLE_WIDTH:
        raise ValueError(f"expected 16-bit WAV, got {width * 8}-bit ({path})")
    data = np.frombuffer(frames, dtype=np.int16)
    if n_ch > 1:
        data = data.reshape(-1, n_ch).mean(axis=1).astype(np.int16)
    return AudioClip(np.ascontiguousarray(data), sr)


def write_wav(path: str | object, clip: AudioClip) -> None:
    clip.write_wav(path)


def _write_wav_to(handle, clip: "AudioClip") -> None:
    """Write 16-bit mono PCM into an already-open binary file object."""
    with wave.open(handle, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(clip.sample_rate)
        w.writeframes(clip.to_bytes())


def rms_db(chunk: np.ndarray) -> float:
    """RMS level of an int16 chunk in dBFS. Returns -120 dB for digital silence."""
    if not len(chunk):
        return -120.0
    x = chunk.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(np.square(x))))
    return float(20.0 * np.log10(rms)) if rms > 1e-7 else -120.0
