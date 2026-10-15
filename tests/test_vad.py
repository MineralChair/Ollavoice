"""VAD state machine, trimming and WAV utilities."""

from __future__ import annotations

import numpy as np
import pytest

from ollavoice import wav
from ollavoice.vad import (
    GateState, RecordingStats, SpeechGate, VadSettingsLite, split_on_silence, trim_clip,
)
from tests.conftest import silence_clip, tone_clip


def _chunks(clip, ms=30):
    size = max(1, int(clip.sample_rate * ms / 1000))
    return [clip.samples[i:i + size] for i in range(0, len(clip.samples), size)]


# --------------------------------------------------------------------------- #
# SpeechGate
# --------------------------------------------------------------------------- #

def test_gate_detects_start_and_end():
    gate = SpeechGate(VadSettingsLite(silence_ms=300, min_speech_ms=100, threshold_db=-45,
                                      adaptive=False, pre_speech_pad_ms=0,
                                      calibration_frames=0))
    events = []
    for chunk in _chunks(silence_clip(0.3)):
        events += gate.push(chunk, 16000)
    assert gate.state == GateState.IDLE

    for chunk in _chunks(tone_clip(0.8, amplitude=0.5)):
        events += gate.push(chunk, 16000)
    assert any(e.kind == "speech_start" for e in events)

    for chunk in _chunks(silence_clip(0.6)):
        events += gate.push(chunk, 16000)
    assert any(e.kind == "speech_end" for e in events)
    assert gate.clip.duration_s > 0.5


def test_gate_ignores_blips():
    gate = SpeechGate(VadSettingsLite(silence_ms=300, min_speech_ms=500, threshold_db=-45,
                                      adaptive=False, calibration_frames=0))
    events = []
    for chunk in _chunks(silence_clip(0.2)):
        events += gate.push(chunk, 16000)
    for chunk in _chunks(tone_clip(0.08, amplitude=0.5)):
        events += gate.push(chunk, 16000)
    for chunk in _chunks(silence_clip(0.8)):
        events += gate.push(chunk, 16000)
    assert any(e.kind == "speech_end" and "too short" in e.detail for e in events)


def test_gate_adapts_to_a_noisy_room():
    noisy = np.random.default_rng(0).normal(0, 900, 16000 * 2).astype(np.int16)  # ~-31 dBFS
    gate = SpeechGate(VadSettingsLite(threshold_db=-60, adaptive=True, silence_ms=250,
                                      min_speech_ms=100, calibration_frames=15))
    for i in range(0, len(noisy), 480):
        gate.push(noisy[i:i + 480], 16000)
    # The floor converges on the room noise, which lifts the speech threshold
    # above it so a steady hum is not mistaken for a voice.
    assert gate.noise_floor_db > -40
    assert gate.threshold_db > gate.noise_floor_db
    assert gate.speech_ms < 500


def test_gate_reports_levels():
    gate = SpeechGate()
    events = gate.push(tone_clip(0.05).samples, 16000)
    assert any(e.kind == "level" for e in events)


def test_calibration_window_holds_off_detection():
    """While calibrating only level events appear, then a single 'calibrated'."""
    gate = SpeechGate(VadSettingsLite(calibration_frames=5, threshold_db=-60))
    assert not gate.calibrated
    events = []
    for chunk in _chunks(tone_clip(0.5, amplitude=0.5), ms=30):
        events += gate.push(chunk, 16000)
        if gate.calibrated:
            break
    assert gate.calibrated
    assert any(e.kind == "calibrated" for e in events)
    assert not any(e.kind == "speech_start" for e in events)


def test_quiet_room_keeps_the_floor_low():
    """Calibrating on near-silence must not produce an unusable threshold."""
    gate = SpeechGate(VadSettingsLite(calibration_frames=10, threshold_db=-45))
    for chunk in _chunks(silence_clip(0.4), ms=30):
        gate.push(chunk, 16000)
    assert gate.calibrated
    assert gate.noise_floor_db <= -44
    events = []
    for chunk in _chunks(tone_clip(0.5, amplitude=0.5), ms=30):
        events += gate.push(chunk, 16000)
    assert any(e.kind == "speech_start" for e in events)


def test_extremely_loud_room_is_clamped():
    """A floor above full scale would make detection impossible - clamp it."""
    gate = SpeechGate(VadSettingsLite(calibration_frames=6, threshold_db=-70))
    loud = (np.ones(6 * 480) * 30000).astype(np.int16)
    for i in range(0, len(loud), 480):
        gate.push(loud[i:i + 480], 16000)
    assert gate.noise_floor_db <= SpeechGate.FLOOR_CEILING_DB


# --------------------------------------------------------------------------- #
# Offline helpers
# --------------------------------------------------------------------------- #

def test_trim_clip_removes_silence():
    clip = silence_clip(0.5).concat(tone_clip(0.5, amplitude=0.5)).concat(silence_clip(0.5))
    trimmed = trim_clip(clip, threshold_db=-40, pad_ms=50)
    assert trimmed.duration_s < clip.duration_s
    assert 0.4 < trimmed.duration_s < 0.9


def test_split_on_silence_separates_utterances():
    parts = [tone_clip(0.6, amplitude=0.5), silence_clip(1.2), tone_clip(0.6, amplitude=0.5)]
    clip = parts[0]
    for p in parts[1:]:
        clip = clip.concat(p)
    out = split_on_silence(clip, threshold_db=-40, min_gap_ms=600, min_part_ms=200)
    assert len(out) == 2
    assert all(0.4 < p.duration_s < 1.0 for p in out)


def test_split_on_silence_keeps_continuous_speech_together():
    clip = tone_clip(2.0, amplitude=0.5)
    assert len(split_on_silence(clip)) == 1


def test_recording_stats():
    stats = RecordingStats.from_clip(tone_clip(1.0, amplitude=0.5))
    assert stats.duration_s == pytest.approx(1.0, abs=0.05)
    assert stats.peak_db > -10
    assert stats.voiced_ratio > 0.9
    quiet = RecordingStats.from_clip(silence_clip(1.0))
    assert quiet.voiced_ratio == 0.0


# --------------------------------------------------------------------------- #
# WAV helpers
# --------------------------------------------------------------------------- #

def test_wav_roundtrip(tmp_path):
    clip = tone_clip(0.5)
    path = tmp_path / "tone.wav"
    wav.write_wav(path, clip)
    back = wav.read_wav(path)
    assert back.sample_rate == clip.sample_rate
    assert len(back.samples) == len(clip.samples)
    assert np.array_equal(back.samples, clip.samples)


def test_resample_changes_length_not_duration():
    clip = tone_clip(1.0, rate=24000)
    out = clip.resample(16000)
    assert out.sample_rate == 16000
    assert out.duration_s == pytest.approx(1.0, abs=0.02)
    assert len(out.samples) == pytest.approx(16000, abs=2)


def test_resample_is_a_noop_when_rates_match():
    clip = tone_clip(0.2, rate=22050)
    assert clip.resample(22050) is clip


def test_float_to_clip_handles_stereo_float():
    stereo = np.zeros((1000, 2), dtype=np.float32)
    stereo[:, 0] = 0.5
    stereo[:, 1] = -0.5
    clip = wav.float_to_clip(stereo, 24000)
    assert clip.samples.dtype == np.int16
    assert clip.sample_rate == 24000


def test_concat_and_silence():
    a = wav.AudioClip.silence(0.25, 16000)
    b = tone_clip(0.25)
    joined = a.concat(b)
    assert joined.duration_s == pytest.approx(0.5, abs=0.01)
    assert joined.concat(wav.AudioClip(np.zeros(0, dtype=np.int16), 16000)) is not None


def test_rms_db_silence_is_very_negative():
    assert wav.rms_db(np.zeros(160, dtype=np.int16)) < -100
    assert wav.rms_db(tone_clip(0.05, amplitude=0.5).samples) > -20
