"""Microphone capture and audio playback, with three interchangeable backends.

==============  ========================================================================
``qt``          Qt Multimedia (PySide6). Used by the desktop GUI; also works headless via
                a :class:`QCoreApplication`. One dependency covers record *and* playback,
                and it needs no native audio libraries of its own on Windows.
``sounddevice`` PortAudio through the ``sounddevice`` wheel. Used by the CLI/server on
                machines without Qt. Needs PortAudio (bundled in the Windows wheel).
``system``      Last resort: record/play with external tools (``ffmpeg``/``arecord``/
                ``aplay`` on Linux, PowerShell + ``System.Media.SoundPlayer`` on Windows).
==============  ========================================================================

The recorders hand you int16 mono numpy chunks at the configured sample rate; the players
take a :class:`~ollavoice.wav.AudioClip` and, importantly, tell you when playback *ends* -
that is what lets the app mute the microphone while it is talking, so the assistant does
not transcribe its own voice back to itself.
"""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .wav import AudioClip, read_wav

ChunkFn = Callable[[np.ndarray], None]
StateFn = Callable[[str], None]


class AudioError(RuntimeError):
    pass


@dataclass
class AudioDevice:
    name: str
    kind: str            # "input" | "output"
    backend: str
    index: int = -1
    channels: int = 1
    default_rate: int = 16000

    def label(self) -> str:
        return f"{self.name} ({self.backend})"


# --------------------------------------------------------------------------- #
# Backend selection
# --------------------------------------------------------------------------- #

BACKENDS = ("qt", "sounddevice", "system", "null")


def qt_available() -> bool:
    """True when Qt Multimedia can actually be imported *and* loaded.

    PySide6 being installed is not enough: on Linux a missing libglib/libGL makes
    the import raise OSError, and the answer must be False so we fall back to
    sounddevice instead of crashing at startup.
    """
    if os.environ.get("OLLAVOICE_NO_QT"):
        return False
    try:
        from PySide6 import QtMultimedia  # noqa: PLC0415,F401

        return True
    except Exception:      # ImportError, OSError (missing Qt system libs), ...
        return False


def sounddevice_available() -> bool:
    try:
        import sounddevice  # noqa: PLC0415,F401

        return True
    except Exception:
        return False


def system_tools() -> dict[str, str]:
    found = {}
    for tool in ("ffmpeg", "arecord", "aplay", "ffplay", "sox", "pwsh", "powershell"):
        path = shutil.which(tool)
        if path:
            found[tool] = path
    return found


def choose_backend(preferred: str = "auto") -> str:
    if preferred == "null":
        return "null"
    order = [preferred] if preferred in BACKENDS else [b for b in BACKENDS if b != "null"]
    for backend in order:
        if backend == "qt" and qt_available():
            return "qt"
        if backend == "sounddevice" and sounddevice_available():
            return "sounddevice"
        if backend == "system" and system_tools():
            return "system"
    raise AudioError(
        "No usable audio backend.\n"
        "Install one of:\n"
        "  pip install PySide6        (recommended: mic + playback, no system deps)\n"
        "  pip install sounddevice    (needs PortAudio; bundled in the Windows wheel)\n"
        "or make sure ffmpeg / arecord+aplay is on PATH.\n"
        "(Use --audio-backend null to run with no audio hardware at all.)"
    )


def list_devices(backend: str = "auto") -> list[AudioDevice]:
    backend = choose_backend(backend)
    if backend == "sounddevice":
        return _sd_devices()
    if backend == "qt":
        return _qt_devices()
    return []


# --------------------------------------------------------------------------- #
# Recorder interface
# --------------------------------------------------------------------------- #


class Recorder(ABC):
    """Captures int16 mono PCM and pushes chunks to a callback."""

    backend = "base"

    def __init__(self, sample_rate: int = 16000, channels: int = 1,
                 chunk_ms: int = 30, device: str = "", on_chunk: ChunkFn | None = None,
                 on_state: StateFn | None = None) -> None:
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.chunk_ms = int(chunk_ms)
        self.device = device or ""
        self.on_chunk = on_chunk
        self.on_state = on_state
        self.recording = False
        self.started_at = 0.0
        self._frames: list[np.ndarray] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> AudioClip: ...

    def abort(self) -> None:
        """Stop and throw the audio away."""
        try:
            self.stop()
        except AudioError:
            pass
        with self._lock:
            self._frames.clear()

    @property
    def elapsed_s(self) -> float:
        return time.time() - self.started_at if self.recording else 0.0

    def buffered_clip(self) -> AudioClip:
        with self._lock:
            if not self._frames:
                return AudioClip(np.zeros(0, dtype=np.int16), self.sample_rate)
            return AudioClip(np.concatenate(self._frames).astype(np.int16), self.sample_rate)

    # ------------------------------------------------------------------ #
    def _ingest(self, chunk: np.ndarray) -> None:
        if chunk is None or chunk.size == 0:
            return
        if chunk.dtype != np.int16:
            if np.issubdtype(chunk.dtype, np.floating):
                chunk = np.clip(chunk, -1.0, 1.0)
                chunk = (chunk * 32767).astype(np.int16)
            else:
                chunk = chunk.astype(np.int16)
        if self.channels > 1:
            chunk = chunk.reshape(-1, self.channels).mean(axis=1).astype(np.int16)
        with self._lock:
            self._frames.append(chunk)
        if self.on_chunk:
            try:
                self.on_chunk(chunk)
            except Exception:
                pass

    def _take_clip(self) -> AudioClip:
        with self._lock:
            frames, self._frames = self._frames, []
        if not frames:
            return AudioClip(np.zeros(0, dtype=np.int16), self.sample_rate)
        return AudioClip(np.concatenate(frames).astype(np.int16), self.sample_rate)

    def _state(self, state: str) -> None:
        if self.on_state:
            try:
                self.on_state(state)
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# Qt Multimedia backend
# --------------------------------------------------------------------------- #


class QtRecorder(Recorder):
    backend = "qt"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._source = None
        self._io = None
        self._app = ensure_qapp(gui=False)

    def start(self) -> None:
        if self.recording:
            return
        from PySide6.QtMultimedia import QAudioFormat, QAudioSource  # noqa: PLC0415

        fmt = QAudioFormat()
        fmt.setChannelCount(self.channels)
        fmt.setSampleRate(self.sample_rate)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)

        device = _qt_input_device(self.device)
        if not device.isNull() and not device.isFormatSupported(fmt):
            preferred = device.preferredFormat()
            # Keep the sample rate we asked for when the device allows it.
            fmt = QAudioFormat(preferred)
            fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
            fmt.setChannelCount(self.channels)
            if not device.isFormatSupported(fmt):
                raise AudioError(
                    f"Microphone '{device.description()}' does not support 16-bit PCM capture."
                )
            self.sample_rate = fmt.sampleRate()

        self._source = QAudioSource(device, fmt)
        self._source.setErrorChanged(lambda err: self._state(f"error:{err}"))
        self._io = self._source.start()
        if self._io is None:
            raise AudioError(f"Qt could not start capture (error {self._source.error()})")
        self._io.readyRead.connect(self._on_ready)
        self.recording = True
        self.started_at = time.time()
        self._state("recording")

    def _on_ready(self) -> None:
        try:
            data = self._io.readAll()  # type: ignore[union-attr]
        except RuntimeError:
            return
        if not data:
            return
        raw = bytes(data)
        if self.channels > 1:
            arr = np.frombuffer(raw, dtype=np.int16).reshape(-1, self.channels)
            arr = arr.mean(axis=1).astype(np.int16)
        else:
            arr = np.frombuffer(raw, dtype=np.int16).copy()
        self._ingest(arr)

    def stop(self) -> AudioClip:
        if not self.recording:
            return self._take_clip()
        self.recording = False
        try:
            if self._io is not None:
                self._on_ready()
            if self._source is not None:
                self._source.stop()
        except RuntimeError:
            pass
        self._source = None
        self._io = None
        clip = self._take_clip()
        self._state("stopped")
        return clip


class QtPlayer:
    """Plays :class:`AudioClip`s through QAudioSink."""

    backend = "qt"

    def __init__(self, device: str = "", on_finished: Callable[[], None] | None = None) -> None:
        self.device = device
        self.on_finished = on_finished
        self._sink = None
        self._buffer = None
        self._playing = threading.Event()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._app = ensure_qapp(gui=False)

    # ------------------------------------------------------------------ #
    @property
    def is_playing(self) -> bool:
        return self._playing.is_set()

    def play(self, clip: AudioClip, *, blocking: bool = False) -> None:
        if not len(clip.samples):
            return
        self._stop.clear()
        if blocking:
            self._play_sync(clip)
            return
        self.stop()
        self._thread = threading.Thread(target=self._play_sync, args=(clip,), daemon=True)
        self._thread.start()

    def _play_sync(self, clip: AudioClip) -> None:
        from PySide6.QtCore import QBuffer, QByteArray, QIODevice  # noqa: PLC0415
        from PySide6.QtMultimedia import QAudioFormat, QAudioSink  # noqa: PLC0415

        fmt = QAudioFormat()
        fmt.setChannelCount(1)
        fmt.setSampleRate(clip.sample_rate)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)

        device = _qt_output_device(self.device)
        sink = QAudioSink(device, fmt)
        sink.setVolume(1.0)
        buf = QBuffer()
        buf.setData(QByteArray(clip.to_bytes()))
        buf.open(QIODevice.OpenModeFlag.ReadOnly)

        self._sink, self._buffer = sink, buf
        self._playing.set()
        try:
            sink.start(buf)
            while sink.state() != _idle_state() and not self._stop.is_set():
                time.sleep(0.02)
                _pump_events(20)
            if not self._stop.is_set():
                # Wait for the last samples to reach the hardware.
                deadline = time.time() + clip.duration_s + 1.0
                while sink.bytesFree() < sink.bufferSize() and time.time() < deadline:
                    time.sleep(0.02)
                    _pump_events(20)
        finally:
            try:
                sink.stop()
            except RuntimeError:
                pass
            buf.close()
            self._sink = self._buffer = None
            self._playing.clear()
            if self.on_finished:
                try:
                    self.on_finished()
                except Exception:
                    pass

    def stop(self) -> None:
        self._stop.set()
        sink = self._sink
        if sink is not None:
            try:
                sink.stop()
            except RuntimeError:
                pass
        if self._thread and self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)
        self._playing.clear()


def _idle_state():
    from PySide6.QtMultimedia import QAudio  # noqa: PLC0415

    return QAudio.State.IdleState


def _pump_events(ms: int) -> None:
    try:
        from PySide6.QtCore import QCoreApplication  # noqa: PLC0415

        app = QCoreApplication.instance()
        if app is not None:
            app.processEvents()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# sounddevice backend
# --------------------------------------------------------------------------- #


class SoundDeviceRecorder(Recorder):
    backend = "sounddevice"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._stream = None
        self._q: "queue.Queue[np.ndarray]" = queue.Queue()
        self._pump: threading.Thread | None = None
        self._running = threading.Event()

    def start(self) -> None:
        if self.recording:
            return
        try:
            import sounddevice as sd  # noqa: PLC0415
        except OSError as exc:
            raise AudioError(f"sounddevice could not load PortAudio: {exc}") from exc

        blocksize = max(160, int(self.sample_rate * self.chunk_ms / 1000))
        device = self.device or None
        try:
            self._stream = sd.RawInputStream(
                samplerate=self.sample_rate, blocksize=blocksize, device=device,
                channels=self.channels, dtype="int16", callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:
            raise AudioError(f"Could not open microphone: {exc}") from exc
        self._running.set()
        self._pump = threading.Thread(target=self._drain, daemon=True)
        self._pump.start()
        self.recording = True
        self.started_at = time.time()
        self._state("recording")

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        self._q.put(bytes(indata))

    def _drain(self) -> None:
        while self._running.is_set():
            try:
                raw = self._q.get(timeout=0.05)
            except queue.Empty:
                continue
            arr = np.frombuffer(raw, dtype=np.int16).copy()
            self._ingest(arr)
        while not self._q.empty():
            raw = self._q.get_nowait()
            self._ingest(np.frombuffer(raw, dtype=np.int16).copy())

    def stop(self) -> AudioClip:
        if not self.recording:
            return self._take_clip()
        self.recording = False
        self._running.clear()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._pump:
            self._pump.join(timeout=1.0)
            self._pump = None
        clip = self._take_clip()
        self._state("stopped")
        return clip


class SoundDevicePlayer:
    backend = "sounddevice"

    def __init__(self, device: str = "", on_finished: Callable[[], None] | None = None) -> None:
        self.device = device or None
        self.on_finished = on_finished
        self._playing = threading.Event()

    @property
    def is_playing(self) -> bool:
        return self._playing.is_set()

    def play(self, clip: AudioClip, *, blocking: bool = False) -> None:
        if not len(clip.samples):
            return
        try:
            import sounddevice as sd  # noqa: PLC0415
        except OSError as exc:
            raise AudioError(f"sounddevice could not load PortAudio: {exc}") from exc
        self._playing.set()
        try:
            sd.play(clip.samples, samplerate=clip.sample_rate, device=self.device,
                    blocking=blocking)
            if not blocking:
                sd.wait()
        finally:
            self._playing.clear()
            if self.on_finished:
                try:
                    self.on_finished()
                except Exception:
                    pass

    def stop(self) -> None:
        try:
            import sounddevice as sd  # noqa: PLC0415

            sd.stop()
        except Exception:
            pass
        self._playing.clear()


def _sd_devices() -> list[AudioDevice]:
    import sounddevice as sd  # noqa: PLC0415

    out = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0:
            out.append(AudioDevice(dev["name"], "input", "sounddevice", idx,
                                   int(dev["max_input_channels"]), int(dev["default_samplerate"])))
        if dev.get("max_output_channels", 0) > 0:
            out.append(AudioDevice(dev["name"], "output", "sounddevice", idx,
                                   int(dev["max_output_channels"]), int(dev["default_samplerate"])))
    return out


# --------------------------------------------------------------------------- #
# System-tool backend (no Python audio deps at all)
# --------------------------------------------------------------------------- #


class SystemRecorder(Recorder):
    """Records with ffmpeg (Windows/Linux/macOS) or arecord (Linux)."""

    backend = "system"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._running = threading.Event()

    def _command(self, duration_hint: float = 600) -> list[str]:
        tools = system_tools()
        if "ffmpeg" in tools:
            if sys.platform == "win32":
                device = self.device or "audio=Microphone"
                return [tools["ffmpeg"], "-hide_banner", "-loglevel", "error", "-y",
                        "-f", "dshow", "-i", device, "-ac", "1", "-ar", str(self.sample_rate),
                        "-f", "s16le", "-"]
            if sys.platform == "darwin":
                device = self.device or "default"
                return [tools["ffmpeg"], "-hide_banner", "-loglevel", "error", "-y",
                        "-f", "avfoundation", "-i", f":{device}", "-ac", "1",
                        "-ar", str(self.sample_rate), "-f", "s16le", "-"]
            device = self.device or "default"
            return [tools["ffmpeg"], "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "alsa", "-i", device, "-ac", "1", "-ar", str(self.sample_rate),
                    "-f", "s16le", "-"]
        if "arecord" in tools:
            device = self.device or "default"
            return [tools["arecord"], "-q", "-D", device, "-f", "S16_LE", "-c", "1",
                    "-r", str(self.sample_rate), "-t", "raw"]
        raise AudioError(
            "No system recorder available. Install ffmpeg (or arecord on Linux), "
            "or: pip install sounddevice"
        )

    def start(self) -> None:
        if self.recording:
            return
        cmd = self._command()
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as exc:
            raise AudioError(f"Could not start recorder: {exc}") from exc
        self._running.set()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self.recording = True
        self.started_at = time.time()
        self._state("recording")

    def _read_loop(self) -> None:
        assert self._proc and self._proc.stdout
        bytes_per_chunk = max(320, int(self.sample_rate * 2 * self.chunk_ms / 1000))
        while self._running.is_set():
            raw = self._proc.stdout.read(bytes_per_chunk)
            if not raw:
                break
            self._ingest(np.frombuffer(raw, dtype=np.int16).copy())

    def stop(self) -> AudioClip:
        if not self.recording:
            return self._take_clip()
        self.recording = False
        self._running.clear()
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if self._reader:
            self._reader.join(timeout=1.0)
            self._reader = None
        clip = self._take_clip()
        self._state("stopped")
        return clip


class SystemPlayer:
    """Plays a WAV with ffplay/aplay/afplay, or on Windows with SoundPlayer."""

    backend = "system"

    def __init__(self, device: str = "", on_finished: Callable[[], None] | None = None) -> None:
        self.device = device
        self.on_finished = on_finished
        self._proc: subprocess.Popen | None = None
        self._playing = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def is_playing(self) -> bool:
        return self._playing.is_set()

    def play(self, clip: AudioClip, *, blocking: bool = False) -> None:
        if not len(clip.samples):
            return
        path = _write_temp_wav(clip)
        self._playing.set()
        if blocking:
            self._run(path)
            return
        self._thread = threading.Thread(target=self._run, args=(path,), daemon=True)
        self._thread.start()

    def _run(self, path: Path) -> None:
        try:
            cmd = self._command(path)
            if cmd is None:
                raise AudioError("No system audio player available (need ffplay, aplay, afplay "
                                 "or PowerShell on Windows).")
            self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL,
                                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self._proc.wait()
        except Exception:
            pass
        finally:
            self._proc = None
            self._playing.clear()
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            if self.on_finished:
                try:
                    self.on_finished()
                except Exception:
                    pass

    def _command(self, path: Path) -> list[str] | None:
        tools = system_tools()
        if sys.platform == "win32":
            ps = tools.get("powershell") or tools.get("pwsh")
            if ps:
                return [ps, "-NoProfile", "-Command",
                        "(New-Object Media.SoundPlayer $args[0]).PlaySync()", str(path)]
        if "ffplay" in tools:
            return [tools["ffplay"], "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)]
        if "aplay" in tools:
            return [tools["aplay"], "-q", str(path)]
        if sys.platform == "darwin" and shutil.which("afplay"):
            return [shutil.which("afplay"), str(path)]  # type: ignore[list-item]
        return None

    def stop(self) -> None:
        proc = self._proc
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
        self._playing.clear()


def _write_temp_wav(clip: AudioClip) -> Path:
    from . import downloads  # noqa: PLC0415

    path = downloads.temp_wav_path("play-")
    clip.write_wav(path)
    return path


# --------------------------------------------------------------------------- #
# Qt device helpers + QApplication management
# --------------------------------------------------------------------------- #


def _qt_input_device(name: str = ""):
    from PySide6.QtMultimedia import QMediaDevices  # noqa: PLC0415

    if name:
        for d in QMediaDevices.audioInputs():
            if d.description() == name:
                return d
    return QMediaDevices.defaultAudioInput()


def _qt_output_device(name: str = ""):
    from PySide6.QtMultimedia import QMediaDevices  # noqa: PLC0415

    if name:
        for d in QMediaDevices.audioOutputs():
            if d.description() == name:
                return d
    return QMediaDevices.defaultAudioOutput()


def _qt_devices() -> list[AudioDevice]:
    from PySide6.QtMultimedia import QMediaDevices  # noqa: PLC0415

    out = []
    for d in QMediaDevices.audioInputs():
        out.append(AudioDevice(d.description(), "input", "qt", -1,
                               d.maximumChannelCount() or 1,
                               d.preferredFormat().sampleRate() or 16000))
    for d in QMediaDevices.audioOutputs():
        out.append(AudioDevice(d.description(), "output", "qt", -1,
                               d.maximumChannelCount() or 2,
                               d.preferredFormat().sampleRate() or 48000))
    return out


_QAPP = None


def ensure_qapp(gui: bool = False):
    """Create (once) a QCoreApplication, or a QApplication when ``gui`` is set."""
    global _QAPP
    try:
        from PySide6.QtCore import QCoreApplication  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover
        raise AudioError(f"PySide6 is not available: {exc}") from exc
    existing = QCoreApplication.instance()
    if existing is not None:
        _QAPP = existing
        return existing
    argv = [sys.argv[0] if sys.argv else "ollavoice"]
    if gui:
        from PySide6.QtWidgets import QApplication  # noqa: PLC0415

        _QAPP = QApplication(argv)
    else:
        _QAPP = QCoreApplication(argv)
    return _QAPP


# --------------------------------------------------------------------------- #
# Null backend (no audio hardware: headless servers, CI, containers)
# --------------------------------------------------------------------------- #


class NullRecorder(Recorder):
    """A recorder that produces silence.

    Useful for headless testing, CI, and for running the HTTP API on a machine
    with no microphone - transcription and synthesis still work on bytes, and
    typed input still works, so only live capture is lost.
    """

    backend = "null"

    def start(self) -> None:
        self.recording = True
        self.started_at = time.time()
        self._state("recording")

    def stop(self) -> AudioClip:
        self.recording = False
        self._state("stopped")
        return self._take_clip()


class NullPlayer:
    """A player that discards audio but keeps the clips it was given."""

    backend = "null"

    def __init__(self, device: str = "", on_finished: Callable[[], None] | None = None) -> None:
        self.device = device
        self.on_finished = on_finished
        self.clips: list[AudioClip] = []

    @property
    def is_playing(self) -> bool:
        return False

    def play(self, clip: AudioClip, *, blocking: bool = False) -> None:
        self.clips.append(clip)
        if self.on_finished:
            try:
                self.on_finished()
            except Exception:
                pass

    def stop(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #


def make_recorder(backend: str = "auto", **kwargs) -> Recorder:
    backend = choose_backend(backend)
    if backend == "null":
        return NullRecorder(**kwargs)
    if backend == "qt":
        return QtRecorder(**kwargs)
    if backend == "sounddevice":
        return SoundDeviceRecorder(**kwargs)
    return SystemRecorder(**kwargs)


def make_player(backend: str = "auto", **kwargs):
    backend = choose_backend(backend)
    if backend == "null":
        return NullPlayer(**kwargs)
    if backend == "qt":
        return QtPlayer(**kwargs)
    if backend == "sounddevice":
        return SoundDevicePlayer(**kwargs)
    return SystemPlayer(**kwargs)


def record_clip(duration_s: float, *, backend: str = "auto", sample_rate: int = 16000,
                device: str = "", on_chunk: ChunkFn | None = None) -> AudioClip:
    """Blocking helper: record exactly ``duration_s`` seconds. Used by tests/CLI."""
    rec = make_recorder(backend, sample_rate=sample_rate, device=device, on_chunk=on_chunk)
    rec.start()
    time.sleep(duration_s)
    return rec.stop()


def play_file(path: str | Path, *, backend: str = "auto", blocking: bool = True) -> None:
    clip = read_wav(path)
    player = make_player(backend)
    player.play(clip, blocking=blocking)
    if not blocking:
        while player.is_playing:
            time.sleep(0.05)


def play_clip(clip: AudioClip, *, backend: str = "auto", blocking: bool = True) -> None:
    player = make_player(backend)
    player.play(clip, blocking=blocking)
    if not blocking:
        while player.is_playing:
            time.sleep(0.05)
