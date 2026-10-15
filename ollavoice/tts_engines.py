"""Text-to-speech engines. All of them run locally, on the CPU by default.

=============  =========================================================================
``piper``      (default) Piper neural TTS 1.x. ~60 MB per voice, faster than realtime on
               a modest CPU, and it streams audio chunk-by-chunk so the voice can start
               before the whole sentence is synthesised.
``kokoro``     Kokoro-82M through ``kokoro-onnx`` (~114 MB int8 + 28 MB voices). More
               expressive and more natural; roughly 4x slower than Piper on CPU.
``sapi``       The voices already installed in Windows (``System.Speech``), called through
               PowerShell. Zero download, robotic but instant and always present.
``none``       Silent mode - the chat still works, nothing is spoken. Useful for testing
               or when you only want speech *input*.
=============  =========================================================================

Every engine returns a :class:`~ollavoice.wav.AudioClip` (mono int16 PCM). The GUI plays
clips through Qt Multimedia; the CLI uses sounddevice/ffplay/PowerShell as available.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from . import downloads, paths
from .config import TtsSettings, localize_piper_voice
from .textsplit import expand_symbols, strip_for_speech
from .wav import AudioClip, float_to_clip


class TtsError(RuntimeError):
    pass


@dataclass
class SynthesisResult:
    clip: AudioClip
    synth_s: float
    engine: str
    voice: str

    @property
    def realtime_factor(self) -> float:
        """synth time / audio duration. Under 1.0 means faster than realtime."""
        if self.clip.duration_s <= 0:
            return 0.0
        return self.synth_s / self.clip.duration_s


class TtsEngine(ABC):
    name = "base"

    def __init__(self, settings: TtsSettings, *, language: str = "en") -> None:
        self.settings = settings
        self.language = language or "en"
        self.voice_name = ""
        self._loaded = False

    # ------------------------------------------------------------------ #
    @abstractmethod
    def synthesize(self, text: str) -> SynthesisResult:
        ...

    def load(self) -> None:
        """Pre-load weights so the first sentence is not slow. Idempotent."""
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def synthesize_stream(self, text: str) -> Iterator[AudioClip]:
        """Yield audio pieces as they are produced. Default: one piece."""
        result = self.synthesize(text)
        if len(result.clip.samples):
            yield result.clip

    def list_voices(self) -> list[str]:
        return []

    def describe(self) -> dict[str, Any]:
        return {"engine": self.name, "voice": self.voice_name, "loaded": self._loaded}

    # ------------------------------------------------------------------ #
    def _prepare_text(self, text: str) -> str:
        if self.settings.strip_markdown:
            text = strip_for_speech(text)
        return expand_symbols(text).strip()


# --------------------------------------------------------------------------- #
# Piper
# --------------------------------------------------------------------------- #


class PiperTts(TtsEngine):
    """Piper neural TTS (ONNX + espeak-ng phonemiser, both bundled in the wheel)."""

    name = "piper"

    def __init__(self, settings: TtsSettings, *, language: str = "en",
                 auto_download: bool = True) -> None:
        super().__init__(settings, language=language)
        self.auto_download = auto_download
        self.voice_name = self._resolve_voice(settings.piper_voice)
        self.model_dir = paths.piper_dir()
        self._voice: Any = None
        self.load_s = 0.0

    def _resolve_voice(self, requested: str) -> str:
        if requested and requested != "auto":
            return requested
        # Prefer a voice that is already on disk, else fall back to the tier default.
        d = paths.piper_dir()
        if d.exists():
            installed = sorted(p.stem for p in d.glob("*.onnx"))
            if installed:
                same_lang = [v for v in installed if v.split("-")[0].split("_")[0]
                             == self.language.split("-")[0]]
                return (same_lang or installed)[0]
        return localize_piper_voice("en_US-lessac-medium", self.language)

    # ------------------------------------------------------------------ #
    def _model_path(self) -> Path:
        onnx = self.model_dir / f"{self.voice_name}.onnx"
        if onnx.exists():
            return onnx
        if not self.auto_download:
            raise TtsError(
                f"Piper voice '{self.voice_name}' is not installed.\n"
                f"Run: ollavoice setup --tts-voice {self.voice_name}"
            )
        return downloads.ensure_piper_voice(self.voice_name, self.model_dir)

    def load(self) -> None:
        if self._voice is not None:
            self._loaded = True
            return
        try:
            from piper.voice import PiperVoice  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise TtsError(
                "Piper TTS is not installed.\n  pip install piper-tts\n"
                "or run: scripts\\setup.ps1 --with-tts"
            ) from exc

        model = self._model_path()
        kwargs: dict[str, Any] = {}
        espeak_dir = _find_espeak_data()
        if espeak_dir:
            kwargs["espeak_data_dir"] = str(espeak_dir)
        use_cuda = self._cuda_ok()
        t0 = time.time()
        try:
            self._voice = PiperVoice.load(str(model), use_cuda=use_cuda, **kwargs)
        except Exception as exc:
            if use_cuda:
                self._voice = PiperVoice.load(str(model), use_cuda=False, **kwargs)
            else:
                raise TtsError(f"Could not load Piper voice '{self.voice_name}': {exc}") from exc
        self.load_s = time.time() - t0
        self._loaded = True

    @staticmethod
    def _cuda_ok() -> bool:
        try:
            import onnxruntime as rt  # noqa: PLC0415

            return "CUDAExecutionProvider" in rt.get_available_providers()
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    def set_voice(self, voice: str) -> None:
        """Switch voices.

        Only invalidates the cache - loading is deferred to the next synthesis so
        that selecting a voice which is not downloaded yet produces a clear error
        at that point instead of blowing up inside the settings dialog.
        """
        if voice == self.voice_name:
            return
        self.voice_name = voice
        self._voice = None
        self._loaded = False

    def synthesize(self, text: str) -> SynthesisResult:
        pieces = list(self.synthesize_stream(text))
        clip = AudioClip(np.zeros(0, dtype=np.int16), 22050)
        for p in pieces:
            clip = clip.concat(p)
        return SynthesisResult(clip, self._last_synth_s, self.name, self.voice_name)

    def synthesize_stream(self, text: str) -> Iterator[AudioClip]:
        self.load()
        from piper.voice import SynthesisConfig  # noqa: PLC0415

        prepared = self._prepare_text(text)
        if not prepared:
            return
        cfg = SynthesisConfig(
            length_scale=float(np.clip(1.0 / max(self.settings.rate, 0.25), 0.4, 3.0)),
            volume=float(np.clip(self.settings.volume, 0.0, 1.0)),
            normalize_audio=False,
        )
        t0 = time.time()
        sr = 22050
        for chunk in self._voice.synthesize(prepared, cfg):  # type: ignore[union-attr]
            sr = int(chunk.sample_rate)
            data = np.asarray(chunk.audio_int16_array, dtype=np.int16)
            if data.size:
                yield AudioClip(np.ascontiguousarray(data), sr)
        self._last_synth_s = time.time() - t0

    _last_synth_s = 0.0

    # ------------------------------------------------------------------ #
    def list_voices(self) -> list[str]:
        """Everything Piper knows about (from the catalogue bundled in the wheel)."""
        from .voices import piper_catalogue  # noqa: PLC0415

        return piper_catalogue()

    def installed_voices(self) -> list[str]:
        d = paths.piper_dir()
        return sorted(p.stem for p in d.glob("*.onnx")) if d.exists() else []


def _find_espeak_data() -> str | None:
    """Locate espeak-ng phoneme data (bundled by piper or by espeakng-loader)."""
    env = os.environ.get("ESPEAK_DATA_DIR")
    if env and Path(env).exists():
        return env
    try:
        from piper.voice import ESPEAK_DATA_DIR  # noqa: PLC0415

        if ESPEAK_DATA_DIR and Path(str(ESPEAK_DATA_DIR)).exists():
            return str(ESPEAK_DATA_DIR)
    except Exception:
        pass
    try:
        import espeakng_loader  # noqa: PLC0415

        return str(espeakng_loader.get_data_path())
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Kokoro-82M (onnxruntime)
# --------------------------------------------------------------------------- #

# Kokoro voice prefixes -> language tag understood by kokoro-onnx.
KOKORO_LANG = {
    "a": "en-us",   # American English
    "b": "en-gb",   # British English
    "j": "ja",
    "z": "zh",
    "e": "es",
    "f": "fr",
    "h": "hi",
    "i": "it",
    "p": "pt-br",
    "_": "en-us",   # multilingual voices from v1.1-zh
}


class KokoroTts(TtsEngine):
    """Kokoro-82M through kokoro-onnx: the best-sounding fully local option here."""

    name = "kokoro"

    def __init__(self, settings: TtsSettings, *, language: str = "en",
                 auto_download: bool = True) -> None:
        super().__init__(settings, language=language)
        self.auto_download = auto_download
        self.voice_name = settings.kokoro_voice or "af_heart"
        self.model_name = settings.kokoro_model or "kokoro-v1.0.int8.onnx"
        self.model_dir = paths.kokoro_dir()
        self._engine: Any = None
        self.load_s = 0.0
        self._last_synth_s = 0.0

    # ------------------------------------------------------------------ #
    def _files(self) -> tuple[Path, Path]:
        model = self.model_dir / self.model_name
        voices = self.model_dir / downloads.KOKORO_VOICES_FOR.get(self.model_name, "voices-v1.0.bin")
        if model.exists() and voices.exists():
            return model, voices
        if not self.auto_download:
            raise TtsError(
                f"Kokoro model not installed ({model.name}).\n"
                "Run: ollavoice setup --tts-engine kokoro"
            )
        return downloads.ensure_kokoro(self.model_name, self.model_dir)

    def load(self) -> None:
        if self._engine is not None:
            self._loaded = True
            return
        try:
            from kokoro_onnx import Kokoro  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise TtsError(
                "Kokoro TTS is not installed.\n  pip install kokoro-onnx\n"
                "or run: scripts\\setup.ps1 --with-tts"
            ) from exc
        model, voices = self._files()
        t0 = time.time()
        try:
            self._engine = Kokoro(str(model), str(voices))
        except Exception as exc:
            raise TtsError(f"Could not load Kokoro ({model.name}): {exc}") from exc
        self.load_s = time.time() - t0
        self._loaded = True

    def set_voice(self, voice: str) -> None:
        self.voice_name = voice

    def _lang(self) -> str:
        prefix = self.voice_name.split("_")[0][:1].lower()
        return KOKORO_LANG.get(prefix, "en-us")

    # ------------------------------------------------------------------ #
    def synthesize(self, text: str) -> SynthesisResult:
        self.load()
        prepared = self._prepare_text(text)
        if not prepared:
            return SynthesisResult(AudioClip(np.zeros(0, dtype=np.int16), 24000),
                                   0.0, self.name, self.voice_name)
        t0 = time.time()
        try:
            audio, sr = self._engine.create(  # type: ignore[union-attr]
                prepared,
                voice=self.voice_name,
                speed=float(np.clip(self.settings.rate, 0.5, 2.0)),
                lang=self._lang(),
            )
        except Exception as exc:
            raise TtsError(f"Kokoro synthesis failed: {exc}") from exc
        self._last_synth_s = time.time() - t0
        clip = float_to_clip(audio, sr)
        clip = _apply_volume(clip, self.settings.volume)
        return SynthesisResult(clip, self._last_synth_s, self.name, self.voice_name)

    def synthesize_stream(self, text: str) -> Iterator[AudioClip]:
        """Kokoro supports an async streaming API; we stream per sentence instead,
        which plays nicely with the pipeline's sentence splitter and needs no
        event loop of its own."""
        from .textsplit import chunk_for_tts  # noqa: PLC0415

        for piece in chunk_for_tts(text, max_chars=300):
            result = self.synthesize(piece)
            if len(result.clip.samples):
                yield result.clip

    # ------------------------------------------------------------------ #
    def list_voices(self) -> list[str]:
        try:
            self.load()
            return list(self._engine.get_voices())  # type: ignore[union-attr]
        except Exception:
            return ["af_heart", "af_bella", "af_sky", "af_nicole", "am_adam", "am_michael",
                    "bf_emma", "bm_george"]


# --------------------------------------------------------------------------- #
# Windows SAPI (zero-download fallback)
# --------------------------------------------------------------------------- #


class SapiTts(TtsEngine):
    """Voices already installed in Windows, via ``System.Speech`` + PowerShell.

    Nothing to download, works on a clean install, sounds like 2009. Good as a
    safety net when the neural voices have not been fetched yet.
    """

    name = "sapi"

    PS_LIST = r"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name + '|' + $_.VoiceInfo.Culture.Name }
$s.Dispose()
"""

    PS_SPEAK = r"""
param([string]$Text, [string]$Voice, [string]$Out, [double]$Rate, [double]$Volume)
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
if ($Voice) { try { $s.SelectVoice($Voice) } catch { } }
$s.Rate = [int][Math]::Round(($Rate - 1.0) * 5)
$s.Volume = [int][Math]::Round($Volume * 100)
$s.SetOutputToWaveFile($Out)
$s.Speak($Text)
$s.SetOutputToNull(); $s.Dispose()
"""

    def __init__(self, settings: TtsSettings, *, language: str = "en") -> None:
        super().__init__(settings, language=language)
        self.voice_name = settings.sapi_voice or "(Windows default)"

    def load(self) -> None:
        if sys.platform != "win32":
            raise TtsError("The 'sapi' engine only runs on Windows.")
        if not (shutil.which("powershell") or shutil.which("pwsh")):
            raise TtsError("PowerShell not found; cannot use the SAPI engine.")
        self._loaded = True

    def synthesize(self, text: str) -> SynthesisResult:
        self.load()
        prepared = self._prepare_text(text)
        if not prepared:
            return SynthesisResult(AudioClip(np.zeros(0, dtype=np.int16), 22050), 0.0,
                                   self.name, self.voice_name)
        out = downloads.temp_wav_path("sapi-")
        t0 = time.time()
        try:
            _run_ps(self.PS_SPEAK, [
                prepared,
                "" if self.voice_name.startswith("(") else self.voice_name,
                str(out), str(float(self.settings.rate)), str(float(self.settings.volume)),
            ])
            from .wav import read_wav  # noqa: PLC0415

            clip = read_wav(out)
        finally:
            try:
                out.unlink(missing_ok=True)
            except OSError:
                pass
        return SynthesisResult(clip, time.time() - t0, self.name, self.voice_name)

    def list_voices(self) -> list[str]:
        if sys.platform != "win32":
            return []
        try:
            out = _run_ps(self.PS_LIST, [])
        except TtsError:
            return []
        voices = []
        for line in out.splitlines():
            line = line.strip()
            if "|" in line:
                voices.append(line.split("|", 1)[0])
        return voices


def _run_ps(script: str, args: list[str], timeout: int = 300) -> str:
    exe = shutil.which("powershell") or shutil.which("pwsh") or "powershell"
    cmd = [exe, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
           "-Command", script, *args]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired as exc:
        raise TtsError("SAPI timed out - the text may be too long for one call") from exc
    except OSError as exc:
        raise TtsError(f"Could not run PowerShell: {exc}") from exc
    if proc.returncode != 0:
        raise TtsError(f"SAPI synthesis failed: {(proc.stderr or '').strip()[:300]}")
    return (proc.stdout or "").strip()


# --------------------------------------------------------------------------- #
# Silent engine
# --------------------------------------------------------------------------- #


class SilentTts(TtsEngine):
    """Speech output disabled. Everything else keeps working."""

    name = "none"

    def __init__(self, settings: TtsSettings, *, language: str = "en") -> None:
        super().__init__(settings, language=language)
        self.voice_name = "(silent)"

    def synthesize(self, text: str) -> SynthesisResult:
        return SynthesisResult(AudioClip(np.zeros(0, dtype=np.int16), 22050), 0.0,
                               self.name, self.voice_name)

    def synthesize_stream(self, text: str) -> Iterator[AudioClip]:
        return iter(())


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

ENGINES: dict[str, type[TtsEngine]] = {
    PiperTts.name: PiperTts,
    KokoroTts.name: KokoroTts,
    SapiTts.name: SapiTts,
    SilentTts.name: SilentTts,
}


def create_engine(settings: TtsSettings, *, language: str = "en",
                  auto_download: bool = True) -> TtsEngine:
    key = (settings.engine or "piper").strip().lower()
    if key in ("silent", "off", "disabled", ""):
        key = "none"
    cls = ENGINES.get(key)
    if cls is None:
        raise TtsError(f"unknown TTS engine {settings.engine!r}; "
                       f"choose from {', '.join(sorted(ENGINES))}")
    if cls is SilentTts or cls is SapiTts:
        return cls(settings, language=language)
    return cls(settings, language=language, auto_download=auto_download)


def available_engines() -> list[dict[str, Any]]:
    out = [
        {"name": "piper", "available": _has_module("piper"),
         "hint": "" if _has_module("piper") else "pip install piper-tts",
         "description": "Fast neural TTS, many voices/languages (recommended default)"},
        {"name": "kokoro", "available": _has_module("kokoro_onnx"),
         "hint": "" if _has_module("kokoro_onnx") else "pip install kokoro-onnx",
         "description": "Kokoro-82M: more natural, ~4x slower on CPU"},
        {"name": "sapi", "available": sys.platform == "win32",
         "hint": "" if sys.platform == "win32" else "Windows only",
         "description": "Built-in Windows voices, zero download"},
        {"name": "none", "available": True, "hint": "",
         "description": "Silent: text only, speech input still works"},
    ]
    return out


def _has_module(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def _apply_volume(clip: AudioClip, volume: float) -> AudioClip:
    if volume >= 0.999 or not len(clip.samples):
        return clip
    scaled = clip.samples.astype(np.float32) * float(np.clip(volume, 0.0, 1.0))
    return AudioClip(np.clip(scaled, -32768, 32767).astype(np.int16), clip.sample_rate)


def installed_voices() -> dict[str, Any]:
    """What is on disk right now - used by the GUI and ``ollavoice doctor``."""
    piper_dir = paths.piper_dir()
    kokoro_dir = paths.kokoro_dir()
    return {
        "piper": sorted(p.stem for p in piper_dir.glob("*.onnx")) if piper_dir.exists() else [],
        "kokoro": sorted(p.name for p in kokoro_dir.glob("*.onnx")) if kokoro_dir.exists() else [],
        "kokoro_voices": [p.name for p in kokoro_dir.glob("voices-*.bin")] if kokoro_dir.exists() else [],
        "piper_size_mb": round(downloads.dir_size_mb(piper_dir), 1),
        "kokoro_size_mb": round(downloads.dir_size_mb(kokoro_dir), 1),
    }
