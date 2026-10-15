"""Speech-to-text engines. All of them run locally.

===========================  ==========================================================
``faster-whisper``  (default) OpenAI Whisper on CTranslate2. int8 on CPU, fp16 on CUDA.
                              Wheels ship for Windows, so no compiler needed.
``whisper-cpp``               Optional: point it at a ``whisper.cpp`` build you already
                              have (``whisper-cli.exe`` + a ``ggml-*.bin`` model). Zero
                              Python dependencies at inference time.
``ollama``                    Optional/experimental: uses Ollama's audio transcription
                              API when your installed server exposes it, with a
                              Whisper-family model you have pulled.
``windows``                   Optional: the recogniser built into Windows, called via
                              PowerShell/System.Speech. On-device, no model download,
                              lowest accuracy of the four.
===========================  ==========================================================

Every engine returns :class:`Transcription`, which keeps the text, per-segment
timings, the detected language and how long inference took (shown in the GUI).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import downloads, paths
from .config import SttSettings, effective_language, localize_whisper_model, resolve_whisper_model
from .hardware import detect, whisper_compute_type, whisper_threads
from .wav import AudioClip

WHISPER_SIZES = ("tiny", "base", "small", "medium", "large-v3", "distil-small.en",
                 "distil-medium.en", "distil-large-v3")


class SttError(RuntimeError):
    pass


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Transcription:
    text: str
    language: str = ""
    language_probability: float = 0.0
    duration_s: float = 0.0
    decode_s: float = 0.0
    segments: list[Segment] = field(default_factory=list)
    engine: str = ""
    model: str = ""

    @property
    def realtime_factor(self) -> float:
        """decode time / audio time. Below 1.0 means faster than realtime."""
        if self.duration_s <= 0:
            return 0.0
        return self.decode_s / self.duration_s

    def summary(self) -> str:
        rt = self.realtime_factor
        return (f"{self.engine}/{self.model} - {self.duration_s:.1f}s audio in "
                f"{self.decode_s:.2f}s" + (f" ({rt:.2f}x realtime)" if rt else "") +
                (f" [{self.language} {self.language_probability:.0%}]" if self.language else ""))


class SttEngine(ABC):
    name = "base"

    def __init__(self, settings: SttSettings, *, app_language: str = "") -> None:
        self.settings = settings
        #: The app-wide language (``config.language``), used as a fallback when
        #: ``stt.language`` is left on "auto". Whisper model choice depends on it:
        #: the English-only ``*.en`` checkpoints are better at English but cannot
        #: detect or handle any other language.
        self.app_language = app_language or ""
        self._loaded = False

    @property
    def language(self) -> str:
        """The effective language, or ``auto`` to let Whisper detect it."""
        return effective_language(self.settings.language, self.app_language)

    @abstractmethod
    def transcribe(self, clip: AudioClip | Path | str, *, language: str | None = None,
                   initial_prompt: str | None = None) -> Transcription:
        ...

    def load(self) -> None:
        """Warm the model up (called on a worker thread by the GUI)."""
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def describe(self) -> dict[str, Any]:
        return {"engine": self.name, "model": getattr(self, "model_name", ""),
                "loaded": self._loaded}

    # ------------------------------------------------------------------ #
    def _resolve_model(self) -> str:
        """Turn the configured model name into a real faster-whisper model id.

        ``auto`` means "the tier default". The effective language decides whether
        an English-only ``*.en`` checkpoint can be used: those are smaller and
        noticeably better at English, but they cannot do anything else, and they
        cannot auto-detect the language either.
        """
        model = (self.settings.model or "auto").strip() or "auto"
        if model == "auto":
            model = "base.en"
        return resolve_whisper_model(model, self.language)


# --------------------------------------------------------------------------- #
# faster-whisper (default)
# --------------------------------------------------------------------------- #


class FasterWhisperStt(SttEngine):
    """Whisper on CTranslate2. The workhorse: accurate, fast, easy to install."""

    name = "faster-whisper"

    def __init__(self, settings: SttSettings, *, app_language: str = "") -> None:
        super().__init__(settings, app_language=app_language)
        self.model_name = self._resolve_model()
        self._model: Any = None
        hw = detect()
        device, compute = whisper_compute_type(hw, settings.device)
        if settings.device != "auto":
            device = settings.device
        if settings.compute_type != "auto":
            compute = settings.compute_type
        if device == "cuda":
            try:
                import ctranslate2  # noqa: PLC0415

                if not ctranslate2.get_cuda_device_count():
                    device, compute = "cpu", "int8_float16" if hw.avx2 else "int8"
            except Exception:
                device, compute = "cpu", "int8_float16" if hw.avx2 else "int8"
        self.device = device
        self.compute_type = compute
        self.cpu_threads = whisper_threads(hw, settings.cpu_threads)
        self.download_root = str(paths.whisper_dir())

    # ------------------------------------------------------------------ #
    def load(self) -> None:
        if self._model is not None:
            self._loaded = True
            return
        try:
            from faster_whisper import WhisperModel  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise SttError(
                "faster-whisper is not installed.\n"
                "  pip install \"faster-whisper>=1.0\"\n"
                "or run: scripts\\setup.ps1 --with-stt"
            ) from exc

        _prepare_offline_hf_cache(self.model_name)
        t0 = time.time()
        self._model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
            download_root=self.download_root,
        )
        self.load_s = time.time() - t0
        self._loaded = True

    def unload(self) -> None:
        self._model = None
        self._loaded = False

    # ------------------------------------------------------------------ #
    def transcribe(self, clip: AudioClip | Path | str, *, language: str | None = None,
                   initial_prompt: str | None = None) -> Transcription:
        self.load()
        assert self._model is not None

        source, cleanup = _as_audio_source(clip)
        lang = language or self.language
        lang = None if lang in ("auto", "", None) else lang
        prompt = initial_prompt if initial_prompt is not None else self.settings.initial_prompt

        t0 = time.time()
        # NB: faster-whisper rejects task=None, so only pass it when translating.
        extra: dict[str, Any] = {}
        if self.settings.translate_to_english and lang is None:
            extra["task"] = "translate"
        try:
            segments_iter, info = self._model.transcribe(
                source if not isinstance(source, Path) else str(source),
                language=lang,
                beam_size=int(self.settings.beam_size),
                vad_filter=bool(self.settings.vad_filter),
                condition_on_previous_text=bool(self.settings.condition_on_previous_text),
                initial_prompt=prompt or None,
                **extra,
            )
            segs = [Segment(float(s.start), float(s.end), s.text.strip())
                    for s in segments_iter if s.text and s.text.strip()]
        finally:
            if cleanup and isinstance(source, Path):
                _safe_unlink(source)

        text = _join_segments([s.text for s in segs])
        return Transcription(
            text=text,
            language=getattr(info, "language", "") or (lang or ""),
            language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
            duration_s=float(getattr(info, "duration", 0.0) or 0.0),
            decode_s=time.time() - t0,
            segments=segs,
            engine=self.name,
            model=self.model_name,
        )

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        base.update({"device": self.device, "compute_type": self.compute_type,
                     "cpu_threads": self.cpu_threads, "model": self.model_name})
        return base


def _prepare_offline_hf_cache(model: str = "") -> None:
    """Point huggingface_hub at our own cache, and go offline when we can.

    Two reasons this matters:

    * a vendor bundle (``ollavoice bundle`` / ``setup --offline``) is only used if
      ``HF_HOME`` points at it;
    * once a model is on disk, ``huggingface_hub`` still phones home to check for
      a newer revision. That adds a stall - or a hard failure on a machine with
      no network - on every start, which defeats the point of a local-only tool.
    """
    vendor_hf = paths.vendor_dir() / "hf-cache"
    if vendor_hf.exists() and not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = str(vendor_hf)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        return

    if not model or os.environ.get("HF_HUB_OFFLINE"):
        return
    root = paths.whisper_dir()
    candidates = [
        root / f"models--Systran--faster-whisper-{model}",
        root / model,
    ]
    if any(p.exists() and any(p.iterdir()) for p in candidates if p.exists()):
        os.environ.setdefault("HF_HOME", str(root.parent))
        os.environ["HF_HUB_OFFLINE"] = "1"


# --------------------------------------------------------------------------- #
# whisper.cpp (optional, bring-your-own binary)
# --------------------------------------------------------------------------- #


class WhisperCppStt(SttEngine):
    """Drive a local ``whisper.cpp`` build.

    Put ``whisper-cli.exe`` (or ``main.exe``) and a ``ggml-*.bin`` model into
    ``%LOCALAPPDATA%\\OllaVoice\\models\\whisper.cpp`` - or set
    ``OLLAVOICE_WHISPERCPP_BIN`` / ``OLLAVOICE_WHISPERCPP_MODEL``.

    This path is popular with people who already run llama.cpp/whisper.cpp and
    want one binary handling all inference.
    """

    name = "whisper-cpp"

    def __init__(self, settings: SttSettings, *, app_language: str = "") -> None:
        super().__init__(settings, app_language=app_language)
        self.model_name = settings.model
        self.binary = self._find_binary()
        self.model_path = self._find_model()

    def _find_binary(self) -> str:
        env = os.environ.get("OLLAVOICE_WHISPERCPP_BIN")
        if env:
            return env
        d = paths.whispercpp_dir()
        names = ["whisper-cli.exe", "whisper-cli", "main.exe", "main"] if sys.platform == "win32" \
            else ["whisper-cli", "main"]
        for n in names:
            p = d / n
            if p.exists():
                return str(p)
        found = shutil.which("whisper-cli") or shutil.which("whisper.cpp")
        return found or ""

    def _find_model(self) -> str:
        env = os.environ.get("OLLAVOICE_WHISPERCPP_MODEL")
        if env:
            return env
        d = paths.whispercpp_dir()
        if d.exists():
            cands = sorted(d.glob("ggml-*.bin"))
            if cands:
                return str(cands[0])
        return ""

    def load(self) -> None:
        if not self.binary:
            raise SttError(
                "whisper.cpp binary not found.\n"
                f"Download a Windows build from https://github.com/ggml-org/whisper.cpp/releases\n"
                f"and put whisper-cli.exe in {paths.whispercpp_dir()}\n"
                "(or set OLLAVOICE_WHISPERCPP_BIN)"
            )
        if not self.model_path or not Path(self.model_path).exists():
            raise SttError(
                "No ggml Whisper model found.\n"
                f"Put e.g. ggml-base.en.bin in {paths.whispercpp_dir()}\n"
                "(or set OLLAVOICE_WHISPERCPP_MODEL)"
            )
        self._loaded = True

    def transcribe(self, clip: AudioClip | Path | str, *, language: str | None = None,
                   initial_prompt: str | None = None) -> Transcription:
        self.load()
        wav_path, cleanup = _as_wav_path(clip, force_16k=True)
        cmd = [self.binary, "-m", self.model_path, "-f", str(wav_path),
               "-nt", "--no-prints", "-of", str(wav_path.with_suffix("")), "-otxt"]
        lang = language or self.settings.language
        if lang and lang != "auto":
            cmd += ["-l", lang]
        else:
            cmd += ["-l", "auto"]
        prompt = initial_prompt if initial_prompt is not None else self.settings.initial_prompt
        if prompt:
            cmd += ["--prompt", prompt]
        threads = whisper_threads(detect(), self.settings.cpu_threads)
        cmd += ["-t", str(threads)]

        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        finally:
            pass
        txt_path = wav_path.with_suffix(".txt")
        text = ""
        if txt_path.exists():
            text = txt_path.read_text(encoding="utf-8", errors="replace").strip()
            _safe_unlink(txt_path)
        if not text:
            text = (proc.stdout or "").strip()
        if cleanup:
            _safe_unlink(wav_path)
        if proc.returncode != 0 and not text:
            raise SttError(f"whisper.cpp failed ({proc.returncode}): {(proc.stderr or '')[:400]}")
        return Transcription(text=_clean_transcript(text), decode_s=time.time() - t0,
                             engine=self.name, model=Path(self.model_path).stem)


# --------------------------------------------------------------------------- #
# Ollama-hosted transcription (optional / experimental)
# --------------------------------------------------------------------------- #


class OllamaStt(SttEngine):
    """Use Ollama's own transcription endpoint, when the installed server has one.

    Ollama does not (as of v0.40.x) document an audio transcription API, so this
    engine probes for ``POST /api/transcriptions`` at runtime and raises a clear
    error if the server rejects it. Nothing else about OllaVoice depends on it.
    """

    name = "ollama"

    def __init__(self, settings: SttSettings, host: str = "http://127.0.0.1:11434",
                 *, app_language: str = "") -> None:
        super().__init__(settings, app_language=app_language)
        self.host = host.rstrip("/")
        self.model_name = settings.model if settings.model not in ("", "auto") else "whisper"

    def load(self) -> None:
        self._loaded = True

    def transcribe(self, clip: AudioClip | Path | str, *, language: str | None = None,
                   initial_prompt: str | None = None) -> Transcription:
        wav_path, cleanup = _as_wav_path(clip, force_16k=True)
        try:
            boundary = "----ollavoice"
            body = _multipart(boundary, {"model": self.model_name},
                              ("file", wav_path.name, wav_path.read_bytes(), "audio/wav"))
            req = urllib.request.Request(
                self.host + "/api/transcriptions", data=body, method="POST",
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            )
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    data = json.loads(resp.read().decode("utf-8", "replace"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:300]
                raise SttError(
                    f"This Ollama server rejected /api/transcriptions (HTTP {exc.code}).\n"
                    "Ollama has no native speech-to-text API in current releases; use the "
                    "'faster-whisper' engine instead.\n" f"Server said: {detail}"
                ) from exc
        finally:
            if cleanup:
                _safe_unlink(wav_path)

        text = str(data.get("text") or data.get("transcript") or "").strip()
        return Transcription(text=_clean_transcript(text), decode_s=time.time() - t0,
                             engine=self.name, model=self.model_name)


def _multipart(boundary: str, fields: dict[str, str],
               file_field: tuple[str, str, bytes, str]) -> bytes:
    out = bytearray()
    for k, v in fields.items():
        out += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    name, filename, blob, ctype = file_field
    out += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; "
            f"filename=\"{filename}\"\r\nContent-Type: {ctype}\r\n\r\n").encode()
    out += blob + f"\r\n--{boundary}--\r\n".encode()
    return bytes(out)


# --------------------------------------------------------------------------- #
# Windows built-in speech recognition (optional)
# --------------------------------------------------------------------------- #


class WindowsStt(SttEngine):
    """The recogniser built into Windows, driven through ``System.Speech``.

    Zero model download and no Python ML stack - it calls the same on-device
    speech engine Windows itself uses, via PowerShell (so there is nothing extra
    to install and it works on a clean Windows 10/11 machine). Accuracy is well
    below Whisper and only languages with an installed Windows speech pack work,
    but it is a handy no-download fallback.
    """

    name = "windows"

    def __init__(self, settings: SttSettings, *, app_language: str = "") -> None:
        super().__init__(settings, app_language=app_language)
        self.model_name = "windows-system-speech"

    def load(self) -> None:
        if sys.platform != "win32":
            raise SttError("The 'windows' STT engine only runs on Windows.")
        if not (shutil.which("powershell") or shutil.which("pwsh")):
            raise SttError("PowerShell was not found on PATH; cannot use the Windows engine.")
        self._loaded = True

    def transcribe(self, clip: AudioClip | Path | str, *, language: str | None = None,
                   initial_prompt: str | None = None) -> Transcription:
        self.load()
        wav_path, cleanup = _as_wav_path(clip, force_16k=True)
        lang = language or self.settings.language or "auto"
        culture = _bcp47(lang) if lang not in ("auto", "") else ""
        t0 = time.time()
        try:
            text = _powershell(PS_TRANSCRIBE, [str(wav_path), culture])
        finally:
            if cleanup:
                _safe_unlink(wav_path)
        return Transcription(text=_clean_transcript(text), decode_s=time.time() - t0,
                             engine=self.name, model=self.model_name)


def _powershell(script: str, args: list[str], timeout: int = 300) -> str:
    """Run a PowerShell snippet, returning stdout. Raises SttError with stderr."""
    exe = shutil.which("powershell") or shutil.which("pwsh") or "powershell"
    cmd = [exe, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
           "-Command", script, *args]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired as exc:
        raise SttError(f"PowerShell timed out after {timeout}s") from exc
    except OSError as exc:
        raise SttError(f"Could not run PowerShell: {exc}") from exc
    if proc.returncode != 0 and not proc.stdout.strip():
        raise SttError(f"PowerShell failed: {(proc.stderr or '').strip()[:400]}")
    return (proc.stdout or "").strip()


# args: [0] = wav path, [1] = culture ("en-US"; "" => Windows UI culture)
PS_TRANSCRIBE = r"""
param([string]$Wav, [string]$Culture)
Add-Type -AssemblyName System.Speech
try {
  if ($Culture) { $ci = New-Object System.Globalization.CultureInfo($Culture) }
  else { $ci = [System.Globalization.CultureInfo]::CurrentUICulture }
  $eng = New-Object System.Speech.Recognition.SpeechRecognitionEngine($ci)
  $eng.LoadGrammar((New-Object System.Speech.Recognition.DictationGrammar))
  $eng.SetInputToWaveFile($Wav)
  $sb = New-Object System.Text.StringBuilder
  while ($true) {
    $r = $eng.Recognize()
    if ($null -eq $r) { break }
    [void]$sb.Append($r.Text); [void]$sb.Append(' ')
  }
  $eng.SetInputToNull(); $eng.Dispose()
  Write-Output $sb.ToString().Trim()
} catch {
  Write-Error $_.Exception.Message
  exit 1
}
"""


def _bcp47(language: str) -> str:
    mapping = {"en": "en-US", "zh": "zh-CN", "pt": "pt-BR", "es": "es-ES",
               "de": "de-DE", "fr": "fr-FR", "it": "it-IT", "ja": "ja-JP",
               "ko": "ko-KR", "ru": "ru-RU", "nl": "nl-NL", "sv": "sv-SE",
               "da": "da-DK", "no": "nb-NO", "fi": "fi-FI", "pl": "pl-PL",
               "cs": "cs-CZ", "tr": "tr-TR", "ar": "ar-SA", "he": "he-IL"}
    if not language:
        return ""
    if "-" in language:
        return language
    return mapping.get(language.split("_")[0].lower(), language)


# --------------------------------------------------------------------------- #
# Factory + shared helpers
# --------------------------------------------------------------------------- #

ENGINES: dict[str, type[SttEngine]] = {
    FasterWhisperStt.name: FasterWhisperStt,
    WhisperCppStt.name: WhisperCppStt,
    OllamaStt.name: OllamaStt,
    WindowsStt.name: WindowsStt,
}


def create_engine(settings: SttSettings, *, ollama_host: str = "http://127.0.0.1:11434",
                  app_language: str = "") -> SttEngine:
    key = (settings.engine or "faster-whisper").strip().lower()
    if key in ("whisper_cpp", "whispercpp", "whisper.cpp"):
        key = "whisper-cpp"
    cls = ENGINES.get(key)
    if cls is None:
        raise SttError(f"unknown STT engine {settings.engine!r}; "
                       f"choose from {', '.join(sorted(ENGINES))}")
    if cls is OllamaStt:
        return cls(settings, host=ollama_host, app_language=app_language)
    return cls(settings, app_language=app_language)


def available_engines() -> list[dict[str, Any]]:
    """Status of every STT engine, for the GUI dropdown and ``ollavoice doctor``."""
    out = []
    out.append(_probe("faster-whisper", _has_module("faster_whisper"),
                      "pip install faster-whisper", "Whisper on CTranslate2 (recommended)"))
    wcpp = WhisperCppStt(SttSettings(engine="whisper-cpp"))
    out.append(_probe("whisper-cpp", bool(wcpp.binary and wcpp.model_path),
                      f"put whisper-cli + a ggml model in {paths.whispercpp_dir()}",
                      "Optional bring-your-own whisper.cpp build"))
    out.append(_probe("ollama", True, "", "Experimental: needs an Ollama build with /api/transcriptions"))
    out.append(_probe("windows", sys.platform == "win32"
                      and bool(shutil.which("powershell") or shutil.which("pwsh")),
                      "Windows only - needs PowerShell (present by default)",
                      "Built-in on-device Windows recogniser, no model download"))
    return out


def _probe(name: str, ok: bool, hint: str, description: str) -> dict[str, Any]:
    return {"name": name, "available": bool(ok), "hint": "" if ok else hint,
            "description": description}


def _has_module(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False


AudioInput = AudioClip | Path | str | bytes | Any


def _as_wav_path(clip: AudioInput, *, force_16k: bool = False) -> tuple[Path, bool]:
    """Return a path to a 16-bit mono WAV, writing a temp file if needed."""
    if isinstance(clip, AudioClip):
        data = clip.resample(16000) if force_16k and clip.sample_rate != 16000 else clip
        p = downloads.temp_wav_path()
        data.write_wav(p)
        return p, True
    if isinstance(clip, (bytes, bytearray)):
        p = downloads.temp_wav_path("upload-")
        p.write_bytes(bytes(clip))
        return p.with_suffix(p.suffix), True
    p = Path(str(clip))
    if not p.exists():
        raise SttError(f"audio file not found: {p}")
    return p, False


def _as_audio_source(clip: AudioInput, *, force_16k: bool = False) -> tuple[Any, bool]:
    """Like :func:`_as_wav_path` but also accepts encoded bytes (mp3/ogg/webm).

    faster-whisper decodes through its bundled ffmpeg libraries, so a browser
    upload of ``audio/webm`` works without any external tool installed. The
    second return value says whether a temp file was written and must be deleted.
    """
    if isinstance(clip, (bytes, bytearray)):
        blob = bytes(clip)
        if blob[:4] == b"RIFF" and blob[8:12] == b"WAVE":
            path, cleanup = _as_wav_path(blob, force_16k=force_16k)
            return path, cleanup
        import io

        return io.BytesIO(blob), False
    return _as_wav_path(clip, force_16k=force_16k)


def _safe_unlink(p: Path | str) -> None:
    try:
        os.unlink(p)
    except OSError:
        pass


def _join_segments(texts: list[str]) -> str:
    """Whisper segments often start with a space; join them without doubling up."""
    out: list[str] = []
    for t in texts:
        t = t.strip()
        if not t:
            continue
        if out and not out[-1].endswith((".", "!", "?", ":", ",", ";")) and not t.startswith(("'", "-", ",")):
            out.append(" ")
        out.append(t)
    return "".join(out).strip()


def _clean_transcript(text: str) -> str:
    text = (text or "").strip()
    # Whisper loves to hallucinate these on silence.
    for junk in ("[BLANK_AUDIO]", "[BLANK AUDIO]", "(blank audio)", "[Music]", "[MUSIC]",
                 "Thanks for watching!", "Thank you.", "you", " subtitles by"):
        if text.lower().startswith(junk.lower()):
            text = text[len(junk):].strip()
    return text.strip()


def whisper_model_files(model: str) -> list[Path]:
    """Which files a faster-whisper model occupies on disk (for the doctor)."""
    root = paths.whisper_dir()
    if not root.exists():
        return []
    stem = localize_whisper_model(model, "en") if model.endswith(".en") else model
    hits = [p for p in root.rglob("*") if stem.split(".")[0] in p.name and p.is_file()]
    return hits
