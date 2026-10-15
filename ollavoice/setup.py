"""First-run setup: detect hardware, pick a tier, fetch models, verify the loop.

``ollavoice setup`` is interactive by default; ``--yes`` makes it fully
non-interactive (what the Windows installer script uses).

Setup is the *only* moment OllaVoice touches the network. It downloads:

* a Whisper model (via ``faster-whisper`` / huggingface_hub)
* one or two Piper voice ONNX files (~60 MB each), and/or a Kokoro checkpoint
* optionally an Ollama chat model (via ``ollama pull``)

``--offline`` skips all of that and installs from a bundle made by
``ollavoice bundle`` on a machine that *does* have internet access.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import downloads, hardware, paths as _paths, stt_engines, tts_engines
from .cli import _make_progress, bold, cyan, dim, green, red, yellow
from .config import TIERS, TIER_ORDER, Config, apply_tier, load as load_config, save as save_config
from .ollama_client import OllamaClient, OllamaError


def _ask(prompt: str, default: str = "", choices: list[str] | None = None,
         yes: bool = False) -> str:
    if yes:
        return default
    suffix = f" [{default}]" if default else ""
    if choices:
        suffix = f" ({'/'.join(choices)}, default {default})"
    while True:
        try:
            answer = input(f"{prompt}{suffix}: ").strip()
        except EOFError:
            return default
        if not answer:
            return default
        if not choices or answer.lower() in [c.lower() for c in choices]:
            return answer
        print(yellow(f"  please choose one of: {', '.join(choices)}"))


def _step(n: int, title: str) -> None:
    print(f"\n{bold(f'[{n}]')} {bold(title)}")


def run_setup(args: Any) -> int:
    yes = bool(getattr(args, "yes", False))
    offline = bool(getattr(args, "offline", False)) or downloads.offline_mode()
    if offline:
        os.environ["OLLAVOICE_OFFLINE"] = "1"
    _paths.ensure_dirs()

    print(bold("OllaVoice setup"))
    print(dim("Speech input and speech output for Ollama - everything stays on this machine."))
    print(dim(f"config: {_paths.config_file()}"))
    print(dim(f"models: {_paths.data_root()}"))
    if offline:
        print(yellow("OFFLINE mode: no network access; installing from the vendor bundle."))

    cfg = load_config(getattr(args, "config", None), create_if_missing=True)

    # ---------------------------------------------------------------- 1
    _step(1, "Hardware")
    hw = hardware.detect()
    print("  " + hw.summary().replace("\n", "\n  "))
    tier, reasons = hardware.recommend_tier(hw)
    for r in reasons:
        print(dim(f"  -> {r}"))
    if getattr(args, "tier", None):
        tier = args.tier
    elif not yes:
        chosen = _ask("  Performance tier", tier, [*TIER_ORDER, "custom"])
        tier = chosen if chosen in TIERS else tier
    print(green(f"  using tier: {tier}") + dim(f"  ({TIERS.get(tier, {}).get('_note', '')})"))
    if tier != cfg.tier:
        apply_tier(cfg, tier, language=getattr(args, "language", None) or cfg.language)

    # ---------------------------------------------------------------- 2
    _step(2, "Ollama")
    client = OllamaClient(cfg.ollama.host, cfg.ollama.api_key, timeout=5)
    if client.is_alive():
        version = client.version()
        print(green(f"  online at {cfg.ollama.host} (v{version})"))
        have = client.model_names()
        if have:
            print(dim(f"  {len(have)} model(s): " + ", ".join(have[:10])
                      + (" ..." if len(have) > 10 else "")))
        want_model = getattr(args, "model", None) or cfg.ollama.model
        if not want_model:
            want_model = TIERS.get(tier, {}).get("ollama.model", "llama3.2:3b")
        if client.has_model(want_model):
            cfg.ollama.model = want_model
            print(green(f"  model '{want_model}' is already installed"))
        else:
            print(yellow(f"  model '{want_model}' is NOT installed"))
            do_pull = not getattr(args, "no_pull", False) and (
                getattr(args, "pull", False) or yes or
                _ask("  Pull it now?", "y", ["y", "n"]).lower().startswith("y"))
            cfg.ollama.model = want_model
            if do_pull and not offline:
                _pull_model(client, want_model)
            else:
                print(dim(f"  (pull it later with: ollama pull {want_model})"))
    else:
        print(red(f"  cannot reach Ollama at {cfg.ollama.host}"))
        if sys.platform == "win32":
            print(dim("  Install/start it: winget install Ollama.Ollama  (then run 'ollama serve')"))
            print(dim("  Or download OllamaSetup.exe from https://ollama.com/download/windows"))
        else:
            print(dim("  Start it with: ollama serve   (install: https://ollama.com/download)"))
        want_model = getattr(args, "model", None) or cfg.ollama.model or \
            TIERS.get(tier, {}).get("ollama.model", "llama3.2:3b")
        cfg.ollama.model = want_model
        print(dim(f"  Continuing anyway; model recorded as '{want_model}'."))

    # ---------------------------------------------------------------- 3
    _step(3, "Speech input (STT)")
    if getattr(args, "stt_engine", None):
        cfg.stt.engine = args.stt_engine
    if getattr(args, "stt_model", None):
        cfg.stt.model = args.stt_model
    if getattr(args, "language", None):
        cfg.stt.language = args.language
        cfg.language = args.language
        cfg.stt.model = _localize(cfg.stt.model, args.language)

    engines = {e["name"]: e for e in stt_engines.available_engines()}
    for name, info in engines.items():
        mark = green("ok  ") if info["available"] else dim("--  ")
        print(f"  {mark}{name:<16}{dim(info['description'])}")
        if not info["available"] and info["hint"]:
            print(dim(f"        install: {info['hint']}"))

    if cfg.stt.engine not in engines or not engines[cfg.stt.engine]["available"]:
        if engines["faster-whisper"]["available"]:
            cfg.stt.engine = "faster-whisper"
        else:
            print(red("  No usable STT engine. Install one with:"))
            print(cyan("    pip install faster-whisper"))
            print(dim("    (or run scripts\\setup.ps1 --with-stt)"))

    device, compute = hardware.whisper_compute_type(hw, cfg.stt.device)
    cfg.stt.device, cfg.stt.compute_type = device, compute
    cfg.stt.cpu_threads = hardware.whisper_threads(hw, cfg.stt.cpu_threads)
    print(dim(f"  device={device} compute={compute} threads={cfg.stt.cpu_threads} "
              f"model={cfg.stt.model}"))

    if not getattr(args, "skip_stt", False) and cfg.stt.engine == "faster-whisper":
        _fetch_whisper(cfg, offline=offline)

    # ---------------------------------------------------------------- 4
    _step(4, "Speech output (TTS)")
    if getattr(args, "tts_engine", None):
        cfg.tts.engine = args.tts_engine
    if getattr(args, "tts_voice", None):
        cfg.tts.piper_voice = args.tts_voice
        if not getattr(args, "tts_engine", None):
            cfg.tts.engine = "piper"
    if getattr(args, "kokoro_voice", None):
        cfg.tts.kokoro_voice = args.kokoro_voice
        if not getattr(args, "tts_engine", None):
            cfg.tts.engine = "kokoro"

    for info in tts_engines.available_engines():
        mark = green("ok  ") if info["available"] else dim("--  ")
        print(f"  {mark}{info['name']:<16}{dim(info['description'])}")
        if not info["available"] and info["hint"]:
            print(dim(f"        install: {info['hint']}"))

    if cfg.tts.engine == "piper" and not tts_engines._has_module("piper"):
        print(red("  Piper is not installed. Either:"))
        print(cyan("    pip install piper-tts"))
        print(dim("  ...or fall back to the built-in Windows voices with:"))
        print(cyan("    ollavoice config set tts.engine sapi"))
    if cfg.tts.engine == "kokoro" and not tts_engines._has_module("kokoro_onnx"):
        print(red("  kokoro-onnx is not installed: pip install kokoro-onnx"))

    if not getattr(args, "skip_tts", False):
        _fetch_tts(cfg, offline=offline, yes=yes)

    # ---------------------------------------------------------------- 5
    _step(5, "Save + verify")
    save_config(cfg, getattr(args, "config", None))
    print(green(f"  config saved to {_paths.config_file()}"))
    print(dim(f"  disk used by speech models: "
              f"{downloads.dir_size_mb(_paths.data_root()):.0f} MB"))

    failures = _verify(cfg, offline=offline)

    print()
    if failures:
        print(yellow("Setup finished with warnings:"))
        for f in failures:
            print(f"  - {f}")
    else:
        print(green("Setup complete. The speech loop works end to end."))
    print()
    print("Next:")
    print("  " + cyan("ollavoice gui") + dim("    launch the desktop app"))
    print("  " + cyan("ollavoice chat") + dim("   talk to Ollama in this terminal"))
    print("  " + cyan("ollavoice doctor --test") + dim("  re-run the self-test"))
    return 1 if failures else 0


# --------------------------------------------------------------------------- #


def _localize(model: str, language: str) -> str:
    from .config import effective_language, resolve_whisper_model

    return resolve_whisper_model(model, effective_language(language))


def _pull_model(client: OllamaClient, model: str) -> bool:
    state = {"last": 0.0}

    def progress(evt: dict[str, Any]) -> None:
        status = evt.get("status", "")
        total = evt.get("total") or 0
        done = evt.get("completed") or 0
        now = time.time()
        if now - state["last"] < 0.2 and total and done < total:
            return
        state["last"] = now
        if total:
            sys.stdout.write("\r  " + status[:24].ljust(24) + " " +
                             _bar(done, total) + "   ")
        else:
            sys.stdout.write("\r  " + status[:60].ljust(60))
        sys.stdout.flush()

    try:
        client.pull(model, progress=progress)
        sys.stdout.write("\r" + " " * 78 + "\r")
        print(green(f"  pulled {model}"))
        return True
    except OllamaError as exc:
        sys.stdout.write("\n")
        print(red(f"  pull failed: {exc}"))
        return False
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        print(yellow("  pull cancelled"))
        return False


def _bar(done: float, total: float, width: int = 28) -> str:
    if not total:
        return f"{done / 1e6:.1f} MB"
    frac = min(1.0, done / total)
    filled = int(width * frac)
    return f"[{'#' * filled}{'-' * (width - filled)}] {frac * 100:5.1f}%"


def _fetch_whisper(cfg: Config, *, offline: bool) -> None:
    """Make sure the faster-whisper model is on disk."""
    from .config import effective_language, resolve_whisper_model

    model = resolve_whisper_model(cfg.stt.model or "auto",
                                  effective_language(cfg.stt.language, cfg.language))
    root = _paths.whisper_dir()
    root.mkdir(parents=True, exist_ok=True)

    if offline:
        vendor = _paths.vendor_dir() / "hf-cache"
        if vendor.exists():
            os.environ["HF_HOME"] = str(vendor)
            os.environ["HF_HUB_OFFLINE"] = "1"
            print(dim(f"  offline: using HF cache {vendor}"))
        else:
            print(yellow("  offline: no vendored Whisper cache found; skipping download"))
            return

    try:
        from faster_whisper import WhisperModel  # noqa: PLC0415
    except ImportError:
        print(red("  faster-whisper is not installed - skipping the model download"))
        return

    print(dim(f"  downloading Whisper '{model}' (~{_whisper_size_mb(model):.0f} MB) "
              f"into {root}"))
    t0 = time.time()
    try:
        device, compute = hardware.whisper_compute_type(None, cfg.stt.device)
        WhisperModel(model, device=device, compute_type=compute,
                     cpu_threads=max(1, cfg.stt.cpu_threads or 1), download_root=str(root))
        print(green(f"  Whisper '{model}' ready ({time.time() - t0:.1f}s)"))
    except Exception as exc:
        print(red(f"  Whisper download/load failed: {exc}"))
        print(dim("  If your network blocks huggingface.co, set HF_ENDPOINT to a mirror, e.g."))
        print(cyan("    set HF_ENDPOINT=https://hf-mirror.com"))


def _whisper_size_mb(model: str) -> float:
    base = {"tiny": 75, "base": 145, "small": 484, "medium": 1533, "large-v3": 3095,
            "distil-small.en": 262, "distil-medium.en": 769, "distil-large-v3": 1550}
    stem = model.replace(".en", "")
    return float(base.get(stem, 200))


def _fetch_tts(cfg: Config, *, offline: bool, yes: bool) -> None:
    want_piper = cfg.tts.engine == "piper"
    want_kokoro = cfg.tts.engine == "kokoro"
    if not yes and not offline:
        extra = _ask("  Also install the other neural engine for switching later?",
                     "n", ["y", "n"]).lower().startswith("y")
        if extra:
            want_piper = want_kokoro = True

    if cfg.tts.engine == "sapi":
        if sys.platform != "win32":
            print(yellow("  'sapi' is Windows-only; keeping it configured anyway"))
        else:
            print(dim("  SAPI uses the voices already installed in Windows - nothing to fetch"))
        return
    if cfg.tts.engine == "none":
        print(dim("  speech output disabled"))
        return

    if want_piper and tts_engines._has_module("piper"):
        voice = cfg.tts.piper_voice or "en_US-lessac-medium"
        onnx = _paths.piper_dir() / f"{voice}.onnx"
        if onnx.exists():
            print(green(f"  Piper voice '{voice}' already installed"))
        elif offline:
            vendored = downloads.copy_from_vendor(f"{voice}.onnx", _paths.piper_dir())
            downloads.copy_from_vendor(f"{voice}.onnx.json", _paths.piper_dir())
            print(green(f"  Piper voice '{voice}' installed from bundle") if vendored
                  else yellow(f"  offline: voice '{voice}' not in the bundle"))
        else:
            print(dim(f"  downloading Piper voice '{voice}' (~60 MB)"))
            try:
                downloads.ensure_piper_voice(voice, _paths.piper_dir(),
                                             progress=_make_progress("piper"))
                print(green(f"  Piper voice '{voice}' ready"))
            except Exception as exc:
                print(red(f"  Piper voice download failed: {exc}"))

    if want_kokoro and tts_engines._has_module("kokoro_onnx"):
        model = cfg.tts.kokoro_model or "kokoro-v1.0.int8.onnx"
        if (_paths.kokoro_dir() / model).exists():
            print(green(f"  Kokoro '{model}' already installed"))
        elif offline:
            print(yellow("  offline: Kokoro weights must be copied manually into "
                         f"{_paths.kokoro_dir()}"))
        else:
            print(dim(f"  downloading Kokoro '{model}' + voices (~140 MB)"))
            try:
                downloads.ensure_kokoro(model, _paths.kokoro_dir(),
                                        progress=_make_progress("kokoro"))
                print(green(f"  Kokoro '{model}' ready"))
            except Exception as exc:
                print(red(f"  Kokoro download failed: {exc}"))


def _verify(cfg: Config, *, offline: bool) -> list[str]:
    """Synthesise a sentence, transcribe it back, and ping Ollama."""
    problems: list[str] = []

    # --- TTS
    try:
        engine = tts_engines.create_engine(cfg.tts, language=cfg.language,
                                           auto_download=not offline)
        engine.load()
        t0 = time.time()
        result = engine.synthesize("OllaVoice is ready. Everything runs on this computer.")
        synth = time.time() - t0
        if result.clip.duration_s < 0.4:
            problems.append("TTS produced almost no audio")
        else:
            rtf = result.realtime_factor
            note = "" if rtf < 1.0 else yellow(
                "  (slower than realtime - expect a pause before it speaks)")
            print(green(f"  TTS {cfg.tts.engine}/{engine.voice_name}: "
                        f"{result.clip.duration_s:.1f}s audio in {synth:.1f}s "
                        f"({rtf:.2f}x realtime) {note}"))
        test_wav = _paths.audio_tmp_dir() / "setup-selftest.wav"
        result.clip.write_wav(test_wav)
    except Exception as exc:
        problems.append(f"TTS: {exc}")
        print(red(f"  TTS self-test failed: {exc}"))
        test_wav = None

    # --- STT
    if test_wav is not None:
        try:
            engine = stt_engines.create_engine(cfg.stt, ollama_host=cfg.ollama.host,
                                               app_language=cfg.language)
            t0 = time.time()
            tr = engine.transcribe(test_wav)
            print(green(f"  STT {tr.engine}/{tr.model}: {tr.text.strip()!r} in "
                        f"{time.time() - t0:.1f}s ({tr.realtime_factor:.2f}x realtime)"))
            if not tr.text.strip():
                problems.append("STT returned an empty transcript for a known-good clip")
            test_wav.unlink(missing_ok=True)
        except Exception as exc:
            problems.append(f"STT: {exc}")
            print(red(f"  STT self-test failed: {exc}"))

    # --- Ollama
    client = OllamaClient(cfg.ollama.host, cfg.ollama.api_key, timeout=120)
    if not client.is_alive():
        problems.append(f"Ollama is not reachable at {cfg.ollama.host}")
        print(red(f"  Ollama: not reachable at {cfg.ollama.host}"))
    elif not cfg.ollama.model:
        problems.append("No Ollama model configured")
    elif not client.has_model(cfg.ollama.model):
        problems.append(f"Ollama model '{cfg.ollama.model}' is not pulled")
        print(yellow(f"  Ollama: '{cfg.ollama.model}' not pulled yet"))
    else:
        try:
            t0 = time.time()
            answer = client.chat_once(
                [{"role": "user", "content": "Reply with exactly: ready"}],
                cfg.ollama.model, options={"num_predict": 8, "temperature": 0})
            print(green(f"  Ollama {cfg.ollama.model}: {answer.strip()!r} "
                        f"({time.time() - t0:.1f}s)"))
        except OllamaError as exc:
            problems.append(f"Ollama chat failed: {exc}")
            print(red(f"  Ollama chat failed: {exc}"))

    # --- audio devices
    try:
        from . import audioio

        backend = audioio.choose_backend("auto")
        devices = audioio.list_devices(backend)
        inputs = [d for d in devices if d.kind == "input"]
        outputs = [d for d in devices if d.kind == "output"]
        print(dim(f"  audio backend '{backend}': {len(inputs)} input, "
                  f"{len(outputs)} output device(s)"))
        if not inputs:
            problems.append("No microphone was detected")
        if not outputs:
            problems.append("No audio output device was detected")
    except audioio.AudioError as exc:
        problems.append(f"audio: {exc}")
        print(yellow("  audio: no usable backend yet. Install one with:"))
        print(cyan("    pip install PySide6      (recommended - mic + playback)"))
        print(cyan("    pip install sounddevice  (needs PortAudio)"))
    except Exception as exc:
        problems.append(f"audio: {exc}")
        print(yellow(f"  audio devices: {exc}"))

    return problems


# --------------------------------------------------------------------------- #
# Non-interactive install helpers used by scripts/setup.ps1
# --------------------------------------------------------------------------- #


def pip_install(packages: list[str], *, index_url: str | None = None) -> int:
    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"]
    if index_url:
        cmd += ["--index-url", index_url]
    cmd += packages
    print(cyan("  $ " + " ".join(cmd)))
    return subprocess.call(cmd)


def install_python_deps(*, stt: bool = True, tts: bool = True, gui: bool = True,
                        offline_wheels: Path | None = None,
                        index_url: str | None = None) -> int:
    """Install the extras. ``offline_wheels`` points at a directory of .whl files."""
    packages: list[str] = ["numpy", "soundfile"]
    if stt:
        packages.append("faster-whisper>=1.0")
    if tts:
        packages += ["piper-tts>=1.3", "kokoro-onnx>=0.9"]
    if gui:
        packages.append("PySide6>=6.6")

    if offline_wheels and Path(offline_wheels).exists():
        cmd = [sys.executable, "-m", "pip", "install", "--no-index",
               "--find-links", str(offline_wheels), *packages]
        print(cyan("  $ " + " ".join(cmd)))
        return subprocess.call(cmd)
    return pip_install(packages, index_url=index_url)


def check_python() -> list[str]:
    problems = []
    if sys.version_info < (3, 10):
        problems.append(f"Python {sys.version.split()[0]} is too old; 3.10+ is required")
    if sys.version_info >= (3, 14):
        problems.append(f"Python {sys.version.split()[0]} is very new; if a wheel is missing, "
                        "install Python 3.11 or 3.12")
    return problems


def check_ollama_binary() -> str | None:
    """Where is the ollama executable, if anywhere?"""
    found = shutil.which("ollama")
    if found:
        return found
    if sys.platform == "win32":
        for candidate in (
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
            Path("C:/Program Files/Ollama/ollama.exe"),
        ):
            if candidate.exists():
                return str(candidate)
    return None
