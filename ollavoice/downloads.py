"""Model/binary downloads with progress, resume and checksum verification.

Everything here runs **only during setup**. Once the files are on disk OllaVoice
never touches the network again - you can pull the cable and it keeps working.

Supported sources:

* Hugging Face (Whisper CT2 models, Piper voices, Kokoro checkpoints)
* GitHub release assets (kokoro-onnx model files)
* A local ``vendor/`` directory, for air-gapped installs
  (``ollavoice setup --offline`` uses it and refuses to reach the network)

Set ``OLLAVOICE_OFFLINE=1`` to make every download call fail fast with a clear
message instead of hanging on a blocked network.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

CHUNK = 256 * 1024
USER_AGENT = "OllaVoice/1.0 (+local; offline-first)"

ProgressFn = Callable[[float, float, str], None]  # (bytes_done, bytes_total, filename)


class OfflineError(RuntimeError):
    """Raised when a download is attempted while offline mode is on."""


def offline_mode() -> bool:
    return os.environ.get("OLLAVOICE_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}


def _quiet_hook(progress: ProgressFn | None, url: str) -> Callable[[int, int], None]:
    """Throttle progress callbacks to ~4/s and normalise the numbers."""
    if progress is None:
        return lambda *_: None
    name = url.rsplit("/", 1)[-1].split("?")[0] or url
    state = {"last": 0.0}

    def hook(written: int, total: int) -> None:
        written = int(written or 0)
        total = int(total or 0)
        now = time.time()
        if now - state["last"] < 0.25 and not (total and written >= total):
            return
        state["last"] = now
        progress(min(written, total) if total else written, total, name)

    return hook


@dataclass
class DownloadResult:
    path: Path
    bytes: int
    seconds: float
    from_cache: bool
    source: str


def download(url: str, dest: Path | str, *, expected_sha256: str | None = None,
             progress: ProgressFn | None = None, timeout: int = 60,
             resume: bool = True, mirrors: list[str] | None = None) -> DownloadResult:
    """Download ``url`` to ``dest`` atomically, with optional resume + checksum.

    ``mirrors`` are tried in order if the primary URL fails (useful for users
    behind a firewall with a Hugging Face mirror, see ``HF_ENDPOINT``).
    """
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        if not expected_sha256 or sha256(dest) == expected_sha256.lower():
            return DownloadResult(dest, dest.stat().st_size, 0.0, True, "cache")
        dest.unlink(missing_ok=True)

    if offline_mode():
        raise OfflineError(
            f"OLLAVOICE_OFFLINE is set but {dest.name} is missing.\n"
            f"Copy it manually to: {dest}"
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    urls = [url] + list(mirrors or [])
    last_error: Exception | None = None

    for attempt_url in urls:
        partial = dest.with_suffix(dest.suffix + ".part")
        start_byte = partial.stat().st_size if (resume and partial.exists()) else 0
        request = urllib.request.Request(attempt_url, headers={"User-Agent": USER_AGENT})
        if start_byte:
            request.add_header("Range", f"bytes={start_byte}-")
        t0 = time.time()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                if resp.status == 200 and start_byte:
                    # Server ignored Range; start over.
                    start_byte = 0
                total = int(resp.headers.get("Content-Length") or 0) + start_byte
                mode = "ab" if start_byte and resp.status == 206 else "wb"
                hook = _quiet_hook(progress, attempt_url)
                written = start_byte if mode == "ab" else 0
                with open(partial, mode) as fh:
                    while True:
                        block = resp.read(CHUNK)
                        if not block:
                            break
                        fh.write(block)
                        written += len(block)
                        hook(written, total)
            if expected_sha256:
                actual = sha256(partial)
                if actual != expected_sha256.lower():
                    raise ValueError(
                        f"checksum mismatch for {partial.name}: "
                        f"expected {expected_sha256[:12]}..., got {actual[:12]}..."
                    )
            os.replace(partial, dest)
            return DownloadResult(dest, dest.stat().st_size, time.time() - t0,
                                  False, attempt_url)
        except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
            last_error = exc
            continue

    raise RuntimeError(f"download failed for {url}: {last_error}")


def sha256(path: Path | str, block_size: int = CHUNK) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def hf_url(repo: str, filename: str, *, revision: str = "main") -> str:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    return f"{endpoint}/{repo}/resolve/{revision}/{filename}?download=true"


def hf_mirror_urls(repo: str, filename: str, *, revision: str = "main") -> list[str]:
    """Fallbacks for users whose network blocks huggingface.co directly."""
    endpoint = os.environ.get("HF_ENDPOINT", "").rstrip("/")
    out = []
    if endpoint and "huggingface.co" not in endpoint:
        out.append(f"{endpoint}/{repo}/resolve/{revision}/{filename}?download=true")
    mirror = os.environ.get("OLLAVOICE_HF_MIRROR", "").rstrip("/")
    if mirror:
        out.append(f"{mirror}/{repo}/resolve/{revision}/{filename}?download=true")
    return out


# --------------------------------------------------------------------------- #
# Named artifacts
# --------------------------------------------------------------------------- #

KOKORO_RELEASE_BASE = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1"
)
KOKORO_FILES = {
    "kokoro-v1.0.onnx": "kokoro-v1.0.onnx",
    "kokoro-v1.0.int8.onnx": "kokoro-v1.0.int8.onnx",
    "kokoro-v1.0.fp16.onnx": "kokoro-v1.0.fp16.onnx",
    "voices-v1.0.bin": "voices-v1.0.bin",
}
KOKORO_VOICES_FOR = {
    "kokoro-v1.0.onnx": "voices-v1.0.bin",
    "kokoro-v1.0.int8.onnx": "voices-v1.0.bin",
    "kokoro-v1.0.fp16.onnx": "voices-v1.0.bin",
}


def ensure_kokoro(model_name: str, dest_dir: Path, *, progress: ProgressFn | None = None) -> tuple[Path, Path]:
    """Download a Kokoro ONNX checkpoint + its voices file. Returns both paths."""
    from . import paths

    dest_dir = Path(dest_dir) if dest_dir else paths.kokoro_dir()
    if model_name not in KOKORO_FILES:
        raise ValueError(
            f"unknown Kokoro model {model_name!r}; pick one of {', '.join(sorted(KOKORO_FILES))}"
        )
    voices_name = KOKORO_VOICES_FOR[model_name]
    model_path = dest_dir / model_name
    voices_path = dest_dir / voices_name
    download(f"{KOKORO_RELEASE_BASE}/{model_name}", model_path, progress=progress)
    download(f"{KOKORO_RELEASE_BASE}/{voices_name}", voices_path, progress=progress)
    return model_path, voices_path


def ensure_piper_voice(voice: str, dest_dir: Path, *, progress: ProgressFn | None = None) -> Path:
    """Download a Piper voice via the official downloader, falling back to a
    direct HF fetch. Returns the ``.onnx`` path.

    Piper's downloader writes ``<voice>.onnx`` + ``<voice>.onnx.json`` into
    ``dest_dir`` and is a no-op when the files already exist.
    """
    from . import paths

    from .voices import is_valid_piper_voice_name  # noqa: PLC0415

    if not is_valid_piper_voice_name(voice):
        raise ValueError(
            "Piper voice names look like 'en_US-lessac-medium' "
            "(language_REGION, name, quality); got " + repr(voice)
        )
    parts = voice.split("-")

    dest_dir = Path(dest_dir) if dest_dir else paths.piper_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    onnx = dest_dir / f"{voice}.onnx"
    if onnx.exists() and (dest_dir / f"{voice}.onnx.json").exists():
        return onnx

    if offline_mode():
        # Air-gapped: only the vendor bundle can satisfy this.
        vendored = copy_from_vendor(f"{voice}.onnx", dest_dir)
        copy_from_vendor(f"{voice}.onnx.json", dest_dir)
        if vendored:
            return onnx
        raise OfflineError(
            f"OLLAVOICE_OFFLINE is set and the Piper voice '{voice}' is not in the "
            f"vendor bundle.\nCopy '{voice}.onnx' and '{voice}.onnx.json' into: {dest_dir}"
        )

    if not offline_mode():
        try:
            from piper.download_voices import download_voice  # noqa: PLC0415

            download_voice(voice, download_dir=dest_dir)
            if onnx.exists():
                return onnx
        except Exception as exc:  # network hiccup, unknown voice name, ...
            sys.stderr.write(f"[ollavoice] piper downloader failed ({exc}); trying direct URL\n")

    # Direct HF fetch as a fallback (also the path used in offline-bundle mode).
    lang_code = parts[0]
    if voice.endswith("-x_low"):
        quality, voice_name = "x_low", "-".join(parts[1:-1])
    else:
        quality, voice_name = parts[-1], "-".join(parts[1:-1])
    lang_family = lang_code.split("_")[0]
    filename = f"{lang_family}/{lang_code}/{voice_name}/{quality}/{lang_code}-{voice_name}-{quality}"
    for ext in (".onnx", ".onnx.json"):
        download(
            hf_url("rhasspy/piper-voices", filename + ext),
            dest_dir / f"{voice}{ext}",
            progress=progress,
            mirrors=hf_mirror_urls("rhasspy/piper-voices", filename + ext),
        )
    return onnx


def copy_from_vendor(name: str, dest_dir: Path) -> Path | None:
    """Air-gapped installs: look for ``name`` inside the vendor directory."""
    from . import paths

    vendor = paths.vendor_dir()
    for candidate in (vendor / "models" / name, vendor / name):
        if candidate.exists():
            dest_dir.mkdir(parents=True, exist_ok=True)
            target = dest_dir / name
            if not target.exists():
                shutil.copy2(candidate, target)
            return target
    return None


def dir_size_mb(path: Path | str) -> float:
    path = Path(path)
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def make_offline_bundle(dest: Path | str, *, wheels_dir: Path | str | None = None) -> Path:
    """Copy every downloaded model into a portable bundle for another machine.

    The result can be zipped, copied to an offline PC and unpacked into
    ``%LOCALAPPDATA%\\OllaVoice\\models`` (or pointed at with ``OLLAVOICE_HOME``).
    """
    from . import paths

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for sub in ("whisper", "piper", "kokoro"):
        src = paths.data_root() / sub
        if src.exists():
            target = dest / sub
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(src, target)
            copied.append(f"{sub} ({dir_size_mb(target):.0f} MB)")
    if wheels_dir and Path(wheels_dir).exists():
        shutil.copytree(wheels_dir, dest / "wheels", dirs_exist_ok=True)
        copied.append("wheels")
    (dest / "BUNDLE-INFO.txt").write_text(
        "OllaVoice offline model bundle\n"
        f"created: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"python: {sys.version.split()[0]} on {platform_tag()}\n\n"
        "contents:\n  " + "\n  ".join(copied) +
        "\n\nTo use: copy this folder's subdirectories into\n"
        f"  {paths.data_root()}\n",
        encoding="utf-8",
    )
    return dest


def platform_tag() -> str:
    if sys.platform == "win32":
        return f"win-{os.environ.get('PROCESSOR_ARCHITECTURE', 'amd64').lower()}"
    if sys.platform == "darwin":
        return "macos-" + ("arm64" if os.uname().machine == "arm64" else "x86_64")
    return f"linux-{os.uname().machine}"


def temp_wav_path(prefix: str = "ollavoice-") -> Path:
    from . import paths

    paths.audio_tmp_dir().mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=prefix, suffix=".wav", dir=str(paths.audio_tmp_dir()))
    os.close(fd)
    return Path(name)
