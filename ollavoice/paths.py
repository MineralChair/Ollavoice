"""Filesystem locations used by OllaVoice.

Everything OllaVoice stores is on the local machine, inside standard per-user
application directories. Nothing is uploaded anywhere.

Windows layout (default)::

    %APPDATA%\\OllaVoice\\config.json        settings
    %APPDATA%\\OllaVoice\\logs\\             logs
    %APPDATA%\\OllaVoice\\transcripts\\      chat history (opt-in)
    %LOCALAPPDATA%\\OllaVoice\\models\\      whisper / piper / kokoro weights
    %LOCALAPPDATA%\\OllaVoice\\audio\\       scratch recordings

Set ``OLLAVOICE_HOME`` to move the whole lot to a single directory (handy for a
portable USB install or a non-system drive).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "OllaVoice"


def _is_windows() -> bool:
    return sys.platform.startswith("win")


def home_root() -> Path:
    """Root directory for config/state (roaming-style)."""
    override = os.environ.get("OLLAVOICE_HOME")
    if override:
        return Path(override).expanduser()
    if _is_windows():
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / APP_NAME
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / APP_NAME.lower()


def data_root() -> Path:
    """Root directory for large, re-downloadable model files (local-style)."""
    override = os.environ.get("OLLAVOICE_HOME")
    if override:
        return Path(override).expanduser() / "models"
    if _is_windows():
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / APP_NAME / "models"
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / APP_NAME.lower() / "models"


def config_file() -> Path:
    return home_root() / "config.json"


def models_dir() -> Path:
    return data_root()


def whisper_dir() -> Path:
    return data_root() / "whisper"


def piper_dir() -> Path:
    return data_root() / "piper"


def kokoro_dir() -> Path:
    return data_root() / "kokoro"


def whispercpp_dir() -> Path:
    """Location for an optional user-supplied whisper.cpp build (ggml models + binary)."""
    return data_root() / "whisper.cpp"


def audio_tmp_dir() -> Path:
    return home_root() / "audio"


def logs_dir() -> Path:
    return home_root() / "logs"


def transcripts_dir() -> Path:
    return home_root() / "transcripts"


def vendor_dir() -> Path:
    """Offline bundle location: wheels + model files for air-gapped installs."""
    return home_root() / "vendor"


def ensure_dirs() -> None:
    for p in (home_root(), data_root(), whisper_dir(), piper_dir(), kokoro_dir(),
              audio_tmp_dir(), logs_dir(), transcripts_dir()):
        try:
            p.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass


def describe() -> dict[str, str]:
    ensure_dirs()
    return {
        "config": str(config_file()),
        "models": str(data_root()),
        "whisper": str(whisper_dir()),
        "piper": str(piper_dir()),
        "kokoro": str(kokoro_dir()),
        "audio_tmp": str(audio_tmp_dir()),
        "logs": str(logs_dir()),
        "transcripts": str(transcripts_dir()),
    }


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    for k, v in describe().items():
        print(f"{k:12s} {v}")
