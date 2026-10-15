"""Voice catalogue helpers.

``piper.download_voices.list_voices()`` prints the whole catalogue to stdout as a
side effect (it doubles as a CLI), which is rude when you only wanted the list.
These wrappers capture that output and hand back clean data.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path


def piper_catalogue() -> list[str]:
    """Every Piper voice name known to the installed piper package.

    Reads the catalogue bundled inside the wheel, so it works with no network.
    """
    try:
        from piper.download_voices import list_voices  # noqa: PLC0415
    except ImportError:
        return []

    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            result = list_voices()
    except Exception:
        return _catalogue_from_stdout(buffer.getvalue())
    if isinstance(result, (list, tuple, set)):
        names = sorted({str(v) for v in result if v})
        if names:
            return names
    return _catalogue_from_stdout(buffer.getvalue())


def _catalogue_from_stdout(text: str) -> list[str]:
    names = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("INFO", "WARN", "ERROR")):
            continue
        if line.count("-") >= 2 and "_" in line.split("-", 1)[0]:
            names.append(line)
    return sorted(set(names))


def filter_by_language(names: list[str], language: str) -> list[str]:
    """``de``, ``de_DE`` and ``German``-style codes all select the same set."""
    if not language:
        return list(names)
    lang = language.split("-")[0].split("_")[0].lower()
    return [n for n in names if n.split("-", 1)[0].split("_")[0].lower() == lang]


def installed_piper_voices(directory: Path | str | None = None) -> list[str]:
    """Voices already downloaded on this machine."""
    from . import paths as _paths  # noqa: PLC0415

    folder = Path(directory) if directory else _paths.piper_dir()
    if not folder.exists():
        return []
    return sorted(p.stem for p in folder.glob("*.onnx"))


def is_valid_piper_voice_name(voice: str) -> bool:
    """Shape check for ``lang_REGION-name-quality`` (quality may contain a dash)."""
    parts = (voice or "").split("-")
    if len(parts) < 3:
        return False
    locale = parts[0]
    if len(locale) != 5 or locale[2] != "_" or not locale[:2].isalpha():
        return False
    return voice.endswith(("-x_low", "-low", "-medium", "-high"))
