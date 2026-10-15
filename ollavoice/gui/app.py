"""Application entry point for the OllaVoice desktop GUI.

``ollavoice gui`` and the ``ollavoice-gui`` console script both land here, as
does a PyInstaller build of the Windows executable.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from .. import __version__, paths as _paths
from ..config import Config, load as load_config


def _setup_logging(level: str = "INFO") -> Path:
    _paths.ensure_dirs()
    log_path = _paths.logs_dir() / f"gui-{time.strftime('%Y%m%d')}.log"
    try:
        import logging

        logging.basicConfig(
            filename=str(log_path), level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
        for noisy in ("faster_whisper", "piper", "onnxruntime", "httpx", "matplotlib"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    except OSError:
        pass
    return log_path


def _single_instance_guard() -> bool:
    """Best-effort 'already running?' check using a lock file with a PID."""
    lock = _paths.home_root() / "gui.lock"
    try:
        if lock.exists():
            raw = lock.read_text(encoding="utf-8").strip()
            pid = int(raw.split(":")[0]) if raw else 0
            if pid and pid != os.getpid() and _pid_alive(pid):
                return False
        lock.write_text(f"{os.getpid()}:{time.time()}", encoding="utf-8")
    except (OSError, ValueError):
        return True
    return True


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        if sys.platform == "win32":
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return False
            code = ctypes.c_ulong()
            ok = ctypes.windll.kernel32.GetExitCodeProcess(  # type: ignore[attr-defined]
                handle, ctypes.byref(code))
            ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
            return bool(ok) and code.value == 259  # STILL_ACTIVE
        os.kill(pid, 0)
        return True
    except (OSError, AttributeError):
        return False


def main(cfg: Config | None = None, *, start_minimized: bool = False,
         argv: list[str] | None = None) -> int:
    """Launch the GUI. Returns a process exit code."""
    argv = list(sys.argv if argv is None else argv)
    _paths.ensure_dirs()
    log_path = _setup_logging(os.environ.get("OLLAVOICE_LOG_LEVEL", "INFO"))

    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication, QMessageBox
    except ImportError as exc:  # pragma: no cover - environment problem
        sys.stderr.write(
            "OllaVoice's desktop GUI needs PySide6 (Qt 6).\n"
            "  pip install PySide6\n"
            f"(import failed with: {exc})\n\n"
            "You can still use the terminal interface:\n"
            "  ollavoice chat\n"
        )
        return 2
    except OSError as exc:  # pragma: no cover - missing Qt system libraries
        sys.stderr.write(
            "PySide6 is installed but its Qt libraries could not be loaded.\n"
            f"  {exc}\n\n"
            "On Linux this usually means missing system packages, e.g.\n"
            "  sudo apt install libglib2.0-0 libgl1 libegl1 libxkbcommon0 "
            "libdbus-1-3 libfontconfig1\n"
            "On Windows, reinstalling PySide6 (pip install --force-reinstall PySide6) "
            "usually fixes it.\n\n"
            "You can still use the terminal interface:\n"
            "  ollavoice chat\n"
        )
        return 2

    # Qt on Windows: keep audio in the main process, avoid the ANGLE/D3D quirks.
    os.environ.setdefault("QT_MEDIA_BACKEND", "windowsmediafoundation")
    os.environ.setdefault("QT_AUDIO_RESUMER", "1")
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(  # type: ignore[attr-defined]
                "OllaVoice.Desktop.1")
        except Exception:
            pass

    if not _single_instance_guard():
        app = QApplication(argv)
        QMessageBox.information(
            None, "OllaVoice",
            "OllaVoice is already running.\nLook for the microphone icon in the system tray.")
        return 0

    app = QApplication(argv)
    app.setApplicationName("OllaVoice")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("OllaVoice")
    app.setQuitOnLastWindowClosed(False)   # the tray keeps it alive

    cfg = cfg or load_config()
    if start_minimized or cfg.gui.start_minimized:
        pass

    from .main_window import MainWindow

    window = MainWindow(cfg)
    if start_minimized or cfg.gui.start_minimized:
        window.hide()
        window.tray.showMessage("OllaVoice", "Running in the background. "
                                f"Press {cfg.hotkeys.push_to_talk or 'the tray icon'} to talk.",
                                4000)
    else:
        window.show()

    # A tiny first-run nudge when nothing has been configured yet.
    if not cfg.ollama.model:
        QTimer.singleShot(800, window.show_paths)

    try:
        return int(app.exec())
    except Exception as exc:  # pragma: no cover
        sys.stderr.write(f"OllaVoice crashed: {exc}\nLog: {log_path}\n")
        raise


def run() -> None:  # pragma: no cover - console-script shim
    raise SystemExit(main())


if __name__ == "__main__":  # pragma: no cover
    run()
