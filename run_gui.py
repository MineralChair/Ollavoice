"""PyInstaller entry point: launch the OllaVoice desktop GUI.

Kept at the project root (rather than inside the package) so the frozen app has
an obvious, stable script to bundle. ``console=False`` in the spec means there is
no terminal, so anything that would normally be printed is written to the log
file under %APPDATA%\\OllaVoice\\logs.
"""

from __future__ import annotations

import sys
import traceback


def main() -> int:
    try:
        from ollavoice.gui.app import main as gui_main

        return gui_main(argv=sys.argv[:1])
    except SystemExit as exc:                     # a clean shutdown
        return int(exc.code or 0)
    except Exception:
        # No console in a windowed build: put the traceback somewhere findable
        # and tell the user where to look.
        message = traceback.format_exc()
        log_path = ""
        try:
            from ollavoice import paths

            paths.ensure_dirs()
            log_path = str(paths.logs_dir() / "crash.log")
            with open(log_path, "w", encoding="utf-8") as fh:
                fh.write(message)
        except Exception:
            pass
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox

            app = QApplication.instance() or QApplication(sys.argv[:1])  # noqa: F841
            box = QMessageBox()
            box.setIcon(QMessageBox.Icon.Critical)
            box.setWindowTitle("OllaVoice failed to start")
            box.setText("OllaVoice could not start.")
            box.setInformativeText(
                "If this is a packaged build, the most likely causes are a missing "
                "Visual C++ runtime or antivirus quarantine of a bundled DLL.")
            box.setDetailedText(message + (f"\n\nLog written to {log_path}" if log_path else ""))
            box.exec()
        except Exception:
            sys.stderr.write(message)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
