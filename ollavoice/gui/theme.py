"""Icons and stylesheets for the GUI.

Icons are drawn as inline SVG data URIs so the app has no binary asset
dependencies - which also means a PyInstaller build cannot forget to bundle
them.
"""

from __future__ import annotations

import base64

DARK_QSS = """
* { font-family: "Segoe UI", "Inter", system-ui, sans-serif; }
QWidget {
    background-color: #16181d;
    color: #e6e8ee;
    font-size: %(font_size)spx;
}
QMainWindow, QDialog { background-color: #16181d; }
QToolBar {
    background-color: #1b1e25;
    border-bottom: 1px solid #262a33;
    padding: 5px 8px;
    spacing: 6px;
}
QToolBar QToolButton {
    background: transparent; border: none; padding: 4px 6px; border-radius: 6px;
    color: #cfd4e0;
}
QToolBar QToolButton:hover { background: #262b36; }
QStatusBar { background: #12141a; color: #8b93a7; border-top: 1px solid #262a33; }
QStatusBar::item { border: none; }

QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {
    background-color: #1f232b;
    border: 1px solid #303644;
    border-radius: 7px;
    padding: 5px 8px;
    color: #e6e8ee;
    selection-background-color: #3d6df0;
}
QComboBox:hover, QLineEdit:focus, QPlainTextEdit:focus { border-color: #4a7dff; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background-color: #1f232b; border: 1px solid #303644;
    selection-background-color: #33406b; color: #e6e8ee;
}
QLineEdit#userInput {
    background-color: #1f232b; padding: 9px 12px; font-size: %(font_size)spx;
}
QLineEdit#userInput:disabled { color: #6d7488; }

QPushButton {
    background-color: #262b36; border: 1px solid #39404f; border-radius: 8px;
    padding: 7px 14px; color: #e6e8ee;
}
QPushButton:hover { background-color: #2f3644; border-color: #4a5468; }
QPushButton:pressed { background-color: #1c2029; }
QPushButton:disabled { color: #5f6675; background-color: #1e222a; border-color: #2a2f3a; }

QPushButton#sendButton { background-color: #2d4bd6; border-color: #3a5cf0; color: white; }
QPushButton#sendButton:hover { background-color: #3859e8; }
QPushButton#sendButton:disabled { background-color: #232a3d; color: #6b7488; border-color: #2b3346; }

QPushButton#micButton {
    background-color: #22303f; border: 1px solid #33506e; border-radius: 22px;
    padding: 8px 20px; font-weight: 600; color: #cfe4ff;
}
QPushButton#micButton:hover { background-color: #294057; }
QPushButton#micButton:checked, QPushButton#micButton[recording="true"] {
    background-color: #a32635; border-color: #d63b4d; color: #ffe9ec;
}
QPushButton#stopButton { background-color: #3a2530; border-color: #5c3546; color: #ffd9e2; }
QPushButton#stopButton:hover { background-color: #4a2d3b; }

QCheckBox { spacing: 7px; color: #b9c0d0; }
QCheckBox::indicator { width: 15px; height: 15px; border-radius: 4px;
                       border: 1px solid #4a5468; background: #1f232b; }
QCheckBox::indicator:checked { background: #3d6df0; border-color: #3d6df0; }

QProgressBar#levelMeter {
    background-color: #1f232b; border: 1px solid #303644; border-radius: 6px;
    text-align: center; color: #9aa3b8; height: 20px; font-size: 10px;
}
QProgressBar#levelMeter::chunk { background-color: #35c46a; border-radius: 5px; }
QProgressBar#levelMeter[hot="true"]::chunk { background-color: #e0a53a; }
QProgressBar#levelMeter[clip="true"]::chunk { background-color: #e04a4a; }

QProgressBar#busyBar::chunk { background-color: #3d6df0; }

QTabWidget::pane { border: 1px solid #262a33; border-radius: 8px; top: -1px; }
QTabBar::tab {
    background: transparent; color: #8b93a7; padding: 8px 16px;
    border: 1px solid transparent; border-top-left-radius: 7px; border-top-right-radius: 7px;
}
QTabBar::tab:selected { color: #e6e8ee; background: #1f232b; border-color: #262a33; }
QTabBar::tab:hover { color: #cfd4e0; }

QGroupBox {
    border: 1px solid #262a33; border-radius: 8px; margin-top: 14px;
    padding-top: 10px; font-weight: 600; color: #cfd4e0;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }

QSlider::groove:horizontal { height: 4px; background: #303644; border-radius: 2px; }
QSlider::handle:horizontal {
    width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; background: #7aa2ff;
}
QScrollBar:vertical { background: #16181d; width: 11px; margin: 0; }
QScrollBar::handle:vertical { background: #303644; border-radius: 5px; min-height: 28px; }
QScrollBar::handle:vertical:hover { background: #3d4658; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
QScrollBar:horizontal { background: #16181d; height: 11px; }
QScrollBar::handle:horizontal { background: #303644; border-radius: 5px; min-width: 28px; }

QMenu { background-color: #1f232b; border: 1px solid #303644; padding: 5px; }
QMenu::item { padding: 6px 22px; border-radius: 5px; color: #e6e8ee; }
QMenu::item:selected { background-color: #33406b; }
QMenu::separator { height: 1px; background: #303644; margin: 5px 8px; }

QToolTip { background-color: #262b36; color: #e6e8ee; border: 1px solid #39404f; padding: 5px; }
QLabel#hintLabel { color: #7e879c; font-size: 11px; }
QLabel#errorLabel { color: #ff9aa6; }
QLabel#phaseLabel { color: #9fb6ff; font-weight: 600; }
"""

LIGHT_QSS = """
* { font-family: "Segoe UI", "Inter", system-ui, sans-serif; }
QWidget { background-color: #f6f7f9; color: #1c2027; font-size: %(font_size)spx; }
QToolBar { background-color: #ffffff; border-bottom: 1px solid #dfe3ea; padding: 5px 8px; spacing: 6px; }
QStatusBar { background: #eceef2; color: #61697a; border-top: 1px solid #dfe3ea; }
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {
    background-color: #ffffff; border: 1px solid #cfd5e0; border-radius: 7px;
    padding: 5px 8px; color: #1c2027; selection-background-color: #3d6df0;
}
QLineEdit#userInput { padding: 9px 12px; }
QPushButton { background-color: #eceff5; border: 1px solid #cfd5e0; border-radius: 8px;
              padding: 7px 14px; color: #1c2027; }
QPushButton:hover { background-color: #e2e7f0; }
QPushButton#sendButton { background-color: #2d4bd6; border-color: #2d4bd6; color: white; }
QPushButton#micButton { background-color: #dcecfb; border: 1px solid #a9c9e8;
                        border-radius: 22px; padding: 8px 20px; font-weight: 600; color: #1d4e7d; }
QPushButton#micButton:checked { background-color: #d63b4d; border-color: #d63b4d; color: white; }
QPushButton#stopButton { background-color: #fbe3e7; border-color: #e8b3bd; color: #8c2434; }
QProgressBar#levelMeter { background-color: #eceff5; border: 1px solid #cfd5e0;
                          border-radius: 6px; text-align: center; height: 20px; color: #61697a; }
QProgressBar#levelMeter::chunk { background-color: #35c46a; border-radius: 5px; }
QGroupBox { border: 1px solid #dfe3ea; border-radius: 8px; margin-top: 14px; padding-top: 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
QTabBar::tab { background: transparent; color: #61697a; padding: 8px 16px; }
QTabBar::tab:selected { color: #1c2027; background: #ffffff; border: 1px solid #dfe3ea; }
QLabel#hintLabel { color: #7b8496; font-size: 11px; }
QLabel#errorLabel { color: #b3283c; }
QLabel#phaseLabel { color: #2d4bd6; font-weight: 600; }
"""


def _svg_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.strip().encode()).decode()


def _wrap(body: str, color: str = "#e6e8ee", size: int = 24) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
            f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.9" '
            f'stroke-linecap="round" stroke-linejoin="round">{body}</svg>')


_MIC_BODY = ('<rect x="9" y="2.5" width="6" height="11" rx="3"/>'
             '<path d="M5.5 11a6.5 6.5 0 0 0 13 0"/><path d="M12 17.5V21"/>')
_SPEAKER_BODY = ('<path d="M4 9.5h3.2L12 5.5v13L7.2 14.5H4z"/>'
                 '<path d="M15.8 9.2a4 4 0 0 1 0 5.6"/><path d="M18.3 6.7a7.5 7.5 0 0 1 0 10.6"/>')
_STOP_BODY = '<rect x="6" y="6" width="12" height="12" rx="2"/>'
_GEAR_BODY = ('<circle cx="12" cy="12" r="3.2"/>'
              '<path d="M19.4 15a1.7 1.7 0 0 0 .34 1.87l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06'
              'a1.7 1.7 0 0 0-1.87-.34 1.7 1.7 0 0 0-1 1.56V21a2 2 0 1 1-4 0v-.09'
              'A1.7 1.7 0 0 0 8.9 19.3a1.7 1.7 0 0 0-1.87.34l-.06.06a2 2 0 1 1-2.83-2.83'
              'l.06-.06A1.7 1.7 0 0 0 4.7 15.2a1.7 1.7 0 0 0-1.56-1H3a2 2 0 1 1 0-4h.09'
              'A1.7 1.7 0 0 0 4.7 9.1a1.7 1.7 0 0 0-.34-1.87l-.06-.06a2 2 0 1 1 2.83-2.83'
              'l.06.06A1.7 1.7 0 0 0 9.1 4.7 1.7 1.7 0 0 0 10 3.14V3a2 2 0 1 1 4 0v.09'
              'a1.7 1.7 0 0 0 1 1.56 1.7 1.7 0 0 0 1.87-.34l.06-.06a2 2 0 1 1 2.83 2.83'
              'l-.06.06A1.7 1.7 0 0 0 19.3 9.1a1.7 1.7 0 0 0 1.56 1H21a2 2 0 1 1 0 4h-.09'
              'a1.7 1.7 0 0 0-1.51 1z"/>')
_SEND_BODY = '<path d="M4 12h15"/><path d="M13 6l6 6-6 6"/>'
_APP_BODY = ('<path d="M12 3a4 4 0 0 1 4 4v5a4 4 0 0 1-8 0V7a4 4 0 0 1 4-4z"/>'
             '<path d="M5 11a7 7 0 0 0 14 0"/><path d="M12 18v3"/><circle cx="12" cy="8" r="1.2"/>')
_TRASH_BODY = '<path d="M4 7h16"/><path d="M9 7V4.8h6V7"/><path d="M6 7l1 13h10l1-13"/>'


def icons(color: str = "#e6e8ee") -> dict[str, str]:
    """Data-URI icons, tinted for the active theme."""
    return {
        "mic": _svg_uri(_wrap(_MIC_BODY, color)),
        "speaker": _svg_uri(_wrap(_SPEAKER_BODY, color)),
        "stop": _svg_uri(_wrap(_STOP_BODY, "#ffd9e2")),
        "gear": _svg_uri(_wrap(_GEAR_BODY, color)),
        "send": _svg_uri(_wrap(_SEND_BODY, "#ffffff")),
        "app": _svg_uri(_wrap(_APP_BODY, "#7aa2ff")),
        "trash": _svg_uri(_wrap(_TRASH_BODY, color)),
    }


def app_icon(color: str = "#7aa2ff"):
    """A QIcon for the window/taskbar/tray."""
    from PySide6.QtGui import QIcon, QPixmap
    from PySide6.QtCore import QByteArray

    pixmap = QPixmap()
    pixmap.loadFromData(QByteArray.fromBase64(
        base64.b64encode(_wrap(_APP_BODY, color).encode())))
    return QIcon(pixmap)


def stylesheet(theme: str = "dark", font_size: int = 12) -> str:
    template = DARK_QSS if theme != "light" else LIGHT_QSS
    return template % {"font_size": int(font_size)}
