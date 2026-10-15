"""Global (system-wide) hotkeys.

On Windows this uses ``RegisterHotKey`` through Qt's native event filter, so a
hotkey works even when OllaVoice is not the focused window - which is the whole
point of a push-to-talk key. No third-party keyboard-hook package is needed, and
nothing is sent off the machine.

On other platforms (and if registration fails) we fall back to an in-app
``QShortcut``, which only fires while the window has focus. The GUI reports which
mode is active in the status bar.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312

# Virtual-key codes for the keys we realistically need in a hotkey string.
VK = {
    "SPACE": 0x20, "ENTER": 0x0D, "RETURN": 0x0D, "TAB": 0x09, "ESC": 0x1B,
    "BACKSPACE": 0x08, "DELETE": 0x2E, "INSERT": 0x2D, "HOME": 0x24, "END": 0x23,
    "PAGEUP": 0x21, "PAGEDOWN": 0x22,
    "F1": 0x70, "F2": 0x71, "F3": 0x72, "F4": 0x73, "F5": 0x74, "F6": 0x75,
    "F7": 0x76, "F8": 0x77, "F9": 0x78, "F10": 0x79, "F11": 0x7A, "F12": 0x7B,
    "F13": 0x7C, "F14": 0x7D, "F15": 0x7E, "F16": 0x7F,
}


@dataclass
class HotkeySpec:
    name: str
    combo: str
    modifiers: int = 0
    vk: int = 0
    id: int = 0

    @property
    def valid(self) -> bool:
        return bool(self.vk)


def parse_combo(combo: str) -> HotkeySpec:
    """Parse ``"Ctrl+Alt+Space"`` / ``"F9"`` / ``"Ctrl+Shift+M"`` into a spec."""
    spec = HotkeySpec(name=combo, combo=combo or "")
    if not combo:
        return spec
    mods = MOD_NOREPEAT
    parts = [p.strip() for p in combo.replace("-", "+").split("+") if p.strip()]
    if not parts:
        return spec
    key = parts[-1].upper()
    for part in parts[:-1]:
        p = part.upper()
        if p in ("CTRL", "CONTROL"):
            mods |= MOD_CONTROL
        elif p == "ALT":
            mods |= MOD_ALT
        elif p == "SHIFT":
            mods |= MOD_SHIFT
        elif p in ("WIN", "META", "SUPER", "CMD"):
            mods |= MOD_WIN
    if len(key) == 1:
        spec.vk = ord(key)
    elif key in VK:
        spec.vk = VK[key]
    elif len(key) > 1 and key.isdigit():
        spec.vk = int(key)
    spec.modifiers = mods
    return spec


class GlobalHotkeyManager:
    """Registers Win32 hotkeys against the Qt application's native window."""

    def __init__(self, window, callbacks: dict[str, object]) -> None:
        self.window = window
        self.callbacks = callbacks
        self.specs: dict[int, HotkeySpec] = {}
        self._id_to_name: dict[int, str] = {}
        self._installed = False
        self._native_ok = False
        self._fallback_shortcuts: list[object] = []

    # ------------------------------------------------------------------ #
    def register(self, combos: dict[str, str]) -> dict[str, bool]:
        """Register ``{name: "Ctrl+Alt+Space"}``. Returns which ones succeeded."""
        self.unregister()
        results: dict[str, bool] = {}
        for index, (name, combo) in enumerate(combos.items(), start=1):
            if not combo:
                results[name] = False
                continue
            spec = parse_combo(combo)
            spec.name = name
            spec.id = 0x011A + index
            if not spec.valid:
                results[name] = False
                continue
            if self._register_win32(spec):
                self.specs[spec.id] = spec
                self._id_to_name[spec.id] = name
                results[name] = True
            else:
                results[name] = self._register_fallback(name, combo)
        if any(results.values()):
            self._install_filter()
        return results

    def _install_filter(self) -> None:
        if self._installed:
            return
        try:
            from PySide6.QtCore import QAbstractNativeEventFilter

            manager = self

            class _Filter(QAbstractNativeEventFilter):
                def nativeEventFilter(self, event_type, message):  # noqa: N802
                    return manager._native_event(event_type, message)

            self._filter = _Filter()
            from PySide6.QtGui import QGuiApplication

            QGuiApplication.instance().installNativeEventFilter(self._filter)
            self._installed = True
        except Exception:
            self._installed = False

    def _register_win32(self, spec: HotkeySpec) -> bool:
        if sys.platform != "win32":
            return False
        try:
            import ctypes
            from ctypes import wintypes

            hwnd = int(self.window.winId())
            user32 = ctypes.windll.user32  # type: ignore[attr-defined]
            ok = user32.RegisterHotKey(wintypes.HWND(hwnd), spec.id,
                                       wintypes.UINT(spec.modifiers),
                                       wintypes.UINT(spec.vk))
            self._native_ok = bool(ok)
            return bool(ok)
        except Exception:
            self._native_ok = False
            return False

    def _register_fallback(self, name: str, combo: str) -> bool:
        try:
            from PySide6.QtGui import QKeySequence, QShortcut
            from PySide6.QtCore import Qt

            seq = QKeySequence(combo.replace("Ctrl", "Ctrl").replace("Alt", "Alt"))
            shortcut = QShortcut(seq, self.window)
            shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            callback = self.callbacks.get(name)
            if callback is not None:
                shortcut.activated.connect(callback)  # type: ignore[arg-type]
            self._fallback_shortcuts.append(shortcut)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    def _native_event(self, event_type, message):
        if event_type not in (b"windows_generic_MSG", b"windows_dispatcher_MSG"):
            return False, 0
        try:
            import ctypes

            msg = ctypes.wintypes.MSG.from_address(int(message))
            if msg.message != WM_HOTKEY:
                return False, 0
            spec_id = int(msg.wParam)
            spec = self.specs.get(spec_id)
            if spec is None:
                return False, 0
            name = self._id_to_name.get(spec_id, spec.name)
            callback = self.callbacks.get(name)
            if callback is not None:
                callback()  # type: ignore[misc]
            return True, 0
        except Exception:
            return False, 0

    # ------------------------------------------------------------------ #
    def unregister(self) -> None:
        if sys.platform == "win32" and self.specs:
            try:
                import ctypes
                from ctypes import wintypes

                hwnd = int(self.window.winId())
                user32 = ctypes.windll.user32  # type: ignore[attr-defined]
                for spec_id in list(self.specs):
                    user32.UnregisterHotKey(wintypes.HWND(hwnd), spec_id)
            except Exception:
                pass
        self.specs.clear()
        self._id_to_name.clear()
        for sc in self._fallback_shortcuts:
            try:
                sc.setEnabled(False)
                sc.deleteLater()
            except Exception:
                pass
        self._fallback_shortcuts.clear()

    @property
    def native_mode(self) -> bool:
        return self._native_ok and bool(self.specs)

    def describe(self) -> str:
        if self.native_mode:
            return "global hotkeys active"
        if self._fallback_shortcuts:
            return "in-app shortcuts only (window must be focused)"
        return "no hotkeys"
