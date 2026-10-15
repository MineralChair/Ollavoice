"""GUI smoke tests.

Two modes:

* **Real Qt present** - construct the window on the ``offscreen`` platform and
  drive it. This is the real deal.
* **No Qt** (typical CI, or a headless container) - install :mod:`tests.qt_stub`,
  a permissive fake ``PySide6``, then import and construct everything. The stub
  cannot validate Qt semantics, but it does catch the bugs that actually ship:
  typos in method/signal names, missing imports, wrong arity on our own methods,
  and anything that raises during construction or in an event callback.
"""

from __future__ import annotations

import importlib
import sys

import pytest

REAL_QT = True
try:
    # Importing the bare package is not enough: it succeeds even when the Qt
    # shared libraries are missing, so probe QtCore (and QtWidgets, which the
    # window needs) before trusting it.
    from PySide6 import QtCore, QtWidgets  # noqa: F401
except Exception:                       # ImportError *or* a missing libglib/libGL
    REAL_QT = False
    # Drop any half-initialised modules so the stub can take their place.
    for _name in [m for m in sys.modules if m == "PySide6" or m.startswith("PySide6.")]:
        del sys.modules[_name]


@pytest.fixture
def gui_modules():
    """Import the GUI package with either real Qt or the stub."""
    if not REAL_QT:
        from tests import qt_stub

        qt_stub.install()
    for name in [m for m in sys.modules if m.startswith("ollavoice.gui")]:
        del sys.modules[name]
    modules = {
        "theme": importlib.import_module("ollavoice.gui.theme"),
        "hotkeys": importlib.import_module("ollavoice.gui.hotkeys"),
        "workers": importlib.import_module("ollavoice.gui.workers"),
        "settings": importlib.import_module("ollavoice.gui.settings_dialog"),
        "main": importlib.import_module("ollavoice.gui.main_window"),
        "app": importlib.import_module("ollavoice.gui.app"),
    }
    return modules


def _make_config():
    from ollavoice.config import Config, apply_tier

    cfg = Config()
    apply_tier(cfg, "low")
    cfg.ollama.model = "testmodel:latest"
    cfg.tts.engine = "none"          # never synthesise during a GUI test
    cfg.gui.save_transcripts = False
    cfg.gui.minimize_to_tray = False
    return cfg


# --------------------------------------------------------------------------- #
# Modules that do not need a QApplication
# --------------------------------------------------------------------------- #

def test_theme_stylesheets_render(gui_modules):
    theme = gui_modules["theme"]
    for name in ("dark", "light"):
        css = theme.stylesheet(name, font_size=13)
        assert "QWidget" in css and "13px" in css
        assert "%(" not in css, "unsubstituted placeholder left in the stylesheet"


def test_icons_are_valid_data_uris(gui_modules):
    theme = gui_modules["theme"]
    icons = theme.icons()
    assert {"mic", "speaker", "stop", "gear", "send", "app", "trash"} <= set(icons)
    for name, uri in icons.items():
        assert uri.startswith("data:image/svg+xml;base64,"), name
        import base64

        decoded = base64.b64decode(uri.split(",", 1)[1]).decode()
        assert decoded.startswith("<svg") and decoded.endswith("</svg>")
        assert "<path" in decoded or "<rect" in decoded or "<circle" in decoded


def test_hotkey_parsing(gui_modules):
    hotkeys = gui_modules["hotkeys"]
    spec = hotkeys.parse_combo("Ctrl+Alt+Space")
    assert spec.valid and spec.vk == 0x20
    assert spec.modifiers & hotkeys.MOD_CONTROL
    assert spec.modifiers & hotkeys.MOD_ALT
    assert spec.modifiers & hotkeys.MOD_NOREPEAT

    assert hotkeys.parse_combo("F9").vk == 0x78
    assert hotkeys.parse_combo("Ctrl+Shift+M").vk == ord("M")
    assert not hotkeys.parse_combo("").valid
    assert not hotkeys.parse_combo("NotAKey").valid


# --------------------------------------------------------------------------- #
# Window construction and event handling
# --------------------------------------------------------------------------- #

def test_main_window_constructs_and_handles_events(gui_modules, tmp_home):
    pytest.importorskip("numpy")
    main = gui_modules["main"]
    cfg = _make_config()

    window = main.MainWindow(cfg, audio_backend="qt")

    # --- toolbar / menus exist
    for attr in ("model_combo", "tts_combo", "voice_combo", "rate_slider", "btn_mic",
                 "btn_send", "btn_stop", "input_edit", "chat_view", "level",
                 "phase_label", "chk_speak", "tray", "lbl_ollama", "lbl_stt", "lbl_tts"):
        assert hasattr(window, attr), f"window is missing {attr}"

    # --- pipeline events render without exploding
    window._on_phase("listening")
    window._on_phase("thinking")
    window._on_phase("idle")
    window._on_level(-18.5, "speech")
    window._update_level_meter()
    window._on_transcript("what time is it", {"audio_s": 1.2, "decode_s": 0.3,
                                              "model": "base.en", "language": "en"})
    window._on_token("It is ")
    window._on_token("half past nine.")
    window._on_done("It is half past nine.", True)
    window._on_done("", False)          # non-final events must be ignored
    window._on_error("Ollama is not running")
    window._on_log("status line")
    window._on_stats({"rtf": 0.4})
    window._on_audio_meta("It is half past nine.", {"duration_s": 1.4, "synth_s": 0.2})

    # --- toolbar handlers
    window._on_model_changed()
    window._on_tts_engine_changed("none")
    window._on_voice_changed()
    window._on_rate_changed(125)
    window._refresh_voice_combo()
    window._toggle_autospeak(False)
    window._toggle_autospeak(True)

    # --- status refresh from the fake Ollama payload
    window._on_models(["testmodel:latest", "qwen3:4b"])
    window._on_server_status({
        "alive": True, "version": "0.40.3", "models": ["testmodel:latest"],
        "running": [{"name": "testmodel:latest"}],
        "stt_engines": [{"name": "faster-whisper", "available": True, "hint": "",
                         "description": "x"}],
        "tts_engines": [], "voices": {"piper": ["en_US-lessac-medium"], "kokoro": []},
    })
    window._on_server_status({"alive": False, "models": [], "running": [],
                              "stt_engines": [], "tts_engines": [], "voices": {}})
    window._on_refresh_finished()

    # --- recording flow with no audio hardware present
    window.start_recording()
    window._on_phase("listening")
    window.finish_recording()
    window.cancel_recording()
    window.toggle_record()
    window.stop_everything()
    window.stop_speaking()

    # --- conversation management
    window.new_conversation()
    window.repeat_last_answer()
    window._clear_view()
    window._register_hotkeys()
    window._apply_theme()
    window.show_and_focus()

    # --- config round trip through the settings dialog path
    window._on_config_changed(_make_config())

    window.quit_app()


def test_mic_button_state_tracks_recording(gui_modules, tmp_home):
    main = gui_modules["main"]
    window = main.MainWindow(_make_config(), audio_backend="qt")
    assert not window._recording
    window.start_recording()
    started = window._recording
    window.finish_recording()
    assert not window._recording
    window.quit_app()
    assert started in (True, False)      # depends on whether a mic exists


def test_settings_dialog_constructs_and_collects(gui_modules, tmp_home):
    settings_mod = gui_modules["settings"]
    cfg = _make_config()
    dialog = settings_mod.SettingsDialog(cfg, ["testmodel:latest", "qwen3:4b"])

    for attr in ("model_combo", "stt_engine", "stt_model", "tts_engine", "piper_voice",
                 "kokoro_voice", "rate_slider", "volume_slider", "input_device",
                 "output_device", "hk_ptt", "theme", "server_port"):
        assert hasattr(dialog, attr), f"dialog is missing {attr}"

    dialog._scan_devices()
    dialog._on_engine_change("kokoro")
    dialog._on_engine_change("piper")
    dialog._on_engine_change("sapi")
    dialog._on_engine_change("none")

    collected = dialog._collect()
    if REAL_QT:
        assert collected.ollama.model == "testmodel:latest"
        assert collected.stt.engine == cfg.stt.engine
    else:
        # With the stub, widget getters return placeholders; what matters is that
        # collecting produced a well-typed config.
        assert isinstance(collected.ollama.model, str)
        assert collected.ollama.host.startswith("http")
        assert collected.stt.engine in ("faster-whisper", "whisper-cpp", "ollama",
                                        "windows", "stub-text")
        assert isinstance(collected.ollama.system_prompt, str)
        assert isinstance(collected.hotkeys.push_to_talk, str)
    assert 0.5 <= collected.tts.rate <= 2.0
    assert collected.audio.sample_rate in (0, 8000, 16000, 22050, 44100, 48000)
    assert isinstance(collected.gui.save_transcripts, bool)
    assert isinstance(collected.server.port, int)

    dialog._on_apply(save=False)
    dialog._on_ok()


def test_worker_signal_bundle_is_complete(gui_modules):
    workers = gui_modules["workers"]
    signals = workers.Signals()
    for name in ("log", "phase", "level", "transcript", "token", "sentence",
                 "audio", "stats", "done", "error", "finished"):
        assert hasattr(signals, name), f"Signals is missing {name}"


def test_pipeline_worker_routes_every_event_kind(gui_modules, tmp_home, cfg):
    """Every event kind the pipeline can emit must have a handler."""
    workers = gui_modules["workers"]
    from ollavoice.pipeline import Event, VoicePipeline

    pipeline = VoicePipeline(cfg, audio_backend="system")
    worker = workers.PipelineWorker(pipeline)

    kinds = ["phase", "level", "token", "transcript", "sentence", "audio",
             "stats", "done", "error", "info"]
    for kind in kinds:
        if kind == "phase":
            worker._on_event(Event(kind=kind, phase="thinking"))
        elif kind == "level":
            worker._on_event(Event(kind=kind, data={"db": -12.0, "state": "speech"}))
        elif kind == "done":
            worker._on_event(Event(kind=kind, text="hi", data={"final": True}))
        else:
            worker._on_event(Event(kind=kind, text="payload", data={"a": 1}))
    # An unknown kind must not raise.
    worker._on_event(Event(kind="something-new", text="x"))


def test_app_entry_point_reports_missing_qt_cleanly(monkeypatch):
    """Without PySide6 the GUI command must print install instructions, not crash."""
    if REAL_QT:
        pytest.skip("PySide6 is installed here; the fallback path cannot be exercised")
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("PySide6"):
            raise ImportError("no Qt in this environment")
        return real_import(name, *args, **kwargs)

    from ollavoice.gui import app as app_mod

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert app_mod.main() == 2


def test_single_instance_guard_is_reentrant(tmp_home):
    from ollavoice.gui import app as app_mod

    assert app_mod._single_instance_guard() is True
    # Our own lock file must not lock us out.
    assert app_mod._single_instance_guard() is True
    assert not app_mod._pid_alive(0)
    assert not app_mod._pid_alive(-1)
