"""The OllaVoice main window."""

from __future__ import annotations

import html
import json
import re
import time
from typing import Any

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence, QTextCursor
from PySide6.QtMultimedia import QAudioFormat, QAudioSource, QMediaDevices
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMenu, QMessageBox, QProgressBar, QPushButton, QSizePolicy, QSlider,
    QSystemTrayIcon, QTextBrowser, QToolBar, QVBoxLayout, QWidget,
)

from .. import __version__, paths as _paths
from ..config import Config, save as save_config
from ..pipeline import VoicePipeline
from ..vad import SpeechGate, VadSettingsLite
from ..wav import AudioClip
from . import theme
from .hotkeys import GlobalHotkeyManager
from .settings_dialog import SettingsDialog
from .workers import PipelineWorker, RefreshWorker, ThreadHandle, save_turn_to_transcript

PHASE_TEXT = {
    "idle": "Ready",
    "listening": "Listening...",
    "transcribing": "Transcribing...",
    "thinking": "Thinking...",
    "speaking": "Speaking...",
    "error": "Error",
}

CSS_USER = ("background:#243049;border-radius:12px;padding:9px 14px;margin:6px 0 6px 40px;"
            "color:#dce6ff;")
CSS_ASSISTANT = ("background:#1f242d;border-radius:12px;padding:9px 14px;margin:6px 40px 6px 0;"
                 "color:#e6e8ee;")
CSS_SYSTEM = "color:#8b93a7;font-style:italic;margin:4px 20px;"
CSS_META = "color:#6c7488;font-size:10px;margin:0 0 4px 0;"


class MainWindow(QMainWindow):
    def __init__(self, cfg: Config, *, audio_backend: str = "qt") -> None:
        super().__init__()
        self.cfg = cfg
        self.audio_backend = audio_backend
        self.session_id = time.strftime("%H%M%S")

        # capture state
        self._audio_source: QAudioSource | None = None
        self._audio_io: Any = None
        self._mic_chunks: list[np.ndarray] = []
        self._capture_rate = int(cfg.audio.sample_rate)
        self._gate: SpeechGate | None = None
        self._vad_poll: QTimer | None = None
        self._recording = False
        self._recording_started = 0.0
        self._level_peak = -120.0

        # render state
        self._assistant_buffer: list[str] = []
        self._in_assistant_block = False
        self._last_answer = ""
        self._phase = "idle"

        self.setWindowTitle(f"OllaVoice {__version__} - local voice for Ollama")
        self.resize(int(cfg.gui.window_width), int(cfg.gui.window_height))
        self.setWindowIcon(theme.app_icon())
        if cfg.gui.always_on_top:
            self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)

        self.pipeline = VoicePipeline(cfg, audio_backend=audio_backend)
        self.worker_handle = ThreadHandle(PipelineWorker(self.pipeline), "ollavoice-pipeline")
        self.refresh_handle: ThreadHandle | None = None
        self._server: Any = None
        self._hotkeys: GlobalHotkeyManager | None = None
        self.w = self.worker_handle.worker
        self.s = self.w.signals

        self._build_menubar()
        self._build_toolbar()
        self._build_central()
        self._build_statusbar()
        self._build_tray()
        self._connect_signals()
        self._apply_theme()
        self._register_hotkeys()

        self._level_timer = QTimer(self)
        self._level_timer.setInterval(90)
        self._level_timer.timeout.connect(self._update_level_meter)

        self._status_timer = QTimer(self)
        self._status_timer.setInterval(20000)
        self._status_timer.timeout.connect(self.refresh_models)
        self._status_timer.start()

        self._append_system(
            "Everything runs on this computer: Whisper hears you, Ollama answers, "
            "Piper or Kokoro speaks. Hold <b>Space</b> - or the mic button - to talk.")
        QTimer.singleShot(60, self._startup)
        QTimer.singleShot(500, self.refresh_models)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def _build_menubar(self) -> None:
        bar = self.menuBar()

        m_file = bar.addMenu("&File")
        self.act_settings = QAction("&Settings...", self)
        self.act_settings.setShortcut(QKeySequence("Ctrl+,"))
        self.act_settings.triggered.connect(self.open_settings)
        m_file.addAction(self.act_settings)
        act_import = QAction("Transcribe an audio &file...", self)
        act_import.triggered.connect(self.import_audio)
        m_file.addAction(act_import)
        act_export = QAction("&Export transcript...", self)
        act_export.setShortcut("Ctrl+S")
        act_export.triggered.connect(self.export_transcript)
        m_file.addAction(act_export)
        m_file.addSeparator()
        act_quit = QAction("&Quit", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.quit_app)
        m_file.addAction(act_quit)

        m_chat = bar.addMenu("&Chat")
        act_new = QAction("&New conversation", self)
        act_new.setShortcut("Ctrl+N")
        act_new.triggered.connect(self.new_conversation)
        m_chat.addAction(act_new)
        act_clear = QAction("C&lear the view", self)
        act_clear.setShortcut("Ctrl+K")
        act_clear.triggered.connect(self._clear_view)
        m_chat.addAction(act_clear)
        m_chat.addSeparator()
        act_respeak = QAction("&Read the last answer again", self)
        act_respeak.setShortcut("Ctrl+R")
        act_respeak.triggered.connect(self.repeat_last_answer)
        m_chat.addAction(act_respeak)

        m_voice = bar.addMenu("&Voice")
        self.act_listen = QAction("&Listen now", self)
        self.act_listen.setShortcut("Ctrl+M")
        self.act_listen.triggered.connect(self.toggle_record)
        m_voice.addAction(self.act_listen)
        act_stop = QAction("&Stop speaking", self)
        act_stop.setShortcut(QKeySequence("Esc"))
        act_stop.triggered.connect(self.stop_everything)
        m_voice.addAction(act_stop)
        m_voice.addSeparator()
        self.act_autospeak = QAction("Read answers &aloud", self)
        self.act_autospeak.setCheckable(True)
        self.act_autospeak.setChecked(bool(self.cfg.tts.auto_speak))
        self.act_autospeak.toggled.connect(self._toggle_autospeak)
        m_voice.addAction(self.act_autospeak)

        m_tools = bar.addMenu("&Tools")
        act_doctor = QAction("Run &health check", self)
        act_doctor.triggered.connect(self.show_doctor)
        m_tools.addAction(act_doctor)
        act_paths = QAction("Show &folders...", self)
        act_paths.triggered.connect(self.show_paths)
        m_tools.addAction(act_paths)
        act_reload = QAction("&Reload speech engines", self)
        act_reload.triggered.connect(self.reload_engines)
        m_tools.addAction(act_reload)
        m_tools.addSeparator()
        self.act_server = QAction("Run the local speech &API", self)
        self.act_server.setCheckable(True)
        self.act_server.setChecked(bool(self.cfg.server.enabled))
        self.act_server.toggled.connect(self._toggle_server)
        m_tools.addAction(self.act_server)

        m_help = bar.addMenu("&Help")
        act_help = QAction("&How to use this", self)
        act_help.setShortcut("F1")
        act_help.triggered.connect(self.show_help)
        m_help.addAction(act_help)
        act_about = QAction("&About OllaVoice", self)
        act_about.triggered.connect(self.show_about)
        m_help.addAction(act_about)

    def _build_toolbar(self) -> None:
        bar = QToolBar("Main")
        bar.setMovable(False)
        self.addToolBar(bar)

        bar.addWidget(QLabel(" Model "))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(180)
        self.model_combo.setEditable(True)
        self.model_combo.setCurrentText(self.cfg.ollama.model)
        self.model_combo.lineEdit().editingFinished.connect(self._on_model_changed)
        bar.addWidget(self.model_combo)

        bar.addSeparator()
        bar.addWidget(QLabel(" Voice "))
        self.tts_combo = QComboBox()
        self.tts_combo.addItems(["piper", "kokoro", "sapi", "none"])
        self.tts_combo.setCurrentText(self.cfg.tts.engine)
        self.tts_combo.currentTextChanged.connect(self._on_tts_engine_changed)
        bar.addWidget(self.tts_combo)

        self.voice_combo = QComboBox()
        self.voice_combo.setMinimumWidth(170)
        self.voice_combo.setEditable(True)
        self.voice_combo.lineEdit().editingFinished.connect(self._on_voice_changed)
        bar.addWidget(self.voice_combo)
        self._refresh_voice_combo()

        bar.addWidget(QLabel("  Rate "))
        self.rate_slider = QSlider(Qt.Orientation.Horizontal)
        self.rate_slider.setRange(60, 180)
        self.rate_slider.setValue(int(float(self.cfg.tts.rate) * 100))
        self.rate_slider.setFixedWidth(92)
        self.rate_slider.setToolTip("Speaking rate (0.6x - 1.8x)")
        self.rate_slider.valueChanged.connect(self._on_rate_changed)
        bar.addWidget(self.rate_slider)
        self.rate_label = QLabel(f" {float(self.cfg.tts.rate):.2f}x ")
        bar.addWidget(self.rate_label)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        bar.addWidget(spacer)

        self.btn_settings = QPushButton("Settings")
        self.btn_settings.clicked.connect(self.open_settings)
        bar.addWidget(self.btn_settings)

    def _build_central(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(8)

        self.chat_view = QTextBrowser()
        self.chat_view.setOpenExternalLinks(False)
        self.chat_view.setFrameShape(QTextBrowser.Shape.NoFrame)
        self.chat_view.document().setDocumentMargin(10)
        root.addWidget(self.chat_view, 1)

        input_row = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.input_edit.setObjectName("userInput")
        self.input_edit.setPlaceholderText(
            "Type a message, or hold Space to talk...")
        self.input_edit.returnPressed.connect(self.send_text)
        input_row.addWidget(self.input_edit, 1)
        self.btn_send = QPushButton("Send")
        self.btn_send.setObjectName("sendButton")
        self.btn_send.clicked.connect(self.send_text)
        input_row.addWidget(self.btn_send)
        root.addLayout(input_row)

        ctrl_row = QHBoxLayout()
        self.btn_mic = QPushButton("  Hold to talk  ")
        self.btn_mic.setObjectName("micButton")
        self.btn_mic.setCheckable(True)
        self.btn_mic.setMinimumHeight(42)
        self.btn_mic.pressed.connect(self.start_recording)
        self.btn_mic.released.connect(self._mic_released)
        self.btn_mic.toggled.connect(self._mic_toggled)
        ctrl_row.addWidget(self.btn_mic)

        self.level = QProgressBar()
        self.level.setObjectName("levelMeter")
        self.level.setRange(-60, 0)
        self.level.setValue(-60)
        self.level.setFormat("%v dB")
        self.level.setFixedHeight(22)
        ctrl_row.addWidget(self.level, 1)

        self.chk_speak = QCheckBox("Speak replies")
        self.chk_speak.setChecked(bool(self.cfg.tts.auto_speak))
        self.chk_speak.toggled.connect(self._toggle_autospeak)
        ctrl_row.addWidget(self.chk_speak)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("stopButton")
        self.btn_stop.setToolTip("Stop speaking and cancel the current turn (Esc)")
        self.btn_stop.clicked.connect(self.stop_everything)
        ctrl_row.addWidget(self.btn_stop)
        root.addLayout(ctrl_row)

        self.phase_label = QLabel(PHASE_TEXT["idle"])
        self.phase_label.setObjectName("phaseLabel")
        root.addWidget(self.phase_label)

        self.setCentralWidget(central)

    def _build_statusbar(self) -> None:
        bar = self.statusBar()
        self.lbl_ollama = QLabel("ollama: checking...")
        self.lbl_stt = QLabel("stt: -")
        self.lbl_tts = QLabel("tts: -")
        self.lbl_backend = QLabel(f"audio: {self.audio_backend}")
        self.lbl_hotkeys = QLabel("hotkeys: -")
        for widget in (self.lbl_ollama, self.lbl_stt, self.lbl_tts,
                       self.lbl_backend, self.lbl_hotkeys):
            bar.addPermanentWidget(widget)
        bar.showMessage("Starting up...")

    def _build_tray(self) -> None:
        self.tray = QSystemTrayIcon(theme.app_icon(), self)
        menu = QMenu()
        act_show = QAction("Show OllaVoice", self)
        act_show.triggered.connect(self.show_and_focus)
        menu.addAction(act_show)
        act_talk = QAction("Talk", self)
        act_talk.triggered.connect(self.toggle_record)
        menu.addAction(act_talk)
        act_stop = QAction("Stop speaking", self)
        act_stop.triggered.connect(self.stop_everything)
        menu.addAction(act_stop)
        menu.addSeparator()
        act_settings = QAction("Settings...", self)
        act_settings.triggered.connect(self.open_settings)
        menu.addAction(act_settings)
        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self.quit_app)
        menu.addAction(act_quit)
        self.tray.setContextMenu(menu)
        self.tray.setToolTip("OllaVoice - local voice for Ollama")
        self.tray.activated.connect(self._tray_activated)
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.tray.show()

    def _connect_signals(self) -> None:
        self.s.phase.connect(self._on_phase)
        self.s.level.connect(self._on_level)
        self.s.token.connect(self._on_token)
        self.s.transcript.connect(self._on_transcript)
        self.s.done.connect(self._on_done)
        self.s.error.connect(self._on_error)
        self.s.log.connect(self._on_log)
        self.s.stats.connect(self._on_stats)
        self.s.sentence.connect(lambda text: self._on_log(f"speaking: {text[:80]}"))
        self.s.audio.connect(self._on_audio_meta)
        self.w.signals.finished.connect(lambda: None)

    def _startup(self) -> None:
        self.s.request_warm.emit()
        self._on_log(f"config: {_paths.config_file()}")
        if self.cfg.server.enabled:
            self.act_server.setChecked(True)

    # ------------------------------------------------------------------ #
    # Theme
    # ------------------------------------------------------------------ #
    def _apply_theme(self) -> None:
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(theme.stylesheet(self.cfg.gui.theme, self.cfg.gui.font_size))
        chat_bg = "#12141a" if self.cfg.gui.theme != "light" else "#ffffff"
        self.chat_view.setStyleSheet(f"QTextBrowser{{background:{chat_bg};border:none;}}")
        self.setWindowIcon(theme.app_icon())

    # ------------------------------------------------------------------ #
    # Chat rendering
    # ------------------------------------------------------------------ #
    def _append_html(self, fragment: str) -> None:
        self.chat_view.moveCursor(QTextCursor.MoveOperation.End)
        self.chat_view.insertHtml(fragment)
        self._scroll_to_end()

    def _scroll_to_end(self) -> None:
        bar = self.chat_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _append_system(self, text: str) -> None:
        self._append_html(f'<div style="{CSS_SYSTEM}">{text}</div>')

    def _append_user(self, text: str, meta: str = "") -> None:
        stamp = time.strftime("%H:%M")
        tail = f" &middot; {html.escape(meta)}" if meta else ""
        self._append_html(
            f'<div style="{CSS_META}">You &middot; {stamp}{tail}</div>'
            f'<div style="{CSS_USER}">{self._render(text)}</div>')

    def _begin_assistant(self) -> None:
        if self._in_assistant_block:
            return
        stamp = time.strftime("%H:%M")
        self._assistant_buffer = []
        self._in_assistant_block = True
        self._append_html(
            f'<div style="{CSS_META}">Ollama &middot; {stamp}</div>'
            f'<div style="{CSS_ASSISTANT}">&hellip;</div>')

    def _on_token(self, delta: str) -> None:
        self._begin_assistant()
        self._assistant_buffer.append(delta)
        self._rewrite_assistant()

    def _rewrite_assistant(self, *, final: bool = False) -> None:
        """Replace the trailing assistant bubble with the accumulated text."""
        text = "".join(self._assistant_buffer).strip()
        doc = self.chat_view.document()
        cursor = QTextCursor(doc.lastBlock())
        cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
        cursor.removeSelectedText()
        cursor.insertHtml(f'<div style="{CSS_ASSISTANT}">{self._render(text, final)}</div>')
        self._scroll_to_end()

    @staticmethod
    def _render(text: str, final: bool = True) -> str:
        escaped = html.escape(text or "")
        escaped = re.sub(
            r"```(\w*)\n?(.*?)```",
            lambda m: ('<pre style="background:#10131a;padding:8px;border-radius:8px;'
                       f'color:#c8d3ff;">{m.group(2)}</pre>'),
            escaped, flags=re.DOTALL)
        escaped = re.sub(
            r"`([^`]+)`",
            r'<code style="background:#10131a;padding:1px 4px;border-radius:4px;'
            r'color:#9fd0ff;">\1</code>', escaped)
        escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
        escaped = re.sub(r"(?<!\w)\*([^*]+)\*(?!\w)", r"<i>\1</i>", escaped)
        escaped = escaped.replace("\n", "<br/>" if final else " ")
        return escaped or "&hellip;"

    def _finish_assistant(self, meta: str = "") -> None:
        if not self._in_assistant_block:
            return
        self._rewrite_assistant(final=True)
        if meta:
            self._append_html(f'<div style="{CSS_META}">{html.escape(meta)}</div>')
        self._last_answer = "".join(self._assistant_buffer).strip()
        self._in_assistant_block = False

    # ------------------------------------------------------------------ #
    # Pipeline events
    # ------------------------------------------------------------------ #
    def _on_phase(self, phase: str) -> None:
        self._phase = phase
        label = PHASE_TEXT.get(phase, phase)
        if phase == "listening" and self.cfg.vad.enabled:
            label += " (hands-free: stop talking to send)"
        self.phase_label.setText(label)
        self.btn_mic.setProperty("recording", phase == "listening")
        style = self.btn_mic.style()
        style.unpolish(self.btn_mic)
        style.polish(self.btn_mic)

    def _on_level(self, db: float, state: str) -> None:
        self._level_peak = max(self._level_peak, float(db))

    def _update_level_meter(self) -> None:
        db = self._level_peak
        self.level.setValue(int(max(-60, min(0, db))))
        self.level.setProperty("clip", db > -3.0)
        self.level.setProperty("hot", -12.0 < db <= -3.0)
        style = self.level.style()
        style.unpolish(self.level)
        style.polish(self.level)
        self._level_peak = max(-120.0, self._level_peak - 4.0)   # decay between frames

    def _on_transcript(self, text: str, meta: dict) -> None:
        """One path for "the user said this", whether it came from the mic or the keyboard."""
        if not text.strip():
            self._append_system("(nothing was recognised)")
            return
        bits: list[str] = []
        if meta.get("audio_s"):
            bits.append(f"{meta['audio_s']}s audio")
        if meta.get("decode_s"):
            bits.append(f"transcribed in {meta['decode_s']}s")
        if meta.get("model"):
            bits.append(str(meta["model"]))
        if meta.get("language"):
            bits.append(str(meta["language"]))
        self._append_user(text, " · ".join(bits))
        self.pipeline.history.add_user(text)
        if self.cfg.gui.save_transcripts:
            save_turn_to_transcript({"session": self.session_id, "role": "user",
                                     "text": text}, self.session_id)
        self._begin_assistant()

    def _on_done(self, answer: str, is_final: bool) -> None:
        if not is_final:
            return
        self._finish_assistant(self._timing_summary())
        if self.cfg.gui.save_transcripts:
            text = answer or self._last_answer
            if text:
                save_turn_to_transcript({"session": self.session_id, "role": "assistant",
                                         "text": text,
                                         "timings": self.pipeline.last_stats}, self.session_id)
        self._set_busy(False)
        self.input_edit.setFocus()

    def _timing_summary(self) -> str:
        timings = self.pipeline.last_stats or {}
        bits = []
        for key, label in (("stt_s", "whisper"), ("ttft_s", "first token"),
                           ("llm_s", "answer")):
            if key in timings:
                bits.append(f"{label} {timings[key]:.1f}s")
        return " · ".join(bits)

    def _on_error(self, message: str) -> None:
        self._in_assistant_block = False
        self._append_html(f'<div style="color:#ff9aa6;margin:6px 20px;">'
                          f'{html.escape(message)}</div>')
        self.statusBar().showMessage(message[:200], 9000)
        self._set_busy(False)
        self.input_edit.setEnabled(True)

    def _on_log(self, message: str) -> None:
        self.statusBar().showMessage(message[:200], 5000)

    def _on_stats(self, data: dict) -> None:
        if data.get("rtf"):
            self.lbl_stt.setText(f"stt: {self.cfg.stt.model} ({data['rtf']}x realtime)")

    def _on_audio_meta(self, sentence: str, meta: dict) -> None:
        dur, synth = meta.get("duration_s", 0), meta.get("synth_s", 0)
        if dur:
            self.statusBar().showMessage(
                f"spoke {dur:.1f}s of audio (synthesised in {synth:.1f}s)", 4000)

    # ------------------------------------------------------------------ #
    # Busy state
    # ------------------------------------------------------------------ #
    def _set_busy(self, busy: bool) -> None:
        for widget in (self.btn_send, self.model_combo, self.tts_combo, self.voice_combo,
                       self.rate_slider, self.btn_settings):
            widget.setEnabled(not busy)
        self.input_edit.setEnabled(not busy)
        self.btn_mic.setEnabled(not busy)

    # ------------------------------------------------------------------ #
    # Sending
    # ------------------------------------------------------------------ #
    def send_text(self) -> None:
        text = self.input_edit.text().strip()
        if not text or self.pipeline.is_busy:
            return
        self.input_edit.clear()
        self._set_busy(True)
        self.s.request_turn.emit(None, text, self.chk_speak.isChecked())

    def start_turn_from_audio(self, clip: AudioClip) -> None:
        self._set_busy(True)
        self.s.request_turn.emit(clip, "", self.chk_speak.isChecked())

    # ------------------------------------------------------------------ #
    # Microphone capture (push-to-talk + hands-free)
    # ------------------------------------------------------------------ #
    def _mic_released(self) -> None:
        # Only a real hold-to-talk release ends the recording; a plain click
        # (press+release in quick succession) keeps it running as a toggle.
        if not self._recording:
            return
        held = time.time() - self._recording_started
        if held > 0.35 and self._gate is None:
            self.finish_recording()

    def _mic_toggled(self, checked: bool) -> None:
        if checked and not self._recording and not self.pipeline.is_busy:
            self.start_recording()
        elif not checked and self._recording and self._gate is None:
            self.finish_recording()

    def toggle_record(self) -> None:
        if self._recording:
            self.finish_recording()
        else:
            self.start_recording()

    def start_recording(self) -> None:
        if self._recording or self.pipeline.is_busy:
            return
        self.stop_speaking()
        self._mic_chunks = []
        self._level_peak = -120.0

        fmt = QAudioFormat()
        fmt.setChannelCount(1)
        fmt.setSampleRate(int(self.cfg.audio.sample_rate))
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)

        device = self._selected_input_device()
        if device.isNull():
            self._on_error("No microphone was found. Plug one in, or pick a device in "
                           "Settings -> Audio devices.")
            return
        if not device.isFormatSupported(fmt):
            preferred = device.preferredFormat()
            fmt = QAudioFormat(preferred)
            fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
            fmt.setChannelCount(1)
        self._capture_rate = int(fmt.sampleRate()) or int(self.cfg.audio.sample_rate)

        try:
            self._audio_source = QAudioSource(device, fmt, self)
            self._audio_io = self._audio_source.start()
        except Exception as exc:
            self._on_error(f"Could not open the microphone: {exc}")
            self._audio_source = None
            return
        if self._audio_io is None:
            self._on_error(
                "Microphone capture would not start. On Windows, check Settings -> "
                "Privacy & security -> Microphone and make sure desktop apps are allowed.")
            self._audio_source = None
            return

        self._audio_io.readyRead.connect(self._on_mic_data)
        self._recording = True
        self._recording_started = time.time()

        if self.cfg.vad.enabled:
            self._gate = SpeechGate(VadSettingsLite(
                silence_ms=int(self.cfg.vad.silence_ms),
                min_speech_ms=int(self.cfg.vad.min_speech_ms),
                max_record_s=float(self.cfg.vad.max_record_s),
                threshold_db=float(self.cfg.vad.threshold_db),
                adaptive=bool(self.cfg.vad.adaptive),
                adaptive_margin_db=float(self.cfg.vad.adaptive_margin_db),
                pre_speech_pad_ms=int(self.cfg.vad.pre_speech_pad_ms),
                calibration_frames=int(self.cfg.vad.calibration_frames),
            ))
            self._vad_poll = QTimer(self)
            self._vad_poll.setInterval(120)
            self._vad_poll.timeout.connect(self._poll_gate)
            self._vad_poll.start()

        self._level_timer.start()
        self.btn_mic.blockSignals(True)
        self.btn_mic.setChecked(True)
        self.btn_mic.blockSignals(False)
        self.btn_mic.setText("  Listening - release to send  " if not self.cfg.vad.enabled
                             else "  Listening - stop talking to send  ")
        self._on_phase("listening")
        self.input_edit.setEnabled(False)
        self.statusBar().showMessage(
            "Recording. Release Space / click the mic again to send, Esc to discard.")

    def _selected_input_device(self):
        if self.cfg.audio.input_device:
            for d in QMediaDevices.audioInputs():
                if d.description() == self.cfg.audio.input_device:
                    return d
        return QMediaDevices.defaultAudioInput()

    def _on_mic_data(self) -> None:
        try:
            data = self._audio_io.readAll()
        except RuntimeError:
            return
        if not data:
            return
        arr = np.frombuffer(bytes(data), dtype=np.int16).copy()
        if arr.size == 0:
            return
        self._mic_chunks.append(arr)
        rms = float(np.sqrt(np.mean(np.square(arr.astype(np.float32) / 32768.0))))
        self._level_peak = max(self._level_peak,
                               20.0 * float(np.log10(rms)) if rms > 1e-7 else -120.0)

    def _poll_gate(self) -> None:
        """Hands-free mode: feed the VAD and cut when the speaker pauses."""
        gate = self._gate
        if gate is None:
            return
        chunks, self._mic_chunks = self._mic_chunks, []
        for chunk in chunks:
            for ev in gate.push(chunk, self._capture_rate):
                if ev.kind in ("speech_end", "timeout"):
                    clip = gate.clip if len(gate.clip.samples) else self._clip_from(chunks)
                    self._stop_capture()
                    self._dispatch_clip(clip)
                    return
        # Keep the chunks we just consumed for the fallback buffer.
        self._mic_chunks = chunks + self._mic_chunks

    @staticmethod
    def _clip_from(chunks: list[np.ndarray], rate: int = 16000) -> AudioClip:
        if not chunks:
            return AudioClip(np.zeros(0, dtype=np.int16), rate)
        return AudioClip(np.concatenate(chunks).astype(np.int16), rate)

    def _buffered_clip(self) -> AudioClip:
        return self._clip_from(self._mic_chunks, self._capture_rate)

    def finish_recording(self) -> None:
        if not self._recording:
            return
        if self._gate is not None:
            clip = self._gate.clip
            if not len(clip.samples):
                clip = self._buffered_clip()
        else:
            clip = self._buffered_clip()
        self._stop_capture()
        self._dispatch_clip(clip)

    def _stop_capture(self) -> None:
        self._recording = False
        self._level_timer.stop()
        if self._vad_poll is not None:
            self._vad_poll.stop()
            self._vad_poll = None
        self._gate = None
        if self._audio_source is not None:
            try:
                self._audio_io.readyRead.disconnect(self._on_mic_data)
            except (RuntimeError, AttributeError, TypeError):
                pass
            try:
                self._audio_source.stop()
                self._audio_source.deleteLater()
            except RuntimeError:
                pass
        self._audio_source = None
        self._audio_io = None
        self._mic_chunks = []
        self.btn_mic.blockSignals(True)
        self.btn_mic.setChecked(False)
        self.btn_mic.blockSignals(False)
        self.btn_mic.setText("  Hold to talk  ")
        self.level.setValue(-60)
        self.input_edit.setEnabled(True)

    def cancel_recording(self) -> None:
        self._stop_capture()
        self._on_phase("idle")
        self.statusBar().showMessage("Recording discarded", 3000)

    def _dispatch_clip(self, clip: AudioClip) -> None:
        if clip.sample_rate != int(self.cfg.audio.sample_rate):
            clip = clip.resample(int(self.cfg.audio.sample_rate))
        if clip.duration_s < 0.25:
            self._append_system("(that recording was too short - nothing to transcribe)")
            self._on_phase("idle")
            return
        self._on_phase("transcribing")
        self._on_log(f"captured {clip.duration_s:.1f}s of audio")
        self.start_turn_from_audio(clip)

    # ------------------------------------------------------------------ #
    # Keyboard
    # ------------------------------------------------------------------ #
    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            if QApplication.focusWidget() is not self.input_edit:
                self.start_recording()
                event.accept()
                return
        if event.key() == Qt.Key.Key_Escape:
            if self._recording:
                self.cancel_recording()
            else:
                self.stop_everything()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            if QApplication.focusWidget() is not self.input_edit and self._recording \
                    and self._gate is None:
                self.finish_recording()
                event.accept()
                return
        super().keyReleaseEvent(event)

    # ------------------------------------------------------------------ #
    # Stop
    # ------------------------------------------------------------------ #
    def stop_speaking(self) -> None:
        try:
            self.pipeline.cancel()
        except Exception:
            pass
        QTimer.singleShot(150, self.pipeline._cancel.clear)

    def stop_everything(self) -> None:
        if self._recording:
            self.cancel_recording()
        self.stop_speaking()
        self._on_phase("idle")
        self._in_assistant_block = False
        self._set_busy(False)
        self.input_edit.setEnabled(True)
        self.statusBar().showMessage("Stopped", 3000)

    # ------------------------------------------------------------------ #
    # Toolbar handlers
    # ------------------------------------------------------------------ #
    def _on_model_changed(self) -> None:
        model = self.model_combo.currentText().strip()
        if model and model != self.cfg.ollama.model:
            self.cfg.ollama.model = model
            save_config(self.cfg)
            self.statusBar().showMessage(f"model: {model}", 4000)

    def _on_tts_engine_changed(self, engine: str) -> None:
        if engine == self.cfg.tts.engine:
            return
        self.cfg.tts.engine = engine
        save_config(self.cfg)
        self._refresh_voice_combo()
        self.pipeline.reload_engines()
        self.lbl_tts.setText(f"tts: {engine}")
        self.statusBar().showMessage(f"speech output: {engine}", 4000)
        self.s.request_warm.emit()

    def _on_voice_changed(self) -> None:
        voice = self.voice_combo.currentText().strip()
        if not voice:
            return
        if self.cfg.tts.engine == "piper":
            if voice == self.cfg.tts.piper_voice:
                return
            self.cfg.tts.piper_voice = voice
        elif self.cfg.tts.engine == "kokoro":
            if voice == self.cfg.tts.kokoro_voice:
                return
            self.cfg.tts.kokoro_voice = voice
        else:
            return
        save_config(self.cfg)
        self.pipeline.reload_engines()
        self.statusBar().showMessage(f"voice: {voice}", 4000)
        self.s.request_speak.emit("Voice changed.")

    def _on_rate_changed(self, value: int) -> None:
        rate = value / 100.0
        self.rate_label.setText(f" {rate:.2f}x ")
        if abs(rate - float(self.cfg.tts.rate)) > 0.005:
            self.cfg.tts.rate = rate
            save_config(self.cfg)
            engine = self.pipeline._tts
            if engine is not None:
                engine.settings = self.cfg.tts

    def _refresh_voice_combo(self) -> None:
        from ..tts_engines import installed_voices

        combo = self.voice_combo
        voices = installed_voices()
        engine = self.cfg.tts.engine
        if engine == "piper":
            items = voices["piper"] or [self.cfg.tts.piper_voice]
            current = self.cfg.tts.piper_voice
        elif engine == "kokoro":
            items = ["af_heart", "af_bella", "af_sky", "af_nicole", "af_nova", "af_sarah",
                     "af_sol", "am_adam", "am_michael", "am_echo", "bf_emma", "bm_george",
                     "zf_xiaobei", "zm_yunjian", "jf_alpha", "ef_dora", "hf_alpha",
                     "if_sara", "pf_driella", "lf_alfa"]
            current = self.cfg.tts.kokoro_voice
        elif engine == "sapi":
            items = [self.cfg.tts.sapi_voice or "(Windows default)"]
            current = items[0]
        else:
            items = ["(speech output disabled)"]
            current = items[0]
        combo.blockSignals(True)
        combo.clear()
        combo.addItems([i for i in items if i])
        combo.setCurrentText(current)
        combo.setEnabled(engine in ("piper", "kokoro"))
        combo.blockSignals(False)

    def _toggle_autospeak(self, checked: bool) -> None:
        self.cfg.tts.auto_speak = bool(checked)
        save_config(self.cfg)
        for widget in (self.act_autospeak, self.chk_speak):
            widget.blockSignals(True)
            widget.setChecked(bool(checked))
            widget.blockSignals(False)
        if not checked:
            self.stop_speaking()

    # ------------------------------------------------------------------ #
    # Ollama status refresh
    # ------------------------------------------------------------------ #
    def refresh_models(self) -> None:
        if self.refresh_handle is not None:
            return
        worker = RefreshWorker(self.cfg)
        worker.models.connect(self._on_models)
        worker.status.connect(self._on_server_status)
        worker.error.connect(self._on_log)
        worker.finished.connect(self._on_refresh_finished)
        self.refresh_handle = ThreadHandle(worker, "ollavoice-refresh")
        self.refresh_handle.worker.request.emit()

    def _on_models(self, models: list[str]) -> None:
        if not models:
            return
        current = self.model_combo.currentText()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems(models)
        if current in models:
            self.model_combo.setCurrentText(current)
        elif self.cfg.ollama.model in models:
            self.model_combo.setCurrentText(self.cfg.ollama.model)
        else:
            self.model_combo.setCurrentText(models[0])
            self.cfg.ollama.model = models[0]
        self.model_combo.blockSignals(False)

    def _on_server_status(self, status: dict[str, Any]) -> None:
        if status.get("alive"):
            text = f"ollama: v{status.get('version', '?')}"
            running = status.get("running") or []
            if running:
                text += f" | loaded: {running[0].get('name', '')}"
            self.lbl_ollama.setText(text)
        else:
            self.lbl_ollama.setText("ollama: OFFLINE")
        engines = {e["name"]: e for e in status.get("stt_engines", [])}
        info = engines.get(self.cfg.stt.engine) or {}
        suffix = "" if info.get("available", True) else " (missing!)"
        self.lbl_stt.setText(f"stt: {self.cfg.stt.engine}/{self.cfg.stt.model}{suffix}")
        voices = status.get("voices", {})
        if self.cfg.tts.engine == "piper":
            self.lbl_tts.setText(f"tts: piper/{len(voices.get('piper', []))} voices")
        else:
            self.lbl_tts.setText(f"tts: {self.cfg.tts.engine}")

    def _on_refresh_finished(self) -> None:
        if self.refresh_handle is not None:
            self.refresh_handle.stop()
            self.refresh_handle = None

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #
    def _clear_view(self) -> None:
        self.chat_view.clear()

    def new_conversation(self) -> None:
        self.stop_everything()
        self.pipeline.history.clear()
        self.chat_view.clear()
        self.session_id = time.strftime("%H%M%S")
        self._append_system("Started a new conversation.")

    def repeat_last_answer(self) -> None:
        if not self._last_answer:
            self.statusBar().showMessage("Nothing to repeat yet", 3000)
            return
        self.s.request_speak.emit(self._last_answer)

    def import_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Transcribe an audio file", "",
            "Audio (*.wav);;All files (*)")
        if not path:
            return
        from ..wav import read_wav

        try:
            clip = read_wav(path)
        except Exception as exc:
            QMessageBox.warning(
                self, "Import",
                f"Could not read that file as 16-bit WAV:\n{exc}\n\n"
                "Convert it first, for example:\n"
                "  ffmpeg -i in.mp3 -ar 16000 -ac 1 out.wav")
            return
        self._append_system(f"Transcribing {path} ({clip.duration_s:.1f}s of audio)...")
        self.start_turn_from_audio(clip)

    def export_transcript(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export transcript",
            f"ollavoice-{time.strftime('%Y%m%d-%H%M')}.txt",
            "Text (*.txt);;HTML (*.html);;JSON (*.json)")
        if not path:
            return
        try:
            if path.lower().endswith(".json"):
                payload = [{"role": m.role, "text": m.content}
                           for m in self.pipeline.history.messages]
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, indent=2, ensure_ascii=False)
            elif path.lower().endswith(".html"):
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write("<html><body>" + self.chat_view.toHtml() + "</body></html>")
            else:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(self.pipeline.history.as_transcript() + "\n")
            self.statusBar().showMessage(f"Saved {path}", 5000)
        except OSError as exc:
            QMessageBox.warning(self, "Export", str(exc))

    def reload_engines(self) -> None:
        self.pipeline.reload_engines()
        self.s.request_warm.emit()
        self.statusBar().showMessage("Reloading speech engines...", 4000)

    def open_settings(self) -> None:
        models = [self.model_combo.itemText(i) for i in range(self.model_combo.count())]
        dialog = SettingsDialog(self.cfg, models, parent=self,
                                refresh_models=self.refresh_models)
        dialog.config_changed.connect(self._on_config_changed)
        dialog.exec()

    def _on_config_changed(self, cfg: Config) -> None:
        self.cfg = cfg
        self.pipeline.config = cfg
        self.pipeline.history.turns = cfg.ollama.history_turns
        self.pipeline.history.system = cfg.ollama.system_prompt
        self.pipeline.reload_engines()

        self.model_combo.blockSignals(True)
        self.model_combo.setCurrentText(cfg.ollama.model)
        self.model_combo.blockSignals(False)
        self.tts_combo.blockSignals(True)
        self.tts_combo.setCurrentText(cfg.tts.engine)
        self.tts_combo.blockSignals(False)
        self.rate_slider.blockSignals(True)
        self.rate_slider.setValue(int(float(cfg.tts.rate) * 100))
        self.rate_slider.blockSignals(False)
        self.rate_label.setText(f" {float(cfg.tts.rate):.2f}x ")
        self._toggle_autospeak(bool(cfg.tts.auto_speak))
        self._refresh_voice_combo()
        self._apply_theme()
        self._register_hotkeys()

        if bool(cfg.gui.always_on_top) != bool(self.windowFlags()
                                               & Qt.WindowType.WindowStaysOnTopHint):
            self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, bool(cfg.gui.always_on_top))
            self.show()

        self.s.request_warm.emit()
        self.refresh_models()
        self._toggle_server(bool(cfg.server.enabled))
        self.act_server.blockSignals(True)
        self.act_server.setChecked(bool(cfg.server.enabled))
        self.act_server.blockSignals(False)
        self.statusBar().showMessage("Settings saved", 4000)

    def show_doctor(self) -> None:
        from .. import status_report

        report = status_report()
        box = QMessageBox(self)
        box.setWindowTitle("Health check")
        box.setText("Health report - click 'Show Details' for the full output.")
        box.setInformativeText(report.splitlines()[3] if len(report.splitlines()) > 3 else "")
        box.setDetailedText(report)
        box.setIcon(QMessageBox.Icon.Information)
        box.exec()

    def show_paths(self) -> None:
        lines = "\n".join(f"{k}: {v}" for k, v in _paths.describe().items())
        QMessageBox.information(
            self, "Folders",
            "Everything OllaVoice stores stays on this machine:\n\n" + lines)

    def show_help(self) -> None:
        QMessageBox.information(
            self, "How to use OllaVoice",
            "<b>Talk</b><br/>"
            "Hold <b>Space</b>, the mic button, or your global hotkey, and speak. "
            "Release to send. Whisper transcribes you locally, Ollama answers locally, "
            "and a local voice reads the answer.<br/><br/>"
            "<b>Type</b><br/>"
            "Click the text box and press Enter. The reply is still spoken if "
            "'Speak replies' is ticked.<br/><br/>"
            "<b>Hands-free</b><br/>"
            "Settings &rarr; Voice input &rarr; 'Hands-free' ends the recording for you "
            "when you stop talking.<br/><br/>"
            "<b>Stop</b><br/>"
            "Escape, the Stop button, or your stop hotkey cuts the voice off "
            "immediately.<br/><br/>"
            "<b>Nothing leaves this computer.</b> No cloud API, no telemetry, no account.")

    def show_about(self) -> None:
        from .. import hardware

        hw = hardware.detect()
        QMessageBox.about(
            self, "About OllaVoice",
            f"<h3>OllaVoice {__version__}</h3>"
            "<p>Speech input and speech output for <b>Ollama</b> - the layer Ollama does "
            "not ship natively.</p>"
            "<p>Local Whisper is the ear, local Piper or Kokoro is the mouth, and your "
            "local Ollama model is the brain. No cloud services, no telemetry, no "
            "account, no licence key.</p>"
            f"<p><small>{html.escape(hw.summary().replace(chr(10), ' | '))}</small></p>"
            "<p><small>Ollama is a trademark of its respective owner; OllaVoice is an "
            "independent companion tool. Whisper, Piper and Kokoro belong to their "
            "respective authors.</small></p>")

    # ------------------------------------------------------------------ #
    # Local speech API
    # ------------------------------------------------------------------ #
    def _toggle_server(self, enabled: bool) -> None:
        if enabled:
            if self._server is not None:
                return
            try:
                from ..server import start_background

                self._server = start_background(self.cfg, self.cfg.server.host,
                                                int(self.cfg.server.port))
                self.statusBar().showMessage(
                    f"speech API on http://{self.cfg.server.host}:{self.cfg.server.port}", 6000)
            except Exception as exc:
                self.act_server.blockSignals(True)
                self.act_server.setChecked(False)
                self.act_server.blockSignals(False)
                QMessageBox.warning(self, "Local speech API", f"Could not start:\n{exc}")
        elif self._server is not None:
            self._server.shutdown()
            self._server = None
            self.statusBar().showMessage("speech API stopped", 4000)

    # ------------------------------------------------------------------ #
    # Global hotkeys
    # ------------------------------------------------------------------ #
    def _register_hotkeys(self) -> None:
        if not self.cfg.hotkeys.enabled:
            self.lbl_hotkeys.setText("hotkeys: off")
            return
        try:
            if self._hotkeys is None:
                self._hotkeys = GlobalHotkeyManager(self, {
                    "push_to_talk": self._hotkey_talk,
                    "stop_speaking": self.stop_everything,
                })
            results = self._hotkeys.register({
                "push_to_talk": self.cfg.hotkeys.push_to_talk,
                "stop_speaking": self.cfg.hotkeys.stop_speaking,
            })
            active = [name for name, ok in results.items() if ok]
            mode = "" if self._hotkeys.native_mode else " (focus needed)"
            self.lbl_hotkeys.setText(f"hotkeys: {', '.join(active) or 'none'}{mode}")
        except Exception as exc:
            self.lbl_hotkeys.setText(f"hotkeys: unavailable ({exc})")

    def _hotkey_talk(self) -> None:
        QTimer.singleShot(0, self.toggle_record)

    # ------------------------------------------------------------------ #
    # Window lifecycle
    # ------------------------------------------------------------------ #
    def _tray_activated(self, reason) -> None:
        if reason in (QSystemTrayIcon.ActivationReason.Trigger,
                      QSystemTrayIcon.ActivationReason.DoubleClick,
                      QSystemTrayIcon.ActivationReason.MiddleClick):
            self.show_and_focus()

    def show_and_focus(self) -> None:
        self.showNormal()
        self.activateWindow()
        self.raise_()
        self.input_edit.setFocus()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt naming
        if self.cfg.gui.minimize_to_tray and self.tray.isVisible():
            event.ignore()
            self.hide()
            self.tray.showMessage(
                "OllaVoice",
                "Still running in the tray - use the tray icon or your hotkey to talk.",
                QSystemTrayIcon.MessageIcon.Information, 2500)
            return
        self.quit_app()
        event.accept()

    def quit_app(self) -> None:
        self._status_timer.stop()
        self._level_timer.stop()
        try:
            self.cfg.gui.window_width = int(self.width())
            self.cfg.gui.window_height = int(self.height())
            save_config(self.cfg)
        except Exception:
            pass
        self.stop_everything()
        if self._hotkeys is not None:
            try:
                self._hotkeys.unregister()
            except Exception:
                pass
        for handle in (self.worker_handle, self.refresh_handle):
            if handle is not None:
                handle.stop(timeout_ms=2500)
        if self._server is not None:
            try:
                self._server.shutdown()
            except Exception:
                pass
        self.tray.hide()
        QApplication.quit()
