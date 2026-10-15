"""The Settings dialog: every knob in ``config.json``, grouped into tabs.

Nothing here talks to the network. Buttons that *look* like they might (pull a
model, download a voice) shell out to the local Ollama server or the downloader
module, and both report progress inline.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
    QPushButton, QSlider, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from ..config import TIER_ORDER, Config, apply_tier
from .. import hardware, paths as _paths


def _normalise_url(text: str, fallback: str) -> str:
    """Accept ``localhost:11434`` as well as a full URL; keep the old value if
    what is in the box is not a usable address."""
    value = (text or "").strip().rstrip("/")
    if not value:
        return fallback
    if "://" not in value:
        value = "http://" + value
    if not value.startswith(("http://", "https://")) or not value.split("://", 1)[1]:
        return fallback
    return value


def _safe_int(value: Any, fallback: int) -> int:
    """Widgets can hand back non-numeric text; never lose the previous value."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def _form_group(title: str) -> tuple[QGroupBox, QFormLayout]:
    box = QGroupBox(title)
    layout = QFormLayout(box)
    layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
    layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft)
    layout.setHorizontalSpacing(14)
    layout.setVerticalSpacing(9)
    return box, layout


class SettingsDialog(QDialog):
    """Modal settings editor. Emits ``config_changed`` when the user hits OK."""

    config_changed = Signal(object)

    def __init__(self, cfg: Config, models: list[str], parent: QWidget | None = None,
                 refresh_models: Callable[[], None] | None = None) -> None:
        super().__init__(parent)
        self.cfg = cfg
        self.refresh_models = refresh_models or (lambda: None)
        self.setWindowTitle("OllaVoice settings")
        self.setMinimumSize(700, 620)
        self.setModal(True)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_general(models), "General")
        self.tabs.addTab(self._build_input(), "Voice input")
        self.tabs.addTab(self._build_output(), "Voice output")
        self.tabs.addTab(self._build_audio(), "Audio devices")
        self.tabs.addTab(self._build_advanced(), "Advanced")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Apply)
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self._on_apply)

        root = QVBoxLayout(self)
        root.addWidget(self.tabs, 1)
        root.addWidget(buttons)

    # ------------------------------------------------------------------ #
    # General
    # ------------------------------------------------------------------ #
    def _build_general(self, models: list[str]) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        box, form = _form_group("Ollama")
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(models or [])
        self.model_combo.setCurrentText(self.cfg.ollama.model)
        self.model_combo.setMinimumWidth(300)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_models)
        row = QHBoxLayout()
        row.addWidget(self.model_combo, 1)
        row.addWidget(refresh)
        wrap = QWidget()
        wrap.setLayout(row)
        form.addRow("Chat model", wrap)

        self.host_edit = QLineEdit(self.cfg.ollama.host)
        form.addRow("Server URL", self.host_edit)

        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(float(self.cfg.ollama.temperature))
        form.addRow("Temperature", self.temp_spin)

        self.ctx_spin = QSpinBox()
        self.ctx_spin.setRange(512, 131072)
        self.ctx_spin.setSingleStep(1024)
        self.ctx_spin.setValue(int(self.cfg.ollama.num_ctx))
        form.addRow("Context length", self.ctx_spin)

        self.predict_spin = QSpinBox()
        self.predict_spin.setRange(-1, 32768)
        self.predict_spin.setValue(int(self.cfg.ollama.num_predict))
        self.predict_spin.setSpecialValueText("model default")
        form.addRow("Max reply tokens", self.predict_spin)

        self.history_spin = QSpinBox()
        self.history_spin.setRange(0, 200)
        self.history_spin.setValue(int(self.cfg.ollama.history_turns))
        form.addRow("Remember turns", self.history_spin)
        layout.addWidget(box)

        box2, form2 = _form_group("Behaviour")
        self.system_prompt = QPlainTextEdit(self.cfg.ollama.system_prompt)
        self.system_prompt.setFixedHeight(96)
        form2.addRow("System prompt", self.system_prompt)

        self.save_transcripts = QCheckBox("Save transcripts to disk")
        self.save_transcripts.setChecked(bool(self.cfg.gui.save_transcripts))
        form2.addRow("", self.save_transcripts)
        hint = QLabel(str(_paths.transcripts_dir()))
        hint.setObjectName("hintLabel")
        form2.addRow("", hint)
        layout.addWidget(box2)

        box3, form3 = _form_group("Performance preset")
        self.tier_combo = QComboBox()
        self.tier_combo.addItems([*TIER_ORDER, "custom"])
        self.tier_combo.setCurrentText(self.cfg.tier)
        form3.addRow("Tier", self.tier_combo)
        self.tier_btn = QPushButton("Apply tier defaults")
        self.tier_btn.clicked.connect(self._apply_tier)
        form3.addRow("", self.tier_btn)
        hw = hardware.detect()
        rec, why = hardware.recommend_tier(hw)
        info = QLabel(f"This machine looks like a '{rec}' tier device.\n{why[0] if why else ''}")
        info.setObjectName("hintLabel")
        info.setWordWrap(True)
        form3.addRow("", info)
        layout.addWidget(box3)
        layout.addStretch(1)
        return page

    def _apply_tier(self) -> None:
        tier = self.tier_combo.currentText()
        if tier not in TIER_ORDER:
            QMessageBox.information(self, "Tier", "Choose low, balanced or high first.")
            return
        answer = QMessageBox.question(
            self, "Apply tier",
            f"Apply the '{tier}' preset?\n\nThis overwrites the model, Whisper size and "
            "voice choices on this dialog. Anything you already typed is kept.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        scratch = Config.from_dict(self._collect().to_dict())
        apply_tier(scratch, tier)
        self.model_combo.setCurrentText(scratch.ollama.model)
        self.stt_model.setCurrentText(scratch.stt.model)
        self.tts_engine.setCurrentText(scratch.tts.engine)
        self.piper_voice.setCurrentText(scratch.tts.piper_voice)
        self.kokoro_voice.setCurrentText(scratch.tts.kokoro_voice)
        self.stt_device.setCurrentText(scratch.stt.device)
        self.stt_compute.setCurrentText(scratch.stt.compute_type)

    # ------------------------------------------------------------------ #
    # Voice input
    # ------------------------------------------------------------------ #
    def _build_input(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        box, form = _form_group("Speech-to-text")
        self.stt_engine = QComboBox()
        self.stt_engine.addItems(["faster-whisper", "whisper-cpp", "ollama", "windows"])
        self.stt_engine.setCurrentText(self.cfg.stt.engine)
        form.addRow("Engine", self.stt_engine)

        self.stt_model = QComboBox()
        self.stt_model.setEditable(True)
        self.stt_model.addItems(["tiny.en", "base.en", "small.en", "medium.en", "large-v3",
                                 "distil-small.en", "distil-medium.en", "distil-large-v3",
                                 "tiny", "base", "small", "medium"])
        self.stt_model.setCurrentText(self.cfg.stt.model)
        form.addRow("Whisper model", self.stt_model)

        self.stt_language = QComboBox()
        self.stt_language.setEditable(True)
        self.stt_language.addItems(["auto", "en", "de", "fr", "es", "it", "pt", "nl", "pl",
                                    "ru", "uk", "tr", "ar", "fa", "hi", "zh", "ja", "ko",
                                    "vi", "th", "cs", "sv", "no", "da", "fi", "el", "he"])
        self.stt_language.setCurrentText(self.cfg.stt.language)
        form.addRow("Language", self.stt_language)

        self.stt_device = QComboBox()
        self.stt_device.addItems(["auto", "cpu", "cuda"])
        self.stt_device.setCurrentText(self.cfg.stt.device)
        form.addRow("Device", self.stt_device)

        self.stt_compute = QComboBox()
        self.stt_compute.addItems(["auto", "int8", "int8_float16", "float16", "float32"])
        self.stt_compute.setCurrentText(self.cfg.stt.compute_type)
        form.addRow("Compute type", self.stt_compute)

        self.beam_spin = QSpinBox()
        self.beam_spin.setRange(1, 10)
        self.beam_spin.setValue(int(self.cfg.stt.beam_size))
        form.addRow("Beam size", self.beam_spin)

        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(0, 64)
        self.threads_spin.setSpecialValueText("auto")
        self.threads_spin.setValue(int(self.cfg.stt.cpu_threads))
        form.addRow("CPU threads", self.threads_spin)

        self.vad_filter = QCheckBox("Filter non-speech (reduces hallucinations)")
        self.vad_filter.setChecked(bool(self.cfg.stt.vad_filter))
        form.addRow("", self.vad_filter)

        self.stt_prompt = QLineEdit(self.cfg.stt.initial_prompt)
        form.addRow("Vocabulary hint", self.stt_prompt)
        hint = QLabel("A short sentence of proper nouns Whisper should expect "
                      "(improves accuracy on names and jargon).")
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        form.addRow("", hint)
        layout.addWidget(box)

        box2, form2 = _form_group("Listening mode")
        self.vad_enabled = QCheckBox("Hands-free: stop automatically when I finish speaking")
        self.vad_enabled.setChecked(bool(self.cfg.vad.enabled))
        form2.addRow("", self.vad_enabled)
        self.silence_spin = QSpinBox()
        self.silence_spin.setRange(200, 5000)
        self.silence_spin.setSingleStep(50)
        self.silence_spin.setSuffix(" ms")
        self.silence_spin.setValue(int(self.cfg.vad.silence_ms))
        form2.addRow("Trailing silence", self.silence_spin)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(-90.0, -6.0)
        self.threshold_spin.setSuffix(" dB")
        self.threshold_spin.setValue(float(self.cfg.vad.threshold_db))
        form2.addRow("Noise floor", self.threshold_spin)
        self.adaptive_vad = QCheckBox("Adapt the noise floor to the room automatically")
        self.adaptive_vad.setChecked(bool(self.cfg.vad.adaptive))
        form2.addRow("", self.adaptive_vad)
        layout.addWidget(box2)
        layout.addStretch(1)
        return page

    # ------------------------------------------------------------------ #
    # Voice output
    # ------------------------------------------------------------------ #
    def _build_output(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        box, form = _form_group("Text-to-speech")
        self.tts_engine = QComboBox()
        self.tts_engine.addItems(["piper", "kokoro", "sapi", "none"])
        self.tts_engine.setCurrentText(self.cfg.tts.engine)
        self.tts_engine.currentTextChanged.connect(self._on_engine_change)
        form.addRow("Engine", self.tts_engine)

        self.piper_voice = QComboBox()
        self.piper_voice.setEditable(True)
        self.piper_voice.addItems(self._installed_piper_voices())
        self.piper_voice.setCurrentText(self.cfg.tts.piper_voice)
        form.addRow("Piper voice", self.piper_voice)

        self.piper_get = QPushButton("Download another voice...")
        self.piper_get.clicked.connect(self._download_piper_voice)
        form.addRow("", self.piper_get)

        self.kokoro_voice = QComboBox()
        self.kokoro_voice.setEditable(True)
        self.kokoro_voice.addItems(self._installed_kokoro_voices())
        self.kokoro_voice.setCurrentText(self.cfg.tts.kokoro_voice)
        form.addRow("Kokoro voice", self.kokoro_voice)

        self.kokoro_model = QComboBox()
        self.kokoro_model.addItems(["kokoro-v1.0.int8.onnx", "kokoro-v1.0.fp16.onnx",
                                    "kokoro-v1.0.onnx"])
        self.kokoro_model.setCurrentText(self.cfg.tts.kokoro_model)
        form.addRow("Kokoro model", self.kokoro_model)

        self.sapi_voice = QComboBox()
        self.sapi_voice.setEditable(True)
        self.sapi_voice.addItem("")
        self.sapi_voice.setCurrentText(self.cfg.tts.sapi_voice)
        self.sapi_probe = QPushButton("Detect Windows voices")
        self.sapi_probe.clicked.connect(self._probe_sapi)
        sapi_row = QHBoxLayout()
        sapi_row.addWidget(self.sapi_voice, 1)
        sapi_row.addWidget(self.sapi_probe)
        sapi_wrap = QWidget()
        sapi_wrap.setLayout(sapi_row)
        form.addRow("Windows voice", sapi_wrap)

        self.auto_speak = QCheckBox("Read answers aloud automatically")
        self.auto_speak.setChecked(bool(self.cfg.tts.auto_speak))
        form.addRow("", self.auto_speak)

        self.stream_sentences = QCheckBox("Start speaking sentence-by-sentence while generating")
        self.stream_sentences.setChecked(bool(self.cfg.tts.stream_sentences))
        form.addRow("", self.stream_sentences)

        self.rate_slider = QSlider(Qt.Orientation.Horizontal)
        self.rate_slider.setRange(50, 200)
        self.rate_slider.setValue(int(float(self.cfg.tts.rate) * 100))
        self.rate_label = QLabel(f"{float(self.cfg.tts.rate):.2f}x")
        self.rate_slider.valueChanged.connect(
            lambda v: self.rate_label.setText(f"{v / 100:.2f}x"))
        rate_row = QHBoxLayout()
        rate_row.addWidget(self.rate_slider, 1)
        rate_row.addWidget(self.rate_label)
        rate_wrap = QWidget()
        rate_wrap.setLayout(rate_row)
        form.addRow("Speaking rate", rate_wrap)

        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(int(float(self.cfg.tts.volume) * 100))
        self.volume_label = QLabel(f"{int(float(self.cfg.tts.volume) * 100)}%")
        self.volume_slider.valueChanged.connect(
            lambda v: self.volume_label.setText(f"{v}%"))
        vol_row = QHBoxLayout()
        vol_row.addWidget(self.volume_slider, 1)
        vol_row.addWidget(self.volume_label)
        vol_wrap = QWidget()
        vol_wrap.setLayout(vol_row)
        form.addRow("Volume", vol_wrap)

        self.min_chars = QSpinBox()
        self.min_chars.setRange(1, 300)
        self.min_chars.setValue(int(self.cfg.tts.min_speak_chars))
        form.addRow("Min chars per sentence", self.min_chars)

        self.strip_md = QCheckBox("Strip markdown before speaking")
        self.strip_md.setChecked(bool(self.cfg.tts.strip_markdown))
        form.addRow("", self.strip_md)

        test = QPushButton("Test voice")
        test.clicked.connect(self._test_voice)
        form.addRow("", test)
        layout.addWidget(box)
        layout.addStretch(1)
        self._on_engine_change(self.cfg.tts.engine)
        return page

    def _installed_piper_voices(self) -> list[str]:
        from ..tts_engines import installed_voices

        voices = installed_voices()["piper"]
        return voices or ["en_US-lessac-medium", "en_US-lessac-high", "en_GB-alan-medium",
                          "en_US-ryan-high"]

    def _installed_kokoro_voices(self) -> list[str]:
        return ["af_heart", "af_bella", "af_sky", "af_nicole", "af_nova", "am_adam",
                "am_michael", "am_echo", "bf_emma", "bm_george"]

    def _on_engine_change(self, engine: str) -> None:
        piper = engine == "piper"
        kokoro = engine == "kokoro"
        sapi = engine == "sapi"
        for widget, on in ((self.piper_voice, piper), (self.piper_get, piper),
                           (self.kokoro_voice, kokoro), (self.kokoro_model, kokoro),
                           (self.sapi_voice, sapi), (self.sapi_probe, sapi)):
            widget.setEnabled(on)

    def _download_piper_voice(self) -> None:
        from PySide6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(
            self, "Download a Piper voice",
            "Voice name (see the catalogue at huggingface.co/rhasspy/piper-voices):\n"
            "format is lang_REGION-name-quality, e.g. en_GB-alan-medium",
            text="en_GB-alan-medium")
        if not ok or not name.strip():
            return
        name = name.strip()
        self.piper_get.setEnabled(False)
        self.piper_get.setText(f"Downloading {name} ...")

        def worker() -> None:
            from .. import downloads

            try:
                downloads.ensure_piper_voice(name, _paths.piper_dir())
                result = ("ok", name)
            except Exception as exc:
                result = ("err", str(exc))
            self._voice_download_done(result)

        threading.Thread(target=worker, daemon=True).start()

    def _voice_download_done(self, result: tuple[str, str]) -> None:
        status, detail = result
        self.piper_get.setEnabled(True)
        self.piper_get.setText("Download another voice...")
        if status == "ok":
            items = self._installed_piper_voices()
            current = self.piper_voice.currentText()
            self.piper_voice.clear()
            self.piper_voice.addItems(items)
            self.piper_voice.setCurrentText(detail if detail in items else current)
            QMessageBox.information(self, "Voice", f"Installed {detail}.")
        else:
            QMessageBox.warning(self, "Voice", f"Download failed:\n{detail}")

    def _probe_sapi(self) -> None:
        def worker() -> None:
            from ..tts_engines import SapiTts

            try:
                names = SapiTts(self.cfg.tts).list_voices()
            except Exception as exc:
                names = [f"(error: {exc})"]
            self._sapi_result(names)

        threading.Thread(target=worker, daemon=True).start()

    def _sapi_result(self, names: list[str]) -> None:
        current = self.sapi_voice.currentText()
        self.sapi_voice.clear()
        self.sapi_voice.addItem("")
        self.sapi_voice.addItems(names)
        self.sapi_voice.setCurrentText(current)

    def _test_voice(self) -> None:
        """Synthesise + play a test sentence. Config is captured on the GUI thread."""
        self._on_apply(save=False)
        cfg = Config.from_dict(self.cfg.to_dict())      # snapshot: workers must not race
        text = ("This is your local voice. Everything you hear was synthesised "
                "on this computer.")

        def worker() -> None:
            from .. import audioio
            from ..tts_engines import create_engine

            try:
                engine = create_engine(cfg.tts, language=cfg.language)
                engine.load()
                result = engine.synthesize(text)
                audioio.play_clip(result.clip, backend="qt", blocking=True)
            except Exception as exc:
                self._voice_test_failed(str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _voice_test_failed(self, message: str) -> None:
        QMessageBox.warning(self, "Test voice", message)

    # ------------------------------------------------------------------ #
    # Audio devices
    # ------------------------------------------------------------------ #
    def _build_audio(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        box, form = _form_group("Devices")

        self.input_device = QComboBox()
        self.output_device = QComboBox()
        self.backend_label = QLabel("-")
        self.backend_label.setObjectName("hintLabel")
        rescan = QPushButton("Rescan devices")
        rescan.clicked.connect(self._scan_devices)
        self._scan_devices()

        form.addRow("Microphone", self.input_device)
        form.addRow("Speakers", self.output_device)
        form.addRow("Backend", self.backend_label)
        form.addRow("", rescan)

        self.mute_while_speaking = QCheckBox("Mute the microphone while speaking "
                                             "(prevents hearing itself)")
        self.mute_while_speaking.setChecked(bool(self.cfg.audio.mute_mic_while_speaking))
        form.addRow("", self.mute_while_speaking)

        self.sample_rate = QComboBox()
        self.sample_rate.addItems(["8000", "16000", "22050", "44100", "48000"])
        self.sample_rate.setCurrentText(str(self.cfg.audio.sample_rate))
        form.addRow("Capture rate", self.sample_rate)
        layout.addWidget(box)

        box2, form2 = _form_group("Hotkeys")
        self.hk_enabled = QCheckBox("Enable global hotkeys (work even when unfocused)")
        self.hk_enabled.setChecked(bool(self.cfg.hotkeys.enabled))
        form2.addRow("", self.hk_enabled)
        self.hk_ptt = QLineEdit(self.cfg.hotkeys.push_to_talk)
        form2.addRow("Push to talk", self.hk_ptt)
        self.hk_stop = QLineEdit(self.cfg.hotkeys.stop_speaking)
        form2.addRow("Stop speaking", self.hk_stop)
        hk_hint = QLabel("Format: Ctrl+Alt+Space, Ctrl+Shift+F9, ... "
                         "Leave blank to disable one.")
        hk_hint.setObjectName("hintLabel")
        form2.addRow("", hk_hint)
        layout.addWidget(box2)
        layout.addStretch(1)
        return page

    def _scan_devices(self) -> None:
        from .. import audioio

        self.input_device.clear()
        self.output_device.clear()
        self.input_device.addItem("(system default)", "")
        self.output_device.addItem("(system default)", "")
        try:
            backend = audioio.choose_backend("auto")
            devices = audioio.list_devices(backend)
        except Exception as exc:
            self.backend_label.setText(f"unavailable ({exc})")
            return
        self.backend_label.setText(f"{backend} - {len(devices)} device(s)")
        for d in devices:
            if d.kind == "input":
                self.input_device.addItem(d.label(), d.name)
            else:
                self.output_device.addItem(d.label(), d.name)
        self._select_device(self.input_device, self.cfg.audio.input_device)
        self._select_device(self.output_device, self.cfg.audio.output_device)

    @staticmethod
    def _select_device(combo: QComboBox, name: str) -> None:
        for i in range(combo.count()):
            if combo.itemData(i) == name:
                combo.setCurrentIndex(i)
                return
        combo.setCurrentIndex(0)

    # ------------------------------------------------------------------ #
    # Advanced
    # ------------------------------------------------------------------ #
    def _build_advanced(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        box, form = _form_group("Appearance")
        self.theme = QComboBox()
        self.theme.addItems(["dark", "light"])
        self.theme.setCurrentText(self.cfg.gui.theme)
        form.addRow("Theme", self.theme)
        self.font_size = QSpinBox()
        self.font_size.setRange(9, 20)
        self.font_size.setValue(int(self.cfg.gui.font_size))
        form.addRow("Font size", self.font_size)
        self.always_on_top = QCheckBox("Keep the window on top")
        self.always_on_top.setChecked(bool(self.cfg.gui.always_on_top))
        form.addRow("", self.always_on_top)
        self.minimize_to_tray = QCheckBox("Minimise to the system tray on close")
        self.minimize_to_tray.setChecked(bool(self.cfg.gui.minimize_to_tray))
        form.addRow("", self.minimize_to_tray)
        self.start_minimized = QCheckBox("Start minimised")
        self.start_minimized.setChecked(bool(self.cfg.gui.start_minimized))
        form.addRow("", self.start_minimized)
        layout.addWidget(box)

        box2, form2 = _form_group("Local speech API")
        self.server_enabled = QCheckBox("Run the local HTTP API alongside the GUI")
        self.server_enabled.setChecked(bool(self.cfg.server.enabled))
        form2.addRow("", self.server_enabled)
        self.server_host = QLineEdit(self.cfg.server.host)
        form2.addRow("Bind address", self.server_host)
        self.server_port = QSpinBox()
        self.server_port.setRange(1024, 65535)
        self.server_port.setValue(int(self.cfg.server.port))
        form2.addRow("Port", self.server_port)
        api_hint = QLabel("Serves OpenAI-compatible /v1/audio/transcriptions and "
                          "/v1/audio/speech on loopback only. No authentication - do not "
                          "expose it to your network.")
        api_hint.setObjectName("hintLabel")
        api_hint.setWordWrap(True)
        form2.addRow("", api_hint)
        layout.addWidget(box2)

        box3, form3 = _form_group("Paths")
        for label, value in _paths.describe().items():
            field = QLineEdit(str(value))
            field.setReadOnly(True)
            form3.addRow(label, field)
        open_btn = QPushButton("Open models folder")
        open_btn.clicked.connect(self._open_models_folder)
        form3.addRow("", open_btn)
        layout.addWidget(box3)
        layout.addStretch(1)
        return page

    @staticmethod
    def _open_models_folder() -> None:
        import subprocess
        import sys

        target = str(_paths.data_root())
        try:
            if sys.platform == "win32":
                subprocess.Popen(["explorer", target])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", target])
            else:
                subprocess.Popen(["xdg-open", target])
        except OSError:
            pass

    # ------------------------------------------------------------------ #
    # Collect / apply
    # ------------------------------------------------------------------ #
    def _collect(self) -> Config:
        cfg = Config.from_dict(self.cfg.to_dict())
        cfg.ollama.model = self.model_combo.currentText().strip()
        cfg.ollama.host = _normalise_url(self.host_edit.text(), self.cfg.ollama.host)
        cfg.ollama.temperature = float(self.temp_spin.value())
        cfg.ollama.num_ctx = int(self.ctx_spin.value())
        cfg.ollama.num_predict = int(self.predict_spin.value())
        cfg.ollama.history_turns = int(self.history_spin.value())
        cfg.ollama.system_prompt = self.system_prompt.toPlainText().strip()
        cfg.tier = self.tier_combo.currentText()

        cfg.stt.engine = self.stt_engine.currentText()
        cfg.stt.model = self.stt_model.currentText().strip()
        cfg.stt.language = self.stt_language.currentText().strip()
        cfg.language = cfg.stt.language if cfg.stt.language not in ("auto", "") else cfg.language
        cfg.stt.device = self.stt_device.currentText()
        cfg.stt.compute_type = self.stt_compute.currentText()
        cfg.stt.beam_size = int(self.beam_spin.value())
        cfg.stt.cpu_threads = int(self.threads_spin.value())
        cfg.stt.vad_filter = self.vad_filter.isChecked()
        cfg.stt.initial_prompt = self.stt_prompt.text().strip()

        cfg.vad.enabled = self.vad_enabled.isChecked()
        cfg.vad.mode = "hands_free" if cfg.vad.enabled else "push_to_talk"
        cfg.vad.silence_ms = int(self.silence_spin.value())
        cfg.vad.threshold_db = float(self.threshold_spin.value())
        cfg.vad.adaptive = self.adaptive_vad.isChecked()

        cfg.tts.engine = self.tts_engine.currentText()
        cfg.tts.piper_voice = self.piper_voice.currentText().strip()
        cfg.tts.kokoro_voice = self.kokoro_voice.currentText().strip()
        cfg.tts.kokoro_model = self.kokoro_model.currentText()
        cfg.tts.sapi_voice = self.sapi_voice.currentText().strip()
        cfg.tts.auto_speak = self.auto_speak.isChecked()
        cfg.tts.stream_sentences = self.stream_sentences.isChecked()
        # Clamped here as well as in the widget ranges: a config file edited by
        # hand must not be able to produce an unusable voice.
        cfg.tts.rate = round(min(2.0, max(0.5, _safe_int(self.rate_slider.value(), 100) / 100.0)), 2)
        cfg.tts.volume = round(min(1.0, max(0.0, _safe_int(self.volume_slider.value(), 100) / 100.0)), 2)
        cfg.tts.min_speak_chars = max(1, _safe_int(self.min_chars.value(), 24))
        cfg.tts.strip_markdown = self.strip_md.isChecked()

        cfg.audio.input_device = self.input_device.currentData() or ""
        cfg.audio.output_device = self.output_device.currentData() or ""
        cfg.audio.mute_mic_while_speaking = self.mute_while_speaking.isChecked()
        cfg.audio.sample_rate = _safe_int(self.sample_rate.currentText(),
                                          self.cfg.audio.sample_rate)

        cfg.hotkeys.enabled = self.hk_enabled.isChecked()
        cfg.hotkeys.push_to_talk = self.hk_ptt.text().strip()
        cfg.hotkeys.stop_speaking = self.hk_stop.text().strip()

        cfg.gui.theme = self.theme.currentText()
        cfg.gui.font_size = _safe_int(self.font_size.value(), self.cfg.gui.font_size)
        cfg.gui.always_on_top = self.always_on_top.isChecked()
        cfg.gui.minimize_to_tray = self.minimize_to_tray.isChecked()
        cfg.gui.start_minimized = self.start_minimized.isChecked()

        cfg.server.enabled = self.server_enabled.isChecked()
        cfg.server.host = self.server_host.text().strip() or self.cfg.server.host or "127.0.0.1"
        cfg.server.port = _safe_int(self.server_port.value(), self.cfg.server.port)

        cfg.gui.save_transcripts = self.save_transcripts.isChecked()
        return cfg

    def _on_apply(self, save: bool = True) -> None:
        new_cfg = self._collect()
        self.cfg.__dict__.clear()
        self.cfg.__dict__.update(new_cfg.__dict__)
        if save:
            from ..config import save as _save

            _save(self.cfg)
            self.config_changed.emit(self.cfg)

    def _on_ok(self) -> None:
        self._on_apply(save=True)
        self.accept()

    def reject(self) -> None:  # noqa: D102 - Qt override
        super().reject()

    # ------------------------------------------------------------------ #
    def result_config(self) -> Config:
        return self._collect()
