"""Configuration model for OllaVoice.

Stored as a single JSON document (see :mod:`ollavoice.paths` for the location).
The dataclasses below are the schema; ``load()`` merges the stored document over
the defaults so adding new options never breaks an existing config file.

``apply_tier()`` swaps in one of the low / balanced / high presets. The setup
wizard picks a tier automatically from the detected hardware, and you can change
it later from the GUI (Settings -> Performance) or with::

    ollavoice config set tier high
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from . import paths

CONFIG_VERSION = 1

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


@dataclass
class OllamaSettings:
    host: str = "http://127.0.0.1:11434"
    model: str = ""            # empty => chosen by the setup wizard / tier
    system_prompt: str = (
        "You are a helpful assistant running entirely on the user's own computer "
        "through Ollama. Your reply will be read aloud by a text-to-speech voice, "
        "so keep answers natural and conversational. Prefer plain prose: no markdown "
        "headings, no bullet symbols, no code blocks unless the user asks for code. "
        "Keep answers under 120 words unless the user asks for more detail."
    )
    temperature: float = 0.7
    num_ctx: int = 4096
    num_predict: int = -1          # -1 = model default
    keep_alive: str = "10m"
    api_key: str = ""              # only needed if you proxy Ollama; local server ignores it
    timeout_s: float = 600.0
    history_turns: int = 20        # how many turns of context to send (0 = stateless)


@dataclass
class SttSettings:
    engine: str = "faster-whisper"   # faster-whisper | whisper-cpp | ollama | windows
    model: str = "auto"              # "auto" => resolved from tier
    language: str = "auto"           # "auto", "en", "de", "zh", ...
    device: str = "auto"             # auto | cpu | cuda
    compute_type: str = "auto"       # auto | int8 | int8_float16 | float16 | float32
    cpu_threads: int = 0             # 0 => auto
    beam_size: int = 5
    vad_filter: bool = True
    condition_on_previous_text: bool = False
    initial_prompt: str = "Ollama, OllaVoice, Piper, Kokoro, Whisper."  # biases proper nouns
    translate_to_english: bool = False
    max_segment_s: float = 30.0


@dataclass
class TtsSettings:
    engine: str = "piper"            # piper | kokoro | sapi | none
    piper_voice: str = "auto"        # e.g. en_US-lessac-medium
    kokoro_voice: str = "af_heart"
    kokoro_model: str = "kokoro-v1.0.int8.onnx"
    sapi_voice: str = ""             # empty => Windows default installed voice
    rate: float = 1.0                # 0.5 (slow) .. 2.0 (fast)
    volume: float = 1.0
    auto_speak: bool = True          # read assistant replies aloud
    stream_sentences: bool = True    # speak sentence-by-sentence as tokens arrive
    #: A sentence shorter than this is merged with the next one before being
    #: spoken - single-word fragments sound stilted, but waiting for 100+
    #: characters adds audible latency, so the sweet spot is quite low.
    min_speak_chars: int = 24
    speak_own_messages: bool = False
    strip_markdown: bool = True


@dataclass
class AudioSettings:
    input_device: str = ""           # empty => system default microphone
    output_device: str = ""
    sample_rate: int = 16000         # mic capture rate (Whisper wants 16 kHz mono)
    channels: int = 1
    chunk_ms: int = 30
    mute_mic_while_speaking: bool = True   # crude echo cancellation
    record_format: str = "wav"


@dataclass
class VadSettings:
    enabled: bool = False            # hands-free end-of-speech detection
    mode: str = "push_to_talk"       # push_to_talk | hands_free
    silence_ms: int = 900            # stop after this much trailing silence
    min_speech_ms: int = 350         # ignore blips shorter than this
    max_record_s: float = 120.0
    threshold_db: float = -38.0      # absolute floor, used before noise floor is known
    adaptive: bool = True            # track the ambient noise floor
    adaptive_margin_db: float = 9.0
    pre_speech_pad_ms: int = 300     # keep audio from just before speech onset
    calibration_frames: int = 15     # ~0.45 s of room tone measured before deciding


@dataclass
class HotkeySettings:
    enabled: bool = True
    push_to_talk: str = "Ctrl+Alt+Space"
    stop_speaking: str = "Ctrl+Alt+X"
    toggle_mic: str = ""


@dataclass
class GuiSettings:
    theme: str = "dark"              # dark | light
    font_size: int = 12
    show_level_meter: bool = True
    start_minimized: bool = False
    minimize_to_tray: bool = True
    always_on_top: bool = False
    window_width: int = 980
    window_height: int = 700
    save_transcripts: bool = True
    show_partial_transcript: bool = True


@dataclass
class ServerSettings:
    enabled: bool = False
    host: str = "127.0.0.1"          # keep it loopback-only: this endpoint has no auth
    port: int = 8787


@dataclass
class Config:
    version: int = CONFIG_VERSION
    tier: str = "balanced"           # low | balanced | high | custom
    language: str = "en"
    ollama: OllamaSettings = field(default_factory=OllamaSettings)
    stt: SttSettings = field(default_factory=SttSettings)
    tts: TtsSettings = field(default_factory=TtsSettings)
    audio: AudioSettings = field(default_factory=AudioSettings)
    vad: VadSettings = field(default_factory=VadSettings)
    hotkeys: HotkeySettings = field(default_factory=HotkeySettings)
    gui: GuiSettings = field(default_factory=GuiSettings)
    server: ServerSettings = field(default_factory=ServerSettings)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Config":
        data = data or {}
        cfg = cls()
        for f in fields(cls):
            if f.name not in data:
                continue
            value = data[f.name]
            default = getattr(cfg, f.name)
            # Under `from __future__ import annotations` the declared types are
            # strings, so decide on the *default value* instead of the annotation.
            if is_dataclass(default) and isinstance(value, dict):
                setattr(cfg, f.name, _merge_dataclass(default, value))
            elif is_dataclass(default) or not isinstance(default, (str, int, float, bool, list, dict, type(None))):
                continue  # unknown shape in the file -> keep the default
            else:
                setattr(cfg, f.name, value)
        cfg.version = CONFIG_VERSION
        return cfg

    # ------------------------------------------------------------------ #
    def get(self, dotted: str) -> Any:
        obj: Any = self
        for part in dotted.split("."):
            if is_dataclass(obj):
                obj = getattr(obj, part)
            elif isinstance(obj, dict):
                obj = obj[part]
            else:
                raise KeyError(dotted)
        return obj

    def set(self, dotted: str, raw: str) -> Any:
        """Set ``section.option`` from a CLI string, coercing to the current type."""
        parts = dotted.split(".")
        obj: Any = self
        for part in parts[:-1]:
            obj = getattr(obj, part)
        leaf = parts[-1]
        current = getattr(obj, leaf)
        value: Any = _coerce(raw, current)
        setattr(obj, leaf, value)
        return value

    def save(self, path: Path | str | None = None) -> Path:
        path = Path(path) if path else paths.config_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(_jsonable(self.to_dict(), self), indent=2,
                                  ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        return path


_PRIMITIVES = (str, int, float, bool, type(None))

#: Every settings section, including the root. Compared with isinstance() rather
#: than duck-typed with is_dataclass(), which would accept any object exposing a
#: __dataclass_fields__ attribute and silently serialise it to {}.
_SECTION_TYPES = (Config, OllamaSettings, SttSettings, TtsSettings, AudioSettings,
                  VadSettings, HotkeySettings, GuiSettings, ServerSettings)


def _section_defaults(section: Any) -> dict[str, Any]:
    """Declared defaults for a settings section.

    The *declared* default, not the current value: the current value is exactly
    what may be corrupted, so it cannot be used as the fallback.
    """
    out: dict[str, Any] = {}
    for f in fields(section):
        if f.default is not MISSING:
            out[f.name] = f.default
        elif f.default_factory is not MISSING:      # type: ignore[misc]
            try:
                out[f.name] = f.default_factory()   # type: ignore[misc]
            except TypeError:
                out[f.name] = None
        else:
            out[f.name] = None
    return out


def _jsonable(value: Any, template: Any) -> Any:
    """Coerce a config tree to plain JSON values before writing it.

    A GUI widget or a bad merge can hand back something that is not serialisable.
    Rather than crash - or worse, write a file that will not load next time - fall
    back to the *declared default* for that one field.
    """
    # Compare against our own known section classes: duck-typing with
    # is_dataclass() would accept any object that happens to expose a
    # __dataclass_fields__ attribute, which would silently serialise to {}.
    if isinstance(template, _SECTION_TYPES):
        defaults = _section_defaults(template)

        def _field(name: str) -> Any:
            if isinstance(value, dict):
                return value.get(name, defaults.get(name))
            return getattr(value, name, defaults.get(name))

        return {name: _jsonable(_field(name), default)
                for name, default in defaults.items()}
    if isinstance(value, bool):
        return value
    if isinstance(value, _PRIMITIVES):
        if isinstance(template, bool) and not isinstance(value, bool):
            return bool(template)
        if isinstance(template, int) and not isinstance(template, bool) \
                and not isinstance(value, int):
            return template
        if isinstance(template, float) and not isinstance(value, (int, float)):
            return template
        if isinstance(template, str) and not isinstance(value, str):
            return template
        return value
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, _PRIMITIVES)]
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items() if isinstance(v, _PRIMITIVES)}
    # Not a JSON value at all: fall back to the declared default, and if that is
    # somehow not serialisable either, use None so the write still succeeds.
    return template if isinstance(template, _PRIMITIVES) else None


def _coerce(raw: str, current: Any) -> Any:
    if isinstance(current, bool):
        return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


def _merge_dataclass(obj: Any, data: dict[str, Any]) -> Any:
    out = copy.deepcopy(obj)
    for f in fields(out):
        if f.name in data:
            setattr(out, f.name, data[f.name])
    return out


# --------------------------------------------------------------------------- #
# Tier presets
# --------------------------------------------------------------------------- #
# Sizes are the download footprint of the speech models only (the Ollama LLM is
# pulled separately). "low" is tuned for a modest laptop with no GPU; "high"
# assumes a discrete GPU with >= 8 GB VRAM.

TIERS: dict[str, dict[str, Any]] = {
    "low": {
        "ollama.model": "qwen3:4b",
        "stt.model": "base.en",
        "stt.beam_size": 1,
        "stt.device": "cpu",
        "stt.compute_type": "int8",
        "tts.engine": "piper",
        "tts.piper_voice": "en_US-lessac-medium",
        "tts.stream_sentences": True,
        "ollama.num_ctx": 2048,
        "_approx_download_mb": 320,
        "_note": "CPU-only, ~320 MB of speech models. Fastest to first run.",
    },
    "balanced": {
        "ollama.model": "qwen3:8b",
        "stt.model": "small.en",
        "stt.beam_size": 5,
        "stt.device": "auto",
        "stt.compute_type": "auto",
        "tts.engine": "piper",
        "tts.piper_voice": "en_US-lessac-high",
        "tts.stream_sentences": True,
        "ollama.num_ctx": 4096,
        "_approx_download_mb": 700,
        "_note": "~700 MB of speech models. Good accuracy, still snappy on CPU.",
    },
    "high": {
        "ollama.model": "qwen3:14b",
        "stt.model": "medium.en",
        "stt.beam_size": 5,
        "stt.device": "auto",
        "stt.compute_type": "auto",
        "tts.engine": "kokoro",
        "tts.kokoro_voice": "af_heart",
        "tts.stream_sentences": True,
        "ollama.num_ctx": 8192,
        "_approx_download_mb": 1700,
        "_note": "~1.7 GB of speech models. Best transcription + most natural voice.",
    },
}

TIER_ORDER = ("low", "balanced", "high")


def apply_tier(cfg: Config, tier: str, *, language: str | None = None) -> Config:
    """Apply a preset tier. ``language`` overrides the English-only model choice."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; choose from {', '.join(TIER_ORDER)}")
    cfg.tier = tier
    lang = language or cfg.language
    cfg.language = lang
    for dotted, value in TIERS[tier].items():
        if dotted.startswith("_"):
            continue
        if dotted == "stt.model":
            value = resolve_whisper_model(str(value), effective_language(lang, cfg.stt.language))
        if dotted in ("tts.piper_voice",):
            value = localize_piper_voice(str(value), lang)
        cfg.set(dotted, value) if isinstance(value, str) else _set_typed(cfg, dotted, value)
    return cfg


def _set_typed(cfg: Config, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    obj = cfg
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)


def localize_whisper_model(model: str, language: str) -> str:
    """``base.en`` only exists for English; fall back to multilingual otherwise.

    "auto" counts as not-English here: an English-only checkpoint cannot detect
    the language, so it would silently mis-transcribe anything else.
    """
    if language in ("en", "english"):
        return model
    return model.replace(".en", "")


def resolve_whisper_model(model: str, language: str) -> str:
    """Like :func:`localize_whisper_model` but also expands ``auto`` model names."""
    model = (model or "auto").strip() or "auto"
    if model == "auto":
        model = "base.en"
    return localize_whisper_model(model, language)


def effective_language(*candidates: str) -> str:
    """First language code that is actually a choice, ignoring ``auto``/empty.

    ``stt.language`` (what Whisper should expect) wins over the app-wide
    ``language`` (which also drives TTS voice selection).
    """
    for value in candidates:
        if value and str(value).strip().lower() not in ("auto", ""):
            return str(value).strip()
    return "auto"

def localize_piper_voice(voice: str, language: str) -> str:
    """Map a language code onto a sensible default Piper voice, if we know one."""
    if language in ("en", "english", ""):
        return voice
    defaults = {
        "de": "de_DE-thorsten-high",
        "fr": "fr_FR-siwis-high",
        "es": "es_ES-davefx-medium",
        "it": "it_IT-riccardo_fabl-high",
        "pt": "pt_BR-faber-medium",
        "nl": "nl_NL-merijn-medium",
        "pl": "pl_PL-darkman-medium",
        "ru": "ru_RU-irina-medium",
        "uk": "uk_UA-lada-x_low",
        "zh": "zh_CN-huayan-medium",
        "ja": "ja_JP-krita-medium",
        "ko": "ko_KR-jc-medium",
        "ar": "ar_JO-kareem-medium",
        "tr": "tr_TR-dfki-medium",
        "sv": "sv_SE-nst-medium",
        "no": "nb_NO-talesyntese-medium",
        "da": "da_DK-talesyntese-medium",
        "fi": "fi_FI-harri-medium",
        "el": "el_GR-rapunzelina-low",
        "cs": "cs_CZ-jirka-medium",
        "vi": "vi_VN-25hours_single-low",
        "fa": "fa_IR-amir-medium",
        "hu": "hu_HU-berta-medium",
    }
    code = language.split("-")[0].split("_")[0].lower()
    return defaults.get(code, voice)


# --------------------------------------------------------------------------- #
# Load / save helpers
# --------------------------------------------------------------------------- #


def default_config() -> Config:
    cfg = Config()
    return apply_tier(cfg, "balanced")


def load(path: Path | str | None = None, *, create_if_missing: bool = True) -> Config:
    path = Path(path) if path else paths.config_file()
    if path.exists():
        try:
            return Config.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            backup = path.with_suffix(".corrupt.json")
            try:
                os.replace(path, backup)
                print(f"[ollavoice] config unreadable ({exc}); moved to {backup}", flush=True)
            except OSError:
                pass
    cfg = default_config()
    if create_if_missing:
        try:
            cfg.save(path)
        except OSError:
            pass
    return cfg


def save(cfg: Config, path: Path | str | None = None) -> Path:
    return cfg.save(path)
