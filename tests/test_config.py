"""Config loading, tier presets and dotted get/set."""

from __future__ import annotations

import json

from ollavoice.config import (
    TIERS, Config, apply_tier, default_config, load, localize_piper_voice,
    localize_whisper_model, save,
)


def test_defaults_are_consistent():
    cfg = default_config()
    assert cfg.tier == "balanced"
    assert cfg.ollama.model
    assert cfg.stt.model
    assert cfg.tts.engine == "piper"


def test_every_tier_produces_a_valid_config():
    for tier in TIERS:
        cfg = apply_tier(Config(), tier)
        assert cfg.tier == tier
        assert cfg.ollama.model
        assert cfg.stt.model
        assert cfg.tts.engine in ("piper", "kokoro", "sapi", "none")
        assert 0 < cfg.tts.rate <= 2.0


def test_tier_overrides_are_writable_keys():
    """A typo in TIERS would silently do nothing - make sure every key resolves."""
    cfg = Config()
    for tier, preset in TIERS.items():
        for dotted in preset:
            if dotted.startswith("_"):
                continue
            cfg.set(dotted, str(preset[dotted]))   # raises if the key does not exist


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "config.json"
    cfg = apply_tier(Config(), "high")
    cfg.tts.rate = 1.25
    cfg.ollama.system_prompt = "Be brief."
    save(cfg, path)

    again = load(path)
    assert again.tier == "high"
    assert again.tts.rate == 1.25
    assert again.ollama.system_prompt == "Be brief."
    assert json.loads(path.read_text())["version"] == cfg.version


def test_unknown_keys_in_file_are_ignored(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "tier": "low",
        "stt": {"model": "tiny.en", "from_the_future": True},
        "something_new": {"a": 1},
    }))
    cfg = load(path)
    assert cfg.stt.model == "tiny.en"
    assert cfg.ollama.host.startswith("http://127.0.0.1")   # default survived


def test_corrupt_config_is_replaced(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{not json at all")
    cfg = load(path)
    assert cfg.tier == "balanced"
    assert path.with_suffix(".corrupt.json").exists()


def test_dotted_get_set_and_coercion():
    cfg = Config()
    cfg.set("tts.rate", "1.35")
    assert cfg.get("tts.rate") == 1.35
    cfg.set("tts.auto_speak", "false")
    assert cfg.get("tts.auto_speak") is False
    cfg.set("ollama.num_ctx", "8192")
    assert cfg.get("ollama.num_ctx") == 8192
    cfg.set("gui.theme", "light")
    assert cfg.get("gui.theme") == "light"


def test_language_localisation():
    assert localize_whisper_model("base.en", "en") == "base.en"
    assert localize_whisper_model("base.en", "de") == "base"
    assert localize_whisper_model("small.en", "zh") == "small"
    assert localize_piper_voice("en_US-lessac-medium", "en") == "en_US-lessac-medium"
    assert localize_piper_voice("en_US-lessac-medium", "de") == "de_DE-thorsten-high"
    assert localize_piper_voice("en_US-lessac-medium", "zh") == "zh_CN-huayan-medium"
    # Unknown language keeps the English voice rather than inventing one.
    assert localize_piper_voice("en_US-lessac-medium", "xx") == "en_US-lessac-medium"


def test_apply_tier_respects_language():
    cfg = apply_tier(Config(), "low", language="fr")
    assert cfg.stt.model == "base"          # multilingual variant
    assert cfg.tts.piper_voice.startswith("fr_")


def test_apply_tier_keeps_the_english_checkpoint_for_english():
    cfg = apply_tier(Config(), "low", language="en")
    assert cfg.stt.model == "base.en"
    assert cfg.tts.piper_voice.startswith("en_")


def test_effective_language_prefers_an_explicit_choice():
    from ollavoice.config import effective_language

    assert effective_language("auto", "en") == "en"
    assert effective_language("de", "en") == "de"
    assert effective_language("", "auto") == "auto"
    assert effective_language() == "auto"
    assert effective_language(" EN ") == "EN"


def test_resolve_whisper_model():
    from ollavoice.config import resolve_whisper_model

    assert resolve_whisper_model("auto", "en") == "base.en"
    assert resolve_whisper_model("", "auto") == "base"
    assert resolve_whisper_model("small.en", "zh") == "small"
    assert resolve_whisper_model("large-v3", "en") == "large-v3"


def test_saving_a_corrupted_value_falls_back_to_the_default(tmp_path):
    """A widget (or a bad merge) must never be able to write an unloadable file."""

    class Junk:
        pass

    cfg = default_config()
    cfg.tts.rate = Junk()                    # not a number
    cfg.ollama.model = None                  # not a string
    cfg.gui.theme = 42                       # wrong type entirely
    path = tmp_path / "config.json"
    cfg.save(path)

    raw = json.loads(path.read_text())
    assert isinstance(raw["tts"]["rate"], float)
    assert raw["tts"]["rate"] == default_config().tts.rate
    # A None model falls back to the schema default ("" = "not chosen yet"),
    # which is loadable and prompts the user rather than breaking startup.
    assert isinstance(raw["ollama"]["model"], str)
    assert isinstance(raw["gui"]["theme"], str) and raw["gui"]["theme"] == "dark"

    reloaded = load(path)
    assert reloaded.tts.rate == default_config().tts.rate
    assert reloaded.gui.theme == "dark"


def test_jsonable_keeps_valid_values_untouched():
    from ollavoice.config import _jsonable

    cfg = default_config()
    cfg.tts.rate = 1.35
    cfg.ollama.history_turns = 7
    cfg.tts.auto_speak = False
    out = _jsonable(cfg.to_dict(), cfg)
    assert out["tts"]["rate"] == 1.35
    assert out["ollama"]["history_turns"] == 7
    assert out["tts"]["auto_speak"] is False
