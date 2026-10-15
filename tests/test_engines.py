"""Engine factories, voice resolution and the text preparation shared by TTS."""

from __future__ import annotations

import os
import sys

import pytest

from ollavoice import stt_engines, tts_engines
from ollavoice.config import SttSettings, TtsSettings


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #

def test_stt_factory_knows_every_engine():
    for name in stt_engines.ENGINES:
        settings = SttSettings(engine=name)
        engine = stt_engines.create_engine(settings)
        assert engine.name == name or name in engine.name


def test_stt_factory_normalises_whisper_cpp_spellings():
    for spelling in ("whisper-cpp", "whisper_cpp", "whispercpp", "whisper.cpp"):
        engine = stt_engines.create_engine(SttSettings(engine=spelling))
        assert engine.name == "whisper-cpp"


def test_stt_factory_rejects_nonsense():
    with pytest.raises(stt_engines.SttError) as exc:
        stt_engines.create_engine(SttSettings(engine="deepgram-cloud"))
    assert "faster-whisper" in str(exc.value)      # the message lists the real options


def test_tts_factory_rejects_nonsense():
    with pytest.raises(tts_engines.TtsError):
        tts_engines.create_engine(TtsSettings(engine="elevenlabs"))


@pytest.mark.parametrize("engine", ["piper", "kokoro", "sapi", "none", "silent", "off", ""])
def test_tts_factory_accepts_known_names(engine):
    created = tts_engines.create_engine(TtsSettings(engine=engine))
    assert created.name in ("piper", "kokoro", "sapi", "none")


def test_availability_reports_are_well_formed():
    for report in stt_engines.available_engines() + tts_engines.available_engines():
        assert {"name", "available", "hint", "description"} <= set(report)
        assert isinstance(report["available"], bool)
        assert report["description"]


def test_faster_whisper_is_the_recommended_default():
    reports = {e["name"]: e for e in stt_engines.available_engines()}
    assert "recommended" in reports["faster-whisper"]["description"].lower()


# --------------------------------------------------------------------------- #
# Whisper model naming
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("configured,language,expected", [
    ("base.en", "en", "base.en"),
    ("base.en", "auto", "base"),        # detection needs the multilingual model
    ("small.en", "de", "small"),
    ("large-v3", "zh", "large-v3"),
    ("auto", "en", "base.en"),
    ("auto", "auto", "base"),
    ("", "en", "base.en"),
])
def test_model_resolution_follows_the_language(configured, language, expected):
    engine = stt_engines.FasterWhisperStt(SttSettings(model=configured, language=language))
    assert engine.model_name == expected


def test_whisper_device_falls_back_to_cpu_without_cuda():
    engine = stt_engines.FasterWhisperStt(SttSettings(device="cuda", compute_type="auto"))
    # No CUDA in CI: the engine must quietly downgrade rather than explode later.
    assert engine.device in ("cpu", "cuda")
    if engine.device == "cpu":
        assert engine.compute_type in ("int8", "int8_float16")


# --------------------------------------------------------------------------- #
# Piper voice resolution
# --------------------------------------------------------------------------- #

def test_piper_prefers_an_installed_voice(tmp_home, monkeypatch):
    piper_dir = tmp_home / "models" / "piper"
    piper_dir.mkdir(parents=True)
    (piper_dir / "en_GB-alan-medium.onnx").write_bytes(b"x")
    (piper_dir / "en_GB-alan-medium.onnx.json").write_bytes(b"{}")

    engine = tts_engines.PiperTts(TtsSettings(piper_voice="auto"), language="en",
                                  auto_download=False)
    assert engine.voice_name == "en_GB-alan-medium"


def test_piper_prefers_a_voice_matching_the_language(tmp_home):
    piper_dir = tmp_home / "models" / "piper"
    piper_dir.mkdir(parents=True)
    for name in ("en_US-lessac-high", "de_DE-thorsten-high"):
        (piper_dir / f"{name}.onnx").write_bytes(b"x")
        (piper_dir / f"{name}.onnx.json").write_bytes(b"{}")

    engine = tts_engines.PiperTts(TtsSettings(piper_voice="auto"), language="de",
                                  auto_download=False)
    assert engine.voice_name == "de_DE-thorsten-high"


def test_piper_auto_falls_back_to_the_english_default(tmp_home):
    engine = tts_engines.PiperTts(TtsSettings(piper_voice="auto"), language="en",
                                  auto_download=False)
    assert engine.voice_name == "en_US-lessac-medium"


def test_piper_missing_voice_without_download_gives_a_command(tmp_home):
    engine = tts_engines.PiperTts(TtsSettings(piper_voice="en_US-lessac-high"),
                                  language="en", auto_download=False)
    with pytest.raises(tts_engines.TtsError) as exc:
        engine.load()
    assert "ollavoice setup" in str(exc.value)


def test_piper_set_voice_invalidates_the_cached_model(tmp_home):
    engine = tts_engines.PiperTts(TtsSettings(piper_voice="auto"), language="en",
                                  auto_download=False)
    engine._voice = object()
    engine._loaded = True
    engine.set_voice("en_GB-cori-high")
    assert engine.voice_name == "en_GB-cori-high"
    assert engine._voice is None and not engine._loaded
    # Setting the same voice again must not throw the cache away.
    engine._voice = object()
    engine.set_voice("en_GB-cori-high")
    assert engine._voice is not None


# --------------------------------------------------------------------------- #
# Kokoro
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("voice,expected", [
    ("af_heart", "en-us"),
    ("am_michael", "en-us"),
    ("bf_emma", "en-gb"),
    ("zf_xiaobei", "zh"),
    ("jf_alpha", "ja"),
    ("ef_dora", "es"),
    ("hf_alpha", "hi"),
    ("if_sara", "it"),
    ("pf_driella", "pt-br"),
    ("ff_siwis", "fr"),
    ("nonsense", "en-us"),
])
def test_kokoro_language_follows_the_voice_prefix(voice, expected):
    engine = tts_engines.KokoroTts(TtsSettings(kokoro_voice=voice), auto_download=False)
    assert engine._lang() == expected


def test_kokoro_missing_model_gives_a_command(tmp_home):
    engine = tts_engines.KokoroTts(TtsSettings(kokoro_model="kokoro-v1.0.int8.onnx"),
                                   auto_download=False)
    with pytest.raises(tts_engines.TtsError) as exc:
        engine.load()
    assert "ollavoice setup" in str(exc.value)


def test_kokoro_voice_list_without_weights_is_a_sane_default(tmp_home):
    engine = tts_engines.KokoroTts(TtsSettings(), auto_download=False)
    voices = engine.list_voices()
    assert "af_heart" in voices and len(voices) >= 5


# --------------------------------------------------------------------------- #
# SAPI / silent
# --------------------------------------------------------------------------- #

def test_sapi_refuses_to_load_off_windows(tmp_home):
    engine = tts_engines.SapiTts(TtsSettings())
    if sys.platform != "win32":
        with pytest.raises(tts_engines.TtsError):
            engine.load()
        assert engine.list_voices() == []
    else:
        engine.load()


def test_sapi_powershell_scripts_are_syntactically_balanced():
    """A stray brace here would only show up on a user's Windows machine."""
    for script in (tts_engines.SapiTts.PS_LIST, tts_engines.SapiTts.PS_SPEAK,
                   stt_engines.PS_TRANSCRIBE):
        assert script.count("{") == script.count("}"), script
        assert script.count("(") == script.count(")")


def test_silent_engine_produces_no_audio():
    engine = tts_engines.SilentTts(TtsSettings())
    result = engine.synthesize("hello")
    assert len(result.clip.samples) == 0
    assert list(engine.synthesize_stream("hello")) == []
    assert engine.voice_name == "(silent)"


# --------------------------------------------------------------------------- #
# Text preparation shared by every TTS engine
# --------------------------------------------------------------------------- #

def test_tts_prepare_text_strips_markdown():
    engine = tts_engines.SilentTts(TtsSettings(strip_markdown=True))
    out = engine._prepare_text("## Heading\n\n- a **bold** item\n- `code`")
    assert "#" not in out and "*" not in out and "`" not in out
    assert "bold" in out


def test_tts_prepare_text_can_keep_markdown():
    engine = tts_engines.SilentTts(TtsSettings(strip_markdown=False))
    assert engine._prepare_text("**keep**") == "**keep**"


def test_tts_prepare_text_expands_symbols():
    engine = tts_engines.SilentTts(TtsSettings())
    assert "percent" in engine._prepare_text("It is 50% faster")


def test_installed_voices_shape(tmp_home):
    info = tts_engines.installed_voices()
    assert set(info) == {"piper", "kokoro", "kokoro_voices", "piper_size_mb", "kokoro_size_mb"}
    assert info["piper"] == [] and info["kokoro"] == []


def test_base_engine_describe():
    engine = tts_engines.SilentTts(TtsSettings())
    assert engine.describe() == {"engine": "none", "voice": "(silent)", "loaded": False}

    class _Minimal(stt_engines.SttEngine):
        name = "minimal"

        def transcribe(self, clip, *, language=None, initial_prompt=None):
            return stt_engines.Transcription(text="", engine=self.name)

    stt = _Minimal(SttSettings())
    assert stt.describe() == {"engine": "minimal", "model": "", "loaded": False}
    stt.load()
    assert stt.loaded and stt.describe()["loaded"] is True


# --------------------------------------------------------------------------- #
# Transcript clean-up
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("raw,expected", [
    ("[BLANK_AUDIO]", ""),
    ("  Thanks for watching!  ", ""),
    ("(blank audio)", ""),
    ("Hello there", "Hello there"),
    ("", ""),
])
def test_whisper_hallucinations_are_stripped(raw, expected):
    assert stt_engines._clean_transcript(raw) == expected


def test_segment_joining_does_not_double_spaces():
    joined = stt_engines._join_segments(["Hello", " world", "again.", " Yes"])
    assert "  " not in joined
    assert joined.startswith("Hello")


def test_transcription_realtime_factor():
    tr = stt_engines.Transcription(text="x", duration_s=4.0, decode_s=1.0)
    assert tr.realtime_factor == 0.25
    assert "4.0s audio" in tr.summary()
    assert stt_engines.Transcription(text="x").realtime_factor == 0.0


def test_as_audio_source_handles_every_input_type(tmp_path):
    from ollavoice.wav import float_to_clip

    made = float_to_clip([0.1, 0.2, 0.3], 16000)
    source, cleanup = stt_engines._as_audio_source(made)
    assert cleanup and source.exists()
    source.unlink()

    wav_path = tmp_path / "a.wav"
    made.write_wav(wav_path)
    source, cleanup = stt_engines._as_audio_source(wav_path)
    assert not cleanup and source == wav_path

    source, cleanup = stt_engines._as_audio_source(made.to_wav_bytes())
    assert cleanup and source.exists()
    source.unlink()

    import io
    source, cleanup = stt_engines._as_audio_source(b"not-a-wav-file")
    assert not cleanup and isinstance(source, io.BytesIO)

    with pytest.raises(stt_engines.SttError):
        stt_engines._as_audio_source(tmp_path / "missing.wav")


# --------------------------------------------------------------------------- #
# Optional engines: whisper.cpp, Ollama-hosted, Windows
# --------------------------------------------------------------------------- #

def test_whisper_cpp_discovery_uses_env_and_folder(tmp_home, monkeypatch):
    engine = stt_engines.WhisperCppStt(SttSettings(engine="whisper-cpp"))
    assert engine.binary == "" and engine.model_path == ""

    bin_dir = tmp_home / "bin"
    bin_dir.mkdir()
    fake_bin = bin_dir / ("whisper-cli.exe" if sys.platform == "win32" else "whisper-cli")
    fake_bin.write_text("#!/bin/sh\n")
    fake_bin.chmod(0o755)
    model = tmp_home / "models" / "whisper.cpp" / "ggml-base.en.bin"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"weights")

    monkeypatch.setenv("OLLAVOICE_WHISPERCPP_BIN", str(fake_bin))
    monkeypatch.setenv("OLLAVOICE_WHISPERCPP_MODEL", str(model))
    engine = stt_engines.WhisperCppStt(SttSettings(engine="whisper-cpp"))
    assert engine.binary == str(fake_bin)
    assert engine.model_path == str(model)
    engine.load()
    assert engine.loaded


def test_whisper_cpp_load_error_points_at_the_release_page(tmp_home):
    engine = stt_engines.WhisperCppStt(SttSettings(engine="whisper-cpp"))
    with pytest.raises(stt_engines.SttError) as exc:
        engine.load()
    assert "whisper.cpp" in str(exc.value)
    assert "releases" in str(exc.value)


def test_ollama_engine_explains_that_ollama_has_no_stt_api(fake_ollama):
    base, _ = fake_ollama
    engine = stt_engines.OllamaStt(SttSettings(engine="ollama", model="whisper"), host=base)
    from ollavoice.wav import float_to_clip

    clip = float_to_clip([0.0] * 1600, 16000)
    with pytest.raises(stt_engines.SttError) as exc:
        engine.transcribe(clip)
    message = str(exc.value)
    assert "no native speech-to-text" in message.lower() or "faster-whisper" in message


def test_windows_engine_refuses_to_load_off_windows():
    engine = stt_engines.WindowsStt(SttSettings(engine="windows"))
    if sys.platform != "win32":
        with pytest.raises(stt_engines.SttError):
            engine.load()


def test_bcp47_mapping():
    assert stt_engines._bcp47("en") == "en-US"
    assert stt_engines._bcp47("zh") == "zh-CN"
    assert stt_engines._bcp47("en-GB") == "en-GB"
    assert stt_engines._bcp47("") == ""
    assert stt_engines._bcp47("klingon") == "klingon"


# --------------------------------------------------------------------------- #
# Offline behaviour: a downloaded model must not cause a network check
# --------------------------------------------------------------------------- #

def test_offline_cache_is_used_when_the_model_is_already_downloaded(tmp_home, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)

    model_dir = tmp_home / "models" / "whisper" / "models--Systran--faster-whisper-base.en"
    model_dir.mkdir(parents=True)
    (model_dir / "model.bin").write_bytes(b"weights")

    stt_engines._prepare_offline_hf_cache("base.en")
    assert os.environ.get("HF_HUB_OFFLINE") == "1", "a cached model must not phone home"


def test_offline_cache_is_not_forced_when_the_model_is_missing(tmp_home, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    stt_engines._prepare_offline_hf_cache("large-v3")
    assert os.environ.get("HF_HUB_OFFLINE") is None


def test_vendor_bundle_takes_priority(tmp_home, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    bundle = tmp_home / "vendor" / "hf-cache"
    bundle.mkdir(parents=True)
    stt_engines._prepare_offline_hf_cache("base.en")
    assert os.environ.get("HF_HOME") == str(bundle)
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
