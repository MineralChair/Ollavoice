"""CLI surface tests: argument parsing, config round-trips, doctor, help text.

These run the real ``ollavoice`` entry point in-process (fast) and, for a couple
of cases, as a subprocess (to prove the console script works after install).
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from ollavoice import cli

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(argv: list[str]) -> tuple[int, str]:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = cli.main(argv)
    return int(code or 0), buffer.getvalue()


# --------------------------------------------------------------------------- #
# Top level
# --------------------------------------------------------------------------- #

def test_no_command_prints_a_quick_start(tmp_home):
    code, out = run([])
    assert code == 0
    assert "ollavoice setup" in out and "ollavoice gui" in out


def test_version_flag(tmp_home):
    with pytest.raises(SystemExit) as exc:
        run(["--version"])
    assert exc.value.code == 0


def test_help_lists_every_command(tmp_home):
    """Every subcommand must be discoverable from --help."""
    parser = cli.build_parser()
    out = parser.format_help()
    for command in ("chat", "gui", "setup", "doctor", "listen", "transcribe",
                    "speak", "voices", "models", "serve", "config", "bundle"):
        assert command in out, f"'{command}' is missing from --help"
    assert "no cloud API" in out
    # ...and each subcommand has its own help text.
    for command in ("chat", "setup", "doctor", "speak"):
        sub = cli.build_parser()
        with pytest.raises(SystemExit):
            sub.parse_args([command, "--help"])


def test_unknown_command_exits_nonzero(tmp_home):
    with pytest.raises(SystemExit) as exc:
        run(["definitely-not-a-command"])
    assert exc.value.code != 0


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #

def test_config_path_points_into_the_app_dir(tmp_home):
    code, out = run(["config", "path"])
    assert code == 0
    assert str(tmp_home) in out.strip()
    assert out.strip().endswith("config.json")


def test_config_show_is_valid_json(tmp_home):
    code, out = run(["config", "show"])
    assert code == 0
    data = json.loads(out)
    assert {"ollama", "stt", "tts", "audio", "vad", "gui", "server"} <= set(data)


def test_config_get_single_key(tmp_home):
    run(["config", "set", "tts.rate", "1.4"])
    code, out = run(["config", "get", "tts.rate"])
    assert code == 0 and out.strip() == "1.4"


def test_config_set_persists_and_coerces_types(tmp_home):
    run(["config", "set", "ollama.num_ctx", "8192"])
    run(["config", "set", "tts.auto_speak", "false"])
    run(["config", "set", "gui.theme", "light"])
    code, out = run(["config", "show"])
    data = json.loads(out)
    assert data["ollama"]["num_ctx"] == 8192
    assert data["tts"]["auto_speak"] is False
    assert data["gui"]["theme"] == "light"


def test_config_set_rejects_unknown_keys(tmp_home):
    code, out = run(["config", "set", "tts.not_a_real_option", "1"])
    assert code == 2
    assert "unknown key" in out


def test_config_tier_switch_updates_the_models(tmp_home):
    run(["config", "--tier", "high"])
    code, out = run(["config", "show"])
    data = json.loads(out)
    assert data["tier"] == "high"
    assert data["stt"]["model"].startswith("medium")
    assert data["tts"]["engine"] == "kokoro"

    run(["config", "--tier", "low"])
    _, out = run(["config", "show"])
    data = json.loads(out)
    assert data["tier"] == "low"
    assert data["stt"]["model"].startswith("base")
    assert data["tts"]["engine"] == "piper"


def test_config_reset_restores_defaults(tmp_home):
    run(["config", "set", "gui.theme", "light"])
    run(["config", "reset"])
    _, out = run(["config", "show"])
    assert json.loads(out)["gui"]["theme"] == "dark"


def test_alternate_config_file_is_honoured(tmp_path, tmp_home):
    alt = tmp_path / "custom.json"
    run(["--config", str(alt), "config", "set", "tts.rate", "1.8"])
    assert alt.exists()
    assert json.loads(alt.read_text())["tts"]["rate"] == 1.8
    # ...and the default file was left alone.
    _, out = run(["config", "get", "tts.rate"])
    assert out.strip() != "1.8"


# --------------------------------------------------------------------------- #
# doctor / models / voices
# --------------------------------------------------------------------------- #

def test_doctor_reports_every_section(tmp_home):
    code, out = run(["doctor"])
    assert code == 0
    for section in ("[hardware]", "[ollama]", "[speech-to-text]", "[text-to-speech]",
                    "[paths]"):
        assert section in out
    assert "OllaVoice" in out


def test_doctor_json_is_parseable(tmp_home):
    code, out = run(["doctor", "--json"])
    data = json.loads(out)
    assert data["version"]
    assert data["recommended_tier"] in ("low", "balanced", "high")
    assert data["paths"]["models"]
    assert isinstance(data["stt"]["engines"], list)


def test_doctor_detects_a_dead_ollama(tmp_home):
    _, out = run(["doctor"])
    assert "OFFLINE" in out


def test_doctor_test_fails_cleanly_without_models(tmp_home):
    """--test must report failures rather than traceback."""
    code, out = run(["doctor", "--test"])
    assert code == 1
    assert "[self-test]" in out
    assert "fail" in out.lower()


def test_models_command_handles_a_dead_server(tmp_home):
    code, out = run(["models"])
    assert code == 2
    assert "ollama serve" in out or "Cannot reach" in out


def test_voices_lists_directories(tmp_home):
    code, out = run(["voices"])
    assert code == 0
    assert "Installed Piper voices" in out
    assert "Installed Kokoro models" in out


def test_voices_catalogue_can_be_filtered(tmp_home):
    try:
        import piper  # noqa: F401
    except ImportError:
        pytest.skip("piper is not installed")
    code, out = run(["voices", "--all", "--language", "de"])
    assert code == 0
    assert "de_DE" in out
    assert "en_US-lessac" not in out


# --------------------------------------------------------------------------- #
# speak / listen / transcribe argument handling
# --------------------------------------------------------------------------- #

def test_speak_without_text_is_a_usage_error(tmp_home):
    code, out = run(["speak"])
    assert code == 2
    assert "Nothing to speak" in out


def test_speak_with_silent_engine_is_a_no_op(tmp_home, tmp_path):
    out_wav = tmp_path / "out.wav"
    code, out = run(["speak", "hello there", "--engine", "none", "-o", str(out_wav)])
    # 'none' is a deliberate configuration, not an error.
    assert code == 0
    assert "disabled" in out
    assert not out_wav.exists()


def test_transcribe_missing_file(tmp_home, tmp_path):
    code, out = run(["transcribe", str(tmp_path / "nope.wav")])
    assert code == 2
    assert "No such file" in out


def test_listen_rejects_non_wav_files(tmp_home, tmp_path):
    fake = tmp_path / "audio.mp3"
    fake.write_bytes(b"not really mp3")
    code, out = run(["listen", "--file", str(fake)])
    assert code == 2
    assert "WAV" in out


def test_bundle_writes_an_offline_folder(tmp_home):
    (tmp_home / "models" / "piper").mkdir(parents=True, exist_ok=True)
    (tmp_home / "models" / "piper" / "en_US-x.onnx").write_bytes(b"1234")
    dest = tmp_home / "bundle-out"
    code, out = run(["bundle", str(dest)])
    assert code == 0
    assert (dest / "piper" / "en_US-x.onnx").exists()
    assert "OllaVoice offline model bundle" in (dest / "BUNDLE-INFO.txt").read_text()


# --------------------------------------------------------------------------- #
# Subprocess smoke tests (prove the module entry points really work)
# --------------------------------------------------------------------------- #

def _env(tmp_home) -> dict[str, str]:
    env = dict(os.environ)
    env["OLLAVOICE_HOME"] = str(tmp_home)
    env["NO_COLOR"] = "1"
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    # Never let a test try to open a real window or grab the machine's audio.
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["OLLAVOICE_NO_QT"] = "1"
    return env


def test_python_dash_m_help(tmp_home):
    proc = subprocess.run([sys.executable, "-m", "ollavoice", "--help"],
                          capture_output=True, text=True, timeout=120,
                          env=_env(tmp_home), cwd=str(PROJECT_ROOT))
    assert proc.returncode == 0
    assert "local speech input and speech output" in proc.stdout.lower()


def test_python_dash_m_doctor_json(tmp_home):
    proc = subprocess.run([sys.executable, "-m", "ollavoice", "doctor", "--json"],
                          capture_output=True, text=True, timeout=180,
                          env=_env(tmp_home), cwd=str(PROJECT_ROOT))
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["app"] == "OllaVoice"


def test_gui_command_explains_the_missing_dependency(tmp_home):
    """Without PySide6 the GUI command must say exactly what to install."""
    proc = subprocess.run([sys.executable, "-m", "ollavoice", "gui"],
                          capture_output=True, text=True, timeout=180,
                          env=_env(tmp_home), cwd=str(PROJECT_ROOT))
    try:
        import PySide6  # noqa: F401
        have_qt = True
    except ImportError:
        have_qt = False
    if have_qt:
        # Qt is present but the sandbox has no display: either exit code is fine,
        # as long as it did not traceback with an unhelpful message.
        assert proc.returncode in (0, 1, 2, 3)
    else:
        assert proc.returncode == 2
        assert "pip install PySide6" in proc.stderr
