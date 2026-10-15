"""Voice catalogue helpers."""

from __future__ import annotations

import pytest

from ollavoice import voices


def test_catalogue_is_read_without_polluting_stdout(capsys):
    catalogue = voices.piper_catalogue()
    captured = capsys.readouterr()
    assert captured.out == "", "list_voices() prints; the wrapper must swallow it"
    try:
        import piper  # noqa: F401
    except ImportError:
        pytest.skip("piper is not installed")
    assert len(catalogue) > 50
    assert all(voices.is_valid_piper_voice_name(v) for v in catalogue)


@pytest.mark.parametrize("language,prefix,unexpected", [
    ("de", "de_", "en_"),
    ("zh", "zh_", "en_"),
    ("en", "en_", "de_"),
    ("fr", "fr_", "en_"),
])
def test_filter_by_language(language, prefix, unexpected):
    catalogue = voices.piper_catalogue()
    if not catalogue:
        pytest.skip("piper is not installed")
    filtered = voices.filter_by_language(catalogue, language)
    assert filtered, f"no {language} voices in the catalogue?"
    # "en" covers en_US, en_GB and en_AR; what matters is nothing else leaks in.
    assert all(v.startswith(prefix) for v in filtered)
    assert not any(v.startswith(unexpected) for v in filtered)


def test_filter_by_language_accepts_full_codes():
    catalogue = voices.piper_catalogue()
    if not catalogue:
        pytest.skip("piper is not installed")
    assert voices.filter_by_language(catalogue, "de_DE") == \
        voices.filter_by_language(catalogue, "de")
    assert voices.filter_by_language(catalogue, "") == catalogue


@pytest.mark.parametrize("name,ok", [
    ("en_US-lessac-medium", True),
    ("en_GB-alan-medium", True),
    ("uk_UA-lada-x_low", True),       # quality may contain a dash
    ("zh_CN-huayan-x_low", True),
    ("lessac", False),
    ("en_US-lessac", False),
    ("enUS-lessac-medium", False),
    ("", False),
    ("en_US-lessac-ultra", False),
])
def test_voice_name_validation(name, ok):
    assert voices.is_valid_piper_voice_name(name) is ok


def test_installed_voices_lists_the_directory(tmp_home):
    folder = tmp_home / "models" / "piper"
    folder.mkdir(parents=True)
    assert voices.installed_piper_voices(folder) == []
    (folder / "en_US-lessac-high.onnx").write_bytes(b"x")
    (folder / "en_US-lessac-high.onnx.json").write_bytes(b"{}")
    (folder / "not-a-model.txt").write_text("ignore me")
    assert voices.installed_piper_voices(folder) == ["en_US-lessac-high"]


def test_stdout_fallback_parser():
    text = ("INFO:__main__:something\n"
            "ar_JO-kareem-low\n"
            "en_US-lessac-medium\n"
            "uk_UA-lada-x_low\n"
            "not a voice\n")
    parsed = voices._catalogue_from_stdout(text)
    assert parsed == ["ar_JO-kareem-low", "en_US-lessac-medium", "uk_UA-lada-x_low"]
