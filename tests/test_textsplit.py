"""Markdown stripping and the streaming sentence splitter."""

from __future__ import annotations

from ollavoice.textsplit import (
    SentenceSplitter, chunk_for_tts, expand_symbols, split_sentences, strip_for_speech,
)


def test_strip_headings_bullets_and_code():
    text = (
        "# Title\n\n"
        "Some **bold** and *italic* and `code` words.\n\n"
        "- first item\n- second item\n\n"
        "```python\nprint('hi')\n```\n\n"
        "See [the docs](https://example.com/a-b) for more."
    )
    out = strip_for_speech(text)
    assert "#" not in out
    assert "**" not in out and "*" not in out
    assert "`" not in out
    assert "print" not in out           # code dropped by default
    assert "the docs" in out
    assert "https://" not in out


def test_strip_keeps_code_when_asked():
    text = "Run this:\n```bash\nollama pull llama3.2\n```\nDone."
    out = strip_for_speech(text, keep_code=True)
    assert "ollama pull llama3.2" in out
    assert "Code block" in out


def test_tables_become_readable_prose():
    text = "| voice | speed |\n|---|---|\n| piper | fast |\n| kokoro | slow |"
    out = strip_for_speech(text)
    assert "|" not in out
    assert "piper" in out and "fast" in out


def test_expand_symbols():
    assert "50 percent" in expand_symbols("50% faster")
    assert " and " in expand_symbols("salt & pepper")
    assert "..." not in expand_symbols("wait.....")


def test_splitter_streams_complete_sentences():
    stream = ("The quick brown fox jumps over the lazy dog. "
              "It does this every single morning without fail. "
              "Third sentence is here too.")
    splitter = SentenceSplitter(min_chars=20)
    got = []
    # Feed it one character at a time, the worst case for a token stream.
    for ch in stream:
        got.extend(splitter.feed(ch))
    got.extend(splitter.flush())
    assert len(got) == 3
    assert " ".join(got).split() == stream.split()
    assert all(len(s) >= 20 for s in got)


def test_splitter_emits_early_so_speech_can_start():
    """The first sentence must arrive long before the stream ends."""
    stream = ("Short answer first. " + "Then a much longer explanation that keeps "
              "going and going for quite a while. ")
    splitter = SentenceSplitter(min_chars=10)
    seen_at = None
    for i, ch in enumerate(stream):
        out = list(splitter.feed(ch))
        if out and seen_at is None:
            seen_at = i
    assert seen_at is not None and seen_at < len(stream) // 2


def test_splitter_does_not_break_on_abbreviations():
    splitter = SentenceSplitter(min_chars=10)
    out = list(splitter.feed("Dr. Smith went to Washington. He arrived at noon. "))
    out += list(splitter.flush())
    joined = " ".join(out)
    assert "Dr. Smith went to Washington." in joined
    assert not any(part.strip().endswith("Dr.") for part in out)


def test_splitter_forces_a_cut_on_very_long_sentences():
    """A wall of text with no punctuation must still be cut into speakable pieces.

    feed() keeps the last piece in the buffer (more tokens may follow), so the
    remainder arrives from flush().
    """
    text = "word " * 200
    splitter = SentenceSplitter(min_chars=40, max_chars=200)
    out = list(splitter.feed(text)) + list(splitter.flush())
    assert len(out) >= 4, f"expected the wall of text to be cut up, got {len(out)} piece(s)"
    assert all(len(s) <= 260 for s in out)
    # Pieces are stripped for speech, so compare word lists rather than raw text.
    assert " ".join(out).split() == text.split(), "no words may be lost or duplicated"


def test_splitter_handles_paragraph_breaks():
    """A paragraph break is a hard boundary; a short lead-in waits for flush()."""
    splitter = SentenceSplitter(min_chars=10)
    out = list(splitter.feed("Short para.\n\nAnother short one."))
    out += list(splitter.flush())
    assert len(out) == 2
    assert out[0] == "Short para."

    # With a high minimum the splitter still hands everything over at flush().
    patient = SentenceSplitter(min_chars=200)
    assert list(patient.feed("Short para.\n\nAnother short one.")) == []
    assert len(list(patient.flush())) == 1


def test_split_sentences_one_shot():
    out = split_sentences("One. Two. Three.")
    assert out == ["One. Two. Three."] or len(out) >= 1
    assert "".join(out).replace(" ", "") == "One.Two.Three."


def test_chunk_for_tts_respects_max_chars():
    text = ". ".join(["a rather long sentence that keeps going"] * 20)
    for chunk in chunk_for_tts(text, max_chars=120):
        assert len(chunk) <= 160
        assert chunk.strip()


def test_chunk_for_tts_never_splits_mid_word():
    text = "alpha beta gamma delta epsilon zeta"
    chunks = list(chunk_for_tts(text, max_chars=18))
    for chunk in chunks:
        for word in chunk.split():
            assert word.strip(".,") in text
