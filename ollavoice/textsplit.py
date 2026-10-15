"""Text post-processing for speech.

Two jobs:

1. :func:`strip_for_speech` - turn a markdown-flavoured LLM answer into something
   a TTS voice can read without saying "hash hash hash" or "backtick".
2. :class:`SentenceSplitter` - cut a *stream* of tokens into speakable sentences
   so the voice can start talking while Ollama is still generating. This is the
   single biggest perceived-latency win in a voice assistant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Iterator, List

# Terminal punctuation that ends a sentence. A following quote/bracket is
# absorbed so we do not speak `"...end." )` as two chunks.
_SENT_END = re.compile(r"""(?<=[.!?…])["'”’)\]]*(?=\s|$)""")
# Headings, bullets, rules, tables, code fences, inline code, emphasis, links.
_MD_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\n?")
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_MD_BULLET = re.compile(r"^\s*[*+-]\s+", re.MULTILINE)
_MD_NUMBER = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
_MD_QUOTE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_MD_RULE = re.compile(r"^\s*([-*_])\1{2,}\s*$", re.MULTILINE)
_MD_LINK = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MD_INLINE_CODE = re.compile(r"`([^`]*)`")
_MD_BOLD = re.compile(r"(\*\*|__)(.+?)\1")
_MD_ITALIC = re.compile(r"(?<!\w)([*_])(?!\s)(.+?)(?<!\s)\1(?!\w)")
_MD_STRIKE = re.compile(r"~~(.+?)~~")
_MD_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_MD_TABLE_SEP = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$", re.MULTILINE)
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE = re.compile(r"\n{3,}")

# Abbreviations that must not end a sentence.
_NO_SPLIT = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
    "al", "inc", "ltd", "co", "corp", "no", "nos", "fig", "approx", "dept",
    "ave", "blvd", "gen", "gov", "hon", "rev", "sgt", "capt", "lt", "col",
}

# Things a TTS voice should not try to pronounce.
_SPEECH_NOISE = re.compile(
    r"(\[.*?\]\(.*?\))|(<[^>]{0,80}>)|(\bhttps?://\S+)|(\b[\w./-]+@[\w.-]+\.\w+)",
)


def strip_for_speech(text: str, *, keep_code: bool = False) -> str:
    """Convert markdown/HTML-flavoured text into plain spoken prose."""
    if not text:
        return ""
    out = text.replace("\r\n", "\n").replace("\r", "\n")

    if keep_code:
        # Keep code blocks verbatim but flag them so the voice can say so.
        def _code(m: re.Match[str]) -> str:
            body = m.group(1).strip()
            return f" Code block. {body}. End code block. "
        out = re.sub(r"```[a-zA-Z0-9_+-]*\n(.*?)```", _code, out, flags=re.DOTALL)
    else:
        out = re.sub(r"```.*?```", " ", out, flags=re.DOTALL)
        out = _MD_FENCE.sub(" ", out)

    out = _MD_IMAGE.sub(lambda m: m.group(1) or "", out)
    out = _MD_LINK.sub(lambda m: m.group(1) or m.group(2), out)
    out = _MD_TABLE_SEP.sub(" ", out)
    out = _MD_TABLE_LINE.sub(lambda m: ", ".join(
        c.strip() for c in m.group(0).strip().strip("|").split("|") if c.strip()) + ".", out)
    out = _MD_RULE.sub(" ", out)
    out = _MD_QUOTE.sub("", out)
    out = _MD_HEADING.sub("", out)
    out = _MD_BULLET.sub("", out)
    out = _MD_NUMBER.sub("", out)
    out = _MD_INLINE_CODE.sub(lambda m: m.group(1), out)
    out = _MD_BOLD.sub(lambda m: m.group(2), out)
    out = _MD_ITALIC.sub(lambda m: m.group(2), out)
    out = _MD_STRIKE.sub(lambda m: m.group(1), out)
    out = _HTML_TAG.sub(" ", out)
    out = _SPEECH_NOISE.sub(" ", out)

    # Markdown tables/bullets leave stray pipes and asterisks behind.
    out = out.replace("|", " ").replace("*", " ").replace("#", " ").replace("_", " ")
    out = out.replace("~", " ").replace(">", " ").replace("`", " ")
    out = _MULTI_SPACE.sub(" ", out)
    out = _MULTI_NEWLINE.sub("\n\n", out)
    out = re.sub(r"\n+", ". ", out)
    out = re.sub(r"\.\s*\.", ".", out)
    return out.strip()


def expand_symbols(text: str) -> str:
    """Rewrite characters that espeak-ng/Piper pronounce badly."""
    table = {
        "%": " percent",
        "&": " and ",
        "@": " at ",
        "+": " plus ",
        "=": " equals ",
        "->": " to ",
        "=>": " to ",
        "<3": " love ",
        "°C": " degrees Celsius",
        "°F": " degrees Fahrenheit",
        "€": " euros",
        "£": " pounds",
        "¥": " yen",
        "•": ". ",
        "‣": ". ",
        "→": " to ",
        "…": "...",
        "\u00a0": " ",
        "\u200b": "",
    }
    for k, v in table.items():
        text = text.replace(k, v)
    # Collapse runs of punctuation that make voices stutter.
    text = re.sub(r"([!?.])\1{2,}", r"\1\1", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text


def looks_like_proper_noun_hint(text: str) -> str:
    """Build a Whisper ``initial_prompt`` that biases domain vocabulary."""
    words = re.findall(r"\b[A-Z][a-zA-Z0-9_.+-]{2,}\b", text or "")
    seen: list[str] = []
    for w in words:
        if w not in seen:
            seen.append(w)
        if len(seen) >= 12:
            break
    return ", ".join(seen)


@dataclass
class SentenceSplitter:
    """Incremental sentence splitter for token streams.

    Feed it chunks of text with :meth:`feed`; it yields complete sentences. Call
    :meth:`flush` at the end to get whatever is left over.

    ``min_chars`` avoids firing on fragments like "Yes." mid-stream; the splitter
    keeps accumulating until the sentence is long enough or a stronger boundary
    (paragraph break) appears.
    """

    min_chars: int = 40
    max_chars: int = 320
    buffer: str = ""
    _emitted: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._emitted is None:
            self._emitted = []

    # ------------------------------------------------------------------ #
    def feed(self, chunk: str) -> Iterator[str]:
        if not chunk:
            return
        self.buffer += chunk
        yield from self._drain(final=False)

    def flush(self) -> Iterator[str]:
        yield from self._drain(final=True)
        tail = self.buffer.strip()
        self.buffer = ""
        if tail:
            yield tail

    def fed_text(self) -> str:
        return "".join(self._emitted)

    # ------------------------------------------------------------------ #
    def _drain(self, *, final: bool) -> Iterator[str]:
        while True:
            cut = self._find_cut(final=final)
            if cut is None:
                break
            sentence = self.buffer[:cut].strip()
            self.buffer = self.buffer[cut:].lstrip()
            if sentence:
                self._emitted.append(sentence)
                yield sentence

    def _find_cut(self, *, final: bool) -> int | None:
        buf = self.buffer
        if not buf:
            return None

        if final:
            return len(buf) if buf.strip() else None

        # A paragraph break is always a boundary. If the paragraph so far is
        # still very short we wait, because more text may be on its way and a
        # two-word "sentence" sounds jarring when spoken.
        para = buf.find("\n\n")
        if para != -1:
            head = buf[:para].strip()
            if head and (len(head) >= self.min_chars or para == len(buf) - 2):
                return para + 2

        # No sentence end in sight: force a cut at a clause or word boundary so
        # the voice starts speaking instead of stalling on a wall of text.
        # The search is bounded by max_chars - an unbounded rfind() would just
        # hand back the whole buffer again.
        if len(buf) >= self.max_chars:
            return self._force_cut(buf)

        for m in _SENT_END.finditer(buf):
            end = m.end()
            if end < len(buf):                      # something follows the boundary
                candidate = buf[:end].strip()
                if len(candidate) >= self.min_chars and not _ends_with_abbreviation(candidate):
                    return end
        return None

    def _force_cut(self, buf: str) -> int:
        """Pick the best break point inside ``[min_chars, max_chars]``."""
        best: int | None = None
        for sep in ("; ", ". ", ", ", " - ", "\n", " "):
            idx = buf.rfind(sep, self.min_chars, self.max_chars + 1)
            if idx < 0:
                continue
            end = idx + len(sep)
            if best is None or end > best:
                best = end
            if best and best >= self.max_chars:
                break
        if best and best > self.min_chars:
            return best
        # No punctuation, no clause, not even a space: hard-cut so the buffer can
        # never grow without bound.
        return self.max_chars


def _ends_with_abbreviation(sentence: str) -> bool:
    """True when the sentence ends in something like ``Dr`` so we keep waiting."""
    m = re.search(r"([A-Za-z][A-Za-z.]{0,6})\.$", sentence.strip())
    if not m:
        return False
    return m.group(1).rstrip(".").lower() in _NO_SPLIT


def split_sentences(text: str, *, min_chars: int = 40) -> list[str]:
    """One-shot convenience wrapper around :class:`SentenceSplitter`."""
    splitter = SentenceSplitter(min_chars=min_chars)
    out = list(splitter.feed(text)) + list(splitter.flush())
    return [s for s in out if s.strip()]


def chunk_for_tts(text: str, *, max_chars: int = 400) -> Iterable[str]:
    """Split text into TTS-sized pieces without breaking words.

    Kokoro has a 510-phoneme ceiling and long inputs make every engine slower to
    first audio, so we cap chunk length even after sentence splitting.
    """
    for sentence in split_sentences(text, min_chars=1):
        if len(sentence) <= max_chars:
            yield sentence
            continue
        words = sentence.split()
        current: list[str] = []
        size = 0
        for w in words:
            if size + len(w) + 1 > max_chars and current:
                yield " ".join(current)
                current, size = [], 0
            current.append(w)
            size += len(w) + 1
        if current:
            yield " ".join(current)
