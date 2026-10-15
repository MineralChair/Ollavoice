"""Native desktop GUI for OllaVoice (PySide6 / Qt 6).

Layout::

    +------------------------------------------------------------------+
    | toolbar: [Ollama model] [voice in] [voice out] [rate] ... [gear] |
    +------------------------------------------------------------------+
    |                                                                  |
    |  chat transcript (HTML, dark theme)                              |
    |                                                                  |
    +------------------------------------------------------------------+
    | [ text input ........................................ ] [ Send ] |
    | [  Hold to talk  ] [ level meter ] [ speak replies x ] [ Stop ]  |
    +------------------------------------------------------------------+
    | status: Ollama ok | whisper small.en | piper en_US-lessac-high   |
    +------------------------------------------------------------------+

Heavy work (Whisper, TTS, the Ollama stream) runs on a worker thread; the GUI
only ever touches widgets from the main thread, via signals.
"""
