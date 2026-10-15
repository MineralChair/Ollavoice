"""The dependency-free Ollama HTTP client, against the fake server."""

from __future__ import annotations

import pytest

from ollavoice.ollama_client import (
    ChatHistory, ChatMessage, OllamaClient, OllamaError,
    OllamaNotRunning, wait_for_server,
)


def test_version_and_liveness(fake_ollama):
    base, _ = fake_ollama
    client = OllamaClient(base, timeout=5)
    assert client.is_alive()
    assert client.version() == "0.0.0-fake"


def test_dead_server_raises_a_helpful_error():
    client = OllamaClient("http://127.0.0.1:1", timeout=1)
    assert not client.is_alive()
    assert client.version() == ""
    with pytest.raises(OllamaNotRunning) as exc:
        client.list_models()
    assert "ollama serve" in str(exc.value)


def test_list_models_and_has_model(fake_ollama):
    base, _ = fake_ollama
    client = OllamaClient(base, timeout=5)
    names = client.model_names()
    assert "testmodel:latest" in names
    assert client.has_model("testmodel:latest")
    assert client.has_model("testmodel")          # tag-insensitive
    assert not client.has_model("nosuchmodel")


def test_chat_non_streaming(fake_ollama):
    base, handler = fake_ollama
    client = OllamaClient(base, timeout=5)
    answer = client.chat_once([ChatMessage.user("hi")], "testmodel:latest")
    assert answer == handler.answer
    sent = handler.requests[-1]
    assert sent["messages"][0] == {"role": "user", "content": "hi"}
    assert sent["stream"] is False


def test_chat_streaming_concatenates_to_the_same_answer(fake_ollama):
    base, handler = fake_ollama
    client = OllamaClient(base, timeout=5)
    pieces = []
    events = client.chat([ChatMessage.user("hi")], "testmodel:latest", stream=True)
    done_seen = False
    for evt in events:
        delta = (evt.get("message") or {}).get("content", "")
        if delta:
            pieces.append(delta)
        if evt.get("done"):
            done_seen = True
    assert done_seen
    assert "".join(pieces) == handler.answer


def test_options_and_keep_alive_are_forwarded(fake_ollama):
    base, handler = fake_ollama
    client = OllamaClient(base, timeout=5)
    client.chat_once([{"role": "user", "content": "x"}], "testmodel:latest",
                     options={"temperature": 0.2, "num_ctx": 1024})
    sent = handler.requests[-1]
    assert sent["options"]["temperature"] == 0.2
    assert sent["keep_alive"] == "10m"


def test_system_message_goes_first(fake_ollama):
    base, handler = fake_ollama
    client = OllamaClient(base, timeout=5)
    history = ChatHistory(turns=4, system="Be brief.")
    history.add_user("hello")
    client.chat_once(history.for_api(), "testmodel:latest")
    sent = handler.requests[-1]
    assert sent["messages"][0]["role"] == "system"
    assert sent["messages"][1]["content"] == "hello"


def test_pull_reports_success(fake_ollama):
    base, handler = fake_ollama
    client = OllamaClient(base, timeout=5)
    client.pull("something:latest", stream=False)
    assert handler.requests[-1]["model"] == "something:latest"


def test_http_error_becomes_ollama_error(fake_ollama):
    base, _ = fake_ollama
    client = OllamaClient(base, timeout=5)
    with pytest.raises(OllamaError):
        client._request("/api/does-not-exist")


def test_a_stray_path_in_the_host_is_normalised(fake_ollama):
    base, _ = fake_ollama
    client = OllamaClient(base + "/some/prefix", timeout=5)
    assert client.host == base
    assert client.is_alive()


def test_wait_for_server(fake_ollama):
    base, _ = fake_ollama
    assert wait_for_server(base, timeout_s=3)
    assert not wait_for_server("http://127.0.0.1:1", timeout_s=1, poll_s=0.2)


# --------------------------------------------------------------------------- #
# Chat history
# --------------------------------------------------------------------------- #

def test_history_keeps_a_sliding_window():
    history = ChatHistory(turns=2)
    for i in range(10):
        history.add_user(f"q{i}")
        history.add_assistant(f"a{i}")
    assert len(history.messages) == 4
    assert history.messages[0].content == "q8"
    assert history.messages[-1].content == "a9"


def test_history_zero_turns_is_stateless():
    history = ChatHistory(turns=0)
    history.add_user("hello")
    assert history.messages == []


def test_history_transcript_and_clear():
    history = ChatHistory(turns=5, system="sys")
    history.add_user("what is 2+2?")
    history.add_assistant("4")
    text = history.as_transcript()
    assert "You: what is 2+2?" in text and "Ollama: 4" in text
    history.clear()
    assert history.messages == []
    assert history.for_api() == [{"role": "system", "content": "sys"}]


def test_message_serialisation_drops_empty_images():
    assert ChatMessage.user("hi").to_dict() == {"role": "user", "content": "hi"}
    assert ChatMessage("user", "hi", images=["abc"]).to_dict()["images"] == ["abc"]
