"""Download manager: resume, checksums, mirrors, offline mode, bundles."""

from __future__ import annotations

import hashlib
import http.server
import os
import threading

import pytest

from ollavoice import downloads

PAYLOAD = b"ollavoice-test-payload" * 500


class _Handler(http.server.BaseHTTPRequestHandler):
    hits = {"count": 0, "ranges": 0}
    fail_first = False

    def log_message(self, *args) -> None:
        pass

    def do_GET(self) -> None:  # noqa: N802
        type(self).hits["count"] += 1
        if self.path == "/missing":
            self.send_error(404)
            return
        if self.path == "/broken" and type(self).fail_first:
            self.send_error(500, "nope")
            return
        rng = self.headers.get("Range")
        start = 0
        if rng and rng.startswith("bytes="):
            start = int(rng.split("=")[1].split("-")[0])
            type(self).hits["ranges"] += 1
        body = PAYLOAD[start:]
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Type", "application/octet-stream")
        if start:
            self.send_header("Content-Range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}")
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def http_server():
    _Handler.hits = {"count": 0, "ranges": 0}
    _Handler.fail_first = False
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    yield f"http://{host}:{port}", _Handler
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=2)


def test_download_writes_the_file(http_server, tmp_path):
    base, _ = http_server
    dest = tmp_path / "blob.bin"
    result = downloads.download(f"{base}/blob.bin", dest)
    assert dest.read_bytes() == PAYLOAD
    assert not result.from_cache
    assert result.bytes == len(PAYLOAD)
    assert not dest.with_suffix(".bin.part").exists()


def test_download_is_cached(http_server, tmp_path):
    base, handler = http_server
    dest = tmp_path / "blob.bin"
    downloads.download(f"{base}/blob.bin", dest)
    before = handler.hits["count"]
    again = downloads.download(f"{base}/blob.bin", dest)
    assert again.from_cache
    assert handler.hits["count"] == before


def test_checksum_mismatch_is_rejected(http_server, tmp_path):
    base, _ = http_server
    with pytest.raises(RuntimeError):
        downloads.download(f"{base}/blob.bin", tmp_path / "bad.bin",
                           expected_sha256="0" * 64)


def test_checksum_match_is_accepted(http_server, tmp_path):
    base, _ = http_server
    digest = hashlib.sha256(PAYLOAD).hexdigest()
    dest = tmp_path / "good.bin"
    downloads.download(f"{base}/blob.bin", dest, expected_sha256=digest)
    assert downloads.sha256(dest) == digest


def test_resume_after_a_failed_attempt(http_server, tmp_path):
    base, handler = http_server
    dest = tmp_path / "resume.bin"
    partial = dest.with_suffix(".bin.part")
    partial.write_bytes(PAYLOAD[:1000])
    result = downloads.download(f"{base}/resume.bin", dest)
    assert dest.read_bytes() == PAYLOAD
    assert handler.hits["ranges"] >= 1
    assert not result.from_cache


def test_mirror_is_used_when_primary_fails(http_server, tmp_path):
    base, handler = http_server
    handler.fail_first = True
    dest = tmp_path / "mirror.bin"
    result = downloads.download(f"{base}/broken", dest, mirrors=[f"{base}/blob.bin"])
    assert dest.read_bytes() == PAYLOAD
    assert result.source.endswith("/blob.bin")


def test_progress_callback_fires(http_server, tmp_path):
    base, _ = http_server
    seen = []
    downloads.download(f"{base}/blob.bin", tmp_path / "p.bin",
                       progress=lambda done, total, name: seen.append((done, total, name)))
    assert seen and seen[-1][0] == len(PAYLOAD)
    assert all(name for _, _, name in seen)


def test_offline_mode_refuses_to_download(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAVOICE_OFFLINE", "1")
    assert downloads.offline_mode()
    with pytest.raises(downloads.OfflineError):
        downloads.download("http://127.0.0.1:9/nothing", tmp_path / "x.bin")


def test_url_helpers_respect_mirrors(monkeypatch):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.delenv("OLLAVOICE_HF_MIRROR", raising=False)
    url = downloads.hf_url("rhasspy/piper-voices", "en/en_US/lessac/high/x.onnx")
    assert url.startswith("https://huggingface.co/rhasspy/piper-voices/resolve/main/")
    assert downloads.hf_mirror_urls("a/b", "c") == []

    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
    assert downloads.hf_url("a/b", "c").startswith("https://hf-mirror.com/a/b/")
    assert downloads.hf_mirror_urls("a/b", "c")[0].startswith("https://hf-mirror.com/")


def test_unknown_kokoro_model_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        downloads.ensure_kokoro("kokoro-v9.onnx", tmp_path)


def test_bad_piper_voice_name_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAVOICE_OFFLINE", "1")
    with pytest.raises(ValueError):
        downloads.ensure_piper_voice("not-a-voice", tmp_path)


def test_copy_from_vendor(tmp_home):
    vendor = tmp_home / "vendor" / "models"
    vendor.mkdir(parents=True)
    (vendor / "en_US-lessac-medium.onnx").write_bytes(b"weights")
    dest = tmp_home / "models" / "piper"
    got = downloads.copy_from_vendor("en_US-lessac-medium.onnx", dest)
    assert got is not None and got.read_bytes() == b"weights"
    assert downloads.copy_from_vendor("does-not-exist.onnx", dest) is None


def test_make_offline_bundle(tmp_home):
    (tmp_home / "models" / "piper").mkdir(parents=True, exist_ok=True)
    (tmp_home / "models" / "piper" / "v.onnx").write_bytes(b"12345")
    bundle = downloads.make_offline_bundle(tmp_home / "bundle")
    assert (bundle / "piper" / "v.onnx").exists()
    assert "piper" in (bundle / "BUNDLE-INFO.txt").read_text()


def test_dir_size_mb(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 2048)
    assert downloads.dir_size_mb(tmp_path) == pytest.approx(2048 / (1024 * 1024))
    assert downloads.dir_size_mb(tmp_path / "nope") == 0.0


def test_temp_wav_path_is_inside_the_app_dir(tmp_home):
    p = downloads.temp_wav_path()
    try:
        assert str(tmp_home) in str(p)
        assert p.suffix == ".wav"
    finally:
        os.unlink(p)


def test_platform_tag_is_not_empty():
    assert downloads.platform_tag()
