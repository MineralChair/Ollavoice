"""Hardware detection and tier recommendation."""

from __future__ import annotations

from ollavoice import hardware


def test_detect_returns_sane_values():
    info = hardware.detect()
    assert info.cpu_cores >= 1
    assert info.os_name
    assert isinstance(info.avx2, bool)
    assert info.ram_gb >= 0
    assert isinstance(info.gpus, list)


def test_summary_and_to_dict():
    info = hardware.detect()
    text = info.summary()
    assert "CPU:" in text and "RAM:" in text and "GPU:" in text
    d = info.to_dict()
    assert d["cpu_cores"] == info.cpu_cores
    assert "notes" in d


def test_recommend_tier_is_one_of_the_known_tiers():
    tier, reasons = hardware.recommend_tier()
    assert tier in hardware.TIER_ORDER
    assert reasons


def test_tier_scales_with_vram():
    small = hardware.HardwareInfo(vram_gb=0.0, ram_gb=8.0, cpu_cores=4, avx2=True)
    mid = hardware.HardwareInfo(vram_gb=10.0, ram_gb=32.0, cpu_cores=8, avx2=True)
    big = hardware.HardwareInfo(vram_gb=24.0, ram_gb=64.0, cpu_cores=16, avx2=True)
    assert hardware.recommend_tier(small)[0] == "low"
    assert hardware.recommend_tier(mid)[0] == "balanced"
    assert hardware.recommend_tier(big)[0] == "high"


def test_big_cpu_only_machine_gets_balanced():
    info = hardware.HardwareInfo(vram_gb=0.0, ram_gb=32.0, cpu_cores=12, avx2=True)
    assert hardware.recommend_tier(info)[0] == "balanced"


def test_compute_type_matches_device():
    info = hardware.HardwareInfo(vram_gb=0.0, cpu_cores=8, avx2=True)
    device, compute = hardware.whisper_compute_type(info, "auto")
    assert device == "cpu"
    assert compute in ("int8", "int8_float16")

    info_cuda = hardware.HardwareInfo(vram_gb=12.0, cpu_cores=8, avx2=True,
                                      gpus=[{"name": "RTX", "vram_gb": 12.0, "cuda": True}])
    device, compute = hardware.whisper_compute_type(info_cuda, "auto")
    assert device == "cuda" and compute == "float16"

    no_avx = hardware.HardwareInfo(vram_gb=0.0, cpu_cores=2, avx2=False)
    assert hardware.whisper_compute_type(no_avx, "auto") == ("cpu", "int8")


def test_thread_count_leaves_headroom():
    assert hardware.whisper_threads(hardware.HardwareInfo(cpu_cores=2)) == 1
    assert hardware.whisper_threads(hardware.HardwareInfo(cpu_cores=32)) == 8
    assert hardware.whisper_threads(hardware.HardwareInfo(cpu_cores=8), requested=3) == 3


def test_detect_fills_notes_list():
    real = hardware.detect()
    assert isinstance(real.notes, list)
    for note in real.notes:
        assert isinstance(note, str) and note
