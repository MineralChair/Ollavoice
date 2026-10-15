"""Hardware detection and automatic tier selection.

Pure standard library: no psutil, no WMI package. On Windows it shells out to
``wmic``/``PowerShell`` for VRAM and CPU details, and always tries Ollama's own
``/api/ps`` first because that is the ground truth for what the LLM runtime can
see.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MB = 1024 * 1024

#: Tier names, cheapest first. Mirrors :data:`ollavoice.config.TIERS`.
TIER_ORDER = ("low", "balanced", "high")


@dataclass
class HardwareInfo:
    os_name: str = platform.system()
    arch: str = platform.machine()
    python: str = platform.python_version()
    cpu_name: str = ""
    cpu_cores: int = os.cpu_count() or 2
    ram_gb: float = 0.0
    avx2: bool = False
    gpus: list[dict[str, Any]] = field(default_factory=list)
    vram_gb: float = 0.0
    ollama_gpus: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def has_nvidia(self) -> bool:
        return any(str(g.get("vendor", "")).lower().startswith(("nvidia", "cuda"))
                   or g.get("cuda") for g in self.gpus + self.ollama_gpus)

    def summary(self) -> str:
        gpu = ", ".join(
            f"{g.get('name', '?')} ({g.get('vram_gb', '?')} GB)" for g in (self.ollama_gpus or self.gpus)
        ) or "none detected"
        return (
            f"OS: {self.os_name} {platform.release()} ({self.arch})\n"
            f"CPU: {self.cpu_name or 'unknown'} - {self.cpu_cores} logical cores"
            f"{' (AVX2)' if self.avx2 else ' (no AVX2)'}\n"
            f"RAM: {self.ram_gb:.1f} GB\n"
            f"GPU: {gpu}\n"
            f"Python: {self.python}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "os_name": self.os_name, "arch": self.arch, "python": self.python,
            "cpu_name": self.cpu_name, "cpu_cores": self.cpu_cores,
            "ram_gb": round(self.ram_gb, 2), "avx2": self.avx2,
            "gpus": self.gpus, "vram_gb": round(self.vram_gb, 2),
            "ollama_gpus": self.ollama_gpus, "notes": self.notes,
        }


def _run(cmd: list[str], timeout: int = 15) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return (out.stdout or "") + (out.stderr or "")
    except (OSError, subprocess.SubprocessError):
        return ""


def detect_ram_gb() -> float:
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024 / (1024 ** 3)
        except OSError:
            pass
    if sys.platform == "win32":
        txt = _run(["wmic", "ComputerSystem", "get", "TotalPhysicalMemory", "/value"])
        m = re.search(r"TotalPhysicalMemory=(\d+)", txt)
        if m:
            return int(m.group(1)) / (1024 ** 3)
        txt = _run([
            "powershell", "-NoProfile", "-Command",
            "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
        ])
        m = re.search(r"(\d{6,})", txt)
        if m:
            return int(m.group(1)) / (1024 ** 3)
    try:
        import ctypes  # noqa: PLC0415

        if sys.platform == "darwin":
            libc = ctypes.CDLL("libc.dylib")
            size = ctypes.c_uint64()
            libc.sysctlbyname(b"hw.memsize", ctypes.byref(size),
                              ctypes.byref(ctypes.c_size_t(8)), None, 0)
            return size.value / (1024 ** 3)
    except Exception:  # pragma: no cover - best effort only
        pass
    return 0.0


def detect_cpu() -> tuple[str, int, bool]:
    name = platform.processor() or ""
    cores = os.cpu_count() or 2
    flags = ""
    if sys.platform.startswith("linux"):
        try:
            info = Path("/proc/cpuinfo").read_text()
            m = re.search(r"model name\s*:\s*(.+)", info)
            name = m.group(1).strip() if m else name
            flags = info
        except OSError:
            pass
    elif sys.platform == "win32":
        txt = _run(["wmic", "cpu", "get", "Name", "/value"])
        m = re.search(r"Name=(.+)", txt)
        if m:
            name = m.group(1).strip()
        flags = _run(["powershell", "-NoProfile", "-Command",
                      "[System.Environment]::Is64BitOperatingSystem"])
        # Cheap AVX2 probe: .NET reports the ISA via Vector256 only in newer
        # runtimes, so fall back to the CPU brand string heuristic below.
    avx2 = "avx2" in flags.lower()
    if not avx2:
        # Heuristic: every Intel CPU from Haswell (2013) and every AMD CPU from
        # Excavator/Zen (2015/2017) onwards has AVX2.
        m = re.search(r"Intel\(R\) Core\(TM\) i([3579])-(\d{4,5})", name)
        if m and int(m.group(2)[:1]) >= 4 and len(m.group(2)) >= 4:
            gen = int(m.group(2)[:2]) if len(m.group(2)) == 5 else int(m.group(2)[0])
            avx2 = gen >= 4
        if re.search(r"AMD Ryzen|EPYC|Threadripper", name, re.I):
            avx2 = True
    return name.strip(), cores, avx2


def detect_gpus() -> tuple[list[dict[str, Any]], float]:
    gpus: list[dict[str, Any]] = []
    vram_gb = 0.0

    smi = shutil.which("nvidia-smi")
    if smi:
        txt = _run([smi, "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits"])
        for line in txt.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[0] and not parts[0].lower().startswith("nvidia-smi"):
                try:
                    mem = float(parts[1]) / 1024.0
                except ValueError:
                    mem = 0.0
                gpus.append({"name": parts[0], "vram_gb": round(mem, 1),
                             "vendor": "nvidia", "cuda": True,
                             "driver": parts[2] if len(parts) > 2 else ""})
                vram_gb = max(vram_gb, mem)

    if sys.platform == "win32" and not gpus:
        txt = _run([
            "powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object Name,"
            "AdapterRAM,DriverVersion | ConvertTo-Json -Compress",
        ])
        txt = txt.strip()
        if txt.startswith("[") or txt.startswith("{"):
            try:
                data = json.loads(txt)
                if isinstance(data, dict):
                    data = [data]
                for item in data:
                    mem = float(item.get("AdapterRAM") or 0) / (1024 ** 3)
                    if mem > 4.0:  # AdapterRAM is a 32-bit counter; cap nonsense values
                        mem = 0.0
                    gpus.append({"name": item.get("Name", "?"), "vram_gb": round(mem, 1),
                                 "vendor": "unknown", "cuda": False,
                                 "driver": item.get("DriverVersion", "")})
                    vram_gb = max(vram_gb, mem)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
    return gpus, vram_gb


def detect_ollama_gpus(host: str = "http://127.0.0.1:11434") -> list[dict[str, Any]]:
    """Ask a running Ollama what it sees. This is the most reliable signal."""
    import urllib.error
    import urllib.request

    out: list[dict[str, Any]] = []
    try:
        with urllib.request.urlopen(host.rstrip("/") + "/api/ps", timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for m in data.get("models", []) or []:
            out.append({
                "name": m.get("name", "?"),
                "vram_gb": round((m.get("size_vram") or 0) / (1024 ** 3), 2),
                "size_gb": round((m.get("size") or 0) / (1024 ** 3), 2),
                "vendor": "ollama",
                "details": (m.get("details") or {}).get("family", ""),
            })
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
        pass
    return out


def detect() -> HardwareInfo:
    info = HardwareInfo()
    info.cpu_name, info.cpu_cores, info.avx2 = detect_cpu()
    info.ram_gb = detect_ram_gb()
    info.gpus, info.vram_gb = detect_gpus()
    info.ollama_gpus = detect_ollama_gpus()

    if not info.avx2:
        info.notes.append(
            "No AVX2 detected: CTranslate2 (faster-whisper) still runs, but expect "
            "roughly half the throughput. The 'low' tier is recommended."
        )
    if info.ram_gb and info.ram_gb < 8:
        info.notes.append("Less than 8 GB of RAM: keep the Ollama model at 4B parameters or smaller.")
    if not info.gpus and not info.ollama_gpus:
        info.notes.append("No GPU detected - everything will run on the CPU.")
    return info


# --------------------------------------------------------------------------- #
# Tier recommendation
# --------------------------------------------------------------------------- #


def recommend_tier(info: HardwareInfo | None = None) -> tuple[str, list[str]]:
    """Return ``(tier, reasons)``. Deliberately conservative: a too-small model
    is a much better first experience than a 20-second pause after every word."""
    info = info or detect()
    reasons: list[str] = []
    vram = info.vram_gb
    ram = info.ram_gb

    if vram >= 16 or (vram >= 12 and ram >= 32):
        reasons.append(f"{vram:.0f} GB VRAM -> high tier (medium.en Whisper + Kokoro TTS)")
        return "high", reasons
    if vram >= 8:
        reasons.append(f"{vram:.0f} GB VRAM -> balanced tier, GPU-accelerated Whisper")
        return "balanced", reasons
    if vram >= 6:
        reasons.append(f"{vram:.0f} GB VRAM -> balanced tier")
        return "balanced", reasons
    if ram >= 16 and info.cpu_cores >= 8 and info.avx2:
        reasons.append(f"No GPU but {info.cpu_cores} cores / {ram:.0f} GB RAM -> balanced tier")
        return "balanced", reasons
    reasons.append(f"CPU-only with {info.cpu_cores} cores / {ram:.0f} GB RAM -> low tier")
    return "low", reasons


def whisper_compute_type(info: HardwareInfo | None = None, device: str = "auto") -> tuple[str, str]:
    """Pick ``(device, compute_type)`` for faster-whisper."""
    info = info or detect()
    if device == "auto":
        device = "cuda" if info.has_nvidia else "cpu"
    if device == "cuda":
        return "cuda", "float16"
    return "cpu", ("int8" if not info.avx2 else "int8_float16")


def whisper_threads(info: HardwareInfo | None = None, requested: int = 0) -> int:
    if requested > 0:
        return requested
    info = info or detect()
    # Leave a core for the GUI and for Ollama; ONNX/CT2 scale poorly past ~8.
    return max(1, min(8, info.cpu_cores - 1))


def cuda_available() -> bool:
    try:
        import ctranslate2  # noqa: PLC0415

        return bool(ctranslate2.get_cuda_device_count())
    except Exception:
        return False


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    hw = detect()
    print(hw.summary())
    tier, why = recommend_tier(hw)
    print("\nrecommended tier:", tier)
    for w in why:
        print("  -", w)
    for n in hw.notes:
        print("  note:", n)
