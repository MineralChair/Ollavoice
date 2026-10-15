"""OllaVoice - local speech input and speech output for Ollama.

Ollama itself is a text-in/text-out runtime; it has no native speech-to-text or
text-to-speech API (the feature request ollama/ollama#11021 is still open as of
v0.40.x). OllaVoice adds that missing layer *without* changing Ollama: it runs a
native desktop app that talks to your local Ollama server and puts local Whisper
(ear) and local Piper/Kokoro (mouth) around it.

Everything is offline after setup - no cloud APIs, no telemetry, no accounts.
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__", "APP_NAME", "status_report"]

APP_NAME = "OllaVoice"


def status_report(as_dict: bool = False):
    """One-call health summary: hardware, Ollama, engines, models on disk."""
    from . import hardware, paths as _paths, stt_engines, tts_engines
    from .config import load as load_config
    from .ollama_client import OllamaClient

    cfg = load_config(create_if_missing=False)
    hw = hardware.detect()
    tier, why = hardware.recommend_tier(hw)
    client = OllamaClient(cfg.ollama.host, cfg.ollama.api_key, timeout=3)
    alive = client.is_alive()
    report = {
        "app": APP_NAME,
        "version": __version__,
        "hardware": hw.to_dict(),
        "recommended_tier": tier,
        "tier_reasons": why,
        "tier_in_use": cfg.tier,
        "ollama": {
            "host": cfg.ollama.host,
            "alive": alive,
            "version": client.version() if alive else "",
            "model": cfg.ollama.model,
            "model_present": client.has_model(cfg.ollama.model) if alive else False,
            "models": client.model_names() if alive else [],
            "running": client.ps() if alive else [],
        },
        "stt": {"configured": cfg.stt.engine, "model": cfg.stt.model,
                "engines": stt_engines.available_engines()},
        "tts": {"configured": cfg.tts.engine, "voices": tts_engines.installed_voices(),
                "engines": tts_engines.available_engines()},
        "paths": _paths.describe(),
        "notes": list(hw.notes),
    }
    if as_dict:
        return report
    return _format_report(report)


def _format_report(r: dict) -> str:
    lines: list[str] = []
    add = lines.append
    add(f"{r['app']} {r['version']} - health report")
    add("=" * 62)
    add("\n[hardware]")
    hw = r["hardware"]
    add(f"  os      : {hw['os_name']} {hw['arch']}")
    add(f"  cpu     : {hw['cpu_name'] or 'unknown'} ({hw['cpu_cores']} cores, "
        f"AVX2={'yes' if hw['avx2'] else 'no'})")
    add(f"  ram     : {hw['ram_gb']:.1f} GB")
    add("  gpu     : " + (", ".join(f"{g.get('name')} {g.get('vram_gb')}GB"
                                     for g in (hw['ollama_gpus'] or hw['gpus'])) or "none"))
    add(f"  tier    : {r['tier_in_use']} (recommended: {r['recommended_tier']})")
    for w in r["tier_reasons"]:
        add(f"            {w}")

    add("\n[ollama]")
    o = r["ollama"]
    add(f"  host    : {o['host']} - {'ONLINE' if o['alive'] else 'OFFLINE'}"
        + (f" (v{o['version']})" if o["version"] else ""))
    add(f"  model   : {o['model'] or '(not set)'}"
        + (" [installed]" if o["model_present"] else (" [MISSING]" if o["alive"] else "")))
    if o["models"]:
        add(f"  local   : {len(o['models'])} model(s): " + ", ".join(o["models"][:8]))

    add("\n[speech-to-text]")
    add(f"  engine  : {r['stt']['configured']} / model {r['stt']['model']}")
    for e in r["stt"]["engines"]:
        mark = "ok " if e["available"] else "-- "
        add(f"  {mark}{e['name']:<16} {e['description']}"
            + (f"\n                      -> {e['hint']}" if e["hint"] else ""))

    add("\n[text-to-speech]")
    add(f"  engine  : {r['tts']['configured']}")
    voices = r["tts"]["voices"]
    add(f"  piper   : {len(voices['piper'])} voice(s) on disk ({voices['piper_size_mb']} MB)"
        + (f" - {', '.join(voices['piper'][:4])}" if voices["piper"] else ""))
    add(f"  kokoro  : {len(voices['kokoro'])} model(s) ({voices['kokoro_size_mb']} MB)"
        + (f" voices: {', '.join(voices['kokoro_voices'])}" if voices["kokoro_voices"] else ""))
    for e in r["tts"]["engines"]:
        mark = "ok " if e["available"] else "-- "
        add(f"  {mark}{e['name']:<16} {e['description']}"
            + (f"\n                      -> {e['hint']}" if e["hint"] else ""))

    add("\n[paths]")
    for k, v in r["paths"].items():
        add(f"  {k:<12}: {v}")

    if r["notes"]:
        add("\n[notes]")
        for n in r["notes"]:
            add(f"  - {n}")
    return "\n".join(lines)
