#!/usr/bin/env bash
# =============================================================================
#  OllaVoice setup for Linux and macOS.
#
#  Windows users: run install-windows.bat (or scripts\setup.ps1) instead.
#
#  Usage:
#      ./scripts/setup.sh                    # auto-detect everything
#      ./scripts/setup.sh --tier high        # pick a performance preset
#      ./scripts/setup.sh --no-gui           # terminal-only install
#      ./scripts/setup.sh --offline          # install from ./vendor (air-gapped)
#      ./scripts/setup.sh --python python3.12
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
PY="${PYTHON:-}"
TIER="" MODEL="" LANGUAGE="" TTS_ENGINE="" TTS_VOICE="" STT_MODEL=""
GUI=1 STT=1 TTS=1 PULL_ARG="" OFFLINE=0 YES=0 SHORTCUTS=1

cyan()  { printf '\033[36m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
grey()  { printf '\033[2m%s\033[0m\n' "$*"; }
warn()  { printf '\033[33m%s\033[0m\n' "$*"; }
die()   { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier)        TIER="$2"; shift 2 ;;
    --model)       MODEL="$2"; shift 2 ;;
    --language)    LANGUAGE="$2"; shift 2 ;;
    --tts-engine)  TTS_ENGINE="$2"; shift 2 ;;
    --tts-voice)   TTS_VOICE="$2"; shift 2 ;;
    --stt-model)   STT_MODEL="$2"; shift 2 ;;
    --python)      PY="$2"; shift 2 ;;
    --no-gui)      GUI=0; shift ;;
    --no-stt)      STT=0; shift ;;
    --no-tts)      TTS=0; shift ;;
    --pull)        PULL_ARG="--pull"; shift ;;
    --no-pull)     PULL_ARG="--no-pull"; shift ;;
    --offline)     OFFLINE=1; shift ;;
    --no-shortcuts) SHORTCUTS=0; shift ;;
    -y|--yes)      YES=1; shift ;;
    -h|--help)     sed -n '2,20p' "$0"; exit 0 ;;
    *) die "unknown argument: $1 (try --help)" ;;
  esac
done

echo
echo "  OllaVoice - local speech for Ollama"
echo "  ==================================="
echo

# --------------------------------------------------------------------------- #
cyan "==> Checking Python"
if [[ -z "$PY" ]]; then
  for candidate in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then PY="$candidate"; break; fi
  done
fi
[[ -n "$PY" ]] || die "Python 3 was not found. Install Python 3.10+ and re-run."

VERSION="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
MINOR="${VERSION#*.}"
[[ "$VERSION" == 3.* ]] || die "Python 3 is required (found $VERSION)."
if (( MINOR < 10 || MINOR > 13 )); then
  die "Python $VERSION is not supported - OllaVoice needs 3.10 to 3.13."
fi
green "    Python $VERSION ($PY)"

# --------------------------------------------------------------------------- #
cyan "==> Creating the virtual environment"
if [[ -x "$VENV/bin/python" ]]; then
  green "    .venv already exists"
else
  "$PY" -m venv "$VENV"
  green "    created $VENV"
fi
VPY="$VENV/bin/python"
"$VPY" -m pip install --upgrade pip --disable-pip-version-check --quiet || \
  warn "    pip upgrade failed (continuing)"

# --------------------------------------------------------------------------- #
cyan "==> Installing Python packages"
PACKAGES=(numpy soundfile sounddevice)
[[ $STT -eq 1 ]] && PACKAGES+=("faster-whisper>=1.0")
[[ $TTS -eq 1 ]] && PACKAGES+=("piper-tts>=1.3" "kokoro-onnx>=0.9")
[[ $GUI -eq 1 ]] && PACKAGES+=("PySide6>=6.6")

PIP_ARGS=(-m pip install --disable-pip-version-check)
if [[ $OFFLINE -eq 1 || -d "$ROOT/vendor/wheels" ]]; then
  [[ -d "$ROOT/vendor/wheels" ]] || die "--offline needs $ROOT/vendor/wheels"
  PIP_ARGS+=(--no-index --find-links "$ROOT/vendor/wheels")
  grey "    installing from vendor/wheels (no network)"
fi
"$VPY" "${PIP_ARGS[@]}" "${PACKAGES[@]}" -e "$ROOT"
green "    installed: ${PACKAGES[*]}"

# --------------------------------------------------------------------------- #
cyan "==> Checking Ollama"
if command -v ollama >/dev/null 2>&1; then
  green "    found $(command -v ollama)"
  if curl -sf --max-time 3 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
    green "    the server is running"
  else
    warn "    the server is not running - start it with: ollama serve"
  fi
else
  warn "    ollama was not found. Install it from https://ollama.com/download"
fi

# --------------------------------------------------------------------------- #
cyan "==> Downloading speech models and verifying the loop"
SETUP=(-m ollavoice setup)
[[ $YES -eq 1 ]]     && SETUP+=(--yes)
[[ $OFFLINE -eq 1 ]] && SETUP+=(--offline)
[[ -n "$TIER" ]]       && SETUP+=(--tier "$TIER")
[[ -n "$MODEL" ]]      && SETUP+=(--model "$MODEL")
[[ -n "$LANGUAGE" ]]   && SETUP+=(--language "$LANGUAGE")
[[ -n "$TTS_ENGINE" ]] && SETUP+=(--tts-engine "$TTS_ENGINE")
[[ -n "$TTS_VOICE" ]]  && SETUP+=(--tts-voice "$TTS_VOICE")
[[ -n "$STT_MODEL" ]]  && SETUP+=(--stt-model "$STT_MODEL")
[[ -n "$PULL_ARG" ]]   && SETUP+=("$PULL_ARG")
[[ $STT -eq 0 ]] && SETUP+=(--skip-stt)
[[ $TTS -eq 0 ]] && SETUP+=(--skip-tts)

set +e
"$VPY" "${SETUP[@]}"
RC=$?
set -e
[[ $RC -eq 0 ]] || warn "    setup finished with warnings (exit $RC)"

# --------------------------------------------------------------------------- #
if [[ $SHORTCUTS -eq 1 && $GUI -eq 1 && "$(uname)" == "Linux" ]]; then
  cyan "==> Writing a .desktop launcher"
  APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
  mkdir -p "$APPS"
  cat > "$APPS/ollavoice.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=OllaVoice
Comment=Local speech input and output for Ollama
Exec=$VENV/bin/python -m ollavoice gui
Path=$ROOT
Terminal=false
Categories=Utility;AudioVideo;
EOF
  green "    $APPS/ollavoice.desktop"
fi

echo
green "  Done."
echo
echo "  Start the desktop app:"
echo "      $VENV/bin/python -m ollavoice gui"
echo "  Talk in this terminal:"
echo "      $VENV/bin/ollavoice chat"
echo "  Activate the venv:"
echo "      source $VENV/bin/activate"
echo
exit $RC
