#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Install Ollama and start the server.
#
#   bash scripts/install_ollama.sh
#
# Ollama is a single binary that downloads open-weight models and serves them
# behind an OpenAI-compatible HTTP API on localhost:11434. Free, offline,
# no account, no rate limit.
# ---------------------------------------------------------------------------
set -euo pipefail

info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

OS="$(uname -s)"

if command -v ollama >/dev/null 2>&1; then
  info "Ollama already installed: $(ollama --version 2>&1 | head -1)"
else
  case "$OS" in
    Linux)
      info "Installing Ollama for Linux ..."
      curl -fsSL https://ollama.com/install.sh | sh
      ;;
    Darwin)
      info "Installing Ollama for macOS ..."
      if command -v brew >/dev/null 2>&1; then
        brew install --cask ollama
      else
        die "Homebrew not found. Download the app from https://ollama.com/download"
      fi
      ;;
    MINGW*|MSYS*|CYGWIN*)
      die "On Windows, download and run the installer from https://ollama.com/download
Then re-run the rest of this repo from PowerShell or WSL."
      ;;
    *)
      die "Unsupported OS: $OS. See https://ollama.com/download"
      ;;
  esac
fi

# --- make sure the server is up --------------------------------------------
if curl -fsS -m 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
  info "Ollama server is already running on http://localhost:11434"
else
  info "Starting the Ollama server ..."
  if [[ "$OS" == "Linux" ]] && command -v systemctl >/dev/null 2>&1 \
     && systemctl list-unit-files 2>/dev/null | grep -q '^ollama.service'; then
    sudo systemctl enable --now ollama
  else
    # Fall back to a detached process; log to a file so students can read it.
    nohup ollama serve > "${TMPDIR:-/tmp}/ollama-serve.log" 2>&1 &
    info "Server log: ${TMPDIR:-/tmp}/ollama-serve.log"
  fi

  for _ in $(seq 1 30); do
    curl -fsS -m 2 http://localhost:11434/api/tags >/dev/null 2>&1 && break
    sleep 1
  done
fi

curl -fsS -m 3 http://localhost:11434/api/tags >/dev/null 2>&1 \
  || die "Server did not come up. Try running 'ollama serve' in another terminal."

info "Ollama is up."
echo
warn "Context-window note (read this, it bites everyone):"
cat <<'NOTE'
    Ollama picks a context window based on your VRAM: 4k on small GPUs,
    32k on big ones. Agents send long prompts (system prompt + tool schemas
    + every past observation), and Ollama TRUNCATES SILENTLY when they do not
    fit. The symptom is an agent that loops or forgets its task halfway.

    Force a large window by setting OLLAMA_CONTEXT_LENGTH before the server
    starts:

      Linux (systemd):
        sudo systemctl edit ollama
        # add:  [Service]
        #       Environment="OLLAMA_CONTEXT_LENGTH=32768"
        sudo systemctl restart ollama

      macOS / manual:
        OLLAMA_CONTEXT_LENGTH=32768 ollama serve

    Check what you actually got, while a model is loaded:  ollama ps
NOTE
echo
info "Next:  bash scripts/pull_models.sh"
