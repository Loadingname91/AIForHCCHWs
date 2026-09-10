#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Download a Qwen3 model sized for your machine.
#
#   bash scripts/pull_models.sh            # auto-detect and pick for you
#   bash scripts/pull_models.sh qwen3:14b  # or name one explicitly
#
# Why Qwen3? It is Apache-2.0, it is genuinely good at emitting structured
# tool calls at small sizes, and Ollama ships tool-calling support for it.
# That combination is what an agent course needs.
# ---------------------------------------------------------------------------
set -euo pipefail

info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }

command -v ollama >/dev/null 2>&1 || {
  echo "Ollama not found. Run: bash scripts/install_ollama.sh" >&2; exit 1; }

# --- how much VRAM (or RAM) do we have? ------------------------------------
detect_vram_gb() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
      | sort -rn | head -1 | awk '{printf "%d", $1/1024}'
    return
  fi
  # Apple Silicon shares RAM with the GPU; use half of system memory.
  if [[ "$(uname -s)" == "Darwin" ]]; then
    sysctl -n hw.memsize 2>/dev/null | awk '{printf "%d", $1/1073741824/2}'
    return
  fi
  echo 0
}

pick_model() {
  local gb="$1"
  if   (( gb >= 30 )); then echo "qwen3-coder:30b"
  elif (( gb >= 22 )); then echo "qwen3:14b"
  elif (( gb >= 11 )); then echo "qwen3:8b"
  elif (( gb >=  6 )); then echo "qwen3:4b-instruct"
  else                      echo "qwen3:1.7b"
  fi
}

MODEL="${1:-}"
if [[ -z "$MODEL" ]]; then
  VRAM_GB="$(detect_vram_gb)"
  if (( VRAM_GB > 0 )); then
    info "Detected ~${VRAM_GB} GB of GPU memory."
  else
    info "No GPU detected — you will run on CPU. Expect slow but working."
  fi
  MODEL="$(pick_model "$VRAM_GB")"
fi

cat <<TABLE

  Model                VRAM     Speed        Agent/tool-call quality
  ------------------------------------------------------------------
  qwen3:1.7b           ~2 GB    fast         poor  — demos only
  qwen3:4b-instruct    ~3 GB    fast         ok    — usable for Unit 1-2
  qwen3:8b             ~6 GB    medium       good  — recommended baseline
  qwen3:14b            ~10 GB   slower       very good
  qwen3-coder:30b      ~19 GB   fast (MoE)   best  — tuned for tool use

  Picked: ${MODEL}

TABLE

info "Pulling ${MODEL} (first run downloads several GB) ..."
ollama pull "$MODEL"

info "Smoke test ..."
ollama run "$MODEL" "Reply with exactly: READY" --hidethinking 2>/dev/null \
  || ollama run "$MODEL" "Reply with exactly: READY"

echo
info "Now point .env at it:"
echo "    LLM_BACKEND=ollama"
echo "    OLLAMA_MODEL_ID=${MODEL}"
