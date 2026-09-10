#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Create the conda environment and install every dependency for the course.
#
#   bash scripts/setup_env.sh              # env name: cs6961-agents
#   bash scripts/setup_env.sh my-env-name
# ---------------------------------------------------------------------------
set -euo pipefail

ENV_NAME="${1:-cs6961-agents}"
PY_VERSION="3.12"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

command -v conda >/dev/null 2>&1 \
  || die "conda not found. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html"

# `conda activate` needs the shell hook; plain `conda` on PATH is not enough.
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  info "Environment '${ENV_NAME}' already exists — reusing it."
else
  info "Creating conda environment '${ENV_NAME}' (python ${PY_VERSION}) ..."
  conda create -y -n "$ENV_NAME" "python=${PY_VERSION}" pip
fi

conda activate "$ENV_NAME"
info "Using $(python -V) at $(which python)"

info "Installing dependencies (a few minutes) ..."
python -m pip install --upgrade pip
python -m pip install -r "${REPO_ROOT}/requirements.txt"

info "Installing this repo in editable mode so 'import cs6961_agents' works ..."
python -m pip install -e "${REPO_ROOT}"

if [[ ! -f "${REPO_ROOT}/.env" ]]; then
  cp "${REPO_ROOT}/.env.example" "${REPO_ROOT}/.env"
  info "Created .env from .env.example — open it and check STEP 1."
fi

cat <<NEXT

  Done. Every new terminal needs:

      conda activate ${ENV_NAME}

  Next steps:
      bash scripts/install_ollama.sh
      bash scripts/pull_models.sh
      python scripts/doctor.py

NEXT
