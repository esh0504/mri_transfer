#!/usr/bin/env bash
# Download SeunghoEum/mri-tongue-dataset into V2/datasets/
#
# Prerequisites:
#   1) Request + receive access: https://huggingface.co/datasets/SeunghoEum/mri-tongue-dataset
#   2) Authenticate:  hf auth login   (or export HF_TOKEN=...)
#   3) pip install huggingface_hub
#
# Usage:
#   ./datasets/dataset_download.sh
#   ./datasets/dataset_download.sh --skip-dicom
#   ./datasets/dataset_download.sh --out ./datasets

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V2_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${V2_DIR}"

if ! python3 -c "import huggingface_hub" 2>/dev/null; then
  echo "Installing huggingface_hub …"
  python3 -m pip install -q huggingface_hub
fi

if command -v hf >/dev/null 2>&1; then
  if ! hf auth whoami >/dev/null 2>&1; then
    echo "Not logged in to Hugging Face."
    echo "  Request access: https://huggingface.co/datasets/SeunghoEum/mri-tongue-dataset"
    echo "  Then run: hf auth login"
    exit 1
  fi
elif [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Warning: neither 'hf auth login' nor HF_TOKEN found; download may fail for gated data."
fi

exec python3 "${SCRIPT_DIR}/download.py" "$@"
