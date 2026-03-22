#!/usr/bin/env bash
# install.sh – Install saccard and all dependencies into the active Python environment.
#
# Assumptions:
#   • You are already inside a Python environment (conda or venv).
#   • For GPU acceleration: CUDA 12.x drivers and cupy must be installed separately.
#   • Run from the repository root: `bash install.sh`

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Installing pip packages ..."
pip install \
    "numpy>=1.21" \
    "scipy>=1.7" \
    "opencv-python>=4.5" \
    "mediapipe==0.10.14" \
    "plotly>=5.3.1" \
    "scikit-learn>=1.0" \
    "scikit-image>=0.19" \
    "numba>=0.57" \
    "lmfit>=1.0" \
    "torch>=2.0" \
    "tensorflow>=2.13.0,<2.16" \
    "requests>=2.25" \
    "ipython>=8.0" \
    "ipywidgets>=8.0"

echo ""
echo "==> Installing saccard in editable mode ..."
pip install -e "$REPO_DIR"

echo ""
echo "==> Installation complete."
echo "    Verify with: python -c \"from saccard import saccard; print('saccard ready')\""
echo ""
echo "    Optional GPU support (requires CUDA 12.x drivers):"
echo "      conda install -c pytorch -c nvidia pytorch=2.1.0 pytorch-cuda=12.1"
echo "      conda install -c rapidsai -c conda-forge cupy=13.0.0"
