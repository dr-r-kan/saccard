#!/usr/bin/env bash
# install.sh – Install pyVHR and all dependencies into the active conda environment.
#
# Assumptions:
#   • You are already inside a fresh conda environment (e.g. `conda create -n pyvhr python=3.10 && conda activate pyvhr`).
#   • CUDA 12.x drivers are installed on the host machine.
#   • Run from the repository root: `bash install.sh`
#
# Dependency highlights:
#   • PyTorch 2.1 + CUDA 12.1  (via pytorch/nvidia conda channels)
#   • CuPy 13  (CUDA 12 build, via conda-forge)
#   • MediaPipe 0.10.14  (uses the legacy mp.solutions.* API that pyVHR relies on)
#   • TensorFlow ≥2.13  (required by pyVHR's MTTS-CAN deep-rPPG module)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Installing conda packages (PyTorch 2.1 / CUDA 12.1, CuPy 13, HDF5, PyTables) ..."
conda install -y \
    -c pytorch -c nvidia -c rapidsai -c conda-forge \
    "pytorch=2.1.0" \
    "torchvision=0.16.0" \
    "torchaudio=2.1.0" \
    "pytorch-cuda=12.1" \
    "cupy=13.0.0" \
    "hdf5>=1.12" \
    "pytables>=3.7"

echo ""
echo "==> Installing pip packages ..."
# Note: tensorflow is capped below 2.16 because TF 2.16+ restructured the keras
# API (keras 3) in a way that breaks pyVHR's MTTS-CAN deep-rPPG module.
pip install \
    "asposestorage==1.0.2" \
    "autorank==1.1.1" \
    "biosppy==0.7.3" \
    "ipython>=8.0" \
    "ipywidgets>=8.0" \
    "lmfit==1.0.3" \
    "mediapipe==0.10.14" \
    "plotly>=5.3.1" \
    "pybdf==0.2.5" \
    "PySimpleGUI==4.60.5" \
    "scikit-posthocs==0.6.7" \
    "scikit-image>=0.19" \
    "tqdm>=4.62.3" \
    "tensorflow>=2.13.0,<2.16" \
    "opencv-python>=4.5" \
    "numba>=0.57"

echo ""
echo "==> Installing pyVHR in editable mode ..."
pip install -e "$REPO_DIR"

echo ""
echo "==> Installation complete. Verify with: python -c \"import pyVHR; print('pyVHR imported successfully')\""
