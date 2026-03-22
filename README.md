# saccard

**saccard** extracts cardiac data (BVP signals, heart rate BPM, metadata, and plots) from a video file or live webcam stream using remote photoplethysmography (rPPG). It is a fork from pyVHR - with reduced cardiac complexity, and instead a focus on ocular behaviour.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

## Methods

Nine classical rPPG methods are supported:

| Method | Reference |
|--------|-----------|
| `cpu_CHROM` | De Haan & Jeanne (2013) |
| `cpu_LGI`   | Pilz et al. (2018) |
| `cpu_POS`   | Wang et al. (2016) |
| `cpu_PCA`   | Lewandowska et al. (2011) |
| `cpu_ICA`   | Poh et al. (2010) |
| `cpu_GREEN` | Verkruysse et al. (2008) |
| `cpu_SSR`   | Wang et al. (2015) |
| `cpu_PBV`   | De Haan & Van Leest (2014) |
| `cpu_OMIT`  | Álvarez Casado & Bordallo López (2022) |

Plus the deep-learning method **`MTTS_CAN`** (Liu et al., 2020).

## Installation

One-command reproducible GPU setup (Windows PowerShell):

```powershell
./scripts/setup_gpu_env.ps1
```

Compatibility alias (same behavior):

```powershell
./scripts/setup_gpu_envs.ps1
```

If you want to rebuild from scratch:

```powershell
./scripts/setup_gpu_env.ps1 -Recreate
```

Manual equivalent:

```bash
conda create -n saccard-gpu python=3.9 -y
conda activate saccard-gpu

# CUDA 12 compatible GPU stack (recommended, reproducible)
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Project dependencies (excluding torch packages by design)
pip install -r requirements.txt
pip install -e .
```

Verify GPU runtime:

```bash
python -c "import torch, cupy; print('torch', torch.__version__); print('torch_cuda', torch.cuda.is_available()); print('cupy_devices', cupy.cuda.runtime.getDeviceCount())"
```

Expected output includes `torch_cuda True` and `cupy_devices >= 1`.

Important:

- Do not run `pip install torch torchvision torchaudio` in this environment after the conda step above.
- This project auto-selects GPU execution when CUDA backends are available.

## Quick start

```python
from saccard import saccard

# Process a video file (all methods, default settings)
result = saccard('path/to/video.mp4')

# Process a live webcam stream for 30 seconds
result = saccard(0, stream_duration=30)

# Use only specific methods
result = saccard('video.mp4', methods=['cpu_CHROM', 'cpu_POS', 'MTTS_CAN'])
```

## Return value

`saccard()` returns a ``dict`` with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `'bvp'` | `dict[str, list]` | Windowed BVP signal arrays per method |
| `'bpm'` | `dict[str, ndarray]` | Per-window heart rate (BPM) per method |
| `'times'` | `ndarray` | Window-centre timestamps (seconds) |
| `'fps'` | `float` | Video frame rate |
| `'metadata'` | `dict` | `video`, `fps`, `width`, `height`, `total_frames`, `duration`, `methods` |
| `'plot'` | `plotly.Figure` | BPM-over-time chart for all methods |

### Example

```python
from saccard import saccard

result = saccard('video.mp4', winsize=10, methods=['cpu_CHROM', 'cpu_POS'])

print(result['fps'])                      # e.g. 30.0
print(result['metadata'])                 # dict of video info
print(result['bpm']['cpu_CHROM'])         # BPM array, one value per window
result['plot'].show()                     # open interactive Plotly chart
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `video` | — | Video file path (`str`) or webcam index (`int`) |
| `winsize` | `10` | Analysis window size in seconds |
| `methods` | all | List of method names to run |
| `roi_method` | `'convexhull'` | Skin ROI: `'convexhull'` or `'faceparsing'` |
| `pre_filt` | `False` | Bandpass-filter the RGB signal before BVP extraction |
| `post_filt` | `True` | Bandpass-filter the BVP signal after extraction |
| `stream_duration` | `30` | Seconds to capture when `video` is a webcam index |
| `verb` | `False` | Print progress messages |

## License

GPL-3.0 – see [LICENSE](LICENSE).
