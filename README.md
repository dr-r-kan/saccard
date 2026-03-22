# saccard

**saccard** extracts cardiac data (BVP signals, heart rate BPM, metadata, and plots) from a video file or live webcam stream using remote photoplethysmography (rPPG).

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

```bash
pip install -e .
```

> A GPU is not required. CUDA support is used automatically when available.

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
