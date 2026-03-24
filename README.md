# saccard

**saccard** is a project aiming to measure implicit interoceptive behaviour using passive video recordings of faces.

It uses a very simplified fork of pyVHR(link) in order to extract cardiac data, and then opencv to measure blinks and eye movement.

We then compile blink rate and eye movement from the video, and use circular statistics to determine the relative occurence per point in the cardiac cycle.



[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

## Methods

Nine classical rPPG methods are supported, transposed from the original pyVHR package.

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

The default install path is CPU-first and should be the safest option on a fresh machine.

Clone the repository, then run:
```bash
pip install -r requirements.txt
```

Optional installs:

- GPU/CuPy acceleration for compatible CUDA 12 environments:
```bash
pip install -r requirements-gpu.txt
```

- Torch/torchvision for the optional face-parsing ROI path:
```bash
pip install -r requirements-faceparsing.txt
```

Notes:

- The default pipeline does not require GPU packages.
- GPU execution is enabled only when the environment is appropriate and CUDA is actually available.
- You can force CPU mode even in a GPU-capable environment by setting `SACCARD_DISABLE_GPU=1`.

## Quick start

```python
from saccard import saccard

# Process a video file (all methods, default settings)
saccard('path/to/video.mp4')

```

You can also select a subset of the methods.

## CPU/GPU behavior

- CPU is the default and fallback mode.
- Classical `pyVHR` methods will use CUDA only if `cupy` is installed and a CUDA device is available.
- Optional face-parsing code paths require `torch` and `torchvision`; they are not needed for the default convex-hull workflow.

## Performance defaults

The pipeline now uses speed-oriented defaults tuned for longer recordings:

- FaceMesh landmark refresh stride defaults to `3` frames.
- Eye face-detection refresh stride defaults to `5` frames.

These defaults improve throughput while preserving output compatibility.

## Benchmarking

Use the benchmark helper to record runtime, memory, and stage timings:

```bash
python benchmark_pipeline.py path/to/video.mp4 --label baseline
python benchmark_pipeline.py path/to/video.mp4 --label optimized
```

Each run writes a JSON benchmark artifact to `outputs/`.

## License

GPL-3.0 – see [LICENSE](LICENSE).
