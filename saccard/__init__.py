"""
saccard – remote cardiac data extraction from video.

Usage::

    from saccard import saccard

    result = saccard('video.mp4')
    # or for a live webcam stream (index 0):
    result = saccard(0, stream_duration=30)

The returned dict contains BVP signals, per-window BPM estimates, timestamps,
video metadata, and a Plotly figure for every requested method.
"""

import os
import warnings
import cv2
import numpy as np
import plotly.graph_objects as go
from importlib import import_module


def _configure_runtime_warnings():
    # Configure runtime knobs before TensorFlow/MediaPipe are imported.
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ.setdefault('GLOG_minloglevel', '2')

    warnings.filterwarnings(
        'ignore',
        message=r'.*tf\.losses\.sparse_softmax_cross_entropy is deprecated.*',
    )
    warnings.filterwarnings(
        'ignore',
        message=r'.*cupyx\.jit\.rawkernel is experimental.*',
        category=FutureWarning,
    )
    warnings.filterwarnings(
        'ignore',
        message=r'.*SymbolDatabase\.GetPrototype\(\) is deprecated.*',
        category=UserWarning,
    )


_configure_runtime_warnings()

import pyVHR
from pyVHR.extraction.sig_processing import (
    SignalProcessing,
    SignalProcessingParams,
    SkinProcessingParams,
)
from pyVHR.extraction.skin_extraction_methods import (
    SkinExtractionConvexHull,
    SkinExtractionFaceParsing,
)
from pyVHR.extraction.utils import sig_windowing, get_fps, sliding_straded_win_idx
from pyVHR.BVP.BVP import RGB_sig_to_BVP
from pyVHR.BVP.filters import BPfilter, apply_filter
from pyVHR.BPM.BPM import BVP_to_BPM, BPM_median

# ── constants ──────────────────────────────────────────────────────────────────
_MIN_HZ = 0.65   # 39 BPM
_MAX_HZ = 4.0    # 240 BPM

# All nine classical CPU-based rPPG methods
_CPU_METHODS = [
    'cpu_CHROM',
    'cpu_LGI',
    'cpu_POS',
    'cpu_PCA',
    'cpu_ICA',
    'cpu_GREEN',
    'cpu_SSR',
    'cpu_PBV',
    'cpu_OMIT',
]

# Methods that need fps forwarded as a parameter
_FPS_PARAM_METHODS = {'cpu_POS', 'cpu_SSR'}
# Methods that need a component selector
_COMP_PARAM_METHODS = {'cpu_PCA', 'cpu_ICA'}


def _cupy_cuda_available():
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _torch_cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _select_bvp_backend(method_name, bvp_module):
    """Pick GPU implementation when available, else fall back to the requested method."""
    if method_name.startswith('cpu_') and _cupy_cuda_available():
        gpu_method = f"cupy_{method_name[4:]}"
        if hasattr(bvp_module, gpu_method):
            return gpu_method, 'cuda'
    return method_name, 'cpu'

# ── helpers ────────────────────────────────────────────────────────────────────

def _video_metadata(source):
    """Return a dict with basic metadata for *source* (file path or webcam index)."""
    cap = cv2.VideoCapture(source)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = (total / fps) if total > 0 else None
    return {
        'video': source,
        'fps': fps,
        'width': width,
        'height': height,
        'total_frames': total if total > 0 else None,
        'duration': duration,
    }


def _capture_stream(source, duration_s, fps_hint=30):
    """Capture *duration_s* seconds of frames from a live stream."""
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or fps_hint
    max_frames = int(duration_s * fps)
    frames = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def _bpm_from_bvp_windows(bvps_win, fps):
    """Return a 1-D array of per-window BPM medians (one value per window)."""
    bpms_per_window = BVP_to_BPM(bvps_win, fps, minHz=_MIN_HZ, maxHz=_MAX_HZ)
    medians, _ = BPM_median(bpms_per_window)
    return medians


def _bpm_from_1d_bvp(bvp_signal, fps, winsize_s):
    """Window a 1-D BVP signal and return per-window BPM medians."""
    from pyVHR.extraction.utils import sliding_straded_win_idx
    N = len(bvp_signal)
    block_idx, _ = sliding_straded_win_idx(N, winsize_s, 1, fps)
    bvps_win = []
    for idx in block_idx:
        st, en = int(idx[0]), int(idx[-1]) + 1
        seg = bvp_signal[st:en].reshape(1, -1).astype(np.float32)
        bvps_win.append(seg)
    return _bpm_from_bvp_windows(bvps_win, fps)


def _window_raw_frames(raw_frames, winsize_s, stride_s, fps):
    """Window raw RGB frames into a list compatible with cpu_SSR."""
    n_frames = len(raw_frames)
    if n_frames == 0:
        return []
    block_idx, _ = sliding_straded_win_idx(n_frames, winsize_s, stride_s, fps)
    windows = []
    for idx in block_idx:
        frame_idx = idx.astype(np.int32)
        windows.append(raw_frames[frame_idx])
    return windows


def _make_plot(times, bpm_dict):
    """Return a Plotly figure of BPM over time for each method."""
    fig = go.Figure()
    for method, bpms in bpm_dict.items():
        t = times[:len(bpms)]
        fig.add_trace(go.Scatter(x=t, y=bpms, mode='lines', name=method))
    fig.update_layout(
        title='Heart Rate (BPM) over time',
        xaxis_title='Time (s)',
        yaxis_title='BPM',
        legend_title='Method',
    )
    return fig


# ── public API ─────────────────────────────────────────────────────────────────

def saccard(
    video,
    winsize=10,
    methods=None,
    roi_method='convexhull',
    pre_filt=False,
    post_filt=True,
    stream_duration=30,
    verb=False,
):
    """
    Extract cardiac data from a video file or live webcam stream.

    Parameters
    ----------
    video : str or int
        Path to a video file, or an integer webcam index (e.g. ``0``).
    winsize : float
        Analysis window size in **seconds** (default 10).
    methods : list of str, optional
        rPPG methods to run.  Defaults to all nine classical CPU methods plus
        MTTS-CAN.  Available classical methods:

        ``'cpu_CHROM'``, ``'cpu_LGI'``, ``'cpu_POS'``, ``'cpu_PCA'``,
        ``'cpu_ICA'``, ``'cpu_GREEN'``, ``'cpu_SSR'``, ``'cpu_PBV'``,
        ``'cpu_OMIT'``

        Deep-learning method: ``'MTTS_CAN'``
    roi_method : str
        Skin region of interest extraction: ``'convexhull'`` (default) or
        ``'faceparsing'``.
    pre_filt : bool
        Apply a bandpass filter to the RGB signal before BVP extraction
        (default ``False``).
    post_filt : bool
        Apply a bandpass filter to the BVP signal after extraction
        (default ``True``).
    stream_duration : float
        When *video* is a webcam index, capture this many seconds before
        processing (default 30).
    verb : bool
        Print progress messages (default ``False``).

    Returns
    -------
    dict with keys:

    ``'bvp'``
        ``dict`` mapping each method name to a list of windowed BVP arrays
        (each array has shape ``[num_estimators, num_frames]``).
    ``'bpm'``
        ``dict`` mapping each method name to a 1-D :class:`numpy.ndarray`
        of BPM values (one per window).
    ``'times'``
        1-D :class:`numpy.ndarray` of window-centre timestamps in seconds.
    ``'fps'``
        Video frame rate (float).
    ``'metadata'``
        ``dict`` with keys ``video``, ``fps``, ``width``, ``height``,
        ``total_frames``, ``duration``, ``methods``.
    ``'plot'``
        A :class:`plotly.graph_objects.Figure` with all BPM traces.
    """
    if methods is None:
        methods = _CPU_METHODS + ['MTTS_CAN']

    classical = [m for m in methods if m != 'MTTS_CAN']
    run_mtts  = 'MTTS_CAN' in methods
    run_ssr = 'cpu_SSR' in classical

    # ── 1. video source ────────────────────────────────────────────────────────
    is_stream = isinstance(video, int)
    if not is_stream:
        if not os.path.isfile(video):
            raise FileNotFoundError(f"Video file not found: {video}")

    meta = _video_metadata(video)
    fps  = meta['fps']

    if verb:
        print(f"[saccard] source: {video}  fps: {fps:.1f}")

    # ── 2. signal extraction ───────────────────────────────────────────────────
    sig_proc = SignalProcessing()
    sig_proc.set_total_frames(0)

    if roi_method == 'convexhull':
        skin_device = 'GPU' if _cupy_cuda_available() else 'CPU'
        sig_proc.set_skin_extractor(SkinExtractionConvexHull(skin_device))
    elif roi_method == 'faceparsing':
        skin_device = 'GPU' if _torch_cuda_available() else 'CPU'
        sig_proc.set_skin_extractor(SkinExtractionFaceParsing(skin_device))
    else:
        raise ValueError(f"Unknown roi_method '{roi_method}'")

    if is_stream:
        # capture live frames and write to a temp file so existing code paths work
        import tempfile
        raw_frames_list, fps = _capture_stream(video, stream_duration)
        if verb:
            print(f"[saccard] captured {len(raw_frames_list)} frames from stream")
        tmp = tempfile.NamedTemporaryFile(suffix='.avi', delete=False)
        tmp.close()
        h, w = raw_frames_list[0].shape[:2]
        writer = cv2.VideoWriter(
            tmp.name,
            cv2.VideoWriter_fourcc(*'XVID'),
            fps, (w, h),
        )
        for f in raw_frames_list:
            writer.write(f)
        writer.release()
        video_source = tmp.name
        meta['fps'] = fps
        meta['total_frames'] = len(raw_frames_list)
        meta['duration'] = len(raw_frames_list) / fps
    else:
        video_source = video

    if verb:
        print("[saccard] extracting holistic RGB signal …")

    sig = sig_proc.extract_holistic(video_source)   # [frames, 1, 3]

    # Also get raw frames for MTTS-CAN
    raw_frames = None
    if run_mtts or run_ssr:
        if verb:
            print("[saccard] extracting raw frames …")
        raw_frames = sig_proc.extract_raw(video_source)  # [frames, H, W, 3]

    # clean up temp file if we created one
    if is_stream:
        os.unlink(video_source)

    # ── 3. windowing ──────────────────────────────────────────────────────────
    windowed_sig, times = sig_windowing(sig, winsize, 1, fps)
    raw_windowed_sig = None
    if run_ssr and raw_frames is not None and len(raw_frames) > 0:
        raw_windowed_sig = _window_raw_frames(raw_frames, winsize, 1, fps)
    if verb:
        print(f"[saccard] {len(windowed_sig)} windows of {winsize}s")

    # bandpass filter helper
    bp_params = {'minHz': _MIN_HZ, 'maxHz': _MAX_HZ, 'fps': 'adaptive', 'order': 6}

    if pre_filt:
        windowed_sig = apply_filter(windowed_sig, BPfilter, fps=fps, params=bp_params)

    # ── 4. classical methods ───────────────────────────────────────────────────
    bvp_out = {}
    bpm_out = {}
    bvp_mod = import_module('pyVHR.BVP.methods')

    for method in classical:
        if verb:
            print(f"[saccard] running {method} …")

        runtime_method, runtime_device = _select_bvp_backend(method, bvp_mod)
        func = getattr(bvp_mod, runtime_method)

        if method in _FPS_PARAM_METHODS:
            params = {'fps': 'adaptive'}
        elif method in _COMP_PARAM_METHODS:
            params = {'component': 'all_comp'}
        else:
            params = {}

        method_input = raw_windowed_sig if method == 'cpu_SSR' and raw_windowed_sig is not None else windowed_sig

        bvps_win = RGB_sig_to_BVP(
            method_input, fps,
            device_type=runtime_device,
            method=func,
            params=params,
        )

        if post_filt:
            bvps_win = apply_filter(bvps_win, BPfilter, fps=fps, params=bp_params)

        bvp_out[method] = bvps_win
        bpm_out[method] = _bpm_from_bvp_windows(bvps_win, fps)

    # ── 5. MTTS-CAN ───────────────────────────────────────────────────────────
    if run_mtts and raw_frames is not None and len(raw_frames) > 0:
        if verb:
            print("[saccard] running MTTS-CAN …")
        try:
            from pyVHR.deepRPPG.mtts_can import MTTS_CAN_deep
            pulse = MTTS_CAN_deep(raw_frames, fps, verb=0)
            bvp_out['MTTS_CAN'] = pulse          # 1-D array
            bpm_out['MTTS_CAN'] = _bpm_from_1d_bvp(pulse, fps, winsize)
        except Exception as exc:
            if verb:
                print(f"[saccard] MTTS-CAN skipped: {exc}")

    # ── 6. plots ──────────────────────────────────────────────────────────────
    plot = _make_plot(times, bpm_out)

    # ── 7. finalise metadata ──────────────────────────────────────────────────
    meta['methods'] = list(bvp_out.keys())

    return {
        'bvp':      bvp_out,
        'bpm':      bpm_out,
        'times':    times,
        'fps':      fps,
        'metadata': meta,
        'plot':     plot,
    }
