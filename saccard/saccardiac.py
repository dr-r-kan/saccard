"""
End-to-end cardiac and eye-movement extraction from a single video.

The pipeline is designed to run with common scientific Python packages and
OpenCV-based eye tracking. Cardiac extraction is delegated to the local pyVHR
wrapper in this repository.
"""

from __future__ import annotations

import argparse
import collections
import importlib
import importlib.util
import json
import logging
import math
import multiprocessing as mp
import os
import site
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import sys
import traceback
import faulthandler

from tqdm import tqdm
TQDM_AVAILABLE = True

from scipy import signal as scipy_signal
from scipy.interpolate import interp1d
from scipy.stats import chi2_contingency, kruskal, t
SCIPY_AVAILABLE = True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Ensure uncaught exceptions are logged
def _log_uncaught_exceptions(exc_type, exc_value, exc_tb):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    logger.exception("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))

sys.excepthook = _log_uncaught_exceptions
try:
    faulthandler.enable()
except Exception:
    pass


@dataclass
class MultimodalExtractionConfig:
    video_path: str
    output_dir: str = "outputs"
    output_fps: Optional[float] = None
    max_frames: Optional[int] = None

    rppg_method: str = "cpu_CHROM"
    rppg_methods: Tuple[str, ...] = field(
        default_factory=lambda: (
            "cpu_CHROM",
            "cpu_POS",
            "cpu_LGI",
            "cpu_OMIT",
            "cpu_GREEN",
            "cpu_ICA",
        )
    )
    rppg_window_size: float = 6.0
    rppg_roi_approach: str = "patches"
    rppg_estimate_type: str = "clustering"
    heart_band_hz: Tuple[float, float] = (0.7, 3.5)
    face_margin_px: int = 10
    cardiac_backend: str = "pyvhr"
    isolate_pyvhr: bool = False
    pyvhr_timeout_s: float = 120.0
    pyvhr_preflight_timeout_s: float = 30.0
    pyvhr_preflight_isolated: bool = True
    pyvhr_consensus_strategy: str = "promac"
    pyvhr_consensus_sigma_s: float = 0.10
    pyvhr_consensus_threshold: float = 0.35
    pyvhr_min_methods_for_consensus: int = 2
    pyvhr_drop_failed_methods: bool = True
    pyvhr_patch_size: int = 0
    pyvhr_rgb_low_high_th: Tuple[int, int] = (5, 230)

    eye_confidence_threshold: float = 0.5
    eye_smoothing_window: int = 5
    optical_flow_points: int = 30
    blink_motion_quantile: float = 0.15
    blink_min_separation_s: float = 0.15
    blink_metric_smoothing_window: int = 7
    head_motion_smoothing_window: int = 5
    movement_velocity_quantile: float = 0.75
    movement_velocity_floor: float = 0.01
    movement_context_s: float = 0.08

    phase_bin_count: int = 12
    min_phase_bin_samples: int = 10
    posterior_df0: float = 2.0
    posterior_kappa0: float = 1.0
    posterior_mu0: float = 0.0
    posterior_sigma0: float = 0.05

    output_csv: Optional[str] = None
    verbose: bool = True

@dataclass
class LiveMetrics:
    timestamp_s: float
    heart_bpm: float = math.nan
    saccade_modulation_pct: float = math.nan
    blink_modulation_pct: float = math.nan
    mean_eye_velocity: float = math.nan
    mean_eye_velocity_compensated: float = math.nan
    blink_rate_hz: float = math.nan
    sync_valid_fraction: float = math.nan


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _moving_average_nan(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return values.copy()
    window = min(int(window), len(values))
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=float)
    valid = np.isfinite(values)
    filled = np.where(valid, values, 0.0)
    numerator = np.convolve(filled, kernel, mode="same")
    denominator = np.convolve(valid.astype(float), kernel, mode="same")
    return np.divide(numerator, denominator, out=np.full_like(values, np.nan), where=denominator > 0)


def _interpolate_nan(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values.copy()
    return pd.Series(values, dtype=float).interpolate(limit_direction="both").to_numpy(dtype=float)


def _interp1d_eval(
    x: np.ndarray,
    y: np.ndarray,
    x_new: np.ndarray,
    kind: str = "linear",
    fill_value: Any = np.nan,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_new = np.asarray(x_new, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) == 0:
        return np.full_like(x_new, np.nan, dtype=float)
    if len(x) == 1:
        return np.full_like(x_new, y[0], dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if SCIPY_AVAILABLE:
        return interp1d(x, y, kind=kind, bounds_error=False, fill_value=fill_value)(x_new)

    if kind == "nearest":
        idx = np.searchsorted(x, x_new, side="left")
        idx = np.clip(idx, 0, len(x) - 1)
        left = np.clip(idx - 1, 0, len(x) - 1)
        choose_left = np.abs(x_new - x[left]) <= np.abs(x[idx] - x_new)
        nearest = np.where(choose_left, left, idx)
        out = y[nearest].astype(float)
    else:
        if fill_value == "extrapolate":
            out = np.interp(x_new, x, y)
            left_mask = x_new < x[0]
            right_mask = x_new > x[-1]
            left_slope = (y[1] - y[0]) / max(x[1] - x[0], 1e-12)
            right_slope = (y[-1] - y[-2]) / max(x[-1] - x[-2], 1e-12)
            out[left_mask] = y[0] + left_slope * (x_new[left_mask] - x[0])
            out[right_mask] = y[-1] + right_slope * (x_new[right_mask] - x[-1])
        else:
            out = np.interp(x_new, x, y, left=np.nan, right=np.nan)
            if np.isscalar(fill_value) and np.isfinite(fill_value):
                out[x_new < x[0]] = float(fill_value)
                out[x_new > x[-1]] = float(fill_value)
        out = out.astype(float)
    return out


def _detrend(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if SCIPY_AVAILABLE:
        return scipy_signal.detrend(values)
    if len(values) <= 1:
        return values.copy()
    x = np.arange(len(values), dtype=float)
    valid = np.isfinite(values)
    if np.sum(valid) < 2:
        return values - np.nanmean(values)
    slope, intercept = np.polyfit(x[valid], values[valid], 1)
    return values - (slope * x + intercept)


def _hilbert_phase(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if SCIPY_AVAILABLE:
        analytic = scipy_signal.hilbert(values)
    else:
        n = len(values)
        spectrum = np.fft.fft(values)
        h = np.zeros(n, dtype=float)
        if n % 2 == 0:
            h[0] = 1.0
            h[n // 2] = 1.0
            h[1:n // 2] = 2.0
        else:
            h[0] = 1.0
            h[1:(n + 1) // 2] = 2.0
        analytic = np.fft.ifft(spectrum * h)
    return np.mod(np.angle(analytic), 2 * np.pi)


def _find_peaks(values: np.ndarray, distance: int = 1, prominence: float = 0.0, height: Optional[float] = None) -> np.ndarray:
    series = np.asarray(values, dtype=float)
    if SCIPY_AVAILABLE:
        peaks, _ = scipy_signal.find_peaks(series, distance=distance, prominence=prominence, height=height)
        return peaks

    peaks: List[int] = []
    last_peak = -max(distance, 1)
    for idx in range(1, len(series) - 1):
        current = series[idx]
        if not np.isfinite(current):
            continue
        if height is not None and current < height:
            continue
        if current <= series[idx - 1] or current < series[idx + 1]:
            continue
        local_left = np.nanmin(series[max(0, idx - distance):idx + 1])
        local_right = np.nanmin(series[idx:min(len(series), idx + distance + 1)])
        local_prominence = current - max(local_left, local_right)
        if local_prominence < prominence:
            continue
        if idx - last_peak < distance:
            if peaks and current > series[peaks[-1]]:
                peaks[-1] = idx
                last_peak = idx
            continue
        peaks.append(idx)
        last_peak = idx
    return np.asarray(peaks, dtype=int)


def _bandpass_filter(values: np.ndarray, fps: float, low_hz: float, high_hz: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if SCIPY_AVAILABLE:
        nyq = fps / 2.0
        low = max(low_hz / nyq, 1e-4)
        high = min(high_hz / nyq, 0.99)
        if low >= high:
            raise ValueError("Invalid heart band for current FPS")
        b, a = scipy_signal.butter(3, [low, high], btype="bandpass")
        return scipy_signal.filtfilt(b, a, values)

    if len(values) < 4:
        return values.copy()
    centered = values - np.nanmean(values)
    centered = np.where(np.isfinite(centered), centered, 0.0)
    freqs = np.fft.rfftfreq(len(centered), d=1.0 / fps)
    spectrum = np.fft.rfft(centered)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    spectrum[~mask] = 0.0
    return np.fft.irfft(spectrum, n=len(centered))


def _confidence_interval_95(loc: float, scale: float, dof: float) -> Tuple[float, float]:
    if not np.isfinite(loc) or not np.isfinite(scale):
        return math.nan, math.nan
    if SCIPY_AVAILABLE and t is not None:
        return tuple(float(v) for v in t.interval(0.95, dof, loc=loc, scale=scale))
    z = 1.96
    return float(loc - z * scale), float(loc + z * scale)


def _circular_interp(source_times: np.ndarray, phase_rad: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    unwrapped = np.unwrap(phase_rad)
    out = _interp1d_eval(source_times, unwrapped, target_times, kind="linear", fill_value=np.nan)
    return np.mod(out, 2 * np.pi)


def _circular_mean(angles: np.ndarray) -> float:
    if len(angles) == 0:
        return math.nan
    return math.atan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))) % (2 * np.pi)


def _phase_from_peaks(times: np.ndarray, peak_times: np.ndarray) -> np.ndarray:
    phase = np.full(times.shape, np.nan, dtype=float)
    if len(peak_times) < 2:
        return phase
    for idx in range(len(peak_times) - 1):
        start = peak_times[idx]
        end = peak_times[idx + 1]
        mask = (times >= start) & (times < end)
        if end <= start or not np.any(mask):
            continue
        phase[mask] = 2 * np.pi * (times[mask] - start) / (end - start)
    return phase


def _compute_modulation_depth_pct(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 2:
        return math.nan
    peak = float(np.max(finite))
    trough = float(np.min(finite))
    denom = peak + trough
    if abs(denom) < 1e-12:
        return math.nan
    return float((peak - trough) / denom * 100.0)


def detect_blink_events(
    blink_metric: np.ndarray,
    timestamps: np.ndarray,
    config: MultimodalExtractionConfig,
) -> np.ndarray:
    metric = np.asarray(blink_metric, dtype=float)
    times = np.asarray(timestamps, dtype=float)
    events = np.zeros(len(metric), dtype=bool)
    if len(metric) < 3 or len(times) != len(metric):
        return events

    smoothed = _moving_average_nan(_interpolate_nan(metric), config.blink_metric_smoothing_window)
    baseline_window = max(5, int(round(len(smoothed) * 0.1)))
    baseline = pd.Series(smoothed).rolling(window=baseline_window, center=True, min_periods=1).median().to_numpy()
    closure_signal = baseline - smoothed
    finite = closure_signal[np.isfinite(closure_signal)]
    if len(finite) < 3:
        return events

    min_distance = max(1, int(round(config.blink_min_separation_s / max(np.nanmedian(np.diff(times)), 1e-3))))
    height = max(np.nanpercentile(finite, 75), np.nanmean(finite) + 0.5 * np.nanstd(finite))
    prominence = max(np.nanstd(finite) * 0.35, 1e-6)
    peaks = _find_peaks(closure_signal, height=height, prominence=prominence, distance=min_distance)
    events[peaks] = True
    return events


def _estimate_blink_metric(gray_frame: np.ndarray, eye_boxes: List[Tuple[int, int, int, int]]) -> float:
    scores: List[float] = []
    for x0, y0, x1, y1 in eye_boxes:
        roi = gray_frame[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        grad_y = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
        scores.append(float(np.mean(np.abs(grad_y))))
    if not scores:
        return math.nan
    return float(np.mean(scores))


def _detect_eye_boxes(
    gray_frame: np.ndarray,
    face_rect: Optional[Tuple[int, int, int, int]],
    eye_cascade: cv2.CascadeClassifier,
) -> List[Tuple[int, int, int, int]]:
    h, w = gray_frame.shape[:2]
    eye_boxes: List[Tuple[int, int, int, int]] = []
    if face_rect is not None:
        x0, y0, x1, y1 = face_rect
        face_gray = gray_frame[y0:y1, x0:x1]
        eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
        eyes = sorted(eyes, key=lambda rect: rect[0])[:2]
        for ex, ey, ew, eh in eyes:
            ex0 = x0 + ex
            ey0 = y0 + ey
            ex1 = ex0 + ew
            ey1 = ey0 + eh
            eye_boxes.append((ex0, ey0, ex1, ey1))

    if len(eye_boxes) == 0:
        default_centers, eye_box = _default_eye_regions((h, w), face_rect)
        box_w, box_h = int(eye_box[0]), int(eye_box[1])
        for center in default_centers:
            cx, cy = int(round(center[0])), int(round(center[1]))
            ex0 = max(cx - box_w // 2, 0)
            ey0 = max(cy - box_h // 2, 0)
            ex1 = min(cx + box_w // 2, w)
            ey1 = min(cy + box_h // 2, h)
            eye_boxes.append((ex0, ey0, ex1, ey1))
    return eye_boxes


def _estimate_pupil_center(
    gray_frame: np.ndarray,
    eye_box: Tuple[int, int, int, int],
) -> Tuple[np.ndarray, float]:
    x0, y0, x1, y1 = eye_box
    roi = gray_frame[y0:y1, x0:x1]
    if roi.size == 0 or roi.shape[0] < 6 or roi.shape[1] < 6:
        return np.array([math.nan, math.nan], dtype=float), 0.0

    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    threshold = float(np.percentile(blurred, 35))
    _, dark_mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))

    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.array([math.nan, math.nan], dtype=float), 0.0

    roi_area = float(roi.shape[0] * roi.shape[1])
    candidates: List[Tuple[float, np.ndarray]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 4.0 or area > roi_area * 0.35:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            continue
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
        center_penalty = abs(cx - roi.shape[1] / 2.0) / max(roi.shape[1], 1) + abs(cy - roi.shape[0] / 2.0) / max(roi.shape[0], 1)
        score = area / max(1.0 + 2.0 * center_penalty, 1e-6)
        candidates.append((score, np.array([x0 + cx, y0 + cy], dtype=float)))

    if not candidates:
        return np.array([math.nan, math.nan], dtype=float), 0.0

    best_score, best_center = max(candidates, key=lambda item: item[0])
    confidence = min(1.0, float(best_score / max(roi_area * 0.02, 1.0)))
    return best_center, confidence


def _estimate_eye_motion_sample(
    gray_frame: np.ndarray,
    frame_shape: Tuple[int, int],
    face_rect: Optional[Tuple[int, int, int, int]],
    eye_cascade: cv2.CascadeClassifier,
) -> Tuple[np.ndarray, float, np.ndarray, List[Tuple[int, int, int, int]]]:
    h, w = frame_shape
    eye_boxes = _detect_eye_boxes(gray_frame, face_rect, eye_cascade)
    pupil_centers: List[np.ndarray] = []
    confidences: List[float] = []
    for eye_box in eye_boxes:
        center, confidence = _estimate_pupil_center(gray_frame, eye_box)
        if np.all(np.isfinite(center)):
            pupil_centers.append(center)
            confidences.append(confidence)

    if pupil_centers:
        mean_center = np.mean(np.vstack(pupil_centers), axis=0)
        confidence = float(np.mean(confidences))
    else:
        mean_center = np.array([math.nan, math.nan], dtype=float)
        confidence = 0.0

    if face_rect is not None:
        x0, y0, x1, y1 = face_rect
        head_xy = np.array([(x0 + x1) / 2.0 / w, (y0 + y1) / 2.0 / h], dtype=float)
    else:
        head_xy = np.array([math.nan, math.nan], dtype=float)

    movement_xy = np.array([mean_center[0] / w, mean_center[1] / h], dtype=float)
    return movement_xy, confidence, head_xy, eye_boxes


def _detect_face_roi(frame_bgr: np.ndarray, margin_px: int) -> Optional[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    x0 = max(0, x - margin_px)
    y0 = max(0, y - margin_px)
    x1 = min(frame_bgr.shape[1], x + w + margin_px)
    y1 = min(frame_bgr.shape[0], y + h + margin_px)
    return x0, y0, x1, y1


def _default_eye_regions(frame_shape: Tuple[int, int], face_rect: Optional[Tuple[int, int, int, int]]) -> Tuple[List[np.ndarray], np.ndarray]:
    h, w = frame_shape
    if face_rect is not None:
        x0, y0, x1, y1 = face_rect
        face_w = max(x1 - x0, 1)
        face_h = max(y1 - y0, 1)
        left = np.array([x0 + 0.35 * face_w, y0 + 0.38 * face_h], dtype=float)
        right = np.array([x0 + 0.65 * face_w, y0 + 0.38 * face_h], dtype=float)
        box_w = max(int(face_w * 0.18), 20)
        box_h = max(int(face_h * 0.12), 16)
    else:
        left = np.array([w * 0.42, h * 0.40], dtype=float)
        right = np.array([w * 0.58, h * 0.40], dtype=float)
        box_w = max(int(w * 0.08), 20)
        box_h = max(int(h * 0.06), 16)
    return [left, right], np.array([box_w, box_h], dtype=int)


def _normalise_method_list(methods: Iterable[str]) -> List[str]:
    aliases = {
        "CHROM": "cpu_CHROM",
        "LGI": "cpu_LGI",
        "POS": "cpu_POS",
        "PBV": "cpu_PBV",
        "OMIT": "cpu_OMIT",
        "GREEN": "cpu_GREEN",
        "ICA": "cpu_ICA",
        "PCA": "cpu_PCA",
        "SSR": "cpu_SSR",
        "MTTS_CAN": "MTTS_CAN",
    }
    cleaned: List[str] = []
    for method in methods:
        name = str(method).strip()
        if not name:
            continue
        upper = name.upper()
        if upper in aliases:
            name = aliases[upper]
        elif name.startswith("cpu_"):
            pass
        elif f"cpu_{upper}" in aliases.values():
            name = f"cpu_{upper}"
        if name not in cleaned:
            cleaned.append(name)
    return cleaned


def _gaussian_kernel_1d(sigma_samples: float) -> np.ndarray:
    sigma_samples = float(max(sigma_samples, 1e-3))
    radius = max(1, int(round(4.0 * sigma_samples)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_samples) ** 2)
    kernel_sum = np.sum(kernel)
    if kernel_sum <= 0:
        return np.array([1.0], dtype=float)
    return kernel / kernel_sum


def _signal_to_1d_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return np.asarray([float(arr)], dtype=float)
    if arr.ndim == 1:
        return arr.astype(float)
    if arr.ndim == 2:
        time_axis = int(np.argmax(arr.shape))
        reduce_axis = 1 - time_axis
        return np.nanmedian(arr, axis=reduce_axis).astype(float)
    time_axis = int(np.argmax(arr.shape))
    moved = np.moveaxis(arr, time_axis, -1)
    flattened = moved.reshape(-1, moved.shape[-1])
    return np.nanmedian(flattened, axis=0).astype(float)


def _resample_numeric_to_grid(source_times: np.ndarray, values: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.full(len(target_times), np.nan, dtype=float)
    if len(values) == len(target_times) and np.allclose(source_times, target_times, atol=1e-9, rtol=1e-6):
        return values.copy()
    return _interp1d_eval(source_times, values, target_times, kind="linear", fill_value=np.nan)


def _synthesise_phase_from_bpm(times: np.ndarray, bpm: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    bpm = np.asarray(bpm, dtype=float)
    phase = np.full(len(times), np.nan, dtype=float)
    valid = np.isfinite(times) & np.isfinite(bpm)
    if np.sum(valid) < 2:
        return phase
    t = times[valid]
    f = np.clip(bpm[valid] / 60.0, 0.4, 4.0)
    dt = np.diff(t, prepend=t[0])
    dt[0] = np.nanmedian(np.diff(t)) if len(t) > 1 else 0.0
    unwrapped = np.cumsum(2.0 * np.pi * f * dt)
    phase_valid = np.mod(unwrapped, 2.0 * np.pi)
    phase[valid] = phase_valid
    return phase


def _orient_bvp_signal(
    bvp: np.ndarray,
    times: np.ndarray,
    bpm_hint: np.ndarray,
    heart_band_hz: Tuple[float, float],
) -> Dict[str, Any]:
    bvp = np.asarray(bvp, dtype=float)
    times = np.asarray(times, dtype=float)
    bpm_hint = np.asarray(bpm_hint, dtype=float)
    finite = np.isfinite(bvp) & np.isfinite(times)
    if np.sum(finite) < 10:
        raise RuntimeError("Insufficient finite BVP samples")

    sig = bvp.copy()
    sig[~np.isfinite(sig)] = np.nanmedian(sig[finite])
    sig = _detrend(sig)
    dt = np.diff(times[finite])
    fs = 1.0 / float(np.nanmedian(dt)) if len(dt) else math.nan
    if np.isfinite(fs) and fs > 2 * heart_band_hz[0]:
        try:
            sig = _bandpass_filter(sig, fs, heart_band_hz[0], heart_band_hz[1])
        except Exception:
            pass
    hint = float(np.nanmedian(bpm_hint[np.isfinite(bpm_hint)])) if np.any(np.isfinite(bpm_hint)) else math.nan

    def evaluate(candidate: np.ndarray) -> Dict[str, Any]:
        cand = np.asarray(candidate, dtype=float)
        prominence = max(float(np.nanstd(cand)) * 0.25, 1e-6)
        min_distance = 1
        if np.isfinite(fs) and fs > 0:
            min_distance = max(1, int(fs * 60.0 / 200.0))
        peaks = _find_peaks(cand, distance=min_distance, prominence=prominence)
        peak_times = times[peaks] if len(peaks) else np.array([], dtype=float)
        if len(peak_times) >= 2:
            ibi = np.diff(peak_times)
            ibi = ibi[np.isfinite(ibi) & (ibi > 0)]
            median_bpm = 60.0 / float(np.nanmedian(ibi)) if len(ibi) else math.nan
        else:
            median_bpm = math.nan
        if np.isfinite(hint) and np.isfinite(median_bpm):
            hr_score = math.exp(-abs(median_bpm - hint) / 12.0)
        elif np.isfinite(median_bpm):
            hr_score = 1.0 if 40.0 <= median_bpm <= 180.0 else 0.2
        else:
            hr_score = 0.0
        periodicity_score = min(len(peaks) / max(len(times) / max(fs, 1.0), 1.0), 4.0) / 4.0 if np.isfinite(fs) and fs > 0 else 0.0
        amplitude_score = float(np.nanstd(cand))
        score = hr_score + periodicity_score + 0.05 * amplitude_score
        return {
            "signal": cand,
            "peaks": peaks,
            "peak_times": peak_times,
            "median_bpm": median_bpm,
            "score": score,
        }

    pos = evaluate(sig)
    neg = evaluate(-sig)
    best = pos if pos["score"] >= neg["score"] else neg
    best["sign"] = 1.0 if best is pos else -1.0
    return best


def _parse_pyvhr_outputs(run_result: Any, fps: float, total_frames: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(run_result, (tuple, list)) or len(run_result) < 3:
        raise RuntimeError(f"Unexpected pyVHR run_on_video return type: {type(run_result).__name__}")

    arrays = [_signal_to_1d_array(item) for item in run_result[:3]]
    durations = total_frames / fps

    time_idx = None
    bpm_idx = None
    bvp_idx = None

    for idx, arr in enumerate(arrays):
        finite = arr[np.isfinite(arr)]
        if len(arr) >= 2 and np.all(np.diff(arr[np.isfinite(arr)]) >= 0) and np.nanmin(arr) >= 0 and np.nanmax(arr) <= durations * 1.2:
            time_idx = idx
            break

    for idx, arr in enumerate(arrays):
        if idx == time_idx:
            continue
        finite = arr[np.isfinite(arr)]
        if len(finite) >= 2:
            med = float(np.nanmedian(finite))
            if 30.0 <= med <= 220.0:
                bpm_idx = idx
                break

    remaining = [idx for idx in range(3) if idx not in {time_idx, bpm_idx}]
    if remaining:
        bvp_idx = max(remaining, key=lambda i: len(arrays[i]))

    if time_idx is None:
        if bpm_idx is not None:
            time_idx = [i for i in range(3) if i != bpm_idx][0]
        else:
            lengths = [len(arr) for arr in arrays]
            time_idx = int(np.argmax(lengths))

    if bpm_idx is None:
        candidates = [i for i in range(3) if i != time_idx]
        bpm_idx = min(candidates, key=lambda i: abs(float(np.nanmedian(arrays[i][np.isfinite(arrays[i])])) - 75.0) if np.any(np.isfinite(arrays[i])) else 1e9)

    if bvp_idx is None:
        candidates = [i for i in range(3) if i not in {time_idx, bpm_idx}]
        bvp_idx = candidates[0] if candidates else time_idx

    times = arrays[time_idx]
    bpm = arrays[bpm_idx]
    bvp = arrays[bvp_idx]

    if len(times) < 2 or not np.all(np.isfinite(times[: min(len(times), 5)])):
        duration = total_frames / fps
        n = max(len(bvp), len(bpm), 10)
        times = np.linspace(0.0, duration, n, endpoint=False)

    if len(bpm) == 0:
        bpm = np.full(len(times), np.nan, dtype=float)
    elif len(bpm) != len(times):
        bpm_time = np.linspace(times[0], times[-1], len(bpm)) if len(bpm) > 1 else np.asarray([times[0]], dtype=float)
        bpm = _interp1d_eval(bpm_time, bpm, times, kind="linear", fill_value="extrapolate")

    if len(bvp) == 0:
        bvp = np.full(len(times), np.nan, dtype=float)
    elif len(bvp) != len(times):
        bvp_time = np.linspace(times[0], times[-1], len(bvp)) if len(bvp) > 1 else np.asarray([times[0]], dtype=float)
        bvp = _interp1d_eval(bvp_time, bvp, times, kind="linear", fill_value="extrapolate")

    return np.asarray(times, dtype=float), np.asarray(bpm, dtype=float), np.asarray(bvp, dtype=float)


def _run_single_pyvhr_method(video_path: str, config: MultimodalExtractionConfig, method: str, import_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    print(f"[TRACE] _run_single_pyvhr_method start method={method}", file=sys.stderr, flush=True)
    _apply_import_context(import_context)
    print(f"[TRACE] _run_single_pyvhr_method preflight import start method={method}", file=sys.stderr, flush=True)
    preflight = _preflight_pyvhr_import(import_context)
    print(f"[TRACE] _run_single_pyvhr_method preflight import done method={method}", file=sys.stderr, flush=True)
    saccard_module = importlib.import_module("saccard")
    saccard_fn = getattr(saccard_module, "saccard", None)
    if saccard_fn is None:
        raise ImportError("Imported package 'saccard' but could not find callable 'saccard'")
    print(f"[TRACE] _run_single_pyvhr_method imported saccard.saccard method={method}", file=sys.stderr, flush=True)

    result = saccard_fn(
        video_path,
        winsize=config.rppg_window_size,
        methods=[method],
        roi_method="convexhull",
        pre_filt=True,
        post_filt=True,
        verb=config.verbose,
    )
    print(f"[TRACE] _run_single_pyvhr_method saccard returned method={method}", file=sys.stderr, flush=True)

    fps = _safe_float(result.get("fps"))
    metadata = result.get("metadata", {}) if isinstance(result, dict) else {}
    total_frames = int(metadata.get("total_frames") or 0)
    times = np.asarray(result.get("times", []), dtype=float)
    bpm = np.asarray(result.get("bpm", {}).get(method, []), dtype=float)
    if len(times) == 0 or len(bpm) == 0:
        raise RuntimeError(f"saccard() returned no cardiac samples for method {method}")
    if len(bpm) != len(times):
        bpm_time = np.linspace(times[0], times[-1], len(bpm)) if len(bpm) > 1 else np.asarray([times[0]], dtype=float)
        bpm = _interp1d_eval(bpm_time, bpm, times, kind="linear", fill_value="extrapolate")

    phase = _synthesise_phase_from_bpm(times, bpm)
    bvp = np.sin(phase)
    peak_times = np.array([], dtype=float)
    peak_indices = np.array([], dtype=int)
    finite_bpm = bpm[np.isfinite(bpm)]
    median_bpm = float(np.nanmedian(finite_bpm)) if len(finite_bpm) else math.nan
    quality = float(np.mean(np.isfinite(bpm))) if len(bpm) else 0.0
    print(f"[TRACE] _run_single_pyvhr_method complete method={method} n={len(times)} median_bpm={median_bpm}", file=sys.stderr, flush=True)
    return {
        "method": method,
        "times": times,
        "bpm": bpm,
        "bvp": bvp,
        "phase": phase,
        "peak_times": peak_times,
        "peak_indices": peak_indices,
        "sign": 1.0,
        "median_bpm": float(median_bpm) if np.isfinite(median_bpm) else math.nan,
        "quality": quality,
        "fps": float(fps),
        "total_frames": int(total_frames),
        "import_diagnostics": preflight,
    }


def _build_pyvhr_payloads_from_saccard_result(
    result: Dict[str, Any],
    methods: Iterable[str],
    import_diagnostics: Dict[str, Any],
) -> List[Dict[str, Any]]:
    times = np.asarray(result.get("times", []), dtype=float)
    if len(times) == 0:
        raise RuntimeError("saccard() returned no cardiac timestamps")

    fps = _safe_float(result.get("fps"))
    metadata = result.get("metadata", {}) if isinstance(result, dict) else {}
    total_frames = int(metadata.get("total_frames") or 0)
    bpm_dict = result.get("bpm", {}) if isinstance(result, dict) else {}

    payloads: List[Dict[str, Any]] = []
    for method in methods:
        bpm = np.asarray(bpm_dict.get(method, []), dtype=float)
        if len(bpm) == 0:
            raise RuntimeError(f"saccard() returned no cardiac samples for method {method}")
        if len(bpm) != len(times):
            bpm_time = np.linspace(times[0], times[-1], len(bpm)) if len(bpm) > 1 else np.asarray([times[0]], dtype=float)
            bpm = _interp1d_eval(bpm_time, bpm, times, kind="linear", fill_value="extrapolate")

        phase = _synthesise_phase_from_bpm(times, bpm)
        bvp = np.sin(phase)
        finite_bpm = bpm[np.isfinite(bpm)]
        median_bpm = float(np.nanmedian(finite_bpm)) if len(finite_bpm) else math.nan
        quality = float(np.mean(np.isfinite(bpm))) if len(bpm) else 0.0
        payloads.append(
            {
                "method": method,
                "times": times.copy(),
                "bpm": bpm,
                "bvp": bvp,
                "phase": phase,
                "peak_times": np.array([], dtype=float),
                "peak_indices": np.array([], dtype=int),
                "sign": 1.0,
                "median_bpm": float(median_bpm) if np.isfinite(median_bpm) else math.nan,
                "quality": quality,
                "fps": float(fps),
                "total_frames": int(total_frames),
                "import_diagnostics": import_diagnostics,
            }
        )
    return payloads


def _run_pyvhr_method_batch(
    video_path: str,
    config: MultimodalExtractionConfig,
    methods: Iterable[str],
    import_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    methods = _normalise_method_list(methods)
    print(f"[TRACE] _run_pyvhr_method_batch start methods={methods}", file=sys.stderr, flush=True)
    _apply_import_context(import_context)
    preflight = _preflight_pyvhr_import(import_context)
    saccard_module = importlib.import_module("saccard")
    saccard_fn = getattr(saccard_module, "saccard", None)
    if saccard_fn is None:
        raise ImportError("Imported package 'saccard' but could not find callable 'saccard'")
    result = saccard_fn(
        video_path,
        winsize=config.rppg_window_size,
        methods=methods,
        roi_method="convexhull",
        pre_filt=True,
        post_filt=True,
        verb=config.verbose,
    )
    print(f"[TRACE] _run_pyvhr_method_batch complete methods={methods}", file=sys.stderr, flush=True)
    return _build_pyvhr_payloads_from_saccard_result(result, methods, preflight)


def _pyvhr_preflight_worker(queue: "mp.Queue", import_context: Optional[Dict[str, Any]] = None) -> None:
    try:
        payload = _preflight_pyvhr_import(import_context)
        queue.put({"ok": True, "payload": payload})
    except BaseException as exc:
        tb = traceback.format_exc()
        try:
            queue.put({
                "ok": False,
                "error_type": type(exc).__name__,
                "error": repr(exc),
                "traceback": tb,
            })
        finally:
            raise


def _preflight_pyvhr_import_isolated(import_context: Optional[Dict[str, Any]], timeout_s: float) -> Dict[str, Any]:
    mp.set_executable(sys.executable)
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_pyvhr_preflight_worker, args=(queue, import_context), daemon=True)
    proc.start()
    print(f"[TRACE] Spawned pyVHR preflight child pid={proc.pid} timeout_s={timeout_s}", file=sys.stderr, flush=True)
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(5.0)
        raise RuntimeError(f"pyVHR import preflight timed out after {timeout_s:.1f} s")
    payload = None
    if not queue.empty():
        payload = queue.get_nowait()
    if payload is None:
        raise RuntimeError(f"pyVHR import preflight exited without payload; exitcode={proc.exitcode}. This usually means pyVHR or one of its native dependencies terminated the interpreter during import.")
    if payload.get("ok"):
        return payload["payload"]
    tb = payload.get("traceback", "")
    if tb:
        print(tb, file=sys.stderr, flush=True)
    raise RuntimeError(f"pyVHR import preflight failed with {payload.get('error_type', 'Exception')}: {payload.get('error', 'unknown error')}")


def _pyvhr_method_worker(video_path: str, config: MultimodalExtractionConfig, method: str, queue: "mp.Queue", import_context: Optional[Dict[str, Any]] = None) -> None:
    try:
        payload = _run_single_pyvhr_method(video_path=video_path, config=config, method=method, import_context=import_context)
        queue.put({"ok": True, "payload": payload})
    except BaseException as exc:
        tb = traceback.format_exc()
        try:
            queue.put({
                "ok": False,
                "error_type": type(exc).__name__,
                "error": repr(exc),
                "traceback": tb,
                "method": method,
            })
        finally:
            raise


def _pyvhr_batch_worker(video_path: str, config: MultimodalExtractionConfig, methods: List[str], queue: "mp.Queue", import_context: Optional[Dict[str, Any]] = None) -> None:
    try:
        payload = _run_pyvhr_method_batch(video_path=video_path, config=config, methods=methods, import_context=import_context)
        queue.put({"ok": True, "payload": payload})
    except BaseException as exc:
        tb = traceback.format_exc()
        try:
            queue.put({
                "ok": False,
                "error_type": type(exc).__name__,
                "error": repr(exc),
                "traceback": tb,
                "methods": methods,
            })
        finally:
            raise


class CardiacPhaseExtractor:
    def __init__(self, config: MultimodalExtractionConfig):
        self.config = config
        self.fps: Optional[float] = None
        self.last_diagnostics: Dict[str, Any] = {}
        self._import_context: Dict[str, Any] = _make_import_context()

    def extract(self, video_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        backend = str(getattr(self.config, "cardiac_backend", "pyvhr")).strip().lower()
        print(f"[TRACE] CardiacPhaseExtractor.extract backend={backend} video={video_path}", file=sys.stderr, flush=True)
        if backend not in {"auto", "pyvhr"}:
            raise ValueError(f"Unknown cardiac backend: {backend}")
        return self._extract_with_pyvhr_consensus(video_path)

    def _run_pyvhr_methods_isolated(self, video_path: str, methods: List[str]) -> List[Dict[str, Any]]:
        timeout_s = float(getattr(self.config, "pyvhr_timeout_s", 120.0))
        mp.set_executable(sys.executable)
        ctx = mp.get_context("spawn")
        queue: mp.Queue = ctx.Queue()
        proc = ctx.Process(target=_pyvhr_batch_worker, args=(video_path, self.config, methods, queue, self._import_context), daemon=True)
        proc.start()
        print(f"[TRACE] Spawned pyVHR child pid={proc.pid} methods={methods} timeout_s={timeout_s}", file=sys.stderr, flush=True)
        proc.join(timeout_s)
        if proc.is_alive():
            proc.terminate()
            proc.join(5.0)
            raise RuntimeError(f"pyVHR methods {methods} timed out after {timeout_s:.1f} s")
        payload = None
        if not queue.empty():
            payload = queue.get_nowait()
        if payload is None:
            raise RuntimeError(f"pyVHR methods {methods} exited without payload; exitcode={proc.exitcode}")
        if payload.get("ok"):
            return payload["payload"]
        tb = payload.get("traceback", "")
        if tb:
            print(tb, file=sys.stderr, flush=True)
        raise RuntimeError(f"pyVHR methods {methods} failed with {payload.get('error_type', 'Exception')}: {payload.get('error', 'unknown error')}")

    def _run_pyvhr_methods(self, video_path: str, methods: List[str]) -> List[Dict[str, Any]]:
        print(f"[TRACE] Running pyVHR methods in batch: {methods}", file=sys.stderr, flush=True)
        if getattr(self.config, "isolate_pyvhr", False):
            return self._run_pyvhr_methods_isolated(video_path, methods)
        return _run_pyvhr_method_batch(video_path=video_path, config=self.config, methods=methods, import_context=self._import_context)

    def _check_pyvhr_ready(self) -> Dict[str, Any]:
        print("[TRACE] pyVHR preflight starting", file=sys.stderr, flush=True)
        diagnostics: Dict[str, Any]
        isolated_enabled = bool(getattr(self.config, "pyvhr_preflight_isolated", True))
        timeout_s = float(getattr(self.config, "pyvhr_preflight_timeout_s", 30.0))
        if isolated_enabled:
            try:
                diagnostics = _preflight_pyvhr_import_isolated(
                    self._import_context,
                    timeout_s=timeout_s,
                )
            except Exception as exc:
                print(
                    f"[TRACE] Isolated pyVHR preflight failed; retrying in-process: {repr(exc)}",
                    file=sys.stderr,
                    flush=True,
                )
                self.last_diagnostics["pyvhr_preflight_retry"] = {
                    "isolated": True,
                    "timeout_s": timeout_s,
                    "error": repr(exc),
                }
                diagnostics = _preflight_pyvhr_import(self._import_context)
                diagnostics["preflight_retry_after_error"] = repr(exc)
                diagnostics["preflight_retry_mode"] = "in_process"
        else:
            diagnostics = _preflight_pyvhr_import(self._import_context)
        self.last_diagnostics.setdefault("pyvhr_import", diagnostics)
        print(
            f"[TRACE] pyVHR import OK module={diagnostics.get('module_file')} saccard={diagnostics.get('saccard_module_file')} executable={diagnostics.get('sys_executable')}",
            file=sys.stderr,
            flush=True,
        )
        return diagnostics

    def _combine_pyvhr_methods(self, method_results: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        if not method_results:
            raise RuntimeError("No successful pyVHR methods available for consensus")

        best = max(method_results, key=lambda d: (d.get("quality", 0.0), len(d.get("times", []))))
        common_times = np.asarray(best["times"], dtype=float)
        if len(common_times) < 10:
            raise RuntimeError("Consensus time grid too short")

        dt = np.diff(common_times)
        median_dt = float(np.nanmedian(dt)) if len(dt) else math.nan
        if not np.isfinite(median_dt) or median_dt <= 0:
            raise RuntimeError("Invalid consensus timestep")
        fs = 1.0 / median_dt
        sigma_samples = max(1.0, float(self.config.pyvhr_consensus_sigma_s) * fs)
        kernel = _gaussian_kernel_1d(sigma_samples)
        min_distance = max(1, int(fs * 60.0 / 200.0))

        consensus_prob = np.zeros(len(common_times), dtype=float)
        weight_sum = 0.0
        bvp_accum = np.zeros(len(common_times), dtype=float)
        bvp_weight_sum = np.zeros(len(common_times), dtype=float)
        bpm_accum = np.zeros(len(common_times), dtype=float)
        bpm_weight_sum = np.zeros(len(common_times), dtype=float)
        phase_vectors = np.zeros(len(common_times), dtype=complex)
        phase_weight_sum = np.zeros(len(common_times), dtype=float)
        method_diags: List[Dict[str, Any]] = []

        for result in method_results:
            weight = float(max(result.get("quality", 0.0), 1e-6))
            times = np.asarray(result["times"], dtype=float)
            bpm = _resample_numeric_to_grid(times, np.asarray(result["bpm"], dtype=float), common_times)
            phase = _circular_interp(times, np.asarray(result["phase"], dtype=float), common_times)
            bvp = _resample_numeric_to_grid(times, np.asarray(result["bvp"], dtype=float), common_times)
            bvp_z = bvp.copy()
            finite_bvp = np.isfinite(bvp_z)
            if np.any(finite_bvp):
                mu = float(np.nanmean(bvp_z[finite_bvp]))
                sd = float(np.nanstd(bvp_z[finite_bvp]))
                if sd > 0:
                    bvp_z[finite_bvp] = (bvp_z[finite_bvp] - mu) / sd
                else:
                    bvp_z[finite_bvp] = bvp_z[finite_bvp] - mu
                bvp_accum[finite_bvp] += weight * bvp_z[finite_bvp]
                bvp_weight_sum[finite_bvp] += weight

            finite_bpm = np.isfinite(bpm)
            bpm_accum[finite_bpm] += weight * bpm[finite_bpm]
            bpm_weight_sum[finite_bpm] += weight

            finite_phase = np.isfinite(phase)
            phase_vectors[finite_phase] += weight * np.exp(1j * phase[finite_phase])
            phase_weight_sum[finite_phase] += weight

            peak_train = np.zeros(len(common_times), dtype=float)
            peak_times = np.asarray(result.get("peak_times", []), dtype=float)
            if len(peak_times):
                peak_indices = np.searchsorted(common_times, peak_times)
                peak_indices = np.clip(peak_indices, 0, len(common_times) - 1)
                peak_train[peak_indices] = 1.0
                peak_prob = np.convolve(peak_train, kernel, mode="same")
                if np.nanmax(peak_prob) > 0:
                    peak_prob = peak_prob / np.nanmax(peak_prob)
                consensus_prob += weight * peak_prob
                weight_sum += weight

            method_diags.append({
                "method": result.get("method"),
                "quality": weight,
                "median_bpm": result.get("median_bpm", math.nan),
                "n_peaks": int(len(peak_times)),
                "sign": result.get("sign", math.nan),
                "n_samples": int(len(times)),
            })

        if weight_sum > 0:
            consensus_prob = consensus_prob / weight_sum
        threshold = float(self.config.pyvhr_consensus_threshold)
        consensus_peak_idx = _find_peaks(consensus_prob, distance=min_distance, prominence=max(0.05, threshold * 0.25), height=threshold)
        if len(consensus_peak_idx) < 2 and len(method_results) >= 1:
            consensus_peak_idx = np.asarray(best.get("peak_indices", []), dtype=int)
            consensus_peak_idx = consensus_peak_idx[(consensus_peak_idx >= 0) & (consensus_peak_idx < len(common_times))]

        consensus_peak_times = common_times[consensus_peak_idx] if len(consensus_peak_idx) else np.array([], dtype=float)
        phase = _phase_from_peaks(common_times, consensus_peak_times)
        valid_phase = np.isfinite(phase)
        phase_mean = np.mod(np.angle(phase_vectors), 2.0 * np.pi)
        phase = np.where(valid_phase, phase, phase_mean)

        bpm = np.divide(bpm_accum, bpm_weight_sum, out=np.full(len(common_times), np.nan, dtype=float), where=bpm_weight_sum > 0)
        if len(consensus_peak_times) >= 2:
            ibi = np.diff(consensus_peak_times)
            instant_bpm = 60.0 / ibi
            midpoint = consensus_peak_times[:-1] + ibi / 2.0
            bpm_from_peaks = _interp1d_eval(midpoint, instant_bpm, common_times, kind="linear", fill_value="extrapolate") if len(midpoint) >= 2 else np.full(len(common_times), instant_bpm[0], dtype=float)
            finite_peak_bpm = np.isfinite(bpm_from_peaks)
            bpm[finite_peak_bpm] = bpm_from_peaks[finite_peak_bpm]

        bvp = np.divide(bvp_accum, bvp_weight_sum, out=np.full(len(common_times), np.nan, dtype=float), where=bvp_weight_sum > 0)
        if not np.any(np.isfinite(bvp)):
            bvp = np.sin(phase)
        else:
            bvp = np.where(np.isfinite(bvp), bvp, np.sin(phase))

        diagnostics = {
            "backend": "pyvhr_consensus",
            "import": best.get("import_diagnostics", {}),
            "consensus_strategy": self.config.pyvhr_consensus_strategy,
            "n_methods_successful": int(len(method_results)),
            "methods": method_diags,
            "consensus_threshold": threshold,
            "consensus_sigma_s": float(self.config.pyvhr_consensus_sigma_s),
            "consensus_peak_count": int(len(consensus_peak_times)),
            "consensus_mean_bpm": float(np.nanmean(bpm)) if np.any(np.isfinite(bpm)) else math.nan,
            "consensus_weight_sum": float(weight_sum),
            "consensus_probability_max": float(np.nanmax(consensus_prob)) if len(consensus_prob) else math.nan,
            "timebase_method": best.get("method"),
        }
        return phase, common_times, bpm, bvp, diagnostics

    def _extract_with_pyvhr_consensus(self, video_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        methods = _normalise_method_list(getattr(self.config, "rppg_methods", ()) or (self.config.rppg_method,))
        if not methods:
            methods = [self.config.rppg_method]

        try:
            import_diag = self._check_pyvhr_ready()
        except Exception as exc:
            self.last_diagnostics = {
                "backend": "pyvhr_consensus",
                "preflight_error": repr(exc),
            }
            raise

        failures: List[Dict[str, Any]] = []
        try:
            successes = self._run_pyvhr_methods(video_path=video_path, methods=methods)
        except Exception as exc:
            failures.extend({"method": method, "error": repr(exc)} for method in methods)
            print(f"[TRACE] pyVHR batch failed for methods {methods}: {repr(exc)}", file=sys.stderr, flush=True)
            if not self.config.pyvhr_drop_failed_methods:
                raise
            successes = []

        for payload in successes:
            print(
                f"[TRACE] pyVHR method {payload.get('method')} succeeded: n={len(payload['times'])} median_bpm={payload.get('median_bpm', math.nan):.2f} quality={payload.get('quality', math.nan):.3f}",
                file=sys.stderr,
                flush=True,
            )

        if len(successes) < max(1, int(self.config.pyvhr_min_methods_for_consensus)):
            if not successes:
                raise RuntimeError("All pyVHR methods failed")
            print("[TRACE] Fewer methods than requested for consensus; using best successful method only", file=sys.stderr, flush=True)
            best = max(successes, key=lambda d: d.get("quality", 0.0))
            self.fps = best.get("fps")
            self.last_diagnostics = {
                "backend": "pyvhr_single_best",
                "selected_method": best.get("method"),
                "n_methods_successful": int(len(successes)),
                "import": best.get("import_diagnostics", import_diag),
                "failures": failures,
            }
            return best["phase"], best["times"], best["bpm"], best["bvp"]

        phase, times, bpm, bvp, diagnostics = self._combine_pyvhr_methods(successes)
        diagnostics.setdefault("import", import_diag)
        diagnostics["failures"] = failures
        self.fps = float(1.0 / np.nanmedian(np.diff(times))) if len(times) > 1 else None
        self.last_diagnostics = diagnostics
        return phase, times, bpm, bvp

    def _extract_with_pyvhr_isolated(self, video_path: str, allow_fallback: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self._extract_with_pyvhr_consensus(video_path)

    def _extract_with_pyvhr(self, video_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self._extract_with_pyvhr_consensus(video_path)

    def _bandpass_filter(self, values: np.ndarray, fps: float) -> np.ndarray:
        return _bandpass_filter(values, fps, self.config.heart_band_hz[0], self.config.heart_band_hz[1])

    def _find_bvp_peaks(self, values: np.ndarray, fps: float) -> np.ndarray:
        prominence = max(float(np.nanstd(values)) * 0.25, 1e-6)
        min_distance = max(1, int(float(fps) * 60.0 / 200.0))
        return _find_peaks(values, distance=min_distance, prominence=prominence)



class EyeMovementExtractor:
    def __init__(self, config: MultimodalExtractionConfig):
        self.config = config
        self.fps: Optional[float] = None
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    def extract(
        self,
        video_path: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        print("[TRACE] Entering cv2 pupil tracking extractor", file=sys.stderr, flush=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        fps = _safe_float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            cap.release()
            raise ValueError("Invalid video FPS")

        gaze_xy: List[List[float]] = []
        confidence: List[float] = []
        head_xy: List[List[float]] = []
        blink_metric: List[float] = []
        timestamps: List[float] = []

        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames, desc="Eye (cv2 pupil) frames", unit="fr") if tqdm and total_frames and total_frames > 0 and self.config.verbose else None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if self.config.max_frames is not None and frame_idx >= self.config.max_frames:
                break

            face_rect = _detect_face_roi(frame, self.config.face_margin_px)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            movement_xy, sample_confidence, sample_head_xy, eye_boxes = _estimate_eye_motion_sample(
                gray_frame=gray,
                frame_shape=gray.shape[:2],
                face_rect=face_rect,
                eye_cascade=self.eye_cascade,
            )
            blink_metric.append(_estimate_blink_metric(gray, eye_boxes))
            gaze_xy.append(movement_xy.tolist())
            confidence.append(sample_confidence)
            head_xy.append(sample_head_xy.tolist())
            timestamps.append(frame_idx / fps)
            frame_idx += 1
            if pbar:
                pbar.update(1)

        cap.release()
        if pbar:
            pbar.close()
        blink_flags = detect_blink_events(np.asarray(blink_metric, dtype=float), np.asarray(timestamps, dtype=float), self.config)
        gaze = np.asarray(gaze_xy, dtype=float)
        if len(gaze):
            gaze[:, 0] = _moving_average_nan(gaze[:, 0], self.config.eye_smoothing_window)
            gaze[:, 1] = _moving_average_nan(gaze[:, 1], self.config.eye_smoothing_window)
        self.fps = fps
        print(f"[TRACE] Eye extractor method produced {len(timestamps)} samples", file=sys.stderr, flush=True)
        return (
            gaze,
            np.asarray(timestamps, dtype=float),
            np.asarray(confidence, dtype=float),
            blink_flags,
            np.asarray(head_xy, dtype=float),
            np.asarray(blink_metric, dtype=float),
        )


class MultimodalSynchronizer:
    def __init__(self, config: MultimodalExtractionConfig):
        self.config = config

    def synchronize(
        self,
        cardiac_phase: np.ndarray,
        cardiac_times: np.ndarray,
        bpm: np.ndarray,
        bvp: np.ndarray,
        gaze_xy: np.ndarray,
        gaze_times: np.ndarray,
        gaze_confidence: np.ndarray,
        blink_flags: np.ndarray,
        head_xy: Optional[np.ndarray] = None,
        blink_metric: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        logger.info("Synchronizing cardiac and eye signals")
        print("[TRACE] Entering MultimodalSynchronizer.synchronize", file=sys.stderr, flush=True)

        t_start = max(cardiac_times[0], gaze_times[0])
        t_end = min(cardiac_times[-1], gaze_times[-1])
        if t_end <= t_start:
            raise RuntimeError("No overlapping cardiac and gaze time range")

        sample_fps = self.config.output_fps
        if sample_fps is None:
            cardiac_dt = np.nanmedian(np.diff(cardiac_times))
            gaze_dt = np.nanmedian(np.diff(gaze_times))
            sample_fps = 1.0 / max(cardiac_dt, gaze_dt)
        n_samples = int(math.floor((t_end - t_start) * sample_fps)) + 1
        t_common = np.linspace(t_start, t_end, n_samples)

        phase_sync = _circular_interp(cardiac_times, cardiac_phase, t_common)
        bpm_values = _interp1d_eval(cardiac_times, bpm, t_common, kind="linear", fill_value=np.nan)
        bvp_values = _interp1d_eval(cardiac_times, bvp, t_common, kind="linear", fill_value=np.nan)
        gaze_x_values = _interp1d_eval(gaze_times, gaze_xy[:, 0], t_common, kind="linear", fill_value=np.nan)
        gaze_y_values = _interp1d_eval(gaze_times, gaze_xy[:, 1], t_common, kind="linear", fill_value=np.nan)
        conf_values = _interp1d_eval(gaze_times, gaze_confidence, t_common, kind="nearest", fill_value=0.0)
        blink_values = _interp1d_eval(gaze_times, blink_flags.astype(float), t_common, kind="nearest", fill_value=0.0)
        head_x_interp = None
        head_y_interp = None
        if head_xy is not None and len(head_xy) == len(gaze_times):
            head_x_interp = _interp1d_eval(gaze_times, head_xy[:, 0], t_common, kind="linear", fill_value=np.nan)
            head_y_interp = _interp1d_eval(gaze_times, head_xy[:, 1], t_common, kind="linear", fill_value=np.nan)
        blink_metric_interp = None
        if blink_metric is not None and len(blink_metric) == len(gaze_times):
            blink_metric_interp = _interp1d_eval(gaze_times, blink_metric, t_common, kind="linear", fill_value=np.nan)

        df_dict: Dict[str, Any] = {
            "time_s": t_common,
            "cardiac_phase_rad": phase_sync,
            "heart_bpm": bpm_values,
            "bvp_signal": bvp_values,
            "gaze_x": gaze_x_values,
            "gaze_y": gaze_y_values,
            "gaze_confidence": conf_values,
            "blink": blink_values > 0.5,
        }
        if head_x_interp is not None and head_y_interp is not None:
            df_dict["head_x"] = head_x_interp
            df_dict["head_y"] = head_y_interp
        if blink_metric_interp is not None:
            df_dict["blink_metric"] = blink_metric_interp

        df = pd.DataFrame(df_dict)

        df["gaze_valid"] = (
            df["gaze_confidence"] >= self.config.eye_confidence_threshold
        ) & df["gaze_x"].notna() & df["gaze_y"].notna()
        df["sync_valid"] = df["gaze_valid"] & df["cardiac_phase_rad"].notna()
        print(f"[TRACE] Synchronizer created dataframe with shape={df.shape}", file=sys.stderr, flush=True)
        return df


def add_eye_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = out["time_s"].diff().to_numpy(dtype=float)
    dx = out["gaze_x"].diff().to_numpy(dtype=float)
    dy = out["gaze_y"].diff().to_numpy(dtype=float)

    vel_x = np.divide(dx, dt, out=np.full_like(dx, np.nan), where=np.isfinite(dt) & (dt > 0))
    vel_y = np.divide(dy, dt, out=np.full_like(dy, np.nan), where=np.isfinite(dt) & (dt > 0))
    speed = np.sqrt(vel_x ** 2 + vel_y ** 2)

    invalid = ~out["gaze_valid"].to_numpy(dtype=bool)
    vel_x[invalid] = np.nan
    vel_y[invalid] = np.nan
    speed[invalid] = np.nan

    out["gaze_velocity_x"] = vel_x
    out["gaze_velocity_y"] = vel_y
    out["gaze_velocity_abs"] = speed
    out["gaze_displacement"] = np.sqrt(dx ** 2 + dy ** 2)
    out.loc[invalid, "gaze_displacement"] = np.nan

    if {"head_x", "head_y"}.issubset(out.columns):
        head_x = out["head_x"].to_numpy(dtype=float)
        head_y = out["head_y"].to_numpy(dtype=float)
        head_dx = np.diff(head_x, prepend=np.nan)
        head_dy = np.diff(head_y, prepend=np.nan)
        head_vel_x = np.divide(head_dx, dt, out=np.full_like(head_dx, np.nan), where=np.isfinite(dt) & (dt > 0))
        head_vel_y = np.divide(head_dy, dt, out=np.full_like(head_dy, np.nan), where=np.isfinite(dt) & (dt > 0))
        comp_vel_x = vel_x - head_vel_x
        comp_vel_y = vel_y - head_vel_y
        head_speed = np.sqrt(head_vel_x ** 2 + head_vel_y ** 2)
        comp_speed = np.sqrt(comp_vel_x ** 2 + comp_vel_y ** 2)

        invalid_head = invalid | ~np.isfinite(head_x) | ~np.isfinite(head_y)
        head_speed[invalid_head] = np.nan
        head_vel_x[invalid_head] = np.nan
        head_vel_y[invalid_head] = np.nan
        comp_vel_x[invalid_head] = np.nan
        comp_vel_y[invalid_head] = np.nan
        comp_speed[invalid_head] = np.nan

        out["head_velocity_x"] = head_vel_x
        out["head_velocity_y"] = head_vel_y
        out["head_velocity_abs"] = head_speed
        out["gaze_velocity_compensated_x"] = comp_vel_x
        out["gaze_velocity_compensated_y"] = comp_vel_y
        out["gaze_velocity_compensated_abs"] = comp_speed
        out["analysis_velocity_x"] = np.where(np.isfinite(comp_vel_x), comp_vel_x, vel_x)
        out["analysis_velocity_y"] = np.where(np.isfinite(comp_vel_y), comp_vel_y, vel_y)
        out["analysis_velocity_abs"] = np.where(np.isfinite(comp_speed), comp_speed, speed)
    else:
        out["analysis_velocity_x"] = vel_x
        out["analysis_velocity_y"] = vel_y
        out["analysis_velocity_abs"] = speed
    return out


def add_eye_movement_state_features(
    df: pd.DataFrame,
    config: MultimodalExtractionConfig,
) -> pd.DataFrame:
    out = df.copy()
    velocity = out["analysis_velocity_abs"].to_numpy(dtype=float)
    valid = out["sync_valid"].to_numpy(dtype=bool) & np.isfinite(velocity)
    finite_velocity = velocity[valid]

    if len(finite_velocity) == 0:
        out["movement_state"] = "unclassified"
        out["movement_state_code"] = -1
        out["movement_onset"] = False
        return out

    threshold = max(
        float(np.nanquantile(finite_velocity, config.movement_velocity_quantile)),
        float(config.movement_velocity_floor),
    )
    moving = np.isfinite(velocity) & (velocity >= threshold) & valid

    dt = out["time_s"].diff().to_numpy(dtype=float)
    median_dt = float(np.nanmedian(dt[np.isfinite(dt) & (dt > 0)])) if np.any(np.isfinite(dt) & (dt > 0)) else math.nan
    context_samples = 1
    if np.isfinite(median_dt) and median_dt > 0:
        context_samples = max(1, int(round(config.movement_context_s / median_dt)))

    just_about_to_move = np.zeros(len(out), dtype=bool)
    just_moved = np.zeros(len(out), dtype=bool)
    for offset in range(1, context_samples + 1):
        just_about_to_move[:-offset] |= moving[offset:]
        just_moved[offset:] |= moving[:-offset]

    just_about_to_move &= ~moving
    just_moved &= ~moving & ~just_about_to_move
    fixating = valid & ~moving & ~just_about_to_move & ~just_moved

    state = np.full(len(out), "unclassified", dtype=object)
    state[fixating] = "fixating"
    state[just_moved] = "just_moved"
    state[just_about_to_move] = "just_about_to_move"
    state[moving] = "moving"
    onset = moving & ~np.roll(moving, 1)
    if len(onset):
        onset[0] = moving[0]

    code_map = {
        "fixating": 0,
        "just_moved": 1,
        "just_about_to_move": 2,
        "moving": 3,
        "unclassified": -1,
    }
    out["movement_state"] = state
    out["movement_state_code"] = [code_map.get(str(value), -1) for value in state]
    out["movement_onset"] = onset & valid
    out["movement_velocity_threshold"] = threshold
    return out


def circular_phase_summary(df: pd.DataFrame, config: MultimodalExtractionConfig) -> pd.DataFrame:
    velocity_col = "analysis_velocity_abs" if "analysis_velocity_abs" in df.columns else "gaze_velocity_abs"
    valid = df["sync_valid"] & df[velocity_col].notna()
    phase = df.loc[valid, "cardiac_phase_rad"].to_numpy(dtype=float)
    speed = df.loc[valid, velocity_col].to_numpy(dtype=float)

    edges = np.linspace(0.0, 2 * np.pi, config.phase_bin_count + 1)
    bin_ids = np.digitize(phase, edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, config.phase_bin_count - 1)

    rows: List[Dict[str, Any]] = []
    for idx in range(config.phase_bin_count):
        mask = bin_ids == idx
        values = speed[mask]
        phase_values = phase[mask]
        row: Dict[str, Any] = {
            "phase_bin": idx,
            "phase_start_rad": edges[idx],
            "phase_end_rad": edges[idx + 1],
            "phase_center_rad": (edges[idx] + edges[idx + 1]) / 2.0,
            "sample_count": int(np.sum(mask)),
            "mean_abs_velocity": float(np.nanmean(values)) if np.sum(mask) else math.nan,
            "median_abs_velocity": float(np.nanmedian(values)) if np.sum(mask) else math.nan,
            "std_abs_velocity": float(np.nanstd(values, ddof=1)) if np.sum(mask) > 1 else math.nan,
            "mean_phase_rad": _circular_mean(phase_values) if np.sum(mask) else math.nan,
        }
        row.update(_bayesian_normal_posterior(values, config))
        rows.append(row)
    return pd.DataFrame(rows)


def blink_phase_summary(df: pd.DataFrame, config: MultimodalExtractionConfig) -> pd.DataFrame:
    valid = df["sync_valid"] & df["cardiac_phase_rad"].notna()
    phase = df.loc[valid, "cardiac_phase_rad"].to_numpy(dtype=float)
    blink = df.loc[valid, "blink"].to_numpy(dtype=bool)

    edges = np.linspace(0.0, 2 * np.pi, config.phase_bin_count + 1)
    bin_ids = np.digitize(phase, edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, config.phase_bin_count - 1)

    rows: List[Dict[str, Any]] = []
    for idx in range(config.phase_bin_count):
        mask = bin_ids == idx
        blink_values = blink[mask].astype(float)
        rows.append(
            {
                "phase_bin": idx,
                "phase_start_rad": edges[idx],
                "phase_end_rad": edges[idx + 1],
                "phase_center_rad": (edges[idx] + edges[idx + 1]) / 2.0,
                "sample_count": int(np.sum(mask)),
                "blink_count": int(np.sum(blink_values)),
                "blink_event_rate": float(np.mean(blink_values)) if np.sum(mask) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def movement_state_phase_summary(df: pd.DataFrame, config: MultimodalExtractionConfig) -> pd.DataFrame:
    valid = df["sync_valid"] & df["cardiac_phase_rad"].notna() & df["movement_state"].notna()
    phase = df.loc[valid, "cardiac_phase_rad"].to_numpy(dtype=float)
    states = df.loc[valid, "movement_state"].astype(str).to_numpy()
    edges = np.linspace(0.0, 2 * np.pi, config.phase_bin_count + 1)
    bin_ids = np.digitize(phase, edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, config.phase_bin_count - 1)

    state_order = ["just_about_to_move", "moving", "just_moved", "fixating"]
    rows: List[Dict[str, Any]] = []
    for idx in range(config.phase_bin_count):
        mask = bin_ids == idx
        row: Dict[str, Any] = {
            "phase_bin": idx,
            "phase_start_rad": edges[idx],
            "phase_end_rad": edges[idx + 1],
            "phase_center_rad": (edges[idx] + edges[idx + 1]) / 2.0,
            "sample_count": int(np.sum(mask)),
        }
        for state in state_order:
            state_mask = mask & (states == state)
            row[f"{state}_count"] = int(np.sum(state_mask))
            row[f"{state}_fraction"] = float(np.mean(states[mask] == state)) if np.any(mask) else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _circular_resultant_length(angles: np.ndarray) -> float:
    if len(angles) == 0:
        return math.nan
    return float(np.sqrt(np.mean(np.cos(angles)) ** 2 + np.mean(np.sin(angles)) ** 2))


def compute_phase_modulation_statistics(df: pd.DataFrame, config: MultimodalExtractionConfig) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    valid = df["sync_valid"] & df["cardiac_phase_rad"].notna()
    if not np.any(valid):
        return stats

    valid_df = df.loc[valid].copy()
    phase = valid_df["cardiac_phase_rad"].to_numpy(dtype=float)
    blink = valid_df["blink"].to_numpy(dtype=bool)
    if np.any(blink):
        blink_phase = phase[blink]
        stats["blink_preferred_phase_rad"] = _circular_mean(blink_phase)
        stats["blink_phase_resultant_length"] = _circular_resultant_length(blink_phase)
    else:
        stats["blink_preferred_phase_rad"] = math.nan
        stats["blink_phase_resultant_length"] = math.nan

    onset_mask = valid_df["movement_onset"].to_numpy(dtype=bool) if "movement_onset" in valid_df.columns else np.zeros(len(valid_df), dtype=bool)
    if np.any(onset_mask):
        onset_phase = phase[onset_mask]
        stats["movement_onset_preferred_phase_rad"] = _circular_mean(onset_phase)
        stats["movement_onset_resultant_length"] = _circular_resultant_length(onset_phase)
    else:
        stats["movement_onset_preferred_phase_rad"] = math.nan
        stats["movement_onset_resultant_length"] = math.nan

    edges = np.linspace(0.0, 2 * np.pi, config.phase_bin_count + 1)
    phase_bins = np.digitize(phase, edges, right=False) - 1
    phase_bins = np.clip(phase_bins, 0, config.phase_bin_count - 1)

    if SCIPY_AVAILABLE and chi2_contingency is not None:
        contingency = np.zeros((config.phase_bin_count, 2), dtype=int)
        for idx in range(config.phase_bin_count):
            mask = phase_bins == idx
            contingency[idx, 0] = int(np.sum(blink[mask]))
            contingency[idx, 1] = int(np.sum(mask) - np.sum(blink[mask]))
        if np.sum(contingency[:, 0]) > 0 and np.sum(contingency[:, 1]) > 0:
            chi2, pvalue, _, _ = chi2_contingency(contingency)
            stats["blink_phase_chi2"] = float(chi2)
            stats["blink_phase_pvalue"] = float(pvalue)

        if "movement_state" in valid_df.columns:
            state_order = ["just_about_to_move", "moving", "just_moved", "fixating"]
            state_matrix = np.zeros((config.phase_bin_count, len(state_order)), dtype=int)
            states = valid_df["movement_state"].astype(str).to_numpy()
            for idx in range(config.phase_bin_count):
                mask = phase_bins == idx
                for jdx, state in enumerate(state_order):
                    state_matrix[idx, jdx] = int(np.sum(states[mask] == state))
            if np.sum(state_matrix) > 0 and np.all(np.sum(state_matrix, axis=0) > 0):
                chi2, pvalue, _, _ = chi2_contingency(state_matrix)
                stats["movement_state_phase_chi2"] = float(chi2)
                stats["movement_state_phase_pvalue"] = float(pvalue)

    velocity_col = "analysis_velocity_abs" if "analysis_velocity_abs" in valid_df.columns else "gaze_velocity_abs"
    if SCIPY_AVAILABLE and kruskal is not None and velocity_col in valid_df.columns:
        grouped = []
        for idx in range(config.phase_bin_count):
            values = valid_df.loc[phase_bins == idx, velocity_col].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values):
                grouped.append(values)
        if len(grouped) >= 2:
            h_stat, pvalue = kruskal(*grouped)
            stats["velocity_phase_kruskal_h"] = float(h_stat)
            stats["velocity_phase_pvalue"] = float(pvalue)

    return stats


def _bayesian_normal_posterior(values: np.ndarray, config: MultimodalExtractionConfig) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "posterior_mean": math.nan,
            "posterior_sd": math.nan,
            "posterior_ci_low": math.nan,
            "posterior_ci_high": math.nan,
        }

    n = len(values)
    mean_x = float(np.mean(values))
    ss = float(np.sum((values - mean_x) ** 2))

    mu0 = config.posterior_mu0
    kappa0 = config.posterior_kappa0
    alpha0 = config.posterior_df0 / 2.0
    beta0 = config.posterior_df0 * (config.posterior_sigma0 ** 2) / 2.0

    kappa_n = kappa0 + n
    mu_n = (kappa0 * mu0 + n * mean_x) / kappa_n
    alpha_n = alpha0 + n / 2.0
    beta_n = beta0 + 0.5 * ss + (kappa0 * n * (mean_x - mu0) ** 2) / (2.0 * kappa_n)

    if alpha_n <= 1:
        posterior_var = math.nan
    else:
        posterior_var = beta_n / (alpha_n - 1.0) / kappa_n

    scale = math.sqrt(max(beta_n / (alpha_n * kappa_n), 0.0))
    dof = 2.0 * alpha_n
    ci_low, ci_high = _confidence_interval_95(mu_n, scale, dof)
    return {
        "posterior_mean": float(mu_n),
        "posterior_sd": float(math.sqrt(posterior_var)) if np.isfinite(posterior_var) else math.nan,
        "posterior_ci_low": float(ci_low),
        "posterior_ci_high": float(ci_high),
    }


def compute_global_summary(
    df: pd.DataFrame,
    phase_summary: pd.DataFrame,
    blink_summary: pd.DataFrame,
    config: MultimodalExtractionConfig,
    phase_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    velocity_col = "analysis_velocity_abs" if "analysis_velocity_abs" in df.columns else "gaze_velocity_abs"
    valid = df["sync_valid"] & df[velocity_col].notna()
    sync_ratio = float(df["sync_valid"].mean()) if len(df) else math.nan
    phase_values = df.loc[valid, "cardiac_phase_rad"].to_numpy(dtype=float)
    velocity_values = df.loc[valid, velocity_col].to_numpy(dtype=float)

    vector_x = np.sum(velocity_values * np.cos(phase_values)) if len(velocity_values) else math.nan
    vector_y = np.sum(velocity_values * np.sin(phase_values)) if len(velocity_values) else math.nan
    weight_sum = np.sum(velocity_values) if len(velocity_values) else math.nan
    preferred_phase = math.atan2(vector_y, vector_x) % (2 * np.pi) if np.isfinite(vector_x) and np.isfinite(vector_y) else math.nan
    coupling_strength = math.sqrt(vector_x ** 2 + vector_y ** 2) / weight_sum if np.isfinite(weight_sum) and weight_sum > 0 else math.nan

    low_sample_bins = int(np.sum(phase_summary["sample_count"] < config.min_phase_bin_samples))
    duration_s = float(df["time_s"].iloc[-1] - df["time_s"].iloc[0]) if len(df) > 1 else math.nan
    blink_count = int(df["blink"].sum()) if "blink" in df else 0
    valid_phase_summary = phase_summary.loc[
        phase_summary["sample_count"] >= config.min_phase_bin_samples,
        "mean_abs_velocity",
    ].to_numpy(dtype=float)
    valid_blink_summary = blink_summary.loc[
        blink_summary["sample_count"] >= config.min_phase_bin_samples,
        "blink_event_rate",
    ].to_numpy(dtype=float)
    summary = {
        "sample_count": int(len(df)),
        "sync_valid_fraction": sync_ratio,
        "mean_heart_bpm": float(df["heart_bpm"].mean(skipna=True)),
        "mean_abs_eye_velocity": float(df["gaze_velocity_abs"].mean(skipna=True)),
        "median_abs_eye_velocity": float(df["gaze_velocity_abs"].median(skipna=True)),
        "mean_abs_eye_velocity_compensated": float(
            df.get("gaze_velocity_compensated_abs", df["gaze_velocity_abs"]).mean(skipna=True)
        ),
        "mean_head_velocity": float(df.get("head_velocity_abs", pd.Series(dtype=float)).mean(skipna=True))
        if "head_velocity_abs" in df
        else math.nan,
        "blink_count": blink_count,
        "blink_rate_hz": float(blink_count / duration_s) if np.isfinite(duration_s) and duration_s > 0 else math.nan,
        "preferred_cardiac_phase_rad": preferred_phase,
        "velocity_phase_coupling_strength": float(coupling_strength) if np.isfinite(coupling_strength) else math.nan,
        "saccade_modulation_depth_pct": _compute_modulation_depth_pct(valid_phase_summary),
        "blink_modulation_depth_pct": _compute_modulation_depth_pct(valid_blink_summary),
        "phase_bins_below_min_samples": low_sample_bins,
    }
    if phase_stats:
        summary["phase_statistics"] = phase_stats
    return summary


class LiveFrameAnalyzer:
    def __init__(
        self,
        config: Optional[MultimodalExtractionConfig] = None,
        analysis_window_s: Optional[float] = None,
    ):
        self.config = config or MultimodalExtractionConfig(video_path="live_stream")
        self.analysis_window_s = analysis_window_s
        self.cardiac_extractor = CardiacPhaseExtractor(self.config)
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

        self.times: Deque[float] = collections.deque()
        self.raw_green: Deque[float] = collections.deque()
        self.gaze_xy: Deque[np.ndarray] = collections.deque()
        self.gaze_confidence: Deque[float] = collections.deque()
        self.blink_metric: Deque[float] = collections.deque()
        self.blink_flag: Deque[bool] = collections.deque()
        self.head_xy: Deque[np.ndarray] = collections.deque()

        self.prev_gray: Optional[np.ndarray] = None
        self.prev_points: Optional[np.ndarray] = None
        self.prev_relative_position: Optional[np.ndarray] = None
        self.prev_head_center: Optional[np.ndarray] = None

    def process_frame(self, frame_bgr: np.ndarray, timestamp_s: float) -> Tuple[LiveMetrics, pd.DataFrame, Dict[str, Any]]:
        gaze_xy, confidence, blink_metric, head_xy, overlays = self._process_eye_frame(frame_bgr)
        green_signal = self._extract_live_green_signal(frame_bgr, overlays.get("face_rect"))

        self.times.append(timestamp_s)
        self.raw_green.append(green_signal)
        self.gaze_xy.append(gaze_xy)
        self.gaze_confidence.append(confidence)
        self.blink_metric.append(blink_metric)
        self.head_xy.append(head_xy)
        self._trim_buffers(timestamp_s)

        blink_flags = detect_blink_events(
            np.asarray(self.blink_metric, dtype=float),
            np.asarray(self.times, dtype=float),
            self.config,
        )
        self.blink_flag = collections.deque(blink_flags.tolist(), maxlen=len(self.times))

        live_df, summary = self._analyze_buffers()
        metrics = LiveMetrics(
            timestamp_s=timestamp_s,
            heart_bpm=float(summary.get("mean_heart_bpm", math.nan)),
            saccade_modulation_pct=float(summary.get("saccade_modulation_depth_pct", math.nan)),
            blink_modulation_pct=float(summary.get("blink_modulation_depth_pct", math.nan)),
            mean_eye_velocity=float(summary.get("mean_abs_eye_velocity", math.nan)),
            mean_eye_velocity_compensated=float(summary.get("mean_abs_eye_velocity_compensated", math.nan)),
            blink_rate_hz=float(summary.get("blink_rate_hz", math.nan)),
            sync_valid_fraction=float(summary.get("sync_valid_fraction", math.nan)),
        )
        return metrics, live_df, overlays

    def _trim_buffers(self, timestamp_s: float) -> None:
        if self.analysis_window_s is None:
            return
        cutoff = timestamp_s - self.analysis_window_s
        while self.times and self.times[0] < cutoff:
            self.times.popleft()
            self.raw_green.popleft()
            self.gaze_xy.popleft()
            self.gaze_confidence.popleft()
            self.blink_metric.popleft()
            self.head_xy.popleft()

    def _extract_live_green_signal(
        self,
        frame_bgr: np.ndarray,
        face_rect: Optional[Tuple[int, int, int, int]],
    ) -> float:
        if face_rect is None:
            h, w = frame_bgr.shape[:2]
            face_rect = (w // 4, h // 6, 3 * w // 4, 5 * h // 6)
        x0, y0, x1, y1 = face_rect
        roi = frame_bgr[y0:y1, x0:x1]
        if roi.size == 0:
            return math.nan
        return float(np.mean(roi[:, :, 1]))

    def _process_eye_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float, float, np.ndarray, Dict[str, Any]]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        face_rect = _detect_face_roi(frame_bgr, self.config.face_margin_px)
        movement_xy, confidence, head_xy, eye_boxes = _estimate_eye_motion_sample(
            gray_frame=gray,
            frame_shape=gray.shape[:2],
            face_rect=face_rect,
            eye_cascade=self.eye_cascade,
        )
        blink_metric = _estimate_blink_metric(gray, eye_boxes)
        return movement_xy, confidence, blink_metric, head_xy, {"face_rect": face_rect, "eye_boxes": eye_boxes}

    def _analyze_buffers(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        times = np.asarray(self.times, dtype=float)
        if len(times) < 10:
            print(f"[TRACE] _analyze_buffers returning empty output: len(times)={len(times)} < 10", file=sys.stderr)
            return pd.DataFrame(), {}

        raw_green = np.asarray(self.raw_green, dtype=float)
        raw_green = _moving_average_nan(raw_green, max(3, int(round(len(raw_green) * 0.05))))
        raw_green = _interpolate_nan(raw_green)
        dt = np.diff(times)
        median_dt = float(np.nanmedian(dt)) if len(dt) else math.nan
        if not np.isfinite(median_dt) or median_dt <= 0:
            print(f"[TRACE] _analyze_buffers returning empty output: invalid median_dt={median_dt}", file=sys.stderr)
            return pd.DataFrame(), {}
        fps = 1.0 / median_dt

        try:
            filtered = self.cardiac_extractor._bandpass_filter(_detrend(raw_green), fps)
        except Exception as exc:
            print("[TRACE] Bandpass filtering failed; falling back to detrended signal:", repr(exc), file=sys.stderr)
            traceback.print_exc()
            logger.exception("Bandpass filtering failed; falling back to detrended signal: %s", exc)
            filtered = _detrend(raw_green)
        peaks = self.cardiac_extractor._find_bvp_peaks(filtered, fps)
        peak_times = times[peaks] if len(peaks) else np.array([], dtype=float)
        phase = _phase_from_peaks(times, peak_times)
        analytic_phase = _hilbert_phase(filtered)
        phase = np.where(np.isfinite(phase), phase, analytic_phase)

        bpm = np.full(len(times), np.nan, dtype=float)
        if len(peak_times) >= 2:
            ibi = np.diff(peak_times)
            instant_bpm = 60.0 / ibi
            midpoint = peak_times[:-1] + ibi / 2.0
            if len(midpoint) >= 2:
                bpm = _interp1d_eval(midpoint, instant_bpm, times, fill_value="extrapolate")
            else:
                bpm[:] = instant_bpm[0]

        gaze_xy = np.asarray(self.gaze_xy, dtype=float)
        gaze_confidence = np.asarray(self.gaze_confidence, dtype=float)
        blink_flags = np.asarray(self.blink_flag, dtype=bool)
        head_xy = np.asarray(self.head_xy, dtype=float)
        blink_metric = np.asarray(self.blink_metric, dtype=float)

        df = MultimodalSynchronizer(self.config).synchronize(
            cardiac_phase=phase,
            cardiac_times=times,
            bpm=bpm,
            bvp=filtered,
            gaze_xy=gaze_xy,
            gaze_times=times,
            gaze_confidence=gaze_confidence,
            blink_flags=blink_flags,
            head_xy=head_xy,
            blink_metric=blink_metric,
        )
        df = add_eye_velocity_features(df)
        df = add_eye_movement_state_features(df, self.config)
        phase_summary = circular_phase_summary(df, self.config)
        blink_summary = blink_phase_summary(df, self.config)
        phase_stats = compute_phase_modulation_statistics(df, self.config)
        summary = compute_global_summary(df, phase_summary, blink_summary, self.config, phase_stats=phase_stats)
        return df, summary


def extract_cardiac_and_eye_timeseries(
    video_path: str,
    output_csv: Optional[str] = None,
    config: Optional[MultimodalExtractionConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")
    if config is None:
        config = MultimodalExtractionConfig(video_path=str(video))

    print(f"[TRACE] Starting extraction for video: {video}", file=sys.stderr, flush=True)
    cardiac_extractor = CardiacPhaseExtractor(config)
    phase, cardiac_times, bpm, bvp = cardiac_extractor.extract(str(video))
    print("[TRACE] Cardiac extraction returned successfully", file=sys.stderr, flush=True)

    eye_extractor = EyeMovementExtractor(config)
    gaze_xy, gaze_times, gaze_confidence, blink_flags, head_xy, blink_metric = eye_extractor.extract(str(video))

    synchronizer = MultimodalSynchronizer(config)
    df = synchronizer.synchronize(
        cardiac_phase=phase,
        cardiac_times=cardiac_times,
        bpm=bpm,
        bvp=bvp,
        gaze_xy=gaze_xy,
        gaze_times=gaze_times,
        gaze_confidence=gaze_confidence,
        blink_flags=blink_flags,
        head_xy=head_xy,
        blink_metric=blink_metric,
    )
    df = add_eye_velocity_features(df)
    df = add_eye_movement_state_features(df, config)
    phase_summary = circular_phase_summary(df, config)
    blink_summary = blink_phase_summary(df, config)
    movement_summary = movement_state_phase_summary(df, config)
    phase_stats = compute_phase_modulation_statistics(df, config)
    summary = compute_global_summary(df, phase_summary, blink_summary, config, phase_stats=phase_stats)
    summary["movement_state_phase_summary"] = movement_summary.to_dict(orient="records")
    cardiac_diagnostics = getattr(cardiac_extractor, "last_diagnostics", None)
    if cardiac_diagnostics:
        summary["cardiac_diagnostics"] = cardiac_diagnostics

    if output_csv:
        df.to_csv(output_csv, index=False)
    return df, phase_summary, blink_summary, summary


def save_outputs(
    df: pd.DataFrame,
    phase_summary: pd.DataFrame,
    blink_summary: pd.DataFrame,
    summary: Dict[str, Any],
    config: MultimodalExtractionConfig,
) -> Dict[str, Path]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(config.video_path).stem

    timeseries_path = output_dir / f"{stem}_timeseries.csv"
    phase_summary_path = output_dir / f"{stem}_phase_summary.csv"
    blink_summary_path = output_dir / f"{stem}_blink_phase_summary.csv"
    movement_summary_path = output_dir / f"{stem}_movement_state_phase_summary.csv"
    summary_path = output_dir / f"{stem}_summary.json"
    config_path = output_dir / f"{stem}_config.json"
    array_path = output_dir / f"{stem}_timeseries.npy"
    array_columns_path = output_dir / f"{stem}_timeseries_columns.json"
    cardiac_diag_json_path = output_dir / f"{stem}_cardiac_diagnostics.json"
    cardiac_diag_csv_path = output_dir / f"{stem}_cardiac_method_diagnostics.csv"

    df.to_csv(timeseries_path, index=False)
    phase_summary.to_csv(phase_summary_path, index=False)
    blink_summary.to_csv(blink_summary_path, index=False)
    movement_summary_records = summary.get("movement_state_phase_summary", [])
    if movement_summary_records:
        pd.DataFrame(movement_summary_records).to_csv(movement_summary_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    config_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    array_columns = [
        "time_s",
        "cardiac_phase_rad",
        "heart_bpm",
        "bvp_signal",
        "gaze_x",
        "gaze_y",
        "analysis_velocity_x",
        "analysis_velocity_y",
        "analysis_velocity_abs",
        "blink",
        "movement_state_code",
        "gaze_confidence",
        "sync_valid",
    ]
    available_columns = [column for column in array_columns if column in df.columns]
    array_df = df.loc[:, available_columns].copy()
    for column in ("blink", "sync_valid"):
        if column in array_df.columns:
            array_df[column] = array_df[column].astype(float)
    np.save(array_path, array_df.to_numpy(dtype=float, copy=True))
    array_columns_path.write_text(json.dumps(available_columns, indent=2), encoding="utf-8")

    outputs = {
        "timeseries_csv": timeseries_path,
        "phase_summary_csv": phase_summary_path,
        "blink_phase_summary_csv": blink_summary_path,
        "timeseries_npy": array_path,
        "timeseries_columns_json": array_columns_path,
        "summary_json": summary_path,
        "config_json": config_path,
    }
    if movement_summary_records:
        outputs["movement_state_phase_summary_csv"] = movement_summary_path
    cardiac_diag = summary.get("cardiac_diagnostics")
    if isinstance(cardiac_diag, dict) and cardiac_diag:
        cardiac_diag_json_path.write_text(json.dumps(cardiac_diag, indent=2), encoding="utf-8")
        outputs["cardiac_diagnostics_json"] = cardiac_diag_json_path
        method_rows = cardiac_diag.get("methods")
        if isinstance(method_rows, list) and method_rows:
            pd.DataFrame(method_rows).to_csv(cardiac_diag_csv_path, index=False)
            outputs["cardiac_method_diagnostics_csv"] = cardiac_diag_csv_path

    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract eye movement and cardiac-cycle summaries from a video."
    )
    parser.add_argument("video_path", help="Path to an input video file")
    parser.add_argument("--output-dir", default="outputs", help="Directory for generated outputs")
    parser.add_argument("--output-fps", type=float, default=None, help="Optional resampling FPS")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap for debugging")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Cardiac extraction methods to run, e.g. --methods cpu_CHROM cpu_POS cpu_GREEN",
    )
    parser.add_argument("--phase-bins", type=int, default=12, help="Number of cardiac phase bins")
    parser.add_argument(
        "--eye-confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum confidence retained for eye samples",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional direct path for the synchronized time-series CSV",
    )
    parser.add_argument(
        "--cardiac-backend",
        choices=["auto", "pyvhr"],
        default="pyvhr",
        help="Cardiac extraction backend. pyVHR only; auto is retained as an alias for pyVHR.",
    )
    parser.add_argument(
        "--pyvhr-timeout-s",
        type=float,
        default=120.0,
        help="Timeout for isolated pyVHR extraction before falling back.",
    )
    parser.add_argument(
        "--isolate-pyvhr",
        action="store_true",
        help="Run each pyVHR method in an isolated child process.",
    )
    parser.add_argument(
        "--no-isolated-pyvhr-preflight",
        action="store_true",
        help="Run pyVHR import preflight in-process instead of a spawned child process.",
    )
    parser.add_argument(
        "--pyvhr-preflight-timeout-s",
        type=float,
        default=30.0,
        help="Timeout for isolated pyVHR import preflight.",
    )
    return parser


def process_video(video_path: str, **config_kwargs: Any) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Path]]:
    config = MultimodalExtractionConfig(video_path=video_path, **config_kwargs)
    df, phase_summary, blink_summary, summary = extract_cardiac_and_eye_timeseries(
        video_path=video_path,
        output_csv=config.output_csv,
        config=config,
    )
    outputs = save_outputs(df, phase_summary, blink_summary, summary, config)
    return df, phase_summary, blink_summary, summary, outputs


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    print(f"[TRACE] Parsed arguments: {args}", file=sys.stderr, flush=True)
    rppg_methods = tuple(_normalise_method_list(args.methods)) if args.methods else MultimodalExtractionConfig(video_path=args.video_path).rppg_methods
    config = MultimodalExtractionConfig(
        video_path=args.video_path,
        output_dir=args.output_dir,
        output_fps=args.output_fps,
        max_frames=args.max_frames,
        rppg_methods=rppg_methods,
        rppg_method=rppg_methods[0] if len(rppg_methods) else "cpu_CHROM",
        phase_bin_count=args.phase_bins,
        eye_confidence_threshold=args.eye_confidence_threshold,
        output_csv=args.output_csv,
        cardiac_backend=args.cardiac_backend,
        isolate_pyvhr=getattr(args, "isolate_pyvhr", False),
        pyvhr_timeout_s=args.pyvhr_timeout_s,
        pyvhr_preflight_isolated=not getattr(args, "no_isolated_pyvhr_preflight", False),
        pyvhr_preflight_timeout_s=getattr(args, "pyvhr_preflight_timeout_s", 30.0),
    )

    try:
        df, phase_summary, blink_summary, summary = extract_cardiac_and_eye_timeseries(
            video_path=args.video_path,
            output_csv=args.output_csv,
            config=config,
        )
        outputs = save_outputs(df, phase_summary, blink_summary, summary, config)
    except Exception as exc:
        print("[TRACE] Pipeline failed:", repr(exc), file=sys.stderr)
        traceback.print_exc()
        logger.exception("Pipeline failed: %s", exc)
        return 1

    logger.info("Pipeline complete")
    for name, path in outputs.items():
        logger.info("%s: %s", name, path)
    logger.info("Summary: %s", json.dumps(summary, indent=2))
    return 0


def _collect_python_site_paths() -> List[str]:
    paths: List[str] = []
    for getter_name in ("getsitepackages",):
        getter = getattr(site, getter_name, None)
        if getter is None:
            continue
        try:
            values = getter()
        except Exception:
            continue
        if isinstance(values, (list, tuple)):
            for value in values:
                if value and value not in paths:
                    paths.append(str(value))
    try:
        user_site = site.getusersitepackages()
        if isinstance(user_site, str) and user_site and user_site not in paths:
            paths.append(user_site)
    except Exception:
        pass
    return paths


def _make_import_context() -> Dict[str, Any]:
    return {
        "sys_path": list(dict.fromkeys(str(p) for p in sys.path if p)),
        "site_paths": _collect_python_site_paths(),
        "sys_executable": str(sys.executable),
        "base_executable": str(getattr(sys, "_base_executable", sys.executable)),
        "sys_prefix": str(sys.prefix),
        "cwd": os.getcwd(),
        "pythonpath": os.environ.get("PYTHONPATH", ""),
    }


def _apply_import_context(import_context: Optional[Dict[str, Any]]) -> None:
    if not import_context:
        return
    prepend: List[str] = []
    for key in ("sys_path", "site_paths"):
        for entry in import_context.get(key, []) or []:
            entry = str(entry)
            if entry and entry not in prepend:
                prepend.append(entry)
    for entry in reversed(prepend):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    pythonpath = str(import_context.get("pythonpath", "") or "")
    if pythonpath:
        existing = os.environ.get("PYTHONPATH", "")
        if existing:
            os.environ["PYTHONPATH"] = pythonpath + os.pathsep + existing
        else:
            os.environ["PYTHONPATH"] = pythonpath


def _preflight_pyvhr_import(import_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    _apply_import_context(import_context)
    pyver = tuple(sys.version_info[:3])
    if pyver < (3, 9):
        raise RuntimeError(
            "pyVHR imports MediaPipe, and this interpreter is too old. "
            f"Detected Python {pyver[0]}.{pyver[1]}.{pyver[2]} at {sys.executable!r}; "
            "MediaPipe wheels support Python 3.9-3.12."
        )

    print(f"[TRACE] _preflight_pyvhr_import find_spec sys.executable={sys.executable}", file=sys.stderr, flush=True)
    spec = importlib.util.find_spec("pyVHR")
    if spec is None:
        raise ModuleNotFoundError(
            "Could not find module 'pyVHR'. Install it in the interpreter running this script. "
            f"sys.executable={sys.executable!r}, sys.prefix={sys.prefix!r}"
        )

    print("[TRACE] _preflight_pyvhr_import import pyVHR start", file=sys.stderr, flush=True)
    module = importlib.import_module("pyVHR")
    print("[TRACE] _preflight_pyvhr_import import saccard start", file=sys.stderr, flush=True)
    saccard_module = importlib.import_module("saccard")
    print("[TRACE] _preflight_pyvhr_import imports done", file=sys.stderr, flush=True)

    saccard_fn = getattr(saccard_module, "saccard", None)
    if saccard_fn is None or not callable(saccard_fn):
        raise ImportError("Imported package 'saccard', but callable 'saccard' was not found")
    return {
        "module_file": str(getattr(module, "__file__", "")),
        "saccard_module_file": str(getattr(saccard_module, "__file__", "")),
        "sys_executable": str(sys.executable),
        "sys_prefix": str(sys.prefix),
        "python_version": f"{pyver[0]}.{pyver[1]}.{pyver[2]}",
        "sys_path_head": list(sys.path[:8]),
    }

if __name__ == "__main__":
    print("Running cardiac and eye movement synchronization pipeline...", flush=True)
    raise SystemExit(main())
