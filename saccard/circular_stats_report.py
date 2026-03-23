from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import chi2_contingency, chisquare


STATE_LABELS = {
    -1: "unclassified",
    0: "fixating",
    1: "just_moved",
    2: "just_about_to_move",
    3: "moving",
}

STATE_ORDER = ["just_about_to_move", "moving", "just_moved", "fixating"]


def circular_mean(angles: np.ndarray, weights: np.ndarray | None = None) -> float:
    if len(angles) == 0:
        return math.nan
    if weights is None:
        weights = np.ones(len(angles), dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = np.sum(weights)
    if total <= 0:
        return math.nan
    x = np.sum(weights * np.cos(angles)) / total
    y = np.sum(weights * np.sin(angles)) / total
    return float(np.mod(np.arctan2(y, x), 2.0 * np.pi))


def resultant_length(angles: np.ndarray, weights: np.ndarray | None = None) -> float:
    if len(angles) == 0:
        return math.nan
    if weights is None:
        weights = np.ones(len(angles), dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = np.sum(weights)
    if total <= 0:
        return math.nan
    x = np.sum(weights * np.cos(angles)) / total
    y = np.sum(weights * np.sin(angles)) / total
    return float(np.sqrt(x * x + y * y))


def load_timeseries(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"cardiac_phase_rad", "sync_valid"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if "blink" in df.columns:
        df["blink"] = df["blink"].astype(bool)
    if "sync_valid" in df.columns:
        df["sync_valid"] = df["sync_valid"].astype(bool)
    if "movement_state_code" in df.columns:
        df["movement_state"] = [STATE_LABELS.get(int(code), "unclassified") for code in df["movement_state_code"].fillna(-1)]
    return df


def phase_bin_edges(phase_bins: int) -> np.ndarray:
    return np.linspace(0.0, 2.0 * np.pi, phase_bins + 1)


def phase_bin_labels(edges: np.ndarray) -> List[str]:
    labels = []
    for idx in range(len(edges) - 1):
        center = 0.5 * (edges[idx] + edges[idx + 1])
        labels.append(f"{math.degrees(center):.0f}°")
    return labels


def assign_phase_bins(phase: np.ndarray, phase_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    edges = phase_bin_edges(phase_bins)
    bin_ids = np.digitize(phase, edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, phase_bins - 1)
    return bin_ids, edges


def summarize_blinks(df: pd.DataFrame, phase_bins: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    valid = df["sync_valid"] & df["cardiac_phase_rad"].notna() & df["blink"].notna()
    phase = df.loc[valid, "cardiac_phase_rad"].to_numpy(dtype=float)
    blink = df.loc[valid, "blink"].to_numpy(dtype=bool)
    bin_ids, edges = assign_phase_bins(phase, phase_bins)

    rows: List[Dict[str, Any]] = []
    event_counts = []
    for idx in range(phase_bins):
        mask = bin_ids == idx
        blink_count = int(np.sum(blink[mask]))
        sample_count = int(np.sum(mask))
        event_counts.append(blink_count)
        rows.append(
            {
                "phase_bin": idx,
                "phase_start_rad": float(edges[idx]),
                "phase_end_rad": float(edges[idx + 1]),
                "phase_center_rad": float(0.5 * (edges[idx] + edges[idx + 1])),
                "sample_count": sample_count,
                "blink_count": blink_count,
                "blink_fraction": float(np.mean(blink[mask])) if sample_count else math.nan,
            }
        )

    blink_phase = phase[blink]
    stats = {
        "blink_event_count": int(np.sum(blink)),
        "blink_preferred_phase_rad": circular_mean(blink_phase),
        "blink_resultant_length": resultant_length(blink_phase),
    }
    if np.sum(event_counts) > 0:
        chi = chisquare(event_counts)
        stats["blink_uniformity_chisquare"] = float(chi.statistic)
        stats["blink_uniformity_pvalue"] = float(chi.pvalue)
    return pd.DataFrame(rows), stats


def summarize_velocity(df: pd.DataFrame, phase_bins: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    velocity_col = "analysis_velocity_abs" if "analysis_velocity_abs" in df.columns else "gaze_velocity_abs"
    valid = df["sync_valid"] & df["cardiac_phase_rad"].notna() & df[velocity_col].notna()
    phase = df.loc[valid, "cardiac_phase_rad"].to_numpy(dtype=float)
    velocity = df.loc[valid, velocity_col].to_numpy(dtype=float)
    bin_ids, edges = assign_phase_bins(phase, phase_bins)

    rows: List[Dict[str, Any]] = []
    for idx in range(phase_bins):
        mask = bin_ids == idx
        values = velocity[mask]
        rows.append(
            {
                "phase_bin": idx,
                "phase_start_rad": float(edges[idx]),
                "phase_end_rad": float(edges[idx + 1]),
                "phase_center_rad": float(0.5 * (edges[idx] + edges[idx + 1])),
                "sample_count": int(np.sum(mask)),
                "mean_velocity": float(np.nanmean(values)) if np.sum(mask) else math.nan,
                "median_velocity": float(np.nanmedian(values)) if np.sum(mask) else math.nan,
            }
        )

    stats = {
        "velocity_preferred_phase_rad": circular_mean(phase, weights=velocity),
        "velocity_resultant_length": resultant_length(phase, weights=velocity),
        "velocity_mean": float(np.nanmean(velocity)) if len(velocity) else math.nan,
    }
    return pd.DataFrame(rows), stats


def summarize_states(df: pd.DataFrame, phase_bins: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if "movement_state" not in df.columns:
        return pd.DataFrame(), {}

    valid = df["sync_valid"] & df["cardiac_phase_rad"].notna() & df["movement_state"].notna()
    phase = df.loc[valid, "cardiac_phase_rad"].to_numpy(dtype=float)
    states = df.loc[valid, "movement_state"].astype(str).to_numpy()
    bin_ids, edges = assign_phase_bins(phase, phase_bins)

    rows: List[Dict[str, Any]] = []
    contingency = np.zeros((phase_bins, len(STATE_ORDER)), dtype=int)
    for idx in range(phase_bins):
        mask = bin_ids == idx
        row: Dict[str, Any] = {
            "phase_bin": idx,
            "phase_start_rad": float(edges[idx]),
            "phase_end_rad": float(edges[idx + 1]),
            "phase_center_rad": float(0.5 * (edges[idx] + edges[idx + 1])),
            "sample_count": int(np.sum(mask)),
        }
        for jdx, state in enumerate(STATE_ORDER):
            count = int(np.sum(states[mask] == state))
            contingency[idx, jdx] = count
            row[f"{state}_count"] = count
            row[f"{state}_fraction"] = float(np.mean(states[mask] == state)) if np.sum(mask) else math.nan
        rows.append(row)

    stats: Dict[str, Any] = {}
    if np.sum(contingency) > 0 and np.all(np.sum(contingency, axis=0) > 0):
        chi2, pvalue, _, _ = chi2_contingency(contingency)
        stats["movement_state_phase_chi2"] = float(chi2)
        stats["movement_state_phase_pvalue"] = float(pvalue)
    return pd.DataFrame(rows), stats


def make_radar_plot(
    labels: List[str],
    traces: List[Tuple[str, np.ndarray]],
    title: str,
) -> go.Figure:
    fig = go.Figure()
    theta = labels + [labels[0]]
    for name, values in traces:
        r = list(np.asarray(values, dtype=float)) + [float(values[0])]
        fig.add_trace(go.Scatterpolar(r=r, theta=theta, mode="lines+markers", fill="toself", name=name))
    fig.update_layout(title=title, polar=dict(radialaxis=dict(visible=True)), showlegend=True)
    return fig


def build_report(df: pd.DataFrame, phase_bins: int) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame], Dict[str, go.Figure]]:
    blink_summary, blink_stats = summarize_blinks(df, phase_bins)
    velocity_summary, velocity_stats = summarize_velocity(df, phase_bins)
    state_summary, state_stats = summarize_states(df, phase_bins)

    edges = phase_bin_edges(phase_bins)
    labels = phase_bin_labels(edges)
    figures = {
        "blink_radar": make_radar_plot(
            labels,
            [("Blink Fraction", blink_summary["blink_fraction"].fillna(0.0).to_numpy(dtype=float))],
            "Blink Modulation by Cardiac Phase",
        ),
        "velocity_radar": make_radar_plot(
            labels,
            [("Mean Velocity", velocity_summary["mean_velocity"].fillna(0.0).to_numpy(dtype=float))],
            "Eye Movement by Cardiac Phase",
        ),
    }
    if not state_summary.empty:
        traces = []
        for state in STATE_ORDER:
            traces.append((state, state_summary[f"{state}_fraction"].fillna(0.0).to_numpy(dtype=float)))
        figures["movement_state_radar"] = make_radar_plot(labels, traces, "Movement State by Cardiac Phase")

    report = {
        "phase_bin_count": phase_bins,
        "blink": blink_stats,
        "velocity": velocity_stats,
        "movement_state": state_stats,
    }
    tables = {
        "blink_summary": blink_summary,
        "velocity_summary": velocity_summary,
    }
    if not state_summary.empty:
        tables["movement_state_summary"] = state_summary
    return report, tables, figures


def infer_prefix(timeseries_path: Path) -> str:
    stem = timeseries_path.stem
    suffix = "_timeseries"
    return stem[:-len(suffix)] if stem.endswith(suffix) else stem


def save_report(
    output_dir: Path,
    prefix: str,
    report: Dict[str, Any],
    tables: Dict[str, pd.DataFrame],
    figures: Dict[str, go.Figure],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{prefix}_circular_stats.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, table in tables.items():
        table.to_csv(output_dir / f"{prefix}_{name}.csv", index=False)
    for name, fig in figures.items():
        fig.write_html(str(output_dir / f"{prefix}_{name}.html"), include_plotlyjs="cdn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run circular statistics and radar plots on saccardiac outputs.")
    parser.add_argument("timeseries_csv", help="Path to the synchronized timeseries CSV produced by saccardiac.")
    parser.add_argument("--phase-bins", type=int, default=12, help="Number of cardiac phase bins.")
    parser.add_argument("--output-dir", default=None, help="Directory for stats tables and radar plots.")
    parser.add_argument("--prefix", default=None, help="Optional output file prefix.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timeseries_path = Path(args.timeseries_csv)
    if not timeseries_path.exists():
        raise FileNotFoundError(f"Timeseries CSV not found: {timeseries_path}")

    output_dir = Path(args.output_dir) if args.output_dir else timeseries_path.parent
    prefix = args.prefix or infer_prefix(timeseries_path)
    df = load_timeseries(timeseries_path)
    report, tables, figures = build_report(df, args.phase_bins)
    save_report(output_dir, prefix, report, tables, figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
