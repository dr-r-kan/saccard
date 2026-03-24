from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import chi2_contingency, chisquare

try:
    import statsmodels.formula.api as smf
    STATSMODELS_AVAILABLE = True
except ImportError:
    smf = None
    STATSMODELS_AVAILABLE = False


STATE_LABELS = {
    -1: "unclassified",
    0: "fixating",
    1: "just_moved",
    2: "just_about_to_move",
    3: "moving",
}

STATE_ORDER = ["just_about_to_move", "moving", "just_moved", "fixating"]


@dataclass
class CohortData:
    df: pd.DataFrame
    source_files: List[Path]
    participants: List[str]


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


def discover_timeseries_files(input_path: Path, pattern: str = "*_timeseries.csv") -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob(pattern))
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def infer_participant_id(timeseries_path: Path) -> str:
    stem = infer_prefix(timeseries_path)
    # Keep participant identity stable for mixed models.
    return stem.strip() if stem.strip() else timeseries_path.stem


def load_cohort_data(files: Sequence[Path]) -> CohortData:
    rows: List[pd.DataFrame] = []
    participants: List[str] = []
    for path in files:
        df = load_timeseries(path)
        participant_id = infer_participant_id(path)
        df = df.copy()
        df["participant_id"] = participant_id
        df["source_file"] = str(path)
        rows.append(df)
        if participant_id not in participants:
            participants.append(participant_id)
    if not rows:
        raise RuntimeError("No timeseries files loaded")
    cohort_df = pd.concat(rows, axis=0, ignore_index=True)
    return CohortData(df=cohort_df, source_files=list(files), participants=participants)


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


def build_participant_weighted_profiles(df: pd.DataFrame, phase_bins: int) -> Dict[str, np.ndarray]:
    velocity_col = "analysis_velocity_abs" if "analysis_velocity_abs" in df.columns else "gaze_velocity_abs"
    required = ["participant_id", "sync_valid", "cardiac_phase_rad", "blink", velocity_col]
    for col in required:
        if col not in df.columns:
            return {
                "blink_fraction": np.full(phase_bins, np.nan, dtype=float),
                "mean_velocity": np.full(phase_bins, np.nan, dtype=float),
            }

    work = df.loc[df["sync_valid"] & df["cardiac_phase_rad"].notna(), ["participant_id", "cardiac_phase_rad", "blink", velocity_col]].copy()
    if work.empty:
        return {
            "blink_fraction": np.full(phase_bins, np.nan, dtype=float),
            "mean_velocity": np.full(phase_bins, np.nan, dtype=float),
        }

    bin_ids, _ = assign_phase_bins(work["cardiac_phase_rad"].to_numpy(dtype=float), phase_bins)
    work["phase_bin"] = bin_ids
    work["blink_num"] = work["blink"].astype(float)

    participant_bin = (
        work.groupby(["participant_id", "phase_bin"], observed=True)
        .agg(
            blink_fraction=("blink_num", "mean"),
            mean_velocity=(velocity_col, "mean"),
        )
        .reset_index()
    )

    agg = participant_bin.groupby("phase_bin", observed=True).agg(
        blink_fraction=("blink_fraction", "mean"),
        mean_velocity=("mean_velocity", "mean"),
    )

    blink_profile = np.full(phase_bins, np.nan, dtype=float)
    velocity_profile = np.full(phase_bins, np.nan, dtype=float)
    for idx, row in agg.iterrows():
        bin_idx = int(idx)
        if 0 <= bin_idx < phase_bins:
            blink_profile[bin_idx] = float(row["blink_fraction"])
            velocity_profile[bin_idx] = float(row["mean_velocity"])
    return {
        "blink_fraction": blink_profile,
        "mean_velocity": velocity_profile,
    }


def fit_circular_lmm(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "engine": "statsmodels.mixedlm" if STATSMODELS_AVAILABLE else "unavailable",
        "available": bool(STATSMODELS_AVAILABLE),
    }
    if not STATSMODELS_AVAILABLE:
        out["error"] = "statsmodels is not installed"
        return out

    if "participant_id" not in df.columns:
        out["error"] = "participant_id column missing; cannot fit mixed-effects model"
        return out

    work = df.copy()
    work["phase_sin"] = np.sin(work["cardiac_phase_rad"].to_numpy(dtype=float))
    work["phase_cos"] = np.cos(work["cardiac_phase_rad"].to_numpy(dtype=float))

    models: Dict[str, Any] = {}

    # Continuous velocity model.
    velocity_col = "analysis_velocity_abs" if "analysis_velocity_abs" in work.columns else "gaze_velocity_abs"
    if velocity_col in work.columns:
        vel = work.loc[
            work["sync_valid"]
            & work[velocity_col].notna()
            & work["cardiac_phase_rad"].notna()
            & work["participant_id"].notna(),
            ["participant_id", velocity_col, "phase_sin", "phase_cos"],
        ].copy()
        vel = vel.rename(columns={velocity_col: "outcome"})
        vel["participant_id"] = vel["participant_id"].astype(str)
        if len(vel) >= 10 and vel["participant_id"].nunique() >= 2:
            try:
                fit = smf.mixedlm("outcome ~ phase_sin + phase_cos", vel, groups=vel["participant_id"]).fit(reml=False)
                b_sin = float(fit.params.get("phase_sin", math.nan))
                b_cos = float(fit.params.get("phase_cos", math.nan))
                models["velocity_lmm"] = {
                    "n": int(len(vel)),
                    "participants": int(vel["participant_id"].nunique()),
                    "fixed_effects": {k: float(v) for k, v in fit.params.to_dict().items()},
                    "pvalues": {k: float(v) for k, v in fit.pvalues.to_dict().items()},
                    "aic": float(fit.aic) if np.isfinite(fit.aic) else math.nan,
                    "bic": float(fit.bic) if np.isfinite(fit.bic) else math.nan,
                    "log_likelihood": float(fit.llf),
                    "circular_amplitude": float(np.sqrt(b_sin ** 2 + b_cos ** 2)),
                    "preferred_phase_rad": float(np.mod(np.arctan2(b_sin, b_cos), 2.0 * np.pi)),
                }
            except Exception as exc:
                models["velocity_lmm"] = {
                    "n": int(len(vel)),
                    "participants": int(vel["participant_id"].nunique()),
                    "error": repr(exc),
                }
        else:
            models["velocity_lmm"] = {
                "n": int(len(vel)),
                "participants": int(vel["participant_id"].nunique()),
                "error": "Insufficient data for MixedLM (need >=10 rows and >=2 participants)",
            }

    # Blink model as Gaussian LMM on binary outcome for participant-level dependence adjustment.
    if "blink" in work.columns:
        blink_df = work.loc[
            work["sync_valid"]
            & work["blink"].notna()
            & work["cardiac_phase_rad"].notna()
            & work["participant_id"].notna(),
            ["participant_id", "blink", "phase_sin", "phase_cos"],
        ].copy()
        blink_df["outcome"] = blink_df["blink"].astype(float)
        blink_df["participant_id"] = blink_df["participant_id"].astype(str)
        if len(blink_df) >= 10 and blink_df["participant_id"].nunique() >= 2:
            try:
                fit = smf.mixedlm("outcome ~ phase_sin + phase_cos", blink_df, groups=blink_df["participant_id"]).fit(reml=False)
                b_sin = float(fit.params.get("phase_sin", math.nan))
                b_cos = float(fit.params.get("phase_cos", math.nan))
                models["blink_lmm_gaussian"] = {
                    "n": int(len(blink_df)),
                    "participants": int(blink_df["participant_id"].nunique()),
                    "fixed_effects": {k: float(v) for k, v in fit.params.to_dict().items()},
                    "pvalues": {k: float(v) for k, v in fit.pvalues.to_dict().items()},
                    "aic": float(fit.aic) if np.isfinite(fit.aic) else math.nan,
                    "bic": float(fit.bic) if np.isfinite(fit.bic) else math.nan,
                    "log_likelihood": float(fit.llf),
                    "circular_amplitude": float(np.sqrt(b_sin ** 2 + b_cos ** 2)),
                    "preferred_phase_rad": float(np.mod(np.arctan2(b_sin, b_cos), 2.0 * np.pi)),
                    "note": "Gaussian mixed model on binary blink outcome used for dependency-adjusted directional effect screening.",
                }
            except Exception as exc:
                models["blink_lmm_gaussian"] = {
                    "n": int(len(blink_df)),
                    "participants": int(blink_df["participant_id"].nunique()),
                    "error": repr(exc),
                }
        else:
            models["blink_lmm_gaussian"] = {
                "n": int(len(blink_df)),
                "participants": int(blink_df["participant_id"].nunique()),
                "error": "Insufficient data for MixedLM (need >=10 rows and >=2 participants)",
            }

    out["models"] = models
    return out


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
    lmm_stats = fit_circular_lmm(df)

    edges = phase_bin_edges(phase_bins)
    labels = phase_bin_labels(edges)
    weighted_profiles = build_participant_weighted_profiles(df, phase_bins)
    figures = {
        "blink_radar": make_radar_plot(
            labels,
            [
                ("Blink Fraction (Pooled)", blink_summary["blink_fraction"].fillna(0.0).to_numpy(dtype=float)),
                ("Blink Fraction (Participant Mean)", np.nan_to_num(weighted_profiles["blink_fraction"], nan=0.0)),
            ],
            "Blink Modulation by Cardiac Phase",
        ),
        "velocity_radar": make_radar_plot(
            labels,
            [
                ("Mean Velocity (Pooled)", velocity_summary["mean_velocity"].fillna(0.0).to_numpy(dtype=float)),
                ("Mean Velocity (Participant Mean)", np.nan_to_num(weighted_profiles["mean_velocity"], nan=0.0)),
            ],
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
        "sample_count": int(len(df)),
        "participant_count": int(df["participant_id"].nunique()) if "participant_id" in df.columns else 1,
        "participants": sorted(df["participant_id"].dropna().astype(str).unique().tolist()) if "participant_id" in df.columns else [],
        "blink": blink_stats,
        "velocity": velocity_stats,
        "movement_state": state_stats,
        "circular_lmm": lmm_stats,
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
    parser.add_argument(
        "input_path",
        nargs="?",
        default="outputs",
        help="Input timeseries CSV file or directory containing *_timeseries.csv files (default: outputs).",
    )
    parser.add_argument("--phase-bins", type=int, default=12, help="Number of cardiac phase bins.")
    parser.add_argument("--output-dir", default=None, help="Directory for stats tables and radar plots.")
    parser.add_argument("--prefix", default=None, help="Optional output file prefix.")
    parser.add_argument(
        "--glob",
        default="*_timeseries.csv",
        help="Glob pattern for timeseries discovery when input_path is a directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path)
    files = discover_timeseries_files(input_path, pattern=args.glob)
    if not files:
        raise RuntimeError(f"No timeseries files found in {input_path} with pattern {args.glob!r}")

    if input_path.is_file():
        output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
        prefix = args.prefix or infer_prefix(input_path)
    else:
        output_dir = Path(args.output_dir) if args.output_dir else input_path
        prefix = args.prefix or "cohort"

    cohort = load_cohort_data(files)
    df = cohort.df
    report, tables, figures = build_report(df, args.phase_bins)
    report["source_files"] = [str(p) for p in cohort.source_files]
    save_report(output_dir, prefix, report, tables, figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
