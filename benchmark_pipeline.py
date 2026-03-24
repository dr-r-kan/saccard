from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict

from saccard.saccardiac import (
    MultimodalExtractionConfig,
    extract_cardiac_and_eye_timeseries,
    save_outputs,
)


def run_benchmark(video_path: str, output_dir: str, phase_bins: int, max_frames: int | None) -> Dict[str, Any]:
    config = MultimodalExtractionConfig(
        video_path=video_path,
        output_dir=output_dir,
        phase_bin_count=phase_bins,
        max_frames=max_frames,
        verbose=False,
    )

    tracemalloc.start()
    t0 = time.perf_counter()
    df, phase_summary, blink_summary, summary = extract_cardiac_and_eye_timeseries(
        video_path=video_path,
        config=config,
    )
    total_s = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    outputs = save_outputs(df, phase_summary, blink_summary, summary, config)

    metrics: Dict[str, Any] = {
        "video_path": video_path,
        "total_runtime_s": float(total_s),
        "peak_tracemalloc_mb": float(peak_bytes / (1024 * 1024)),
        "timeseries_rows": int(len(df)),
        "phase_rows": int(len(phase_summary)),
        "blink_rows": int(len(blink_summary)),
        "summary_keys": sorted(summary.keys()),
        "pipeline_stage_timings_s": summary.get("pipeline_stage_timings_s", {}),
        "outputs": {k: str(v) for k, v in outputs.items()},
    }
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark saccardiac end-to-end pipeline runtime and memory.")
    parser.add_argument("video_path", help="Input video file path")
    parser.add_argument("--output-dir", default="outputs", help="Output directory for pipeline artifacts")
    parser.add_argument("--phase-bins", type=int, default=12, help="Phase bin count")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame cap for quicker runs")
    parser.add_argument("--label", default="run", help="Label used in benchmark artifact names")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = run_benchmark(
        video_path=args.video_path,
        output_dir=args.output_dir,
        phase_bins=args.phase_bins,
        max_frames=args.max_frames,
    )

    artifact_path = out_dir / f"{Path(args.video_path).stem}_{args.label}_benchmark.json"
    artifact_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Benchmark written to: {artifact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
