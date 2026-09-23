"""Recompute paired per-video statistics from saved mask-isolation logs.

The script does not regenerate videos. It joins the saved per-video Cover and
RawErr metrics with the saved incremental-error log, forms RAT-minus-control
paired differences, and reports mean/median differences, percentile bootstrap
confidence intervals, an exact sign-test p-value, and paired Hedges g_z.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = {
    "Cover": "motion_reference_coverage",
    "RawErr": "motion_distortion_share",
    "IncErr": "incremental_motion_distortion_share",
}

PAIRS = {
    "Random": "Random-mask",
    "Shifted fusion": "Shifted-RAT-fusion-mask",
    "VideoMAE-only": "VideoMAE-only-mask",
    "RAT+traj.": "RAT-mask",
}

RAT_METHOD = "RAT-no-trajectory-mask"


def bootstrap_ci(values: np.ndarray, statistic: str, rng: np.random.Generator,
                 n_resamples: int) -> tuple[float, float]:
    """Percentile paired bootstrap CI for the mean or median of differences."""
    n = values.size
    indices = rng.integers(0, n, size=(n_resamples, n))
    samples = values[indices]
    if statistic == "mean":
        estimates = samples.mean(axis=1)
    elif statistic == "median":
        estimates = np.median(samples, axis=1)
    else:
        raise ValueError(statistic)
    return tuple(np.quantile(estimates, [0.025, 0.975]).tolist())


def exact_sign_p(values: np.ndarray) -> float:
    """Two-sided exact sign-test p-value after dropping zero differences."""
    nonzero = values[np.abs(values) > 1e-12]
    n = int(nonzero.size)
    if n == 0:
        return 1.0
    wins = int(np.sum(nonzero > 0))
    tail = sum(math.comb(n, k) for k in range(0, min(wins, n - wins) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def paired_effect(values: np.ndarray) -> tuple[float, float]:
    """Return paired Cohen dz and small-sample corrected Hedges gz."""
    sd = float(np.std(values, ddof=1))
    if sd == 0.0:
        return (float("nan"), float("nan"))
    dz = float(np.mean(values) / sd)
    n = values.size
    correction = 1.0 - 3.0 / (4.0 * n - 9.0)
    return dz, dz * correction


def load_and_validate(metrics_csv: Path, incremental_csv: Path) -> pd.DataFrame:
    main = pd.read_csv(metrics_csv)
    inc = pd.read_csv(incremental_csv)
    inc_cols = ["video_id", "method", "incremental_motion_distortion_share"]
    merged = main.merge(inc[inc_cols], on=["video_id", "method"], how="left", validate="one_to_one")
    required = list(METRICS.values())
    missing = [c for c in required if c not in merged.columns]
    if missing:
        raise ValueError(f"Missing metric columns: {missing}")
    if merged[required].isna().any().any():
        bad = merged.loc[merged[required].isna().any(axis=1), ["video_id", "method"]]
        raise ValueError(f"Missing per-video metrics in rows:\n{bad.to_string(index=False)}")
    expected = set(PAIRS.values()) | {RAT_METHOD}
    found = set(merged["method"].unique())
    if not expected.issubset(found):
        raise ValueError(f"Missing methods: {sorted(expected - found)}")
    counts = merged[merged.method.isin(expected)].groupby("method")["video_id"].nunique()
    if counts.min() != counts.max() or counts.min() != 20:
        raise ValueError(f"Expected 20 paired videos per method, got:\n{counts}")
    return merged


def compute_stats(df: pd.DataFrame, seed: int, n_resamples: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    long_rows: list[dict] = []
    summary_rows: list[dict] = []
    for metric_label, metric_col in METRICS.items():
        rat = df[df.method == RAT_METHOD].set_index("video_id")[metric_col]
        for control_label, control_method in PAIRS.items():
            control = df[df.method == control_method].set_index("video_id")[metric_col]
            joined = pd.concat([rat.rename("rat"), control.rename("control")], axis=1).dropna()
            joined["difference"] = joined["rat"] - joined["control"]
            for video_id, row in joined.iterrows():
                long_rows.append({
                    "video_id": video_id,
                    "metric": metric_label,
                    "control": control_label,
                    "rat_value": float(row.rat),
                    "control_value": float(row.control),
                    "difference": float(row.difference),
                })
            values = joined["difference"].to_numpy(dtype=float)
            dz, gz = paired_effect(values)
            mean_ci = bootstrap_ci(values, "mean", rng, n_resamples)
            median_ci = bootstrap_ci(values, "median", rng, n_resamples)
            summary_rows.append({
                "metric": metric_label,
                "control": control_label,
                "n": int(values.size),
                "mean_difference": float(np.mean(values)),
                "median_difference": float(np.median(values)),
                "difference_sd": float(np.std(values, ddof=1)),
                "mean_ci95_low": mean_ci[0],
                "mean_ci95_high": mean_ci[1],
                "median_ci95_low": median_ci[0],
                "median_ci95_high": median_ci[1],
                "cohen_dz": dz,
                "hedges_gz": gz,
                "positive_pairs": int(np.sum(values > 0)),
                "negative_pairs": int(np.sum(values < 0)),
                "ties": int(np.sum(np.abs(values) <= 1e-12)),
                "exact_sign_p": exact_sign_p(values),
            })
    return pd.DataFrame(long_rows), pd.DataFrame(summary_rows)


def write_markdown(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Per-video paired reanalysis (2026-09-22)",
        "",
        "The saved per-video logs were re-used; no videos were regenerated.",
        "Differences are RAT fusion (RAT-no-trajectory-mask) minus each control.",
        "CI uses 10,000 paired bootstrap resamples (percentile interval, seed 20260922).",
        "Effect size is paired Hedges g_z: the standardized paired difference with the small-sample correction.",
        "",
        "| Metric | Control | n | Mean diff | Median diff | Mean 95% CI | Median 95% CI | Hedges g_z | Sign p |",
        "|---|---|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.metric} | {row.control} | {row.n} | {row.mean_difference:.4f} | "
            f"{row.median_difference:.4f} | [{row.mean_ci95_low:.4f}, {row.mean_ci95_high:.4f}] | "
            f"[{row.median_ci95_low:.4f}, {row.median_ci95_high:.4f}] | {row.hedges_gz:.2f} | "
            f"{row.exact_sign_p:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_forest(summary: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    import matplotlib.pyplot as plt

    metric_order = list(METRICS)
    control_order = list(PAIRS)
    colors = {"Random": "#2f6690", "Shifted fusion": "#d1495b", "VideoMAE-only": "#6a994e", "RAT+traj.": "#7b2cbf"}
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5), sharey=True)
    y = np.arange(len(control_order))
    for ax, metric in zip(axes, metric_order):
        part = summary[summary.metric == metric].set_index("control").loc[control_order]
        x = part["mean_difference"].to_numpy()
        lo = part["mean_ci95_low"].to_numpy()
        hi = part["mean_ci95_high"].to_numpy()
        med = part["median_difference"].to_numpy()
        for i, control in enumerate(control_order):
            ax.errorbar(x[i], i, xerr=[[x[i] - lo[i]], [hi[i] - x[i]]], fmt="o",
                        color=colors[control], ecolor=colors[control], capsize=3,
                        markersize=5, linewidth=1.2)
            ax.plot(med[i], i, marker="D", markersize=4.5, color="black", markeredgewidth=0.4)
            ax.text(max(hi) + 0.008, i, f"g={part.iloc[i].hedges_gz:.2f}", va="center", fontsize=7)
        ax.axvline(0.0, color="#555555", linewidth=0.8)
        ax.set_title(metric)
        ax.set_xlabel("RAT - control")
        ax.grid(axis="x", color="#dddddd", linewidth=0.6)
        ax.set_xlim(left=min(0.0, float(lo.min()) - 0.02), right=float(hi.max()) + 0.055)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(control_order)
    fig.suptitle("Per-video paired reanalysis (n=20)", y=1.02, fontsize=11)
    fig.text(0.5, -0.02, "Circles: mean difference with 95% paired-bootstrap CI; diamonds: median difference; labels: paired Hedges g_z.",
             ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--incremental-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--resamples", type=int, default=10000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_and_validate(args.metrics_csv, args.incremental_csv)
    long_df, summary = compute_stats(df, args.seed, args.resamples)
    long_df.to_csv(args.output_dir / "per_video_differences.csv", index=False, float_format="%.9f")
    summary.to_csv(args.output_dir / "paired_stats_mean_median_effect.csv", index=False, float_format="%.9f")
    write_markdown(summary, args.output_dir / "reanalysis_summary.md")
    plot_forest(summary, args.output_dir / "per_video_effects.png", args.output_dir / "per_video_effects.pdf")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
