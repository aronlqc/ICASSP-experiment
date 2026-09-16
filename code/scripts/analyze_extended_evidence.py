from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from run_mask_isolation_icassp import stable_seed


METRICS = ["cover", "raw_errshare", "inc_errshare"]
EXPECTED_ROWS = {"E1": 1400, "E2": 560}
PAIRINGS = {
    "E1": [("RAT-fusion", "Random"), ("RAT-fusion", "Shifted-RAT-fusion"), ("RAT-fusion", "VideoMAE-only"), ("RAT-fusion", "RAT+traj")],
    "E2": [("RAT-fusion", "Random"), ("RAT-fusion", "Shifted-RAT-fusion")],
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def exact_sign_test(diffs: Sequence[float]) -> float:
    positives = sum(value > 0 for value in diffs)
    negatives = sum(value < 0 for value in diffs)
    n = positives + negatives
    if n == 0:
        return 1.0
    tail = min(positives, negatives)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(tail + 1)) / (2**n))


def bootstrap_ci(values: Sequence[float], seed: int) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(10000, len(array)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(samples, [0.025, 0.975]))


def fmt(value: float) -> str:
    return f"{value:.6f}" if np.isfinite(value) else "nan"


def grouped_means(rows: List[Dict[str, str]], keys: Sequence[str]) -> Dict[Tuple[str, ...], Dict[str, float]]:
    buckets: Dict[Tuple[str, ...], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = tuple(row[item] for item in keys)
        for metric in METRICS:
            buckets[key][metric].append(float(row[metric]))
    return {
        key: {metric: float(np.mean(values)) for metric, values in metrics.items()}
        for key, metrics in buckets.items()
    }


def paired_rows(rows: List[Dict[str, str]], experiment: str) -> List[Dict[str, object]]:
    if experiment == "E1":
        means = grouped_means(rows, ["video_id", "policy"])
        by_unit = defaultdict(dict)
        for (video_id, policy), values in means.items():
            by_unit[video_id][policy] = values
        pairs = PAIRINGS[experiment]
    elif experiment == "E2":
        means = grouped_means(rows, ["video_id", "top_k", "policy"])
        by_k = defaultdict(lambda: defaultdict(dict))
        for (video_id, top_k, policy), values in means.items():
            by_k[top_k][video_id][policy] = values
        output = []
        for top_k, video_units in sorted(by_k.items(), key=lambda item: int(item[0])):
            for left, right in PAIRINGS[experiment]:
                for metric in METRICS:
                    diffs = [
                        policies[left][metric] - policies[right][metric]
                        for policies in video_units.values()
                        if left in policies and right in policies
                    ]
                    ci = bootstrap_ci(diffs, stable_seed(20260906, experiment, top_k, left, right, metric))
                    output.append({
                        "experiment": experiment,
                        "top_k": top_k,
                        "comparison": f"{left} vs {right}",
                        "metric": metric,
                        "paired_units": len(diffs),
                        "left_wins": sum(value > 0 for value in diffs),
                        "right_wins": sum(value < 0 for value in diffs),
                        "ties": sum(value == 0 for value in diffs),
                        "mean_diff": fmt(float(np.mean(diffs)) if diffs else float("nan")),
                        "ci95_low": fmt(ci[0]),
                        "ci95_high": fmt(ci[1]),
                        "exact_sign_p": fmt(exact_sign_test(diffs)),
                    })
        return output
    else:
        raise ValueError(experiment)

    output = []
    for left, right in pairs:
        for metric in METRICS:
            diffs = [
                policies[left][metric] - policies[right][metric]
                for policies in by_unit.values()
                if left in policies and right in policies
            ]
            ci = bootstrap_ci(diffs, stable_seed(20260906, experiment, left, right, metric))
            output.append({
                "experiment": experiment,
                "comparison": f"{left} vs {right}",
                "metric": metric,
                "paired_units": len(diffs),
                "left_wins": sum(value > 0 for value in diffs),
                "right_wins": sum(value < 0 for value in diffs),
                "ties": sum(value == 0 for value in diffs),
                "mean_diff": fmt(float(np.mean(diffs)) if diffs else float("nan")),
                "ci95_low": fmt(ci[0]),
                "ci95_high": fmt(ci[1]),
                "exact_sign_p": fmt(exact_sign_test(diffs)),
            })
    return output


def aggregate_rows(rows: List[Dict[str, str]], experiment: str) -> List[Dict[str, object]]:
    keys = ["policy"]
    if experiment == "E1":
        keys = ["seed", "policy"]
    elif experiment == "E2":
        keys = ["top_k", "policy"]
    means = grouped_means(rows, keys)
    output = []
    for key, values in sorted(means.items()):
        selected = [row for row in rows if all(row[name] == value for name, value in zip(keys, key))]
        item: Dict[str, object] = dict(zip(keys, key))
        item["n"] = len(selected)
        for metric in METRICS:
            data = [float(row[metric]) for row in selected]
            item[f"{metric}_mean"] = fmt(values[metric])
            item[f"{metric}_std"] = fmt(float(np.std(data)))
        output.append(item)
    return output


def plot_e1(rows: List[Dict[str, str]], output: Path) -> None:
    per_video = grouped_means(rows, ["video_id", "seed", "policy"])
    seeds = sorted({int(row["seed"]) for row in rows})
    run_positions = np.arange(1, len(seeds) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), constrained_layout=True)
    for axis, metric, title in zip(axes, METRICS, ["Cover", "RawErr", "IncErr"]):
        for control, color in [("Random", "#3973ac"), ("Shifted-RAT-fusion", "#7b5eb5")]:
            means, lows, highs = [], [], []
            for seed in seeds:
                diffs = []
                for video_id in sorted({key[0] for key in per_video}):
                    left = per_video.get((video_id, str(seed), "RAT-fusion"))
                    right = per_video.get((video_id, str(seed), control))
                    if left and right:
                        diffs.append(left[metric] - right[metric])
                mean = float(np.mean(diffs))
                low, high = bootstrap_ci(diffs, stable_seed(20260906, "E1-plot", seed, control, metric))
                means.append(mean)
                lows.append(low)
                highs.append(high)
            axis.errorbar(
                run_positions,
                means,
                yerr=[np.asarray(means) - lows, np.asarray(highs) - means],
                marker="o",
                capsize=2,
                label=control,
                color=color,
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("Repeated stochastic run")
        axis.set_xticks(run_positions)
    axes[0].set_ylabel("RAT fusion minus control")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def write_e1_trace_tables(rows: List[Dict[str, str]], output_dir: Path) -> None:
    by_video_policy: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_video_policy[(row["video_id"], row["policy"])].append(row)

    per_video = []
    for (video_id, policy), selected in sorted(by_video_policy.items()):
        item: Dict[str, object] = {
            "video_id": video_id,
            "policy": policy,
            "n_seeds": len(selected),
        }
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            item[f"{metric}_mean"] = fmt(float(np.mean(values)))
            item[f"{metric}_std"] = fmt(float(np.std(values)))
            item[f"{metric}_min"] = fmt(float(np.min(values)))
            item[f"{metric}_max"] = fmt(float(np.max(values)))
        per_video.append(item)
    write_csv(output_dir / "per_video_seed_aggregated.csv", per_video)

    indexed = grouped_means(rows, ["video_id", "seed", "policy"])
    seed_level = []
    videos = sorted({row["video_id"] for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    for seed in seeds:
        for left, right in PAIRINGS["E1"]:
            for metric in METRICS:
                diffs = []
                for video_id in videos:
                    left_values = indexed.get((video_id, str(seed), left))
                    right_values = indexed.get((video_id, str(seed), right))
                    if left_values and right_values:
                        diffs.append(left_values[metric] - right_values[metric])
                low, high = bootstrap_ci(
                    diffs,
                    stable_seed(20260906, "E1-seed-level", seed, left, right, metric),
                )
                seed_level.append({
                    "seed": seed,
                    "comparison": f"{left} vs {right}",
                    "metric": metric,
                    "paired_videos": len(diffs),
                    "left_wins": sum(value > 0 for value in diffs),
                    "right_wins": sum(value < 0 for value in diffs),
                    "ties": sum(value == 0 for value in diffs),
                    "mean_diff": fmt(float(np.mean(diffs)) if diffs else float("nan")),
                    "ci95_low": fmt(low),
                    "ci95_high": fmt(high),
                })
    write_csv(output_dir / "seed_level_deltas.csv", seed_level)


def plot_e2(rows: List[Dict[str, str]], output: Path) -> None:
    per_video = grouped_means(rows, ["video_id", "top_k", "policy"])
    ks = sorted({int(row["top_k"]) for row in rows})
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), constrained_layout=True)
    for axis, metric, title in zip(axes, METRICS, ["Cover", "RawErr", "IncErr"]):
        for control, color in [("Random", "#3973ac"), ("Shifted-RAT-fusion", "#7b5eb5")]:
            means, lows, highs = [], [], []
            for top_k in ks:
                diffs = []
                for video_id in sorted({key[0] for key in per_video}):
                    left = per_video.get((video_id, str(top_k), "RAT-fusion"))
                    right = per_video.get((video_id, str(top_k), control))
                    if left and right:
                        diffs.append(left[metric] - right[metric])
                mean = float(np.mean(diffs))
                low, high = bootstrap_ci(diffs, stable_seed(20260906, "E2-plot", top_k, control, metric))
                means.append(mean)
                lows.append(low)
                highs.append(high)
            axis.errorbar(
                ks,
                means,
                yerr=[np.asarray(means) - lows, np.asarray(highs) - means],
                marker="o",
                capsize=3,
                label=control,
                color=color,
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("Selected blocks k")
    axes[0].set_ylabel("RAT fusion minus control")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def evidence_status(experiment: str, row_count: int, failures: int) -> str:
    expected = EXPECTED_ROWS[experiment]
    if failures:
        return f"INCOMPLETE ({failures} failures)"
    if row_count != expected:
        return f"INCOMPLETE ({row_count}/{expected} rows)"
    return "COMPLETE"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the frozen E1/E2 evidence and regenerate figures.")
    parser.add_argument("--suite-root", required=True)
    args = parser.parse_args()
    root = Path(args.suite_root)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    all_stats: Dict[str, List[Dict[str, object]]] = {}
    all_aggregates: Dict[str, List[Dict[str, object]]] = {}
    statuses = {}

    for experiment in ["E1", "E2"]:
        exp_dir = root / {
            "E1": "E1_multi_seed",
            "E2": "E2_k_sensitivity",
        }[experiment]
        rows = read_csv(exp_dir / "metrics.csv")
        failures = read_csv(exp_dir / "failures.csv")
        statuses[experiment] = evidence_status(experiment, len(rows), len(failures))
        if not rows:
            continue
        aggregate = aggregate_rows(rows, experiment)
        stats = paired_rows(rows, experiment)
        write_csv(exp_dir / "aggregate.csv", aggregate)
        write_csv(exp_dir / "paired_stats.csv", stats)
        all_stats[experiment] = stats
        all_aggregates[experiment] = aggregate
        if experiment == "E1" and len(rows) == EXPECTED_ROWS[experiment]:
            write_e1_trace_tables(rows, exp_dir)
            plot_e1(rows, figures / "E1_multi_seed_deltas.png")
        elif experiment == "E2" and len(rows) == EXPECTED_ROWS[experiment]:
            plot_e2(rows, figures / "E2_k_sensitivity_deltas.png")

    lines = ["# Extended Experiment Report", "", "Protocol status is determined from saved per-run metrics and failure logs.", ""]
    for experiment, title in [
        ("E1", "Multi-seed robustness"),
        ("E2", "Selected-area sensitivity"),
    ]:
        lines.extend([f"## {experiment}: {title}", "", f"Status: **{statuses[experiment]}**.", ""])
        if experiment in all_stats:
            lines.extend(["| Setting | Comparison | Metric | Units | Wins | Mean diff | 95% CI | Sign p |", "|---|---|---|---:|---:|---:|---:|---:|"])
            for row in all_stats[experiment]:
                setting = f"k={row['top_k']}" if row.get("top_k", "") != "" else "--"
                lines.append(
                    f"| {setting} | {row['comparison']} | {row['metric']} | {row['paired_units']} | {row['left_wins']}/{row['paired_units']} | {row['mean_diff']} | [{row['ci95_low']}, {row['ci95_high']}] | {row['exact_sign_p']} |"
                )
            lines.append("")
        else:
            lines.extend(["No completed metrics are available yet.", ""])
    (root / "EXTENDED_EXPERIMENT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[Done] reports under {root}")


if __name__ == "__main__":
    main()
