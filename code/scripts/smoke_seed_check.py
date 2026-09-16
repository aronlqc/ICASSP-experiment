from __future__ import annotations

import argparse
import csv
import hashlib
import math
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "code/scripts/run_mask_isolation_icassp.py"
MANIFEST = ROOT / "experiments_24h_k64/manifest.csv"
SCORE_ROOT = ROOT / "experiments_icassp4_ablation_scores_20260905"


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    value.update(path.read_bytes())
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a one-clip smoke test under multiple fresh seeds.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    args = parser.parse_args()
    assert len(set(args.seeds)) >= 2, "provide at least two distinct seeds"

    random_hashes = []
    rat_hashes = []
    with tempfile.TemporaryDirectory(prefix="rat_dp_seed_smoke_") as temporary:
        temp_root = Path(temporary)
        for seed in args.seeds:
            output = temp_root / f"seed_{seed}"
            command = [
                sys.executable,
                str(RUNNER),
                "--manifest",
                str(MANIFEST),
                "--output-dir",
                str(output),
                "--seed",
                str(seed),
                "--top-k",
                "20",
                "--dp-c-blk",
                "2.0",
                "--limit",
                "1",
                "--extra-score-root",
                str(SCORE_ROOT),
                "--include-no-trajectory",
                "--include-videomae-only",
                "--include-shifted-no-trajectory",
            ]
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            if completed.returncode:
                print(completed.stdout)
                print(completed.stderr, file=sys.stderr)
                raise SystemExit(completed.returncode)
            metrics = read_rows(output / "mask_isolation_metrics.csv")
            runs = read_rows(output / "run_manifest.csv")
            assert len(metrics) == 7 and len(runs) == 7
            assert (output / "failures.csv").stat().st_size <= 2
            for row in metrics:
                for key in ["motion_reference_coverage", "motion_distortion_share"]:
                    assert math.isfinite(float(row[key])), (seed, row["method"], key)
            by_method = {row["method"]: row for row in runs}
            random_hashes.append(digest(Path(by_method["Random-mask"]["score_maps"])))
            rat_hashes.append(digest(Path(by_method["RAT-no-trajectory-mask"]["score_maps"])))
            print(f"[seed-smoke] seed={seed}: 7/7 policies PASS")

    assert len(set(random_hashes)) == len(args.seeds), "random masks did not change across seeds"
    assert len(set(rat_hashes)) == 1, "deterministic RAT-fusion maps changed across seeds"
    print("[seed-smoke] random masks differ; deterministic RAT-fusion maps match")
    print("[seed-smoke] PASS")


if __name__ == "__main__":
    main()
