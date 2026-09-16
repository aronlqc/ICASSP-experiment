from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SEEDS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 42}
EXPECTED_K = {10, 20, 30, 40}
EXPECTED_POLICIES = {
    "RAT+traj",
    "Motion-reference",
    "Random",
    "Shifted-RAT+traj",
    "RAT-fusion",
    "Shifted-RAT-fusion",
    "VideoMAE-only",
}


def rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise AssertionError(f"cannot open video: {path.relative_to(ROOT)}")
    count = 0
    while True:
        ok, _frame = capture.read()
        if not ok:
            break
        count += 1
    capture.release()
    if count <= 0:
        raise AssertionError(f"empty video: {path.relative_to(ROOT)}")
    return count


def check_checksums() -> None:
    checksum_file = ROOT / "CHECKSUMS.sha256"
    if not checksum_file.exists():
        print("[verify] checksum file not present yet; structural checks continue")
        return
    checked = 0
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        assert path.is_file(), f"missing checksummed file: {relative}"
        assert sha256(path) == expected, f"checksum mismatch: {relative}"
        checked += 1
    print(f"[verify] checksums: {checked} files")


def write_checksums() -> None:
    checksum_file = ROOT / "CHECKSUMS.sha256"
    selected = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and path != checksum_file
    )
    checksum_file.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in selected),
        encoding="utf-8",
    )
    print(f"[verify] wrote checksums for {len(selected)} files")


def check_inventory() -> None:
    bad_names = {"__pycache__", ".pytest_cache", ".git", ".venv"}
    bad_suffixes = {".aux", ".log", ".pyc", ".zip"}
    for path in ROOT.rglob("*"):
        assert not any(part in bad_names for part in path.parts), f"excluded directory found: {path}"
        if path.is_file():
            assert path.suffix not in bad_suffixes, f"excluded artifact found: {path}"
    for name in ["E3_cue_ablation", "E4_confirmatory_100", "E5_shift_offsets"]:
        assert not (ROOT / "extended_evidence_suite_20260906" / name).exists(), name
    checkpoints = list((ROOT / "code/VideoMAE/checkpoints").glob("*.pth*"))
    assert len(checkpoints) == 1, f"expected one VideoMAE checkpoint, got {len(checkpoints)}"
    assert sha256(checkpoints[0]) == "2e0c3f4bc73c6d287b96c30975b43d60c9fc004c548150f252278f1becd89a7d"
    raft = ROOT / "code/RAFT/models/raft-things.pth"
    assert sha256(raft) == "fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1"
    print("[verify] minimal inventory and model checkpoints")


def check_manifests_and_arrays() -> None:
    source_manifest = rows(ROOT / "data/ucf101_rescue/data_manifest.csv")
    assert len(source_manifest) == 20
    for row in source_manifest:
        path = ROOT / row["path"]
        assert path.is_file() and sha256(path) == row["sha256"]

    deterministic_manifest = rows(ROOT / "experiments_icassp4_ablation_scores_20260905/score_manifest.csv")
    assert len(deterministic_manifest) == 40
    for row in deterministic_manifest:
        path = ROOT / row["score_maps"]
        assert path.is_file() and sha256(path) == row["score_sha256"]

    trajectory_manifest = rows(ROOT / "extended_evidence_suite_20260906/E1_trajectory_scores/score_manifest.csv")
    assert len(trajectory_manifest) == 200
    for row in trajectory_manifest:
        path = ROOT / row["score_maps"]
        assert path.is_file() and sha256(path) == row["score_sha256"]

    manifest = rows(ROOT / "experiments_24h_k64/manifest.csv")
    assert len(manifest) == 20
    assert {row["method"] for row in manifest} == {"RAT-DP"}
    for row in manifest:
        raw = ROOT / row["input_video"]
        backend_dir = ROOT / row["output_dir"]
        backend = backend_dir / "rat_intermediate_merged.mp4"
        private_scores = backend_dir / "masks/private_risk_scores.npy"
        assert raw.is_file() and backend.is_file() and private_scores.is_file()
        raw_frames = frame_count(raw)
        backend_frames = frame_count(backend)
        assert raw_frames == backend_frames, (row["video_id"], raw_frames, backend_frames)
        expected_frames = raw_frames
        array = np.load(private_scores, mmap_mode="r")
        assert array.shape == (expected_frames, 14, 14), (row["video_id"], array.shape)
        score_dir = ROOT / "experiments_icassp4_ablation_scores_20260905" / row["video_id"]
        for name in ["score_maps_no_trajectory.npy", "score_maps_videomae_only.npy"]:
            score = np.load(score_dir / name, mmap_mode="r")
            assert score.shape == (expected_frames, 14, 14), (row["video_id"], name, score.shape)
        for seed in EXPECTED_SEEDS:
            score = np.load(
                ROOT
                / "extended_evidence_suite_20260906/E1_trajectory_scores"
                / f"seed_{seed}"
                / row["video_id"]
                / "score_maps_trajectory.npy",
                mmap_mode="r",
            )
            assert score.shape == (expected_frames, 14, 14), (row["video_id"], seed, score.shape)
    main_runs = rows(ROOT / "experiments_icassp4_mask_ablation_20260905/run_manifest.csv")
    assert len(main_runs) == 140
    for row in main_runs:
        for key in ["input_video", "backend_input_video", "score_maps"]:
            assert (ROOT / row[key]).is_file(), f"broken main manifest reference: {key}={row[key]}"
    print("[verify] 20 raw videos, 20 backend videos, and all retained score maps")


def check_results() -> None:
    main_metrics = rows(ROOT / "experiments_icassp4_mask_ablation_20260905/mask_isolation_metrics.csv")
    e1 = rows(ROOT / "extended_evidence_suite_20260906/E1_multi_seed/metrics.csv")
    e2 = rows(ROOT / "extended_evidence_suite_20260906/E2_k_sensitivity/metrics.csv")
    assert len(main_metrics) == 140
    assert len(e1) == 1400 and {int(row["seed"]) for row in e1} == EXPECTED_SEEDS
    assert len(e2) == 560 and {int(row["top_k"]) for row in e2} == EXPECTED_K
    assert {row["policy"] for row in e1} == EXPECTED_POLICIES
    assert {row["policy"] for row in e2} == EXPECTED_POLICIES
    for relative in [
        "experiments_icassp4_mask_ablation_20260905/failures.csv",
        "experiments_icassp4_mask_ablation_20260905/incremental_errshare_failures.csv",
        "extended_evidence_suite_20260906/E1_multi_seed/failures.csv",
        "extended_evidence_suite_20260906/E2_k_sensitivity/failures.csv",
    ]:
        path = ROOT / relative
        assert path.exists() and path.stat().st_size <= 2, f"failure record is not empty: {relative}"
    e1_pairs = rows(ROOT / "extended_evidence_suite_20260906/E1_multi_seed/paired_stats.csv")
    e2_pairs = rows(ROOT / "extended_evidence_suite_20260906/E2_k_sensitivity/paired_stats.csv")
    assert len(e1_pairs) == 12 and len(e2_pairs) == 24
    print("[verify] main=140 rows, E1=1400 rows, E2=560 rows, no recorded failures")


def check_paper() -> None:
    paper_dir = ROOT / "icassp2026_allocator_rewrite_20260901"
    text = (paper_dir / "paper.tex").read_text(encoding="utf-8")
    assert "Motion ref. (upper bound)" in text
    assert "proxy-aligned empirical upper bound" in text
    assert (paper_dir / "paper.pdf").is_file()
    assert (paper_dir / "figures/mask_policy_example.png").is_file()
    assert (ROOT / "extended_evidence_suite_20260906/figures/E2_k_sensitivity_deltas.png").is_file()
    print("[verify] frozen TeX/PDF and both referenced figures")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the frozen Minimal_package handoff.")
    parser.add_argument("--write-checksums", action="store_true")
    args = parser.parse_args()
    check_inventory()
    check_manifests_and_arrays()
    check_results()
    check_paper()
    if args.write_checksums:
        write_checksums()
    check_checksums()
    print("[verify] PASS")


if __name__ == "__main__":
    main()
