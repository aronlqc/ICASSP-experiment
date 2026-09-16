from __future__ import annotations

import argparse
import csv
import math
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def count_frames(video_path: Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n <= 0:
        n = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            n += 1
    cap.release()
    if n <= 0:
        raise RuntimeError(f"No frames found: {video_path}")
    return n


def rdp_eps_for_sigma(
    T: int,
    k: int,
    C_blk: float,
    sigma: float,
    delta: float,
    alphas: Iterable[int],
) -> float:
    delta = min(max(float(delta), 1e-12), 1.0 - 1e-12)
    Delta = 2.0 * float(C_blk) * math.sqrt(float(k))
    best = float("inf")
    for alpha in alphas:
        a = float(alpha)
        rho = int(T) * a * (Delta ** 2) / (2.0 * (float(sigma) ** 2))
        eps = rho + math.log(1.0 / delta) / (a - 1.0)
        best = min(best, eps)
    return float(best)


def sigma_for_target_eps(
    T: int,
    target_eps: float,
    k: int,
    C_blk: float,
    delta: float,
    alphas: Iterable[int],
) -> float:
    lo, hi = 1e-6, 1.0
    while rdp_eps_for_sigma(T, k, C_blk, hi, delta, alphas) > target_eps:
        hi *= 2.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if rdp_eps_for_sigma(T, k, C_blk, mid, delta, alphas) > target_eps:
            lo = mid
        else:
            hi = mid
    return float(hi)


def select_ucf_videos(root: Path, num_classes: int, videos_per_class: int) -> List[Path]:
    class_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    selected: List[Path] = []
    chosen_classes = 0
    for class_dir in class_dirs:
        videos = sorted([
            p for p in class_dir.iterdir()
            if p.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}
        ])
        if len(videos) < videos_per_class:
            continue
        selected.extend(videos[:videos_per_class])
        chosen_classes += 1
        if chosen_classes >= num_classes:
            break
    if chosen_classes < num_classes:
        raise RuntimeError(f"Only found {chosen_classes} classes with enough videos under {root}")
    return selected


def build_rows(args: argparse.Namespace) -> List[Dict[str, str]]:
    videos = select_ucf_videos(Path(args.ucf_root), args.num_classes, args.videos_per_class)
    rows: List[Dict[str, str]] = []
    alphas = range(2, 129)
    for video in videos:
        class_name = video.parent.name
        video_id = f"{class_name}_{video.stem}"
        T = count_frames(video)
        sigma = sigma_for_target_eps(
            T=T,
            target_eps=args.epsilon,
            k=args.dp_top_k,
            C_blk=args.dp_c_blk,
            delta=args.delta,
            alphas=alphas,
        )
        for seed in args.seeds:
            common_out = Path(args.output_root) / video_id / f"seed_{seed}"
            commands = {
                "RAT-DP": [
                    sys.executable, "main.py",
                    "--input", str(video),
                    "--output-dir", str(common_out / "rat_dp"),
                    "--seed", str(seed),
                    "--dp-delta", str(args.delta),
                    "--dp-sigma", f"{sigma:.10f}",
                    "--dp-top-k", str(args.dp_top_k),
                    "--dp-c-blk", str(args.dp_c_blk),
                    "--mask-mode", "private_risk_experimental",
                    "--videomae-root", str(args.videomae_root),
                    "--videomae-weight-path", str(args.videomae_weight_path),
                    "--raft-root", str(args.raft_root),
                    "--raft-ckpt", str(args.raft_ckpt),
                    "--device", str(args.device),
                    "--raft-iters", str(args.raft_iters),
                ],
                "Uniform-DCT-Gaussian": [
                    sys.executable, "baseline/uniform_dct_gaussian.py",
                    "--input", str(video),
                    "--output-dir", str(common_out / "uniform_dct"),
                    "--seed", str(seed),
                    "--dp-delta", str(args.delta),
                    "--dp-sigma", f"{sigma:.10f}",
                    "--dp-top-k", str(args.dp_top_k),
                    "--dp-c-blk", str(args.dp_c_blk),
                    "--epsilon-label", str(args.epsilon),
                ],
                "Video-DPRP": [
                    sys.executable, "baseline/video_dprp.py",
                    "--input", str(video),
                    "--output-dir", str(common_out / "video_dprp"),
                    "--output-name", "videodprp.mp4",
                    "--epsilon", str(args.epsilon),
                    "--delta", str(args.delta),
                    "--k", str(args.dprp_k),
                    "--seed", str(seed),
                ],
            }
            for method, cmd in commands.items():
                output_dir = {
                    "RAT-DP": common_out / "rat_dp",
                    "Uniform-DCT-Gaussian": common_out / "uniform_dct",
                    "Video-DPRP": common_out / "video_dprp",
                }[method]
                rows.append({
                    "video_id": video_id,
                    "class": class_name,
                    "input_video": str(video),
                    "frames": str(T),
                    "method": method,
                    "seed": str(seed),
                    "epsilon": str(args.epsilon),
                    "delta": str(args.delta),
                    "rat_sigma": f"{sigma:.10f}",
                    "dprp_k": str(args.dprp_k),
                    "output_dir": str(output_dir),
                    "command": shlex.join(cmd),
                })
    return rows


def write_manifest(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_rows(rows: List[Dict[str, str]]) -> None:
    for row in rows:
        out_dir = Path(row["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "run.log"
        cmd = shlex.split(row["command"])
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed for {row['method']} {row['video_id']}; see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or run the 24h rescue experiment manifest.")
    parser.add_argument("--ucf-root", required=True, help="Path to UCF-101 class-folder root.")
    parser.add_argument("--output-root", default="experiments_24h")
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--videos-per-class", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--delta", type=float, default=1e-4)
    parser.add_argument("--dp-top-k", type=int, default=20)
    parser.add_argument("--dp-c-blk", type=float, default=2.0)
    parser.add_argument("--dprp-k", type=int, default=64)
    parser.add_argument("--videomae-root", default=str(PROJECT_ROOT / "VideoMAE"))
    parser.add_argument(
        "--videomae-weight-path",
        default=str(PROJECT_ROOT / "VideoMAE" / "checkpoints" / "videomae_pretrain_base_patch16_224.pth"),
    )
    parser.add_argument("--raft-root", default=str(PROJECT_ROOT / "RAFT"))
    parser.add_argument("--raft-ckpt", default=str(PROJECT_ROOT / "RAFT" / "models" / "raft-things.pth"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--raft-iters", type=int, default=12)
    parser.add_argument("--run", action="store_true", help="Actually execute commands after writing manifest.")
    args = parser.parse_args()

    rows = build_rows(args)
    manifest_path = Path(args.output_root) / "manifest.csv"
    write_manifest(rows, manifest_path)
    print(f"[Manifest] {manifest_path} rows={len(rows)}")
    if args.run:
        run_rows(rows)


if __name__ == "__main__":
    main()
