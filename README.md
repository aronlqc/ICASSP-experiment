# Minimal_package

This is the frozen student handoff package for the ICASSP manuscript on
same-backend video perturbation allocation. The scientific scope is the main
mask-isolation experiment plus E1 (multi-seed robustness) and E2 (selected-area
sensitivity). Run commands from this directory so the relative paths in the
portable manifests resolve correctly.

## Package map

- `icassp2026_allocator_rewrite_20260901/`: frozen `paper.tex`, compiled
  `paper.pdf`, ICASSP style file, and the qualitative figure.
- `extended_evidence_suite_20260906/`: E1/E2 result tables, trace CSVs, saved
  trajectory score maps, and E1/E2 figures.
- `experiments_icassp4_mask_ablation_20260905/`: main-experiment summary,
  paired statistics, per-run metrics, portable run manifest, and score maps.
- `experiments_icassp4_ablation_scores_20260905/`: deterministic RAT-fusion
  and VideoMAE-only score maps used by the main experiment and E1/E2.
- `experiments_24h_k64/manifest.csv`: portable 20-video backend manifest.
- `data/ucf101_rescue/`: the exact 20 UCF101 input clips and source manifest.
- `data/backend_inputs/`: the exact 20 shared backend inputs and original
  trajectory-smoothed RAT score maps.
- `code/`: only the source needed to generate selectors, run the fixed-mask
  release, analyze the main experiment/E1/E2, and regenerate figures.
- `docs/`: experiment provenance, privacy scope, and release-path audit.
- `CHECKSUMS.sha256`: integrity list for all frozen files except itself.
- `VALIDATION_REPORT.md`: checks performed before handoff.

The large E1/E2 release videos are intentionally omitted because they are
regenerable from the included raw clips, shared backend inputs, score maps,
seeds, and code. Their per-run metrics and run manifests are retained. E3--E5,
old drafts, preview images, virtual environments, caches, duplicated
checkpoints, and unrelated baseline runs are not included.

## Environment

Use Linux, Python 3.10 or newer, FFmpeg/OpenCV video codec support, and a TeX
installation containing `pdflatex`. A CUDA GPU is recommended only when
regenerating VideoMAE/RAFT score maps. Install PyTorch for the student's CUDA
version first, then install the remaining packages:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install torch and torchvision from https://pytorch.org/get-started/locally/
python -m pip install -r requirements.txt
```

The frozen host used Python 3.14.4, NumPy 2.5.1, OpenCV 5.0.0, Matplotlib
3.11.1, SciPy 1.18.1, PyTorch 2.13.0+cu130, torchvision 0.28.0+cu130,
Transformers 5.14.1, and scikit-image 0.26.0.

## First checks

```bash
python code/scripts/verify_package.py
python code/scripts/smoke_seed_check.py --seeds 101 202 303
```

The first command checks the inventory, hashes, video readability, array
shapes, manifest references, and expected main/E1/E2 row counts. The second
runs one complete seven-policy clip with three fresh seeds and confirms that
random-mask realizations change while deterministic score maps remain fixed.

## Compile the frozen paper

```bash
cd icassp2026_allocator_rewrite_20260901
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
cd ..
```

The expected output is five PDF pages: four technical pages and one references
page.

## Regenerate current tables and qualitative figure

```bash
python code/scripts/make_icassp_tables_figures.py \
  --experiment-dir experiments_icassp4_mask_ablation_20260905 \
  --figure-dir icassp2026_allocator_rewrite_20260901/figures \
  --example-video BalanceBeam_v_BalanceBeam_g01_c01 \
  --print-tables
```

To regenerate E1/E2 statistics and plots from the frozen metric rows:

```bash
python code/scripts/analyze_extended_evidence.py \
  --suite-root extended_evidence_suite_20260906
```

## Rerun E1 or E2

E1 uses seeds `0,1,2,3,4,5,6,7,8,42`; E2 uses seed 42 and
`k=10,20,30,40`. These defaults are frozen in the runner.

```bash
python code/scripts/run_extended_allocation_suite.py \
  --experiment E1 \
  --base-manifest experiments_24h_k64/manifest.csv \
  --score-root experiments_icassp4_ablation_scores_20260905 \
  --trajectory-score-root extended_evidence_suite_20260906/E1_trajectory_scores \
  --output-dir new_runs/E1_multi_seed

python code/scripts/run_extended_allocation_suite.py \
  --experiment E2 \
  --base-manifest experiments_24h_k64/manifest.csv \
  --score-root experiments_icassp4_ablation_scores_20260905 \
  --output-dir new_runs/E2_k_sensitivity
```

Full reruns create several gigabytes of generated release videos; keep them
outside this frozen package.

## Regenerate selector score maps

The package includes the single checkpoints actually used:

- `code/VideoMAE/checkpoints/videomae_pretrain_base_patch16_224.pth`
- `code/RAFT/models/raft-things.pth`

Example deterministic score generation:

```bash
python code/scripts/generate_mask_ablation_scores_icassp.py \
  --manifest experiments_24h_k64/manifest.csv \
  --output-dir new_runs/selector_scores \
  --seed 42 --top-k 20 --dp-c-blk 2.0 --device cuda --raft-iters 12 \
  --variants no_trajectory,videomae_only \
  --videomae-root code/VideoMAE \
  --videomae-weight-path code/VideoMAE/checkpoints/videomae_pretrain_base_patch16_224.pth \
  --raft-root code/RAFT \
  --raft-ckpt code/RAFT/models/raft-things.pth
```

See `docs/EXPERIMENT_TRACE.md` and
`extended_evidence_suite_20260906/E1_E2_REPORT.md` for exact provenance and
interpretation.
