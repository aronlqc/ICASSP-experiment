# Validation report

Validation was performed on 2026-09-06 from the root of `Minimal_package`.

## Packaging boundary

- The LaTeX dependency scanner found four required source assets: `paper.tex`,
  `spconf.sty`, the E2 sensitivity panel, and the qualitative mask panel.
- The frozen compiled `paper.pdf` was retained as the handoff snapshot.
- The package contains exactly 20 UCF101 inputs, 20 common backend inputs, the
  main/E1/E2 result records, retained score maps, one VideoMAE checkpoint, one
  RAFT checkpoint, and the necessary source files.
- E3--E5, old manuscripts, preview renders, virtual environments, Python
  caches, duplicated checkpoints, large regenerable release videos, and
  unrelated baseline outputs were excluded.

## Checks completed

1. Structural/data verification passed: all 20 raw videos and all 20 backend
   videos decode; retained score arrays match the decoded frame counts and
   `14 x 14` block grid; every active portable-manifest reference resolves.
2. Frozen result verification passed: main experiment 140 rows, E1 1,400 rows,
   E2 560 rows, E1 seeds `{0,1,2,3,4,5,6,7,8,42}`, E2 budgets
   `{10,20,30,40}`, seven policies per setting, and empty failure records.
3. Fresh-seed release test passed for seeds 101, 202, and 303: 7/7 policies
   completed for each seed; random-mask hashes differed across seeds and the
   deterministic RAT-fusion score-map hash remained identical.
4. Frozen E1 entry-point test passed on one full clip: all ten frozen seeds and
   seven policies completed, giving 70/70 rows and zero failures.
5. Frozen E2 entry-point test passed on one full clip: all four budgets and
   seven policies completed, giving 28/28 rows and zero failures.
6. Figure/table regeneration passed: the qualitative figure regenerated with
   SHA-256 `cbe476c1044c3781cd9fbd23362ae24dbf51a5da40a19900d37ce656d3f5cc79`;
   E1/E2 aggregate tables, paired tables, trace tables, and figures matched the
   frozen files byte-for-byte.
7. LaTeX compilation passed in an isolated output directory. The PDF has five
   US-Letter pages: four technical pages followed by references.
8. The included VideoMAE and RAFT checkpoints both loaded successfully through
   the included wrappers on CPU. Their hashes are checked by
   `code/scripts/verify_package.py`.

Run `python code/scripts/verify_package.py` after transfer. It verifies the
complete `CHECKSUMS.sha256` list as well as the structural and scientific
inventory checks above.
