# Validation

## Isolated recheck: 29 October 2026

A fresh per-project virtual environment was created on Windows with Python
3.12.10. The pinned runtime and development dependencies installed from wheels;
`pip check` passed. The environment's pip was updated to 26.2.2. The final
`pip-audit` scan reported no known vulnerabilities in that installed package
set at the time of the check. This is a dated advisory check, not a security guarantee.

- The preprocessing/model regression test passed; CLI help passed.
- Real MNIST completed all three default 12-epoch variants with seed 42 and
  51,000/9,000/10,000 train/validation/test samples.
- Test accuracy was 0.9649 for 28×28, 0.9528 for 16×16 and 0.9542 for 16×16
  with 16 intensity levels. See [docs/results.json](docs/results.json).
- Notebook format and syntax checks passed. Every code cell also ran sequentially
  on real MNIST with its original settings in a Python harness; no interactive
  Jupyter UI was exercised.
- Source/test formatting and focused Ruff syntax/import checks passed.

This is a single-seed experiment, not a statistically established benefit from
pixel quantization. Weights and arithmetic remain float32. Recorded model-only
throughput excludes preprocessing and transfers, and other experiments ran
concurrently on the host. There was no controlled hardware benchmark, GPU run
or end-to-end serving measurement. Remote CI has not exercised these local edits.

## Earlier preparation record: 27 September 2026

Validated on 2026-09-27 with Python 3.12 on Linux CPU.

Full corrected MNIST training and hardware throughput measurements are pending.

The source notebook passed schema checks and sequential execution during cleanup. For ML notebooks, external data were replaced only inside the test harness by small explicitly synthetic fixtures, and training was shortened. These checks verify code paths, not scientific performance. Published notebooks retain the full experiment settings and contain no synthetic scores.

The extracted command-line modules are additionally checked for importability, help output and the regression cases included in `tests/`. No training or download occurs on module import.
