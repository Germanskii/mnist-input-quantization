# MNIST Input Quantization

[![Checks](https://github.com/Germanskii/mnist-input-quantization/actions/workflows/checks.yml/badge.svg)](https://github.com/Germanskii/mnist-input-quantization/actions/workflows/checks.yml)
Measure the accuracy and runtime trade-offs of image resizing and pixel quantization in a small PyTorch MLP.

**Python · PyTorch · torchvision · pandas · scikit-learn · Matplotlib**

## Experiment

Three input variants share the same hidden-layer widths: original 28×28 images, resized 16×16 images, and 16×16 images with 16 pixel intensity levels. The first linear layer has fewer parameters for the smaller input. Training uses an 85/15 split of the official training set; evaluation uses the official test set.

The experiment quantizes **input values**, not weights or arithmetic. Tensors and model weights remain float32: this is not INT4 inference or packed 4-bit storage. Inference throughput is the median of 30 model-only timed batches after 10 warm-up batches; preprocessing and host-to-device transfers are excluded. A single seed does not establish statistical significance.

Outputs include CSV metrics, training histories, checkpoints and full-test confusion matrices.

## Quick start

Use **Python 3.12** for the pinned environment. The October recheck used Python
3.12.10 on Windows CPU. Run commands from the repository root.

```bash
git clone https://github.com/Germanskii/mnist-input-quantization.git
cd mnist-input-quantization
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m src.benchmark
```

For the PyTorch projects, the pinned versions reproduce the tested CPU environment. Install compatible GPU wheels for your platform before running on CUDA.

```bash
python -m src.benchmark --epochs 1
```

[Open in Colab](https://colab.research.google.com/github/Germanskii/mnist-input-quantization/blob/main/notebooks/experiment.ipynb). The cleaned Colab/Jupyter experiment is in [`notebooks/experiment.ipynb`](notebooks/experiment.ipynb). To open it locally, install Jupyter separately (`python -m pip install jupyterlab`) and run `jupyter lab`. Command-line runs save figures instead of requiring an interactive window.

## Data

MNIST is downloaded by `torchvision.datasets.MNIST` on the first run into `data/`. No image files are bundled. Consult the original dataset terms before redistribution.

## Validation and results

On **29 October 2026**, the default CLI completed **12 epochs per variant** on
real MNIST, using seed **42** and a **51,000 / 9,000 / 10,000** train/validation/test
split, on Windows / Python 3.12.10 / CPU:

| Input variant | Test accuracy | Parameters |
| --- | ---: | ---: |
| Original 28×28 | 96.49% | 39,106 |
| Resized 16×16 | 95.28% | 13,762 |
| Resized 16×16, 16 intensity levels | 95.42% | 13,762 |

The smaller input reduces parameter count by changing the first layer. Pixel
quantization adds no parameter reduction, and the small accuracy difference
between the two 16×16 variants is not evidence of an improvement from one seed.
Model-only throughput observations are retained in [docs/results.json](docs/results.json)
with their timing limits. Other experiments ran concurrently on the same host,
so those timings should not be treated as stable speedup or deployment estimates.
See [VALIDATION.md](VALIDATION.md) for the executed checks.

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

GitHub Actions runs the regression tests and validates notebook structure on pushes and pull requests. It does not download training datasets or establish model accuracy.

## Repository layout

- `src/`: importable implementation and command-line entry points.
- `notebooks/`: cleaned experiment notebook; original explanatory notes are in Russian.
- `tests/`: focused regression checks.
- `requirements.txt`: direct dependency versions used during validation.
- `DATA.md`: data access and redistribution notes.

This project was developed from a university Colab experiment and subsequently cleaned up for reproducibility. Generated data, trained weights and local paths are excluded from version control.


The portfolio cleanup and packaging used AI-assisted development. The notebooks derive from the original university work; validation limits are documented above.
