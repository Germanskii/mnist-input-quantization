# MNIST Input Quantization

Measure the accuracy and runtime trade-offs of image resizing and pixel quantization in a small PyTorch MLP.

## Experiment

Three input variants share the same hidden-layer widths: original 28×28 images, resized 16×16 images, and 16×16 images with 16 pixel intensity levels. The first linear layer has fewer parameters for the smaller input. Training uses an 85/15 split of the official training set; evaluation uses the official test set.

The experiment quantizes **input values**, not weights or arithmetic. Tensors and model weights remain float32: this is not INT4 inference or packed 4-bit storage. Inference throughput is the median of 30 model-only timed batches after 10 warm-up batches; preprocessing and host-to-device transfers are excluded. A single seed does not establish statistical significance.

Outputs include CSV metrics, training histories, checkpoints and full-test confusion matrices.

## Quick start

Python 3.12 was used for local validation. Run commands from the repository root.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.benchmark
```

For the PyTorch projects, the pinned versions reproduce the tested CPU environment. Install compatible GPU wheels for your platform before running on CUDA.

```bash
python -m src.benchmark --epochs 1
```

The cleaned Colab/Jupyter experiment is in [`notebooks/experiment.ipynb`](notebooks/experiment.ipynb). To open it locally, install Jupyter separately (`python -m pip install jupyterlab`) and run `jupyter lab`. Command-line runs save figures instead of requiring an interactive window.

## Data

MNIST is downloaded by `torchvision.datasets.MNIST` on the first run into `data/`. No image files are bundled. Consult the original dataset terms before redistribution.

## Validation and results

Full corrected MNIST training and hardware throughput measurements are pending. See [VALIDATION.md](VALIDATION.md) for exactly what was checked. No historical notebook output is used as evidence for the corrected implementation.

```bash
python -m unittest discover -s tests -v
```

## Repository layout

- `src/`: importable implementation and command-line entry points.
- `notebooks/`: cleaned experiment notebook; original explanatory notes are in Russian.
- `tests/`: focused regression checks.
- `requirements.txt`: direct dependency versions used during validation.
- `DATA.md`: data access and redistribution notes.

This project was developed from a university Colab experiment and subsequently cleaned up for reproducibility. Generated data, trained weights and local paths are excluded from version control.
