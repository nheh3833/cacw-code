# Federated Learning Experiments

Research code for personalized federated learning experiments in PyTorch,
comparing a custom clustering/personalization approach ("our_approach")
against baselines and prior methods (FedAvg-style trivial training, an
oracle clustering baseline, IFCA, FedAMP, a threshold-based clustering
algorithm, FedRep, FedFomo, FedPAC, FedDisco, and Ditto) across MNIST,
rotated/relabeled MNIST, CIFAR-100, FEMNIST, CelebA, EMNIST, and the LEAF
Shakespeare next-character-prediction dataset.

## Project layout

- `models.py` — neural network architectures (MLPs, small CNNs, a VGG16
  variant, and a character-level LSTM).
- `federated_learning.py` — dataset loading/client-splitting utilities,
  the personalization/clustering algorithms, and the experiment runners.
- `main.py` — entry point that runs the experiment suite.
- `requirements.txt` / `pyproject.toml` — pinned dependencies.
- `INSTALL.md` — detailed installation and troubleshooting guide.
- `run_remoute.py` — uploads the code to a remote host over SSH, installs the
  dependencies there, and launches the experiment under `nohup`.
- `view_exp.py` — reconnects to that host and streams the experiment log.

`run_remoute.py` and `view_exp.py` ship with empty `HOST` / `USER` /
`PASSWORD` fields. Fill them in locally and do not commit real credentials.

## Setup

### 1. Create a virtual environment (recommended)

```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
# source .venv/bin/activate  # On Linux/Mac
```

### 2. Install dependencies

**For CPU (default):**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**For NVIDIA GPU (CUDA 12.1):**
```bash
pip install --upgrade pip
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install numpy scikit-learn matplotlib
```

Replace `cu121` with your CUDA version if different (e.g., `cu118` for CUDA 11.8).

### Troubleshooting

- **PyTorch won't install**: Ensure your Python is 3.8+. Check available PyTorch wheels for your Python version at https://pytorch.org/get-started/locally/
- **Torchvision version mismatch**: `torch` and `torchvision` must be a matching pair. This project uses `torch==2.2.2` with `torchvision==0.17.2`.
- **CUDA/GPU issues**: If you have an NVIDIA GPU, follow the GPU installation step above; CPU-only is fine for development.

## Running

```bash
python main.py
```

Which experiment runs is chosen by commenting and uncommenting the calls in
`main()` in `main.py`. By default it runs the two MNIST experiments
(`experiment_rel_mnist` and `experiment_rot_mnist`); the CIFAR-100 and
Shakespeare calls are present but commented out.

Datasets that aren't already present locally (MNIST, CIFAR-100, and the
Shakespeare corpus) are downloaded automatically into `./data` on first use.
Loaders for FEMNIST, CelebA and EMNIST also exist in `federated_learning.py`
but are not wired into any of the experiments in `main.py`.

## Known limitations

- **BatchNorm statistics in the CIFAR-100 experiments.** The gradient step
  evaluates every client in turn with the model in training mode, so the
  BatchNorm running statistics used at evaluation time are an exponential
  moving average dominated by the clients processed last, rather than by the
  client being evaluated. This depresses the reported CIFAR-100 accuracies for
  every method that shares this code path. Replacing BatchNorm with GroupNorm
  is the usual fix in federated settings.
- **Experiments are not independently seeded.** Each experiment builds its data
  split from whatever global RNG state the previous one left behind, so results
  are reproducible only when the full sequence in `main()` is run in order.
