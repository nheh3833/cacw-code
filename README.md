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

**For NVIDIA GPU (CUDA 11.8):**
```bash
pip install --upgrade pip
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install numpy scikit-learn matplotlib
```

Replace `cu118` with your CUDA version if different (e.g., `cu121` for CUDA 12.1).

### Troubleshooting

- **PyTorch won't install**: Ensure your Python is 3.8+. Check available PyTorch wheels for your Python version at https://pytorch.org/get-started/locally/
- **Torchvision version mismatch**: Must match your PyTorch version. If `torch==2.1.2` fails, update torchvision to `0.16.2`.
- **CUDA/GPU issues**: If you have an NVIDIA GPU, follow the GPU installation step above; CPU-only is fine for development.

## Running

```bash
python main.py
```

Datasets that aren't already present locally (MNIST, CIFAR-100, CelebA,
EMNIST, Shakespeare) are downloaded automatically into `./data` on first
use.
