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

```bash
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

Datasets that aren't already present locally (MNIST, CIFAR-100, CelebA,
EMNIST, Shakespeare) are downloaded automatically into `./data` on first
use.
