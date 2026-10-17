"""
Federated learning experiment code: dataset loading/splitting utilities,
personalization/clustering algorithms, and experiment runners.

This module preserves the original script's structure and global state
(e.g. seed, device, VOCAB_SIZE) so behavior is unchanged from the
single-file version; it only defines names, it does not run experiments
on import. See main.py for the runnable entry point.
"""
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import defaultdict
import math
import torch.nn.utils as utils
import time
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from sklearn.utils.extmath import randomized_svd
from collections import deque
from torch.optim import SGD
import urllib.request
import re
import string
import os
import json
from torch.utils.data import DataLoader, Subset

from models import (
    SimpleNN1,
    SimpleNN,
    SimpleNN_CIFAR100,
    VGG16_CIFAR101,
    VGG16_CIFAR10,
    CharLSTM,
    CNN_FEMNIST,
)

seed = 42
def show_random_samples_cifar(client_data, n=10):
    """
    Display n random images and labels from a CIFAR client dataset.

    client_data: tuple (client_images, client_labels)
        client_images: tensor [N, 3, 32, 32]
        client_labels: tensor [N]
    n: number of images to display
    """
    client_images, client_labels = client_data
    total = client_images.shape[0]
    indices = random.sample(range(total), n)

    plt.figure(figsize=(12, 2))
    for i, idx in enumerate(indices):
        img = client_images[idx]
        label = client_labels[idx].item()

        # Convert from (C, H, W) → (H, W, C)
        img = img.permute(1, 2, 0)

        plt.subplot(1, n, i + 1)
        plt.imshow(img.numpy())
        plt.title(label)
        plt.axis("off")

    plt.show()
def show_random_samples(client_data, n=10):
    """
    Display n random images and labels from a client dataset.

    client_data: tuple (client_images, client_labels)
        client_images: tensor [N,28,28] or [N,1,28,28]
        client_labels: tensor [N]
    n: number of images to display
    """
    client_images, client_labels = client_data
    total = client_images.shape[0]
    indices = random.sample(range(total), n)  # pick n random indices

    plt.figure(figsize=(12, 2))
    for i, idx in enumerate(indices):
        img = client_images[idx]
        label = client_labels[idx].item()

        # if shape is (1,28,28) -> remove channel
        if img.ndim == 3 and img.shape[0] == 1:
            img = img.squeeze(0)

        plt.subplot(1, n, i+1)
        plt.imshow(img.numpy(), cmap="gray")
        plt.title(label)
        plt.axis("off")

    plt.show()

# 2. Apply it everywhere
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# If you use CUDA:
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 1. Load MNIST dataset
transform = transforms.ToTensor()

trainset = torchvision.datasets.MNIST(root="./data", train=True,
                                      download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root="./data", train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # <--- For full determinism:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def rotate_fixed(img, degrees):
    """
    Rotate a MNIST tensor by a fixed number of degrees.
    img: torch.Tensor of shape (H,W) or (1,H,W)
    degrees: angle in degrees (positive = counterclockwise)
    """
    # If MNIST is (H,W) convert to (1,H,W)
    if img.ndim == 2:
        img = img.unsqueeze(0)

    # Use TF.rotate which supports tensor directly
    return TF.rotate(img, degrees)


def split_mnist_among_clients(num_clients, client_distributions,classs,type, seed=42,degre=90):
    """
    Splits MNIST training dataset into num_clients subsets according to given distributions.

    Args:
        num_clients (int): number of clients
        client_distributions (list of arrays): each array has 10 entries (digit proportions for that client)
        seed (int): random seed for reproducibility

    Returns:
        clients_data (list): list of (images, labels) for each client
    """
    rng = np.random.default_rng(seed)

    # Load MNIST train set
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = torchvision.datasets.MNIST(root="./data", train=True,
                                          download=True, transform=transform)

    images = trainset.data
    labels = trainset.targets

    # Group indices by digit
    digit_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        digit_indices[int(label)].append(idx)
    # Shuffle indices inside each digit group
    for d in digit_indices:
        rng.shuffle(digit_indices[d])
    # Compute total counts per digit
    total_digit_counts = {d: len(digit_indices[d]) for d in range(10)}
    # Prepare client datasets
    clients_data = []
    k=-1
    helpp=num_clients/classs
    for c in range(num_clients):
        dist = client_distributions[c]
        client_indices = []

        for digit in range(10):
            take = int(dist[digit] * total_digit_counts[digit])
            take = min(take, len(digit_indices[digit]))
            chosen = digit_indices[digit][:take]
            digit_indices[digit] = digit_indices[digit][take:]
            client_indices.extend(chosen)

        # save this client's dataset
        client_images = images[client_indices]
        client_labels = labels[client_indices]

        if(c%helpp==0):
            k=k+1
        if(type=="rot"):
            client_images=torch.stack([rotate_fixed(cli,degre*k) for cli in client_images])
        if(type=="rel"):
            client_labels=torch.stack([(lab+k)%10 for lab in client_labels])

        clients_data.append((client_images, client_labels))

    return clients_data
def create_custom_mnist_subset(proportions, seed=42):
    """
    Create a MNIST subset with class distribution given by `proportions`.

    Args:
        proportions (list of float): length 10, relative proportions of digits [0-9].
        train (bool): True for training set, False for test set.
        seed (int): random seed.

    Returns:
        torch.utils.data.Subset of MNIST
    """
    assert len(proportions) == 10, "Must give 10 proportions (for digits 0-9)"
    assert sum(proportions) > 0, "At least one class proportion must be > 0"
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load MNIST from torchvision
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    mnist = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )
    # Turn labels into numpy for easy handling
    targets = np.array(mnist.targets)

    indices = []
    for digit, p in enumerate(proportions):
        if p > 0:
            available = np.where(targets == digit)[0]
            # number of samples we want from this digit, based on proportion of available samples
            n_samples = int(len(available) * p )
            # Ensure we don't request more samples than available
            n_samples = min(n_samples, len(available))
            chosen = np.random.choice(available, n_samples, replace=False)
            indices.extend(chosen)

    np.random.shuffle(indices)
    return torch.utils.data.Subset(mnist, indices)


def split_cifar100_among_clients(num_clients,client_distributions,classs,type,seed=42):
    """
    Splits CIFAR-100 training dataset into num_clients subsets
    according to given class distributions.

    Args:
        num_clients (int): number of clients
        client_distributions (list of arrays): each array has 100 entries
        classs (int): number of clusters/classes for transformation logic
        type (str): "rot" or "rel"
        seed (int): random seed

    Returns:
        clients_data (list): list of (images, labels) for each client
    """
    rng = np.random.default_rng(seed)

    # Load CIFAR-100 train set
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761)
        )
    ])
    trainset = torchvision.datasets.CIFAR100(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    images = torch.stack([trainset[i][0] for i in range(len(trainset))])
    labels = torch.tensor([trainset[i][1] for i in range(len(trainset))])

    num_classes = 100

    # Group indices by class
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[int(label)].append(idx)

    # Shuffle indices inside each class
    for c in class_indices:
        rng.shuffle(class_indices[c])

    # Total samples per class
    total_class_counts = {
        c: len(class_indices[c]) for c in range(num_classes)
    }

    clients_data = []
    k = -1
    helpp = num_clients // classs

    for c in range(num_clients):
        dist = client_distributions[c]
        client_indices = []

        for cls in range(num_classes):
            take = int(dist[cls] * total_class_counts[cls])
            take = min(take, len(class_indices[cls]))

            chosen = class_indices[cls][:take]
            class_indices[cls] = class_indices[cls][take:]
            client_indices.extend(chosen)

        client_images = images[client_indices]
        client_labels = labels[client_indices]

        if c % helpp == 0:
            k += 1

        if type == "rot":
            client_images = torch.stack([
                rotate_fixed(img, 90 * k) for img in client_images
            ])

        if type == "rel":
            client_labels = (client_labels + k) % num_classes

        clients_data.append((client_images, client_labels))

    return clients_data
def create_custom_cifar100_subset(proportions, seed=42):
    """
    Create a CIFAR-100 subset with class distribution given by `proportions`.

    Args:
        proportions (list of float): length 100, relative proportions of classes [0-99].
        seed (int): random seed.

    Returns:
        torch.utils.data.Subset of CIFAR-100
    """
    assert len(proportions) == 100, "Must give 100 proportions (for classes 0-99)"
    assert sum(proportions) > 0, "At least one class proportion must be > 0"

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load CIFAR-100 (test set, same as your MNIST code)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761)
        )
    ])
    cifar = torchvision.datasets.CIFAR100(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    # Convert labels to numpy for easy handling
    targets = np.array(cifar.targets)

    indices = []
    for cls, p in enumerate(proportions):
        if p > 0:
            available = np.where(targets == cls)[0]

            # number of samples we want from this class
            n_samples = int(len(available) * p)
            n_samples = min(n_samples, len(available))

            chosen = np.random.choice(available, n_samples, replace=False)
            indices.extend(chosen)

    np.random.shuffle(indices)
    return torch.utils.data.Subset(cifar, indices)

# --- Ensure Vocabulary is defined globally ---
ALL_LETTERS = string.printable
VOCAB_DICT = {letter: i for i, letter in enumerate(ALL_LETTERS)}
VOCAB_SIZE = len(ALL_LETTERS)

def load_federated_shakespeare_python(num_clients, device, users_per_client=5, seq_length=50):

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    try:
        with urllib.request.urlopen(url) as response:
            text = response.read().decode('utf-8')
    except Exception as e:

        return [], None

    # 1. Parse characters and their lines using Regex
    pattern = re.compile(r"^([A-Z ]+):\n(.*?)(?=^[A-Z ]+:\n|\Z)", re.MULTILINE | re.DOTALL)
    matches = pattern.findall(text)

    char_text_map = defaultdict(str)
    for char_name, lines in matches:
        # Clean up newlines so the text flows naturally
        char_text_map[char_name.strip()] += lines.replace('\n', ' ')

    # 2. Filter characters that actually have enough text to form sequences
    # We need characters that have at least a few sequences of text
    valid_chars = [char for char, txt in char_text_map.items() if len(txt) > (seq_length * 4)]

    # Sort by amount of text descending so we get the most "talkative" characters
    valid_chars = sorted(valid_chars, key=lambda c: len(char_text_map[c]), reverse=True)

    total_needed = num_clients * users_per_client
    if len(valid_chars) < total_needed:

        # Sample with replacement if we request more characters than exist in the play
        sampled_chars = random.choices(valid_chars, k=total_needed)
    else:
        # Sample without replacement
        sampled_chars = random.sample(valid_chars, total_needed)

    clients_data = []
    all_test_x = []
    all_test_y = []

    # Helper function to chunk continuous text into (X, Y) sequence tensors
    def chunk_text(combined_text):
        # Map characters to ints, defaulting to 0 for unknown characters
        encoded = [VOCAB_DICT.get(c, 0) for c in combined_text]
        x_seqs, y_seqs = [], []

        # Stride = seq_length to prevent massive overlapping and save GPU RAM (matches LEAF logic)
        stride = seq_length
        for i in range(0, len(encoded) - seq_length, stride):
            x_seqs.append(encoded[i : i + seq_length])
            y_seqs.append(encoded[i + 1 : i + seq_length + 1]) # Next char prediction
        return x_seqs, y_seqs

    # 3. Build clients
    char_idx = 0
    for i in range(num_clients):
        client_train_x, client_train_y = [], []

        # Group 'users_per_client' (e.g., 5) characters into 1 client
        for _ in range(users_per_client):
            char_name = sampled_chars[char_idx]
            char_idx += 1

            txt = char_text_map[char_name]

            # Split 80% train, 20% test for this specific character
            split_point = int(len(txt) * 0.8)
            train_txt = txt[:split_point]
            test_txt = txt[split_point:]

            tr_x, tr_y = chunk_text(train_txt)
            te_x, te_y = chunk_text(test_txt)

            client_train_x.extend(tr_x)
            client_train_y.extend(tr_y)

            if i == 0:
                all_test_x.extend(te_x)
                all_test_y.extend(te_y)

        # Shuffle the client's internal data so the 5 characters are mixed together in batches
        if len(client_train_x) > 0:
            combined = list(zip(client_train_x, client_train_y))
            random.shuffle(combined)
            client_train_x, client_train_y = zip(*combined)

            x_tensor = torch.tensor(client_train_x, dtype=torch.long).to(device)
            y_tensor = torch.tensor(client_train_y, dtype=torch.long).to(device)
        else:
            x_tensor = torch.tensor([], dtype=torch.long).to(device)
            y_tensor = torch.tensor([], dtype=torch.long).to(device)

        clients_data.append((x_tensor, y_tensor))

    # 4. Build global test loader
    x_test_tensor = torch.tensor(all_test_x, dtype=torch.long).to(device)
    y_test_tensor = torch.tensor(all_test_y, dtype=torch.long).to(device)

    testset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)
    # num_workers=0 because it's already on the GPU
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)



    return clients_data, testloader

def get_celeba_clients(num_clients=20, min_images=7, target_attr="Smiling"):


    # 1. Transform to 32x32 so it perfectly fits your VGG16 model
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 2. Download and load the dataset
    dataset = torchvision.datasets.CelebA(
        root='./data',
        split='all',
        target_type=['attr', 'identity'],
        transform=transform,
        download=True
    )

    # Find the column index for "Smiling"
    attr_idx = dataset.attr_names.index(target_attr)

    # 3. Fast grouping: Find identities that have at least 'min_images' photos
    identity_tensor = dataset.identity.squeeze()
    unique_ids, counts = torch.unique(identity_tensor, return_counts=True)

    valid_ids = unique_ids[counts >= min_images]

    # Select exactly the 20 clients we want
    selected_ids = valid_ids[:num_clients]

    clients_data = []

    # 4. Extract and format the data for your framework
    for person_id in selected_ids:
        # Get all image indices for this specific person
        indices = (identity_tensor == person_id).nonzero(as_tuple=True)[0]

        subset = Subset(dataset, indices)
        # Load all images for this client in one giant batch
        loader = DataLoader(subset, batch_size=len(indices), shuffle=True)

        for images, targets in loader:
            # targets[0] contains the 40 attributes. targets[1] is the identity.
            attributes = targets[0]

            # Extract just the Smiling column, and ensure it is a LongTensor for CrossEntropyLoss
            labels = attributes[:, attr_idx].long()

            # Append in your exact (images_tensor, labels_tensor) format
            clients_data.append((images, labels))


    return clients_data



def load_federated_emnist_python(num_clients, device):

    # Download the standard EMNIST "byclass" dataset (62 classes: digits + letters)
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = torchvision.datasets.EMNIST(root='./data', split='byclass', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.EMNIST(root='./data', split='byclass', train=False, download=True, transform=transform)

    clients_data = []

    # Group the indices by class so we can sample them non-IID
    targets = np.array(train_data.targets)
    class_indices = {i: np.where(targets == i)[0] for i in range(62)}

    for i in range(num_clients):
        # 1. Simulate LEAF's non-IID nature: Each client only has 2 to 10 classes
        num_classes_for_client = np.random.randint(2, 11)
        client_classes = np.random.choice(62, num_classes_for_client, replace=False)

        # 2. Simulate LEAF's unbalanced nature: Clients have anywhere from 100 to 1000 samples
        num_samples_for_client = np.random.randint(100, 1000)
        samples_per_class = num_samples_for_client // num_classes_for_client

        client_idx = []
        for c in client_classes:
            avail_idx = class_indices[c]
            if len(avail_idx) > 0:
                take_n = min(samples_per_class, len(avail_idx))
                chosen = np.random.choice(avail_idx, take_n, replace=False)
                client_idx.extend(chosen)

        np.random.shuffle(client_idx)

        # Extract data and move to GPU
        # EMNIST images need to be reshaped and rotated/transposed properly
        x_client = train_data.data[client_idx].float().unsqueeze(1) / 255.0
        # EMNIST images are flipped by default in torchvision, fix them:
        x_client = x_client.transpose(2, 3)

        y_client = train_data.targets[client_idx]

        clients_data.append((x_client.to(device), y_client.to(device)))

    # Process Test Data

    x_test = test_data.data.float().unsqueeze(1) / 255.0
    x_test = x_test.transpose(2, 3)
    y_test = test_data.targets

    testset_gpu = torch.utils.data.TensorDataset(x_test.to(device), y_test.to(device))
    testloader = torch.utils.data.DataLoader(testset_gpu, batch_size=256, shuffle=False)


    return clients_data, testloader

# --- 1. Define the Global Variables ---
# These will hold the dataset in memory so your client-splitting functions can access them instantly.
# ==========================================
SHAKESPEARE_TRAIN = []
SHAKESPEARE_TEST = []
VOCAB_SIZE = 0
NUM_TOTAL_CLASSES = 0
def text_to_tensor(text_list):
    """Converts a list of strings into a tensor of integers."""
    indices = []
    for text in text_list:
        # Default to index 0 if character is strangely missing from printable ASCII
        seq = [VOCAB_DICT.get(c, 0) for c in text]
        indices.append(seq)
    return torch.tensor(indices, dtype=torch.long)

def load_shakespeare_clients(train_dir, test_dir, device, max_clients=50):
    clients_data = []
    all_test_x, all_test_y = [], []

    # 1. Load Train Data
    train_files = [f for f in os.listdir(train_dir) if f.endswith('.json')]
    clients_loaded = 0

    for file_name in train_files:
        with open(os.path.join(train_dir, file_name), 'r') as f:
            data = json.load(f)

        for user in data['users']:
            if clients_loaded >= max_clients:
                break

            x_strings = data['user_data'][user]['x']
            y_strings = data['user_data'][user]['y']

            x_tensor = text_to_tensor(x_strings).to(device)
            y_tensor = text_to_tensor(y_strings).squeeze().to(device) # y is just 1 char

            clients_data.append((x_tensor, y_tensor))
            clients_loaded += 1

    # 2. Load Test Data
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.json')]
    for file_name in test_files:
        with open(os.path.join(test_dir, file_name), 'r') as f:
            data = json.load(f)
        valid_users = list(data['user_data'].keys())[:max_clients]
        for user in valid_users:
            all_test_x.extend(data['user_data'][user]['x'])
            all_test_y.extend(data['user_data'][user]['y'])

    x_test_gpu = text_to_tensor(all_test_x).to(device)
    y_test_gpu = text_to_tensor(all_test_y).squeeze().to(device)

    testset_gpu = torch.utils.data.TensorDataset(x_test_gpu, y_test_gpu)
    testloader = torch.utils.data.DataLoader(testset_gpu, batch_size=32, shuffle=False, num_workers=0)


    return clients_data, testloader

# ==========================================
# 2. DATA PREP & SPLIT FUNCTIONS
# ==========================================



def init_shakespeare_all_classes(seq_length=50, test_split=0.2):
    global SHAKESPEARE_TRAIN, SHAKESPEARE_TEST, VOCAB_SIZE, NUM_TOTAL_CLASSES


    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    file_path = "shakespeare.txt"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except:
        urllib.request.urlretrieve(url, file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

    chars = tuple(set(text))
    char2int = {ch: ii for ii, ch in enumerate(chars)}
    VOCAB_SIZE = len(chars)

    pattern = re.compile(r"^([A-Z ]+):\n(.*?)(?=^[A-Z ]+:\n|\Z)", re.MULTILINE | re.DOTALL)
    matches = pattern.findall(text)

    char_text_map = defaultdict(str)
    for char_name, lines in matches:
        char_text_map[char_name.strip()] += lines.replace('\n', ' ')

    sorted_chars = sorted(char_text_map.items(), key=lambda x: len(x[1]), reverse=True)
    NUM_TOTAL_CLASSES = len(sorted_chars)

    for class_id, (char_name, char_text) in enumerate(sorted_chars):
        encoded = [char2int[ch] for ch in char_text]
        if len(encoded) > seq_length + 1:
            split_idx = int(len(encoded) * (1 - test_split))
            train_encoded = encoded[:split_idx]
            test_encoded = encoded[split_idx:]

            for i in range(0, len(train_encoded) - seq_length):
                x = train_encoded[i : i + seq_length]
                y = train_encoded[i + 1 : i + seq_length + 1]
                SHAKESPEARE_TRAIN.append((torch.tensor(x), torch.tensor(y), class_id))

            for i in range(0, len(test_encoded) - seq_length):
                x = test_encoded[i : i + seq_length]
                y = test_encoded[i + 1 : i + seq_length + 1]
                SHAKESPEARE_TEST.append((torch.tensor(x), torch.tensor(y), class_id))



def split_shakespeare_among_clients(num_clients, client_distributions, seed=42):
    rng = np.random.default_rng(seed)
    global SHAKESPEARE_TRAIN, NUM_TOTAL_CLASSES
    class_indices = defaultdict(list)

    for idx, (_, _, cls_id) in enumerate(SHAKESPEARE_TRAIN):
        class_indices[cls_id].append(idx)

    for c in class_indices:
        rng.shuffle(class_indices[c])

    total_class_counts = {c: len(class_indices.get(c, [])) for c in range(NUM_TOTAL_CLASSES)}
    clients_data = []

    for c in range(num_clients):
        dist = client_distributions[c]
        client_seqs = []
        client_targets = []

        for cls in range(NUM_TOTAL_CLASSES):
            if cls < len(dist):
                take = int(dist[cls] * total_class_counts[cls])
                take = min(take, len(class_indices[cls]))
                chosen = class_indices[cls][:take]
                class_indices[cls] = class_indices[cls][take:]

                for idx in chosen:
                    seq, target, _ = SHAKESPEARE_TRAIN[idx]
                    client_seqs.append(seq)
                    client_targets.append(target)

        if len(client_seqs) > 0:
            client_seqs_tensor = torch.stack(client_seqs)
            client_targets_tensor = torch.stack(client_targets)
        else:
            client_seqs_tensor = torch.tensor([])
            client_targets_tensor = torch.tensor([])

        clients_data.append((client_seqs_tensor, client_targets_tensor))

    return clients_data


def create_custom_shakespeare_subset(proportions, seed=42):
    global SHAKESPEARE_TEST, NUM_TOTAL_CLASSES
    torch.manual_seed(seed)
    np.random.seed(seed)
    test_classes = np.array([item[2] for item in SHAKESPEARE_TEST])
    indices = []

    for cls, p in enumerate(proportions):
        if p > 0 and cls < NUM_TOTAL_CLASSES:
            available = np.where(test_classes == cls)[0]
            n_samples = int(len(available) * p)
            n_samples = min(n_samples, len(available))
            if n_samples > 0:
                chosen = np.random.choice(available, n_samples, replace=False)
                indices.extend(chosen)

    np.random.shuffle(indices)
    subset = []
    for idx in indices:
        seq, target, _ = SHAKESPEARE_TEST[idx]
        subset.append((seq, target))
    return subset



def trivial_approach(clients_data, testloader, batch, model_type, lr=0.0005):

    progress = []

    if model_type == "fully":
        model = SimpleNN()
        model.to(device)
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=decay
        )
    if model_type == "conv":
        model = VGG16_CIFAR10(num_classes=100)
        model.to(device)
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=decay
        )

    criterion = nn.CrossEntropyLoss()
    num_clients = len(clients_data)
    weights = torch.ones(num_clients) / num_clients
    weights = weights.to(torch.float32) # Ensure weights is a tensor of float32
    batch_size = batch
    counter = 0
    num_batches = math.ceil(len(clients_data[0][0]) / batch_size)

    for epoch_idx in range(itr_num):
        clients_data = shuffle_clients_data(clients_data)
        start = 0
        for batch_idx in range(num_batches):
            end = min(start + batch_size, len(clients_data[0][0]))
            current_batch = get_batch(clients_data, start, end)
            grads = get_grad_no_var(current_batch, model, criterion, optimizer)

            step_no_var(grads, weights, model, optimizer)
            start = end
            counter += 1

        if epoch_idx % jump == 0 and epoch_idx != 0:
            if model_type == "fully":
                progress.append(accuracy(model, testloader, True))
            if model_type == "conv":
                progress.append(accuracy(model, testloader, False))
            elif model_type == "lstm": # Added for Shakespeare
                # Sequence data requires a different accuracy calculation
                progress.append(accuracy_sequence(model, testloader))


    return progress


def oracle_approach(clients_data, testloader, num_cluster, batch, model_type, lr=0.0005, rot_ecp="no"):

    progress = []

    if model_type == "fully":
        model = SimpleNN()
        model.to(device)
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=decay
        )
    elif model_type == "conv":
        model = VGG16_CIFAR10(num_classes=100)
        model.to(device)
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=decay
        )
    elif model_type == "lstm" or model_type == "shakespeare":
        model = CharLSTM(vocab_size=VOCAB_SIZE)
        model.to(device)
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=decay
        )

    criterion = nn.CrossEntropyLoss()
    num_clients = len(clients_data)
    weights = torch.zeros(num_clients) / num_clients
    same = num_clients / num_cluster

    if rot_ecp == "no":
        for i in range(int(same)):
            weights[i] = 1 / same
    else:
        for i in range(math.floor(same / 2)):
            weights[i] = 1 / same
            weights[-i] = 1 / same

    weights = weights.to(torch.float32) # Ensure weights is a tensor of float32
    batch_size = batch
    counter = 0
    num_batches = math.ceil(len(clients_data[0][0]) / batch_size)

    for epoch_idx in range(itr_num):
        clients_data = shuffle_clients_data(clients_data)
        start = 0
        for batch_idx in range(num_batches):
            end = min(start + batch_size, len(clients_data[0][0]))
            if (end - start) <= 1:
                # IMPORTANT: You must advance 'start' before skipping,
                # otherwise you get an infinite loop!
                start = end
                continue

            current_batch = get_batch(clients_data, start, end)
            grads = get_grad_no_var(current_batch, model, criterion, optimizer)
            step_no_var(grads, weights, model, optimizer)
            start = end
            counter += 1

        if epoch_idx % jump == 0 and epoch_idx != 0:
            if model_type == "fully":
                progress.append(accuracy(model, testloader, True))
            if model_type == "conv":
                progress.append(accuracy(model, testloader, False))
            elif model_type == "lstm": # Added for Shakespeare
                # Sequence data requires a different accuracy calculation
                progress.append(accuracy_sequence(model, testloader))
    return progress


def only_my_data_approach(clients_data, testloader, batch, model_type, lr=0.0005):

    progress = []

    if model_type == "fully":
        model = SimpleNN()
        model.to(device)
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=decay
        )
    elif model_type == "conv":
        model = VGG16_CIFAR10(num_classes=100)
        model.to(device)
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=decay
        )
    elif model_type == "lstm" or model_type == "shakespeare":
        model = CharLSTM(vocab_size=VOCAB_SIZE)
        model.to(device)
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=decay
        )

    criterion = nn.CrossEntropyLoss()
    num_clients = len(clients_data)
    weights = torch.zeros(num_clients) / num_clients
    weights[0] = 1
    weights = weights.to(torch.float32) # Ensure weights is a tensor of float32
    batch_size = batch
    num_batches = math.ceil(len(clients_data[0][0]) / batch_size)

    for epoch_idx in range(itr_num):
        clients_data = shuffle_clients_data(clients_data)
        start = 0
        for batch_idx in range(num_batches):
            end = min(start + batch_size, len(clients_data[0][0]))
            if (end - start) <= 1:
                # IMPORTANT: You must advance 'start' before skipping,
                # otherwise you get an infinite loop!
                start = end
                continue

            current_batch = get_batch(clients_data, start, end)
            grads = get_grad_no_var(current_batch, model, criterion, optimizer)
            step_no_var(grads, weights, model, optimizer)
            start = end

        if epoch_idx % jump == 0 and epoch_idx != 0:
            if model_type == "fully":
                progress.append(accuracy(model, testloader, True))
            if model_type == "conv":
                progress.append(accuracy(model, testloader, False))

            elif model_type == "lstm": # Added for Shakespeare
                # Sequence data requires a different accuracy calculation
                progress.append(accuracy_sequence(model, testloader))

    return progress

def load_femnist_clients(train_dir, test_dir, device, max_clients=50):
    clients_data = []
    all_test_x, all_test_y = [], []

    # 1. Load Train Data
    train_files = [f for f in os.listdir(train_dir) if f.endswith('.json')]
    clients_loaded = 0

    for file_name in train_files:
        with open(os.path.join(train_dir, file_name), 'r') as f:
            data = json.load(f)

        for user in data['users']:
            if clients_loaded >= max_clients:
                break
            # Reshape to (N, 1, 28, 28) for the CNN
            x_tensor = torch.tensor(data['user_data'][user]['x'], dtype=torch.float32).view(-1, 1, 28, 28).to(device)
            y_tensor = torch.tensor(data['user_data'][user]['y'], dtype=torch.long).to(device)
            clients_data.append((x_tensor, y_tensor))
            clients_loaded += 1

    # 2. Load Test Data
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.json')]
    for file_name in test_files:
        with open(os.path.join(test_dir, file_name), 'r') as f:
            data = json.load(f)
        valid_users = list(data['user_data'].keys())[:max_clients]
        for user in valid_users:
            all_test_x.extend(data['user_data'][user]['x'])
            all_test_y.extend(data['user_data'][user]['y'])

    x_test_gpu = torch.tensor(all_test_x, dtype=torch.float32).view(-1, 1, 28, 28).to(device)
    y_test_gpu = torch.tensor(all_test_y, dtype=torch.long).to(device)

    testset_gpu = torch.utils.data.TensorDataset(x_test_gpu, y_test_gpu)
    testloader = torch.utils.data.DataLoader(testset_gpu, batch_size=32, shuffle=False, num_workers=0)

    return clients_data, testloader




@torch.jit.script
def _squared_2_norm_jit(tensor):
    return torch.sum(tensor ** 2)

@torch.jit.script
def _project_simplex_jit(v):
    n_features = v.shape[0]
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1
    ind = torch.arange(1, n_features + 1, device=v.device, dtype=v.dtype)
    cond = u * ind > cssv
    rho = torch.max(torch.where(cond, ind.long(), torch.zeros_like(ind).long())) - 1
    theta = cssv[rho] / (rho + 1)
    return torch.clamp(v - theta, min=0.0)

@torch.jit.script
def _non_convex_grad_jit(bias, gradients, weights, L: float, reg: float):
    # Vectorized gradient calculation
    coeff2 = 2.0 * L
    val_bias = torch.mv(bias.t(), weights)
    val_grad = torch.mv(gradients.t(), weights)
    term1 = torch.mv(bias, val_bias)
    term2 = torch.mv(gradients, val_grad)
    grad = 2.0 * term1.abs() + coeff2 * term2.abs()
    grad += reg * (weights ** 2)
    return grad / gradients.size(0)

@torch.jit.script
def _find_weights_solver_jit(bias, grad_list, w, L: float, reg: float, lr: float, max_steps: int):
    tolerance = 2e-3
    for _ in range(max_steps):
        w_old = w.clone()

        # Calculate Gradient
        grad = _non_convex_grad_jit(bias, grad_list, w, L, reg)

        # Normalize Gradient
        grad_norm = torch.norm(grad)
        if grad_norm > 1e-12:
            grad = grad / grad_norm

        # Gradient Step
        w_unconstrained = w - lr * grad

        # Projection Step
        w = _project_simplex_jit(w_unconstrained)

        # Convergence Check
        if torch.norm(w - w_old) < tolerance:
            break
    return w

def squared_2_norm(tensor):
    """Return squared 2-norm of a tensor (or matrix row-wise)"""
    # If input is [N, D], returns sum of norms squared.
    # If input is 1D, returns norm squared.
    return torch.sum(tensor ** 2)

def vector_multiply(t1, t2):
    """Dot product optimized for GPU tensors"""
    # Assumes t1 and t2 are already flat tensors
    return torch.abs(torch.dot(t1.view(-1), t2.view(-1)))

def project_simplex2(v):
    """
    Projects vector v onto the probability simplex (w >= 0, sum(w) = 1).
    Efficient GPU implementation.
    """
    if v.dim() > 1: v = v.view(-1)
    n_features = v.shape[0]

    # 1. Sort descending
    u, _ = torch.sort(v, descending=True)

    # 2. Cumulative sum
    cssv = torch.cumsum(u, dim=0)

    # 3. Find the index (rho)
    indices = torch.arange(n_features, device=v.device) + 1
    cond = u - (cssv - 1) / indices > 0

    # Find max index where condition is true
    rho = torch.max(torch.where(cond, indices, torch.zeros_like(indices))) - 1
    if rho < 0: rho = 0

    # 4. Threshold and project
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w = torch.clamp(v - theta, min=0)
    return w

def project_simplex(v):
    n_features = v.shape[0]
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1

    # Vectorized check
    ind = torch.arange(1, n_features + 1, device=v.device)
    cond = u * ind > cssv

    # Use nonzero to find the last True index quickly
    rho = torch.nonzero(cond).max()
    theta = cssv[rho] / (rho + 1)

    return torch.clamp(v - theta, min=0)

def find_weights_all(bias, var, grad_list, D, iteration_size, w, L, model_type, regularization, dataset, lambda_val=0.01, lr=0.01):
    """
    Optimized Iterative Projected Gradient Descent (PGD).
    """
    # 1. Settings based on dataset (Logic preserved)
    if dataset == "mnist":
        lr_internal = 0.1
        max_steps = 200
    else:
        lr_internal = 0.1
        max_steps = 200

    # 2. Optimization Loop (Replaced with JIT solver)
    # We pass 'lr_internal' which overrides the 'lr' argument based on your logic above
    w = _find_weights_solver_jit(bias, grad_list, w, float(L), float(regularization), float(lr_internal), int(max_steps))

    return w

def non_convex_grad(bias, gradients, weights, L, eta, reg):
    # bias: [N, D_pca], gradients: [N, D_pca], weights: [N]
    coeff2 = 2 * L * eta

    # Matrix-Vector product: (N, D) @ (D, 1) -> (N)
    # This computes the interaction of all clients with the weighted average
    val_bias = bias.t() @ weights
    val_grad = gradients.t() @ weights

    # Efficiently compute the gradient for all weights simultaneously
    # term1[i] = bias[i] * weighted_avg_bias
    term1 = bias @ val_bias
    term2 = gradients @ val_grad

    # Use torch.abs directly on the vectors
    grad = 2 * term1.abs() + coeff2 * term2.abs()

    # Regularization
    grad += reg * (weights ** 2)

    return grad / len(gradients)

def update_bias(grads_tensor, counter, bias_buffer):
    """
    Efficiently updates bias history using in-place GPU operations.
    grads_tensor: [N_clients, Total_Params]
    bias_buffer:  [N_clients, Total_Params] or None
    """
    # 1. Calculate Difference (Reference is Client 0)
    ref_grad = grads_tensor[0:1] # Keep dim for broadcasting
    bias_help = grads_tensor - ref_grad

    # 2. Initialize if first run
    if bias_buffer is None:
        return bias_help.clone()

    # 3. Update Moving Average (In-Place)
    if counter > 100:
        # Exponential MA: old * 0.99 + new * 0.01
        bias_buffer.mul_(0.99).add_(bias_help, alpha=0.01)
    else:
        # Cumulative MA
        denom = counter + 1.0
        bias_buffer.mul_(counter / denom).add_(bias_help, alpha=(1.0 / denom))

    return bias_buffer

def our_approach(clients_data, testloader, batch, model_type, lr=0.0005, cluster="no", do_print="yes",do_reg="no",do_pca="no"):

    learn = 5
    progress = []

    grad_avg = None

    if model_type == "fully":
        model = SimpleNN()
        model.to(device)
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=decay
        )
        dataset = "mnist"
        # FIX: Move x and y to the correct device
        x1, y1 = testset[0]
        x = x1.unsqueeze(0).to(device) # Moved to device
        y = (torch.tensor([y1]) if not torch.is_tensor(y1) else y1.unsqueeze(0)).to(device) # Moved to device

    if model_type == "conv":
        model = VGG16_CIFAR10(num_classes=100)
        model.to(device)
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=decay
        )
        x = torch.zeros(1, 3, 32, 32).to(device)
        y = torch.zeros(1, dtype=torch.long).to(device)
        dataset = "cifar"
    elif model_type == "lstm" or model_type == "shakespeare":
        model = CharLSTM(vocab_size=VOCAB_SIZE)
        model.to(device)

        # Switched to Adagrad for adaptive learning rates on text embeddings
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=decay
        )
        # Dummy tensors for text: (batch_size=1, sequence_length=50)
        # Must be dtype=torch.long because they are token indices, not floats
        x = torch.zeros(1, 50, dtype=torch.long).to(device)
        y = torch.zeros(1, 50, dtype=torch.long).to(device)
        dataset = "lstm"
    criterion = nn.CrossEntropyLoss()
    num_clients = len(clients_data)
    weights = torch.zeros(num_clients) / num_clients
    weights[0] = 1
    weights = weights.to(device).to(torch.float32)
    batch_size = batch
    counter = 0
    help = math.ceil(len(clients_data[0][0]) / batch_size)
    lambda_val = 0.01
    bias_buffer = None

    D = 0.05
    bias = None
    var = None
    regularization = 0

    for epoch_idx in range(itr_num):
        clients_data = shuffle_clients_data(clients_data)

        total_grads_sum = None # Will be [N, P]
        sum_first_client = 0.0

        start = 0

        c0_count = 0
        c0_mean = None
        c0_M2 = None  # Sum of squared differences from the mean

        # --- FIX 2: Loop over 'help' (number of batches), not 'batch' (batch size) ---

        for batch_idx in range(help):
            end = min(start + batch_size, len(clients_data[0][0]))
            if (end - start) <= 1:
                # IMPORTANT: You must advance 'start' before skipping,
                # otherwise you get an infinite loop!
                start = end
                continue
            # Get Batch
            current_batch = get_batch(clients_data, start, end)

            # Compute Gradients (Standard PyTorch)
            # This returns list of lists: [[p1, p2], [p1, p2]...]
            grads_list_of_lists = get_grad_no_var(current_batch, model, criterion, optimizer)

            # Step optimizer (Standard)
            step_no_var(grads_list_of_lists, weights, model, optimizer)

            # --- OPTIMIZATION START ---

            # 1. Flatten gradients (Handle None values safely)
            grads_flat = []

            # Create a reference zero vector to use when a client has no data
            # We detach it so it doesn't affect the computation graph
            dummy_vec = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
            zero_vec = torch.zeros_like(dummy_vec)

            for g in grads_list_of_lists:
                if g is None:
                    # If client had no data, use zero gradient
                    grads_flat.append(zero_vec)
                else:
                    # Otherwise, flatten their gradient
                    grads_flat.append(torch.nn.utils.parameters_to_vector(g))

            grads_tensor = torch.stack(grads_flat)

            # 2. Update Bias History (Fast In-Place)
            counter += 1
            # We update 'bias_buffer' here
            bias_buffer = update_bias(grads_tensor, counter, bias_buffer)

            # 3. Accumulate Gradients for this Epoch
            if total_grads_sum is None:
                total_grads_sum = grads_tensor
            else:
                total_grads_sum.add_(grads_tensor)

            # 4. Accumulate for Regularization (Client 0 only)
            sum_first_client += grads_tensor[0]

            # --- OPTIMIZATION END ---

            c0_grad = grads_tensor[0].detach()
            c0_count += 1

            if c0_mean is None:
                c0_mean = c0_grad.clone()
                c0_M2 = torch.zeros_like(c0_grad)
            else:
                delta = c0_grad - c0_mean
                c0_mean += delta / c0_count
                delta2 = c0_grad - c0_mean
                c0_M2 += delta * delta2

            start = end

        # --- AFTER THE LOOP ---
        if c0_count > 1:
            # c0_M2 / (c0_count - 1) gives the exact same round-to-round variance per parameter
            # as torch.var(..., dim=0).
            # .sum() then adds those variances up for the final scalar.
            var = (c0_M2.sum() / (c0_count - 1)).item()
        else:
            var = 0.0

        L = estimate_L(model, criterion, x, y)

        # regularization=(torch.norm(sum_first_client)**2)*2 / help
        if (do_reg=="yes"):
            regularization = var
        else:
            regularization=0
        if dataset == "cifar":
            regularization = regularization

        if epoch_idx >= 0:
            final_avg_grads = total_grads_sum / help

            # --- FIX 3: Pass 'bias_buffer', not 'bias' ---
            weights = update_weights(weights, bias_buffer, var, final_avg_grads, L, lr, lambda_val, D, model_type, regularization, dataset, "no",do_pca)

            if cluster == "yes":
                divide = 1 / num_clients
                weights = torch.where(weights >= divide, weights, torch.zeros_like(weights))
                n = (weights > 0).sum()
                weights = torch.where(weights > 0, torch.ones_like(weights) / n, weights)

            if do_print == "yes":
                print(weights)
        if epoch_idx % jump == 0 and epoch_idx != 0:
            if model_type == "fully":
                progress.append(accuracy(model, testloader, True))
            if model_type == "conv":
                progress.append(accuracy(model, testloader, False))
            elif model_type == "lstm": # Added for Shakespeare
                # Sequence data requires a different accuracy calculation
                progress.append(accuracy_sequence(model, testloader))
    return progress

def our_approach_partial_participation(clients_data, testloader, batch, model_type, lr=0.0005, cluster="no", do_print="yes", do_reg="no", do_pca="no", p=1.0):
    """
    SP-CACW framework extended for Partial Participation using an unbiased estimator.
    p: The uniform probability (0, 1] that auxiliary clients participate in the round.
    Target client (Client 0) ALWAYS participates (p_0 = 1.0).
    """
    learn = 5
    progress = []

    grad_avg = None

    if model_type == "fully":
        model = SimpleNN()
        model.to(device)
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=decay
        )
        dataset = "mnist"
        x1, y1 = testset[0]
        x = x1.unsqueeze(0).to(device)
        y = (torch.tensor([y1]) if not torch.is_tensor(y1) else y1.unsqueeze(0)).to(device)

    elif model_type == "conv":
        model = VGG16_CIFAR10(num_classes=100)
        model.to(device)
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=decay
        )
        x = torch.zeros(1, 3, 32, 32).to(device)
        y = torch.zeros(1, dtype=torch.long).to(device)
        dataset = "cifar"

    elif model_type == "lstm" or model_type == "shakespeare":
        model = CharLSTM(vocab_size=VOCAB_SIZE)
        model.to(device)
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=decay
        )
        x = torch.zeros(1, 50, dtype=torch.long).to(device)
        y = torch.zeros(1, 50, dtype=torch.long).to(device)
        dataset = "lstm"

    criterion = nn.CrossEntropyLoss()
    num_clients = len(clients_data)

    # Initialize FTRL weights
    weights = torch.zeros(num_clients) / num_clients
    weights[0] = 1
    weights = weights.to(device).to(torch.float32)

    batch_size = batch
    counter = 0
    help_batches = math.ceil(len(clients_data[0][0]) / batch_size)
    lambda_val = 0.01
    bias_buffer = None

    D = 0.05
    bias = None
    var = None
    regularization = 0

    for epoch_idx in range(itr_num):
        clients_data = shuffle_clients_data(clients_data)

        # =================================================================
        # NEW: Partial Participation Client Sampling (Bernoulli Indicator Z)
        # =================================================================
        # Draw z_{i,t} ~ Bernoulli(p) for all auxiliary clients
        z_it = (torch.rand(num_clients) < p).float().to(device)

        # CRITICAL FIX: The Target Client (0) must always participate
        z_it[0] = 1.0

        total_grads_sum = None
        sum_first_client = 0.0

        start = 0

        c0_count = 0
        c0_mean = None
        c0_M2 = None

        for batch_idx in range(help_batches):
            end = min(start + batch_size, len(clients_data[0][0]))
            if (end - start) <= 1:
                start = end
                continue

            current_batch = get_batch(clients_data, start, end)

            # Compute True Gradients (Simulation environment captures all)
            grads_list_of_lists = get_grad_no_var(current_batch, model, criterion, optimizer)

            # =================================================================
            # NEW: Unbiased Gradient Estimator Application
            # =================================================================
            scaled_grads_list = []
            for i, g_c in enumerate(grads_list_of_lists):
                # If client i is selected (z_{i,t} == 1)
                if z_it[i].item() == 1.0 and g_c is not None:
                    # Unbiased scaling: (g_t / p_i)
                    # Because Client 0 is ALWAYS selected, its probability is 1.0
                    client_p = 1.0 if i == 0 else p
                    scaled_grads_list.append([g / client_p for g in g_c])
                else:
                    # If not selected, their gradient contribution is exactly 0
                    scaled_grads_list.append(None)

            # Step the global model using the scaled, unbiased estimators
            step_no_var(scaled_grads_list, weights, model, optimizer)

            # --- PREPARE GRADIENTS FOR FTRL META-OPTIMIZER ---
            grads_flat = []
            dummy_vec = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
            zero_vec = torch.zeros_like(dummy_vec)

            for g in scaled_grads_list:
                if g is None:
                    grads_flat.append(zero_vec)
                else:
                    grads_flat.append(torch.nn.utils.parameters_to_vector(g))

            grads_tensor = torch.stack(grads_flat)

            counter += 1
            bias_buffer = update_bias(grads_tensor, counter, bias_buffer)

            if total_grads_sum is None:
                total_grads_sum = grads_tensor
            else:
                total_grads_sum.add_(grads_tensor)

            sum_first_client += grads_tensor[0]

            # =================================================================
            # NEW: Safely Update Variance only if Reference Client Participated
            # (Note: z_it[0] is now guaranteed to be 1.0, but we leave the check for safety)
            # =================================================================
            if z_it[0].item() == 1.0:
                c0_grad = grads_tensor[0].detach()
                c0_count += 1

                if c0_mean is None:
                    c0_mean = c0_grad.clone()
                    c0_M2 = torch.zeros_like(c0_grad)
                else:
                    delta = c0_grad - c0_mean
                    c0_mean += delta / c0_count
                    delta2 = c0_grad - c0_mean
                    c0_M2 += delta * delta2

            start = end

        # --- AFTER THE LOOP ---
        if c0_count > 1:
            var = (c0_M2.sum() / (c0_count - 1)).item()
        else:
            var = 0.0

        L = estimate_L(model, criterion, x, y)

        if do_reg == "yes":
            regularization = var
        else:
            regularization = 0

        if epoch_idx >= 0:
            final_avg_grads = total_grads_sum / help_batches

            # Update weights using FTRL logic
            weights = update_weights(weights, bias_buffer, var, final_avg_grads, L, lr, lambda_val, D, model_type, regularization, dataset, "no", do_pca)

            if cluster == "yes":
                divide = 1 / num_clients
                weights = torch.where(weights >= divide, weights, torch.zeros_like(weights))
                n = (weights > 0).sum()
                weights = torch.where(weights > 0, torch.ones_like(weights) / n, weights)

            if do_print == "yes":
                print(weights)

        # --- EVALUATION ---
        if epoch_idx % jump == 0 and epoch_idx != 0:
            if model_type == "fully":
                progress.append(accuracy(model, testloader, True))
            elif model_type == "conv":
                progress.append(accuracy(model, testloader, False))
            elif model_type == "lstm" or model_type == "shakespeare":
                progress.append(accuracy_sequence(model, testloader))

    return progress



def accuracy_sequence(model, testloader):
    model.eval()
    correct = 0
    total = 0

    # Get the device the model is currently on (CPU or Cuda)
    device = next(model.parameters()).device

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)

            # 1. Initialize a fresh hidden state for this batch
            hidden = model.init_hidden(batch_size)

            # 2. Pass BOTH the inputs and the hidden state to the model
            # Note: LSTMs usually return the output AND the new hidden state
            outputs, hidden = model(inputs, hidden)

            # Reshape for accuracy calculation
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    return correct / total*100

def update_weights(weights, bias, var, final_avg_grads, L, lr, lambda_val, D, model_type, reg, dataset, do_print="yes",do_pca="yes"):
    """
    Performs PCA and solves for weights.
    bias & final_avg_grads are [N, P] tensors.
    """
    device = weights.device
    n_components = bias.shape[0] # Num clients
    if(do_pca == "yes"):
        # 1. PCA on Bias (Fast GPU PCA)
        # bias is already [N, P], no flattening needed
        U_b, S_b, _ = torch.pca_lowrank(bias, q=n_components, center=False, niter=5)
        bias_pca = torch.mm(U_b, torch.diag(S_b))

        # 2. PCA on Gradients
        U_g, S_g, _ = torch.pca_lowrank(final_avg_grads, q=n_components, center=False, niter=5)
        grads_pca = torch.mm(U_g, torch.diag(S_g))

        # 3. Find Weights
        weights = find_weights_all(bias_pca, var, grads_pca, D, lr, weights, L, model_type, reg, dataset, lambda_val)
        if do_print == "yes":
            print_bias_pca = torch.norm(bias_pca, dim=1)**2
            print_grad_pca = torch.norm(grads_pca, dim=1)**2
            print("Bias norms:", print_bias_pca.cpu().numpy())
            print("Grad norms:", print_grad_pca.cpu().numpy())
            print("Weights:", weights.cpu().numpy())
    else:
        weights = find_weights_all(bias, var, final_avg_grads, D, lr, weights, L, model_type, reg, dataset, lambda_val)


    return weights





















def hessian_vector_product(loss, params, v):
    """
    Compute H v using Pearlmutter's trick (forward-over-reverse).
    """
    grad = torch.autograd.grad(loss, params, create_graph=True)
    flat_grad = torch.cat([g.reshape(-1) for g in grad])

    Hv = torch.autograd.grad(flat_grad, params, grad_outputs=v, retain_graph=True)
    Hv_flat = torch.cat([h.reshape(-1) for h in Hv])

    return Hv_flat

def estimate_L(model, criterion, x, y, iters=20):
    """
    Deterministic & SAFE estimate of smoothness constant L.
    - No randomness
    - No model state change
    - No gradient change
    """

    # -----------------------
    # Save state
    # -----------------------
    was_training = model.training

    # cuDNN LSTMs cannot compute backward passes in eval() mode.
    if isinstance(model, CharLSTM):
        model.train()
    else:
        model.eval()

    # Save gradients
    saved_grads = [p.grad.clone() if p.grad is not None else None
                   for p in model.parameters()]

    params = [p for p in model.parameters() if p.requires_grad]

    # -----------------------
    # Detach inputs
    # -----------------------
    x = x.detach()
    y = y.detach()

    # =====================================================================
    # FIX: Temporarily disable cuDNN to allow double-backprop on the LSTM
    # =====================================================================
    with torch.backends.cudnn.flags(enabled=False):

        # -----------------------
        # Forward pass (needs grad)
        # -----------------------
        with torch.enable_grad():
            if hasattr(model, 'init_hidden'):
                # 1. Force integer inputs and pass 'None' for the hidden state
                out, _ = model(x.long(), None)

                # 2. Flatten the 3D output to 2D
                flat_out = out.view(-1, out.size(-1))

                # 3. Calculate loss
                loss = criterion(flat_out, y.long().view(-1))
            else:
                out = model(x)
                loss = criterion(out, y)

        # -----------------------
        # FIXED vector v (no random)
        # -----------------------
        with torch.no_grad():
            n = sum(p.numel() for p in params)
            v = torch.ones(n, device=x.device)   # always the same
            v = v / (v.norm() + 1e-12)

        # -----------------------
        # Power iteration
        # -----------------------
        for _ in range(iters):
            # This is where the double-backward happens
            Hv = hessian_vector_product(loss, params, v)

            with torch.no_grad():
                v = Hv / (Hv.norm() + 1e-12)

        # -----------------------
        # Rayleigh quotient
        # -----------------------
        with torch.no_grad():
            lambda_max = torch.dot(v, Hv).item()

    # =====================================================================
    # cuDNN automatically turns back ON as soon as we exit that 'with' block
    # =====================================================================

    # -----------------------
    # Restore state
    # -----------------------
    model.zero_grad(set_to_none=True)

    for p, g in zip(model.parameters(), saved_grads):
        p.grad = g

    # Safely restore the exact mode the model was in before we started
    if was_training:
        model.train()
    else:
        model.eval()

    return lambda_max

def get_batch(clients_data, start, end):
    batch = []
    for client_images, client_labels in clients_data:
        # Ensure we don't go out of bounds
        b_start = start % len(client_images)
        b_end = min(b_start + (end - start), len(client_images))
        batch.append([client_images[b_start:b_end], client_labels[b_start:b_end]])
    return batch


def shuffle_clients_data(clients_data):
    """
    Shuffle each client's dataset independently.
    clients_data: list of tuples [(images, labels), ...]
    Returns a new list with shuffled data for each client.
    """
    shuffled_clients = []
    for images, labels in clients_data:
        num_samples = images.shape[0]
        # Generate a random permutation of indices
        perm = torch.randperm(num_samples)
        shuffled_images = images[perm]
        shuffled_labels = labels[perm]
        shuffled_clients.append((shuffled_images, shuffled_labels))
    return shuffled_clients


def get_grad_no_var(batch, model, criterion, optimizer):
    grads = []
    model.train()
    for client_images, client_labels in batch:
        optimizer.zero_grad()
        # Skip empty clients
        if client_images.shape[0] <= 2:
            grads.append(None)
            continue

        # Convert to float + feed whole batch at once
        x = client_images.float()                    # shape (B, 784)
        y = client_labels.long()                     # shape (B,)
        # Automatically check if the model is our text sequence model
        if isinstance(model, CharLSTM):
            # 1. Force the inputs to be integers for the Embedding layer
            x = x.long()

            # 2. Pass 'None' for the hidden state to initialize it with zeros
            outputs, _ = model(x, None)

            # 3. Flatten the outputs from [batch, seq_len, vocab] to [batch * seq_len, vocab]
            flat_outputs = outputs.view(-1, outputs.size(-1))

            # 4. Calculate loss with both flattened tensors
            loss = criterion(flat_outputs, y.view(-1))
        else:
            # Standard image forward pass and loss (for CIFAR/MNIST)
            outputs = model(x)
            loss = criterion(outputs, y)
        # Compute loss on full batch
        # Backprop ONCE
        loss.backward()

        # Save gradient for this client
        client_grad = [p.grad.clone() for p in model.parameters()]
        grads.append(client_grad)

    return grads


def step_no_var(grads, weights, model, optimizer):
    # Initialize aggregated gradient
    aggregated_grad = [torch.zeros_like(p) for p in model.parameters()]

    # Weighted sum of per-client gradients
    for c_idx, g_c in enumerate(grads):
        if g_c is None:
            continue
        w = weights[c_idx]
        for p_idx, g in enumerate(g_c):
            aggregated_grad[p_idx] += w * g

    # Assign the aggregated gradient to model parameters
    for p_idx, param in enumerate(model.parameters()):
        param.grad = aggregated_grad[p_idx]

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    optimizer.step()











def ifca_first_client(clients_data, testloader, num_clusters, batch_size, model_type, lr):
    """
    IFCA where accuracy is evaluated only on the first client.
    clients_data: list of (images, labels) per client
    num_clusters: K
    model_type: String indicating the model (e.g., 'mnist' or 'cifar')
    batch_size: local batch size
    itr_num: number of global iterations
    """

    num_clients = len(clients_data)

    if model_type == "mnist":
        cluster_models = [SimpleNN() for _ in range(num_clusters)]
        cluster_optimizers = [
            optim.SGD(
                m.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=decay
            )
            for m in cluster_models
        ]
    elif model_type.lower() == "lstm" or model_type.lower() == "shakespeare":
        # Initialize the LSTM cluster models using the global VOCAB_SIZE
        cluster_models = [CharLSTM(vocab_size=VOCAB_SIZE) for _ in range(num_clusters)]
        cluster_optimizers = [
            optim.SGD(
                m.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=decay
            )
            for m in cluster_models
        ]
    else:
        cluster_models = [VGG16_CIFAR10(num_classes=100) for _ in range(num_clusters)]
        cluster_optimizers = [
            optim.SGD(
                m.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=decay
            )
            for m in cluster_models
        ]

    criterion = nn.CrossEntropyLoss()
    client_assignments = torch.randint(0, num_clusters, (num_clients,))  # random init
    progress = []

    # number of batches per client
    num_batches = math.ceil(len(clients_data[0][0]) / batch_size)

    for epoch in range(itr_num):
        clients_data = shuffle_clients_data(clients_data)
        start = 0
        for batch_idx in range(num_batches):
            end = min(start + batch_size, len(clients_data[0][0]))
            batch = get_batch(clients_data, start, end)

            # Step for each cluster
            for k in range(num_clusters):
                cluster_clients_idx = [i for i, c in enumerate(client_assignments) if c == k]
                if len(cluster_clients_idx) == 0:
                    continue

                cluster_batch = [batch[i] for i in cluster_clients_idx]
                grads = get_grad_no_var(cluster_batch, cluster_models[k], criterion, cluster_optimizers[k])
                weights = torch.ones(len(cluster_clients_idx)) / len(cluster_clients_idx)
                step_no_var(grads, weights, cluster_models[k], cluster_optimizers[k])

            # Reassign clients to clusters based on lowest loss
            for client_idx, (images, labels) in enumerate(batch):
                losses = []
                for k in range(num_clusters):
                    with torch.no_grad():
                        outputs = cluster_models[k](images.float())
                        loss = criterion(outputs, labels.long())
                        losses.append(loss.item())
                client_assignments[client_idx] = int(torch.argmin(torch.tensor(losses)))

            start = end

        # Evaluate accuracy on first client only
        if epoch % jump == 0 and epoch > 0:
            first_client_loader = testloader  # assuming testloader has only first client data
            accs = []
            for k in range(num_clusters):
                if model_type == "mnist":
                    acc = accuracy(cluster_models[k], first_client_loader, True)
                else:
                    acc = accuracy(cluster_models[k], first_client_loader, False)
                accs.append(acc)

            progress.append(max(accs))  # take best cluster for first client


    return progress


def slice_clients(clients_data, start, end):
    batches = []
    for images, labels in clients_data:
        batches.append((images[start:end], labels[start:end]))
    return batches


def shuffle_clients(clients_data):
    shuffled = []
    for images, labels in clients_data:
        perm = torch.randperm(images.size(0))
        shuffled.append((images[perm], labels[perm]))
    return shuffled


EPS = 1e-12

def accuracy(model, testloader, flatten_for_mlp=False, device=None):
    model.eval()  # evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            if device is not None:
                images = images.to(device)
                labels = labels.to(device)

            # Only flatten if using MLP
            if flatten_for_mlp:
                images = images.view(images.size(0), -1)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

def compute_cloud_models(models, sigma=1.0, beta=0.5, target_idx=None):
    """
    Paper-faithful HeurFedAMP (Section 6, Eq. 8)

    xi_{i,j} = exp(sigma * cos(w_i, w_j)) / sum_{k != i} exp(sigma * cos(w_i, w_k))
    xi_{i,i} = beta

    u_i = sum_j xi_{i,j} * w_j
    """
    device = next(models[0].parameters()).device

    with torch.no_grad():
        # Flatten models
        W = torch.stack([
            torch.nn.utils.parameters_to_vector(m.parameters()).to(device)
            for m in models
        ])  # (N, D)

        # Cosine similarity
        W_norm = F.normalize(W, p=2, dim=1)
        cosine_sim = torch.mm(W_norm, W_norm.t())  # [-1, 1]

        # exp(sigma * cosine)
        scores = torch.exp(sigma * cosine_sim)

        # Remove self-interaction
        N = W.size(0)
        mask = torch.eye(N, device=device, dtype=torch.bool)
        scores.masked_fill_(mask, 0.0)

        # Normalize over j != i
        row_sums = scores.sum(dim=1, keepdim=True)
        xi = scores / torch.clamp(row_sums, min=EPS)

        # Add self-weight (paper-consistent)
        xi = (1.0 - beta) * xi
        xi[mask] = beta

        # Cloud anchors
        if target_idx is None:
            u = torch.mm(xi, W)        # (N, D)
            return u
        else:
            u_target = torch.mv(xi[target_idx], W)
            return u_target


def train_local_fedamp(batch, model, u_flat, loss_fn, optimizer, lam, flatten_for_mlp=False):
    """
    Train one batch for a *single* local model with FedAMP regularizer.
    u_flat should be a 1D tensor (detached) on the same device as the model parameters.
    Returns task_loss (float).
    """
    images, labels = batch
    if images.size(0) <= 1:
        return 0.0

    device = next(model.parameters()).device
    if hasattr(model, 'lstm'):
        model.lstm.flatten_parameters()
    optimizer.zero_grad()

    # Forward pass handling
    if hasattr(model, 'init_hidden'):
        images = images.to(device).long()
        labels = labels.to(device).long().view(-1)

        # Properly initialize the hidden state
        hidden = model.init_hidden(images.size(0))
        outputs, _ = model(images, hidden)

        # FLATTEN OUTPUTS FOR LSTM (This absolutely fixes the 32 vs 1600 crash)
        flat_outputs = outputs.view(-1, outputs.size(-1))
        task_loss = loss_fn(flat_outputs, labels)
    else:
        images = images.to(device).float()
        labels = labels.to(device)
        if flatten_for_mlp:
            images = images.view(images.size(0), -1)

        outputs = model(images)
        task_loss = loss_fn(outputs, labels)

    # current flattened parameters (differentiable)
    curr_flat = torch.nn.utils.parameters_to_vector(model.parameters())

    # ensure u_flat is on same device/dtype and detached (not a learnable variable)
    u_flat = u_flat.to(curr_flat.device).detach()

    # regularization: (lambda/2) * ||w - u||^2
    reg_loss = (lam / 2.0) * torch.sum((curr_flat - u_flat) ** 2)

    total_loss = task_loss + reg_loss
    total_loss.backward()

    # optional gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

    optimizer.step()
    return task_loss.item()


def fedamp_train(clients_data, testloader, batch_size, model_type, lr,
                 lam=0.1, sigma=0.5, target_only=False, target_idx=0):
    """
    If target_only=True, only client `target_idx` will be trained locally each round;
    other clients' models are used as contributors for computing the cloud anchor.
    """
    num_clients = len(clients_data)

    # create models on the chosen device
    if model_type == "mnist":
        models = [SimpleNN().to(device) for _ in range(num_clients)]
    elif model_type == "lstm" or model_type == "shakespeare":
        # Initialize the LSTM with the global VOCAB_SIZE we found during parsing
        models = [CharLSTM(vocab_size=VOCAB_SIZE).to(device) for _ in range(num_clients)]
    else:
        models = [VGG16_CIFAR10(num_classes=100).to(device) for _ in range(num_clients)]
    global_init_state = copy.deepcopy(models[0].state_dict())
    for m in models:
        m.load_state_dict(global_init_state)
    optimizers = [torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=decay) for m in models]
    loss_fn = nn.CrossEntropyLoss()
    progress = []

    for rnd in range(itr_num):
        # compute cloud anchors
        if target_only:
            u_target = compute_cloud_models(models, sigma=sigma, target_idx=target_idx)
            # u_target is a 1D tensor on the same device as models
        else:
            cloud_u_params = compute_cloud_models(models, sigma=sigma)  # (n, D)

        # optionally shuffle clients (your helper)
        clients_data = shuffle_clients(clients_data)

        if target_only:
            # train only the target client
            i = target_idx
            models[i].train()
            batches_for_client = slice_clients_single(clients_data[i], batch_size)
            u_i = u_target.to(next(models[i].parameters()).device)

            for batch in batches_for_client:
                train_local_fedamp(
                    batch=batch,
                    model=models[i],
                    u_flat=u_i,
                    loss_fn=loss_fn,
                    optimizer=optimizers[i],
                    lam=lam,
                    flatten_for_mlp=(model_type == "mnist")
                )

            # other clients left unchanged (can be set to eval if needed)
            for j in range(num_clients):
                if j != i:
                    models[j].eval()
        else:
            # train all clients as usual
            for i in range(num_clients):
                models[i].train()
                batches_for_client = slice_clients_single(clients_data[i], batch_size)
                u_i = cloud_u_params[i].to(next(models[i].parameters()).device)

                for batch in batches_for_client:
                    train_local_fedamp(
                        batch=batch,
                        model=models[i],
                        u_flat=u_i,
                        loss_fn=loss_fn,
                        optimizer=optimizers[i],
                        lam=lam,
                        flatten_for_mlp=(model_type == "mnist")
                    )

        # evaluation (you evaluated client 0 before; keep that behavior)
        if rnd > 0 and rnd % jump == 0:
            if model_type == "mnist":
                acc = accuracy(models[0], testloader, True)
            elif model_type == "cifar":
                acc = accuracy(models[0], testloader, False)
            elif model_type == "shakespeare" or model_type == "lstm":
                acc = accuracy_sequence(models[0], testloader)
            progress.append(acc)
            #print(f"Round {rnd}: Accuracy (Client 0) = {acc:.2f}%  progress={progress}")

    return progress


def slice_clients_single(client_data, batch_size):
    """
    Slices one client's data into batches.
    Assumes client_data is (data_tensor, label_tensor) or similar
    """
    data, labels = client_data
    batches = []
    n_samples = len(data)
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        batches.append((data[i:end], labels[i:end]))
    return batches



def train_local1(batch, model, loss_fn, flatten_for_mlp=False):
    # SAFETY FIX: Prevent unpacking error if the batch is completely empty


    # 1. CRITICAL: Force training mode to prevent the CuDNN crash
    model.train()

    images, labels = batch
    if images.size(0) <= 2:
        return False

    # 2. Compact the LSTM weights to stop the memory warning
    if hasattr(model, 'lstm'):
        model.lstm.flatten_parameters()

    # Dynamic Forward Pass
    if hasattr(model, 'lstm'):
        # Shakespeare LSTM logic
        images = images.long()
        hidden = model.init_hidden(images.size(0))
        outputs, _ = model(images, hidden)

        # FLATTEN BOTH FOR LSTM
        flat_outputs = outputs.view(-1, outputs.size(-1))
        loss = loss_fn(flat_outputs, labels.long().view(-1))
    else:
        if flatten_for_mlp:
            images = images.float().view(images.size(0), -1)

        y = labels.long()
        outputs = model(images)
        loss = loss_fn(outputs, y)

    loss.backward()
    return True # Indicates successful gradient calculation
def train_local(batch, model, loss_fn, flatten_for_mlp=False):
    if hasattr(model, 'lstm'):
        model.train()

        images, labels = batch
        if images.size(0) <= 2:
            return False


        # 2. Compact the LSTM weights to stop the memory warning
        model.lstm.flatten_parameters()
        # Dynamic Forward Pass
        # Shakespeare LSTM logic
        images = images.long()
        hidden = model.init_hidden(images.size(0))
        outputs, _ = model(images, hidden)

        # FLATTEN BOTH FOR LSTM
        flat_outputs = outputs.view(-1, outputs.size(-1))
        loss = loss_fn(flat_outputs, labels.long().view(-1))
        loss.backward()
        return True

    else:
        images, labels = batch
        if images.size(0) <= 2:
            return

        if flatten_for_mlp:
            images = images.float().view(images.size(0), -1)

        #x = images.float().view(images)
        y = labels.long()

        outputs = model(images)
        loss = loss_fn(outputs, y)
        loss.backward()

# -------------------------------------------------
# Target-Only Federated Clustering (Client 0 Focus)
# -------------------------------------------------

def fed_cluster_algo(clients_data, testloader, batch_size, model_type, lr=0.01, num_clusters=3, radii=None):

    num_clients = len(clients_data)

    # Initialize models
    if model_type.lower() == "mnist":
        models = [SimpleNN().to(device) for _ in range(num_clients)]
        flatten_for_mlp = True
        radii = 0.25
    elif model_type.lower() == "lstm" or model_type.lower() == "shakespeare":
        models = [CharLSTM(vocab_size=VOCAB_SIZE).to(device) for _ in range(num_clients)]
        flatten_for_mlp = False
        radii = 1
    else:
        models = [VGG16_CIFAR10(num_classes=100).to(device) for _ in range(num_clients)]
        flatten_for_mlp = False
        radii = 1

    loss_fn = nn.CrossEntropyLoss()

    dummy_vec = torch.nn.utils.parameters_to_vector(models[0].parameters())
    velocities = [torch.zeros_like(dummy_vec) for _ in range(num_clients)]
    progress = []

    saved_weights = [copy.deepcopy(m.state_dict()) for m in models]

    for rnd in range(itr_num):
        clients_data = shuffle_clients_data(clients_data)
        steps = math.ceil(len(clients_data[0][0]) / batch_size)

        saved_weights = [copy.deepcopy(m.state_dict()) for m in models]

        for step in range(steps):
            start = step * batch_size
            end = start + batch_size
            batches = slice_clients(clients_data, start, end)

            if batches[0][0].size(0) <= 1:
                continue

            if rnd < 4:
                # === WARM-UP PHASE (CLIENT 0 ONLY) ===
                i = 0

                models[i].load_state_dict(saved_weights[i])
                models[i].zero_grad()
                train_local(batches[i], models[i], loss_fn, flatten_for_mlp)
                torch.nn.utils.clip_grad_norm_(models[i].parameters(), max_norm=max_norm)

                # MINOR FIX: Added your Phase 2 None-check here to prevent LSTM crashes

                if(model_type=="lstm"):
                    g_i = torch.cat([
                        (p.grad.view(-1) if p.grad is not None else torch.zeros_like(p).view(-1))
                        for p in models[i].parameters()
                    ]).detach()
                else:
                    g_i = torch.cat([p.grad.view(-1) for p in models[i].parameters()]).detach()
                vec = torch.nn.utils.parameters_to_vector(models[i].parameters())

                if decay > 0:
                    g_i.add_(vec, alpha=decay)

                if momentum > 0:
                    velocities[i].mul_(momentum).add_(g_i)
                    update_vec = velocities[i]
                else:
                    update_vec = g_i

                new_vec = vec - lr * update_vec
                torch.nn.utils.vector_to_parameters(new_vec, models[i].parameters())

                saved_weights[i] = copy.deepcopy(models[i].state_dict())

            else:
                # === ROBUST CLUSTERING PHASE (CLIENT 0 ONLY) ===
                gradients_for_client_0 = []

                for j in range(num_clients):
                    models[j].load_state_dict(saved_weights[0])
                    models[j].zero_grad()

                    train_local(batches[j], models[j], loss_fn, flatten_for_mlp)
                    torch.nn.utils.clip_grad_norm_(models[j].parameters(), max_norm=max_norm)

                    g_j_on_x_0 = torch.cat([
                        (p.grad.view(-1) if p.grad is not None else torch.zeros_like(p).view(-1))
                        for p in models[j].parameters()
                    ]).detach()
                    gradients_for_client_0.append(g_j_on_x_0)

                i = 0
                my_gradient = gradients_for_client_0[0]

                robust_centers = threshold_clustering(
                    client_vectors=gradients_for_client_0,
                    cluster_centers=[my_gradient],
                    radii=[radii],
                    num_iters=2
                )
                v_it = robust_centers[0]

                models[i].load_state_dict(saved_weights[i])
                vec = torch.nn.utils.parameters_to_vector(models[i].parameters())

                if decay > 0:
                    v_it.add_(vec, alpha=decay)

                if momentum > 0:
                    velocities[i].mul_(momentum).add_(v_it)
                    update_vec = velocities[i]
                else:
                    update_vec = v_it

                new_vec = vec - lr * update_vec
                torch.nn.utils.vector_to_parameters(new_vec, models[i].parameters())

                saved_weights[i] = copy.deepcopy(models[i].state_dict())

        # === SHAKESPEARE EVALUATION ADDITION ===
        if rnd > 0 and rnd % jump == 0:
            if model_type.lower() in ["lstm", "shakespeare"]:
                # Use the sequence accuracy function for text models
                acc = accuracy_sequence(models[0], testloader)
            else:
                # Standard check for image models
                flatten_for_mlp = (model_type.lower() == "mnist")
                acc = accuracy(models[0], testloader, flatten_for_mlp)

            progress.append(acc)

    return progress


def threshold_clustering(client_vectors, cluster_centers, radii, num_iters=5):
    """
    client_vectors: list of tensors [x1, x2, ..., xN] (model parameter vectors)
    cluster_centers: list of tensors [v1, ..., vK]
    radii: list of floats [tau1, ..., tauK]
    """
    N = len(client_vectors)
    K = len(cluster_centers)
    v = [v.clone() for v in cluster_centers]

    for l in range(num_iters):
        new_v = []
        for k in range(K):
            acc = torch.zeros_like(v[k])
            for i in range(N):
                dist = torch.norm(client_vectors[i] - v[k])
                if dist <= radii[k]:
                    acc += client_vectors[i]
                else:
                    acc += v[k]

            # BUG FIX: Use 'acc' instead of 'v[k]'
            # WAS: new_v.append(v[k]/N)
            new_v.append(acc / N)

        v = new_v
    return v


def evaluate_val_loss(model, val_data, batch_size, loss_fn, device, model_type):
    """
    Helper function to calculate the validation loss of a specific model.
    """
    model.eval()
    total_loss = 0.0

    # Assuming slice_clients_single is your custom batching function
    batches = slice_clients_single(val_data, batch_size)
    if not batches:
        return 0.0

    with torch.no_grad():
        for inputs, labels in batches:
            labels = labels.to(device).long()

            if model_type == "shakespeare":
                inputs = inputs.to(device).long()
            elif model_type == "mnist":
                inputs = inputs.to(device).float()
                inputs = inputs.view(inputs.size(0), -1)
            elif model_type == "cifar":
                inputs = inputs.to(device).float()

            outputs = model(inputs)

            if model_type == "shakespeare":
                outputs = outputs.view(-1, outputs.size(-1))
                labels = labels.view(-1)

            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

    return total_loss / len(batches)







def train_local_fedrep(batch, model, loss_fn, optimizer, flatten_for_mlp=False):
    """
    Standard training step used by FedRep.
    The optimizer passed in determines whether the Head or the Representation is being updated.
    """
    images, labels = batch
    if images.size(0) <= 1:
        return 0.0

    device = next(model.parameters()).device
    if hasattr(model, 'lstm'):
        model.lstm.flatten_parameters()

    optimizer.zero_grad()

    # Forward pass handling
    if hasattr(model, 'init_hidden'):
        images = images.to(device).long()
        labels = labels.to(device).long().view(-1)
        hidden = model.init_hidden(images.size(0))
        outputs, _ = model(images, hidden)

        flat_outputs = outputs.view(-1, outputs.size(-1))
        task_loss = loss_fn(flat_outputs, labels)
    else:
        images = images.to(device).float()
        labels = labels.to(device)
        if flatten_for_mlp:
            images = images.view(images.size(0), -1)

        outputs = model(images)
        task_loss = loss_fn(outputs, labels)

    task_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    optimizer.step()

    return task_loss.item()

def fedrep_train(clients_data, testloader, batch_size, model_type, lr, tau_h=1, tau_phi=1,participation_rate=1.0):
    """
    Algorithm 1: FedRep
    Alternates between training the personalized head and the shared representation.
    tau_h: Number of local epochs for the head.
    tau_phi: Number of local epochs for the representation.
    """
    num_clients = len(clients_data)

    # 1. Initialize the Global Model (to hold the shared representation)
    if model_type == "mnist" or model_type == "fully":
        global_model = SimpleNN().to(device)
        flatten_for_mlp = True
    elif model_type in ["lstm", "shakespeare"]:
        global_model = CharLSTM(vocab_size=VOCAB_SIZE).to(device)
        flatten_for_mlp = False
    else:
        global_model = VGG16_CIFAR10(num_classes=100).to(device)
        flatten_for_mlp = False

    # 2. Give every client their own model (this permanently stores their local head h_i)
    models = [copy.deepcopy(global_model) for _ in range(num_clients)]

    # 3. Setup Split Optimizers (Crucial for keeping Momentum states active across rounds!)
    opt_heads = []
    opt_phis = []

    for i in range(num_clients):
        phi_params = []
        head_params = []
        for n, p in models[i].named_parameters():
            # Identify the classifier head
            if ("fc2" in n or "fc." in n or "classifier" in n) and "fc1" not in n:
                head_params.append(p)
            else:
                phi_params.append(p)

        opt_heads.append(torch.optim.SGD(head_params, lr=lr, momentum=momentum, weight_decay=decay))
        opt_phis.append(torch.optim.SGD(phi_params, lr=lr, momentum=momentum, weight_decay=decay))

    loss_fn = nn.CrossEntropyLoss()
    progress = []

    # Filter keys to know exactly what to aggregate
    all_keys = list(global_model.state_dict().keys())
    head_keys = [k for k in all_keys if ("fc2" in k or "fc." in k or "classifier" in k) and "fc1" not in k]
    rep_keys = [k for k in all_keys if k not in head_keys]

    for rnd in range(itr_num):
        clients_data = shuffle_clients(clients_data)

        # Server's current representation (phi^t)
        global_rep_state = {k: global_model.state_dict()[k] for k in rep_keys}

        # ✅ FIX: Force the accumulator to be float32 so it can hold fractions (fixes BatchNorm integer crash)
        next_global_rep = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_rep_state.items()}
        m = max(1, int(participation_rate * num_clients))
        selected_clients = random.sample(range(1, num_clients), m - 1)
        # Always append Client 0 to ensure it is in the active group
        selected_clients.append(0)
        for i in range(len(selected_clients)):
            models[i].train()

            # --- CLIENT INITIALIZATION ---
            client_state = models[i].state_dict()
            for k in rep_keys:
                client_state[k] = global_rep_state[k].clone()
            models[i].load_state_dict(client_state)

            batches = slice_clients_single(clients_data[i], batch_size)

            # Fetch parameters for freezing/unfreezing
            phi_params = [p for n, p in models[i].named_parameters() if n not in head_keys and not any(hk in n for hk in head_keys)]
            head_params = [p for n, p in models[i].named_parameters() if any(hk in n for hk in head_keys)]

            # --- PHASE 1: UPDATE LOCAL HEAD (h_i) ---
            for p in phi_params: p.requires_grad = False
            for p in head_params: p.requires_grad = True

            for _ in range(tau_h):
                for batch in batches:
                    train_local_fedrep(batch, models[i], loss_fn, opt_heads[i], flatten_for_mlp)

            # --- PHASE 2: UPDATE REPRESENTATION (phi_i) ---
            for p in phi_params: p.requires_grad = True
            for p in head_params: p.requires_grad = False

            for _ in range(tau_phi):
                for batch in batches:
                    train_local_fedrep(batch, models[i], loss_fn, opt_phis[i], flatten_for_mlp)

            # Reset requires_grad back to True for safety
            for p in models[i].parameters(): p.requires_grad = True

            # --- SEND TO SERVER ---
            weight = 1.0 / m
            updated_client_state = models[i].state_dict()
            for k in rep_keys:
                next_global_rep[k] += weight * updated_client_state[k].float()

        # --- SERVER UPDATE ---
        new_global_state = global_model.state_dict()
        for k in rep_keys:
            # Safely cast the averaged float32 values back to integers (Long) if required by the layer
            new_global_state[k] = next_global_rep[k].to(new_global_state[k].dtype)
        global_model.load_state_dict(new_global_state)

        # --- EVALUATION ---
        if rnd > 0 and rnd % jump == 0:
            models[0].eval()
            with torch.no_grad():
                if model_type == "mnist" or model_type == "fully":
                    acc = accuracy(models[0], testloader, True)
                elif model_type == "cifar" or model_type == "conv":
                    acc = accuracy(models[0], testloader, False)
                elif model_type in ["lstm", "shakespeare"]:
                    acc = accuracy_sequence(models[0], testloader)
            progress.append(acc)

    return progress








def fedfomo(
        clients_data,
        testloader,
        batch_size,
        model_type,
        lr,
        itr_num,
        jump,
        device,
        decay,
        max_norm,
        M=5,
        val_ratio=0.2,
        epsilon=0.1,
        warmup_rounds=5,     # ⭐ NEW
        eta=0.5              # ⭐ NEW (damping)
):
    num_clients = len(clients_data)

    # Models
    if model_type == "mnist":
        models = [SimpleNN().to(device) for _ in range(num_clients)]
    elif model_type in ["lstm", "shakespeare"]:
        models = [CharLSTM(vocab_size=VOCAB_SIZE).to(device) for _ in range(num_clients)]
    elif model_type == "cifar":
        models = [VGG16_CIFAR10(num_classes=10).to(device) for _ in range(num_clients)]
    else:
        raise ValueError(f"Unknown model_type")

    optimizers = [
        torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=decay)
        for m in models
    ]

    loss_fn = nn.CrossEntropyLoss()
    progress = []

    # Affinity matrix
    P = torch.ones(num_clients, num_clients, device=device) / num_clients

    # Split data
    split_data = []
    for X, Y in clients_data:
        split_idx = int(len(X) * (1 - val_ratio))
        split_data.append(((X[:split_idx], Y[:split_idx]),
                           (X[split_idx:], Y[split_idx:])))

    for rnd in range(itr_num):

        # --- FLATTEN ---
        W_t = torch.stack([
            torch.nn.utils.parameters_to_vector(m.parameters()).detach()
            for m in models
        ])

        new_W = []

        for i in range(num_clients):

            # 🔥 WARM-UP: skip FedFomo early
            if rnd < warmup_rounds:
                new_W.append(W_t[i].clone())
                continue

            _, val_data = split_data[i]

            loss_i = evaluate_val_loss(models[i], val_data, batch_size, loss_fn, device, model_type)

            w_i = torch.zeros(num_clients, device=device)

            # epsilon-greedy sampling
            if random.random() < epsilon:
                sampled = random.sample([j for j in range(num_clients) if j != i],
                                        min(M, num_clients - 1))
            else:
                probs = P[i].clone()
                probs[i] = 0
                probs = probs / probs.sum()
                sampled = torch.multinomial(probs, M, replacement=False).tolist()

            for j in sampled:
                loss_j = evaluate_val_loss(models[j], val_data, batch_size, loss_fn, device, model_type)

                loss_diff = loss_i - loss_j
                dist = torch.norm(W_t[j] - W_t[i], p=2)

                # ignore tiny improvements (noise)
                if dist > 1e-8 and loss_diff > 1e-4:
                    w_i[j] = loss_diff / dist

            # normalize
            if w_i.sum() > 0:
                w_i = w_i / w_i.sum()

                delta = torch.matmul(w_i, (W_t - W_t[i]))

                # ⭐ DAMPED UPDATE
                agg_w = W_t[i] + eta * delta
            else:
                agg_w = W_t[i].clone()

            new_W.append(agg_w)

            # update P
            if w_i.sum() > 0:
                P[i] = 0.9 * P[i] + 0.1 * w_i

        # apply
        for i in range(num_clients):
            torch.nn.utils.vector_to_parameters(new_W[i], models[i].parameters())

        # --- LOCAL TRAINING ---
        for i in range(num_clients):
            models[i].train()
            train_data, _ = split_data[i]
            batches = slice_clients_single(train_data, batch_size)

            for inputs, labels in batches:
                if inputs.size(0) <= 1:
                    continue

                labels = labels.to(device).long()

                if model_type == "mnist":
                    inputs = inputs.to(device).float().view(inputs.size(0), -1)
                elif model_type == "cifar":
                    inputs = inputs.to(device).float()
                else:
                    inputs = inputs.to(device).long()

                optimizers[i].zero_grad()
                outputs = models[i](inputs)

                if model_type in ["lstm", "shakespeare"]:
                    outputs = outputs.view(-1, outputs.size(-1))
                    labels = labels.view(-1)

                loss = loss_fn(outputs, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(models[i].parameters(), max_norm)
                optimizers[i].step()

        # --- EVAL ---
        if rnd > 0 and rnd % jump == 0:
            with torch.no_grad():
                accs = []
                for i in range(num_clients):
                    if model_type == "mnist":
                        acc = accuracy(models[i], testloader, True)
                    elif model_type == "cifar":
                        acc = accuracy(models[i], testloader, False)
                    else:
                        acc = accuracy_sequence(models[i], testloader)
                    accs.append(acc)

            avg_acc = sum(accs) / len(accs)
            progress.append(avg_acc)


    return progress









def compute_stats(features, labels):
    mu = {}
    var = {}
    counts = {}

    for c in torch.unique(labels):
        c_val = c.item()
        mask = (labels == c)
        if mask.sum() == 0: continue

        feats = features[mask]
        mu[c_val] = feats.mean(dim=0).detach()

        # THE FIX: Standard GDA Shrinkage (0.1) to prevent Eq 11 from dividing by zero
        var[c_val] = feats.var(dim=0, unbiased=False).detach() + 0.1
        counts[c_val] = mask.sum().item()

    return mu, var, counts
# =========================
# 2. Solve classifier (Eq 11)

def solve_classifier(global_mu, global_var, num_classes):
    weights = []
    bias = []

    for c in range(num_classes):
        if c not in global_mu:
            weights.append(None)
            bias.append(None)
            continue

        mu_c = global_mu[c]
        var_c = global_var[c]

        # ✅ PUT FIX HERE
        var_c = var_c + 1.0
        w_c = mu_c / var_c
        w_c = w_c / (torch.norm(w_c) + 1e-8)
        b_c = torch.tensor(0.0).to(mu_c.device)

        weights.append(w_c)
        bias.append(b_c)

    return weights, bias

# 3. Apply classifier
# =========================
def apply_classifier(models, weights, bias, num_classes):
    for model in models:
        # 1. Identify the correct final layer
        if hasattr(model, 'fc'):
            layer = model.fc
        elif hasattr(model, 'fc2'):
            layer = model.fc2
        elif hasattr(model, 'classifier'):
            layer = model.classifier[-1]

        # 2. Safely inject the math ONLY for the classes we have
        for c in range(num_classes):
            if weights[c] is not None and bias[c] is not None:
                alpha = 0.5  # try 0.3–0.7
                layer.weight.data[c] = (
                        (1 - alpha) * layer.weight.data[c] +
                        alpha * weights[c].to(layer.weight.device)
                )

                layer.bias.data[c] = (
                        (1 - alpha) * layer.bias.data[c] +
                        alpha * bias[c].to(layer.bias.device)
                )

def apply_classifier_personalized(models, weights, bias, num_classes, client_local_counts_list):
    for i, model in enumerate(models):
        if hasattr(model, 'fc'): layer = model.fc
        elif hasattr(model, 'fc2'): layer = model.fc2
        elif hasattr(model, 'classifier'): layer = model.classifier[-1]

        # Get the classes this specific client owns
        my_classes = client_local_counts_list[i]

        for c in range(num_classes):
            # ONLY overwrite the classifier row if the client owns this digit
            if c in my_classes and weights[c] is not None:
                layer.weight.data[c] = weights[c].clone().to(layer.weight.device)
                layer.bias.data[c] = bias[c].clone().to(layer.bias.device)

def extract_local_centroids_eq17(features, labels):

    c_hat_i = {}
    n_ik = {}
    v_hat_i = {} # We still grab variance here because Line 14 / Eq 11 needs it!

    # Find which classes (digits) are actually in this client's data
    unique_classes = torch.unique(labels)

    for k in unique_classes:
        k_val = k.item()

        # 1. The indicator function 1(y = k) -> Creates a True/False mask
        mask = (labels == k)

        # 2. The Denominator: Count how many samples belong to class k
        count = mask.sum().item()

        # 3. Equation 17 (The Mean): Sum of features / count
        # PyTorch's .mean() does the exact Eq 17 math natively on the GPU
        c_hat_i[k_val] = features[mask].mean(dim=0).detach()

        # Bonus: Grab the variance for this class since Eq 11 needs it later
        if count > 1:
            v_hat_i[k_val] = features[mask].var(dim=0, unbiased=False).detach() + 0.1
        else:
            # Safety fallback if a client only has 1 picture of a digit
            v_hat_i[k_val] = torch.zeros_like(c_hat_i[k_val]) + 0.1

        n_ik[k_val] = count

    return c_hat_i, v_hat_i, n_ik

def get_global_feature_extractor_eq18(models, n_i_list):

    m = len(models)
    total_n = sum(n_i_list)

    # 1. Isolate only the \theta layers (exclude the \phi classifier layers)
    all_keys = models[0].state_dict().keys()
    theta_keys = [k for k in all_keys if not any(x in k for x in ["fc2", "fc", "classifier"])]

    # Initialize \tilde{\theta} with zeros
    theta_tilde = {key: torch.zeros_like(models[0].state_dict()[key].float()) for key in theta_keys}

    # 2. Compute the weighted sum
    for i in range(m):
        beta_i = n_i_list[i] / total_n
        theta_i = models[i].state_dict()

        for key in theta_keys:
            theta_tilde[key] += beta_i * theta_i[key].float()

    # Cast back to original dtype (just in case)
    for key in theta_keys:
        theta_tilde[key] = theta_tilde[key].to(models[0].state_dict()[key].dtype)

    return theta_tilde

def get_global_centroids_eq19_vectorized(c_hat_list, v_hat_list, n_ik_list, num_classes, feature_dim, device):
    """
    Blazing fast Vectorized Eq 19 (PyTorch Tensor Math instead of Python loops)
    """
    m = len(c_hat_list)

    # 1. Prepare empty tensors for vectorized math
    # C_hat shape: (num_clients, num_classes, feature_dim)
    C_hat = torch.zeros((m, num_classes, feature_dim), device=device)
    V_hat = torch.zeros((m, num_classes, feature_dim), device=device)

    # N_ik shape: (num_clients, num_classes, 1) - The '1' allows matrix broadcasting
    N_ik = torch.zeros((m, num_classes, 1), device=device)

    # 2. Unpack the dictionaries into the matrices
    for i in range(m):
        for k, centroid in c_hat_list[i].items():
            C_hat[i, k] = centroid
            V_hat[i, k] = v_hat_list[i][k]
            N_ik[i, k, 0] = n_ik_list[i][k]

    # 3. VECTORIZED EQUATION 19 MATH (One single line!)
    # Multiply centroids by counts, sum across all clients (dim=0), divide by total counts.
    total_n_k = N_ik.sum(dim=0) # Total samples per class across all clients

    # Avoid dividing by zero if a digit (like '9') was completely missing this round
    valid_classes = (total_n_k > 0).squeeze()

    c_global = {}
    v_global = {}

    if valid_classes.any():
        # This one line replaces all the math loops: (C_hat * N_ik)
        C_global_tensor = (C_hat * N_ik).sum(dim=0) / total_n_k
        V_global_tensor = (V_hat * N_ik).sum(dim=0) / total_n_k

        # Convert back to a dictionary only for the valid classes
        for k in range(num_classes):
            if valid_classes[k]:
                c_global[k] = C_global_tensor[k].detach()
                v_global[k] = V_global_tensor[k].detach()

    return c_global, v_global
def solve_eq11_alphas_pgd(mu_list, var_list, n_list, m_clients, num_classes, device):
    alphas = torch.ones(m_clients, m_clients, device=device) / m_clients
    alphas.requires_grad_(True)

    has_data = [len(mu_list[i]) > 0 and n_list[i] > 0 for i in range(m_clients)]

    # You can keep the learning rate standard now that we have the norm constraint
    lr = 0.01

    # 💡 NOTE: Because we are taking smaller steps, you might need more iterations
    # to reach the true minimum. 200 or 300 is safer than 100.
    for _ in range(200):
        if alphas.grad is not None:
            alphas.grad.zero_()

        total_risk = 0
        for i in range(m_clients):
            if not has_data[i]: continue

            for j in range(m_clients):
                if not has_data[j]: continue

                bias = 0
                common = 0
                for k in range(num_classes):
                    if k in mu_list[i] and k in mu_list[j]:
                        bias += torch.norm(mu_list[i][k] - mu_list[j][k]) ** 2
                        common += 1

                if common == 0 and i != j:
                    bias = 100.0

                variance = sum([(v.sum() / n_list[j]) for v in var_list[j].values()])

                total_risk += (alphas[i, j] * bias) + ((alphas[i, j]**2) * variance)

        total_risk.backward()

        # ==========================================
        # YOUR FIX: MAX NORM CLIPPING
        # ==========================================
        # This safely shrinks the gradient without changing its direction
        torch.nn.utils.clip_grad_norm_([alphas], max_norm=0.1)

        with torch.no_grad():
            # Take the safe, small step
            alphas.data -= lr * alphas.grad

            # Apply the Simplex Projection
            for i in range(m_clients):
                if not has_data[i]:
                    alphas.data[i] = 0.0
                    continue

                for j in range(m_clients):
                    if not has_data[j]:
                        alphas.data[i, j] = -1e9

                alphas.data[i] = project_simplex(alphas.data[i])

    return alphas.detach()
def client_update_line12(client_batch, model, loss_fn, c_global, lamda, device, lr, flatten_for_mlp=False):
    images, labels = client_batch
    max_norm = 1.0
    if images.size(0) <= 2: return {}, {}, {}

    # 1. Separate the parameters into \phi (classifier) and \theta (features)
    phi_params = []
    theta_params = []

    for n, p in model.named_parameters():
        if ("fc2" in n or "fc." in n or "classifier" in n) and "fc1" not in n:
            phi_params.append(p)
        else:
            theta_params.append(p)

    # 2. TWO-SPEED OPTIMIZERS
    is_lstm = hasattr(model, 'init_hidden')


    opt_phi = torch.optim.SGD(phi_params, lr=lr*2, momentum=0.5)
    opt_theta = torch.optim.SGD(theta_params, lr=lr/2, momentum=0.5, weight_decay=5e-4)

    model.train()

    # --- DATA PREP ---
    if is_lstm:
        model.lstm.flatten_parameters()
        images = images.long()
        y = labels.long().view(-1)
    else:
        if flatten_for_mlp: images = images.float().view(images.size(0), -1)
        else: images = images.float()
        y = labels.long()

    # ==========================================
    # THE INVISIBLE ADAPTER
    # ==========================================
    def forward_pass(model, x):
        if is_lstm:
            h = model.init_hidden(x.size(0))
            if hasattr(model, 'embedding'):
                embedded = model.embedding(x)
            elif hasattr(model, 'encoder'):
                embedded = model.encoder(x)
            else:
                embedded = x

            lstm_out, _ = model.lstm(embedded, h)
            feats = lstm_out.contiguous().view(-1, lstm_out.size(-1))

            if hasattr(model, 'fc'): outs = model.fc(feats)
            elif hasattr(model, 'classifier'): outs = model.classifier(feats)
            else: outs = model.fc2(feats)

            return outs, feats
        else:
            # Standard models keep using the normal approach
            return model(x, return_features=True)

    # ==========================================
    # Phase 1: Train Classifier (\phi) ONLY
    # ==========================================
    for p in theta_params: p.requires_grad = False
    for p in phi_params: p.requires_grad = True

    opt_phi.zero_grad()

    # SAFELY ROUTED THROUGH ADAPTER
    outputs, _ = forward_pass(model, images)

    loss_ce = loss_fn(outputs, y)
    loss_ce.backward()
    torch.nn.utils.clip_grad_norm_(phi_params, max_norm)
    opt_phi.step()

    # ==========================================
    # Phase 2: Train Feature Extractor (\theta) ONLY
    # ==========================================
    for p in theta_params: p.requires_grad = True
    for p in phi_params: p.requires_grad = False

    opt_theta.zero_grad()

    # SAFELY ROUTED THROUGH ADAPTER
    outputs, features = forward_pass(model, images)

    loss_ce = loss_fn(outputs, y)

    loss_align = torch.tensor(0.0, device=device)
    if len(c_global) > 0:
        for k in torch.unique(y):
            k_val = k.item()
            if k_val in c_global:
                mask = (y == k)
                feats = features[mask]
                global_p = c_global[k_val].to(device).detach().expand_as(feats)
                loss_align += F.mse_loss(feats, global_p)

    loss = loss_ce + lamda * loss_align
    loss.backward()
    torch.nn.utils.clip_grad_norm_(theta_params, max_norm)
    opt_theta.step()

    # Unfreeze all for safety before returning
    for n, p in model.named_parameters(): p.requires_grad = True

    # ==========================================
    # Extract final statistics
    # ==========================================
    with torch.no_grad():
        # SAFELY ROUTED THROUGH ADAPTER
        _, final_features = forward_pass(model, images)
        c_hat_i, v_hat_i, n_ik = extract_local_centroids_eq17(final_features, y)

    return c_hat_i, v_hat_i, n_ik

def get_personalized_classifiers_line9(models, alphas, num_clients):

    # 1. Identify which layers belong to the classifier (\phi)
    # (Excludes the feature extractor layers \theta)
    all_keys = models[0].state_dict().keys()
    phi_keys = [k for k in all_keys if any(x in k for x in ["fc2", "fc", "classifier"])]

    personalized_phis = []

    # 2. Build the customized classifier head for each client
    for i in range(num_clients):
        # Initialize an empty classifier with zeros
        custom_phi = {
            key: torch.zeros_like(models[0].state_dict()[key].float())
            for key in phi_keys
        }

        # Mix in every client's (j) classifier multiplied by their weight alpha[i, j]
        for j in range(num_clients):
            alpha_ij = alphas[i, j].item()
            phi_j = models[j].state_dict()

            for key in phi_keys:
                custom_phi[key] += alpha_ij * phi_j[key].float()

        # Cast back to original dtype to prevent PyTorch type mismatch errors
        for key in phi_keys:
            custom_phi[key] = custom_phi[key].to(models[0].state_dict()[key].dtype)

        personalized_phis.append(custom_phi)

    return personalized_phis, phi_keys

def send_to_clients_line10(models, theta_tilde, personalized_phis, phi_keys, num_clients):

    for i in range(num_clients):
        # Grab the client's current weights
        new_state = models[i].state_dict()

        # 1. Inject the Global Feature Extractor (\tilde{\theta})
        for key in theta_tilde.keys():
            new_state[key] = theta_tilde[key]

        # 2. Inject their specific Personalized Classifier (\tilde{\phi}_i)
        for key in phi_keys:
            new_state[key] = personalized_phis[i][key]

        # 3. Load the newly assembled model back into the client
        models[i].load_state_dict(new_state)

        # Safety requirement for PyTorch RNNs/LSTMs
        if hasattr(models[i], 'lstm'):
            models[i].lstm.flatten_parameters()

def fedpac_algo(clients_data, testloader, batch_size, model_type, lr, feature_dim=128, lamda=0.00):


    num_clients = len(clients_data)

    if model_type.lower() == "fully":
        models = [SimpleNN().to(device) for _ in range(num_clients)]
        flatten_for_mlp = True
        feature_dim = 128  # SimpleNN features

    elif model_type.lower() in ["lstm", "shakespeare"]:
        models = [CharLSTM(vocab_size=VOCAB_SIZE).to(device) for _ in range(num_clients)]
        flatten_for_mlp = False
        feature_dim = 256  # CharLSTM features

    else:
        models = [VGG16_CIFAR10(num_classes=100).to(device) for _ in range(num_clients)]
        flatten_for_mlp = False
        feature_dim = 512

    # Line 1: Input initialization
    # ... (Initialize your models and optimizers here as usual) ...
    num_classes = 10 if model_type.lower() == "fully" else (VOCAB_SIZE if "lstm" in model_type.lower() else 100)

    loss_fn = nn.CrossEntropyLoss()
    optimizers = [torch.optim.SGD(m.parameters(), lr=lr, momentum=momentum, weight_decay=decay) for m in models]
    progress = []

    # Line 3: Initialize w(0) and c(0)
    c_global = {}

    # Line 4: for t = 0, 1, ..., T - 1 do
    pending_weights = None
    pending_bias = None

    c_global = {}

    # ✅ [FIX 1] Removed the 'pending_weights' and 'pending_bias' variables entirely.
    # True FedPAC updates the classifier at the end of the round, not the beginning of the next.

    for rnd in range(itr_num):

        clients_data = shuffle_clients_data(clients_data)
        steps = math.ceil(len(clients_data[0][0]) / batch_size)

        # =====================
        # Collect local stats
        # =====================
        c_hat_list = [{} for _ in range(num_clients)]
        v_hat_list = [{} for _ in range(num_clients)]
        n_ik_list = [{} for _ in range(num_clients)]
        n_i_list = [0] * num_clients

        # =====================
        # LOCAL TRAINING (Lines 5-7)
        # =====================
        for step in range(steps):
            start = step * batch_size
            end = start + batch_size
            batches = slice_clients(clients_data, start, end)

            for i in range(num_clients):
                if i >= len(batches) or not batches[i] or batches[i][0].size(0) <= 1:
                    continue

                optimizers[i].zero_grad()

                c_hat_i, v_hat_i, n_ik = client_update_line12(
                    batches[i],
                    models[i],
                    loss_fn,
                    c_global,
                    lamda,
                    device,
                    lr,
                    flatten_for_mlp
                )

                torch.nn.utils.clip_grad_norm_(models[i].parameters(), max_norm)
                optimizers[i].step()
                has_nans = False
                for p in models[i].parameters():
                    if torch.isnan(p).any():
                        has_nans = True
                        break
                if has_nans:
                    print(f"CRASH: Client {i} model exploded into NaNs during Local Training (Step {step})!")
                    return progress # Emergency stop
                # save stats
                if len(c_hat_i) > 0:
                    for k in c_hat_i:
                        c_hat_list[i][k] = c_hat_i[k]
                        v_hat_list[i][k] = v_hat_i[k]
                        n_ik_list[i][k] = n_ik[k]
                        n_i_list[i] += n_ik[k]

        # =====================
        # SERVER SIDE (Lines 8-10)
        # =====================

        # Eq 18: Aggregate Global Feature Extractor (\tilde{\theta})
        theta_tilde = get_global_feature_extractor_eq18(models, n_i_list)

        # Eq 19: Aggregate Global Centroids for Feature Alignment next round
        c_global, v_global = get_global_centroids_eq19_vectorized(
            c_hat_list,
            v_hat_list,
            n_ik_list,
            num_classes,
            feature_dim,
            device
        )

        # ✅ [FIX 2] Equation 11: Solve for the Collaboration Matrix (Alphas)
        # Instead of one global classifier, we figure out how much clients should share with each other
        alphas = solve_eq11_alphas_pgd(
            c_hat_list,
            v_hat_list,
            n_i_list,
            num_clients,
            num_classes,
            device
        )

        # ✅ [FIX 3] Line 9: Create Personalized Classifiers (\tilde{\phi}_i)
        # Mix the classifier heads based on the alphas we just solved
        personalized_phis, phi_keys = get_personalized_classifiers_line9(
            models,
            alphas,
            num_clients
        )

        # ✅ [FIX 4] Line 10: Broadcast everything back to the clients
        # Replaces the old loop. Sends BOTH the global feature extractor AND the custom classifiers.
        send_to_clients_line10(
            models,
            theta_tilde,
            personalized_phis,
            phi_keys,
            num_clients
        )
        if rnd > 0 and rnd % jump == 0:
            with torch.no_grad():
                if model_type.lower() in ["lstm", "shakespeare"]:
                    acc = accuracy_sequence(models[0], testloader)
                else:
                    acc = accuracy(models[0], testloader, flatten_for_mlp)

            progress.append(acc)


    # Line 11: end for
    return progress

















def compute_feddisco_weights_eq3(d_k_list, n_k_list, a=1.0, b=0.0):
    """
    Computes p_k: The discrepancy-aware aggregation weights using EXACTLY Equation (3).
    p_k = ReLU(n_k - a*d_k + b) / Sum(ReLU(n_m - a*d_m + b))
    """
    K = len(d_k_list)
    raw_weights = torch.zeros(K)

    for k in range(K):
        n_k = n_k_list[k]
        d_k = d_k_list[k]

        # Equation 3 Numerator: ReLU(n_k - a * d_k + b)
        val = torch.tensor(n_k - (a * d_k) + b, dtype=torch.float32)
        raw_weights[k] = F.relu(val)

    # Equation 3 Denominator: Sum of all ReLUs
    # (Adding 1e-9 is a safety net to prevent dividing by zero if all weights hit 0)
    sum_weights = raw_weights.sum() + 1e-9

    # Calculate final p_k
    p_k = (raw_weights / sum_weights).tolist()

    return p_k


def compute_client_discrepancy(client_data, num_classes, target_u=None):
    """
    Computes d_k: The L2 distance between the client's label distribution
    and the target distribution (target_u).
    """
    _, labels = client_data
    total_samples = len(labels)

    if total_samples == 0:
        return 0.0, 0

    # Count all the labels for this client
    counts = torch.bincount(labels.view(-1).long(), minlength=num_classes).float()

    # Client's actual distribution (q_k)
    q_k = counts / total_samples

    # If no target_u is provided, default to a Uniform distribution
    if target_u is None:
        target_u = torch.ones(num_classes, device=q_k.device) / num_classes
    else:
        # Ensure the provided target is on the correct device
        target_u = target_u.to(q_k.device)

    # Discrepancy d_k (L2 distance)
    d_k = torch.norm(q_k - target_u, p=2).item()

    return d_k, total_samples


def feddisco_gradient_approach(clients_data, testloader, batch, model_type, lr=0.0005, a=0.5, b=0.1):

    progress = []

    # 1. Initialize Model and Optimizer
    if model_type == "fully":
        num_classes = 10
        model = SimpleNN().to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=decay)

    elif model_type == "conv":
        num_classes = 100
        model = VGG16_CIFAR10(num_classes=100).to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=decay)

    elif model_type == "lstm" or model_type == "shakespeare":
        num_classes = VOCAB_SIZE
        model = CharLSTM(vocab_size=VOCAB_SIZE).to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=decay)

    criterion = nn.CrossEntropyLoss()
    num_clients = len(clients_data)

    # =========================================================
    # FEDDISCO PHASE 1 & 2: Pre-compute Weights Once
    # =========================================================
    d_k_list = []
    raw_samples_list = []

    # Step A: Determine the Target Distribution (u)
    target_u = None
    if model_type in ["lstm", "shakespeare"]:

        global_counts = torch.zeros(num_classes, device=device)
        total_global_samples = 0

        for k in range(num_clients):
            _, labels = clients_data[k]
            counts = torch.bincount(labels.view(-1).long(), minlength=num_classes).float()
            global_counts += counts.to(device)
            total_global_samples += len(labels)

        # The true global frequency of every character
        target_u = global_counts / (total_global_samples + 1e-9)

    # Step B: Calculate discrepancy for each client against target_u
    for k in range(num_clients):
        d_k, raw_samples = compute_client_discrepancy(clients_data[k], num_classes, target_u=target_u)
        d_k_list.append(d_k)
        raw_samples_list.append(raw_samples)

    # Step C: Calculate relative sizes
    total_samples = sum(raw_samples_list)
    relative_n_k_list = [samples / (total_samples + 1e-9) for samples in raw_samples_list]

    # Step D: Get p_k from Equation 3
    p_k_list = compute_feddisco_weights_eq3(d_k_list, relative_n_k_list, a, b)


    # Convert the FedDisco list into the exact Tensor format your step_no_var expects
    weights = torch.tensor(p_k_list, dtype=torch.float32, device=device)
    # =========================================================

    batch_size = batch
    num_batches = math.ceil(len(clients_data[0][0]) / batch_size)

    # 3. Training Loop
    for epoch_idx in range(itr_num):
        clients_data = shuffle_clients_data(clients_data)
        start = 0

        for batch_idx in range(num_batches):
            end = min(start + batch_size, len(clients_data[0][0]))
            if (end - start) <= 1:
                start = end
                continue

            current_batch = get_batch(clients_data, start, end)

            # Fetch gradients for all clients based on the current global model
            grads = get_grad_no_var(current_batch, model, criterion, optimizer)

            # Take a single optimizer step using the FedDisco gradient weights!
            step_no_var(grads, weights, model, optimizer)

            start = end

        # 4. Evaluation
        if epoch_idx % jump == 0 and epoch_idx != 0:
            model.eval()
            with torch.no_grad():
                if model_type == "fully":
                    progress.append(accuracy(model, testloader, True))
                elif model_type == "conv":
                    progress.append(accuracy(model, testloader, False))
                elif model_type == "lstm" or model_type == "shakespeare":
                    progress.append(accuracy_sequence(model, testloader))
            model.train()


    return progress









def train_local_standard(batch, model, loss_fn, optimizer, flatten_for_mlp=False):
    """
    Standard SGD for the Global Model track (FedAvg).
    """
    images, labels = batch
    if images.size(0) <= 1: return 0.0

    device = next(model.parameters()).device
    if hasattr(model, 'lstm'): model.lstm.flatten_parameters()
    optimizer.zero_grad()

    if hasattr(model, 'init_hidden'):
        images = images.to(device).long()
        labels = labels.to(device).long().view(-1)
        hidden = model.init_hidden(images.size(0))
        outputs, _ = model(images, hidden)
        task_loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels)
    else:
        images = images.to(device).float()
        labels = labels.to(device)
        if flatten_for_mlp: images = images.view(images.size(0), -1)
        task_loss = loss_fn(model(images), labels)

    task_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return task_loss.item()


def train_ditto_personalized(batch, v_model, w_flat, loss_fn, optimizer, lam, flatten_for_mlp=False):
    """
    Trains the Personalized Model (v_k) pulling it toward the flattened Global Model (w_flat).
    """
    images, labels = batch
    if images.size(0) <= 1: return 0.0

    device = next(v_model.parameters()).device
    if hasattr(v_model, 'lstm'): v_model.lstm.flatten_parameters()
    optimizer.zero_grad()

    if hasattr(v_model, 'init_hidden'):
        images = images.to(device).long()
        labels = labels.to(device).long().view(-1)
        hidden = v_model.init_hidden(images.size(0))
        outputs, _ = v_model(images, hidden)
        task_loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels)
    else:
        images = images.to(device).float()
        labels = labels.to(device)
        if flatten_for_mlp: images = images.view(images.size(0), -1)
        task_loss = loss_fn(v_model(images), labels)

    # =========================================================
    # DITTO PENALTY: (lambda / 2) * ||v_k - w||^2
    # =========================================================
    v_flat = torch.nn.utils.parameters_to_vector(v_model.parameters())
    reg_loss = (lam / 2.0) * torch.sum((v_flat - w_flat) ** 2)

    total_loss = task_loss + reg_loss
    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(v_model.parameters(), max_norm=max_norm)
    optimizer.step()
    return total_loss.item()


def ditto_train(clients_data, testloader, batch_size, model_type, lr, lam=0.1):

    num_clients = len(clients_data)
    progress = []


    # 1. Initialize Models
    if model_type == "fully":
        global_model = SimpleNN().to(device)
    elif model_type in ["lstm", "shakespeare"]:
        global_model = CharLSTM(vocab_size=VOCAB_SIZE).to(device)
    else:
        global_model = VGG16_CIFAR10(num_classes=100).to(device)

    # Every client gets their own permanent personalized model
    local_v_models = [copy.deepcopy(global_model) for _ in range(num_clients)]

    # Personalized models persist, so their optimizers keep their momentum across rounds!
    v_optimizers = [torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=decay) for m in local_v_models]
    loss_fn = nn.CrossEntropyLoss()

    for rnd in range(itr_num):
        clients_data = shuffle_clients_data(clients_data)

        global_w_state = global_model.state_dict()
        next_global_state = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_w_state.items()}

        # Flatten the global model ONCE per round to save massive computation
        w_flat = torch.nn.utils.parameters_to_vector(global_model.parameters()).detach()

        for i in range(num_clients):
            batches = slice_clients_single(clients_data[i], batch_size)

            # ---------------------------------------------------------
            # TRACK 1: Train the Global Model (Standard FedAvg)
            # ---------------------------------------------------------
            temp_w_model = copy.deepcopy(global_model)
            # Fresh optimizer for the global track so momentum resets!
            temp_w_opt = torch.optim.SGD(temp_w_model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

            for batch in batches:
                train_local_standard(batch, temp_w_model, loss_fn, temp_w_opt, flatten_for_mlp=(model_type == "fully"))

            # Aggregate to the server (using simple 1/K uniform averaging)
            weight = 1.0 / num_clients
            for key in next_global_state.keys():
                next_global_state[key] += weight * temp_w_model.state_dict()[key].float()

            # ---------------------------------------------------------
            # TRACK 2: Train the Personalized Model (Ditto)
            # ---------------------------------------------------------
            local_v_models[i].train()
            for batch in batches:
                train_ditto_personalized(
                    batch=batch,
                    v_model=local_v_models[i],
                    w_flat=w_flat,
                    loss_fn=loss_fn,
                    optimizer=v_optimizers[i],
                    lam=lam,
                    flatten_for_mlp=(model_type == "fully")
                )

        # Update the global model for the next round
        for key in global_w_state.keys():
            next_global_state[key] = next_global_state[key].to(global_w_state[key].dtype)
        global_model.load_state_dict(next_global_state)

        # ---------------------------------------------------------
        # Evaluation
        # ---------------------------------------------------------
        if rnd > 0 and rnd % jump == 0:
            # We evaluate Client 0's personalized model, as Ditto focuses on personalization
            eval_model = local_v_models[0]
            eval_model.eval()

            with torch.no_grad():
                if model_type == "fully":
                    acc = accuracy(eval_model, testloader, True)
                elif model_type == "conv":
                    acc = accuracy(eval_model, testloader, False)
                elif model_type in ["lstm", "shakespeare"]:
                    acc = accuracy_sequence(eval_model, testloader)

            progress.append(acc)
    return progress





def create_matrix_uni2(n, sigma):
    # each element in a row = 10 / 10 = 1
    x = np.ones((n, 10))
    randoms = 0
    direction = 0
    randoms = np.random.randn(n, 10)
    randoms = np.random.normal(
        loc=0.0,                  # mean
        scale=np.sqrt(sigma),     # std = sqrt(variance)
        size=(n, 10)
    )
    for i in range(n):
        x[i] = x[i] + randoms[i]

    x = np.maximum(x, 0)
    x = np.minimum(x, 1)
    x = x / n
    #print(x)
    return x

def mean_and_std(arr_runs):
    A = np.array(arr_runs)        # shape = (5, T)
    mean = A.mean(axis=0)         # (T,)
    std  = A.std(axis=0)          # (T,)
    return mean, std

def create_matrix_uni(n, sigma=0.1):
    """
    Creates a (n x 10) matrix where every client has a roughly equal
    probability of getting any digit (0-9), with added Gaussian noise.

    Args:
        n (int): Number of clients
        sigma (float): Variance of the noise distribution
    """
    # 1. Create Uniform Base
    # Initialize with 1.0 so everyone has equal weight initially.
    x = np.ones((n, 10))

    # 2. Generate Noise
    # loc=0.0 means the noise is centered around the base value.
    noise = np.random.normal(loc=0.0, scale=np.sqrt(sigma), size=(n, 10))

    # 3. Add Noise to Base
    x = x + noise

    # 4. Clip Negative Values
    # Probabilities cannot be negative.
    x = np.maximum(x, 0)

    # 5. Normalize Columns (CRITICAL STEP)
    # We must ensure that for every digit (0-9), the sum of proportions
    # across all clients equals exactly 1.0 (100% of the data).
    col_sums = x.sum(axis=0)

    # Safety: prevent division by zero if random noise made a whole column 0
    col_sums[col_sums == 0] = 1.0

    # Divide every element by its column sum
    x = x / col_sums[np.newaxis, :]

    return x

def create_matrix_uni_100(n, clusters, client_per_cluster, sigma=0.1):
    # 1. Generate base noise
    # standard dev = sqrt(variance)
    noise = np.random.normal(loc=0.0, scale=np.sqrt(sigma), size=(n, 100))

    # 2. Initialize matrix
    x = np.zeros((n, 100))

    # 3. Inject signals
    pic_in_cli = 100 / clusters
    base_prob = 1 / client_per_cluster

    count = 0
    for i in range(n):
        # Determine which cluster 'count' we are in
        if (i % client_per_cluster == 0 and i != 0):
            count += 1

        # Assign base probability to the specific columns for this cluster
        for j in range(100):
            # Logic: If column j belongs to cluster 'count'
            if (count * pic_in_cli <= j) and ((count * pic_in_cli + pic_in_cli) > j):
                x[i][j] = base_prob

        # Add noise
        x[i] = x[i] + noise[i]

    # 4. Clip to valid range
    x = np.maximum(x, 0)

    # --- CRITICAL FIX: COLUMN NORMALIZATION ---
    # We must ensure the sum of probabilities for every class (column) equals 1.0.
    # Otherwise, the data splitter will crash or run out of images.
    col_sums = x.sum(axis=0)
    # Avoid division by zero
    col_sums[col_sums == 0] = 1.0
    x = x / col_sums[np.newaxis, :]

    return x

def experiment_rel_mnist(sigma=0):
    global jump
    global itr_num
    global momentum
    global decay

    print("sigma")
    print(sigma)
    momentum = 0.9
    decay = 5e-4
    lr = 0.01
    itr_num = 50
    jump =1

    print("\n\nstart experiment rel mnist")
    num_clients_per_cluster = 7
    clusters = 3
    num_clients = num_clients_per_cluster * clusters
    batch = 32
    time0, time01,time02,time03,time04, time3, time4, time5, time6 = [], [], [], [], [], [],[],[],[]
    client_distributions = create_matrix_uni(num_clients, sigma)
    clients_data = split_mnist_among_clients(num_clients, client_distributions, clusters, "rel")

    #for i in range(clusters):
    #    show_random_samples(clients_data[i*num_clients_per_cluster])


    test_copy = copy.deepcopy(client_distributions[0])
    maxx = max(test_copy)
    test_copy = test_copy / maxx
    set_seed()

    testset = create_custom_mnist_subset(test_copy)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
    #print((len(testset)))
    set_seed()

    if 1 == 1:
        model = "fully"
        set_seed()
        seeds = [random.randint(1, 50) for _ in range(5)]
        print(seeds)
        arr0 = []
        arr01 = []
        arr02 = []
        arr03 = []
        arr04=[]
        arr1 = []
        arr2 = []
        arr3 = []
        arr4 = []
        arr5 = []
        arr6 = []
        set_seed()
        arr7=[]
        arr8=[]
        arr9 = []
        for seed in seeds:
            set_seed(seed)
            arr7.append(ditto_train(clients_data,testloader,batch,model,lr,lam=0.05))
            set_seed(seed)
            arr2.append(feddisco_gradient_approach(clients_data,testloader,batch,model,lr))
            # 1. Our Approach (No cluster)
            set_seed(seed)
            arr0.append(our_approach(clients_data, testloader, batch, model, lr, do_print="no"))
            # print(f"arr01: {arr01}") # Added this in case you want to print it too
            # 2. Our Approach pca no reg
            # 2. Our Approach reg no pca
            set_seed(seed)
            start = time.time()
            arr03.append(our_approach(clients_data, testloader, batch, model, lr, do_print="no",do_reg="yes"))

            set_seed(seed)
            arr02.append(our_approach(clients_data, testloader, batch, model, lr, do_print="no",do_pca="yes"))
            time03.append(time.time() - start)
            start = time.time()
            arr3.append(only_my_data_approach(clients_data, testloader, batch, model, lr))
            time3.append(time.time() - start)
            # 4. Oracle Approach
            set_seed(seed)

            arr4.append(oracle_approach(clients_data,testloader,3,batch,"fully",lr))
            # 5. Fed Cluster Algo
            set_seed(seed)
            start = time.time()
            arr5.append(fed_cluster_algo(clients_data, testloader, batch, "mnist", lr, clusters))
            time5.append(time.time() - start)


            # 6. FedAmp Train
            set_seed(seed)
            start = time.time()
            arr6.append(fedamp_train(clients_data, testloader, batch, "mnist", lr))
            time6.append(time.time() - start)





    print(f"\nsigma={sigma}")
    print(f"\narr9={arr9}")
    print(f"\narr0={arr8}")
    print(f"\narr0={arr0}")
    print(f"\narr01={arr01}")
    print(f"\narr02={arr02}")
    print(f"\narr03={arr03}")
    print(f"\narr04={arr04}")
    print(f"\narr01={arr01}")
    print(f"\narr1={arr1}")
    print(f"\narr2={arr2}")
    print(f"\narr3={arr3}")
    print(f"\narr4={arr4}")
    print(f"\narr5={arr5}")
    print(f"\narr6={arr6}")
    print(f"\narr2={arr2}")
    print(f"\narr7={arr7}")

def experiment_rot_mnist(sigma=0):
    # --- 0. SETUP DEVICE ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    global jump
    global itr_num
    global momentum
    global decay

    itr_num=50
    momentum = 0.9
    decay = 1e-4
    lr = 0.01


    num_clients_per_cluster = 1
    clusters = 89
    num_clients = num_clients_per_cluster * clusters
    batch = 32

    client_distributions = create_matrix_uni(num_clients, sigma)

    # Generate data on CPU first
    clients_data_cpu = split_mnist_among_clients(num_clients, client_distributions, clusters, "rot", degre=4)

    # --- 1. OPTIMIZATION: Move all clients' data to GPU immediately ---
    clients_data = [] # We will overwrite the variable with the GPU version


    for imgs, lbls in clients_data_cpu:
        # Move images and labels to GPU
        # Ensure images are Float and labels are Long (integers)
        imgs_gpu = imgs.to(device).float()
        lbls_gpu = lbls.to(device).long()

        clients_data.append((imgs_gpu, lbls_gpu))

    # Validation print (adapted to work with GPU tensors)


    test_copy = copy.deepcopy(client_distributions[0])
    maxx = max(test_copy)
    test_copy = test_copy / maxx
    set_seed()

    # Create the temporary CPU dataset
    raw_testset = create_custom_mnist_subset(test_copy)

    # --- 2. OPTIMIZATION: Move Test Set to GPU (VRAM) immediately ---


    all_test_imgs = []
    all_test_lbls = []

    # Iterate once to collect everything
    for img, lbl in raw_testset:
        all_test_imgs.append(img)
        all_test_lbls.append(lbl)

    # Stack into a single batch and move to DEVICE
    x_test_gpu = torch.stack(all_test_imgs).to(device)
    # Ensure labels are a tensor before moving
    y_test_gpu = torch.tensor(all_test_lbls).long().to(device)

    # Create a TensorDataset that lives entirely on the GPU
    testset_gpu = torch.utils.data.TensorDataset(x_test_gpu, y_test_gpu)

    # Create the loader
    # CRITICAL: Set num_workers=0 because the data is already on the GPU!
    testloader = torch.utils.data.DataLoader(testset_gpu, batch_size=32, shuffle=False, num_workers=0)


    set_seed()

    if 1 == 1:
        model = "fully"
        set_seed()
        seeds = [random.randint(1, 50) for _ in range(5)]


        arr0 = []
        arr01 = []
        arr02 = []
        arr03 = []
        arr04=[]
        arr1 = []
        arr2 = []
        arr3 = []
        arr4 = []
        arr5 = []
        arr6 = []
        set_seed()
        arr7=[]
        arr8=[]

        arr9=[]
        # Run the main experiment loop
        for seed in seeds:
            set_seed(seed)
            arr7.append(ditto_train(clients_data,testloader,batch,model,lr,lam=0.05))
            set_seed(seed)
            arr2.append(feddisco_gradient_approach(clients_data,testloader,batch,model,lr))
            # 1. Our Approach (No cluster)
            set_seed(seed)
            arr0.append(our_approach(clients_data, testloader, batch, model, lr, do_print="no"))
            # print(f"arr01: {arr01}") # Added this in case you want to print it too
            # 2. Our Approach pca no reg
            # 2. Our Approach reg no pca
            set_seed(seed)
            start = time.time()
            arr03.append(our_approach(clients_data, testloader, batch, model, lr, do_print="no",do_reg="yes"))
            set_seed(seed)
            arr02.append(our_approach(clients_data, testloader, batch, model, lr, do_print="no",do_pca="yes"))
            start = time.time()
            arr3.append(only_my_data_approach(clients_data, testloader, batch, model, lr))

            set_seed(seed)
            arr6.append(fedamp_train(clients_data, testloader, batch, "mnist", lr))

            set_seed(seed)
            arr5.append(fed_cluster_algo(clients_data, testloader, batch, "mnist", lr, 7))

        # --- 3. AFTER THE SEED LOOP: Find the best Oracle by average ---
        best_avg = -1
        best_n = -1
        # 4. Assign the winning array to arr4 so it plugs right into your existing plot/print logic


    print(f"\nsigma={sigma}")
    print(f"\narr9={arr9}")
    print(f"\narr0={arr8}")
    print(f"\narr0={arr0}")
    print(f"\narr01={arr01}")
    print(f"\narr02={arr02}")
    print(f"\narr03={arr03}")
    print(f"\narr04={arr04}")
    print(f"\narr01={arr01}")
    print(f"\narr1={arr1}")
    print(f"\narr2={arr2}")
    print(f"\narr3={arr3}")
    print(f"\narr4={arr4}")
    print(f"\narr5={arr5}")
    print(f"\narr6={arr6}")
    print(f"\narr6={arr2}")
    print(f"\narr7={arr7}")

def experiment_cifar100(sigma, lr1=0.1, decay_val=5e-4):
    global jump
    global itr_num
    global momentum
    global decay
    global max_norm
    max_norm=5
    itr_num = 100
    print()
    print()
    print()
    print("start experiment cifar100 ")
    print("sigma=",sigma)
    print()
    print()
    num_clients_per_cluster = 5
    clusters = 4
    num_clients = num_clients_per_cluster * clusters
    batch = 32
    jump = 1
    momentum = 0.9
    decay = decay_val
    lr = lr1
    if 1 == 1:
        client_distributions = create_matrix_uni_100(num_clients, clusters, num_clients_per_cluster, sigma)
        clients_data_cpu = split_cifar100_among_clients(num_clients, client_distributions, clusters, "no")

        # 2. OPTIMIZATION: Move all clients' data to GPU immediately
        clients_data = [] # We will overwrite the variable with the GPU version

        print("Moving data to GPU...")
        for imgs, lbls in clients_data_cpu:
            # Move images and labels to GPU
            # Ensure images are Float and labels are Long (integers)
            imgs_gpu = imgs.to(device).float()
            lbls_gpu = lbls.to(device).long()

            clients_data.append((imgs_gpu, lbls_gpu))

        #for i in range(clusters):
        #   show_random_samples_cifar(clients_data[i*num_clients_per_cluster])

        test_copy = copy.deepcopy(client_distributions[0])
        maxx = max(test_copy)
        test_copy = test_copy / maxx
        set_seed()

        # --- START OF CHANGES ---
        # 1. Create the temporary CPU dataset
        raw_testset = create_custom_cifar100_subset(test_copy)

        # 2. Extract all data and move to GPU (VRAM) immediately
        # This assumes raw_testset returns (image_tensor, label)
        all_test_imgs = []
        all_test_lbls = []

        # Iterate once to collect everything
        for img, lbl in raw_testset:
            all_test_imgs.append(img)
            all_test_lbls.append(lbl)

        # 3. Stack into a single batch and move to DEVICE
        # Make sure 'device' is defined (e.g., device = 'cuda')
        x_test_gpu = torch.stack(all_test_imgs).to(device)
        y_test_gpu = torch.tensor(all_test_lbls).long().to(device)

        # 4. Create a TensorDataset that lives entirely on the GPU
        testset_gpu = torch.utils.data.TensorDataset(x_test_gpu, y_test_gpu)

        # 5. Create the loader
        # CRITICAL: Set num_workers=0 because the data is already on the GPU!
        testloader = torch.utils.data.DataLoader(testset_gpu, batch_size=32, shuffle=False, num_workers=0)
        # --- END OF CHANGES ---


        set_seed()
        model = "conv"
        set_seed()
        seeds = [random.randint(1, 50) for _ in range(5)]

        arr0 = []
        arr01 = []
        arr02 = []
        arr03 = []
        arr04=[]
        arr1 = []
        arr2 = []
        arr3 = []
        arr4 = []
        arr5 = []
        arr6 = []
        arr7=[]
        arr8=[]
        arr9=[]
        arr92=[]
        arr10=[]
        set_seed()
        p=0.1
        time0, time01,time02,time03,time04,time2, time3, time4, time5, time6 = [], [], [], [], [], [],[],[],[],[]
        for seed in seeds:
            set_seed(seed)
            arr02.append(our_approach(clients_data, testloader, batch, model, lr, do_print="no",do_pca="yes"))
            set_seed(seed)
            arr9.append(fedrep_train(clients_data, testloader, batch, model, lr))
            set_seed(seed)
            #arr92.append(fedrep_train(clients_data, testloader, batch, model, lr,participation_rate=p))
            set_seed(seed)
            #arr10.append(our_approach_partial_participation(clients_data,testloader, batch,model,lr,do_print="no",do_reg="yes",p=p ))
            set_seed(seed)
            arr7.append(ditto_train(clients_data,testloader,batch,model,lr,lam=0.05))

            set_seed(seed)
            arr2.append(feddisco_gradient_approach(clients_data,testloader,batch,model,lr))
            # 1. Our Approach (No cluster)
            set_seed(seed)
            arr0.append(our_approach(clients_data, testloader, batch, model, lr, do_print="no"))
            # print(f"arr01: {arr01}") # Added this in case you want to print it too
            # 2. Our Approach pca no reg
            # 2. Our Approach reg no pca
            set_seed(seed)
            start = time.time()
            arr03.append(our_approach(clients_data, testloader, batch, model, lr, do_print="no",do_reg="yes"))
            time03.append(time.time() - start)
            start = time.time()
            arr3.append(only_my_data_approach(clients_data, testloader, batch, model, lr))
            #time3.append(time.time() - start)
            # 4. Oracle Approach
            set_seed(seed)
            start = time.time()
            arr4.append(oracle_approach(clients_data, testloader, clusters, batch, model, lr))
            time4.append(time.time() - start)





            # 5. Fed Cluster Algo
            set_seed(seed)
            start = time.time()
            arr5.append(fed_cluster_algo(clients_data, testloader, batch, "cifar", lr, clusters))
            time5.append(time.time() - start)


            # 6. FedAmp Train
            set_seed(seed)
            start = time.time()
            arr6.append(fedamp_train(clients_data, testloader, batch, "cifar", lr))
            time6.append(time.time() - start)

        # Calculate and print the average times at the end
        num_seeds = len(seeds)
    print(f"\nsigma={sigma}")
    print(f"\narr2={arr2}")
    print(f"\narr7={arr7}")
    print(f"\narr0={arr0}")
    print(f"\narr01={arr01}")
    print(f"\narr02={arr02}")
    print(f"\narr03={arr03}")
    print(f"\narr04={arr04}")
    print(f"\narr01={arr01}")
    print(f"\narr1={arr1}")
    print(f"\narr2={arr2}")
    print(f"\narr3={arr3}")
    print(f"\narr4={arr4}")
    print(f"\narr5={arr5}")
    print(f"\narr6={arr6}")
    print(f"\narr8={arr8}")
    print(f"\narr9={arr9}")
    print(f"\narr92={arr92}")
    print(f"\narr10={arr10}")
    return ([arr0,arr01,arr3,arr4,arr5,arr6])
def experiment_cifar100_label_switch(sigma, lr1=0.1, decay_val=5e-4):
    global jump
    global itr_num
    global momentum
    global decay
    global max_norm
    max_norm = 5
    itr_num = 100

    print("\n\n\n")
    print("start experiment cifar100 with STRICT DISJOINT LABEL SWITCHING (FIRST 10 CLASSES)")
    print("sigma=", sigma)
    print("\n\n")

    num_clients_per_cluster = 7
    clusters = 3
    num_clients = num_clients_per_cluster * clusters
    batch = 32
    jump = 1
    momentum = 0.9
    decay = decay_val
    lr = lr1

    if 1 == 1:
        import torchvision
        import torchvision.transforms as transforms

        # 1. LOAD RAW DATA
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        ])

        print("Loading full CIFAR-100 datasets...")
        full_trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        full_testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

        # ---------------------------------------------------------
        # NEW: FILTER FOR FIRST 10 CLASSES ONLY (0 through 9)
        # ---------------------------------------------------------
        print("Filtering for the first 10 classes...")
        # Get indices of all images where the label is strictly less than 10
        filtered_train_indices = [i for i, target in enumerate(full_trainset.targets) if target < 10]

        # 2. STRICT DISJOINT PARTITIONING (No Overlapping Images)
        # Shuffle the filtered indices
        all_indices = filtered_train_indices
        random.shuffle(all_indices)

        images_per_cluster = len(all_indices) // clusters
        images_per_client = images_per_cluster // num_clients_per_cluster

        clients_data = []

        print(f"Total training images reduced to: {len(all_indices)}")
        print("Slicing data into strict disjoint chunks and moving to GPU...")

        for c in range(clusters):
            # The label shift for this specific cluster
            shift_amount = c * 10

            # Extract the exact indices for this cluster's slice of the pie
            cluster_start_idx = c * images_per_cluster

            for i in range(num_clients_per_cluster):
                client_imgs = []
                client_lbls = []

                # Extract the exact indices for this specific client within the cluster
                client_start_idx = cluster_start_idx + (i * images_per_client)
                client_end_idx = client_start_idx + images_per_client

                for idx in range(client_start_idx, client_end_idx):
                    real_idx = all_indices[idx]
                    img, lbl = full_trainset[real_idx]

                    # Apply label shift and wrap around 100 classes
                    shifted_lbl = (lbl + shift_amount) % 100

                    client_imgs.append(img)
                    client_lbls.append(shifted_lbl)

                # Stack and move to GPU
                imgs_gpu = torch.stack(client_imgs).to(device)
                lbls_gpu = torch.tensor(client_lbls).long().to(device)

                clients_data.append((imgs_gpu, lbls_gpu))

        # 3. TEST SET PREPARATION (Untouched Labels, strictly classes 0-9)
        print("Preparing Test Set on GPU (Filtered)...")
        all_test_imgs = []
        all_test_lbls = []

        for i in range(len(full_testset)):
            if full_testset.targets[i] < 10:
                img, lbl = full_testset[i]
                all_test_imgs.append(img)
                all_test_lbls.append(lbl)

        x_test_gpu = torch.stack(all_test_imgs).to(device)
        y_test_gpu = torch.tensor(all_test_lbls).long().to(device)

        testset_gpu = torch.utils.data.TensorDataset(x_test_gpu, y_test_gpu)
        testloader = torch.utils.data.DataLoader(testset_gpu, batch_size=32, shuffle=False, num_workers=0)


        set_seed()
        model = "conv"
        set_seed()
        seeds = [random.randint(1, 50) for _ in range(5)]

        arr0 = []
        arr01 = []
        arr02 = []
        arr03 = []
        arr04=[]
        arr1 = []
        arr2 = []
        arr3 = []
        arr4 = []
        arr5 = []
        arr6 = []
        arr7=[]
        arr8=[]
        arr9=[]
        arr92=[]
        arr10=[]

        set_seed()
        p = 0.1
        time0, time01, time02, time03, time04, time2, time3, time4, time5, time6 = [], [], [], [], [], [], [], [], [], []

        for seed in seeds:
            set_seed(seed)
            arr9.append(fedrep_train(clients_data, testloader, batch, model, lr))

            set_seed(seed)


            set_seed(seed)
            arr7.append(ditto_train(clients_data,testloader,batch,model,lr,lam=0.05))

            set_seed(seed)
            arr2.append(feddisco_gradient_approach(clients_data,testloader,batch,model,lr))

            # 1. Our Approach (No cluster)
            set_seed(seed)
            arr0.append(our_approach(clients_data, testloader, batch, model, lr, do_print="no"))
            set_seed(seed)
            arr02.append(our_approach(clients_data, testloader, batch, model, lr, do_print="no",do_pca="yes"))
            set_seed(seed)
            start = time.time()
            arr03.append(our_approach(clients_data, testloader, batch, model, lr, do_print="no",do_reg="yes"))
            time03.append(time.time() - start)

            start = time.time()
            arr3.append(only_my_data_approach(clients_data, testloader, batch, model, lr))
            #time3.append(time.time() - start)

            # 4. Oracle Approach
            set_seed(seed)
            start = time.time()
            arr4.append(oracle_approach(clients_data, testloader, clusters, batch, model, lr))
            time4.append(time.time() - start)

            # 5. Fed Cluster Algo
            set_seed(seed)
            start = time.time()
            arr5.append(fed_cluster_algo(clients_data, testloader, batch, "cifar", lr, clusters))
            time5.append(time.time() - start)

            # 6. FedAmp Train
            set_seed(seed)
            start = time.time()
            arr6.append(fedamp_train(clients_data, testloader, batch, "cifar", lr))
            time6.append(time.time() - start)

    print(f"\nsigma={sigma}")
    print(f"\narr2={arr2}")
    print(f"\narr7={arr7}")
    print(f"\narr0={arr0}")
    print(f"\narr01={arr01}")
    print(f"\narr02={arr02}")
    print(f"\narr03={arr03}")
    print(f"\narr04={arr04}")
    print(f"\narr1={arr1}")
    print(f"\narr2={arr2}")
    print(f"\narr3={arr3}")
    print(f"\narr4={arr4}")
    print(f"\narr5={arr5}")
    print(f"\narr6={arr6}")
    print(f"\narr8={arr8}")
    print(f"\narr9={arr9}")
    print(f"\narr92={arr92}")
    print(f"\narr10={arr10}")

    return ([arr0,arr01,arr3,arr4,arr5,arr6])
def create_matrix_uni_dynamic(num_clients, clusters, num_clients_per_cluster, sigma, num_classes, seed=42):
    rng = np.random.default_rng(seed)
    client_distributions = []
    classes_per_cluster = num_classes // clusters

    for cluster_idx in range(clusters):
        start_class = cluster_idx * classes_per_cluster
        end_class = num_classes if cluster_idx == clusters - 1 else (cluster_idx + 1) * classes_per_cluster
        num_primary_classes = end_class - start_class

        for i in range(num_clients_per_cluster):
            dist = np.zeros(num_classes)
            if sigma == 0:
                primary_weight = 1.0 / num_primary_classes
                dist[start_class:end_class] = primary_weight
            else:
                noise_weight = sigma / num_classes
                dist += noise_weight
                primary_weight = (1.0 - sigma) / num_primary_classes
                dist[start_class:end_class] += primary_weight
            client_distributions.append(dist)

    return np.array(client_distributions)


def accuracy_shakespeare(model, testloader, flatten_for_mlp=False, device=None, is_text=False):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            if device is not None:
                inputs = inputs.to(device)
                targets = targets.to(device)

            if is_text:
                outputs, _ = model(inputs, None)
                targets = targets.view(-1)
            else:
                if flatten_for_mlp:
                    inputs = inputs.view(inputs.size(0), -1)
                outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    return 100 * correct / total

# ==========================================
# 3. EXPERIMENT FUNCTION
# ==========================================

def experiment_leaf_shakespeare(num_clients,users,lr1=0.01, decay_val=1e-4):
    global jump, itr_num, momentum, decay, VOCAB_SIZE,max_norm
    max_norm = 1.0
    print()
    print()
    print()
    print(f"\n--- Starting Pure Python Shakespeare Experiment ---")
    print("num clien=",num_clients)
    print("users=",users)
    print()
    print()
    print()
    itr_num = 100 # LSTMs might need more rounds, feel free to increase to 200
    batch = 32
    jump = 1
    momentum = 0.9
    decay = decay_val
    lr = lr1

    # We load 20 clients (each client has 5 unique characters)
    num_clients = num_clients
    # Use the new pure Python loader!
    clients_data, testloader = load_federated_shakespeare_python(
        num_clients=num_clients,
        device=device,
        users_per_client=users,
        seq_length=50
    )

    model = "lstm"

    seeds = [random.randint(1, 50) for _ in range(5)]


    arr0 = []
    arr01 = []
    arr02 = []
    arr03 = []
    arr04=[]
    arr1 = []
    arr2 = []
    arr3 = []
    arr4 = []
    arr5 = []
    arr6 = []
    arr7=[]
    arr8=[]
    arr9=[]
    arr92=[]
    arr10=[]

    set_seed()
    p=0.1
    time0, time01,time02,time03,time04,time2, time3, time4, time5, time6 = [], [], [], [], [], [],[],[],[],[]
    for seed in seeds:
        print(seed)
        set_seed(seed)
        arr3.append(only_my_data_approach(clients_data, testloader, batch, model, lr))

        arr9.append(fedrep_train(clients_data, testloader, batch, model, lr))
        set_seed(seed)
        arr92.append(fedrep_train(clients_data, testloader, batch, model, lr,participation_rate=p))
        set_seed()
        arr10.append(our_approach_partial_participation(clients_data,testloader,batch,model,lr,do_print="no",do_reg="yes",p=p))
        set_seed(seed)
        arr02.append(our_approach(clients_data, testloader, batch, model, lr, do_print="no",do_pca="yes"))
        set_seed(seed)
        arr7.append(ditto_train(clients_data,testloader,batch,model,lr,lam=0.05))
        set_seed(seed)
        arr2.append(feddisco_gradient_approach(clients_data,testloader,batch,model,lr))
        # 1. Our Approach (No cluster)
        set_seed(seed)
        arr0.append(our_approach(clients_data, testloader, batch, model, lr, do_print="no"))
        # print(f"arr01: {arr01}") # Added this in case you want to print it too
        # 2. Our Approach pca no reg
        # 2. Our Approach reg no pca
        set_seed(seed)
        start = time.time()
        arr03.append(our_approach(clients_data, testloader, batch, model, lr, do_print="no",do_reg="yes"))
        time03.append(time.time() - start)
        start = time.time()
        arr3.append(only_my_data_approach(clients_data, testloader, batch, model, lr))
        time3.append(time.time() - start)
        # 4. Oracle Approach
        set_seed(seed)


        # 5. Fed Cluster Algo
        set_seed(seed)
        start = time.time()
        arr5.append(fed_cluster_algo(clients_data, testloader, batch, "lstm", lr, 1))
        time5.append(time.time() - start)


        # 6. FedAmp Train
        set_seed(seed)
        start = time.time()
        arr6.append(fedamp_train(clients_data, testloader, batch, "lstm", lr))
        time6.append(time.time() - start)
    print(f"\narr0={arr0}")
    print(f"\narr01={arr01}")
    print(f"\narr02={arr02}")
    print(f"\narr03={arr03}")
    print(f"\narr04={arr04}")
    print(f"\narr1={arr1}")
    print(f"\narr2={arr2}")
    print(f"\narr3={arr3}")
    print(f"\narr4={arr4}")
    print(f"\narr5={arr5}")
    print(f"\narr6={arr6}")
    print(f"\narr7={arr7}")
    print(f"\narr8={arr8}")
    print(f"\narr9={arr9}")
    print(f"\narr92={arr92}")
    print(f"\narr10={arr10}")
    return [arr0, arr01, arr3, arr5, arr6]

def verify_shakespeare_data(clients_data, num_clients_to_show=3, seqs_per_client=2):
    """
    Decodes and prints the first few sequences from a few clients
    to verify that the text chunking and character mapping worked.
    """
    print("\n" + "="*60)
    print(" SHAKESPEARE DATA SANITY CHECK ")
    print("="*60)

    # 1. Create a reverse dictionary (Integer -> Character)
    int2char = {i: char for char, i in VOCAB_DICT.items()}

    for i in range(min(num_clients_to_show, len(clients_data))):
        print(f"\n--- Client {i} ---")
        x_data, y_data = clients_data[i]

        if len(x_data) == 0:
            print("  [Empty Client - No data]")
            continue

        # 2. Decode and print a few sequences
        for j in range(min(seqs_per_client, len(x_data))):
            # Move tensor to CPU and convert to standard Python list
            seq_ints = x_data[j].cpu().tolist()

            # Reconstruct the string
            text_x = "".join([int2char.get(idx, '?') for idx in seq_ints])

            # Clean up explicit newlines so it prints nicely on one line
            text_x_display = text_x.replace('\n', ' ')

            print(f"  Seq {j+1}: '{text_x_display}'")

    print("\n" + "="*60 + "\n")


def verify_test_set(testloader, num_seqs_to_show=5):
    """
    Decodes and prints the first few sequences from the testloader
    to verify that it only contains the expected Client 0 data.
    """
    print("\n" + "="*60)
    print("🧪 TEST SET SANITY CHECK 🧪")
    print("="*60)

    int2char = {i: char for char, i in VOCAB_DICT.items()}

    # Grab just the first batch from the testloader
    first_batch = next(iter(testloader), None)

    if first_batch is None or len(first_batch[0]) == 0:
        print("  [Test set is empty! Check your split logic.]")
    else:
        x_batch, y_batch = first_batch

        for j in range(min(num_seqs_to_show, len(x_batch))):
            seq_ints = x_batch[j].cpu().tolist()

            # Reconstruct the string and clean up newlines
            text_x = "".join([int2char.get(idx, '?') for idx in seq_ints])
            text_x_display = text_x.replace('\n', ' ')

            print(f"  Test Seq {j+1}: '{text_x_display}'")

    print("\n" + "="*60 + "\n")
max_norm=5
itr_num = 100
jump = 1
momentum = 0.9
decay = 5e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define standard vocabulary for LEAF Shakespeare
ALL_LETTERS = string.printable
VOCAB_DICT = {letter: i for i, letter in enumerate(ALL_LETTERS)}
VOCAB_SIZE = len(ALL_LETTERS) # You will need to make sure your CharLSTM uses this size!
