# Installation Guide

## Prerequisites

- **Python**: 3.8 or higher
  - Check: `python --version`
- **pip**: Latest version
  - Upgrade: `python -m pip install --upgrade pip`

## Quick Start (CPU)

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
python main.py
```

## Detailed Installation Steps

### Step 1: Create Virtual Environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Upgrade pip, setuptools, wheel

```bash
python -m pip install --upgrade pip setuptools wheel
```

### Step 3: Install PyTorch

Choose **one** of the following:

#### Option A: CPU (default, no GPU required)
```bash
pip install -r requirements.txt
```

#### Option B: NVIDIA GPU (CUDA 12.1)
```bash
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install numpy scikit-learn matplotlib
```

#### Option C: NVIDIA GPU (CUDA 11.8)
```bash
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
pip install numpy scikit-learn matplotlib
```

#### Option D: AMD GPU (ROCm)
```bash
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/rocm5.7
pip install numpy scikit-learn matplotlib
```

To find your CUDA version, run:
```bash
nvidia-smi
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Common Issues & Solutions

### Issue 1: "No module named 'torch'"
**Cause**: PyTorch not installed or wrong environment activated.

**Solution**:
```bash
# Make sure virtual environment is activated
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Reinstall PyTorch
pip install torch torchvision
```

### Issue 2: "torchvision is not compatible with torch"
**Cause**: Version mismatch between torch and torchvision.

**Solution**: Use exact versions from requirements.txt
```bash
pip install torch==2.2.2 torchvision==0.17.2
```

### Issue 3: "CUDA out of memory" or GPU errors
**Cause**: GPU-specific version mismatch or incompatible CUDA.

**Solution**:
1. Fall back to CPU: `pip install torch torchvision`
2. Or reinstall with correct CUDA version (see Step 3)

### Issue 4: Permission denied on Linux/Mac
**Cause**: Virtual environment not activated.

**Solution**:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Issue 5: "wheel" or "setuptools" errors during install
**Cause**: Outdated build tools.

**Solution**:
```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Issue 6: "No CUDA devices available" but have GPU
**Cause**: GPU-version PyTorch not installed, or wrong CUDA version.

**Solution**:
```bash
# Check NVIDIA CUDA version
nvidia-smi

# Reinstall with correct CUDA version
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121  # Replace cu121 with your version
```

## Verify Everything Works

```bash
python -c "
import torch
import torchvision
import numpy
import matplotlib
import sklearn

print('✓ All packages installed successfully!')
print(f'PyTorch: {torch.__version__}')
print(f'Torchvision: {torchvision.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

## Platform-Specific Notes

### Windows
- Use `.venv\Scripts\activate` (not `source`)
- Some packages may need Visual C++ build tools
- If issues persist, use WSL2 (Windows Subsystem for Linux)

### macOS
- May need to install Xcode command line tools: `xcode-select --install`
- ARM64 (Apple Silicon) support is built-in for recent PyTorch versions

### Linux
- Ensure Python dev headers: `sudo apt install python3-dev` (Ubuntu/Debian)
- May need CUDA toolkit: https://developer.nvidia.com/cuda-downloads

## Getting Help

- PyTorch Install Guide: https://pytorch.org/get-started/locally/
- Troubleshooting: https://pytorch.org/get-started/locally/
- GitHub Issues: Check if your problem is already reported
