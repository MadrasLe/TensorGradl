# TensorGradl: A Handcrafted Deep Learning Framework 


<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![CUDA](https://img.shields.io/badge/Backend-CUDA%20%2F%20CuPy-76B900?logo=nvidia)
![C++](https://img.shields.io/badge/Extensions-C%2B%2B-00599C?logo=c%2B%2B)
![Build](https://img.shields.io/badge/Build-From%20Scratch-orange)

</div>

##  Overview

**TensorGradl** is a lightweight, backend-agnostic Deep Learning framework built entirely from scratch. It is designed to demystify the "black box" of modern AI by implementing a fully dynamic **Autograd Engine** (Reverse-mode automatic differentiation) capable of training complex architectures like **GPT-2**.

Unlike simple toy libraries, TensorGradl supports **GPU acceleration** via CuPy and integrates **C++ kernels** for performance-critical operations, bridging the gap between educational theory and production-grade engineering.

##  Key Features

* **Dynamic Computational Graph:** Implements a DAG (Directed Acyclic Graph) for automatic gradient tracking and backpropagation (similar to PyTorch).
* **Hardware Agnostic:** Seamlessly switches between **CPU (NumPy)** and **GPU (CuPy)** execution based on environment variables.
* **Transformer-Ready:** Robust enough to handle `LayerNorm`, `Softmax`, `MultiHeadAttention`, and `GELU` activations, allowing for full GPT training.
* **C++ Extensions:** Includes a custom accelerator module demonstrating how to bind low-level C++ code to Python via `ctypes`.
* **Production Optimizations:** Features `Weight Tying` in embeddings and numerical stability tricks in Cross Entropy Loss.

##  Architecture

The framework is structured into modular components:

| Component | File | Description |
| :--- | :--- | :--- |
| **Core** | `tensor.py` | The heart of the library. Handles data storage, graph construction, and recursive backward passes. |
| **Backend** | `backend.py` | Abstract hardware layer. Dispatches operations to `numpy` or `cupy` dynamically. |
| **NN Modules** | `nn.py` | State-of-the-art layers: `Linear`, `Embedding`, `LayerNorm`, `MultiHeadAttention`, `GELU`. |
| **Optimizer** | `optim.py` | Implementation of **SGD** and **Adam** (with bias correction). |
| **Model** | `gpt.py` | A complete **GPT-2** implementation (Transformer Decoder) built using TensorGradl primitives. |
| **Accelerator** | `accelerator.py` | Bridge to compiled C++ kernels for matrix operations. |

##  Usage

### 1. The Tensor (Autograd)
TensorGradl feels just like PyTorch. It tracks operations and calculates gradients automatically.

```python
from minigrad.tensor import Tensor

# Tensors with autograd enabled
x = Tensor([2.0, 3.0])
w = Tensor([0.5, -1.0])
b = Tensor([1.0, 1.0])

# Dynamic Graph Construction
y = x * w + b
loss = y.sum()

# Backward Pass (The Magic)
loss.backward()

print(f"Result: {loss.data}")
print(f"Gradients (dL/dx): {x.grad}") 
2. Training GPT-2 from Scratch
The repository includes a script to train a GPT model on a synthetic dataset (or TinyShakespeare) to demonstrate convergence.
```

```bash
# Set PYTHONPATH to include the library
export PYTHONPATH=$PYTHONPATH:.

# Run the training loop
python train_gpt.py
```

The model will learn the sequence patterns and minimize the CrossEntropy loss, proving the correctness of the Autograd engine.


3. GPU Acceleration
To run on NVIDIA GPUs, simply set the environment variable:

```bash
export MINIGRAD_DEVICE="cuda"
python benchmark.py
```

 Performance Benchmarks
The repository includes benchmark.py and benchmark_cpp.py to evaluate the overhead of the Python autograd engine versus native C++ implementations.

Python/NumPy: Baseline.

C++ Extension: Demonstrates significant speedups for raw matrix multiplications.

CuPy (GPU): massive parallelism for large tensor operations.

 File Structure
Plaintext

```bash
TensorGradl/
├── minigrad/           # The Framework Package
│   ├── tensor.py       # Autograd Engine
│   ├── nn.py           # Neural Network Layers
│   ├── optim.py        # Optimizers (Adam/SGD)
│   ├── gpt.py          # GPT-2 Model Architecture
│   └── backend.py      # CPU/GPU Dispatcher
├── train_gpt.py        # Proof-of-Concept Training Script
├── benchmark.py        # CPU vs GPU Benchmarks
└── benchmark_cpp.py    # Python vs C++ Benchmarks
```

 License

Developed by Gabriel Yogi. Licensed under the GNU GPLv3.
