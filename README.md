# Classical SU(2) Models Match or Exceed Shallow Variational Quantum Circuits on Classical Vision Benchmarks

**Christopher P. Fulton¹, Irene Tsapara², Lawrence V. Fulton³**

¹ United States Air Force Test Pilot School, Edwards Air Force Base, CA  
² Department of Engineering, Data, and Computer Science, National University, San Diego, CA  
³ Applied Analytics, Boston College, Chestnut Hill, MA  

Correspondence: fultonl@bc.edu

---

## Overview

This repository contains all code and data pipelines for the paper *Classical SU(2) Models Match or Exceed Shallow Variational Quantum Circuits on Classical Vision Benchmarks*. The study provides a controlled empirical comparison of real-valued, quaternion-valued, and variational quantum circuit (VQC) classifiers operating on identical frozen feature representations across MNIST, FashionMNIST, and CIFAR-10.

The central finding is that quaternion networks — classical implementations of SU(2) geometry via the Hamilton product — consistently match real-valued baselines (retaining 94–96% of performance) while substantially outperforming shallow product-state and entangled variational quantum classifiers (by 2.4–6.1 percentage points) across all datasets and feature regimes. Introducing entanglement yields modest gains on simple grayscale benchmarks but degrades performance by 9.3 percentage points on CIFAR-10 when paired with high-quality ImageNet-pretrained features.

---

## Repository Structure

```
.
├── MNIST_Quaternion.ipynb          # Full pipeline for MNIST (RealNet, QuatNet, Quantum-NoEnt, Quantum-Ent)
├── FashionMNIST_Quaternion.ipynb   # Full pipeline for FashionMNIST
├── CIFAR_Quaternion.ipynb          # CIFAR-10 with shared 16-D learned bottleneck
├── CIFAR_CNN.ipynb                 # CIFAR-10 with frozen ImageNet-pretrained ResNet18 (512-D)
├── QuantEntanglement.ipynb         # Standalone Block 4: Quantum-Ent training (all datasets)
├── Diag_Test.ipynb                 # Optimization geometry diagnostics (Euclidean vs Adam vs FS/QFI)
├── quat.ipynb                      # Self-contained prototype: Real vs Quaternion vs Quantum
└── README.md
```

---

## Notebook Descriptions

### `MNIST_Quaternion.ipynb`
Full five-block experimental pipeline for MNIST:
- **Block 1 (RealNet):** Trains the real-valued MLP baseline on 3 seeds; saves the frozen preprocessor (`realnet_results.pt`) for downstream use.
- **Block 2 (QuatNet):** Loads frozen preprocessor; trains the quaternion classification head.
- **Block 3 (Quantum-NoEnt):** Loads frozen preprocessor; trains a four-qubit depth-3 VQC without entanglement using PennyLane Lightning-GPU.
- **Block 4 (Quantum-Ent):** Same circuit with CNOT ring entanglement.
- **Block 5 (Aggregation):** Loads all saved results; produces comparative accuracy tables, epoch counts, training times, and parameter counts.

### `FashionMNIST_Quaternion.ipynb`
Identical five-block pipeline applied to FashionMNIST. Mirrors the MNIST protocol exactly, with dataset-appropriate normalization statistics.

### `CIFAR_Quaternion.ipynb`
Five-block pipeline for CIFAR-10 under the shared 16-dimensional learned bottleneck regime. All four models operate on the same frozen 3072→16 feature extractor trained as part of the real-valued baseline.

### `CIFAR_CNN.ipynb`
Five-block pipeline for CIFAR-10 under frozen ImageNet-pretrained ResNet18 features (512-dimensional). The bottleneck is replaced with a frozen ResNet18 backbone; only classification head parameters are trained. Quantum circuits scale to eight qubits in this regime.

### `QuantEntanglement.ipynb`
Standalone Block 4 notebook for training the entangled quantum classifier. Can be run independently once `realnet_results.pt` is available from the corresponding dataset pipeline.

### `Diag_Test.ipynb`
Optimization geometry diagnostic experiments comparing five update rules across MNIST, FashionMNIST, and CIFAR-10:
1. Euclidean gradient descent
2. Adam
3. FS/QFI diagonal natural gradient
4. FS/QFI block-diagonal natural gradient
5. FS/QFI full natural gradient

Reports final cross-entropy loss, wall-clock time per minibatch, and cosine similarity of each update direction relative to the full Fubini–Study natural-gradient step. Results correspond to Table 1 in the paper.

### `quat.ipynb`
Self-contained prototype notebook implementing the full Real vs Quaternion vs Quantum comparison in a single script. Useful for quick replication or adaptation.

---

## Experimental Design

All experiments share the following controls:

| Control | Value |
|---|---|
| Training samples per class | 1,500 (stratified, seed 42) |
| Test samples per class | 300 |
| Total training / test | 15,000 / 3,000 |
| Random seeds | {42, 123, 456} |
| Optimizer | Adam, lr = 1e-3 |
| Loss | Cross-entropy |
| Early stopping patience | 10 epochs (classical); 20 epochs (quantum) |
| Max epochs | 100 (classical); 200 (quantum) |
| Feature representations | Frozen (shared across all heads within regime) |

**Feature regimes:**
- **16-D bottleneck:** Single linear layer + tanh trained as part of RealNet (seed 42), then frozen. Used for MNIST, FashionMNIST, and CIFAR-10.
- **512-D ResNet18:** Frozen ImageNet-pretrained ResNet18 backbone (torchvision default weights), penultimate average-pooling layer. Used for CIFAR-10 control experiment.

**Circuit architecture (quantum models):**
- Depth-3 data re-uploading with RY encoding, trainable RY+RZ rotations per layer
- 4 qubits (16-D regime), 8 qubits (512-D regime)
- Quantum-Ent adds a CNOT ring (q0→q1→q2→q3→q0) after local rotations in each layer
- Measurements: 6 expectation values (4-qubit) or 12 (8-qubit)

---

## Key Results

| Dataset | RealNet | QuatNet | Quantum-NoEnt | Quantum-Ent |
|---|---|---|---|---|
| MNIST | 93.54 ± 0.52% | **93.64 ± 0.15%** | 87.52 ± 1.02% | 88.16 ± 0.62% |
| FashionMNIST | **84.60 ± 0.12%** | 84.47 ± 0.05% | 82.03 ± 0.94% | 82.62 ± 0.52% |
| CIFAR-10 (16-D) | **40.23 ± 0.44%** | 37.92 ± 0.33% | 35.10 ± 0.71% | 34.92 ± 0.22% |
| CIFAR-10 (512-D) | **47.13 ± 0.70%** | 45.21 ± 0.38% | 41.71 ± 0.94% | 32.46 ± 3.18% |

Quaternions retain **94–96% of real-valued performance** across all datasets and feature regimes while outperforming product-state quantum circuits by **2.4–6.1 percentage points**. Entanglement degrades performance by **9.3 percentage points** on CIFAR-10 with ResNet18 features.

---

## Requirements

### Hardware
- CUDA-capable GPU required for quantum circuit simulation via PennyLane Lightning-GPU
- Experiments were run on an NVIDIA GPU with CUDA support
- Classical models (RealNet, QuatNet) run on CPU or GPU

### Software

```
Python >= 3.10
torch >= 2.0
torchvision >= 0.15
pennylane >= 0.35
pennylane-lightning[gpu]
numpy
```

Install dependencies:

```bash
pip install torch torchvision
pip install pennylane pennylane-lightning[gpu]
pip install numpy
```

For PennyLane Lightning-GPU, CUDA toolkit compatibility must match your GPU driver. See the [PennyLane Lightning installation guide](https://docs.pennylane.ai/projects/lightning/en/stable/dev/installation.html) for details.

> **Note:** NVIDIA cuQuantum is used internally by `lightning.gpu` for GPU-accelerated statevector simulation. Ensure your CUDA version is compatible.

---

## Reproducing the Results

### Step 1 — MNIST and FashionMNIST

Run all cells in `MNIST_Quaternion.ipynb` sequentially. Blocks must be executed in order (1→2→3→4→5) because each block loads the frozen preprocessor saved by Block 1.

```
Block 1 → realnet_results.pt
Block 2 → quatnet_results.pt
Block 3 → quantum_noent_results.pt
Block 4 → quantum_ent_results.pt
Block 5 → comparative tables (printed output)
```

Repeat with `FashionMNIST_Quaternion.ipynb`.

### Step 2 — CIFAR-10 (16-D bottleneck)

Run all cells in `CIFAR_Quaternion.ipynb` sequentially. Same block structure as MNIST.

### Step 3 — CIFAR-10 (ResNet18 512-D)

Run all cells in `CIFAR_CNN.ipynb` sequentially. ResNet18 weights are downloaded automatically via torchvision on first run.

### Step 4 — Optimization geometry diagnostics

Run `Diag_Test.ipynb`. This notebook re-trains RealNet (Block 1) for each dataset and then executes the five-optimizer diagnostic. Results correspond to Table 1 in the paper.

### Expected runtimes (per seed, approximate)

| Model | Dataset | Time |
|---|---|---|
| RealNet / QuatNet | MNIST, FashionMNIST | < 2 minutes |
| RealNet / QuatNet | CIFAR-10 (either regime) | < 1 minute |
| Quantum-NoEnt | MNIST | ~6.8 hours |
| Quantum-NoEnt | FashionMNIST | ~6.2 hours |
| Quantum-NoEnt | CIFAR-10 (16-D) | ~4.3 hours |
| Quantum-NoEnt | CIFAR-10 (512-D) | ~18.0 hours |
| Quantum-Ent | MNIST | ~10.1 hours |
| Quantum-Ent | CIFAR-10 (512-D) | ~18.4 hours |

Quantum runtimes reflect GPU-accelerated simulation via PennyLane `lightning.gpu`.

---

## Reproducibility

All random number generators are seeded identically across seeds {42, 123, 456}:

```python
import random, numpy as np, torch
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

PennyLane random state is also fixed where applicable. PyTorch deterministic mode is enabled (`torch.use_deterministic_algorithms(True)`).

---

## Citation

If you use this code or build on these results, please cite:

```bibtex
@article{Fulton2025SU2,
  author  = {Fulton, Christopher P. and Tsapara, Irene and Fulton, Lawrence V.},
  title   = {Classical {SU}(2) Models Match or Exceed Shallow Variational Quantum Circuits
             on Classical Vision Benchmarks},
  year    = {2025},
  note    = {Manuscript under review}
}
```

---

## License

This repository is released for research reproducibility. Please contact the corresponding author (fultonl@bc.edu) regarding reuse or derivative works.

---

## Contact

**Lawrence V. Fulton**  
Applied Analytics, Boston College  
fultonl@bc.edu  

**Christopher P. Fulton**  
United States Air Force Test Pilot School  
fultonc742@gmail.com
