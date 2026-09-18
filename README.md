![ILCI](https://img.shields.io/badge/ILCI-Local%20Cluster%20Intelligence-2563eb?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**Local Cluster Intelligence — distributed ML training and inference on commodity hardware.**

---

## 📊 Benchmarks

| Dataset | Metric | Score |
|---------|--------|-------|
| CIFAR-10 | Accuracy | 94.2% |
| CIFAR-100 | Accuracy | 74.8% |
| Tiny-ImageNet | Top-1 | 62.3% |

> Repository figure; not rerun in this documentation refresh.

![Accuracy boxplot](figs/fig_accuracy_boxplot.png)
*Figure 1: Accuracy distribution across 5 seeds.*

![Training curves](figs/fig_loss_curves.png)
*Figure 2: Training and validation loss curves.*

![Accuracy curves](figs/fig_accuracy_curves.png)
*Figure 3: Top-1 accuracy progression over epochs.*

![Summary bars](figs/fig_summary_bars.png)
*Figure 4: Summary comparison across methods.*

---

## 🏗️ Architecture

```mermaid
graph LR
    subgraph Cluster
        N1[Node 1<br/>GPU 0]
        N2[Node 2<br/>GPU 1]
        N3[Node 3<br/>GPU 2]
    end
    N1 <-->|NCCL| N2
    N2 <-->|NCCL| N3
    N3 <-->|NCCL| N1
    N1 --> R[Reduce Scatter]
    N2 --> R
    N3 --> R
    R --> O[Optimizer<br/>AllReduce]
    O --> N1
    O --> N2
    O --> N3
```

---

## ✨ Features

- **Multi-GPU data-parallel training** with NCCL backend
- **Deterministic seeding** for reproducible experiments
- **Checkpoint resume** with full state restoration
- **Per-seed logging** — every run is auditable
- **Dataset-agnostic** dataloaders — swap CIFAR/Tiny-ImageNet/Custom

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
torchrun --nproc_per_node=3 train.py --dataset cifar10 --epochs 200
```

---

## 📁 Project Structure

```
ILCI/
├── figs/                          # Result figures (4 plots)
├── paper/                         # LaTeX paper source
├── submission/                    # NeurIPS/ICML submission bundle
├── train.py                       # Distributed training entry
├── model.py                       # Model definitions
└── requirements.txt
```

---

## 📄 License

MIT © Md Sadman Bin Masud
