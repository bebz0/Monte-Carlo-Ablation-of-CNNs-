# Monte Carlo Ablation Study of VGG-16 vs ResNet-18

> *Empirical analysis of structural resilience in sequential vs. residual convolutional neural networks.*

---

## Overview

This project conducts a comparative **Monte Carlo ablation study** of two fundamental CNN architectures — **VGG-16** and **ResNet-18** — on the [Imagenette](https://github.com/fastai/imagenette) dataset (10-class subset of ImageNet).

Using PyTorch's Forward Hook mechanism, we randomly disable layers and residual blocks and measure the resulting accuracy degradation across 30 independent trials per ablation level. The study empirically tests whether skip connections provide genuine fault tolerance — and reveals *where* that tolerance breaks down.

---

## Key Results

| Metric | VGG-16 | ResNet-18 |
|---|---|---|
| Baseline accuracy | 99.59% | 99.21% |
| Collapse threshold | N=1/13 **(7.7%)** | N=4/8 **(50%)** |
| Max SD across trials | ~0 pp | ~26 pp |
| Mann-Whitney U (N=1) | — | U=113/125, **p=0.0024** |
| Effect size (r) | — | **+0.808** (large) |

**VGG-16** collapses to random guessing (~10%) upon disabling a single layer. Zero variance across all trials — any break in the sequential chain produces the same catastrophic outcome.

**ResNet-18** degrades gracefully until ~50% ablation, but reveals a critical variance anomaly: at N=1, accuracy spans **8.6% – 95.9%** depending on *which* block is targeted.

### The Positional Hierarchy of Downsampling Blocks

Tracing individual Monte Carlo trials to their specific targets reveals that criticality in ResNet-18 is **positional, not proportional**:

| Block | Transition | Accuracy after ablation |
|---|---|---|
| `layer4[0]` | 7×7, 256→512ch | 82.8% |
| `layer2[0]` | 28×28, 64→128ch | 50.0% |
| `layer3[0]` | 14×14, 128→256ch | **8.6%** ← structural chokepoint |

`layer3[0]` — the transition from spatial to semantic representations — is the true single point of failure in ResNet-18. This contradicts the naive assumption that depth determines criticality.

---

## Repository Structure

```
.
├── notebook/
│   └── Monte_Carlo_Ablation_Study_of_VGG-16_vs_ResNet-18.ipynb
├── scripts/
│   ├── resnet-18.py       # Monte Carlo ablation — ResNet-18
│   └── vgg-16.py          # Monte Carlo ablation — VGG-16
├── csv_results/
│   ├── resnet18_mc_30trials.csv
│   └── vgg16_mc_30trials.csv
└── figures/
    ├── ablation_results.png
    └── variance_anomaly.png
```

---

## Methodology

**Ablation mechanism:** PyTorch `register_forward_hook` replaces a layer's output tensor with `torch.zeros_like(output)`. This nullifies the layer's contribution while preserving the computational graph — no structural deletion, no dimensional mismatches.

**BatchNorm trap (ResNet-18 specific):** Hooks are placed on `block.bn2`, not `block.conv2`. Placing them on `conv2` would feed zeros into BatchNorm, producing non-zero noise ($\gamma \cdot 0 + \beta = \beta$) that corrupts the identity branch. Hooking `bn2` ensures a true zero reaches the residual addition:

$$x_{l+1} = \underbrace{\text{hook}(\text{bn2}(\cdot))}_{= \ 0} + x_l = x_l$$

**Monte Carlo design:** For each ablation level $N$, 30 independent trials sample uniformly without replacement from the target pool. Each `(level, trial)` pair has a deterministic seed `42 + level × 1000 + trial`, ensuring full reproducibility without cross-trial seed dependency.

### Experimental Setup

| Parameter | Value |
|---|---|
| Dataset | Imagenette (10-class, ImageNet subset) |
| Eval subset | 512 images, fixed across all trials |
| Batch size | 256 |
| Monte Carlo trials per level | 30 |
| Random seed | 42 |
| Model weights | ImageNet pretrained (torchvision DEFAULT) |
| Hook target — VGG-16 | `nn.Conv2d` (13 layers) |
| Hook target — ResNet-18 | `block.bn2` (8 blocks) |

---

## Installation

```bash
git clone https://github.com/yourprofile/mc-ablation-study.git
cd mc-ablation-study
pip install torch torchvision pandas numpy scipy tqdm matplotlib seaborn
```

---

## Usage

**Run ResNet-18 ablation:**
```bash
python scripts/resnet-18.py
```

**Run VGG-16 ablation:**
```bash
python scripts/vgg-16.py
```

Configure paths and parameters at the top of each script:
```python
drive_dataset_path = '/path/to/Imagenette'
NUM_TRIALS = 30
EVAL_SUBSET_SIZE = 512  # None = full val set (~3925 images)
BATCH_SIZE = 256
```

Results are saved to `csv_results/` and a sanity-check summary is printed automatically.

**Run the full analysis:**
Open `notebook/Monte_Carlo_Ablation_Study_of_VGG-16_vs_ResNet-18.ipynb` and execute all cells. The notebook loads CSVs from `csv_results/`, reproduces all figures, and runs the statistical tests.

---

## Statistical Validation

At N=1 (single block ablation), Identity vs Downsampling block trials are compared using:

- **Mann-Whitney U** (non-parametric, one-sided): U=113/125, p=0.0024
- **Effect size** (rank-biserial r): +0.808 — large effect, Identity > Downsampling in 90.4% of pairwise comparisons
- **Levene's test**: p=0.116 — not significant, attributed to small Downsampling sample (n=5)

---

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016.
- Veit, A., Wilber, M., & Belongie, S. (2016). *Residual Networks Behave Like Ensembles of Relatively Shallow Networks*. NeurIPS 2016.
- Zeiler, M. D., & Fergus, R. (2014). *Visualizing and Understanding Convolutional Networks*. ECCV 2014.

---

## Author

**Roman Kudermin**
📧 romankudermin@gmail.com
🔗 [GitHub](https://github.com/bebz0)
🔗 [LinkedIn](https://linkedin.com/in/roman-kudermin-261192363/)
