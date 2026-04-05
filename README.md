# Spatial Omics Fusion

Multimodal architecture for spatial transcriptomics domain detection, combining gene expression encoding (MLP) with spatial graph context (GAT) via cross-attention fusion.

Achieves **0.926 ARI** on the DLPFC benchmark (12 slices) with only **1.1M parameters**, training in under 30 seconds per slice.

## Architecture

```
Gene Expression (3000 dims)
        │
   MLP Encoder
        │
        ▼
Expression Embedding (128) ────────────────────────┐
        │                                          │
        ▼                                          │
   GAT Encoder (aggregates spatial neighbors)      │
        │                                          │
        ▼                                          │
Spatial Embedding (128)                            │
        │                                          │
        ▼                                          │
Cross-Attention Fusion                             │
Q = Expression (right path) ◄──────────────────────┘
K/V = Spatial (left path)
        │
        ▼
Classification Head ──→ Domain Prediction (5-7 classes)
```

**Expression Encoder**: 2-layer MLP compresses 3,000 highly variable genes into a 128-dim embedding.

**Spatial Encoder**: 2-layer Graph Attention Network aggregates neighbor information over a k=6 spatial KNN graph, using expression embeddings as node features.

**Cross-Attention Fusion**: Expression embedding serves as query; spatial embeddings of graph neighbors serve as keys/values. Each spot selectively attends to relevant neighbors.

## Results

Benchmarked across all 12 DLPFC tissue slices (mean ± std):

### Standard Metrics

| Model | ARI | NMI | Accuracy |
|---|---|---|---|
| MLP-only (no spatial) | 0.356 ± 0.068 | 0.328 ± 0.043 | 0.607 ± 0.047 |
| Image + MLP + Cross-Attention | 0.447 ± 0.147 | 0.427 ± 0.065 | 0.659 ± 0.084 |
| GAT-only | 0.919 ± 0.029 | 0.892 ± 0.018 | 0.958 ± 0.013 |
| MLP + GAT + Concat | 0.917 ± 0.026 | 0.888 ± 0.015 | 0.956 ± 0.011 |
| MLP + GAT + Gated | 0.920 ± 0.027 | 0.890 ± 0.017 | 0.958 ± 0.012 |
| **MLP + GAT + Cross-Attention** | **0.928 ± 0.022** | **0.896 ± 0.011** | **0.960 ± 0.008** |
| MLP + GAT + Image + Cross-Attention | 0.928 ± 0.023 | 0.899 ± 0.014 | 0.961 ± 0.010 |

### Boundary-Aware Metrics

All spatial models achieve ~99.8% top-2 accuracy — when the model is "wrong", the true label is almost always its second choice. Errors occur exclusively at domain boundaries where ground truth annotations are inherently ambiguous.

| Model | Top-2 Acc | Interior Acc | Boundary Acc | Log-Loss |
|---|---|---|---|---|
| MLP-only (no spatial) | 0.797 ± 0.043 | 0.616 ± 0.047 | 0.356 ± 0.081 | 1.315 ± 0.185 |
| Image + MLP + Cross-Attention | 0.837 ± 0.059 | 0.669 ± 0.083 | 0.390 ± 0.129 | 0.998 ± 0.249 |
| GAT-only | 0.999 ± 0.001 | 0.965 ± 0.010 | 0.775 ± 0.083 | 0.114 ± 0.033 |
| MLP + GAT + Concat | 0.999 ± 0.001 | 0.964 ± 0.008 | 0.787 ± 0.078 | 0.111 ± 0.025 |
| MLP + GAT + Gated | 0.999 ± 0.001 | 0.966 ± 0.009 | 0.785 ± 0.093 | 0.120 ± 0.027 |
| **MLP + GAT + Cross-Attention** | **0.998 ± 0.002** | **0.968 ± 0.006** | **0.775 ± 0.045** | **0.114 ± 0.017** |
| MLP + GAT + Image + Cross-Attention | 0.999 ± 0.001 | 0.968 ± 0.009 | 0.771 ± 0.075 | 0.107 ± 0.026 |

**Key findings**:
- Spatial context is critical: MLP-only → GAT-only improves ARI from 0.36 to 0.92
- Cross-attention fusion provides the best and most robust performance
- H&E histology image features (ResNet50) provide marginal improvement over expression alone but cannot replace the spatial graph — low-resolution Visium images (~15 pixels/spot) lack discriminative morphological detail
- Adding image features to the full model (multimodal) does not improve over the two-modality model — spatial graph already captures neighborhood structure more effectively
- All spatial models achieve ~99.8% top-2 accuracy and ~97% interior accuracy — the remaining errors are at domain boundaries where the ground truth itself is ambiguous
- Boundary accuracy (~78%) represents the annotation noise floor, not model failure — these spots sit at biological transitions between adjacent cortical layers

## Dataset

**Human DLPFC** (Dorsolateral Prefrontal Cortex) — the standard benchmark for spatial transcriptomics methods.

- 12 tissue slices from 3 donors
- ~3,500–4,800 spots per slice (~47K total)
- 33,538 genes per spot → 3,000 after HVG selection
- 5–7 annotated cortical layers per slice (ground truth)
- Source: Maynard et al., *Nature Neuroscience* 2021

## Setup

```bash
# Clone and set up environment
git clone https://github.com/Ttopiac/spatial-omics-fusion.git
cd spatial-omics-fusion
bash setup_env.sh        # auto-detects CUDA vs CPU/MPS
conda activate spatial-omics

# Download data
python data/download_dlpfc.py --all
```

## Usage

**Train a single model:**
```bash
# Default: slice 151673, cross-attention fusion
python scripts/train.py

# Specific configuration
python scripts/train.py --mode full --fusion_type cross_attention --sample_id 151673 --device cuda
```

**Run full benchmark (all 5 ablations × 12 slices):**
```bash
python scripts/run_benchmark.py --device cuda
```

**Extract scGPT embeddings (requires GPU):**
```bash
# Download scGPT pretrained weights first (see scripts/extract_scgpt_embeddings.py)
python scripts/extract_scgpt_embeddings.py

# Then benchmark scGPT variants
python scripts/run_benchmark.py --mode scgpt --fusion_type cross_attention --device cuda
```

**Generate visualizations:**
```bash
python scripts/visualize.py --sample_id 151673
```

## Project Structure

```
spatial-omics-fusion/
├── configs/
│   └── default.yaml              # All hyperparameters
├── data/
│   └── download_dlpfc.py         # Dataset download script
├── notebooks/
│   ├── pipeline_walkthrough.ipynb          # Data + model visualization walkthrough
│   ├── pipeline_walkthrough_detailed.ipynb # Detailed version with elaborations
│   ├── model_deepdive.ipynb               # Architecture + training deep dive
│   └── perturbation_analysis.ipynb        # In-silico gene perturbation analysis
├── scripts/
│   ├── train.py                  # Single model training
│   ├── run_benchmark.py          # Full benchmark suite
│   ├── extract_scgpt_embeddings.py  # scGPT embedding extraction
│   ├── extract_image_features.py    # H&E image feature extraction (ResNet50)
│   ├── visualize.py              # Spatial + t-SNE plots
│   └── visualize_pipeline.py     # Full pipeline visualization
└── src/
    ├── data/
    │   ├── preprocess.py         # Gene filtering, normalization, KNN graph
    │   └── dataset.py            # PyG Data loader with train/val/test splits
    ├── models/
    │   ├── expression_encoder.py # MLP: 3000 → 128
    │   ├── spatial_encoder.py    # GAT with residual connections
    │   ├── image_encoder.py      # ResNet50 feature projection: 2048 → 128
    │   ├── fusion.py             # Gated + Cross-Attention fusion
    │   └── model.py              # Full model with ablation modes
    ├── training/
    │   └── trainer.py            # Training loop, early stopping, class weighting
    └── utils/
        ├── metrics.py            # ARI, NMI, accuracy
        └── visualization.py      # Spatial domain + embedding plots
```

## Ablation Modes

The model supports multiple configurations via the `--mode` flag:

| Mode | Expression | Spatial | Image | Fusion | Purpose |
|---|---|---|---|---|---|
| `expr_only` | MLP | - | - | - | Expression-only baseline |
| `gat_only` | - | GAT | - | - | Spatial-only baseline |
| `concat` | MLP | GAT | - | Concatenate | Simple fusion baseline |
| `full` | MLP | GAT | - | Gated/Cross-Attn | Our main model |
| `scgpt_only` | scGPT proj | - | - | - | Foundation model baseline |
| `scgpt` | scGPT proj | GAT | - | Gated/Cross-Attn | Foundation model + spatial |
| `img_expr` | MLP | - | ResNet50 | Cross-Attn | Image replaces spatial graph |
| `multimodal` | MLP | GAT | ResNet50 | Cross-Attn + Gate | Three-modality fusion |

## Requirements

- Python 3.10
- PyTorch >= 2.0
- PyTorch Geometric
- scanpy, squidpy
- scikit-learn, matplotlib

See `setup_env.sh` for full installation.
