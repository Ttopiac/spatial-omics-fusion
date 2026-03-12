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
Expression Embedding (128) ───────────────────────┐
        │                                          │
        ▼                                          │
   GAT Encoder (aggregates spatial neighbors)      │
        │                                          │
        ▼                                          │
Spatial Embedding (128)                            │
        │                                          │
        ▼                                          ▼
Cross-Attention Fusion (Q = Expression, K/V = Spatial Neighbors)
        │
        ▼
Classification Head ──→ Domain Prediction (5-7 classes)
```

**Expression Encoder**: 2-layer MLP compresses 3,000 highly variable genes into a 128-dim embedding.

**Spatial Encoder**: 2-layer Graph Attention Network aggregates neighbor information over a k=6 spatial KNN graph, using expression embeddings as node features.

**Cross-Attention Fusion**: Expression embedding serves as query; spatial embeddings of graph neighbors serve as keys/values. Each spot selectively attends to relevant neighbors.

## Results

Benchmarked across all 12 DLPFC tissue slices (mean ± std):

| Model | ARI | NMI | Accuracy |
|---|---|---|---|
| MLP-only (no spatial) | 0.366 ± 0.083 | 0.404 ± 0.064 | 0.547 ± 0.067 |
| Frozen scGPT-only | 0.184 ± 0.077 | 0.195 ± 0.039 | 0.338 ± 0.122 |
| GAT-only | 0.920 ± 0.030 | 0.890 ± 0.022 | 0.940 ± 0.019 |
| MLP + GAT + Concat | 0.917 ± 0.029 | 0.889 ± 0.021 | 0.940 ± 0.016 |
| MLP + GAT + Gated | 0.915 ± 0.024 | 0.887 ± 0.019 | 0.938 ± 0.016 |
| **MLP + GAT + Cross-Attention** | **0.926 ± 0.024** | **0.896 ± 0.018** | **0.944 ± 0.015** |
| Frozen scGPT + GAT + Gated | 0.245 ± 0.101 | 0.248 ± 0.050 | 0.401 ± 0.154 |
| Frozen scGPT + GAT + Cross-Attention | 0.257 ± 0.104 | 0.275 ± 0.047 | 0.416 ± 0.122 |

**Key findings**:
- Spatial context is critical: MLP-only → GAT-only improves ARI from 0.37 to 0.92
- Cross-attention fusion provides the best and most robust performance
- Frozen scGPT embeddings (pretrained on 33M cells) underperform task-specific MLP, indicating foundation model representations need fine-tuning for spatial domain detection

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
├── scripts/
│   ├── train.py                  # Single model training
│   ├── run_benchmark.py          # Full benchmark suite
│   ├── extract_scgpt_embeddings.py  # scGPT embedding extraction
│   └── visualize.py              # Spatial + t-SNE plots
└── src/
    ├── data/
    │   ├── preprocess.py         # Gene filtering, normalization, KNN graph
    │   └── dataset.py            # PyG Data loader with train/val/test splits
    ├── models/
    │   ├── expression_encoder.py # MLP: 3000 → 128
    │   ├── spatial_encoder.py    # GAT with residual connections
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

| Mode | Expression | Spatial | Fusion | Purpose |
|---|---|---|---|---|
| `expr_only` | MLP | - | - | Expression-only baseline |
| `gat_only` | - | GAT | - | Spatial-only baseline |
| `concat` | MLP | GAT | Concatenate | Simple fusion baseline |
| `full` | MLP | GAT | Gated/Cross-Attn | Our main model |
| `scgpt_only` | scGPT proj | - | - | Foundation model baseline |
| `scgpt` | scGPT proj | GAT | Gated/Cross-Attn | Foundation model + spatial |

## Requirements

- Python 3.10
- PyTorch >= 2.0
- PyTorch Geometric
- scanpy, squidpy
- scikit-learn, matplotlib

See `setup_env.sh` for full installation.
