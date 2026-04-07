# Spatial Omics Fusion

A benchmark study of graph neural networks for spatial transcriptomics domain detection on the human DLPFC dataset. We compare graph encoders (GCN, GAT), fusion architectures (concatenation, gated, cross-attention), foundation models (scGPT, Geneformer), and image features (ResNet50) — and find that **the simplest model wins**.

## Headline Result

A plain **2-layer GCN** with a wide spatial graph (k=96) achieves the best results across the board:

| | ARI | Top-2 Acc | Interior Acc | Boundary Acc | Params |
|---|---|---|---|---|---|
| **GCN-only (k=96)** | **0.943 ± 0.013** | **1.000** | **1.000** | **0.957 ± 0.009** | **402K** |
| MLP + GAT + Cross-Attention (k=6) | 0.928 ± 0.022 | 0.998 | 0.968 | 0.775 ± 0.045 | 1.1M |
| MLP + GCN + Cross-Attention (k=96) | 0.857 ± 0.073 | 0.995 | 0.999 | 0.883 ± 0.034 | 1.2M |
| MLP-only (no spatial) | 0.356 ± 0.068 | 0.797 | 0.616 | 0.356 ± 0.081 | 0.8M |

100% interior accuracy means **every remaining error sits at a domain boundary** where the manual annotations themselves are ambiguous. Top-2 accuracy is 1.000 — when the model picks "wrong", the ground-truth label is always its second choice.

> Full ablation tables (16+ variants × 12 slices), foundation-model results, and the GAT/GCN KNN sweeps live in **[docs/RESULTS.md](docs/RESULTS.md)**.

## Why the Simple Model Wins

Three findings, in order of how much they surprised us:

1. **Graph construction matters more than model architecture.** Increasing the spatial graph from k=6 to k=96 improved GCN-only ARI from 0.922 to 0.943 — a bigger gain than any fusion strategy delivered. The right data representation beats sophisticated fusion.
2. **GCN beats GAT at every k.** Learned attention weights add no benefit once each spot has enough neighbors — most neighbors are same-domain, so equal averaging is fine, and attention just adds optimization noise.
3. **Adding MLP + Cross-Attention on top of GCN hurts.** MLP+GCN+CrossAttn at k=96 drops to 0.857 ARI. The expression MLP and the cross-attention layer each add ~0.4M parameters of nuisance variance that the lean GCN avoids.

The complex fusion models are not useless — they offer **per-spot interpretability** (you can read which neighbors and which gene-expression channels each spot attended to). For pure accuracy on this task, plain GCN is the right choice.

## Architecture

```
Gene Expression (3000 HVGs per spot)        Spatial coordinates (x, y)
              │                                       │
              │                              k=96 KNN graph
              │                                       │
              └─────────────────┬─────────────────────┘
                                ▼
                  GCN Encoder (2 layers)
                  mean-aggregates over 96 neighbors
                                │
                                ▼
                       Spot Embedding (128)
                                │
                                ▼
                       Classification Head ──→ Cortical Layer (5–7 classes)
```

We also implement an MLP expression encoder, a GAT spatial encoder, gated and cross-attention fusion, image-feature (ResNet50) and foundation-model (scGPT, Geneformer) inputs. All variants are selectable via the `--mode` flag — see [docs/RESULTS.md](docs/RESULTS.md) for the full comparison.

## Dataset

**Human DLPFC** (Maynard et al., *Nature Neuroscience* 2021) — 12 tissue slices from 3 donors, ~47K spots, 33,538 genes per spot (filtered to 3,000 highly variable), 5–7 manually annotated cortical layers per slice.

## Setup

```bash
git clone https://github.com/Ttopiac/spatial-omics-fusion.git
cd spatial-omics-fusion
bash setup_env.sh        # auto-detects CUDA vs CPU/MPS
conda activate spatial-omics
python data/download_dlpfc.py --all
```

## Usage

```bash
# Train the headline model on one slice
python scripts/train.py --mode gcn_only --sample_id 151673

# Run the full benchmark (all modes × 12 slices)
python scripts/run_benchmark.py --device cuda
```

See [docs/RESULTS.md](docs/RESULTS.md) for the full mode list, foundation-model setup, and reproducing each ablation.

## Project Structure

```
spatial-omics-fusion/
├── configs/default.yaml         # Hyperparameters
├── data/download_dlpfc.py       # Dataset download
├── docs/RESULTS.md              # Full ablation report
├── scripts/                     # Training, benchmarks, visualizations
└── src/
    ├── data/                    # Preprocessing, KNN graph, dataset loaders
    ├── models/                  # MLP, GCN, GAT, fusion, image, foundation
    ├── training/trainer.py
    └── utils/                   # Metrics, visualization
```

## Requirements

Python 3.10, PyTorch ≥ 2.0, PyTorch Geometric, scanpy, squidpy, scikit-learn. See `setup_env.sh`.
