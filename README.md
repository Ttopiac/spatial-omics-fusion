# Spatial Omics Fusion

## The Task

Given a slice of human brain tissue measured by spatial transcriptomics, **assign every tissue spot to one of the 5–7 cortical layers** (Layer 1 through Layer 6 + white matter). Each spot reports the expression of ~33,000 genes plus an (x, y) coordinate on the slide. The benchmark is the human **DLPFC** dataset (Maynard et al., *Nature Neuroscience* 2021): 12 brain tissue slices from 3 donors, ~47,000 spots total, with manually annotated layer labels we treat as ground truth.

This is a per-spot multi-class classification problem on graph-structured biological data — the same shape as many drug-discovery and disease-biology questions where you have to label cells, spots, or genes from a mix of measurements and spatial / network context.

## What This Repo Does

A benchmark study of graph neural networks and modality-fusion strategies for the task above. We compare graph encoders (GCN, GAT), fusion architectures (concatenation, gated, cross-attention), foundation models (scGPT, Geneformer), and image features (ResNet50) on all 12 slices — and find that **the simplest model wins**.

## Headline Result

A plain **2-layer GCN** with a wide spatial graph (k=96) achieves the best results across the board:

| | ARI | Top-2 Acc | Interior Acc | Boundary Acc | Params |
|---|---|---|---|---|---|
| **GCN-only (k=96)** | **0.943 ± 0.013** | **1.000** | **1.000** | **0.957 ± 0.009** | **402K** |
| MLP + GAT + Cross-Attention (k=6) | 0.928 ± 0.022 | 0.998 | 0.968 | 0.775 ± 0.045 | 1.1M |
| MLP + GCN + Cross-Attention (k=96) | 0.857 ± 0.073 | 0.995 | 0.999 | 0.883 ± 0.034 | 1.2M |
| MLP-only (no spatial) | 0.356 ± 0.068 | 0.797 | 0.616 | 0.356 ± 0.081 | 0.8M |

100% interior accuracy means **every remaining error sits at a domain boundary** where the manual annotations themselves are ambiguous. Top-2 accuracy is 1.000 — when the model picks "wrong", the ground-truth label is always its second choice.

## vs Foundation Models

A 402K-parameter GCN trained from scratch beats every frozen single-cell foundation model we tested — by a wide margin:

| Model | Pretraining cells | Params | ARI | Boundary Acc |
|---|---|---|---|---|
| **GCN-only (k=96), trained from scratch** | — | **402K** | **0.943 ± 0.013** | **0.957 ± 0.009** |
| Geneformer + GAT + Cross-Attention | 30M dissociated | 104M | 0.491 ± 0.132 | 0.427 ± 0.198 |
| scGPT (brain) + GAT + Gated | 33M dissociated (brain subset) | 51M | 0.327 ± 0.056 | 0.325 ± 0.148 |
| scGPT + GAT + Cross-Attention | 33M dissociated | 51M | 0.250 ± 0.124 | 0.247 ± 0.076 |
| Geneformer-only (no spatial) | 30M dissociated | 104M | 0.264 ± 0.064 | 0.337 ± 0.100 |
| scGPT-only (no spatial) | 33M dissociated | 51M | 0.192 ± 0.092 | 0.209 ± 0.068 |

We used the publicly released checkpoints: scGPT (51M params, whole-human and brain-specific) and Geneformer-V2-104M. Geneformer is also released in smaller sizes (10M, 30M); we report the 104M variant since it is the largest and gives the strongest foundation-model baseline.

**Why the gap is this large:** scGPT and Geneformer were both pretrained on **dissociated** single-cell RNA-seq — cells physically separated from tissue, with all spatial information destroyed during sample preparation. Their embeddings encode cell-type identity, not tissue position. Spatial domain detection needs the opposite signal: where a spot sits in the tissue, not what individual cell type it most resembles. No amount of model capacity (up to 100×) compensates for this domain mismatch.

This is **not a verdict on foundation models in general** — only on these two, on this task. A foundation model pretrained on **spatial** transcriptomics data (Visium, MERFISH, Xenium, etc.) would presumably encode the right inductive bias and could plausibly outperform a 2-layer GCN, especially in low-data or cross-tissue transfer settings. The lesson here is narrower: *match your pretraining domain to your downstream task.* A 2-layer GCN that starts from the spatial graph is doing the right thing on the right data, while frozen embeddings from a dissociated-cell foundation model are not.

> Full ablation tables (16+ variants × 12 slices), all foundation-model variants, and the GAT/GCN KNN sweeps live in **[docs/RESULTS.md](docs/RESULTS.md)**.

## Why the Simple Model Wins

### What each encoder actually represents

The MLP and the GCN look at the same 3,000-dim gene expression vector — but they encode fundamentally different things:

- **MLP encoder** sees only **one spot at a time**. Its embedding answers "*which genes are active in this single spot, in isolation?*" — i.e., a per-spot transcriptional fingerprint. It cannot see neighbors.
- **GCN encoder** sees the spot **plus a 96-spot neighborhood**, and at each layer it averages neighbor features. Its embedding answers "*what does the local tissue patch around this spot look like, on average?*" — i.e., a denoised, spatially-pooled expression profile.

For cortical layer detection, the second question is much closer to the right one. Cortical layers are large contiguous tissue regions, so a spot's true layer is far better predicted by its neighborhood's average expression (GCN) than by its own noisy gene counts (MLP). Per-spot expression is also dropout-heavy at Visium resolution — pooling over 96 neighbors averages out the noise.

This is why **spatial information dominates this task**. Adding spatial context lifts ARI from **0.36 (MLP-only)** → **0.94 (GCN k=96)**, a +0.58 jump that no fusion architecture or foundation model has come close to closing. Once GCN has the spatial signal, the per-spot MLP features become redundant (they are already inside the GCN input) and adding an MLP+Cross-Attention path on top just injects nuisance variance — which is exactly why **MLP+GCN+CrossAttn drops to 0.857 ARI**.

### Three takeaways

1. **Spatial context is the dominant signal.** Going from no-spatial (MLP-only) to spatial (GCN-only) improves ARI by 0.58. No fusion strategy, no foundation model, and no image modality moves the needle by even a tenth of that.
2. **Graph construction matters more than model architecture.** Increasing k from 6 to 96 (still GCN-only) improves ARI from 0.922 to 0.943 — a bigger gain than any fusion strategy ever delivered. With enough neighbors, even a 2-layer GCN saturates the task.
3. **GCN beats GAT at every k.** Once each spot has enough neighbors, most of them share its label and equal averaging is already an excellent estimator. GAT's attention weights have nothing useful to do — they just add learnable parameters that need to be fit.

The fusion models (cross-attention, gated) are not useless: they offer **per-spot interpretability** (you can read which neighbors and which gene channels each spot attended to). For pure accuracy on this task, though, plain GCN is the right choice.

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
