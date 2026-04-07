# Full Ablation Report

This document holds the detailed benchmark results for the Spatial Omics Fusion project. The headline finding lives in the project [README](../README.md): a plain GCN with k=96 spatial neighbors achieves the best ARI, top-2, interior, and boundary accuracy across all variants we tested.

All numbers are mean ± std across 12 DLPFC tissue slices.

## 1. Standard Metrics

The "winning" row (GCN-only k=96) is **bolded**. Bolding follows the headline model — *not* the per-column max — so other rows can be compared against it directly.

| Model | ARI | NMI | Accuracy |
|---|---|---|---|
| MLP-only (no spatial) | 0.356 ± 0.068 | 0.328 ± 0.043 | 0.607 ± 0.047 |
| Image + MLP + Cross-Attention | 0.447 ± 0.147 | 0.427 ± 0.065 | 0.659 ± 0.084 |
| scGPT-only (no spatial) | 0.192 ± 0.092 | 0.190 ± 0.045 | 0.334 ± 0.151 |
| scGPT + GAT + Gated | 0.256 ± 0.093 | 0.256 ± 0.048 | 0.348 ± 0.163 |
| scGPT + GAT + Cross-Attention | 0.250 ± 0.124 | 0.254 ± 0.062 | 0.330 ± 0.162 |
| scGPT (brain)-only (no spatial) | 0.187 ± 0.078 | 0.196 ± 0.035 | 0.332 ± 0.127 |
| scGPT (brain) + GAT + Gated | 0.327 ± 0.056 | 0.341 ± 0.061 | 0.450 ± 0.176 |
| scGPT (brain) + GAT + Cross-Attention | 0.318 ± 0.109 | 0.313 ± 0.074 | 0.414 ± 0.163 |
| Geneformer-only (no spatial) | 0.264 ± 0.064 | 0.283 ± 0.087 | 0.498 ± 0.085 |
| Geneformer + GAT + Cross-Attention | 0.491 ± 0.132 | 0.483 ± 0.145 | 0.618 ± 0.200 |
| GAT-only (k=6) | 0.919 ± 0.029 | 0.892 ± 0.018 | 0.958 ± 0.013 |
| GAT-only (k=12) | 0.924 ± 0.028 | 0.898 ± 0.018 | 0.960 ± 0.012 |
| MLP + GAT + Concat (k=6) | 0.917 ± 0.026 | 0.888 ± 0.015 | 0.956 ± 0.011 |
| MLP + GAT + Gated (k=6) | 0.920 ± 0.027 | 0.890 ± 0.017 | 0.958 ± 0.012 |
| MLP + GAT + Cross-Attention (k=6) | 0.928 ± 0.022 | 0.896 ± 0.011 | 0.960 ± 0.008 |
| MLP + GAT + Image + Cross-Attention (k=6) | 0.928 ± 0.023 | 0.899 ± 0.014 | 0.961 ± 0.010 |
| GCN-only (k=6) | 0.922 ± 0.027 | 0.893 ± 0.017 | 0.959 ± 0.011 |
| GCN-only (k=12) | 0.931 ± 0.025 | 0.898 ± 0.018 | 0.960 ± 0.012 |
| GCN-only (k=24) | 0.932 ± 0.025 | 0.924 ± 0.018 | 0.968 ± 0.013 |
| MLP + GCN + Cross-Attention (k=96) | 0.857 ± 0.073 | 0.835 ± 0.039 | 0.918 ± 0.030 |
| **GCN-only (k=96)** | **0.943 ± 0.013** | **0.922 ± 0.015** | **0.971 ± 0.007** |

## 2. Boundary-Aware Metrics

Standard accuracy under-credits the model: at layer boundaries the manual annotations themselves are ambiguous, so a 50/50 prediction can be the right answer but score as wrong. We split each test set into *interior* spots (all neighbors share the spot's label) and *boundary* spots (at least one neighbor has a different label). Top-2 accuracy is also reported — when the true label is in the model's top two outputs.

| Model | Top-2 Acc | Interior Acc | Boundary Acc | Log-Loss |
|---|---|---|---|---|
| MLP-only (no spatial) | 0.797 ± 0.043 | 0.616 ± 0.047 | 0.356 ± 0.081 | 1.315 ± 0.185 |
| Image + MLP + Cross-Attention | 0.837 ± 0.059 | 0.669 ± 0.083 | 0.390 ± 0.129 | 0.998 ± 0.249 |
| scGPT-only | 0.583 ± 0.135 | 0.339 ± 0.154 | 0.209 ± 0.068 | 1.727 ± 0.170 |
| scGPT + GAT + Gated | 0.546 ± 0.183 | 0.352 ± 0.168 | 0.264 ± 0.093 | 1.524 ± 0.200 |
| scGPT + GAT + Cross-Attention | 0.541 ± 0.188 | 0.333 ± 0.166 | 0.247 ± 0.076 | 1.697 ± 0.214 |
| scGPT (brain)-only | 0.582 ± 0.094 | 0.336 ± 0.129 | 0.230 ± 0.080 | 1.589 ± 0.216 |
| scGPT (brain) + GAT + Gated | 0.710 ± 0.151 | 0.455 ± 0.180 | 0.325 ± 0.148 | 1.235 ± 0.249 |
| scGPT (brain) + GAT + Cross-Attention | 0.657 ± 0.158 | 0.419 ± 0.164 | 0.255 ± 0.141 | 1.410 ± 0.335 |
| Geneformer-only | 0.713 ± 0.081 | 0.504 ± 0.090 | 0.337 ± 0.100 | 1.338 ± 0.230 |
| Geneformer + GAT + Cross-Attention | 0.829 ± 0.117 | 0.624 ± 0.201 | 0.427 ± 0.198 | 0.933 ± 0.386 |
| GAT-only (k=6) | 0.999 ± 0.001 | 0.965 ± 0.010 | 0.775 ± 0.083 | 0.114 ± 0.033 |
| GAT-only (k=12) | 1.000 ± 0.000 | 0.976 ± 0.010 | 0.859 ± 0.034 | 0.100 ± 0.026 |
| MLP + GAT + Concat (k=6) | 0.999 ± 0.001 | 0.964 ± 0.008 | 0.787 ± 0.078 | 0.111 ± 0.025 |
| MLP + GAT + Gated (k=6) | 0.999 ± 0.001 | 0.966 ± 0.009 | 0.785 ± 0.093 | 0.120 ± 0.027 |
| MLP + GAT + Cross-Attention (k=6) | 0.998 ± 0.002 | 0.968 ± 0.006 | 0.775 ± 0.045 | 0.114 ± 0.017 |
| MLP + GAT + Image + Cross-Attention (k=6) | 0.999 ± 0.001 | 0.968 ± 0.009 | 0.771 ± 0.075 | 0.107 ± 0.026 |
| GCN-only (k=6) | 0.999 ± 0.001 | 0.968 ± 0.009 | 0.778 ± 0.084 | 0.102 ± 0.026 |
| GCN-only (k=12) | 1.000 ± 0.000 | 0.981 ± 0.007 | 0.857 ± 0.033 | 0.082 ± 0.021 |
| GCN-only (k=24) | 1.000 ± 0.000 | 0.992 ± 0.003 | 0.901 ± 0.021 | 0.079 ± 0.024 |
| MLP + GCN + Cross-Attention (k=96) | 0.995 ± 0.007 | 0.999 ± 0.003 | 0.883 ± 0.034 | 0.237 ± 0.083 |
| **GCN-only (k=96)** | **1.000 ± 0.000** | **1.000 ± 0.000** | **0.957 ± 0.009** | **0.070 ± 0.018** |

## 3. KNN Neighbor Ablation — GAT-only

| k | ARI | Top-2 | Interior | Boundary | Log-Loss |
|---|---|---|---|---|---|
| 4 | 0.896 ± 0.040 | 0.996 ± 0.004 | 0.949 ± 0.017 | 0.666 ± 0.107 | 0.162 ± 0.045 |
| 6 | 0.919 ± 0.029 | 0.999 ± 0.001 | 0.965 ± 0.010 | 0.775 ± 0.083 | 0.114 ± 0.033 |
| 8 | 0.913 ± 0.031 | 1.000 ± 0.000 | 0.966 ± 0.010 | 0.789 ± 0.048 | 0.119 ± 0.032 |
| 12 | 0.924 ± 0.028 | 1.000 ± 0.000 | 0.976 ± 0.010 | 0.859 ± 0.034 | 0.100 ± 0.026 |
| 18 | 0.919 ± 0.029 | 1.000 ± 0.000 | 0.986 ± 0.005 | 0.861 ± 0.032 | 0.100 ± 0.032 |
| 24 | 0.916 ± 0.026 | 1.000 ± 0.000 | 0.991 ± 0.005 | 0.872 ± 0.022 | 0.110 ± 0.037 |

GAT-only ARI peaks at **k=12** (0.924) and degrades for larger k. The attention mechanism appears to struggle as the neighborhood grows — likely because softmax-normalized attention has to spread mass over more candidates, making gradients noisier. Boundary accuracy still improves with k, but the overall clustering quality starts to suffer.

## 4. KNN Neighbor Ablation — GCN-only

| k | ARI | Top-2 | Interior | Boundary | Log-Loss |
|---|---|---|---|---|---|
| 4 | 0.897 ± 0.035 | 0.997 ± 0.002 | 0.950 ± 0.013 | 0.678 ± 0.079 | 0.148 ± 0.038 |
| 6 | 0.922 ± 0.027 | 0.999 ± 0.001 | 0.968 ± 0.009 | 0.778 ± 0.084 | 0.102 ± 0.026 |
| 8 | 0.919 ± 0.025 | 1.000 ± 0.001 | 0.969 ± 0.007 | 0.803 ± 0.037 | 0.104 ± 0.025 |
| 12 | 0.931 ± 0.025 | 1.000 ± 0.000 | 0.981 ± 0.007 | 0.857 ± 0.033 | 0.082 ± 0.021 |
| 18 | 0.930 ± 0.025 | 1.000 ± 0.000 | 0.986 ± 0.006 | 0.889 ± 0.020 | 0.079 ± 0.020 |
| 24 | 0.932 ± 0.025 | 1.000 ± 0.000 | 0.992 ± 0.003 | 0.901 ± 0.021 | 0.079 ± 0.024 |
| 48 | 0.938 ± 0.024 | 1.000 ± 0.000 | 0.999 ± 0.001 | 0.939 ± 0.017 | 0.079 ± 0.019 |
| 64 | 0.938 ± 0.017 | 1.000 ± 0.000 | 0.999 ± 0.001 | 0.944 ± 0.014 | 0.074 ± 0.019 |
| **96** | **0.943 ± 0.013** | **1.000 ± 0.000** | **1.000 ± 0.000** | **0.957 ± 0.009** | **0.070 ± 0.018** |
| 128 | 0.939 ± 0.015 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.957 ± 0.009 | 0.080 ± 0.020 |
| 192 | 0.942 ± 0.016 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.960 ± 0.017 | 0.084 ± 0.029 |

GCN-only keeps improving until k≈96 and then plateaus. Unlike GAT, equal-weight averaging scales gracefully with the neighborhood size. From k=96 onward, **interior accuracy is 100%** — every remaining error sits at a domain boundary.

## 5. Discussion of Model Variants

### Why GCN beats GAT here

Both encoders aggregate neighbor embeddings through 2 conv layers, with virtually identical parameter counts (402K vs 403K — the 512 difference is GAT's attention weights). The difference is the aggregation rule:

- **GCN** computes a degree-normalized mean of neighbor features.
- **GAT** computes a softmax-weighted sum, where the weights are learned per edge from the source/target features.

Our hypothesis going in was that GAT's selectivity would help at layer boundaries — it should be able to down-weight neighbors from a different layer. Empirically, this is not what happens. With k=96, ~95% of a typical spot's neighbors share its label (cortical layers are large, contiguous regions), so plain averaging is already an excellent estimator. GAT's attention mechanism has nothing useful to do; it just adds learnable weights that need to be fit and that make optimization noisier. As k grows, GAT's softmax has to allocate mass over more candidates, which probably hurts gradient quality, and GAT-only ARI peaks early (k=12) and falls.

### Why MLP + GCN + Cross-Attention is *worse* than GCN-only

The full fusion model has three components on top of GCN: a 2-layer MLP expression encoder, a cross-attention layer, and a classifier. Each of these adds optimization variance without contributing new information — the GCN already sees the raw expression features through the spatial graph, and the spatial neighborhood is already aggregated through GCN. At k=96 the gap widens to ARI 0.943 vs 0.857, confirming that the extra modules are *capacity that the model has to waste*.

### Where the fusion architectures still have value: interpretability

Even though GCN-only is the most accurate, the fusion models (especially cross-attention) offer something GCN does not: **per-spot interpretability**. With cross-attention you can read out, for each spot, which neighbors the model attended to and which expression channels mattered most. We use those attention maps in our perturbation analysis (see `notebooks/perturbation_analysis.ipynb` — kept private) to identify spots whose classification relies on a small set of marker genes.

If the goal is *just* clustering, use GCN. If the goal is biological interpretation — explaining *why* a spot was assigned to a particular layer — the cross-attention model is worth its extra parameters.

### Foundation models (scGPT, Geneformer)

We tested three foundation models with frozen embeddings: scGPT (51M params, whole-human checkpoint), scGPT-brain (brain-specific checkpoint), and Geneformer (104M params). All three underperformed even GAT-only by a wide margin. The strongest configuration (Geneformer + GAT + Cross-Attention) reaches 0.491 ARI, vs 0.919 for GAT-only at k=6.

The reason is pretraining domain mismatch: both models were trained on **dissociated single-cell** RNA-seq, where cells have been physically separated from tissue and all spatial information is destroyed. Their embeddings encode cell-type identity, not tissue position. For a spatial domain detection task, this is the wrong inductive bias — and no amount of model capacity compensates for the missing spatial signal.

This is a verdict on these two specific checkpoints, not on foundation models in general. A foundation model pretrained directly on **spatial** transcriptomics data (Visium, MERFISH, Xenium, etc.) would carry the right inductive bias and could plausibly beat a 2-layer GCN, particularly for low-data slices, cross-tissue transfer, or rare layer types. The takeaway from our experiment is narrower: *match the pretraining domain to the downstream task* — a foundation model is only a head start when it has actually seen data similar to what you are predicting on.

We did not fine-tune scGPT/Geneformer end-to-end. Doing so on a single slice produced *worse* results than the frozen baseline (consistent with Cui et al.'s findings), but a multi-slice fine-tune is on the to-do list.

### Image modality (ResNet50)

We extracted 64×64 H&E patches around each Visium spot and encoded them with a pretrained ResNet50 (ImageNet V2). Adding these features alongside the spatial graph did not improve performance (`multimodal` mode) and replacing the spatial graph with images entirely (`img_expr` mode) degraded ARI to 0.45.

The likely reason is resolution: each Visium spot is ~55 µm in diameter and the underlying H&E image gives you ~15 pixels per spot. That is not enough for ResNet50 to extract layer-distinguishing morphology — cortical layers are defined by cell-density gradients that need much higher resolution to read. On Visium HD or 10x Xenium data, image features would likely contribute more.

## 6. Ablation Modes

The model supports the following configurations via the `--mode` flag:

| Mode | Expression | Spatial | Image | Fusion | Purpose |
|---|---|---|---|---|---|
| `expr_only` | MLP | – | – | – | Expression-only baseline |
| `gat_only` | – | GAT | – | – | GAT-only spatial baseline |
| `gcn_only` | – | GCN | – | – | **GCN-only spatial baseline (winning model)** |
| `concat` | MLP | GAT | – | Concatenate | Simple fusion baseline |
| `full` | MLP | GAT | – | Gated / Cross-Attn | MLP+GAT fusion model |
| `gcn_full` | MLP | GCN | – | Gated / Cross-Attn | MLP+GCN fusion model |
| `scgpt_only` | scGPT proj | – | – | – | scGPT baseline |
| `scgpt` | scGPT proj | GAT | – | Gated / Cross-Attn | scGPT + spatial |
| `scgpt_brain_only` | scGPT (brain) proj | – | – | – | Brain-specific scGPT baseline |
| `scgpt_brain` | scGPT (brain) proj | GAT | – | Gated / Cross-Attn | Brain-specific scGPT + spatial |
| `geneformer_only` | Geneformer proj | – | – | – | Geneformer baseline |
| `geneformer` | Geneformer proj | GAT | – | Gated / Cross-Attn | Geneformer + spatial |
| `img_expr` | MLP | – | ResNet50 | Cross-Attn | Image replaces spatial graph |
| `multimodal` | MLP | GAT | ResNet50 | Cross-Attn + Gate | Three-modality fusion |

## 7. Reproducing the Headline Result

```bash
# Preprocess with k=96 (one-time, ~2 min per slice)
python -m src.data.dataset --k_neighbors 96 --processed_dir data/processed_k96

# Run GCN-only on all 12 slices (~25 min on M1, ~5 min on a single A100)
python scripts/run_benchmark.py --mode gcn_only --processed_dir data/processed_k96
```
