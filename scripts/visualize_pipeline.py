"""
Comprehensive visualization of the full pipeline: data preprocessing steps + model internals.

Generates figures for interview walkthrough:

Part 1 — Data Pipeline:
  1. Raw expression heatmap (before any processing)
  2. After normalization + log transform
  3. After HVG selection + scaling (final input)
  4. Spatial KNN graph overlaid on tissue
  5. Label distribution across classes

Part 2 — Model Internals (requires trained checkpoints):
  6. Expression embeddings (MLP output) — t-SNE
  7. Spatial embeddings (GAT output) — t-SNE
  8. Fused embeddings (cross-attention output) — t-SNE
  9. Embedding progression: MLP → GAT → Fused side-by-side
  10. Cross-attention weights — which neighbors each spot attends to
  11. Gated fusion gate values — expression vs spatial reliance per spot

Usage:
    python scripts/visualize_pipeline.py --sample_id 151673
    python scripts/visualize_pipeline.py --sample_id 151673 --skip_models  # data pipeline only
"""
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Lazy imports for scanpy (only needed for data pipeline viz)
DOMAIN_COLORS = [
    "#E41A1C",  # Layer1 - red
    "#FF7F00",  # Layer2 - orange
    "#FFD700",  # Layer3 - gold
    "#4DAF4A",  # Layer4 - green
    "#377EB8",  # Layer5 - blue
    "#984EA3",  # Layer6 - purple
    "#A65628",  # WM - brown
]
DOMAIN_NAMES = ["Layer1", "Layer2", "Layer3", "Layer4", "Layer5", "Layer6", "WM"]


# ============================================================================
# Part 1: Data Pipeline Visualizations
# ============================================================================

def viz_preprocessing_steps(sample_id, raw_dir, fig_dir):
    """
    Show how expression data transforms at each preprocessing step.
    Generates a 1x3 heatmap panel: raw → normalized → final.
    """
    import scanpy as sc

    raw_path = os.path.join(raw_dir, f"{sample_id}.h5ad")
    print(f"  Loading raw data: {raw_path}")
    adata = sc.read_h5ad(raw_path)

    # Drop unlabeled spots (same as preprocess.py)
    adata = adata[~adata.obs["sce.layer_guess"].isna()].copy()
    sc.pp.filter_genes(adata, min_cells=3)

    # Snapshot 1: raw counts (subset for visualization)
    from scipy.sparse import issparse
    raw_expr = adata.X.toarray() if issparse(adata.X) else adata.X.copy()

    # Normalize + log
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    norm_expr = adata.X.toarray() if issparse(adata.X) else adata.X.copy()

    # HVG selection + scale
    sc.pp.highly_variable_genes(adata, n_top_genes=3000)
    hvg_mask = adata.var.highly_variable.values
    adata_hvg = adata[:, hvg_mask].copy()
    sc.pp.scale(adata_hvg, max_value=10)
    final_expr = adata_hvg.X.toarray() if issparse(adata_hvg.X) else adata_hvg.X.copy()

    # Pick a random subset of spots and genes for the heatmap
    np.random.seed(42)
    n_spots_show = 100
    n_genes_show = 50
    spot_idx = np.sort(np.random.choice(raw_expr.shape[0], n_spots_show, replace=False))

    # For raw and norm, pick top variable genes from the HVG set
    hvg_indices = np.where(hvg_mask)[0][:n_genes_show]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Raw counts
    im0 = axes[0].imshow(raw_expr[np.ix_(spot_idx, hvg_indices)],
                          aspect="auto", cmap="viridis", interpolation="nearest")
    axes[0].set_title("Step 1: Raw Counts", fontsize=13, fontweight="bold")
    axes[0].set_xlabel(f"{n_genes_show} genes (HVGs)")
    axes[0].set_ylabel(f"{n_spots_show} spots (sampled)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # After normalize + log
    im1 = axes[1].imshow(norm_expr[np.ix_(spot_idx, hvg_indices)],
                          aspect="auto", cmap="viridis", interpolation="nearest")
    axes[1].set_title("Step 2: Normalize + Log1p", fontsize=13, fontweight="bold")
    axes[1].set_xlabel(f"{n_genes_show} genes (HVGs)")
    axes[1].set_ylabel("")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # After HVG + scale
    im2 = axes[2].imshow(final_expr[np.ix_(spot_idx, np.arange(n_genes_show))],
                          aspect="auto", cmap="RdBu_r", interpolation="nearest",
                          vmin=-3, vmax=3)
    axes[2].set_title("Step 3: HVG Selection + Scale", fontsize=13, fontweight="bold")
    axes[2].set_xlabel(f"{n_genes_show} / 3000 HVGs")
    axes[2].set_ylabel("")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle(f"Data Preprocessing Pipeline — Slice {sample_id}",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = os.path.join(fig_dir, "01_preprocessing_steps.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def viz_expression_distribution(sample_id, raw_dir, fig_dir):
    """
    Show how the distribution of expression values changes at each step.
    Histogram panel: raw → normalized → scaled.
    """
    import scanpy as sc
    from scipy.sparse import issparse

    adata = sc.read_h5ad(os.path.join(raw_dir, f"{sample_id}.h5ad"))
    adata = adata[~adata.obs["sce.layer_guess"].isna()].copy()
    sc.pp.filter_genes(adata, min_cells=3)

    raw_vals = (adata.X.toarray() if issparse(adata.X) else adata.X).flatten()
    raw_nonzero = raw_vals[raw_vals > 0]

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    norm_vals = (adata.X.toarray() if issparse(adata.X) else adata.X).flatten()
    norm_nonzero = norm_vals[norm_vals > 0]

    sc.pp.highly_variable_genes(adata, n_top_genes=3000)
    adata = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata, max_value=10)
    final_vals = (adata.X.toarray() if issparse(adata.X) else adata.X).flatten()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(raw_nonzero, bins=100, color="#377EB8", alpha=0.8, edgecolor="none")
    axes[0].set_title("Raw Counts (nonzero)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Expression value")
    axes[0].set_ylabel("Frequency")
    axes[0].set_yscale("log")

    axes[1].hist(norm_nonzero, bins=100, color="#4DAF4A", alpha=0.8, edgecolor="none")
    axes[1].set_title("After Normalize + Log1p", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Expression value")

    axes[2].hist(final_vals, bins=100, color="#984EA3", alpha=0.8, edgecolor="none")
    axes[2].set_title("After HVG + Scale (z-scored)", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Expression value")

    plt.suptitle(f"Expression Distribution at Each Step — Slice {sample_id}",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = os.path.join(fig_dir, "02_expression_distributions.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def viz_spatial_graph(sample_id, processed_dir, fig_dir):
    """
    Show the KNN spatial graph overlaid on tissue coordinates, colored by domain.
    Left: spots only, Right: spots + edges.
    """
    data_dir = os.path.join(processed_dir, sample_id)
    coords = torch.load(os.path.join(data_dir, "coordinates.pt"), weights_only=True).numpy()
    labels = torch.load(os.path.join(data_dir, "labels.pt"), weights_only=True).numpy()
    edge_index = torch.load(os.path.join(data_dir, "edge_index.pt"), weights_only=True).numpy()

    with open(os.path.join(data_dir, "metadata.json")) as f:
        meta = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: spots colored by domain
    for i, name in enumerate(DOMAIN_NAMES[:meta["n_classes"]]):
        mask = labels == i
        if mask.sum() > 0:
            axes[0].scatter(coords[mask, 0], coords[mask, 1], c=DOMAIN_COLORS[i],
                            label=name, s=10, alpha=0.8, edgecolors="none")
    axes[0].set_title("Tissue Spots (ground truth)", fontsize=13, fontweight="bold")
    axes[0].set_aspect("equal")
    axes[0].invert_yaxis()
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9, markerscale=2)

    # Right: spots + KNN edges (subsample edges for clarity)
    # Draw edges first (behind spots)
    n_edges = edge_index.shape[1]
    # Subsample if too many edges
    if n_edges > 5000:
        edge_sample = np.random.choice(n_edges, 5000, replace=False)
    else:
        edge_sample = np.arange(n_edges)

    lines = []
    for idx in edge_sample:
        src, tgt = edge_index[0, idx], edge_index[1, idx]
        lines.append([coords[src], coords[tgt]])

    lc = LineCollection(lines, colors="#cccccc", linewidths=0.3, alpha=0.5)
    axes[1].add_collection(lc)

    for i, name in enumerate(DOMAIN_NAMES[:meta["n_classes"]]):
        mask = labels == i
        if mask.sum() > 0:
            axes[1].scatter(coords[mask, 0], coords[mask, 1], c=DOMAIN_COLORS[i],
                            label=name, s=10, alpha=0.8, edgecolors="none", zorder=2)

    axes[1].set_title(f"Spatial KNN Graph (k={meta['knn_k']}, {n_edges} edges)",
                      fontsize=13, fontweight="bold")
    axes[1].set_aspect("equal")
    axes[1].invert_yaxis()
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].autoscale_view()
    axes[1].legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9, markerscale=2)

    plt.suptitle(f"Spatial Structure — Slice {sample_id}",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = os.path.join(fig_dir, "03_spatial_graph.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def viz_label_distribution(sample_id, processed_dir, fig_dir):
    """Bar chart of spots per domain class."""
    data_dir = os.path.join(processed_dir, sample_id)
    labels = torch.load(os.path.join(data_dir, "labels.pt"), weights_only=True).numpy()
    with open(os.path.join(data_dir, "metadata.json")) as f:
        meta = json.load(f)

    n_classes = meta["n_classes"]
    counts = [np.sum(labels == i) for i in range(n_classes)]
    names = DOMAIN_NAMES[:n_classes]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, counts, color=DOMAIN_COLORS[:n_classes], edgecolor="white", linewidth=0.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                str(count), ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title(f"Label Distribution — Slice {sample_id} ({sum(counts)} spots)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of spots")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_path = os.path.join(fig_dir, "04_label_distribution.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================================
# Part 2: Model Internals Visualizations
# ============================================================================

def _load_model(result_dir, data, config, mode, fusion_type):
    """Load a trained model from checkpoint."""
    from src.models.model import SpatialOmicsFusion

    model = SpatialOmicsFusion(
        n_genes=data.x.shape[1],
        n_classes=data.n_classes,
        embed_dim=config["model"]["embed_dim"],
        hidden_dim=config["model"]["expression_encoder"]["hidden_dim"],
        expr_layers=config["model"]["expression_encoder"]["n_layers"],
        gat_heads=config["model"]["spatial_encoder"]["n_heads"],
        gat_layers=config["model"]["spatial_encoder"]["n_layers"],
        fusion_type=fusion_type,
        fusion_heads=config["model"]["fusion"]["n_heads"],
        fusion_layers=config["model"]["fusion"]["n_layers"],
        dropout=0.0,
        mode=mode,
    )
    state = torch.load(os.path.join(result_dir, "model.pt"),
                       map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _tsne_2d(embeddings, perplexity=30):
    """Run t-SNE on embeddings."""
    from sklearn.manifold import TSNE
    return TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(embeddings)


def _plot_tsne(coords_2d, labels, n_classes, title, ax):
    """Plot t-SNE scatter colored by label."""
    for i in range(n_classes):
        mask = labels == i
        if mask.sum() > 0:
            ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1], c=DOMAIN_COLORS[i],
                       label=DOMAIN_NAMES[i], s=3, alpha=0.6, edgecolors="none")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


@torch.no_grad()
def _extract_intermediates(model, data):
    """
    Extract intermediate embeddings from the full model (mode='full').
    Returns dict with expr_embed, spatial_embed, fused_embed.
    """
    intermediates = {}

    # Expression embedding
    expr_embed = model.expr_encoder(data.x)
    intermediates["expr_embed"] = expr_embed.cpu().numpy()

    # Spatial embedding
    spatial_embed = model.spatial_encoder(expr_embed, data.edge_index)
    intermediates["spatial_embed"] = spatial_embed.cpu().numpy()

    # Fused embedding
    fused_embed = model.fusion(expr_embed, spatial_embed, data.edge_index)
    intermediates["fused_embed"] = fused_embed.cpu().numpy()

    # Predictions
    logits = model.classifier(fused_embed)
    intermediates["preds"] = logits.argmax(dim=-1).cpu().numpy()

    return intermediates


def viz_embedding_progression(sample_id, config, fig_dir):
    """
    t-SNE of embeddings at each stage: MLP output → GAT output → Fused output.
    Shows how representations improve through the pipeline.
    """
    from src.data.dataset import load_dlpfc_data

    data = load_dlpfc_data(sample_id, seed=config["training"]["seed"])
    labels = data.y.numpy()
    n_classes = data.n_classes

    # Load the cross-attention model (best performing)
    result_dir = f"results/full_cross_attention_{sample_id}"
    if not os.path.exists(os.path.join(result_dir, "model.pt")):
        print(f"  Skipping embedding progression (no cross-attention checkpoint for {sample_id})")
        return

    model = _load_model(result_dir, data, config, "full", "cross_attention")
    intermediates = _extract_intermediates(model, data)

    # t-SNE for each stage
    print("  Computing t-SNE for expression embeddings...")
    tsne_expr = _tsne_2d(intermediates["expr_embed"])
    print("  Computing t-SNE for spatial embeddings...")
    tsne_spatial = _tsne_2d(intermediates["spatial_embed"])
    print("  Computing t-SNE for fused embeddings...")
    tsne_fused = _tsne_2d(intermediates["fused_embed"])

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    _plot_tsne(tsne_expr, labels, n_classes,
               "Expression Embedding\n(MLP output, 128-dim)", axes[0])
    _plot_tsne(tsne_spatial, labels, n_classes,
               "Spatial Embedding\n(GAT output, 128-dim)", axes[1])
    _plot_tsne(tsne_fused, labels, n_classes,
               "Fused Embedding\n(Cross-Attention output, 128-dim)", axes[2])

    # Shared legend
    handles = [mpatches.Patch(color=DOMAIN_COLORS[i], label=DOMAIN_NAMES[i])
               for i in range(n_classes)]
    fig.legend(handles=handles, loc="lower center", ncol=n_classes,
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle(f"Embedding Progression Through Model — Slice {sample_id}",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    save_path = os.path.join(fig_dir, "05_embedding_progression.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")

    return intermediates, data


def viz_spatial_predictions(sample_id, config, fig_dir, intermediates=None, data=None):
    """
    Side-by-side: ground truth vs predictions on tissue, for each ablation model.
    """
    from src.data.dataset import load_dlpfc_data

    if data is None:
        data = load_dlpfc_data(sample_id, seed=config["training"]["seed"])

    coords = data.pos.numpy()
    labels = data.y.numpy()
    n_classes = data.n_classes

    models_to_show = [
        ("expr_only", "gated", "MLP Only"),
        ("gat_only", "gated", "GAT Only"),
        ("full", "cross_attention", "MLP+GAT+CrossAttn"),
    ]

    available = []
    for mode, ftype, name in models_to_show:
        rdir = f"results/{mode}_{ftype}_{sample_id}"
        if os.path.exists(os.path.join(rdir, "model.pt")):
            available.append((mode, ftype, name, rdir))

    if not available:
        print("  Skipping spatial predictions (no checkpoints found)")
        return

    n_models = len(available)
    fig, axes = plt.subplots(1, n_models + 1, figsize=(6 * (n_models + 1), 6))

    # Ground truth
    for i in range(n_classes):
        mask = labels == i
        if mask.sum() > 0:
            axes[0].scatter(coords[mask, 0], coords[mask, 1], c=DOMAIN_COLORS[i],
                            label=DOMAIN_NAMES[i], s=8, alpha=0.8, edgecolors="none")
    axes[0].set_title("Ground Truth", fontsize=13, fontweight="bold")
    axes[0].set_aspect("equal")
    axes[0].invert_yaxis()
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Each model's predictions
    for j, (mode, ftype, name, rdir) in enumerate(available):
        model = _load_model(rdir, data, config, mode, ftype)
        logits, _ = model(data.x, data.edge_index)
        preds = logits.argmax(dim=-1).cpu().numpy()

        # Load metrics
        metrics_path = os.path.join(rdir, "metrics.json")
        ari_str = ""
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            ari_str = f"\nARI={metrics.get('test_ari', 0):.3f}"

        ax = axes[j + 1]
        for i in range(n_classes):
            mask = preds == i
            if mask.sum() > 0:
                ax.scatter(coords[mask, 0], coords[mask, 1], c=DOMAIN_COLORS[i],
                           s=8, alpha=0.8, edgecolors="none")
        ax.set_title(f"{name}{ari_str}", fontsize=13, fontweight="bold")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])

    handles = [mpatches.Patch(color=DOMAIN_COLORS[i], label=DOMAIN_NAMES[i])
               for i in range(n_classes)]
    fig.legend(handles=handles, loc="lower center", ncol=n_classes,
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle(f"Spatial Domain Predictions — Slice {sample_id}",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    save_path = os.path.join(fig_dir, "06_spatial_predictions.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


@torch.no_grad()
def viz_cross_attention_weights(sample_id, config, fig_dir):
    """
    Visualize cross-attention weights: for a few example spots, show which
    spatial neighbors they attend to most strongly.
    """
    from src.data.dataset import load_dlpfc_data

    data = load_dlpfc_data(sample_id, seed=config["training"]["seed"])
    coords = data.pos.numpy()
    labels = data.y.numpy()
    n_classes = data.n_classes

    result_dir = f"results/full_cross_attention_{sample_id}"
    if not os.path.exists(os.path.join(result_dir, "model.pt")):
        print(f"  Skipping attention viz (no cross-attention checkpoint for {sample_id})")
        return

    model = _load_model(result_dir, data, config, "full", "cross_attention")

    # Extract expression and spatial embeddings
    expr_embed = model.expr_encoder(data.x)
    spatial_embed = model.spatial_encoder(expr_embed, data.edge_index)

    # Get attention weights from the first cross-attention layer
    fusion = model.fusion
    edge_index = data.edge_index
    N = expr_embed.shape[0]
    device = expr_embed.device

    neighbor_idx, attn_mask = fusion._build_neighbor_index(edge_index, N, device)

    # Run first layer's cross-attention with need_weights=True
    kv = spatial_embed[neighbor_idx]  # (N, max_neighbors, d)
    q = expr_embed.unsqueeze(1)       # (N, 1, d)
    _, attn_weights = fusion.cross_attns[0](q, kv, kv,
                                             key_padding_mask=attn_mask,
                                             need_weights=True,
                                             average_attn_weights=True)
    # attn_weights shape: (N, 1, max_neighbors)
    attn_weights = attn_weights.squeeze(1).cpu().numpy()  # (N, max_neighbors)
    neighbor_idx_np = neighbor_idx.cpu().numpy()
    attn_mask_np = attn_mask.cpu().numpy()

    # Pick example spots: one from each layer at boundary regions
    # Find spots near domain boundaries (neighbors with different labels)
    boundary_spots = []
    src, tgt = edge_index[0].numpy(), edge_index[1].numpy()
    for spot in range(N):
        neighbors = src[tgt == spot]
        if len(neighbors) > 0 and len(set(labels[neighbors])) > 1:
            boundary_spots.append(spot)

    # Pick one per class from boundary spots
    example_spots = []
    for cls in range(n_classes):
        cls_boundary = [s for s in boundary_spots if labels[s] == cls]
        if cls_boundary:
            example_spots.append(cls_boundary[len(cls_boundary) // 2])
    example_spots = example_spots[:6]  # max 6 panels

    if not example_spots:
        print("  Skipping attention viz (no boundary spots found)")
        return

    n_examples = len(example_spots)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for panel_idx, spot in enumerate(example_spots):
        if panel_idx >= 6:
            break
        ax = axes[panel_idx]

        # Get this spot's neighbors and attention weights
        nbr_ids = neighbor_idx_np[spot]
        nbr_mask = ~attn_mask_np[spot]
        valid_nbrs = nbr_ids[nbr_mask]
        valid_weights = attn_weights[spot][nbr_mask]

        # Plot local neighborhood (spots within a radius)
        spot_coord = coords[spot]
        dists = np.sqrt(np.sum((coords - spot_coord) ** 2, axis=1))
        radius = np.sort(dists)[min(50, len(dists) - 1)]
        local_mask = dists <= radius

        # Plot local spots (faded)
        for i in range(n_classes):
            cls_mask = (labels == i) & local_mask
            if cls_mask.sum() > 0:
                ax.scatter(coords[cls_mask, 0], coords[cls_mask, 1],
                           c=DOMAIN_COLORS[i], s=30, alpha=0.2, edgecolors="none")

        # Draw attention edges with width proportional to weight
        max_w = valid_weights.max() if len(valid_weights) > 0 else 1
        for nbr, w in zip(valid_nbrs, valid_weights):
            linewidth = 1 + 4 * (w / max_w)
            alpha = 0.3 + 0.7 * (w / max_w)
            ax.plot([spot_coord[0], coords[nbr, 0]],
                    [spot_coord[1], coords[nbr, 1]],
                    color="#333333", linewidth=linewidth, alpha=alpha, zorder=3)

        # Highlight the query spot
        ax.scatter([spot_coord[0]], [spot_coord[1]],
                   c=DOMAIN_COLORS[labels[spot]], s=150, edgecolors="black",
                   linewidths=2, zorder=5, marker="*")

        # Highlight neighbor spots with size proportional to attention
        for nbr, w in zip(valid_nbrs, valid_weights):
            size = 40 + 160 * (w / max_w)
            ax.scatter([coords[nbr, 0]], [coords[nbr, 1]],
                       c=DOMAIN_COLORS[labels[nbr]], s=size,
                       edgecolors="black", linewidths=1, zorder=4)

        ax.set_title(f"Spot {spot} ({DOMAIN_NAMES[labels[spot]]})",
                     fontsize=11, fontweight="bold")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused panels
    for panel_idx in range(n_examples, 6):
        axes[panel_idx].set_visible(False)

    plt.suptitle(f"Cross-Attention Weights at Domain Boundaries — Slice {sample_id}\n"
                 "(star = query spot, circle size = attention weight, line width = attention strength)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_path = os.path.join(fig_dir, "07_cross_attention_weights.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


@torch.no_grad()
def viz_gated_fusion_values(sample_id, config, fig_dir):
    """
    For the Gated Fusion model, visualize gate values on the tissue:
    how much each spot relies on expression (gate~1) vs spatial (gate~0).
    """
    from src.data.dataset import load_dlpfc_data

    data = load_dlpfc_data(sample_id, seed=config["training"]["seed"])
    coords = data.pos.numpy()
    labels = data.y.numpy()
    n_classes = data.n_classes

    result_dir = f"results/full_gated_{sample_id}"
    if not os.path.exists(os.path.join(result_dir, "model.pt")):
        print(f"  Skipping gated fusion viz (no gated checkpoint for {sample_id})")
        return

    model = _load_model(result_dir, data, config, "full", "gated")

    # Extract gate values
    expr_embed = model.expr_encoder(data.x)
    spatial_embed = model.spatial_encoder(expr_embed, data.edge_index)

    gate_values = model.fusion.gate(
        torch.cat([expr_embed, spatial_embed], dim=-1)
    ).cpu().numpy()  # (N, embed_dim)

    # Average gate across embedding dimensions → single value per spot
    mean_gate = gate_values.mean(axis=1)  # (N,)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Left: ground truth
    for i in range(n_classes):
        mask = labels == i
        if mask.sum() > 0:
            axes[0].scatter(coords[mask, 0], coords[mask, 1], c=DOMAIN_COLORS[i],
                            label=DOMAIN_NAMES[i], s=10, alpha=0.8, edgecolors="none")
    axes[0].set_title("Ground Truth", fontsize=13, fontweight="bold")
    axes[0].set_aspect("equal")
    axes[0].invert_yaxis()
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9, markerscale=2)

    # Middle: gate value heatmap
    sc = axes[1].scatter(coords[:, 0], coords[:, 1], c=mean_gate, cmap="RdYlBu_r",
                         s=10, alpha=0.8, edgecolors="none", vmin=0.3, vmax=0.7)
    axes[1].set_title("Gate Value\n(1=expression, 0=spatial)", fontsize=13, fontweight="bold")
    axes[1].set_aspect("equal")
    axes[1].invert_yaxis()
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    plt.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04)

    # Right: histogram of gate values per domain
    for i in range(n_classes):
        mask = labels == i
        if mask.sum() > 0:
            axes[2].hist(mean_gate[mask], bins=30, color=DOMAIN_COLORS[i],
                         alpha=0.5, label=DOMAIN_NAMES[i], density=True)
    axes[2].set_title("Gate Distribution by Domain", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Mean gate value")
    axes[2].set_ylabel("Density")
    axes[2].legend(fontsize=9)
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)

    plt.suptitle(f"Gated Fusion Analysis — Slice {sample_id}",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(fig_dir, "08_gated_fusion_values.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


@torch.no_grad()
def viz_ablation_embeddings(sample_id, config, fig_dir):
    """
    t-SNE comparison of embeddings from each ablation model side by side.
    Shows how adding components progressively improves cluster separation.
    """
    from src.data.dataset import load_dlpfc_data

    data = load_dlpfc_data(sample_id, seed=config["training"]["seed"])
    labels = data.y.numpy()
    n_classes = data.n_classes

    ablations = [
        ("expr_only", "gated", "MLP Only\n(no spatial)"),
        ("gat_only", "gated", "GAT Only\n(no expression branch)"),
        ("full", "gated", "MLP+GAT\n(Gated Fusion)"),
        ("full", "cross_attention", "MLP+GAT\n(Cross-Attention)"),
    ]

    embeddings = {}
    for mode, ftype, name in ablations:
        rdir = f"results/{mode}_{ftype}_{sample_id}"
        if not os.path.exists(os.path.join(rdir, "model.pt")):
            continue
        model = _load_model(rdir, data, config, mode, ftype)
        _, emb = model(data.x, data.edge_index)
        embeddings[name] = emb.cpu().numpy()

    if len(embeddings) < 2:
        print("  Skipping ablation embeddings (need at least 2 models)")
        return

    n = len(embeddings)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (name, emb) in zip(axes, embeddings.items()):
        print(f"  Computing t-SNE for {name.split(chr(10))[0]}...")
        tsne = _tsne_2d(emb)
        _plot_tsne(tsne, labels, n_classes, name, ax)

    handles = [mpatches.Patch(color=DOMAIN_COLORS[i], label=DOMAIN_NAMES[i])
               for i in range(n_classes)]
    fig.legend(handles=handles, loc="lower center", ncol=n_classes,
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle(f"Ablation Study: Embedding Quality — Slice {sample_id}",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    save_path = os.path.join(fig_dir, "09_ablation_embeddings.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate pipeline + model visualizations")
    parser.add_argument("--sample_id", type=str, default="151673")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--skip_models", action="store_true",
                        help="Only generate data pipeline figures (no model loading)")
    parser.add_argument("--skip_data", action="store_true",
                        help="Only generate model figures (skip data pipeline)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    fig_dir = f"results/figures/pipeline_{args.sample_id}"
    os.makedirs(fig_dir, exist_ok=True)

    print(f"=" * 60)
    print(f"Generating visualizations for slice {args.sample_id}")
    print(f"Output: {fig_dir}/")
    print(f"=" * 60)

    if not args.skip_data:
        print(f"\n--- Part 1: Data Pipeline ---")
        print("  [1/4] Preprocessing heatmaps...")
        viz_preprocessing_steps(args.sample_id, "data/raw", fig_dir)
        print("  [2/4] Expression distributions...")
        viz_expression_distribution(args.sample_id, "data/raw", fig_dir)
        print("  [3/4] Spatial graph...")
        viz_spatial_graph(args.sample_id, "data/processed", fig_dir)
        print("  [4/4] Label distribution...")
        viz_label_distribution(args.sample_id, "data/processed", fig_dir)

    if not args.skip_models:
        print(f"\n--- Part 2: Model Internals ---")
        print("  [5/9] Embedding progression (MLP → GAT → Fused)...")
        viz_embedding_progression(args.sample_id, config, fig_dir)
        print("  [6/9] Spatial predictions comparison...")
        viz_spatial_predictions(args.sample_id, config, fig_dir)
        print("  [7/9] Cross-attention weights...")
        viz_cross_attention_weights(args.sample_id, config, fig_dir)
        print("  [8/9] Gated fusion gate values...")
        viz_gated_fusion_values(args.sample_id, config, fig_dir)
        print("  [9/9] Ablation embeddings comparison...")
        viz_ablation_embeddings(args.sample_id, config, fig_dir)

    print(f"\n{'=' * 60}")
    print(f"All figures saved to {fig_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
