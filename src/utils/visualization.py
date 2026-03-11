"""
Visualization utilities for spatial transcriptomics results.

Generates:
1. Spatial domain plots — scatter plots of spots colored by domain label
2. UMAP embeddings — 2D projection of learned representations
3. Side-by-side comparisons — ground truth vs predicted
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

# Try UMAP, fall back to t-SNE
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# Color palette for 7 spatial domains (Layer1-6 + WM)
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


def plot_spatial_domains(coords, labels, title="Spatial Domains", ax=None,
                         label_names=None, colors=None):
    """
    Scatter plot of spots colored by domain label.

    This shows the physical layout of the tissue — each dot is a spot,
    and the color indicates which brain layer it belongs to.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    colors = colors or DOMAIN_COLORS
    label_names = label_names or DOMAIN_NAMES

    coords = np.array(coords)
    labels = np.array(labels)

    for i in range(len(label_names)):
        mask = labels == i
        if mask.sum() > 0:
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=colors[i], label=label_names[i],
                       s=8, alpha=0.8, edgecolors="none")

    ax.set_title(title, fontsize=14)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # tissue images are typically y-inverted
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5),
              fontsize=9, markerscale=2)
    return ax


def plot_spatial_comparison(coords, true_labels, pred_labels, save_path=None):
    """Side-by-side: ground truth vs predicted spatial domains."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    plot_spatial_domains(coords, true_labels, title="Ground Truth", ax=axes[0])
    plot_spatial_domains(coords, pred_labels, title="Predicted", ax=axes[1])

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_embeddings_2d(embeddings, labels, title="Embeddings", ax=None,
                       label_names=None, colors=None, method="tsne"):
    """
    2D projection (UMAP or t-SNE) of learned embeddings.

    If clusters are well-separated in 2D, the model has learned good representations.
    Comparing embeddings from different model variants shows the benefit of fusion.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    colors = colors or DOMAIN_COLORS
    label_names = label_names or DOMAIN_NAMES

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    # Dimensionality reduction
    if method == "umap" and HAS_UMAP:
        reducer = UMAP(n_components=2, random_state=42)
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)

    coords_2d = reducer.fit_transform(embeddings)

    for i in range(len(label_names)):
        mask = labels == i
        if mask.sum() > 0:
            ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                       c=colors[i], label=label_names[i],
                       s=5, alpha=0.7, edgecolors="none")

    ax.set_title(title, fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5),
              fontsize=9, markerscale=2)
    return ax


def plot_embedding_comparison(embeddings_dict, labels, save_path=None, method="tsne"):
    """
    Compare embeddings from different model variants side by side.

    Args:
        embeddings_dict: {"Model Name": embeddings_array, ...}
        labels: ground truth labels
    """
    n = len(embeddings_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (name, emb) in zip(axes, embeddings_dict.items()):
        plot_embeddings_2d(emb, labels, title=name, ax=ax, method=method)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig
