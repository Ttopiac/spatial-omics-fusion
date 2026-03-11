"""
Generate visualizations for trained models.

Usage:
    python scripts/visualize.py --sample_id 151673
"""
import argparse
import json
import os
import sys

import torch
import yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import load_dlpfc_data
from src.models.model import SpatialOmicsFusion
from src.utils import get_device
from src.utils.visualization import (
    plot_spatial_comparison,
    plot_embedding_comparison,
)


def load_trained_model(result_dir, data, config, mode, fusion_type):
    """Load a trained model from checkpoint."""
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
        dropout=0.0,  # no dropout at inference
        mode=mode,
    )
    state = torch.load(os.path.join(result_dir, "model.pt"), map_location="cpu",
                       weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def get_predictions_and_embeddings(model, data):
    """Run inference and return predictions + embeddings."""
    logits, embeddings = model(data.x, data.edge_index)
    preds = logits.argmax(dim=-1).cpu().numpy()
    embeddings = embeddings.cpu().numpy()
    return preds, embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_id", type=str, default="151673")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data = load_dlpfc_data(args.sample_id, seed=cfg["training"]["seed"])
    coords = data.pos.numpy()
    true_labels = data.y.numpy()

    fig_dir = f"results/figures/{args.sample_id}"
    os.makedirs(fig_dir, exist_ok=True)

    # Define models to visualize
    experiments = [
        ("expr_only", "gated", "MLP Only"),
        ("gat_only", "gated", "GAT Only"),
        ("concat", "gated", "Concat Fusion"),
        ("full", "gated", "Gated Fusion"),
        ("full", "cross_attention", "Cross-Attention"),
    ]

    embeddings_dict = {}
    for mode, fusion_type, display_name in experiments:
        result_dir = f"results/{mode}_{fusion_type}_{args.sample_id}"
        if not os.path.exists(os.path.join(result_dir, "model.pt")):
            print(f"  Skipping {display_name} (no checkpoint found)")
            continue

        print(f"Loading {display_name} ...")
        model = load_trained_model(result_dir, data, cfg, mode, fusion_type)
        preds, emb = get_predictions_and_embeddings(model, data)

        # Spatial comparison plot
        plot_spatial_comparison(
            coords, true_labels, preds,
            save_path=os.path.join(fig_dir, f"spatial_{mode}_{fusion_type}.png"),
        )

        embeddings_dict[display_name] = emb

    # Embedding comparison (all models side by side)
    if len(embeddings_dict) >= 2:
        print("Generating embedding comparison (t-SNE) ...")
        plot_embedding_comparison(
            embeddings_dict, true_labels,
            save_path=os.path.join(fig_dir, "embedding_comparison_tsne.png"),
            method="tsne",
        )

    print(f"\nAll figures saved to {fig_dir}/")


if __name__ == "__main__":
    main()
