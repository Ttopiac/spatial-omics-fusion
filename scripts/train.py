"""
Main training script.

Usage:
    python scripts/train.py                          # default config, full model
    python scripts/train.py --mode expr_only         # MLP-only baseline
    python scripts/train.py --mode gat_only          # GAT-only baseline
    python scripts/train.py --mode concat            # concatenation baseline
    python scripts/train.py --mode full              # our model (gated fusion)
    python scripts/train.py --fusion_type cross_attention  # cross-attention variant
"""
import argparse
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import load_dlpfc_data
from src.models.model import SpatialOmicsFusion
from src.training.trainer import Trainer
from src.utils import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--sample_id", type=str, default=None)
    parser.add_argument("--mode", type=str, default=None,
                        choices=["expr_only", "gat_only", "concat", "full", "multimodal", "img_expr"])
    parser.add_argument("--fusion_type", type=str, default=None,
                        choices=["gated", "cross_attention"])
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Override config with CLI args
    sample_id = args.sample_id or cfg["data"]["sample_id"]
    mode = args.mode or "full"
    fusion_type = args.fusion_type or cfg["model"]["fusion"]["type"]
    device = get_device(args.device or cfg["device"])

    print(f"=== Training: mode={mode}, fusion={fusion_type}, sample={sample_id}, device={device} ===\n")

    # Set seed
    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Load data
    data = load_dlpfc_data(
        sample_id,
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
        seed=seed,
        load_image=(mode in ("multimodal", "img_expr")),
    )
    print(f"Data: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges, {data.n_classes} classes")
    print(f"Split: {data.train_mask.sum()} train / {data.val_mask.sum()} val / {data.test_mask.sum()} test\n")

    # Create model
    model = SpatialOmicsFusion(
        n_genes=data.x.shape[1],
        n_classes=data.n_classes,
        embed_dim=cfg["model"]["embed_dim"],
        hidden_dim=cfg["model"]["expression_encoder"]["hidden_dim"],
        expr_layers=cfg["model"]["expression_encoder"]["n_layers"],
        gat_heads=cfg["model"]["spatial_encoder"]["n_heads"],
        gat_layers=cfg["model"]["spatial_encoder"]["n_layers"],
        fusion_type=fusion_type,
        fusion_heads=cfg["model"]["fusion"]["n_heads"],
        fusion_layers=cfg["model"]["fusion"]["n_layers"],
        dropout=cfg["model"]["expression_encoder"]["dropout"],
        mode=mode,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters\n")

    # Train
    trainer = Trainer(
        model=model, data=data, device=device,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        epochs=cfg["training"]["epochs"],
        patience=cfg["training"]["patience"],
    )
    history = trainer.fit()

    # Test
    print()
    test_metrics = trainer.test()

    # Save results
    result_dir = os.path.join("results", f"{mode}_{fusion_type}_{sample_id}")
    os.makedirs(result_dir, exist_ok=True)

    with open(os.path.join(result_dir, "metrics.json"), "w") as f:
        json.dump({"test": test_metrics, "best_val_ari": trainer.best_val_ari}, f, indent=2)

    torch.save(trainer.best_state, os.path.join(result_dir, "model.pt"))
    print(f"\nResults saved to {result_dir}/")


if __name__ == "__main__":
    main()
