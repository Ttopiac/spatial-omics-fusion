"""
Run benchmark across all 12 DLPFC slices.

Usage:
    python scripts/run_benchmark.py                              # all modes, all slices
    python scripts/run_benchmark.py --mode full --fusion_type cross_attention  # single config
    python scripts/run_benchmark.py --device cuda                # for 3090

This script:
1. Preprocesses each slice (if not already done)
2. Trains the specified model(s) on each slice
3. Collects results into a summary table
"""
import argparse
import json
import os
import sys
import time

import torch
import yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocess import preprocess_slice
from src.data.dataset import load_dlpfc_data
from src.models.model import SpatialOmicsFusion
from src.training.trainer import Trainer
from src.utils import get_device

ALL_SAMPLE_IDS = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676",
]

EXPERIMENTS = [
    ("expr_only", "gated"),
    ("gat_only", "gated"),
    ("concat", "gated"),
    ("full", "gated"),
    ("full", "cross_attention"),
]


def run_single(sample_id, mode, fusion_type, cfg, device):
    """Train and evaluate a single experiment."""
    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Preprocess if needed
    processed_dir = os.path.join("data/processed", sample_id)
    if not os.path.exists(os.path.join(processed_dir, "expression.pt")):
        preprocess_slice(
            sample_id, "data/raw", "data/processed",
            n_top_genes=cfg["data"]["n_top_genes"],
            knn_k=cfg["data"]["knn_k"],
        )

    # Load data
    data = load_dlpfc_data(
        sample_id,
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
        seed=seed,
    )

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

    # Train
    trainer = Trainer(
        model=model, data=data, device=device,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        epochs=cfg["training"]["epochs"],
        patience=cfg["training"]["patience"],
    )
    trainer.fit()
    test_metrics = trainer.test()

    # Save
    result_dir = os.path.join("results", f"{mode}_{fusion_type}_{sample_id}")
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, "metrics.json"), "w") as f:
        json.dump({"test": test_metrics, "best_val_ari": trainer.best_val_ari}, f, indent=2)
    torch.save(trainer.best_state, os.path.join(result_dir, "model.pt"))

    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--fusion_type", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--sample_ids", nargs="+", default=None,
                        help="Specific sample IDs to run (default: all 12)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device(args.device)
    sample_ids = args.sample_ids or ALL_SAMPLE_IDS

    # Determine which experiments to run
    if args.mode and args.fusion_type:
        experiments = [(args.mode, args.fusion_type)]
    elif args.mode:
        experiments = [(args.mode, "gated")]
    else:
        experiments = EXPERIMENTS

    print(f"Benchmark: {len(experiments)} experiment(s) x {len(sample_ids)} slice(s) on {device}")
    print("=" * 80)

    all_results = {}
    total_start = time.time()

    for mode, fusion_type in experiments:
        exp_name = f"{mode}_{fusion_type}"
        all_results[exp_name] = {}
        aris, nmis, accs = [], [], []

        print(f"\n{'='*80}")
        print(f"Experiment: {exp_name}")
        print(f"{'='*80}")

        for sid in sample_ids:
            print(f"\n--- Slice {sid} ---")
            metrics = run_single(sid, mode, fusion_type, cfg, device)
            all_results[exp_name][sid] = metrics
            aris.append(metrics["ari"])
            nmis.append(metrics["nmi"])
            accs.append(metrics["accuracy"])

        # Summary for this experiment
        print(f"\n--- {exp_name} Summary ---")
        print(f"  ARI:  {np.mean(aris):.4f} +/- {np.std(aris):.4f}")
        print(f"  NMI:  {np.mean(nmis):.4f} +/- {np.std(nmis):.4f}")
        print(f"  Acc:  {np.mean(accs):.4f} +/- {np.std(accs):.4f}")

    # Final summary table
    total_time = time.time() - total_start
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS (across {len(sample_ids)} slices)")
    print(f"{'='*80}")
    print(f"{'Experiment':<30} {'ARI':>12} {'NMI':>12} {'Acc':>12}")
    print("-" * 70)

    for exp_name, slice_results in all_results.items():
        aris = [m["ari"] for m in slice_results.values()]
        nmis = [m["nmi"] for m in slice_results.values()]
        accs = [m["accuracy"] for m in slice_results.values()]
        print(f"{exp_name:<30} "
              f"{np.mean(aris):.4f}+/-{np.std(aris):.4f} "
              f"{np.mean(nmis):.4f}+/-{np.std(nmis):.4f} "
              f"{np.mean(accs):.4f}+/-{np.std(accs):.4f}")

    print(f"\nTotal time: {total_time:.1f}s")

    # Save summary
    summary_path = "results/benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved to {summary_path}")


if __name__ == "__main__":
    main()
