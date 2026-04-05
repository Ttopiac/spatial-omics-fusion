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
    ("scgpt_only", "gated"),
    ("scgpt", "gated"),
    ("scgpt", "cross_attention"),
    ("geneformer_only", "gated"),
    ("geneformer", "gated"),
    ("geneformer", "cross_attention"),
    ("scgpt_brain_only", "gated"),
    ("scgpt_brain", "gated"),
    ("scgpt_brain", "cross_attention"),
    ("scgpt_finetune", "cross_attention"),
    ("geneformer_finetune", "cross_attention"),
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

    # Load data (with optional embeddings)
    use_scgpt = mode in ("scgpt", "scgpt_only")
    use_image = mode in ("multimodal", "img_expr")
    use_geneformer = mode in ("geneformer", "geneformer_only")
    use_scgpt_brain = mode in ("scgpt_brain", "scgpt_brain_only")
    use_scgpt_tokens = mode == "scgpt_finetune"
    use_geneformer_tokens = mode == "geneformer_finetune"
    data = load_dlpfc_data(
        sample_id,
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
        seed=seed,
        load_scgpt=use_scgpt,
        load_image=use_image,
        load_geneformer=use_geneformer,
        load_scgpt_brain=use_scgpt_brain,
        load_scgpt_tokens=use_scgpt_tokens,
        load_geneformer_tokens=use_geneformer_tokens,
    )

    # Create model
    common_kwargs = dict(
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
    )

    if mode == "scgpt_finetune":
        from src.models.finetune_model import ScGPTFinetune
        model = ScGPTFinetune(model_dir="data/scgpt_human", **common_kwargs)
    elif mode == "geneformer_finetune":
        from src.models.finetune_model import GeneformerFinetune
        model = GeneformerFinetune(**common_kwargs)
    else:
        model = SpatialOmicsFusion(**common_kwargs, mode=mode)

    # Train
    foundation_lr = 1e-5 if mode.endswith("_finetune") else None
    trainer = Trainer(
        model=model, data=data, device=device,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        epochs=cfg["training"]["epochs"],
        patience=cfg["training"]["patience"],
        foundation_lr=foundation_lr,
    )
    trainer.fit()
    test_metrics = trainer.test(extended=True)

    # Save
    result_dir = os.path.join("results", f"{mode}_{fusion_type}_{sample_id}")
    os.makedirs(result_dir, exist_ok=True)
    # Convert numpy types to native Python for JSON serialization
    serializable_metrics = {k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer,)) else v
                            for k, v in test_metrics.items()}
    with open(os.path.join(result_dir, "metrics.json"), "w") as f:
        json.dump({"test": serializable_metrics, "best_val_ari": float(trainer.best_val_ari)}, f, indent=2)
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
        print(f"  ARI:      {np.mean(aris):.4f} +/- {np.std(aris):.4f}")
        print(f"  NMI:      {np.mean(nmis):.4f} +/- {np.std(nmis):.4f}")
        print(f"  Acc:      {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
        # Extended metrics (may not be present for all experiments)
        for key in ["top2_accuracy", "interior_accuracy", "boundary_accuracy", "log_loss"]:
            vals = [m.get(key) for m in all_results[exp_name].values() if m.get(key) is not None]
            if vals:
                print(f"  {key:16s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    # Final summary table
    total_time = time.time() - total_start
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS (across {len(sample_ids)} slices)")
    print(f"{'='*80}")
    print(f"{'Experiment':<30} {'ARI':>12} {'NMI':>12} {'Acc':>12} {'Top-2':>12} {'Interior':>12} {'Boundary':>12}")
    print("-" * 106)

    for exp_name, slice_results in all_results.items():
        aris = [m["ari"] for m in slice_results.values()]
        nmis = [m["nmi"] for m in slice_results.values()]
        accs = [m["accuracy"] for m in slice_results.values()]
        row = (f"{exp_name:<30} "
               f"{np.mean(aris):.4f}+/-{np.std(aris):.4f} "
               f"{np.mean(nmis):.4f}+/-{np.std(nmis):.4f} "
               f"{np.mean(accs):.4f}+/-{np.std(accs):.4f}")
        for key in ["top2_accuracy", "interior_accuracy", "boundary_accuracy"]:
            vals = [m.get(key) for m in slice_results.values() if m.get(key) is not None]
            if vals:
                row += f" {np.mean(vals):.4f}+/-{np.std(vals):.4f}"
        print(row)

    print(f"\nTotal time: {total_time:.1f}s")

    # Save summary (convert numpy types for JSON)
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        return obj

    summary_path = "results/benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f"Saved to {summary_path}")


if __name__ == "__main__":
    main()
