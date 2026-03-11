"""
Preprocessing pipeline for DLPFC spatial transcriptomics data.

What each step does (in ML terms):
1. Load the raw data (a matrix of spots × genes, plus coordinates)
2. Filter out rarely-seen genes (like removing rare tokens)
3. Normalize so each spot has the same total "gene budget" (like TF normalization in NLP)
4. Log-transform to reduce skewness (gene expression is very right-skewed)
5. Select top 3000 most variable genes (feature selection — keeps informative features)
6. Scale to zero mean, unit variance (standard ML preprocessing)
7. Build a spatial KNN graph from (x,y) coordinates (connect nearby spots)
8. Save everything as PyTorch tensors
"""
import argparse
import json
import os

import numpy as np
import scanpy as sc
import squidpy as sq
import torch
from scipy.sparse import issparse


def preprocess_slice(sample_id: str, raw_dir: str, output_dir: str,
                     n_top_genes: int = 3000, knn_k: int = 6,
                     label_col: str = "sce.layer_guess") -> dict:
    """
    Preprocess a single DLPFC slice and save as PyTorch tensors.

    Returns dict with metadata about the processed data.
    """
    # --- Step 1: Load raw data ---
    raw_path = os.path.join(raw_dir, f"{sample_id}.h5ad")
    print(f"  [{sample_id}] Loading {raw_path}")
    adata = sc.read_h5ad(raw_path)
    print(f"  [{sample_id}] Raw shape: {adata.shape[0]} spots × {adata.shape[1]} genes")

    # --- Step 2: Drop spots with no ground truth label ---
    n_before = adata.n_obs
    adata = adata[~adata.obs[label_col].isna()].copy()
    n_dropped = n_before - adata.n_obs
    print(f"  [{sample_id}] Dropped {n_dropped} unlabeled spots → {adata.n_obs} remaining")

    # --- Step 3: Filter genes seen in very few spots ---
    # Why: genes expressed in <3 spots are noise, not signal
    sc.pp.filter_genes(adata, min_cells=3)
    print(f"  [{sample_id}] After gene filter: {adata.n_vars} genes")

    # --- Step 4: Normalize ---
    # Each spot originally has different total counts (some spots captured more mRNA).
    # We normalize so every spot sums to 10,000 — like making word frequencies comparable.
    sc.pp.normalize_total(adata, target_sum=1e4)

    # --- Step 5: Log transform ---
    # Gene expression follows a power-law. log(1+x) compresses the range.
    # After this, the data looks more Gaussian — better for neural networks.
    sc.pp.log1p(adata)

    # --- Step 6: Select top highly variable genes (HVGs) ---
    # Most of 33K genes are uninformative (expressed similarly everywhere).
    # HVGs are the ones that actually differ between cell types — our signal.
    # This is like feature selection: keep the top 3000 most informative features.
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata = adata[:, adata.var.highly_variable].copy()
    print(f"  [{sample_id}] After HVG selection: {adata.n_vars} genes")

    # --- Step 7: Scale to zero mean, unit variance ---
    # Standard ML preprocessing. Clip at ±10 to avoid outlier genes dominating.
    sc.pp.scale(adata, max_value=10)

    # --- Step 8: Build spatial KNN graph ---
    # Connect each spot to its k nearest spatial neighbors.
    # This is like building a graph where nearby spots are connected.
    # Our GNN will later use this graph to aggregate neighborhood info.
    sq.gr.spatial_neighbors(adata, n_neighs=knn_k, coord_type="generic")

    # Extract adjacency as edge_index (PyG format: shape [2, num_edges])
    adj = adata.obsp["spatial_connectivities"]
    rows, cols = adj.nonzero()
    edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long)
    print(f"  [{sample_id}] Spatial graph: {edge_index.shape[1]} edges (k={knn_k})")

    # --- Step 9: Extract tensors ---
    # Expression matrix
    expr = adata.X
    if issparse(expr):
        expr = expr.toarray()
    expression = torch.tensor(expr, dtype=torch.float32)

    # Spatial coordinates
    coordinates = torch.tensor(adata.obsm["spatial"].astype(np.float32))

    # Labels: convert string labels to integers
    label_series = adata.obs[label_col].astype("category")
    label_map = {cat: i for i, cat in enumerate(label_series.cat.categories)}
    labels = torch.tensor(label_series.cat.codes.values, dtype=torch.long)
    n_classes = len(label_map)
    print(f"  [{sample_id}] Classes ({n_classes}): {label_map}")

    # --- Step 10: Save everything ---
    out_dir = os.path.join(output_dir, sample_id)
    os.makedirs(out_dir, exist_ok=True)

    torch.save(expression, os.path.join(out_dir, "expression.pt"))
    torch.save(edge_index, os.path.join(out_dir, "edge_index.pt"))
    torch.save(labels, os.path.join(out_dir, "labels.pt"))
    torch.save(coordinates, os.path.join(out_dir, "coordinates.pt"))

    metadata = {
        "sample_id": sample_id,
        "n_spots": int(expression.shape[0]),
        "n_genes": int(expression.shape[1]),
        "n_edges": int(edge_index.shape[1]),
        "n_classes": n_classes,
        "label_map": label_map,
        "knn_k": knn_k,
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  [{sample_id}] Saved to {out_dir}/")
    print(f"  [{sample_id}] expression.pt: {expression.shape}")
    print(f"  [{sample_id}] edge_index.pt: {edge_index.shape}")
    print(f"  [{sample_id}] labels.pt:     {labels.shape} ({n_classes} classes)")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Preprocess DLPFC data")
    parser.add_argument("--sample_id", type=str, default="151673")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--n_top_genes", type=int, default=3000)
    parser.add_argument("--knn_k", type=int, default=6)
    args = parser.parse_args()

    ALL_SAMPLE_IDS = [
        "151507", "151508", "151509", "151510",
        "151669", "151670", "151671", "151672",
        "151673", "151674", "151675", "151676",
    ]

    if args.all:
        for sid in ALL_SAMPLE_IDS:
            preprocess_slice(sid, args.raw_dir, args.output_dir,
                             args.n_top_genes, args.knn_k)
    else:
        preprocess_slice(args.sample_id, args.raw_dir, args.output_dir,
                         args.n_top_genes, args.knn_k)


if __name__ == "__main__":
    main()
