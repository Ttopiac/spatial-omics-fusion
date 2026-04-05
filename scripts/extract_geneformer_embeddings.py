"""
Extract Geneformer embeddings for all DLPFC slices.

Usage:
    python scripts/extract_geneformer_embeddings.py
    python scripts/extract_geneformer_embeddings.py --sample_ids 151673
    python scripts/extract_geneformer_embeddings.py --batch_size 32

This script:
1. Loads the pretrained Geneformer V2-104M model from Hugging Face
2. For each DLPFC slice, rank-encodes genes and extracts 768-dim cell embeddings
3. Saves embeddings to data/processed/{sample_id}/geneformer_embeddings.pt

Geneformer tokenization:
- Genes are identified by Ensembl IDs
- Expression values are normalized by gene median (from pretraining corpus)
- Genes are ranked by descending normalized expression
- Ranked gene tokens are fed to a BERT model
- Cell embedding = CLS token from 2nd-to-last hidden layer
"""

import argparse
import os
import pickle
import sys

import numpy as np
import torch
import scanpy as sc
from huggingface_hub import hf_hub_download
from transformers import BertForMaskedLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ALL_SAMPLE_IDS = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676",
]

REPO_ID = "ctheodoris/Geneformer"
MODEL_SUBFOLDER = "Geneformer-V2-104M"
CLS_TOKEN_ID = 2
EOS_TOKEN_ID = 3
PAD_TOKEN_ID = 0


def load_geneformer_model(device: torch.device):
    """Load pretrained Geneformer model and dictionaries from HuggingFace."""
    print("Loading Geneformer V2-104M from HuggingFace...")

    model = BertForMaskedLM.from_pretrained(
        REPO_ID,
        subfolder=MODEL_SUBFOLDER,
        output_hidden_states=True,
    )
    model.to(device)
    model.eval()
    print(f"Loaded Geneformer: {sum(p.numel() for p in model.parameters()):,} params")

    # Load token dictionary (Ensembl ID -> token ID)
    token_dict_path = hf_hub_download(REPO_ID, "geneformer/token_dictionary_gc104M.pkl")
    with open(token_dict_path, "rb") as f:
        token_dict = pickle.load(f)

    # Load gene median dictionary (Ensembl ID -> median expression)
    median_dict_path = hf_hub_download(REPO_ID, "geneformer/gene_median_dictionary_gc104M.pkl")
    with open(median_dict_path, "rb") as f:
        median_dict = pickle.load(f)

    print(f"Token vocab size: {len(token_dict)}, Gene medians: {len(median_dict)}")
    return model, token_dict, median_dict


def tokenize_cells(adata, token_dict, median_dict, max_seq_len=2048):
    """
    Tokenize cells for Geneformer using rank-value encoding.

    For each cell:
    1. Normalize total counts to 10k, then log1p
    2. For each gene with an Ensembl ID in both token_dict and median_dict,
       compute normalized expression = expression / gene_median
    3. Rank genes by descending normalized expression
    4. Convert to token IDs, prepend <cls>, append <eos>

    Returns:
        input_ids: (N, max_len) padded token IDs
        attention_mask: (N, max_len) 1 for real tokens, 0 for padding
    """
    adata_pp = adata.copy()
    sc.pp.normalize_total(adata_pp, target_sum=1e4)
    sc.pp.log1p(adata_pp)

    # Map var_names to ensembl IDs
    gene_names = list(adata_pp.var_names)
    ensembl_ids = list(adata_pp.var["gene_ids"]) if "gene_ids" in adata_pp.var.columns else []

    if not ensembl_ids:
        raise ValueError("No Ensembl IDs found in adata.var['gene_ids']")

    # Find genes that are in both token_dict and median_dict
    valid_indices = []
    valid_ensembl = []
    valid_token_ids = []
    valid_medians = []

    for i, ens_id in enumerate(ensembl_ids):
        if ens_id in token_dict and ens_id in median_dict:
            valid_indices.append(i)
            valid_ensembl.append(ens_id)
            valid_token_ids.append(token_dict[ens_id])
            valid_medians.append(median_dict[ens_id])

    print(f"  Genes in Geneformer vocab: {len(valid_indices)}/{len(gene_names)}")

    valid_token_ids = np.array(valid_token_ids)
    valid_medians = np.array(valid_medians)

    # Get expression matrix for valid genes
    expr = adata_pp.X if not hasattr(adata_pp.X, "toarray") else adata_pp.X.toarray()
    expr_valid = expr[:, valid_indices]  # (N, n_valid_genes)

    n_cells = expr_valid.shape[0]
    # +2 for <cls> and <eos>
    max_len = min(len(valid_indices) + 2, max_seq_len)

    input_ids = torch.full((n_cells, max_len), PAD_TOKEN_ID, dtype=torch.long)
    attention_mask = torch.zeros((n_cells, max_len), dtype=torch.long)

    for i in range(n_cells):
        cell_expr = expr_valid[i]

        # Normalize by gene median
        nonzero_mask = cell_expr > 0
        if not nonzero_mask.any():
            # Empty cell - just cls + eos
            input_ids[i, 0] = CLS_TOKEN_ID
            input_ids[i, 1] = EOS_TOKEN_ID
            attention_mask[i, :2] = 1
            continue

        # Rank genes by descending normalized expression
        norm_expr = np.zeros_like(cell_expr)
        norm_expr[nonzero_mask] = cell_expr[nonzero_mask] / (valid_medians[nonzero_mask] + 1e-6)
        ranked_idx = np.argsort(-norm_expr)

        # Keep only expressed genes
        n_expressed = nonzero_mask.sum()
        ranked_idx = ranked_idx[:n_expressed]

        # Truncate to fit in max_len (with room for cls + eos)
        n_tokens = min(len(ranked_idx), max_len - 2)
        ranked_idx = ranked_idx[:n_tokens]

        # Build token sequence: <cls> + ranked genes + <eos>
        tokens = np.empty(n_tokens + 2, dtype=np.int64)
        tokens[0] = CLS_TOKEN_ID
        tokens[1 : n_tokens + 1] = valid_token_ids[ranked_idx]
        tokens[n_tokens + 1] = EOS_TOKEN_ID

        seq_len = n_tokens + 2
        input_ids[i, :seq_len] = torch.tensor(tokens)
        attention_mask[i, :seq_len] = 1

    return input_ids, attention_mask


@torch.no_grad()
def extract_embeddings(model, input_ids, attention_mask, device, batch_size=4):
    """Extract cell embeddings from Geneformer (CLS token, 2nd-to-last layer)."""
    n_cells = input_ids.shape[0]
    all_embeddings = []
    use_fp16 = device.type == "cuda"

    for start in range(0, n_cells, batch_size):
        end = min(start + batch_size, n_cells)
        batch_ids = input_ids[start:end].to(device)
        batch_mask = attention_mask[start:end].to(device)

        with torch.autocast(device_type=device.type, enabled=use_fp16):
            outputs = model(input_ids=batch_ids, attention_mask=batch_mask)
        # hidden_states: tuple of (n_layers+1) tensors of shape (batch, seq, hidden)
        hidden_states = outputs.hidden_states
        # CLS token from 2nd-to-last layer
        cell_embs = hidden_states[-2][:, 0, :].float()  # (batch, 768)
        all_embeddings.append(cell_embs.cpu())

        if (start // batch_size) % 50 == 0:
            print(f"  Processed {end}/{n_cells} cells")

    return torch.cat(all_embeddings, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--sample_ids", nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    model, token_dict, median_dict = load_geneformer_model(device)

    sample_ids = args.sample_ids or ALL_SAMPLE_IDS

    for sid in sample_ids:
        print(f"\n{'='*60}")
        print(f"Processing slice {sid}")
        print(f"{'='*60}")

        h5ad_path = os.path.join(args.raw_dir, f"{sid}.h5ad")
        if not os.path.exists(h5ad_path):
            print(f"  WARNING: {h5ad_path} not found, skipping")
            continue

        adata = sc.read_h5ad(h5ad_path)

        # Filter to spots with ground truth labels (same as preprocessing)
        label_col = None
        for col in ["sce.layer_guess", "layer_guess", "spatialLIBD", "ground_truth"]:
            if col in adata.obs.columns:
                label_col = col
                break

        if label_col:
            valid_mask = adata.obs[label_col].notna()
            adata = adata[valid_mask].copy()
            print(f"  Spots with labels: {adata.n_obs}")

        # Filter genes
        sc.pp.filter_genes(adata, min_cells=3)

        # Tokenize
        input_ids, attention_mask = tokenize_cells(adata, token_dict, median_dict)

        # Save tokenized data (for fine-tuning)
        token_path = os.path.join(args.output_dir, sid, "geneformer_tokens.pt")
        torch.save({"input_ids": input_ids, "attention_mask": attention_mask}, token_path)

        # Extract embeddings
        embeddings = extract_embeddings(
            model, input_ids, attention_mask, device, batch_size=args.batch_size
        )
        print(f"  Embeddings shape: {embeddings.shape}")

        # Save
        output_path = os.path.join(args.output_dir, sid, "geneformer_embeddings.pt")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(embeddings, output_path)
        print(f"  Saved to {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
