"""
Extract scGPT embeddings for all DLPFC slices.

Usage:
    python scripts/extract_scgpt_embeddings.py
    python scripts/extract_scgpt_embeddings.py --sample_ids 151673
    python scripts/extract_scgpt_embeddings.py --model_dir data/scgpt_human --batch_size 64

This script:
1. Loads the pretrained scGPT whole-human model
2. For each DLPFC slice, tokenizes genes and extracts 512-dim cell embeddings
3. Saves embeddings to data/processed/{sample_id}/scgpt_embeddings.pt
"""

import argparse
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
import scanpy as sc

# ── torchtext shim ──────────────────────────────────────────────────────────
# scGPT's tokenizer imports torchtext.vocab.Vocab, which is deprecated and
# incompatible with PyTorch >= 2.4. We create a minimal shim so that scGPT's
# GeneVocab can load the vocab.json without needing the real torchtext.
from collections import OrderedDict


class _Vocab:
    """Minimal drop-in for torchtext.vocab.Vocab used by scGPT's GeneVocab."""

    def __init__(self, vocab_obj=None):
        if vocab_obj is not None and hasattr(vocab_obj, "_stoi"):
            self._stoi = dict(vocab_obj._stoi)
            self._itos = list(vocab_obj._itos)
        else:
            self._stoi = {}
            self._itos = []
        self._default_index = 0

    def __getitem__(self, token):
        return self._stoi.get(token, self._default_index)

    def __contains__(self, token):
        return token in self._stoi

    def __len__(self):
        return len(self._itos)

    def set_default_index(self, idx):
        self._default_index = idx

    def get_stoi(self):
        return self._stoi

    def get_itos(self):
        return self._itos

    def insert_token(self, token, index):
        """Insert token at a specific index."""
        if token not in self._stoi:
            # Shift existing tokens if needed
            for t, i in self._stoi.items():
                if i >= index:
                    self._stoi[t] = i + 1
            self._stoi[token] = index
            self._itos.insert(index, token)
        else:
            # Token exists, just update index
            old_idx = self._stoi[token]
            if old_idx != index:
                self._itos.remove(token)
                self._itos.insert(index, token)
                self._stoi = {t: i for i, t in enumerate(self._itos)}

    def append_token(self, token):
        if token not in self._stoi:
            self._stoi[token] = len(self._itos)
            self._itos.append(token)

    def lookup_token(self, index):
        return self._itos[index]

    def lookup_indices(self, tokens):
        return [self[t] for t in tokens]

    @property
    def vocab(self):
        return self


def _vocab_factory(ordered_dict, min_freq=1):
    v = _Vocab()
    v._stoi = {k: i for i, k in enumerate(ordered_dict.keys())}
    v._itos = list(ordered_dict.keys())
    return v


_torchtext = types.ModuleType("torchtext")
_torchtext_vocab = types.ModuleType("torchtext.vocab")
_torchtext_vocab.Vocab = _Vocab
_torchtext_vocab.vocab = _vocab_factory
_torchtext.vocab = _torchtext_vocab
sys.modules["torchtext"] = _torchtext
sys.modules["torchtext.vocab"] = _torchtext_vocab
# ── end shim ────────────────────────────────────────────────────────────────

from scgpt.model.model import TransformerModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ALL_SAMPLE_IDS = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676",
]


def load_scgpt_model(model_dir: str, device: torch.device):
    """Load pretrained scGPT model and vocabulary."""
    model_dir = Path(model_dir)

    # Load config
    with open(model_dir / "args.json") as f:
        model_config = json.load(f)

    # Load vocabulary
    vocab = GeneVocab.from_file(model_dir / "vocab.json")
    pad_token = model_config.get("pad_token", "<pad>")
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    for t in special_tokens:
        if t not in vocab:
            vocab.append_token(t)
    vocab.set_default_index(vocab[pad_token])
    ntokens = len(vocab)

    # Build model
    model = TransformerModel(
        ntoken=ntokens,
        d_model=model_config["embsize"],
        nhead=model_config["nheads"],
        d_hid=model_config["d_hid"],
        nlayers=model_config["nlayers"],
        vocab=vocab,
        dropout=model_config["dropout"],
        pad_token=pad_token,
        pad_value=model_config["pad_value"],
        do_mvc=model_config.get("MVC", False),
        do_dab=False,
        use_batch_labels=False,
        input_emb_style=model_config.get("input_emb_style", "continuous"),
        n_input_bins=model_config.get("n_bins", 51),
        use_fast_transformer=False,  # don't require flash-attn
    )

    # Load weights
    state_dict = torch.load(model_dir / "best_model.pt", map_location=device)
    # Remove unexpected keys that may be present
    model_keys = set(model.state_dict().keys())
    filtered_state = {k: v for k, v in state_dict.items() if k in model_keys}
    model.load_state_dict(filtered_state, strict=False)
    model.to(device)
    model.eval()

    print(f"Loaded scGPT: {sum(p.numel() for p in model.parameters()):,} params")
    return model, vocab, model_config


def preprocess_for_scgpt(adata, vocab, max_seq_len=1200, pad_value=-2):
    """
    Prepare an AnnData for scGPT embedding extraction.

    scGPT expects:
    - gene_ids: token indices for each gene (padded to max_seq_len)
    - values: expression values for each gene (binned into n_bins)

    Returns gene_ids (N, max_seq_len) and values (N, max_seq_len) as tensors.
    """
    # Normalize and log-transform (scGPT expects log1p-normalized data)
    adata_pp = adata.copy()
    sc.pp.normalize_total(adata_pp, target_sum=1e4)
    sc.pp.log1p(adata_pp)

    # Get gene names
    gene_names = list(adata_pp.var_names)

    # Map genes to vocab indices, keep only genes in vocab
    gene_to_idx = {}
    for g in gene_names:
        if g in vocab:
            gene_to_idx[g] = vocab[g]

    valid_genes = [g for g in gene_names if g in gene_to_idx]
    valid_gene_indices = [gene_names.index(g) for g in valid_genes]
    valid_vocab_ids = [gene_to_idx[g] for g in valid_genes]

    print(f"  Genes in vocab: {len(valid_genes)}/{len(gene_names)}")

    # Truncate to max_seq_len
    if len(valid_genes) > max_seq_len:
        # Keep the most variable genes
        expr_matrix = adata_pp.X if not hasattr(adata_pp.X, 'toarray') else adata_pp.X.toarray()
        gene_vars = np.var(expr_matrix[:, valid_gene_indices], axis=0)
        top_idx = np.argsort(gene_vars)[-max_seq_len:]
        valid_gene_indices = [valid_gene_indices[i] for i in top_idx]
        valid_vocab_ids = [valid_vocab_ids[i] for i in top_idx]
        valid_genes = [valid_genes[i] for i in top_idx]

    n_genes_used = len(valid_genes)
    n_cells = adata_pp.n_obs

    # Extract expression values for valid genes
    expr_matrix = adata_pp.X if not hasattr(adata_pp.X, 'toarray') else adata_pp.X.toarray()
    expr_values = expr_matrix[:, valid_gene_indices]  # (N, n_genes_used)

    # Bin expression values (scGPT uses binned input)
    n_bins = 51
    # Clip and bin
    nonzero_mask = expr_values > 0
    if nonzero_mask.any():
        # Bin non-zero values into n_bins-1 bins, keep 0 as bin 0
        nonzero_vals = expr_values[nonzero_mask]
        bins = np.quantile(nonzero_vals, np.linspace(0, 1, n_bins))
        binned = np.digitize(expr_values, bins, right=True)
        binned = np.clip(binned, 0, n_bins - 1)
    else:
        binned = np.zeros_like(expr_values, dtype=int)

    # Create padded tensors
    gene_ids = torch.full((n_cells, max_seq_len), vocab["<pad>"], dtype=torch.long)
    values = torch.full((n_cells, max_seq_len), pad_value, dtype=torch.float)

    gene_ids[:, :n_genes_used] = torch.tensor(valid_vocab_ids, dtype=torch.long)
    values[:, :n_genes_used] = torch.tensor(binned, dtype=torch.float)

    return gene_ids, values, n_genes_used


@torch.no_grad()
def extract_embeddings(model, gene_ids, values, device, batch_size=64, pad_value=-2):
    """Extract cell embeddings from scGPT model."""
    n_cells = gene_ids.shape[0]
    all_embeddings = []

    for start in range(0, n_cells, batch_size):
        end = min(start + batch_size, n_cells)
        batch_gene_ids = gene_ids[start:end].to(device)
        batch_values = values[start:end].to(device)

        # Forward pass through scGPT
        # The model's _encode method returns transformer output embeddings
        src_key_padding_mask = batch_values.eq(pad_value)

        # Get embeddings from the encoder
        output = model._encode(
            batch_gene_ids,
            batch_values,
            src_key_padding_mask=src_key_padding_mask,
        )
        # output shape: (batch, seq_len, d_model=512)

        # Pool: mean over non-padded gene positions
        mask = ~src_key_padding_mask  # (batch, seq_len)
        mask_expanded = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        pooled = (output * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        # pooled shape: (batch, 512)

        all_embeddings.append(pooled.cpu())

        if (start // batch_size) % 10 == 0:
            print(f"  Processed {end}/{n_cells} cells")

    return torch.cat(all_embeddings, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="data/scgpt_human")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--sample_ids", nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load model
    model, vocab, model_config = load_scgpt_model(args.model_dir, device)
    max_seq_len = model_config.get("max_seq_len", 1200)

    sample_ids = args.sample_ids or ALL_SAMPLE_IDS

    for sid in sample_ids:
        print(f"\n{'='*60}")
        print(f"Processing slice {sid}")
        print(f"{'='*60}")

        # Load raw h5ad
        h5ad_path = os.path.join(args.raw_dir, f"{sid}.h5ad")
        if not os.path.exists(h5ad_path):
            print(f"  WARNING: {h5ad_path} not found, skipping")
            continue

        adata = sc.read_h5ad(h5ad_path)

        # Filter to spots that have ground truth labels (same as preprocessing)
        label_col = None
        for col in ["sce.layer_guess", "layer_guess", "spatialLIBD", "ground_truth"]:
            if col in adata.obs.columns:
                label_col = col
                break

        if label_col:
            valid_mask = adata.obs[label_col].notna()
            adata = adata[valid_mask].copy()
            print(f"  Spots with labels: {adata.n_obs}")

        # Filter genes (same as our preprocessing)
        sc.pp.filter_genes(adata, min_cells=3)

        # Tokenize and prepare for scGPT
        gene_ids, values, n_genes = preprocess_for_scgpt(
            adata, vocab, max_seq_len=max_seq_len
        )

        # Extract embeddings
        embeddings = extract_embeddings(
            model, gene_ids, values, device, batch_size=args.batch_size
        )
        print(f"  Embeddings shape: {embeddings.shape}")

        # Save
        output_path = os.path.join(args.output_dir, sid, "scgpt_embeddings.pt")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(embeddings, output_path)
        print(f"  Saved to {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
