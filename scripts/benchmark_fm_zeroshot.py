"""
Zero-shot spatial domain benchmark on DLPFC.

Usage:
    python scripts/benchmark_fm_zeroshot.py --method baselines --all
    python scripts/benchmark_fm_zeroshot.py --method novae --all
    python scripts/benchmark_fm_zeroshot.py --method novae --novae_model prism-oncology/novae-brain-0 --all

Zero-shot here means NO LABELS ARE USED at any point: each method produces either
a domain assignment directly or an embedding that we cluster into exactly n_classes
groups. Labels enter only to score ARI/NMI afterwards. This is the standard protocol
for spatial domain identification, and it is a different task from the supervised
benchmark in docs/RESULTS.md — the numbers are NOT comparable to the 0.943 ARI the
GCN gets with 60% of spots labeled.

The baselines matter as much as the models. A foundation model that cannot beat
"PCA + KMeans on expression" has not demonstrated anything, and one that cannot beat
"average expression over the k=6 spatial neighborhood, then cluster" has not
demonstrated that its *spatial* pretraining does any work. That second baseline is
the unsupervised analogue of this repo's headline finding.

Results append to results/zeroshot/{method}.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ALL_SAMPLE_IDS = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676",
]

SEED = 42


def score(true, pred):
    return {
        "ari": float(adjusted_rand_score(true, pred)),
        "nmi": float(normalized_mutual_info_score(true, pred)),
        "n_pred_domains": int(len(set(pred))),
    }


def cluster_embedding(emb, k, seed=SEED):
    """KMeans into exactly k groups. Deterministic given the seed."""
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(emb)


# --- baselines ---------------------------------------------------------------

def _lognorm(adata):
    a = adata.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=3000)
    a = a[:, a.var.highly_variable].copy()
    return a


def baseline_expression(adata, k):
    """PCA(50) on log-normalized HVG expression -> KMeans. No spatial information."""
    a = _lognorm(adata)
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    emb = PCA(n_components=50, random_state=SEED).fit_transform(X)
    return cluster_embedding(emb, k)


def baseline_spatial_smooth(adata, k, n_neighs=6):
    """
    Average PCA features over each spot's k=6 spatial neighborhood, then KMeans.

    This is the unsupervised stand-in for what the GCN does: one round of mean
    aggregation over the spatial graph. If a spatial foundation model cannot beat
    this, its pretraining is not buying spatial structure.
    """
    import squidpy as sq

    a = _lognorm(adata)
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    emb = PCA(n_components=50, random_state=SEED).fit_transform(X)

    sq.gr.spatial_neighbors(a, n_neighs=n_neighs, coord_type="grid")
    adj = a.obsp["spatial_connectivities"]
    # mean over neighbors + self, matching GCN's degree-normalized mean
    deg = np.asarray(adj.sum(axis=1)).ravel() + 1.0
    smoothed = (adj @ emb + emb) / deg[:, None]
    return cluster_embedding(smoothed, k)


def baseline_spatial_smooth_2hop(adata, k, n_neighs=6):
    """Two rounds of neighborhood averaging — the unsupervised analogue of 2-layer GCN."""
    import squidpy as sq

    a = _lognorm(adata)
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    emb = PCA(n_components=50, random_state=SEED).fit_transform(X)

    sq.gr.spatial_neighbors(a, n_neighs=n_neighs, coord_type="grid")
    adj = a.obsp["spatial_connectivities"]
    deg = np.asarray(adj.sum(axis=1)).ravel() + 1.0
    for _ in range(2):
        emb = (adj @ emb + emb) / deg[:, None]
    return cluster_embedding(emb, k)


BASELINES = {
    "pca_kmeans": baseline_expression,
    "spatial_smooth_1hop": baseline_spatial_smooth,
    "spatial_smooth_2hop": baseline_spatial_smooth_2hop,
}


# --- foundation models -------------------------------------------------------

def run_novae(sample_ids, model_name, use_native_domains=True):
    """
    Novae zero-shot. Native path: compute_representations(zero_shot=True) then
    assign_domains. Also records KMeans on the latent so it is scored the same way
    as every other model.
    """
    import novae

    model = novae.Novae.from_pretrained(model_name)
    out = []
    for sid in sample_ids:
        a = sc.read_h5ad(f"data/fm_inputs/novae/{sid}.h5ad")
        true = a.obs["label"].values
        k = int(a.obs["label"].nunique())

        novae.spatial_neighbors(a, technology="visium")
        model.compute_representations(a, zero_shot=True)
        latent = np.asarray(a.obsm["novae_latent"])

        rec = {"sample_id": sid, "n_classes": k, "model": model_name}
        rec["kmeans"] = score(true, cluster_embedding(latent, k))
        if use_native_domains:
            key = model.assign_domains(a, n_domains=k)
            rec["native"] = score(true, a.obs[key].astype(str).values)
        out.append(rec)
        native = rec.get("native", {}).get("ari")
        print(f"  [{sid}] k={k}  kmeans ARI={rec['kmeans']['ari']:.4f}"
              + (f"  native ARI={native:.4f}" if native is not None else ""))
    return out


def run_cellplm(sample_ids, ckpt_dir, prefix="20231027_85M"):
    """
    CellPLM zero-shot. Emits a 512-d embedding per spot — it has no native domain
    output, so the embedding is read out with the same fixed KMeans(n_classes) used
    for every embedding-only model. No per-model tuning.

    Input is keyed on Ensembl IDs (var['ensembl_id']) with
    ensembl_auto_conversion=False, which avoids CellPLM's mygene network lookup.
    obs['platform']='cosmx' so the spatial branch actually fires — see
    docs/FM_INPUT_SPECS.md for why 'visium' would silently disable it.
    """
    import functools
    import torch

    torch.load = functools.partial(torch.load, map_location="cpu")
    from CellPLM.pipeline.cell_embedding import CellEmbeddingPipeline

    pipe = CellEmbeddingPipeline(pretrain_prefix=prefix, pretrain_directory=ckpt_dir)
    out = []
    for sid in sample_ids:
        a = sc.read_h5ad(f"data/fm_inputs/cellplm/{sid}.h5ad")
        true = a.obs["label"].values
        k = int(a.obs["label"].nunique())

        a.var.index = a.var["ensembl_id"].astype(str)
        a.var_names_make_unique()
        emb = pipe.predict(a, device="cpu", ensembl_auto_conversion=False)
        emb = emb.cpu().numpy() if hasattr(emb, "cpu") else np.asarray(emb)

        rec = {"sample_id": sid, "n_classes": k, "model": f"CellPLM-{prefix}",
               "embedding_readout": score(true, cluster_embedding(emb, k))}
        out.append(rec)
        print(f"  [{sid}] k={k}  embedding_readout ARI="
              f"{rec['embedding_readout']['ari']:.4f}")
    return out


def _align_to_nicheformer_space(adata, ref_genes):
    """
    Reindex onto Nicheformer's exact 20,310-gene Ensembl space, zero-filling absent
    genes. This is required, not cosmetic: the tokenizer divides by a length-20,310
    technology-mean vector and indexes an embedding table sized to it. The repo's
    own `ad.concat(..., join='inner')` recipe returns the *intersection* instead,
    which desynchronizes both (see theislab/nicheformer issue #12).
    """
    import anndata as ad
    from scipy.sparse import csr_matrix

    a = adata.copy()
    a.var.index = a.var["ensembl_id"].astype(str)
    a.var_names_make_unique()

    idx = {g: i for i, g in enumerate(a.var_names)}
    src = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    X = np.zeros((a.n_obs, len(ref_genes)), dtype=np.float32)
    n_present = 0
    for j, g in enumerate(ref_genes):
        c = idx.get(g, -1)
        if c >= 0:
            X[:, j] = src[:, c]
            n_present += 1

    out = ad.AnnData(X=csr_matrix(X), obs=a.obs.copy())
    out.var_names = list(ref_genes)
    return out, n_present


def run_nicheformer(sample_ids, mean_path, assay_label, batch_size=64, device="cpu"):
    """
    Nicheformer zero-shot. Embedding-only — no native domain output, so the same
    fixed KMeans(n_classes) readout is used as for CellPLM.

    assay_label records WHICH false technology token was declared. DLPFC is Visium,
    which Nicheformer does not support; sweeping this argument measures how much the
    fabricated token drives the result. See docs/FM_INPUT_SPECS.md.
    """
    import anndata as ad
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    ref = list(ad.read_h5ad(hf_hub_download("theislab/Nicheformer", "model.h5ad")).var_names)
    tok = AutoTokenizer.from_pretrained("theislab/Nicheformer", trust_remote_code=True)
    tok._load_technology_mean(mean_path)
    model = AutoModelForMaskedLM.from_pretrained(
        "theislab/Nicheformer", trust_remote_code=True).eval().to(device)
    print(f"  device={device}", flush=True)

    out = []
    for sid in sample_ids:
        a = sc.read_h5ad(f"data/fm_inputs/nicheformer/{sid}.h5ad")
        true = a.obs["label"].values
        k = int(a.obs["label"].nunique())

        aligned, n_present = _align_to_nicheformer_space(a, ref)
        inputs = tok(aligned)
        ids, am = inputs["input_ids"], inputs["attention_mask"]

        embs = []
        with torch.no_grad():
            for i in range(0, ids.shape[0], batch_size):
                e = model.get_embeddings(
                    input_ids=ids[i:i + batch_size].to(device),
                    attention_mask=am[i:i + batch_size].to(device),
                    layer=-1, with_context=False)
                embs.append(e.cpu().numpy() if torch.is_tensor(e) else np.asarray(e))
        emb = np.concatenate(embs, 0)

        rec = {"sample_id": sid, "n_classes": k, "model": "Nicheformer",
               "declared_assay": assay_label, "n_genes_matched": n_present,
               "embedding_readout": score(true, cluster_embedding(emb, k))}
        out.append(rec)
        print(f"  [{sid}] k={k} genes={n_present} assay={assay_label}  "
              f"embedding_readout ARI={rec['embedding_readout']['ari']:.4f}",
              flush=True)
    return out


def run_baselines(sample_ids):
    out = []
    for sid in sample_ids:
        a = sc.read_h5ad(f"data/fm_inputs/novae/{sid}.h5ad")  # any target: same counts
        true = a.obs["label"].values
        k = int(a.obs["label"].nunique())
        rec = {"sample_id": sid, "n_classes": k}
        for name, fn in BASELINES.items():
            rec[name] = score(true, fn(a, k))
        print(f"  [{sid}] k={k}  " + "  ".join(
            f"{n} ARI={rec[n]['ari']:.4f}" for n in BASELINES))
        out.append(rec)
    return out


def summarize(records, metric_keys):
    """Mean +/- std ARI and NMI across slices, per sub-method."""
    summary = {}
    for key in metric_keys:
        aris = [r[key]["ari"] for r in records if key in r]
        nmis = [r[key]["nmi"] for r in records if key in r]
        if not aris:
            continue
        summary[key] = {
            "ari_mean": float(np.mean(aris)), "ari_std": float(np.std(aris)),
            "nmi_mean": float(np.mean(nmis)), "nmi_std": float(np.std(nmis)),
            "n_slices": len(aris),
        }
    return summary


def main():
    p = argparse.ArgumentParser(description="Zero-shot spatial domain benchmark")
    p.add_argument("--method", required=True,
                   choices=["baselines", "novae", "cellplm", "nicheformer"])
    p.add_argument("--nicheformer_mean", type=str,
                   help="Path to {tech}_mean_script.npy from theislab/nicheformer")
    p.add_argument("--nicheformer_assay", type=str, default="MERFISH")
    p.add_argument("--device", type=str, default="cpu",
                   help="cpu | mps | cuda. MPS is ~10x faster on Apple Silicon.")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Nicheformer attention at 1500 tokens is memory-hungry: "
                        "batch 64 needs ~8.6 GiB and OOMs a 24 GB MPS budget. "
                        "8 is safe on MPS; 32-64 fits a 32 GB CUDA card.")
    p.add_argument("--cellplm_ckpt", type=str, default="ckpt",
                   help="Directory holding {prefix}.best.ckpt and .config.json")
    p.add_argument("--cellplm_prefix", type=str, default="20231027_85M")
    p.add_argument("--sample_id", type=str, default="151673")
    p.add_argument("--all", action="store_true")
    p.add_argument("--novae_model", type=str,
                   default="prism-oncology/novae-human-0")
    p.add_argument("--out_dir", type=str, default="results/zeroshot")
    args = p.parse_args()

    sample_ids = ALL_SAMPLE_IDS if args.all else [args.sample_id]
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"=== {args.method} on {len(sample_ids)} slice(s) ===")
    if args.method == "baselines":
        records = run_baselines(sample_ids)
        keys = list(BASELINES)
        tag = "baselines"
    elif args.method == "novae":
        records = run_novae(sample_ids, args.novae_model)
        keys = ["kmeans", "native"]
        tag = f"novae_{args.novae_model.split('/')[-1]}"
    elif args.method == "cellplm":
        records = run_cellplm(sample_ids, args.cellplm_ckpt, args.cellplm_prefix)
        keys = ["embedding_readout"]
        tag = f"cellplm_{args.cellplm_prefix}"
    else:
        records = run_nicheformer(sample_ids, args.nicheformer_mean,
                                  args.nicheformer_assay,
                                  batch_size=args.batch_size, device=args.device)
        keys = ["embedding_readout"]
        tag = f"nicheformer_{args.nicheformer_assay}"

    summary = summarize(records, keys)
    payload = {"method": args.method, "tag": tag, "n_slices": len(sample_ids),
               "summary": summary, "per_slice": records}
    out_path = os.path.join(args.out_dir, f"{tag}.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n--- {tag} ---")
    for key, s in summary.items():
        print(f"  {key:22s} ARI {s['ari_mean']:.4f} +/- {s['ari_std']:.4f}"
              f"   NMI {s['nmi_mean']:.4f}  ({s['n_slices']} slices)")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
