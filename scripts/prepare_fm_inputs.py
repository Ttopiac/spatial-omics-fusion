"""
Annotate DLPFC slices so each foundation model can consume them.

Usage:
    python scripts/prepare_fm_inputs.py --target nicheformer --all
    python scripts/prepare_fm_inputs.py --target all --sample_id 151673

Every target reads data/raw/{sample_id}.h5ad (raw integer counts, 33,538 genes,
gene symbols as var_names and Ensembl IDs in var['gene_ids']) and writes an
annotated AnnData to data/fm_inputs/{target}/{sample_id}.h5ad.

Why this script exists separately from src/data/preprocess.py: the benchmark
pipeline log-normalizes, scales, and reduces to 3,000 HVGs. Every foundation
model here expects RAW COUNTS over its own gene vocabulary, and each one applies
its own normalization internally. Feeding them the processed matrix would
silently double-normalize and make the comparison meaningless.

Spot ordering, label encoding, and the NaN-label drop match preprocess.py
exactly, so spot i here is spot i in data/processed/{sample_id}/. That lets the
same stratified 60/20/20 split (src/data/dataset.py) line up across all models.

IMPORTANT — read docs/FM_INPUT_SPECS.md before trusting any Nicheformer or
CellPLM result. Both models structurally exclude Visium: Nicheformer has no
Visium assay token and no Visium gene-median vector, CellPLM's
SPATIAL_PLATFORM_LIST is ['cosmx', 'merfish']. Running DLPFC through either
requires declaring a technology the data is not. This script makes that
substitution explicit and records it in .uns['assay_substitution'] rather than
burying it.
"""
import argparse
import json
import os

import numpy as np
import scanpy as sc

ALL_SAMPLE_IDS = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676",
]

LABEL_COL = "sce.layer_guess"

# --- Nicheformer token vocabulary --------------------------------------------
# Verbatim from notebooks/tokenization/xenium_human_lung.ipynb in
# theislab/nicheformer. There is no Visium entry; src/nicheformer/data/
# constants.py has VISIUM_SPATIAL_GENE_EXPRESSION commented out.
NICHEFORMER_MODALITY = {"dissociated": 3, "spatial": 4}
NICHEFORMER_SPECIE = {"human": 5, "Homo sapiens": 5, "Mus musculus": 6, "mouse": 6}
NICHEFORMER_TECHNOLOGY = {
    "merfish": 7, "MERFISH": 7,
    "cosmx": 8, "NanoString digital spatial profiling": 8,
    "Xenium": 9,
    "10x 5' v2": 10, "10x 3' v3": 11, "10x 3' v2": 12, "10x 5' v1": 13,
    "10x 3' v1": 14,
    "10x 3' transcription profiling": 15, "10x transcription profiling": 15,
    "10x 5' transcription profiling": 16,
    "CITE-seq": 17, "Smart-seq v4": 18,
}


def load_labeled_slice(sample_id, raw_dir):
    """Load a raw slice and drop unlabeled spots, matching preprocess.py."""
    adata = sc.read_h5ad(os.path.join(raw_dir, f"{sample_id}.h5ad"))
    adata = adata[~adata.obs[LABEL_COL].isna()].copy()
    return adata


def annotate_common(adata, sample_id):
    """Fields every target needs: labels, integer codes, coordinates, ids."""
    label_series = adata.obs[LABEL_COL].astype("category")
    label_map = {cat: i for i, cat in enumerate(label_series.cat.categories)}

    adata.obs["layer"] = label_series.astype(str)
    adata.obs["label"] = label_series.cat.codes.values.astype(np.int64)
    adata.obs["sample_id"] = sample_id

    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    adata.obs["x_coord"] = coords[:, 0]
    adata.obs["y_coord"] = coords[:, 1]

    # Row position in this file == row position in data/processed/{id}/*.pt
    adata.obs["idx"] = np.arange(adata.n_obs, dtype=np.int64)

    adata.uns["label_map"] = label_map
    adata.uns["sample_id"] = sample_id
    return adata, label_map


def prep_nicheformer(adata, assay_name):
    """
    Nicheformer: integer-token obs + raw counts over the model's gene space.

    Emits the obs columns the tokenizer consumes (assay, specie, modality, idx).
    Gene-space intersection against model.h5ad and the rank tokenization itself
    happen at extraction time — they need the checkpoint's model.h5ad and the
    technology-specific *_mean_script.npy, neither of which ships in this repo.
    """
    if assay_name not in NICHEFORMER_TECHNOLOGY:
        raise ValueError(
            f"'{assay_name}' is not in Nicheformer's technology vocabulary. "
            f"Valid: {sorted(NICHEFORMER_TECHNOLOGY)}"
        )

    adata.obs["modality"] = NICHEFORMER_MODALITY["spatial"]
    adata.obs["specie"] = NICHEFORMER_SPECIE["human"]
    adata.obs["assay"] = NICHEFORMER_TECHNOLOGY[assay_name]
    adata.obs["organism"] = "Homo sapiens"
    adata.obs["nicheformer_split"] = "test"  # frozen-embedding extraction only

    adata.uns["assay_substitution"] = {
        "true_assay": "10x Visium Spatial Gene Expression (EFO:0010961)",
        "declared_assay": assay_name,
        "declared_token": NICHEFORMER_TECHNOLOGY[assay_name],
        "reason": (
            "Nicheformer's SpatialCorpus-110M contains only CosMx, ISS, MERFISH "
            "and Xenium. It has no Visium assay token and no Visium gene-median "
            "normalization vector. Any Visium result is therefore an "
            "out-of-distribution extrapolation, not a fair evaluation."
        ),
        "required_action": (
            "Sweep --nicheformer_assay over MERFISH/cosmx/Xenium and report the "
            "spread. If the spread is large, the assay token is driving the "
            "result and the number means nothing."
        ),
    }
    return adata


def prep_cellplm(adata, platform):
    """
    CellPLM: reads coords from obs['x_FOV_px']/obs['y_FOV_px'], but ONLY when
    obs['platform'] is in SPATIAL_PLATFORM_LIST = ['cosmx', 'merfish'].
    Any other value silently substitutes a (-1, -1) coordinate placeholder,
    i.e. the spatial branch never fires and you are running a plain scRNA model.
    """
    adata.obs["x_FOV_px"] = adata.obs["x_coord"].astype(np.float32)
    adata.obs["y_FOV_px"] = adata.obs["y_coord"].astype(np.float32)
    adata.obs["platform"] = platform
    adata.obs["batch"] = adata.obs["sample_id"].astype(str)
    adata.obs["celltype"] = adata.obs["layer"].astype(str)
    adata.obs["split"] = "test"

    spatial_ok = platform in ("cosmx", "merfish")
    adata.uns["assay_substitution"] = {
        "true_assay": "10x Visium Spatial Gene Expression",
        "declared_platform": platform,
        "spatial_branch_active": bool(spatial_ok),
        "reason": (
            "CellPLM gates its spatial encoder on "
            "SPATIAL_PLATFORM_LIST = ['cosmx', 'merfish'] "
            "(CellPLM/utils/data.py:16). Declaring 'visium' disables the "
            "spatial path entirely and fills coords with -1."
        ),
    }
    return adata


def prep_novae(adata):
    """
    Novae: the only candidate that treats Visium as a first-class technology.

    novae/utils/build.py:27 declares
        SpatialTechnology = Literal["cosmx", "merscope", "xenium", "visium", "visium_hd"]
    and _default_visium_arguments() gives plain Visium a 6-neighbor GRID graph —
    the same k=6 hex-lattice neighborhood our GCN baseline uses.

    Novae reads coordinates from .obsm['spatial'] (already present) and maps genes
    by NAME through a learned gene-embedding vocabulary (novae/module/embed.py,
    CellEmbedder), which is how it generalizes across panels. So the full 33,538-gene
    Visium matrix needs no subsetting here.

    Caveat: Visium is supported at INFERENCE but is not in the pretraining corpus
    (~30M cells, MERFISH/Xenium/CosMx only). See docs/FM_INPUT_SPECS.md.
    """
    adata.obs["batch"] = adata.obs["sample_id"].astype(str)
    adata.obs["slide_key"] = adata.obs["sample_id"].astype(str)
    adata.uns["assay_substitution"] = {
        "true_assay": "10x Visium Spatial Gene Expression",
        "declared_technology": "visium",
        "reason": (
            "No substitution needed: novae.spatial_neighbors accepts "
            "technology='visium' natively. Visium is not in Novae's pretraining "
            "corpus (MERFISH/Xenium/CosMx), so this is transfer, not "
            "in-distribution inference."
        ),
        "next_step": (
            "novae.spatial_neighbors(adata, technology='visium'); "
            "model = novae.Novae.from_pretrained('prism-oncology/novae-human-0'); "
            "model.compute_representations(adata, zero_shot=True); "
            "model.assign_domains(adata)"
        ),
    }
    return adata


def prep_plain(adata, organism="homo_sapiens"):
    """
    UCE and scFoundation: raw counts, gene symbols as var_names, species tag.
    Both map onto their own fixed gene vocabulary at extraction time (UCE via
    its ESM2 protein-embedding index, scFoundation via a 19,264-gene ordered
    vector), so no gene subsetting happens here.
    """
    adata.obs["organism"] = organism
    adata.var["gene_symbol"] = adata.var_names.astype(str)
    return adata


def prepare(sample_id, target, raw_dir, out_root, nicheformer_assay,
            cellplm_platform):
    adata = load_labeled_slice(sample_id, raw_dir)
    adata, label_map = annotate_common(adata, sample_id)

    # Keep both gene ID conventions available: scGPT/scGPT-spatial and CellPLM
    # key on symbols, UCE and Geneformer key on Ensembl.
    adata.var["ensembl_id"] = adata.var["gene_ids"].astype(str)
    adata.var["gene_symbol"] = adata.var_names.astype(str)

    if target == "nicheformer":
        adata = prep_nicheformer(adata, nicheformer_assay)
    elif target == "cellplm":
        adata = prep_cellplm(adata, cellplm_platform)
    elif target == "scgpt_spatial":
        # scGPT-spatial's SpatialHuman30M includes Visium — no substitution
        # needed. It reads symbols from var and raw counts from X.
        adata.obs["batch"] = adata.obs["sample_id"].astype(str)
        adata.obs["str_batch"] = adata.obs["sample_id"].astype(str)
        adata.uns["assay_substitution"] = {
            "true_assay": "10x Visium Spatial Gene Expression",
            "declared_assay": "10x Visium Spatial Gene Expression",
            "note": "In-distribution: Visium is part of SpatialHuman30M.",
        }
    elif target == "novae":
        adata = prep_novae(adata)
    elif target in ("uce", "scfoundation"):
        adata = prep_plain(adata)
    else:
        raise ValueError(f"Unknown target: {target}")

    out_dir = os.path.join(out_root, target)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{sample_id}.h5ad")
    # The count matrix is identical across targets — only obs annotation differs.
    # gzip cuts each slice from ~120 MB to ~33 MB, so `--target all --all` costs
    # ~1.8 GB instead of ~6.5 GB.
    adata.write_h5ad(out_path, compression="gzip")

    n_counts = int(adata.X.sum())
    print(f"  [{sample_id}] {target}: {adata.n_obs} spots x {adata.n_vars} genes, "
          f"{len(label_map)} classes, {n_counts:,} total counts -> {out_path}")
    return {
        "sample_id": sample_id,
        "target": target,
        "n_spots": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_classes": len(label_map),
        "label_map": label_map,
    }


def main():
    p = argparse.ArgumentParser(description="Annotate DLPFC for foundation models")
    p.add_argument("--target", type=str, default="all",
                   choices=["novae", "scgpt_spatial", "nicheformer", "cellplm",
                            "uce", "scfoundation", "all"])
    p.add_argument("--sample_id", type=str, default="151673")
    p.add_argument("--all", action="store_true", help="all 12 slices")
    p.add_argument("--raw_dir", type=str, default="data/raw")
    p.add_argument("--out_root", type=str, default="data/fm_inputs")
    p.add_argument("--nicheformer_assay", type=str, default="MERFISH",
                   help="Which technology token to declare. DLPFC is Visium, "
                        "which Nicheformer does not support — sweep this.")
    p.add_argument("--cellplm_platform", type=str, default="cosmx",
                   help="'visium' disables CellPLM's spatial branch entirely.")
    args = p.parse_args()

    sample_ids = ALL_SAMPLE_IDS if args.all else [args.sample_id]
    targets = (["novae", "scgpt_spatial", "nicheformer", "cellplm", "uce",
                "scfoundation"]
               if args.target == "all" else [args.target])

    manifest = []
    for target in targets:
        print(f"\n=== {target} ===")
        for sid in sample_ids:
            manifest.append(prepare(sid, target, args.raw_dir, args.out_root,
                                    args.nicheformer_assay, args.cellplm_platform))

    os.makedirs(args.out_root, exist_ok=True)
    manifest_path = os.path.join(args.out_root, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "nicheformer_assay": args.nicheformer_assay,
            "cellplm_platform": args.cellplm_platform,
            "entries": manifest,
        }, f, indent=2)
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
