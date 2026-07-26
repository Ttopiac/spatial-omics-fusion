# Spatial Foundation Models on DLPFC — Input Specs and Feasibility

Companion to `scripts/prepare_fm_inputs.py`. Records what each candidate foundation
model requires as input, whether it can legitimately consume our data, and what a
result from it would actually mean.

**The headline finding: "spatially pretrained" does not imply "can read Visium."**
DLPFC is 10x Visium. Nicheformer and CellPLM are both genuinely spatial foundation
models, and neither has a code path for Visium — visible in their source as a
commented-out vocabulary entry and a two-element allowlist. Any Visium number from
either is an extrapolation. Meanwhile Novae, a graph-based spatial FM, treats Visium
as a first-class technology and outputs spatial domains zero-shot, which is our task
exactly. Assay support, not spatial pretraining, is the binding constraint.

---

## 1. Candidate summary

Two things gate a candidate independently, and conflating them is the easy mistake:
**(a)** does the model *accept* Visium at inference, and **(b)** was Visium in its
*pretraining* corpus. A model can pass (a) and fail (b) — that is transfer, and it is
still a legitimate experiment. A model that fails (a) cannot be run honestly at all.

| Model | Accepts Visium? | Visium pretrained? | Task alignment | Verdict |
|---|---|---|---|---|
| **Novae** | **Yes — first-class** | No (MERFISH/Xenium/CosMx, ~30M cells) | **Native: zero-shot spatial domains** | **Run first.** Only task-native candidate. |
| **scGPT-spatial** | Yes | **Yes** — SpatialHuman30M (Visium, Visium HD, MERFISH, Xenium) | Generic embedding + our head | **Run first.** Only in-distribution candidate. |
| **SToFM** | Unverified — "high-resolution ST", Geneformer gene vocab | Unverified (6 assays, 88M cells) | Reports DLPFC layer segmentation | Verify next. Checkpoint on Google Drive. |
| **AIDO.Tissue** | Unverified | Unverified | Niche-type prediction | Verify next. |
| **HEIST** | Unverified | Unverified | Graph FM for ST + proteomics | Verify next. |
| **Nicheformer** | **No** — no Visium assay token | No (CosMx, ISS, MERFISH, Xenium) | Generic embedding | Runnable only by declaring a false assay. Report as OOD. |
| **CellPLM** | **No** — `SPATIAL_PLATFORM_LIST = ['cosmx', 'merfish']` | No | Generic embedding | Spatial branch cannot fire on Visium. |
| **UCE** | n/a | No — 36M dissociated cells, 8 species | Generic embedding | Extends the existing negative result. Cheap. |
| **scFoundation** | n/a | No — ~50M dissociated human scRNA-seq | Generic embedding | Same. Cheap. |

UCE and scFoundation are confirmed dissociated-only, exactly as suspected. They are
worth running only as additional points on the existing scGPT/Geneformer trend line,
not as a test of the spatial-pretraining hypothesis.

### Why Novae is the most interesting candidate here

`novae/utils/build.py:27` declares Visium a first-class technology:

```python
SpatialTechnology = Literal["cosmx", "merscope", "xenium", "visium", "visium_hd"]
```

and `_default_visium_arguments()` gives plain Visium a **6-neighbor GRID graph** —
the same k=6 hex-lattice neighborhood as our GAT baseline. Genes are mapped by *name*
through a learned gene-embedding vocabulary (`novae/module/embed.py`, `CellEmbedder`),
which is how it generalizes across panels, so the 33,538-gene Visium matrix needs no
subsetting.

Three properties make it the sharpest test in this repo:

1. **Task-native.** Novae outputs spatial domain assignments zero-shot. Every other
   candidate emits a generic per-cell embedding that we then have to bolt a classifier
   onto — which measures our head as much as their model.
2. **Architecturally matched.** Novae is a self-supervised **graph attention network**
   over spatial neighborhoods. That is a pretrained version of the exact inductive
   bias our from-scratch GCN/GAT encodes. The head-to-head is therefore clean:
   does 30M cells of graph pretraining beat 402K parameters fit on one slice?
3. **Zero-shot means no split leakage.** A zero-shot domain assignment can be scored
   against all labeled spots with ARI directly, no train/val/test split needed.

The caveat is real and belongs in any writeup: Visium is *supported* but not
*pretrained on*. Novae's docs note Visium "differs fundamentally in nature from
imaging technologies" and suggest fine-tuning or retraining for bin data. So a weak
Novae result is evidence about transfer, not about spatial pretraining per se.

---

## 2. The two blockers, verified in source

### Nicheformer has no Visium token

`src/nicheformer/data/constants.py`, `AssayOntologyTermId` — the Visium entry exists
but is **commented out**:

```python
MERFISH_SPATIAL = "EFO:0008992"
COSMX_SPATIAL = "EFO:0030029"
# VISIUM_SPATIAL_GENE_EXPRESSION = "EFO:0010961"
```

The inference-time vocabulary (`notebooks/tokenization/xenium_human_lung.ipynb`)
confirms it — there is no Visium key:

```python
technology_dict = {
    "merfish": 7, "MERFISH": 7,
    "cosmx": 8, "NanoString digital spatial profiling": 8,
    "Xenium": 9,
    "10x 5' v2": 10, "10x 3' v3": 11, ...   # these are dissociated 10x, not Visium
}
```

There is a **second, deeper problem**. Nicheformer's tokenizer divides each spot's
normalized expression by a *technology-specific* median-counts-per-gene vector
(`data/model_means/{cosmx,iss,merfish,xenium,dissociated}_mean_script.npy`) before
ranking genes. There is no Visium mean vector. So running DLPFC requires borrowing
another assay's gene-median normalization — which directly perturbs the gene ranking
that *is* the model's input. This is not a cosmetic mislabel; it changes the tokens.

### CellPLM gates its spatial encoder to two platforms

`CellPLM/utils/data.py:16`:

```python
SPATIAL_PLATFORM_LIST = ['cosmx', 'merfish']
```

and at line 103:

```python
if 'platform' in adata.obs and adata.obs['platform'][...][0] in SPATIAL_PLATFORM_LIST:
    coord_x = torch.tensor(adata.obs['x_FOV_px'][...])[:, None]
    coord_y = torch.tensor(adata.obs['y_FOV_px'][...])[:, None]
    self.coord_list.append(torch.cat([coord_x, coord_y], 1))
else:
    self.coord_list.append(torch.zeros(x.shape[0], 2) - 1)   # <-- silent fallback
```

Declaring `platform='visium'` does not error. It silently substitutes `(-1, -1)`
for every coordinate, so the spatial encoder contributes nothing and you are running
a plain single-cell model that happens to be called spatial. **This failure is
invisible unless you check.** `prepare_fm_inputs.py` records the resolved state in
`.uns['assay_substitution']['spatial_branch_active']`.

### The resolution mismatch underneath both

Even setting the vocabulary aside, the assays differ in kind:

- CosMx / MERFISH / Xenium: single-cell (often subcellular) resolution, **targeted
  panels of ~300–1,000 genes**.
- Visium: **whole transcriptome (33,538 genes)**, 55 µm spots, each a mixture of
  roughly 10–30 cells.

A Nicheformer or CellPLM embedding of a Visium spot asks a model trained on single
cells to interpret a cell mixture, over a gene space ~50× larger than it ever saw.
Declaring `assay='MERFISH'` does not make a Visium spot a MERFISH cell.

---

## 3. What `prepare_fm_inputs.py` emits

`data/fm_inputs/{target}/{sample_id}.h5ad`, gzip-compressed (~2.1 GB for
6 targets × 12 slices).

Common to every target:

| Field | Content |
|---|---|
| `.X` | **Raw integer counts**, 33,538 genes, CSR. Not the pipeline's log-normalized 3K-HVG matrix — every FM normalizes internally. |
| `.var_names`, `.var['gene_symbol']` | HGNC symbols (scGPT, scGPT-spatial, CellPLM key on these) |
| `.var['ensembl_id']` | Ensembl IDs (UCE, Geneformer key on these) |
| `.obs['label']` | Integer class code — **identical to `data/processed/{id}/labels.pt`** |
| `.obs['layer']` | String layer name |
| `.obs['x_coord']`, `.obs['y_coord']` | Visium spot coordinates |
| `.obs['idx']` | Row position, matching the processed tensors |
| `.uns['label_map']`, `.uns['assay_substitution']` | Class encoding; declared-vs-true assay and why |

Verified on 151673: spot count (3,611), label vector, label map, and coordinates are
**bit-identical** to `data/processed/151673/`. Spot *i* is the same spot in both, so
the stratified 60/20/20 split in `src/data/dataset.py` lines up and results are
directly comparable to the GCN/GAT/scGPT/Geneformer numbers in `RESULTS.md`.

Per-target additions:

- **nicheformer** — `modality=4` (spatial), `specie=5` (human), `assay=<token>`,
  `nicheformer_split`, `idx`. Assay token set by `--nicheformer_assay` (default
  `MERFISH`); the script refuses values outside the real vocabulary.
- **cellplm** — `x_FOV_px`, `y_FOV_px`, `platform`, `batch`, `celltype`, `split`.
  Default `--cellplm_platform cosmx` so the spatial branch actually fires.
- **novae** — `batch`, `slide_key`; coordinates already in `.obsm['spatial']`.
  No substitution: `novae.spatial_neighbors(adata, technology="visium")` is native.
- **scgpt_spatial** — `batch`, `str_batch`. No substitution; in-distribution.
- **uce**, **scfoundation** — `organism`, `gene_symbol`. Both map onto their own
  fixed vocabulary at extraction time (UCE via its ESM2 protein-embedding index,
  scFoundation via a 19,264-gene ordered vector), so no subsetting here.

```bash
python scripts/prepare_fm_inputs.py --target all --all
python scripts/prepare_fm_inputs.py --target nicheformer --all --nicheformer_assay Xenium
```

---

## 4. Recommended experimental design

**Run Novae and scGPT-spatial first — they test different things.**

- **Novae** answers *does pretrained spatial-graph structure beat a from-scratch
  GCN?* It is task-native and zero-shot, so it needs no classifier head and no split.
  Visium is supported but not pretrained on, so this measures transfer.
- **scGPT-spatial** answers *does Visium-inclusive pretraining beat a from-scratch
  GCN?* It is the only in-distribution candidate. The repo already has the scGPT
  plumbing (`scripts/extract_scgpt_embeddings.py`, the `scgpt*` modes in
  `run_benchmark.py`) and the same tokenizer, so it is also the cheapest to wire up.

Together they separate the two confounded variables — *spatial inductive bias* and
*matching assay* — that a single experiment would leave tangled.

**For Nicheformer, sweep the assay token.** Run `MERFISH`, `cosmx`, and `Xenium`
and report the spread alongside the mean. If ARI moves materially across three
tokens that are all equally false, the token is driving the result and the number
is not measuring the model. This turns an unavoidable fabrication into a stated,
measured uncertainty.

**For CellPLM, run both platform settings.** `cosmx` (spatial branch on) and
`visium` (branch off, coords = -1). The gap between them *is* the measurement of
how much CellPLM's spatial pretraining transfers to Visium.

**Test where a pretrained model should actually win.** The GCN already hits 1.000
top-2 accuracy and 1.000 interior accuracy at k=96 — only ~0.04 ARI of headroom
remains, all of it at layer boundaries. Full-supervision ARI is close to saturated
and is the *least* discriminating setting available. The claims in `README.md` that
remain untested are about **low-data and cross-slice transfer**. Suggested protocol:
train the classifier head on *n* ∈ {50, 100, 250, 500} labeled spots per slice, and
separately hold out entire slices. That is where a pretrained model should beat a
from-scratch GCN if it ever does, and a negative result there is far stronger than
another saturated full-supervision number.

---

## 5. Prior work on spatial FMs + DLPFC

Two 2025–2026 papers evaluate spatially-aware models on this exact benchmark:

- **SToFM** (arXiv:2507.11588) — a multi-scale spatial transcriptomics foundation
  model that reports DLPFC layer segmentation as one of its headline tasks.
  The closest published analogue to the experiment proposed here.
- **Benchmarking Pathology Foundation Models for Spatial Domain Understanding**
  (arXiv:2605.25764) — 42 ST slides with paired H&E, including the 12 expert-annotated
  DLPFC slides. Reports H-Optimus-1 best overall and MUSK best on the DLPFC subset
  against expert annotation, with CCST+Leiden the strongest aggregate pipeline. This
  is a *pathology/image* FM benchmark, so it is a direct comparison point for our
  ResNet50 image-modality result rather than for the transcriptomic FMs.

I have not verified either paper's numbers or checked whether their DLPFC splits
match ours — worth reading before positioning our results against theirs.

**Caveat on this section and §1:** the source-code findings in §2 are verified
directly against the repositories. The pretraining-corpus compositions, parameter
counts, and checkpoint locations in §1 come from paper abstracts and model cards
read in July 2026, and licences/checkpoint availability change. Re-check before
committing compute.

---

## 6. Status

Data preparation is done and verified for all 12 slices × 5 targets. No foundation
model has been run yet — that requires downloading each checkpoint, and for
Nicheformer additionally the `model.h5ad` gene-space reference and the
`*_mean_script.npy` normalization vectors, which do not ship in this repo.
