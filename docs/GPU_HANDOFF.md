# GPU Handoff — Spatial Foundation Model Benchmark

Brief for an agent with a CUDA GPU (target: RTX 5090, 32 GB). Read this fully before
running anything.

## The question

The repo's supervised result: a plain 2-layer GCN (402K params, k=96 spatial graph)
reaches **0.943 ARI** on DLPFC cortical-layer assignment, beating scGPT and Geneformer
by a wide margin. `README.md` speculates that a foundation model pretrained on
*spatial* data would carry the right inductive bias and could plausibly beat it.

**Your job is to settle that.** Prior work established the framing:

- `docs/FM_INPUT_SPECS.md` — per-model input specs, and which models structurally
  cannot consume Visium at all.
- `docs/ZEROSHOT_RESULTS.md` — the zero-shot results so far, and the caveat on them.

## Task 1 — re-run everything, baselines first

Do **not** treat the existing numbers as done. Re-run the full benchmark on your
hardware in one clean environment. The reason is provenance, not doubt: the previous
run's environment mutated mid-session (installing `scgpt` downgraded
`pytorch-lightning`; `transformers` was installed at 5.x then downgraded to 4.44). The
numbers are believed clean but were not produced in a single immutable env.

**The baselines are the cross-machine checksum.** They are pure numpy/sklearn with
fixed seeds and should reproduce to ~3 decimals anywhere:

| Baseline | Reference ARI | Reference NMI |
|---|---|---|
| `pca_kmeans` | 0.2275 ± 0.0213 | 0.3345 |
| `spatial_smooth_1hop` | 0.2972 ± 0.0404 | 0.4517 |
| `spatial_smooth_2hop` | 0.3065 ± 0.0431 | 0.4662 |

```bash
python scripts/benchmark_fm_zeroshot.py --method baselines --all
```

**If these do not reproduce, stop and debug.** It means data preparation diverged, and
no model result is meaningful until it's resolved. If they do reproduce, the pipeline
is verified equivalent and everything downstream is trustworthy.

Then re-run the models that already have numbers:

| Model | Reference ARI (native / embedding+KMeans) |
|---|---|
| Novae `brain-0` | 0.2161 ± 0.1648 / 0.2455 ± 0.0826 |
| Novae `human-0` | 0.1841 ± 0.0543 / 0.1711 ± 0.0323 |
| CellPLM 85M `20231027` | — / 0.1815 ± 0.0496 |

```bash
python scripts/benchmark_fm_zeroshot.py --method novae --novae_model prism-oncology/novae-brain-0 --all
python scripts/benchmark_fm_zeroshot.py --method novae --novae_model prism-oncology/novae-human-0 --all
python scripts/benchmark_fm_zeroshot.py --method cellplm --cellplm_ckpt <dir> --all
```

Expect **±0.01-ish wobble** on the FM numbers — CUDA and MPS differ in float
accumulation, embeddings shift slightly, and KMeans flips borderline spots. A move of
0.05 or more is not numerics; investigate it. The baselines should *not* wobble.

CellPLM checkpoints: Dropbox link in the CellPLM README, ~2.3 GB zip, gives
`20230926_85M` and `20231027_85M` (`.best.ckpt` + `.config.json`).

## Task 2 — scGPT-spatial (highest priority)

The only candidate with **Visium in its pretraining corpus** (SpatialHuman30M: Visium,
Visium HD, MERFISH, Xenium). DLPFC is Visium. This is the only in-distribution test of
the entire hypothesis, and it has never been run.

- Code: https://github.com/bowang-lab/scGPT-spatial
- Weights: Figshare, "scGPT-spatial V1 Model Weights"
- Input ready at `data/fm_inputs/scgpt_spatial/` — raw counts, gene symbols in
  `var_names`, `batch`/`str_batch` set. No assay substitution needed.

**The blocker, and why you cannot solve it the easy way.** The `scgpt` pip package
imports `torchtext`. torchtext is archived at 0.18, built against torch 2.3, and fails
to load on modern torch (`OSError: Could not load this library: libtorchtext.so` against
torch 2.10). **An RTX 5090 is Blackwell (sm_120) and requires torch ≥2.7 / CUDA 12.8**,
so downgrading torch is not available to you — the two constraints are mutually
exclusive on this hardware.

Options, best first:

1. **Bypass the package.** Load the checkpoint with `torch.load` and reimplement the
   tokenizer (gene symbol → vocab id, binned expression). The vocab ships as
   `vocab.json` alongside the weights. This is the most robust path.
2. **Stub `torchtext.vocab.Vocab`.** scGPT's `GeneVocab` subclasses it and uses very
   little of its API.
3. **Separate torch-2.3 env, CPU only**, purely to obtain the number. Slow but correct,
   and it sidesteps sm_120 entirely.

Do not silently skip this model. If you cannot run it, report which approach you tried
and the exact failure.

## Task 3 — Nicheformer, swept over three assay tokens

Runs correctly via HuggingFace `trust_remote_code`, but is too slow without a GPU (CPU
did not finish one slice in ~12 min; MPS OOMs above batch 8).

**DLPFC is Visium and Nicheformer has no Visium token** — it is commented out in
`src/nicheformer/data/constants.py`, and the corpus is CosMx/ISS/MERFISH/Xenium. Running
it *requires declaring a technology the data is not*, and the tokenizer's gene-median
normalization vector is technology-specific too.

So run it **three times** and report the spread:

```bash
mkdir -p nf_means
for f in merfish cosmx xenium; do
  curl -sL -o nf_means/${f}_mean_script.npy \
    https://raw.githubusercontent.com/theislab/nicheformer/main/data/model_means/${f}_mean_script.npy
done

python scripts/benchmark_fm_zeroshot.py --method nicheformer \
  --nicheformer_mean nf_means/merfish_mean_script.npy \
  --nicheformer_assay MERFISH --device cuda --batch_size 32 --all
# repeat with cosmx / Xenium
```

**The spread is the deliverable, not the mean.** If ARI moves materially across three
equally-false tokens, the token is driving the result and no single number is
meaningful.

Landmines already solved in `scripts/benchmark_fm_zeroshot.py` — do not re-break them:

- **`transformers` must be 4.x.** 5.x cannot instantiate the custom slow tokenizer
  (`ValueError: Couldn't instantiate the backend tokenizer`). Pin `transformers==4.44.2`.
- **Gene space must be reindexed onto Nicheformer's exact 20,310-gene Ensembl space,
  zero-filling absent genes.** The upstream repo's own `ad.concat(..., join='inner')`
  recipe returns the *intersection*, which desynchronizes the mean vector from the
  embedding table and throws `IndexError: index out of range in self`. See
  `_align_to_nicheformer_space()`. Our data matches 19,790 of 20,310 genes.
- **Attention at 1500 tokens is memory-hungry.** Batch 64 wants ~8.6 GiB and OOMed a
  24 GB budget. 32 should fit 32 GB; drop to 16 if not.

## Task 4 — models never attempted

- **UCE** (36M cells, 8 species) and **scFoundation** (~50M human scRNA-seq, 19,264-gene
  ordered vector, 100M params). Both confirmed **dissociated-only**, so they extend the
  existing negative result rather than testing the spatial hypothesis — but they are
  cheap on a GPU and worth having as trend-line points. Inputs at `data/fm_inputs/uce/`
  and `data/fm_inputs/scfoundation/`. Embeddings only → fixed KMeans readout.
- **SToFM** (https://github.com/PharMolix/SToFM, Google Drive checkpoint, Geneformer
  gene vocabulary, described as "high-resolution ST" — may exclude Visium) and
  **AIDO.Tissue** (biorxiv 2025.07.04.663102). Feasibility unverified. Check whether
  each accepts Visium and has a retrievable checkpoint *before* committing compute, and
  report what you find either way.
- **Novae `scConcept`** (`prism-oncology/novae-scConcept-multi-species`) fails with
  `AssertionError: Too few genes (0) are known/used by the model` — gene ID convention
  mismatch. Try Ensembl via `var['ensembl_id']`.

## Setup

```bash
git clone https://github.com/Ttopiac/spatial-omics-fusion.git
cd spatial-omics-fusion
bash setup_env.sh && conda activate spatial-omics   # then upgrade torch for sm_120
python data/download_dlpfc.py --all                      # ~1.2 GB raw
python scripts/prepare_fm_inputs.py --target all --all   # ~2.1 GB, all FM inputs
python scripts/benchmark_fm_zeroshot.py --method baselines --all
```

## Rules

- **Preprocessing is allowed. Post-processing is not.** Format inputs however each model
  requires — that is the job. Do not tune clustering, sweep resolutions, cherry-pick
  checkpoints, or report the best of several readouts. One fixed
  `KMeans(n_classes, random_state=42, n_init=10)` per embedding model, identical across
  models. Where a model has a native domain output, report that.
- **Zero-shot means no labels anywhere**, except to compute ARI/NMI at the end.
- **All 12 slices, always.** Report mean ± std. Flag any model with std > ~0.10 as
  regime-dependent rather than reporting the mean alone — Novae `brain-0` goes
  *negative* on two slices, and its mean hides that.
- Record the exact checkpoint id and gene-match count for every run.
- **If a model cannot be run, report it as blocked with the specific error.** Do not
  quietly drop it from the table. A documented blocker is a result; a missing row is not.

## Deliverable

1. `results/zeroshot/{model}.json` per model — the script writes these.
2. An updated `docs/ZEROSHOT_RESULTS.md`: every model you ran, every model you could
   not and why, and whether the baselines reproduced.
3. Any new input-spec landmine appended to `docs/FM_INPUT_SPECS.md`.

Keep the existing Mac numbers as a replication target rather than overwriting them.
Two independent stacks agreeing is stronger evidence than either run alone.
