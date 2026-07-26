# Zero-Shot Spatial Domain Benchmark — Results

Produced by `scripts/benchmark_fm_zeroshot.py`. Raw scores in `results/zeroshot/*.json`.

**This is a different task from `docs/RESULTS.md`.** There the GCN is trained on 60% of
labeled spots and reaches 0.943 ARI. Here *no labels are used at all* — each method
either assigns domains directly or emits an embedding that is read out with a fixed
`KMeans(n_classes)`. Labels enter only to compute ARI/NMI afterwards. **The numbers on
this page are not comparable to the 0.943.**

## Protocol

- All 12 DLPFC slices, every time. Mean ± std reported across slices.
- Raw counts, full 33,538-gene space, from `data/fm_inputs/{target}/{slice}.h5ad`
  (see `scripts/prepare_fm_inputs.py`). Each model applies its own normalization.
- **Preprocessing is allowed; post-processing is not.** Inputs are formatted to each
  model's requirements. Nothing is tuned on the output side: one fixed
  `KMeans(n_classes, random_state=42, n_init=10)` for every embedding-only model, no
  resolution sweeps, no checkpoint cherry-picking, no best-of-N readouts.
- Where a model has a *native* zero-shot domain output, that is what gets reported.
  Only Novae has one.

## Results

Run on Apple Silicon (MPS/CPU), July 2026.

| Method | Type | ARI | NMI |
|---|---|---|---|
| **`spatial_smooth_2hop`** | **baseline** | **0.3065 ± 0.0431** | 0.4662 |
| `spatial_smooth_1hop` | baseline | 0.2972 ± 0.0404 | 0.4517 |
| Novae `brain-0` | FM, embedding + KMeans | 0.2455 ± 0.0826 | 0.3714 |
| `pca_kmeans` | baseline, no spatial | 0.2275 ± 0.0213 | 0.3345 |
| Novae `brain-0` | **FM, native domains** | 0.2161 ± 0.1648 | 0.3579 |
| Novae `human-0` | FM, native domains | 0.1841 ± 0.0543 | 0.2377 |
| CellPLM 85M `20231027` | FM, embedding + KMeans | 0.1815 ± 0.0496 | 0.3051 |
| Novae `human-0` | FM, embedding + KMeans | 0.1711 ± 0.0323 | 0.2385 |

### Baselines

- `pca_kmeans` — log-normalize, 3000 HVGs, PCA(50), KMeans. **No spatial information.**
- `spatial_smooth_1hop` — the above, then average each spot's PCA vector with its 6
  spatial neighbors once, then KMeans.
- `spatial_smooth_2hop` — two rounds of that averaging. The unsupervised stand-in for
  what a 2-layer GCN does.

The 0.2275 → 0.3065 gap is the value of spatial context alone, with no learning and no
pretraining. It is the floor any *spatial* foundation model must clear to have
demonstrated that its spatial pretraining does work.

## Findings

**1. No foundation model beat the trivial spatial baseline.** Nothing tested reached
0.3065. Nothing reached the 1-hop variant either.

**2. Only one beat the no-spatial baseline, and only on a technicality.** Novae
`brain-0` clears `pca_kmeans` (0.2275) when its latent is KMeans'd — but its *native*
domain output, the thing it is actually designed to produce, does not (0.2161).

**3. Novae `brain-0` is unstable.** ±0.1648 std on the native output, with **negative
ARI** on slices 151669 (−0.1232) and 151670 (−0.0531) — worse than random assignment.
It performs acceptably on the 7-class slices and collapses on the 5-class ones. The
mean alone is misleading here; treat this model as regime-dependent, not as a 0.216 model.

**4. Brain-specific pretraining helped, tissue-matched still lost.** `brain-0` (0.2161)
beats `human-0` (0.1841) on native output, so matching the tissue does buy something.
It is not enough to reach a 5-line baseline.

**5. Only Novae does zero-shot domain inference at all.** CellPLM, Nicheformer, UCE and
scFoundation emit per-spot embeddings only. They have no native concept of a spatial
domain, so *some* clustering is unavoidable just to read their output. That asymmetry
is itself a result about what these models are for.

## Not yet run

| Model | Status |
|---|---|
| **scGPT-spatial** | **Blocked.** The `scgpt` package imports `torchtext`, archived at 0.18 (torch 2.3), which fails to load against torch 2.10 (`OSError: Could not load this library: libtorchtext.so`). Needs the package bypassed or a separate env. **Highest priority — the only candidate with Visium in its pretraining corpus.** |
| **Nicheformer** | Runs correctly (tokenization and gene alignment solved, see `benchmark_fm_zeroshot.py`), but CPU inference did not finish one slice in ~12 min and MPS OOMs above batch 8. Needs a GPU. Must be swept over 3 assay tokens — DLPFC is Visium and Nicheformer has no Visium token. |
| Novae `scConcept` | Fails: `AssertionError: Too few genes (0) are known/used by the model`. Gene ID convention mismatch, unresolved. |
| UCE, scFoundation | Not attempted. Confirmed dissociated-only, so these extend the existing negative result rather than testing the spatial hypothesis. |
| SToFM, AIDO.Tissue | Feasibility unverified — unknown whether either accepts Visium or has a retrievable checkpoint. |

## Caveat on this run

The environment mutated during the session: installing `scgpt` downgraded
`pytorch-lightning` 2.6.5 → 1.9.5 (restored afterwards), and `transformers` was
installed at 5.14 then downgraded to 4.44. The Novae runs predate that churn and
CellPLM postdates the restore, so these numbers are believed clean — but they were not
produced in a single immutable environment. They should be treated as a **replication
target**, not as final. See `docs/GPU_HANDOFF.md`.
