# GENESIS — Reproduction Plan

**Last updated**: 2026-08-11  
**Study**: Ordinal Contrastive Learning Unifies Potency Representation, Molecular Generation, and Drug Repurposing for Cancer Therapy  
**Hardware used**: Apple M3 MacBook Pro (11-core CPU, 14-core GPU, 18 GB unified memory)  
**Core pipeline runtime**: ~8.6 hours (single seed) | ~21.1 hours (5 seeds + ablations)

---

## 1. Prerequisites

### 1.1 Hardware
- macOS with Apple Silicon (MPS backend) **or** Linux with CUDA GPU  
- ≥ 16 GB RAM; ≥ 25 GB free disk

### 1.2 Software
```bash
Python        3.12.3
torch         2.4.0   (MPS or CUDA)
transformers  4.41.2  # MolDeBERTa
selfies       2.1.2
rdkit         2025.09.6
scikit-learn  1.8.0
scipy         1.11.4
pandas        2.1.4
numpy         1.26.4
python-docx   (optional, for manuscript scripts)
AutoDock Vina 1.2.x   # installed separately; must be on PATH as 'vina'
```

Install Python dependencies:
```bash
pip install torch transformers selfies rdkit scikit-learn scipy pandas numpy python-docx
```

### 1.3 Required input files (must be present before starting)

| File | Location | Description |
|------|----------|-------------|
| `All_Compounds_Unq_Std.tsv` | `data/` | Master dataset — 72,162 compounds, 47 columns. |
| `tier1_confirmed_inactives.tsv` | `negatives/` | T1 negatives (20,237 rows) |
| `tier2_other_cancer_actives.tsv` | `negatives/` | T2 negatives (41,909 rows) |
| `tier3a_activity_cliff_pairs.tsv` | `negatives/` | T3A negatives (15,471 rows) |
| `tier3b_scaffold_matched.tsv` | `negatives/` | T3B negatives (478,786 rows) |
| `tier3c_target_switching.tsv` | `negatives/` | T3C negatives (2,363 rows) |
| `tier4_background.tsv` | `negatives/` | T4 negatives (100,000 rows) |
| Crystal structures (PDB) | `docking/structures/` | 8 OSCC target PDB files (see §8) |

> **Note on GENESIS_NP IDs**: 255 natural product compounds carry internal IDs (`GENESIS_NP_00001` … `GENESIS_NP_00255`). All compounds are traceable via their PubChem CID in `All_Compounds_Unq_Std.tsv`.

---

## 2. Phase 0 — Embedding Extraction (~45 min)

Extract 768-d MolDeBERTa CLS embeddings for all 72,162 compounds.

```bash
python scripts/moldeberta_embedding_extract.py \
    --input  data/All_Compounds_Unq_Std.tsv \
    --outdir embeddings/ \
    --device mps          # or cuda / cpu
```

**Outputs**: `embeddings/compound_embeddings.npy`, `embeddings/compound_ids.npy`, `embeddings/progress.json`

---

## 3. Phase 1 — Data Preparation (~3 min)

Build positive and negative training sets, compute Murcko scaffolds, and create scaffold-aware splits.

```bash
# Positives (49,312 compounds with pChEMBL annotations)
python modeling/prepare_positives.py

# Negatives (643,332 across 6 tiers with curriculum weights)
python modeling/prepare_negatives.py

# Murcko scaffolds for scaffold-aware splitting
python modeling/compute_scaffolds.py

# Train / validation / test splits (70 / 15 / 15, zero scaffold leakage)
python modeling/create_splits.py

# Nearest-neighbour embedding index for hard-negative mining
python modeling/build_index.py
```

**Outputs**: `modeling/positives.tsv`, `modeling/negatives.tsv`, `modeling/scaffolds.tsv`, `modeling/splits.json`, `modeling/embedding_index.bin`

---

## 4. Phase 2 — PGCL Contrastive Training (~72 min, 3 seeds)

Train the Potency-Graded Contrastive Learning model. Three seeds are used for statistical reporting of PGCL metrics.

```bash
for SEED in 42 123 456; do
    python modeling/train.py \
        --seed $SEED \
        --epochs 25 \
        --tau 0.07 \
        --dim 128 \
        --device mps
done
```

**Outputs**: `modeling/checkpoints/best_pgcl_seed{42,123,456}.pt`, `modeling/validation/multiseed/logs/multiseed_summary.json`

**Key reported metrics** (from `multiseed_summary.json`):
- Spearman ρ = 0.555 ± 0.003 | Cliff accuracy = 66.4 ± 0.3%
- Cohen's d = 1.542 ± 0.007 | Target-switch AUC = 0.967 ± 0.001

Use seed 42 checkpoint as the canonical model for all downstream phases.

---

## 5. Phase 3 — Generative Model Training & Evaluation (~284 min/seed × 5 seeds)

Train and evaluate all 6 generative architectures. Five seeds are used for primary endpoint statistical testing.

```bash
# Run all 6 methods for a single seed (VAE, cVAE, RL-VAE, Diffusion, CGLF, CRAVE)
python generative/run_all.py --seed 42 --device mps

# Repeat for remaining seeds
python generative/run_all.py --seed 123 --device mps
python generative/run_all.py --seed 456 --device mps
python generative/run_all.py --seed 789 --device mps
python generative/run_all.py --seed 1337 --device mps

# Aggregate multi-seed results
python generative/multiseed_gen.py --aggregate
```

**Outputs**: `generative/checkpoints/`, `generative/outputs/seed_*/`, `generative/multiseed/aggregate/multiseed_summary.json`

**Key reported metrics** (5-seed, CRAVE filtered):
- OSCC potency = 0.771 ± 0.033 | QED = 0.784 ± 0.043 | Uniqueness = 1.000 ± 0.000
- All 5 seeds converge to a single continuous potency regime (0.715-0.796);
  CRAVE significantly exceeds VAE/cVAE/CGLF after Bonferroni correction
  (one-sided Mann-Whitney, p_Bonferroni = 0.016)

> To reproduce a **quick single-seed result only**, run seed 42 alone (~284 min). Full 5-seed evaluation takes ~24 hours for Phase 3 alone.

---

## 6. Phase 4 — Bioactivity Profiling (~2 min)

Screen all 7,000 generated molecules (5,416 unique after deduplication) through the 7-component bioactivity pipeline.

```bash
python generative/evaluate_gen.py \
    --generated-dir generative/outputs/ \
    --pgcl-checkpoint modeling/checkpoints/best_pgcl_seed42.pt \
    --outdir results/phase4/
```

**Outputs**: `results/phase4_bioactivity_profiling.md`, consensus compound lists  
**Key results**: 5,416 unique molecules; 1,031 consensus compounds; 14.6% safety PASS rate

---

## 7. Phase 5 — Validation & Comparative Baselines (~12.5 hours)

Runs the full evaluation battery (VS enrichment, comparative baselines, ablations, cross-cancer transfer). This is the longest phase.

```bash
# All 5 baselines (ECFP4 k-NN, QSAR RF/XGB/LR, Flat SupCon, BCE)
python modeling/baselines/run_all_baselines.py

# Ablation study (7 configurations × 3 seeds)
python modeling/run_ablations.py

# Virtual screening enrichment (PGCL vs ECFP4)
python scripts/virtual_screening_enrichment.py

# Cross-cancer transfer: OSCC → LUAD (requires ChEMBL MySQL at localhost)
python modeling/baselines/cross_cancer_transfer.py \
    --source oscc \
    --target luad \
    --pgcl-checkpoint modeling/checkpoints/best_pgcl_seed42.pt
```

**Key results**:
- Ablation: removing PotencyLoss drops ρ by 18.1%; removing TierMarginRanking drops NDCG@100 by 10.8%
- BEDROC: PGCL 2.52 vs ECFP4 1.90
- Cross-cancer: OSCC→LUAD Spearman ρ = 0.502 (n=29,675 LUAD compounds, 16 targets)

> The cross-cancer transfer requires a local ChEMBL 36 MySQL instance (`localhost`, user=root). Skip if unavailable; result is already in `modeling/baselines/results/cross_cancer/`.

---

## 8. Phase 6 — Drug Repurposing (~included in Phase 5 timing)

Screen 23,308 CompoundList compounds through the 8-component GENESIS scoring system.

```bash
python repurposing/run_drug_repurposing.py \
    --compound-list data/All_Compounds_Unq_Std.tsv \
    --pgcl-checkpoint modeling/checkpoints/best_pgcl_seed42.pt \
    --outdir repurposing/results/
```

**Outputs**: `repurposing/results/global_rankings.tsv`, `repurposing/results/final_dossier.tsv`  
**Key result**: 156 repurposing candidates

---

## 9. Phase 7 — Molecular Docking (~106 min total)

Requires **AutoDock Vina ≥ 1.2** on PATH and 8 OSCC crystal structure PDB files.

### 9.1 Crystal structures required

| Target | PDB ID | Resolution |
|--------|--------|------------|
| FGFR1 | 4V04 | 2.0 Å |
| EGFR | 1IVO | 2.6 Å |
| AKT1 | 4EKK | 2.0 Å |
| PIK3CA | 2RD0 | 3.0 Å |
| mTOR | 4JSV | 3.2 Å |
| MDM2 | 4HG7 | 2.2 Å |
| PTGS2 | 3LN1 | 2.7 Å |
| STAT3 | 6NUQ | 2.4 Å |

Download from https://www.rcsb.org/ and place prepared `.pdbqt` files in `docking/structures/`.

### 9.2 Run docking

```bash
# Single-target docking (364 elite compounds × 8 targets)
python docking/run_molecular_docking.py \
    --compounds results/phase4/elite_compounds.tsv \
    --structures docking/structures/ \
    --exhaustiveness 16 --n-poses 9 --seed 42 \
    --outdir docking/results/

# Multi-target docking (50 elite × 7 targets; excludes STAT3)
python docking/run_multitarget_docking.py \
    --compounds docking/results/top50_elite.tsv \
    --structures docking/structures/ \
    --outdir docking/results/
```

**Key results**: 361/364 successful docks (99.2%); 77.7% achieve ΔG ≤ −7.0 kcal/mol; best = −11.2 kcal/mol (FGFR1); multi-target mean = −8.75 kcal/mol

---

## 10. Phase 8 — ADMET Profiling (~< 1 min)

```bash
python admet/run_admet_profiling.py \
    --input docking/results/docking_results.tsv \
    --outdir admet/results/
```

**Key results**: 375 compounds; 92% Lipinski Ro5; 70.1% mucosal SAFE; 1.1% PAINS

---

## 11. Phase 9b — Lead Optimization (~1 min)

Scaffold-constrained analog generation from the 13 triple-validated elite compounds.

```bash
python scripts/compute_t5_t9.py --phase 9b \
    --elite admet/results/final_candidates.tsv \
    --pgcl-checkpoint modeling/checkpoints/best_pgcl_seed42.pt
```

**Key results**: 69 analogs; 0% ADMET FAIL; mean ΔADMET = +0.276; 95% novel scaffolds

---

## 12. Phase 10 — Final Ranking (~2 min)

```bash
python scripts/compute_t5_t9.py --phase 10 \
    --docking   docking/results/docking_results.tsv \
    --multitarget docking/results/multitarget_binders.tsv \
    --admet     admet/results/final_candidates.tsv \
    --repurposing repurposing/results/global_rankings.tsv \
    --outdir    results/phase10/
```

**Key results**: 316 recommended compounds; 46 Tier-1 dossiers; 13 triple-validated elite; top-ranked overall candidate is OMIPALISIB (REP_CHEMBL1236962); GEN_00182 is the top-ranked de novo generated compound (rank 5 overall)

---

## 13. Figures

```bash
# All main and supplementary figures
python scripts/create_all_paper_figures.py --outdir Images/

# External MOSES/GuacaMol benchmarks (optional)
python scripts/compute_external_benchmarks.py
python scripts/compute_fcd.py
```

---

## 14. Runtime Summary

| Phase | Script | Time |
|-------|--------|-----:|
| 0 — Embedding extraction | `scripts/moldeberta_embedding_extract.py` | 45 min |
| 1 — Data preparation | `modeling/prepare_*.py`, `create_splits.py`, `build_index.py` | 3 min |
| 2 — PGCL training (3 seeds) | `modeling/train.py` | 72 min |
| 3 — Generative (per seed) | `generative/run_all.py` | 284 min |
| 4 — Bioactivity profiling | `generative/evaluate_gen.py` | 2 min |
| 5 — Baselines + ablations | `modeling/baselines/run_all_baselines.py`, `run_ablations.py` | 750 min |
| 6 — Drug repurposing | `repurposing/run_drug_repurposing.py` | ~5 min |
| 7a — Single-target docking | `docking/run_molecular_docking.py` | 75 min |
| 7b — Multi-target docking | `docking/run_multitarget_docking.py` | 31 min |
| 8 — ADMET profiling | `admet/run_admet_profiling.py` | < 1 min |
| 9b — Lead optimization | `scripts/compute_t5_t9.py --phase 9b` | 1 min |
| 10 — Final ranking | `scripts/compute_t5_t9.py --phase 10` | 2 min |
| **Core total (seed 42 only)** | | **~8.6 h** |
| **Full total (5 seeds + ablations)** | | **~21.1 h** |

Peak memory: ~10 GB (Phase 0 embedding extraction)

---

## 15. Expected Output Structure

```
GENESIS/
├── embeddings/          compound_embeddings.npy, compound_ids.npy
├── modeling/
│   ├── checkpoints/     best_pgcl_seed{42,123,456}.pt
│   ├── positives.tsv    49,312 rows
│   ├── negatives.tsv    643,332 rows
│   ├── splits.json
│   └── baselines/results/
├── generative/
│   ├── checkpoints/
│   └── multiseed/aggregate/multiseed_summary.json
├── repurposing/results/ global_rankings.tsv (23,308 rows), final_dossier.tsv
├── docking/results/     docking_results.tsv (364 rows), multitarget_binders.tsv
├── admet/results/       final_candidates.tsv (316 rows)
├── results/             phase{1..10}_*.md summary reports
└── Images/              Main_Figures/ (10), Supplementary/ (11)
```

---

## 16. Skipping Completed Phases

All result files are already present in this repository. To reproduce only a specific phase, all upstream outputs are available:

- To reproduce from **Phase 2** onwards: embeddings and prepared data are ready in `embeddings/` and `modeling/`
- To reproduce from **Phase 4** onwards: PGCL checkpoint `modeling/checkpoints/best_pgcl_seed42.pt` is ready
- To reproduce from **Phase 7** onwards: generated compound lists are in `generative/outputs/` and `results/phase4/`
- To verify final numbers only: all results are in `results/` and `docking/results/` without re-running anything

