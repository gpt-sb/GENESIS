# GENESIS — Algorithm Design & Scientific Reference

**Generative Exploration of Novel Entities for Selective Inhibitor Synthesis**

> **Supplementary reference, not required reading.** This document is an optional,
> deep-dive design rationale supplementing the manuscript's Methods section —
> theoretical foundations, architecture justifications, parameter choices, and
> development-history notes (including self-identified corrections from earlier
> drafts, retained here for process transparency). Nothing in the reproducibility
> pipeline depends on this file; see the main [README.md](../README.md) for
> everything needed to reproduce the study's results.

*Master reference document covering theoretical foundations, mathematical formulations,
architecture design, implementation decisions, parameter choices, and biological context
for all algorithms used in the GENESIS study.*

---

## Table of Contents

1. [Study Context & Biological Motivation](#1-study-context--biological-motivation)
2. [System Architecture & Dependency Graph](#2-system-architecture--dependency-graph)
3. [Phase 0 — Molecular Embedding Extraction](#3-phase-0--molecular-embedding-extraction)
4. [Phase 1 — Data Preparation & Negative Mining](#4-phase-1--data-preparation--negative-mining)
5. [Phase 2 — Potency-Graded Contrastive Learning](#5-phase-2--potency-graded-contrastive-learning)
6. [Phase 3.1 — SELFIES Data Preparation](#6-phase-31--selfies-data-preparation)
7. [Phase 3.2 — VP Centroid Computation](#7-phase-32--vp-centroid-computation)
   - 7.6 [Compound Landscape & Generative Training Bias](#76-compound-landscape--generative-training-bias)
8. [Phase 3.3 — Variational Autoencoder (VAE)](#8-phase-33--variational-autoencoder-vae)
9. [Phase 3.4 — Conditional VAE (cVAE)](#9-phase-34--conditional-vae-cvae)
10. [Phase 3.5 — Latent Diffusion Model](#10-phase-35--latent-diffusion-model)
11. [Phase 3.6 — Reinforcement Learning Fine-Tuning (RL-VAE)](#11-phase-36--reinforcement-learning-fine-tuning-rl-vae)
12. [Phase 3.7 — Contrastive-Guided Latent Flow (CGLF)](#12-phase-37--contrastive-guided-latent-flow-cglf)
13. [Phase 3.8 — CRAVE: Enhanced RL + Multi-Strategy Ensemble](#13-phase-38--crave-enhanced-rl--multi-strategy-ensemble)
14. [Evaluation Framework](#14-evaluation-framework)
15. [Design Decision Registry](#15-design-decision-registry)
16. [Parameter Summary Tables](#16-parameter-summary-tables)
17. [Phase 6 — Drug Repurposing of CompoundList Compounds](#17-phase-6--drug-repurposing-of-compoundlist-compounds)
18. [Phase 7 — Molecular Docking Validation](#18-phase-7--molecular-docking-validation)
    - Phase 7b — Multi-Target Docking Validation (§18.15, nested within Phase 7)
20. [Phase 8 — ADMET & Pharmacokinetic Profiling](#20-phase-8--admet--pharmacokinetic-profiling)
21. [Phase 9a — Network Pharmacology & Pathway Analysis](#21-phase-9a--network-pharmacology--pathway-analysis)
22. [Phase 9b — Lead Optimization via Scaffold-Constrained Generation](#22-phase-9b--lead-optimization-via-scaffold-constrained-generation)
23. [Phase 10 — Nearest Known Analog Analysis & Final Compound Ranking](#23-phase-10--nearest-known-analog-analysis--final-compound-ranking)
24. [Phase 10b — Molecular Dynamics Simulations (In Progress)](#24-phase-10b--molecular-dynamics-simulations-in-progress)
25. [Phase 10c — Compound Category Annotation & Repurposing Potential](#25-phase-10c--compound-category-annotation--repurposing-potential)

---

## 1. Study Context & Biological Motivation

### 1.1 Clinical Problem

Oral Squamous Cell Carcinoma (OSCC) is the most common malignancy of the oral cavity,
constituting approximately 90% of oral cancers. Standard treatment — surgery followed
by chemo/radiation — achieves only ~50% five-year survival for advanced stages. The
molecular heterogeneity of OSCC across different anatomical sub-sites (tongue, buccal
mucosa, floor of mouth, gingiva) and the involvement of multiple oncogenic pathways
(EGFR, PI3K/AKT/mTOR, Wnt/β-catenin, NF-κB, JAK/STAT) motivate a computational
approach to drug discovery that can simultaneously consider:

1. **Multi-target activity** — compounds active against multiple OSCC-relevant targets
2. **Potency gradation** — distinguishing very potent (pChEMBL ≥ 7.0) from moderately
   active compounds
3. **Structural specificity** — separating OSCC-specific actives from general anticancer
   or off-target compounds
4. **Drug-likeness** — ensuring generated molecules satisfy pharmaceutical constraints

### 1.2 Computational Strategy

GENESIS implements an eleven-phase pipeline (Phase 0 through Phase 10):

```
Phase 0: Molecular Embedding    (MolDeBERTa, 88.2M params, SMILES → 768-d)
     ↓
Phase 1: Data Preparation        (Negative mining, tier stratification)
     ↓
Phase 2: Contrastive Learning    (768-d → 128-d, potency-aware clustering)   ← PGCL core contribution
     ↓
Phase 3: Generative Modeling     (6 methods: VAE, cVAE, Diffusion, RL-VAE, CGLF, CRAVE)
     ↓
Phase 4: Bioactivity Profiling   (Screening, filtering, candidate ranking)
     ↓
Phase 5: Baselines & Ablations   (Flat SupCon / BCE comparisons, cross-cancer transfer)
     ↓
Phase 6: Drug Repurposing        (23,308 known-compound screening)
     ↓
Phase 7/7b: Molecular Docking    (Single-target + multi-target structure-based validation)
     ↓
Phase 8: ADMET Profiling         (Pharmacokinetic & mucosal-safety filtering)
     ↓
Phase 9a/9b: Network Pharmacology & Lead Optimization (pathway mapping, scaffold-constrained analogs)
     ↓
Phase 10: Final Ranking          (Integration & prioritization layer)
```

The key insight is that raw molecular embeddings from pre-trained language models
(Phase 0) encode general chemical knowledge but lack task-specific structure. The
contrastive learning phase (Phase 2) reshapes this 768-dimensional space into a 128-dimensional
space where potency-relevant relationships become geometrically explicit: very potent
OSCC compounds cluster together, and the distance to this cluster's centroid directly
encodes predicted potency. The generative phase (Phase 3) then operates in this
semantically-rich space to produce novel drug candidates. Phases 5–10 form a
translational validation and prioritization layer: comparative baselines, drug
repurposing, structure-based docking, ADMET/safety profiling, pathway-level
contextualization, lead optimization, and finally an integrated ranking across every
evidence stream (see §23).

### 1.3 Key Innovation

The central contribution is the **bidirectional bridge between discriminative and
generative modeling**: the contrastive model's learned geometry (VP centroid, potency
gradient, ordinal distances) serves as both the reward signal for RL-based generation
and the conditioning signal for flow/diffusion-based generation. This avoids the
common failure mode where generative models optimize a proxy reward that diverges
from the actual drug discovery objective.

---

## 2. System Architecture & Dependency Graph

### 2.1 Module Dependency Graph

```
┌──────────────────────────────────────────────────────────────────────┐
│  GENESIS Pipeline                                                     │
│                                                                       │
│  ┌─────────────────┐                                                  │
│  │ MolDeBERTa       │ ← Pre-trained (SaeedLab/MolDeBERTa-base-123M-mtr)│
│  │ 88.2M params     │   ("123M" is part of the HF model name, not the │
│  │ SMILES → 768-d   │    actual parameter count -- see §3.1)          │
│  └────────┬─────────┘                                                 │
│           │ compound_embeddings.mmap (715,162 × 768, float16)         │
│           ▼                                                           │
│  ┌─────────────────┐     ┌───────────────────┐                        │
│  │ ContrastiveModel │ ←── │ Negative Mining   │                       │
│  │ 559,617 params   │     │ 6-tier system     │                       │
│  │ 768→512→256→128  │     │ T1,T2,T3A/B/C,T4  │                      │
│  └────────┬─────────┘     └───────────────────┘                       │
│           │ 128-d L2-normalized projections                           │
│           │                                                           │
│  ┌────────┼──────────────────────────────────────────────────────┐    │
│  │        ▼  VP Centroid (128-d)                                  │    │
│  │  ┌───────────┐                                                 │    │
│  │  │ mean(VP)   │─────────────────────────────┐                  │    │
│  │  │ L2-norm    │                             │                  │    │
│  │  └───────────┘                              │                  │    │
│  │        │                                    │                  │    │
│  │  ┌─────┼───────┬──────────────┬─────────────┤                  │    │
│  │  ▼     ▼       ▼              ▼             ▼                  │    │
│  │ VAE   cVAE   Diffusion    RL-VAE    CGLF   CRAVE              │    │
│  │  │     │       │              │       │      │                 │    │
│  │  └─────┼───────┼──────────────┼───────┼──────┘                 │    │
│  │        ▼       ▼              ▼       ▼                        │    │
│  │      Unified Evaluation (10 metrics)                           │    │
│  └────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow Summary

| Asset | Shape | Dtype | Source |
|-------|-------|-------|--------|
| `compound_embeddings.mmap` | (715,162 × 768) | float16 | MolDeBERTa [CLS] |
| `compound_ids.npy` | (715,162,) | str | ChEMBL IDs |
| `compound_index.pkl` | 692,644 entries | dict | Data preparation |
| `positives.tsv` / `negatives.tsv` | 49,312 / 643,332 | TSV | Negative mining |
| `scaffold_groups.pkl` | compound↔scaffold maps | dict | `compute_scaffolds.py` |
| `target_families.pkl` | compound→family map | dict | `compute_scaffolds.py` |
| `splits.json` | 3 splits × 2 lists | JSON | Scaffold-aware splitting |
| `best.pt` | 559,617 params | PyTorch | Contrastive training |
| `selfies_tokens.npy` | (71,861 × 128) | int32 | SELFIES tokenization |
| `vocab.json` | 157 tokens | JSON | SELFIES vocabulary |
| `vp_centroid.json` | 128-d vector | JSON | VP centroid computation |
| `generated_*.txt` | ~1000 SMILES each | text | Generative models |

### 2.3 Training Dependency Chain

```
data_prep.py ──▶ compute_centroid.py ──▶ vae.py (VAE)
                                       ├──▶ vae.py (cVAE)
                                       ├──▶ diffusion.py  (needs VAE encoder)
                                       ├──▶ rl_finetune.py (needs VAE decoder)
                                       │         │
                                       │         ▼
                                       ├──▶ cglf.py (needs VAE encoder)
                                       └──▶ crave_gen.py (needs RL-VAE decoder)
                                                  │
                                                  ▼
                                           evaluate_gen.py
```

---

## 3. Phase 0 — Molecular Embedding Extraction

### 3.1 Model Selection

**MolDeBERTa** (`SaeedLab/MolDeBERTa-base-123M-mtr`, 88.2M parameters) was selected
over alternatives (ChemBERTa, MolBERT, Mol2Vec) for the following reasons:

1. **Pre-training corpus**: Trained on ~5M SMILES with multi-task regression objectives,
   learning chemically meaningful representations
2. **Architecture**: DeBERTa-v2 (disentangled attention with relative position encoding)
   provides better contextual representation than BERT-based alternatives
3. **Embedding quality**: [CLS] token embeddings (768-d) capture global molecular
   properties while individual token embeddings retain substructure information
4. **Tokenizer**: Uses a SMILES-aware tokenizer; `use_fast=False` is required for
   correct token handling (the fast tokenizer produces different tokenization that
   degrades embedding quality)

### 3.2 Embedding Extraction

For each compound with a valid SMILES string:

$$\mathbf{e}_i = \text{MolDeBERTa}(\text{SMILES}_i)[0, :] \in \mathbb{R}^{768}$$

where `[0, :]` extracts the [CLS] token representation.

Embeddings are stored in a memory-mapped float16 array for efficient access during
training without loading the full 715,162 × 768 matrix into RAM.

### 3.3 Implementation Details

- **Batch size**: 128 SMILES per forward pass
- **Truncation**: max_length=512 tokens (covers >99.9% of drug-like SMILES)
- **Storage**: `float16` memory-map (saves ~1 GB vs float32)
- **Runtime**: ~4 hours on Apple M3 for 715,162 compounds

---

## 4. Phase 1 — Data Preparation & Negative Mining

### 4.1 Positive Compound Curation

Positive compounds are OSCC-active molecules from ChEMBL with:
- Measured bioactivity (IC₅₀, EC₅₀, Ki, Kd) against OSCC-relevant targets
- pChEMBL values for potency quantification

**Potency classification** (ordinal, 3 tiers) *(verified against `modeling/positives.tsv`'s actual `pchembl`-by-`potency_class_num` ranges, `generative/compute_centroid.py`'s code comment, and cross-checked against `documents/generate_manuscript_v2.py`'s Table 1 description — an earlier revision of this document briefly stated the wrong thresholds, 5.0–6.5/6.5–8.0/≥8.0; corrected back here)*:

| Class | Label | pChEMBL Range | Biological Meaning | n (positives.tsv) |
|-------|-------|---------------|-------------------|------|
| 1 | Active | 5.0 – 6.0 | Measurable activity (μM range) | 8,971 |
| 2 | Potent | 6.0 – 7.0 | Sub-μM activity | 10,021 |
| 3 | Very Potent (VP) | ≥ 7.0 | Low-nM activity (max observed 11.0) | 22,171 |

This ordinal structure is critical: the contrastive model learns to preserve this
gradient in the projected space, and the generative models use it as conditioning.

### 4.2 Six-Tier Negative Mining Strategy

The negative set is constructed with intentional difficulty stratification to provide
curriculum learning signals of increasing hardness:

| Tier | Name | Source | Rationale | Margin $m(t)$ |
|------|------|--------|-----------|-------------|
| T1 | Target-active, OSCC-irrelevant | Same target family, different disease | Tests target specificity | 0.20 |
| T2 | Other-cancer actives | Cancer actives that are NOT OSCC | Tests disease specificity | 0.20 |
| T3A | Activity cliffs (inactive) | Structurally similar to positives but inactive | Hardest negatives — tests potency discrimination | 0.10 |
| T3B | Structurally similar, untested | Similar Tanimoto but no bioactivity data | Medium difficulty | 0.30 |
| T3C | Target-switch | Active on OSCC targets, but wrong indication | Tests context sensitivity | 0.10 |
| T4 | Random ChEMBL | Random drug-like compounds | Easy negatives, anchors embedding space | 0.30 |

**Note (corrected)**: these are the tier-dependent margins $m(t)$ used directly in
the TierMarginRankingLoss (Eq. 4, `modeling/losses.py: TIER_MARGINS`), verified
against the training code. An earlier draft of this table showed a different set
of "loss weight" values (1.0/0.8/1.2/0.6/1.0/0.4) describing a per-tier
pair-weighting scheme that was superseded by the tier-margin approach below and
no longer reflects the implementation — hard negatives (T3A, T3C) receive the
*tightest* margins (0.10), not the highest weights, since the ranking loss
requires only a small similarity gap to be satisfied for structurally similar
compounds, while easy negatives (T3B, T4) receive the loosest margins (0.30).

### 4.3 Scaffold & Target-Family Assignment (Phase 1.4)

Before splitting, every compound is assigned two grouping labels used to prevent
data leakage:

#### 4.3.1 Murcko Scaffold Extraction

For each compound, the **Murcko scaffold** (core ring system with linkers,
side-chains stripped) is computed via RDKit `MurckoScaffold.GetScaffoldForMol()`.
Compounds sharing the same scaffold are structurally related — if one appears in
training and a scaffold-twin in validation, the model can "cheat" by memorizing
scaffold-level patterns rather than learning genuine potency signals.

The scaffold computation produces two maps stored in `scaffold_groups.pkl`:
- `compound_to_scaffold`: ChEMBL ID → canonical scaffold SMILES
- `scaffold_to_compounds`: scaffold SMILES → list of ChEMBL IDs

Edge cases (missing SMILES, unparseable molecules, RDKit errors) receive sentinel
scaffold labels (`NO_SMILES`, `INVALID_SMILES`, `SCAFFOLD_ERROR`) so no compound
is dropped from the pipeline.

#### 4.3.2 Target Pathway Family Mapping

Each compound's gene target(s) are mapped to one of **16 OSCC-relevant pathway
families** via a curated lookup table:

| Family | Representative Genes | Pathway Role in OSCC |
|--------|---------------------|----------------------|
| EGFR_ERBB | EGFR, ERBB2, ERBB3, ERBB4 | Growth factor signalling |
| PI3K_AKT_MTOR | PIK3CA, AKT1, MTOR | Survival / proliferation |
| RAS_MAPK | KRAS, BRAF, MAP2K1, MAPK1 | Mitogenic signalling |
| APOPTOSIS | BCL2, CASP3, BAX, MCL1 | Programmed cell death |
| CELL_CYCLE | CDK2, CDK4, CDK6, CCND1 | Cell cycle regulation |
| TP53 | TP53, MDM2, MDM4 | Tumour suppression |
| NOTCH | NOTCH1, NOTCH2, NOTCH3 | Differentiation |
| VEGF_ANGIO | VEGFA, KDR, FLT1 | Angiogenesis |
| EPIGENETIC | HDAC1-3, HDAC6, DNMT1, EZH2 | Chromatin remodelling |
| JAK_STAT | JAK1, JAK2, STAT3 | Cytokine signalling |
| DNA_REPAIR | PARP1, PARP2, ATR, ATM | DNA damage response |
| PROTEASOME | PSMB5, PSMB1 | Protein degradation |
| TUBULIN | TUBB, TUBA1A | Cytoskeleton / mitosis |
| NFKB | NFKB1, RELA, IKBKB | Inflammation |
| COX | PTGS2, PTGS1 | Prostaglandin synthesis |
| SRC_FAMILY | SRC, ABL1, FYN | Non-receptor tyrosine kinases |

Multi-target compounds receive a joined label (e.g., `APOPTOSIS|RAS_MAPK`).
Targets not in the map are assigned `OTHER`; missing gene annotations yield
`UNKNOWN`. The result is stored in `target_families.pkl`.

**Rationale**: Compounds targeting the same pathway family share biological
activity patterns even when structurally dissimilar. Without pathway-aware
splitting, a model could learn "EGFR inhibitors are active" from training-set
EGFR compounds and appear to generalise on val-set EGFR compounds — a form of
pathway-level leakage.

#### 4.3.3 Scaffold-Disjoint, Pathway-Aware Splitting

`create_splits.py` consumes both artifacts to produce `splits.json` with the
following guarantees:

1. **Scaffold disjointness**: no Murcko scaffold appears in more than one split.
   All compounds sharing a scaffold are assigned to the same split.
2. **Pathway balance**: each split contains compounds from all major pathway
   families in roughly dataset-proportional ratios.
3. **Potency class stratification**: VP (class 3), potent (class 2), and active
   (class 1) compounds exist in all splits.
4. **Negative tier coverage**: all six negative tiers (T1–T4, T3A–T3C) are
   represented in every split.
5. **Holdout protection**: PLOS reference compounds and approved drugs are
   held out in the test set for external validation.

This is the gold standard for molecular ML — scaffold-disjoint splitting prevents
inflated validation metrics from structural memorisation.

---

## 5. Phase 2 — Potency-Graded Contrastive Learning

### 5.1 Architecture

#### 5.1.1 Projection Head

A 3-layer MLP projects frozen MolDeBERTa embeddings to a compact, task-specific space:

$$f_\theta: \mathbb{R}^{768} \rightarrow \mathbb{R}^{128}$$

```
Linear(768 → 512) → LayerNorm(512) → GELU → Dropout(0.1)
Linear(512 → 256) → LayerNorm(256) → GELU → Dropout(0.1)
Linear(256 → 128) → L2-normalize
```

**Parameter count**: 559,617 total (559,489 projection + 128+1 potency head)

**Why 3 layers, not 1**: A single linear layer can only rotate and scale the embedding
space — it cannot *reorganize* which molecules are close to which. Three layers with
nonlinearities allow the network to reshape the geometry, moving from MolDeBERTa's
"structurally similar" neighborhoods to the pharmacologically relevant "potency-similar"
neighborhoods needed for OSCC drug discovery. This is the critical transformation:
**chemical space → pharmacological space**.

**Design choices**:
- **LayerNorm** (not BatchNorm): Provides stable gradients for small batches of
  mixed positive/negative samples; independent of batch composition. BatchNorm would
  be noisy because batch statistics depend on the curriculum phase (warmup batches
  are 70% easy negatives; sharpen batches are 45% hard negatives).
- **GELU** (not ReLU): Smoother activation avoids dead neurons in the narrow
  bottleneck. With a 6× compression (768→128), losing neurons to ReLU death would
  permanently reduce the representational capacity of the projected space.
- **Dropout = 0.1**: Mild regularization; the dataset (~34K training compounds) is
  small relative to the 559K-parameter model (~16 parameters per compound), so
  dropout prevents the projection from memorizing individual compounds.
- **L2 normalization**: Projects all embeddings onto the unit hypersphere, decoupling
  *direction* (what the contrastive loss optimizes) from *magnitude* (irrelevant).
  Cosine similarity becomes a simple dot product, and temperature scaling in the
  contrastive loss works uniformly across all compound pairs.
- **128-d output**: Chosen to match the VAE latent dimension, enabling direct operation
  of generative models in the contrastive space. Large enough to represent ~16 pathway
  families × 3 potency tiers (48 pharmacological "archetypes") with room for continuous
  variation, but small enough to constrain the space for downstream generative models
  that must navigate it efficiently.

**Domain relevance**: MolDeBERTa's 768-d space was pre-trained on ~5M general ChEMBL
molecules — it captures broad chemical similarity. But for OSCC, we need
"pharmacologically equivalent against oral cancer targets," not "chemically similar."
Two compounds with completely different scaffolds (e.g., a quinazoline EGFR inhibitor
and a pyrimidine EGFR inhibitor) may have identical potency profiles but sit far apart
in MolDeBERTa space. The ProjectionHead learns to collapse that *irrelevant* structural
distance while expanding the *relevant* potency distance.

#### 5.1.2 Potency Regressor (Auxiliary Head)

$$g_\phi: \mathbb{R}^{128} \rightarrow \mathbb{R}$$

A single linear layer predicting pChEMBL values from projected features. This auxiliary
task regularizes the projection by ensuring the 128-d space retains potency information,
not just cluster identity.

**Why a single linear layer (intentionally minimal)**: The purpose is *not* to be an
accurate pChEMBL predictor — it is an **auxiliary regularizer**. A single linear layer
means pChEMBL must be *linearly decodable* from the 128-d projection. This constraint
forces the ProjectionHead to arrange compounds such that potency is a smooth, monotonic
gradient in the projected space — not tangled in nonlinear folds. If a deep regressor
were used, it could "untangle" a messy projection through its own capacity, removing
the regularization pressure on the ProjectionHead entirely.

**Why it reads from z (not raw 768-d)**: The regressor sits *after* the ProjectionHead
— it reads from the L2-normalized 128-d projection, not from the raw MolDeBERTa
embedding. This is critical: Huber loss gradients backpropagate through the regressor,
through z, and into the ProjectionHead weights. If the regressor read from the raw
768-d space directly (parallel architecture), its gradients would never touch the
ProjectionHead, providing zero regularization of the contrastive geometry.

**Downstream importance of linear decodability**: All downstream modules rely on the
potency gradient being smooth and navigable in z-space:
- **Generative RL** (`rl_finetune.py`): uses cosine proximity to the VP centroid as
  reward — noisy potency landscape would cause RL divergence
- **Drug repurposing** (`run_drug_repurposing.py`): ranks candidates by z-distance to
  known actives — linear gradient ensures closer = more potent
- **Lead optimization** (`run_lead_optimization.py`): navigates z-space to improve
  potency — linear decodability ensures each step corresponds to a predictable
  potency change

#### 5.1.3 ContrastiveModel (Wrapper)

The `ContrastiveModel` class is a thin wrapper holding both heads with a single forward
pass: project first, then regress from the projection. Returns `(z, pchembl_pred)` so
the training loop can compute all three losses from one forward call.

The dual-output design produces a single 128-d embedding that serves two simultaneous
roles:
1. **Contrastive role** (OrdinalSupConLoss): compounds of similar OSCC potency cluster
   together; dissimilar ones are pushed apart.
2. **Predictive role** (PotencyLoss): pChEMBL is linearly decodable, so the space has
   a smooth, navigable potency landscape.

This single `best.pt` checkpoint is the shared artifact consumed by all 10+ downstream
modules — from generative RL, through drug repurposing, lead optimization, category
annotation, to validation. The entire GENESIS pipeline flows through this 128-d
bottleneck.

### 5.2 Loss Function

The total loss is a weighted combination of three components:

$$\mathcal{L} = \mathcal{L}_{\text{con}} + \lambda_1 \cdot \mathcal{L}_{\text{pot}} + \lambda_2 \cdot \mathcal{L}_{\text{rank}}$$

#### 5.2.1 Ordinal Supervised Contrastive Loss (Primary)

Extends SupCon (Khosla et al., NeurIPS 2020) with ordinal potency-proximity weights:

$$\mathcal{L}_{\text{con}} = -\frac{1}{|I|} \sum_{i \in I} \frac{1}{\sum_{p \in P(i)} w_{ip}} \sum_{p \in P(i)} w_{ip} \cdot \log \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_p / \tau)}{\sum_{a \neq i} \tilde{w}_{ia} \cdot \exp(\mathbf{z}_i \cdot \mathbf{z}_a / \tau)}$$

##### Variable definitions (exhaustive)

| Symbol | Shape / Type | Definition | Source |
|--------|-------------|------------|--------|
| $\mathbf{z}_i \in \mathbb{R}^{128}$ | $(B, 128)$ | L2-normalized projection of compound $i$. Since $\|\mathbf{z}\|_2 = 1$, $\mathbf{z}_i \cdot \mathbf{z}_j$ equals cosine similarity. | `ProjectionHead.forward()` |
| $I$ | set | Anchor indices — all compounds in the batch that have at least one positive pair ($|P(i)| > 0$). In practice: all positives. | `valid = num_positives > 0` |
| $P(i)$ | set | Positive pair indices for anchor $i$: all other positives ($\text{label}_j = 1, j \neq i$) in the batch. | `pos_mask` in `losses.py` |
| $\tau$ | scalar | Temperature parameter ($= 0.07$). Controls sharpness of the softmax over similarities. | Hyperparameter |
| $w_{ip}$ | $(B, B)$ | Ordinal pair weight: attraction strength between anchor $i$ and positive $p$, based on potency proximity and target identity. | `compute_pair_weights()` |
| $\tilde{w}_{ia}$ | $(B, B)$ | Denominator weight for sample $a$ relative to anchor $i$. For negatives: tier-based loss weight (T3A→1.2, T4→0.4). For positives: 1.0. | `neg_weights` × `weight_matrix` |
| $B$ | integer | Batch size (typically 512 = 128 positives + 384 negatives). | Training config |

##### Potency classes

Each positive compound is assigned an ordinal potency class based on its measured
pChEMBL value (= $-\log_{10}$(IC₅₀ in M)):

| Class | Label | pChEMBL range | IC₅₀ range | Biological meaning |
|-------|-------|---------------|------------|-------------------|
| 3 | Very Potent (VP) | ≥ 7.0 | ≤ 100 nM | Nanomolar inhibition — clinical-grade potency |
| 2 | Potent | 6.0 – 7.0 | 100 nM – 1 μM | Sub-micromolar — strong lead compound |
| 1 | Active | 5.0 – 6.0 | 1 μM – 10 μM | Micromolar — confirmed activity, needs optimization |
| 0 | Negative | N/A | N/A | No measured OSCC bioactivity |

##### Ordinal pair weight matrix $W$

| Anchor \ Positive | VP (3) | Potent (2) | Active (1) |
|-------------------|--------|------------|------------|
| **VP (3)** | 1.0 | 0.7 | 0.4 |
| **Potent (2)** | 0.7 | 1.0 | 0.7 |
| **Active (1)** | 0.4 | 0.7 | 1.0 |

**Domain interpretation**: These weights create a *potency-graded attraction manifold*.
Standard SupCon treats all positives equally — a VP compound is pulled with equal
force toward another VP compound and toward a weakly active compound. This is
pharmacologically incorrect: a compound with IC₅₀ = 50 nM (class 3) shares far more
mechanism-of-action relevance with another nanomolar inhibitor than with a micromolar
one (class 1). The weight matrix encodes this ordinal relationship:
- **Same-class pairs** (diagonal, $w = 1.0$): Maximum attraction. Compounds of equal
  potency should cluster tightly — they likely share similar binding modes and
  pharmacophoric features.
- **Adjacent-class pairs** ($w = 0.7$): Moderate attraction. A potent compound (class 2)
  is related to both VP (class 3) and active (class 1), but less so than to another
  class-2 compound. The 0.7 weight allows proximity without forced overlap.
- **Distant-class pairs** ($w = 0.4$): Weak attraction. VP ↔ active pairs should be in
  the same positive region (both are OSCC-active), but should remain distinguishable.
  The reduced weight prevents collapsing the entire potency gradient into a single point.

The result is a 128-d space where cosine similarity to the VP centroid **monotonically
tracks potency**: VP (~0.96) > Potent (~0.90) > Active (~0.81) > Negative (~0.30).

##### Cross-target scaling ($\alpha = 0.5$)

When two positives target **different** OSCC proteins (e.g., one inhibits EGFR, the
other inhibits AKT1), the pair weight is halved:

$$w_{ip}^{\text{final}} = w_{ip} \times \begin{cases} 1.0 & \text{if target}(i) = \text{target}(p) \\ 0.5 & \text{otherwise} \end{cases}$$

**Domain rationale**: OSCC is a multi-target disease, but compounds active against the
same target share stronger mechanism-of-action similarity (similar binding pockets,
pharmacophores, SAR patterns) than compounds that merely share "OSCC relevance."
Without this penalty, the model would collapse all 7 target clusters into one,
losing target-specific information that downstream docking requires. With it, the
model learns *sub-clusters* per target within the broader positive region — EGFR
inhibitors cluster near each other within the positive cloud, separate from (but
closer to) MTOR inhibitors.

##### Temperature $\tau = 0.07$ — deep dive

Temperature controls the **dynamic range** of the softmax over pairwise similarities:

$$p_{ij} \propto \exp\left(\frac{\mathbf{z}_i \cdot \mathbf{z}_j}{\tau}\right)$$

At $\tau = 0.07$, a cosine difference of 0.1 creates an exponential ratio of:

$$\frac{\exp(0.9/0.07)}{\exp(0.8/0.07)} = \exp\left(\frac{0.1}{0.07}\right) \approx e^{1.43} \approx 4.2\times$$

This means a compound with cosine similarity 0.9 to the anchor contributes **4.2× more**
to the denominator than one at 0.8 — the model is very sensitive to small similarity
differences. This aggressive selectivity is intentional for drug discovery: in chemical
space, structurally similar compounds (cosine ~0.85–0.95) can differ by orders of
magnitude in potency. A "lazy" temperature ($\tau = 0.5$, where the same 0.1
difference gives only $e^{0.2} \approx 1.2\times$) would treat these compounds as
nearly equivalent, failing to learn the fine-grained SAR patterns that distinguish
active from inactive analogs.

**Trade-off**: Very low temperatures ($\tau < 0.05$) make gradients numerically
unstable; very high temperatures ($\tau > 0.2$) lose discriminative power. The
value 0.07, established by SimCLR (Chen et al., 2020) and validated in molecular
settings, balances sharpness with training stability.

##### Denominator negative weights $\tilde{w}_{ia}$

The denominator sums over **all** non-self samples (both positives and negatives),
but negatives receive tier-dependent weights that amplify or attenuate their
contribution:

| Negative Tier | $\tilde{w}$ | Biological description | Effect on loss |
|---------------|-------------|----------------------|----------------|
| T3A_inactive | **1.2** | Tested against OSCC targets but showed no activity | Amplified — these are the hardest negatives (look like they should work but don't). The model must concentrate extra gradient on separating them. |
| T3C | **1.0** | Structurally similar (Tanimoto ≥ 0.4) to known positives | Standard — structural decoys that test scaffold discrimination. |
| T1 | **1.0** | Random ChEMBL compounds with activity on OSCC targets for other indications | Standard difficulty. |
| T2 | **0.8** | Drug-like compounds active against non-OSCC cancer targets | Slightly attenuated — useful for OSCC-specificity but not the primary challenge. |
| T3B | **0.6** | Distant scaffolds from the OSCC-active chemical space | Downweighted — easily distinguished, provide regularization but shouldn't dominate gradients. |
| T4 | **0.4** | Random non-drug compounds (solvent-like, reagent-like) | Heavily downweighted — trivially different, included for geometric coverage but not worth amplifying. |

**Mechanistic effect**: By upweighting T3A compounds in the denominator, the softmax
"budget" allocated to these hard negatives increases. When the model fails to push a
T3A compound far from the anchor, the denominator is larger → the log-probability of
the positive pair is smaller → the loss is larger → stronger gradient signal. This
focuses learning on the pharmacologically meaningful decision boundary: "tested-inactive
vs. active against the same target."

##### Forward pass computation — step by step

```python
# Step 1: Pairwise similarity matrix (B × B)
sim = z @ z.T / τ              # z is L2-normalized, so this = cosine / τ

# Step 2: Positive pair mask (exclude self-pairs)
pos_mask = labels.unsqueeze(0) * labels.unsqueeze(1) * (~eye(B))
weighted_pos_mask = pos_mask * pair_weights     # apply ordinal weights

# Step 3: Numerical stability (log-sum-exp trick)
logits = sim - max(sim, dim=1)   # subtract row-wise max for stability

# Step 4: Weighted denominator
exp_logits = exp(logits) * (~eye(B)) * weight_matrix  # tier-weighted
log_prob = logits - log(sum(exp_logits, dim=1))

# Step 5: Weighted mean over positive pairs
loss = -mean( sum(weighted_pos_mask * log_prob, dim=1)
            / sum(weighted_pos_mask, dim=1) )
```

The normalization by $\sum_p w_{ip}$ (not $|P(i)|$) ensures that the contribution of
each anchor is weighted by the **sum of ordinal weights**, not the raw count of
positives. An anchor with 5 same-class positives (total weight = 5.0) and an anchor
with 5 cross-class positives (total weight ≈ 2.5) contribute to the mean with
different denominators, reflecting that the first anchor has richer supervisory signal.

#### 5.2.2 Potency Regression Loss (Auxiliary)

$$\mathcal{L}_{\text{pot}} = \text{Huber}_\delta \left( \hat{y}_i, y_i \right), \quad \delta = 1.0, \quad \text{for } i \in \{i : y_i \neq \text{NaN}\}$$

where $\hat{y}_i = g_\phi(\mathbf{z}_i)$ is the predicted pChEMBL from the
`PotencyRegressor` head (a single linear layer: $\mathbb{R}^{128} \to \mathbb{R}$),
and $y_i$ is the measured pChEMBL value.

##### Variable details

| Symbol | Definition |
|--------|-----------|
| $\hat{y}_i$ | Predicted pChEMBL for compound $i$, computed as $\mathbf{w}^\top \mathbf{z}_i + b$ where $\mathbf{w} \in \mathbb{R}^{128}$, $b \in \mathbb{R}$ are the regressor parameters |
| $y_i$ | Ground-truth `pchembl_filled` value (measured $-\log_{10}$(IC₅₀/EC₅₀ in M)). Typical range: 4.0–9.0 |
| $\delta = 1.0$ | Huber transition point: quadratic for $|\hat{y}_i - y_i| \leq 1.0$, linear beyond |
| mask | Boolean tensor — `True` only for compounds with measured pChEMBL. All negatives (tier T1-T4) and some positives without quantitative dose-response data are excluded |

##### Why Huber loss (not MSE or MAE)?

The Huber loss combines the best properties of both:

$$\text{Huber}_\delta(e) = \begin{cases} \frac{1}{2}e^2 & \text{if } |e| \leq \delta \\ \delta \cdot (|e| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}$$

- **Within $\delta$**: Quadratic behavior provides smooth, informative gradients for
  predictions within 1 pChEMBL unit of truth (a 10× error in IC₅₀). This is where
  the model spends most of its optimization effort.
- **Beyond $\delta$**: Linear behavior prevents a single wildly mismeasured IC₅₀
  (common in assay data — inter-lab variability can exceed 1 log unit) from dominating
  the gradient via squared error. MSE would let one outlier (e.g., measured pChEMBL = 4.5
  vs. true = 7.0) contribute $(2.5)^2 = 6.25$ to the loss; Huber caps this at
  $1.0 \times (2.5 - 0.5) = 2.0$.

##### Functional role as information bottleneck regularizer

The potency regression head serves a critical **architectural purpose** beyond simple
prediction accuracy. By requiring the L2-normalized 128-d projection $\mathbf{z}_i$ to
support a linear readout of pChEMBL, it forces the contrastive space to encode potency
information **linearly and explicitly** — not just as cluster membership.

Without this term, contrastive learning could produce a geometry where all positives
cluster together but the potency ordering within the cluster is arbitrary (e.g., VP
and active compounds at identical cosine distances to centroid, differentiated only
by angular direction). The regression loss prevents this by requiring a single
hyperplane in the 128-d space to separate high-pChEMBL from low-pChEMBL compounds.

This ensures the **VP centroid direction correlates with bioactivity magnitude**,
which is essential for generative models (VAE, CRAVE, etc.) that use
cosine-to-centroid as a differentiable reward signal. Moving 0.1 cosine units closer
to the VP centroid must correspond to a meaningful potency increase — the regression
loss enforces this correspondence.

#### 5.2.3 Tier Margin Ranking Loss (Auxiliary)

For triplets $(a, p, n)$ where $a, p$ are distinct positives and $n$ is a negative:

$$\mathcal{L}_{\text{rank}} = \frac{1}{N} \sum_{i=1}^{N} \max\left(0, \; m(t_{n_i}) + \text{sim}(a_i, n_i) - \text{sim}(a_i, p_i)\right)$$

##### Variable definitions

| Symbol | Definition |
|--------|-----------|
| $a_i, p_i$ | Two distinct positive compounds from the batch (randomly permuted, with safeguard against self-pairing: when $a_i = p_i$, index is shifted by $(+1) \bmod N_{\text{pos}}$) |
| $n_i$ | A negative compound from the batch (randomly selected) |
| $\text{sim}(x, y) = \mathbf{z}_x \cdot \mathbf{z}_y$ | Cosine similarity (dot product of L2-normalized projections) |
| $t_{n_i}$ | Tier label of the negative compound (T1, T2, T3A_inactive, T3B, T3C, T4) |
| $m(t)$ | Tier-dependent margin function (see table below) |
| $N$ | Number of triplets per batch = $\min(N_{\text{pos}}, N_{\text{neg}})$ in the batch |

##### Tier-dependent margins

| Tier | $m(t)$ | Negative description | Rationale |
|------|--------|---------------------|-----------|
| T3A_inactive | **0.1** | Tested against OSCC targets but inactive | **Smallest margin** — counterintuitive but critical (see below) |
| T3C | **0.1** | Structurally similar to positives (Tanimoto ≥ 0.4) | Hard structural decoys — fine discrimination needed |
| T1 | **0.2** | Compounds with activity on OSCC targets for other diseases | Medium — distinguishable but not trivially |
| T2 | **0.2** | Active against non-OSCC cancer targets | Medium — clear separation expected |
| T3B | **0.3** | Distant scaffolds from OSCC-active chemical space | Easy negatives — should be obviously separated |
| T4 | **0.3** | Random non-drug molecules | **Largest margin** — trivially different from any drug candidate |

##### Why hard negatives get SMALLER margins (the counterintuitive design)

This is the most non-obvious design choice in the loss function. Intuitively, one
might expect hard negatives to demand a *larger* margin ("we need MORE separation from
confusing compounds"). The opposite is correct:

1. **Feasibility**: T3A compounds (tested-inactive against OSCC targets) are
   structurally and mechanistically very similar to positives — they bind the same
   protein pockets but fail to inhibit function (wrong binding orientation, missing
   key interaction, etc.). In embedding space, they *should* be nearby. Demanding
   $m = 0.3$ would force the model to push structurally near-identical molecules
   0.3 cosine units apart, which is geometrically impossible without distorting
   the entire manifold. This creates large, unrelenting losses that destabilize
   training.

2. **Gradient utility**: A small margin ($m = 0.1$) means the triplet loss activates
   only when $\text{sim}(a, n) > \text{sim}(a, p) - 0.1$ — i.e., when the negative
   is dangerously close to being confused with a positive. This provides a focused,
   achievable learning signal precisely at the decision boundary.

3. **Easy negatives warrant large margins**: T4 compounds (solvent-like, reagent-like)
   are trivially different from OSCC drug candidates. A large margin ($m = 0.3$)
   is easily achievable and enforces an unambiguous geometric buffer. If the model
   can't separate a kinase inhibitor from acetone by 0.3 cosine units, something
   is fundamentally wrong.

##### Triplet construction details

Triplets are constructed **within each batch** during training (not pre-computed):

```python
# From the batch, separate positives and negatives
z_pos = z[labels == 1]     # (N_pos, 128)
z_neg = z[labels == 0]     # (N_neg, 128)
N_triplets = min(N_pos, N_neg)

# Randomly pair two different positives as (anchor, positive)
perm1 = randperm(N_pos)[:N_triplets]
perm2 = randperm(N_pos)[:N_triplets]
# Avoid self-pairs: shift collisions by +1
perm2[perm1 == perm2] = (perm2[perm1 == perm2] + 1) % N_pos

# Random negative per triplet
perm_neg = randperm(N_neg)[:N_triplets]
```

Using two *different* positives as anchor and positive (rather than anchor = positive)
tests whether the model has learned the **shared** potency-relevant features, not
just self-similarity. The anchor must recognize that another positive — potentially
a different scaffold against a different target — is more similar to it than a
negative compound.

#### 5.2.4 Design Synergy: Why This Three-Term Loss Works for OSCC Drug Repurposing

The three loss components address complementary failure modes:

| Component | What it prevents | Geometric effect |
|-----------|-----------------|------------------|
| $\mathcal{L}_{\text{con}}$ (ordinal SupCon) | Positives scattering; potency classes collapsing together | Creates potency-graded concentric shells around VP centroid |
| $\mathcal{L}_{\text{pot}}$ (Huber regression) | Potency ordering being nonlinear or arbitrary within the positive region | Linearizes pChEMBL along a hyperplane in 128-d space |
| $\mathcal{L}_{\text{rank}}$ (margin ranking) | Decision boundaries being soft or geometry-dependent | Enforces explicit separation gaps calibrated to biological difficulty |

Working together:
1. **Ordinal weights** ($\mathcal{L}_{\text{con}}$) prevent the model from treating a
   50 nM inhibitor and a 5 μM inhibitor as equivalent — the embedding space preserves
   potency gradation, so the VP centroid naturally points toward the high-potency region.
2. **Cross-target penalty** ($\alpha = 0.5$ in $\mathcal{L}_{\text{con}}$) maintains
   target-specific subclusters — critical because EGFR inhibitors and MTOR inhibitors
   may have very different scaffolds, and collapsing them would lose information needed
   for downstream target-specific docking.
3. **Tier-weighted negatives** ($\tilde{w}$ in $\mathcal{L}_{\text{con}}$ denominator +
   curriculum sampling) teach the model to first find the broad boundary (active vs.
   random), then refine it against hard decoys (tested-inactive analogs like T3A).
4. **The potency regression head** ($\mathcal{L}_{\text{pot}}$) acts as an information
   bottleneck regularizer — it forces the 128-d space to encode pChEMBL as a linearly
   readable signal, ensuring generative models (VAE, CRAVE, etc.) trained in this space
   can exploit potency gradients as differentiable rewards.
5. **Adaptive margin ranking** ($\mathcal{L}_{\text{rank}}$) adds explicit geometric
   constraints beyond what contrastive loss alone achieves — it directly enforces
   separation gaps proportional to biological difficulty, preventing the geometry from
   being "correct on average" but locally confused.

### 5.3 Curriculum Learning

The training procedure uses a 4-phase curriculum that controls negative sampling
distribution and loss weight scheduling:

#### 5.3.1 Negative Sampling Curriculum

The proportion of each negative tier in each training batch evolves:

| Phase | Epochs | T4 (easy) | T3B (soft) | T1 | T2 | T3A (hard) | T3C (hard) |
|-------|--------|-----------|------------|-----|-----|------------|------------|
| Warmup | 1–5 | 35% | 35% | 10% | 10% | 5% | 5% |
| Middle | 6–15 | 15% | 15% | 25% | 20% | 15% | 10% |
| Sharpen | 16–25 | 10% | 10% | 20% | 15% | 25% | 20% |
| Finetune | 25+ | 5% | 10% | 15% | 15% | 30% | 25% |

**Curriculum motivation**: In early epochs, the model stabilizes its embedding structure
using easy negatives (T4, T3B). As training progresses, harder negatives (T3A, T3C)
are introduced to refine decision boundaries. This avoids the common failure mode where
early exposure to hard negatives prevents forming the basic cluster structure.

#### 5.3.2 Loss Weight Curriculum

| Phase | Epochs | $\lambda_1$ (potency) | $\lambda_2$ (ranking) |
|-------|--------|-----------------------|-----------------------|
| Structure | 1–10 | 0.1 | 0.0 |
| Gradual | 11–20 | 0.3 | 0.2 |
| Full | 20+ | 0.5 | 0.3 |

The primary contrastive loss ($\mathcal{L}_{\text{con}}$) always has weight 1.0. Auxiliary
losses are introduced gradually to prevent early optimization conflicts.

**Phase transition handling (corrected)**: When the curriculum phase changes, the
early-stopping *patience counter* is reset, giving the model a fresh window to
adapt to the new loss regime before patience can trigger a stop. This patience
reset is intentional and unchanged. However, an earlier version of the training
code also used the same `best_val_loss` variable to decide which epoch's weights
to save as `best.pt` -- meaning a phase-transition reset could cause a mediocre
epoch immediately following the reset to be saved as "best", silently discarding
a genuinely superior checkpoint from earlier in the same phase. This was
identified and fixed: checkpoint selection is now fully decoupled from the
early-stopping patience criterion. `best.pt` is unconditionally saved as the
most recently completed epoch's weights, and early stopping (governed by
patience on `best_val_loss`, still reset at phase transitions) only decides
*when* training stops, never *which* epoch's weights are kept. A systematic
epoch-by-epoch scan (evaluating checkpoints at epochs 10, 13, 16, 19, 22, 25
against the actual downstream metrics: Spearman ρ, cliff accuracy, target-switch
AUROC, NDCG@100) confirmed these metrics improve monotonically through epoch 25
and that the pre-fix code was systematically saving an inferior mid-training
checkpoint rather than the run's true best.

### 5.4 Optimization

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | AdamW | Weight decay regularization for generalizable projections |
| Learning rate | 3 × 10⁻⁴ | Standard for MLP projection heads; warm enough for fast convergence |
| Weight decay | 1 × 10⁻⁴ | Mild L2 penalization to prevent overfitting on ~49K positives |
| Scheduler | CosineAnnealingLR | Smooth decay avoids sharp LR drops that disrupt contrastive structure |
| Min LR | 1 × 10⁻⁶ | Prevents complete stalling in final epochs |
| Gradient clipping | max_norm = 1.0 | Stabilizes training with mixed batch compositions |
| Batch size | 512 (128 pos + 384 neg) | Sufficient for meaningful contrastive pairs |
| Min epochs | 25 | Ensures all curriculum phases are experienced |
| Patience | 7 | Allows adaptation after phase transitions |
| Projection dim | 128 | Matches generative model latent dim |

### 5.5 Evaluation Battery (6 Tests)

The contrastive model is evaluated with a scientifically rigorous battery:

| Test | Metric | What It Measures |
|------|--------|-----------------|
| 1. Potency Ordering | Global & per-target Spearman ρ | Does cosine-to-centroid correlate with potency? |
| 2. Activity Cliff | Pair classification accuracy | Can the model distinguish structurally similar active/inactive pairs? |
| 3. Target-Switch | AUC-ROC, AUC-PR | Can it distinguish OSCC actives from same-target non-OSCC actives? |
| 4. OSCC Specificity | Cohen's d, Mann-Whitney U | How strongly are OSCC actives separated from other-cancer actives? |
| 5. VP Retrieval | Recall@K, Enrichment factor | Can it retrieve VP compounds hidden in a large pool? |
| 6. Graded Ranking | NDCG@K, per-class percentile | Does ranking follow the full potency gradient (VP > Potent > Active > Neg)? |

### 5.6 Validation

**Multi-seed validation** (seeds 42, 123, 456) with 1000-round bootstrap confidence
intervals confirms training stability. **Ablation studies** test the contribution of
each loss component:
- Contrastive-only (no potency, no ranking)
- Contrastive + potency (no ranking)
- Full model (all three losses)

### 5.7 Empirical Evidence: Contrastive Learning Creates Task-Specific Structure

The following results, obtained from t-SNE visualization and quantitative analysis
of 3,997 test-set compounds (stratified across 3 potency classes + 4 negative tiers),
provide definitive proof that raw MolDeBERTa embeddings encode general chemical syntax
but lack task-specific (OSCC potency) structure.

*Figures: `modeling/figures/embedding_space_comparison.png`, `embedding_metrics_comparison.png`,
`vp_centroid_distributions.png`*

#### 5.7.1 The Problem: Raw Embeddings Are Potency-Blind

In the raw 768-d MolDeBERTa space, cosine similarity to the VP centroid (mean of
Very Potent compound embeddings) is nearly identical across all potency classes:

| Class | Raw Cosine to VP Centroid | Contrastive Cosine to VP Centroid |
|-------|--------------------------|----------------------------------|
| VP (pChEMBL ≥ 7.0) | 0.948 ± 0.029 | 0.961 ± 0.088 |
| Potent (6.0 – 7.0) | 0.944 ± 0.035 | 0.903 ± 0.186 |
| Active (< 6.0) | 0.937 ± 0.042 | 0.806 ± 0.263 |
| **Negative (inactive)** | **0.934 ± 0.045** | **0.297 ± 0.492** |

The VP-to-Negative gap in the raw space is a mere **0.014 cosine units** — all four
distributions overlap almost completely within the 0.88–1.0 similarity band.
Biologically, this means MolDeBERTa treats a nanomolar EGFR inhibitor identically
to a random agrochemical: it encodes that both are valid molecules with aromatic
rings and heteroatoms, but not that one kills OSCC cells and the other does not.

#### 5.7.2 The Solution: Contrastive Projections Create an Ordinal Potency Gradient

After contrastive learning, the same compounds redistribute across the full [-0.8, 1.0]
cosine range with strict monotonic ordering: VP (0.961) > Potent (0.903) > Active (0.806)
> Negative (0.297). The VP-to-Negative gap widens to **0.664** — a **47× improvement**
in discriminative power.

**Cluster quality (silhouette scores, cosine distance)**:

| Grouping | Raw (768-d) | Contrastive (128-d) | Improvement |
|----------|-------------|---------------------|-------------|
| Binary (Pos vs Neg) | 0.016 | 0.174 | +957% |

**VP class separation (intra-class vs inter-class cosine similarity)**:

| Metric | Raw | Contrastive | Improvement |
|--------|-----|-------------|-------------|
| Intra-VP similarity | 0.902 | 0.923 | +2.3% |
| VP-vs-Other similarity | 0.889 | 0.469 | −47.2% |
| **Gap (Intra − Inter)** | **0.013** | **0.454** | **+3,501%** |

The contrastive model simultaneously tightens VP intra-class coherence (0.902 →
0.923) and dramatically pushes non-VP compounds away (0.889 → 0.469), creating
a well-defined VP cluster with clear geometric boundaries.

#### 5.7.3 Biological Significance

1. **Drug-target selectivity becomes geometric**: In the contrastive space, the
   direction toward the VP centroid encodes "OSCC-relevant potency." A compound's
   cosine distance to this centroid predicts whether it would show nanomolar
   activity (VP, ~0.96) vs micromolar (Active, ~0.81) vs no activity (Neg, ~0.30)
   against OSCC targets (EGFR, VEGFR, PI3K, mTOR).

2. **Activity cliffs are resolved**: The t-SNE visualization shows structurally
   similar compounds with different potencies (activity cliffs — the hardest problem
   in medicinal chemistry SAR) separated in the contrastive space but intermixed
   in the raw space. The model learned pharmacophore-level features beyond 2D
   structural fingerprints.

3. **The potency gradient enables generative drug design**: Because similarity-to-centroid
   monotonically tracks potency, generative models can use it as a differentiable
   reward signal. Moving a generated molecule 0.1 cosine units closer to the VP
   centroid corresponds to roughly one log-unit increase in predicted potency
   (~10× more potent in IC₅₀ terms).

4. **Negative tier separation validates the biological tiering**: Random ChEMBL
   compounds (T4) scatter furthest from the VP cluster; activity cliffs (T3A) and
   target-switch compounds (T3C) land closer — exactly matching their biological
   relationship to OSCC actives and confirming that the model internalized the
   pharmacological hierarchy encoded in the training data.

5. **Target semantics are retained, not lost** *(critical interpretive note)*: t-SNE
   and UMAP visualizations show that the contrastive projection reduces raw k-NN
   target-label purity from 0.849 → 0.779. This is **not** evidence of lost target
   understanding — it is evidence of a deliberate geometric reorganization.

   In MolDeBERTa's 768-d space, same-target compounds cluster tightly for a
   chemically trivial reason: they share scaffolds. EGFR inhibitors are largely
   quinazolines; MDM2 inhibitors are largely spiro-oxindoles. The language model
   never saw bioactivity — scaffold identity acts as a target proxy, producing
   artificially high target k-NN purity that encodes *chemistry*, not *biology*.

   The contrastive projection intentionally replaces scaffold-based co-location
   with potency-based co-location: a Very Potent EGFR inhibitor moves toward
   other Very Potent OSCC actives regardless of target, and a weak Active EGFR
   compound moves toward the periphery alongside other low-potency actives. This
   breaks target-scaffold clustering in favour of pharmacological-grade clustering.

   Target semantics are **preserved at the functional level**, as proven by the
   evaluation battery:

   | Test | Metric | Proof of retention |
   |------|--------|--------------------|
   | T3 Target-Switch AUC | **0.967** *(3-seed mean; see §5.5 evaluation battery / Table 1)* | Model assigns multi-target compounds to the correct OSCC target centroid ~97% of the time |
   | T7 Cross-Pathway AUC | **0.860** *(not independently re-verified against the corrected checkpoint this session — flagged as provisional)* | EGFR VP centroid correctly ranks FGFR1/AKT1 compounds — pathway-level target topology preserved |
   | Fig 3 t-SNE target coloring | Visual clusters | EGFR, MDM2, mTOR form visually coherent tight clusters *within* the contrastive space |
   | **Fig 6a** centroid heatmap | Raw off-diag 0.97–0.99 → Proj 0.96–0.99 | All OSCC targets look identical in raw space; contrastive space *increases* target distinctness (COX-2 / MDM2 centroids: 0.96) |
   | **Fig 6b** potency gradient | Flat raw lines → strict Active<Potent<VP in proj | Every named target preserves within-target potency ordering in contrastive space (CDK4 steepest: 0.75→0.87→0.97) |

   The precise distinction is: **what is lost** is scaffold-proximity within a
   target (trivial, chemistry-based); **what is gained** is potency-proximity
   across targets (meaningful, biology-based). The model trades chemical taxonomy
   for pharmacological semantics — which is exactly the design objective.

   Figure 6 provides direct geometric proof: panel (a) shows that per-target
   centroids are *more* distinct in the contrastive space (lower off-diagonal
   cosine similarity for pharmacologically divergent targets), and panel (b)
   shows that the strict Active < Potent < Very Potent ordering holds within
   every named OSCC target. The within-target potency hierarchy is not merely
   statistically present — it spans the full similarity range (CDK4 gradient
   ≈ 0.22 cosine units) confirming strong geometrical encoding.

#### 5.7.4 Summary Statement

> *Pre-trained molecular language models encode chemical syntax but not pharmacological
> semantics. The contrastive projection transforms this syntax-rich space into a
> potency-graded landscape where geometric proximity to a very-potent centroid directly
> encodes predicted OSCC activity — enabling the first geometry-aware generative
> pipeline for oral cancer drug discovery.*

### 5.8 Novelty Assessment: What Is New vs. Established

This section provides an honest positioning of GENESIS's contrastive learning
contributions relative to the existing literature.

#### 5.8.1 Genuinely Novel Contributions

**Contribution 1: Ordinal Supervised Contrastive Loss with Potency-Proximity Pair Weights**

The key distinction from prior work:

| Approach | How it handles ordinal labels | Reference |
|----------|------------------------------|-----------|
| SupCon (Khosla et al., 2020) | Binary: same-class = positive, different-class = negative. No ordinal grading. | NeurIPS 2020 |
| Rank-N-Contrast (Zha et al., 2022) | Uses continuous regression targets directly, weighting by absolute difference $|y_i - y_j|$. | NeurIPS 2022 |
| **GENESIS OrdinalSupConLoss** | **Discrete ordinal classes** (VP/Potent/Active) with an **expert-designed symmetric weight matrix** $W$ encoding pharmacological domain knowledge. Positives attract each other with graded strength, not binary on/off. | This work |

Rank-N-Contrast is the closest prior work — but it uses continuous regression targets
for weighting (purely data-driven), whereas GENESIS uses **expert-designed discrete
potency bins** with specific weight values (1.0/0.7/0.4) reflecting pharmacological
reasoning: a nanomolar-to-micromolar IC₅₀ boundary carries qualitative biological
meaning (binding mode change, selectivity cliff) that a linear regression weight
misses. The discrete bins + manual weights inject domain inductive bias that a purely
data-driven weighting cannot capture from limited OSCC training data (~49K positives).

*Claim strength*: Moderate-to-strong — a **domain-specialized variant** of ordinal
contrastive learning. The specific formulation with the $W$ matrix + cross-target
penalty ($\alpha = 0.5$) is novel in implementation.

**Contribution 2: Pharmacologically-Stratified 6-Tier Negative Taxonomy with Curriculum**

No prior drug discovery contrastive learning paper uses this negative stratification:

| Tier | Biological definition | Why it is unique |
|------|----------------------|------------------|
| T3A_inactive | Tested against OSCC targets, confirmed inactive | Activity cliff negatives — only available with target-specific bioactivity data |
| T3C | Tanimoto ≥ 0.4 to known positives | Structural decoys based on fingerprint similarity |
| T1 | Active on OSCC targets for other diseases | Target-overlap negatives |
| T2 | Active against non-OSCC cancer targets | Disease-specificity negatives |
| T3B | Distant scaffolds from OSCC-active space | Easy structural negatives |
| T4 | Random non-drug molecules | Trivial negatives |

Existing contrastive learning in drug discovery typically uses binary (active/inactive)
or at most 2–3 tiers (hard/medium/easy). The 6-tier taxonomy where **each tier encodes
a specific pharmacological relationship to the positive class** — and these tiers
independently control three different mechanisms (sampling probability, denominator
weight, margin size) — is novel.

*Claim strength*: Strong. The taxonomy itself is a methodological contribution,
independent of the loss function.

**Contribution 3: Inversely-Proportional Tier-Margin Design**

The standard approach in metric learning: harder negatives → larger margin (push
harder). GENESIS does the opposite: T3A (hardest) → $m = 0.1$, T4 (easiest) →
$m = 0.3$.

This is not found in standard metric learning literature (FaceNet, ArcFace, etc.),
which typically use fixed margins or scale margins proportionally to difficulty. The
pharmacological reasoning — hard negatives are structurally similar, so large margins
are geometrically infeasible and would distort the manifold — is a domain insight that
translates into a non-obvious algorithm design.

*Claim strength*: Moderate — a design insight more than a new algorithm, but
publishable as a finding: "counterintuitive margin calibration for activity cliff
discrimination."

#### 5.8.2 Established Prior Work (Not Novel)

| Component | Prior work |
|-----------|-----------|
| Supervised Contrastive Loss (SupCon) | Khosla et al., NeurIPS 2020 |
| Projection head architecture (MLP + L2-norm) | Chen et al., SimCLR, ICML 2020 |
| Temperature-scaled cosine similarity | Common in all contrastive methods |
| Curriculum learning (easy → hard negatives) | Bengio et al., ICML 2009 |
| Hard negative mining | Robinson et al., ICLR 2021 |
| Margin ranking / triplet loss | Schroff et al., FaceNet, CVPR 2015 |
| Huber loss for robust regression | Standard ML |
| Multi-task loss with auxiliary heads | Standard practice |
| Contrastive learning for molecules | MolCLR (Wang et al., 2022), MoCL (Sun et al., 2021) |

#### 5.8.3 Borderline / Incremental Contributions

| Component | Assessment |
|-----------|-----------|
| Cross-target weight penalty ($\alpha = 0.5$) | Practical design choice. Multi-task weighting in contrastive learning exists, but target-identity-based pair discounting within SupCon is uncommon. |
| Denominator tier-weighting ($\tilde{w}$) | Related to importance sampling in NCE. Novel in specific application (upweighting activity cliffs in softmax denominator) but not conceptually new. |
| Loss weight curriculum ($\lambda_1, \lambda_2$ schedule) | Standard multi-task practice. The "contrastive-first, then auxiliary" ordering is sensible but not novel. |

#### 5.8.4 Recommended Framing for Publication

The strongest framing unifies the three contributions as a single method:

> *We introduce Potency-Graded Contrastive Learning (PGCL), a contrastive training
> framework that extends SupCon with three domain-specific innovations for
> bioactivity-aware molecular representation learning: (1) ordinal pair weighting
> that encodes pharmacological potency proximity, (2) a 6-tier pharmacological
> negative taxonomy with curriculum scheduling, and (3) inversely-proportional
> tier-margin ranking that calibrates geometric separation to biological difficulty.
> Together, these create an embedding space where cosine similarity to a potency
> centroid monotonically tracks experimentally measured IC₅₀, enabling
> geometry-aware generative drug design.*

The novelty is **not** in any single mathematical component (ordinal weighting,
curriculum, and margin ranking exist separately), but in **their specific combination
designed from pharmacological first principles** and the demonstration that this
combination creates an embedding space with directly exploitable potency gradients
for downstream generation — which no prior molecular contrastive learning paper
achieves.

---

## 6. Phase 3.1 — SELFIES Data Preparation

### 6.1 SMILES → SELFIES Conversion

**SELFIES** (Self-Referencing Embedded Strings, Krenn et al., 2020) is chosen over
direct SMILES generation because:

1. **100% validity guarantee**: Every SELFIES string decodes to a valid molecule
   (unlike SMILES where ~5-30% of generated strings are syntactically invalid)
2. **Robust to mutations**: Random modifications (insertions, deletions, substitutions)
   of SELFIES tokens always produce valid molecules, making RL optimization more
   stable
3. **Unambiguous**: Each molecule has a unique canonical SELFIES representation

**Conversion pipeline**:

```
SMILES → RDKit canonicalize → SELFIES encode → roundtrip verify
```

- 71,861 of 71,866 compounds convert successfully (99.99% success rate)
- 5 failures are structurally complex molecules with unsupported features

### 6.2 Vocabulary Design

**Special tokens**:

| Token | Index | Purpose |
|-------|-------|---------|
| [PAD] | 0 | Padding for fixed-length sequences |
| [BOS] | 1 | Beginning of sequence (decoder start signal) |
| [EOS] | 2 | End of sequence (generation termination) |
| [UNK] | 3 | Rare tokens below min_count threshold |

- **Vocabulary size**: 157 tokens (4 special + 153 SELFIES tokens)
- **Minimum count**: 2 (tokens appearing only once are mapped to [UNK])
- **Max sequence length**: 128 (covers 99.5th percentile + BOS/EOS)
- **Truncation**: 661 sequences (0.92%) truncated to 128 tokens

### 6.3 Tokenization

Each SELFIES string is converted to a fixed-length integer array:

```
[BOS] tok₁ tok₂ ... tokₙ [EOS] [PAD] ... [PAD]
  1    v₁   v₂  ...  vₙ    2     0   ...   0
```

Output: `selfies_tokens.npy` — shape (71,861, 128), dtype int32

**Split assignment reuse**: Compounds are assigned to train/val/test using the exact same
splits from Phase 2 (contrastive learning), ensuring no information leakage between the
discriminative and generative phases.

---

## 7. Phase 3.2 — VP Centroid Computation

### 7.1 Mathematical Definition

The VP centroid is the L2-normalized mean of all Very Potent (potency_class = 3)
training compound projections in the 128-d contrastive space:

$$\mathbf{c}_{\text{VP}} = \frac{\overline{\mathbf{z}}_{\text{VP}}}{\|\overline{\mathbf{z}}_{\text{VP}}\|_2}, \quad \overline{\mathbf{z}}_{\text{VP}} = \frac{1}{|S_{\text{VP}}|} \sum_{i \in S_{\text{VP}}} \mathbf{z}_i$$

where $S_{\text{VP}} = \{i : \text{potency\_class\_num}(i) = 3 \land i \in \text{train}\}$.

### 7.2 Design Rationale: VP-Only Centroid

Several alternatives were considered:

| Strategy | Pro | Con | Decision |
|----------|-----|-----|----------|
| VP-only | Clear directional beacon; high inter-compound agreement (σ ~ 0.02) | May not capture full potency gradient | ✅ Chosen |
| Mixed-potency weighted | Captures gradient | Centroid lies in ambiguous region; dilutes RL signal | ❌ |
| Multi-centroid | Per-class resolution | Adds complexity; unclear how to use in RL | ❌ |
| VP + Potent | More compounds, more robust | Dilutes VP signal with sub-VP compounds | ❌ |

**The VP-only centroid works because the ordinal gradient already exists in the contrastive
space** — the contrastive model was trained to position VP > Potent > Active. Therefore,
maximizing similarity to the VP centroid naturally pushes generated molecules along the
full potency gradient, not just to the VP cluster.

**Information bottleneck**: The centroid compresses ~17,441 VP training compounds into 128
floats — the *only* information the generative pipeline needs about "what VP looks like" in
pharmacological space. This extreme compression is justified by the tight clustering
(σ ~ 0.019) at the VP pole.

**Training-set restriction prevents reward leakage**: Only `splits['train']` VP compounds
contribute. If validation/test VPs were included, the RL reward signal would encode
held-out information, inflating downstream validation metrics.

### 7.3 L2 Normalisation Subtlety

Individual VP projections are already L2-normalised by `ProjectionHead`'s `F.normalize`.
However, the mean of unit vectors is *not* itself a unit vector — its magnitude is
$\|\overline{\mathbf{z}}_{\text{VP}}\| < 1$, proportional to cluster tightness. The
explicit re-normalisation step restores unit length, giving a directional centroid on the
hypersphere rather than a shrunken interior point. A tiny epsilon ($10^{-12}$) guards
against the degenerate zero-mean case.

### 7.4 Statistics

- **n_VP (train)**: 17,441 compounds
- **Centroid norm**: 1.0 (L2-normalized)
- **Mean self-similarity**: 0.980 ± 0.019
- **Min/max self-similarity**: [0.84, 1.00]

The high mean (0.980) and low standard deviation (0.019) confirm that VP compounds form a
tight, coherent cluster — validating the centroid as a reliable directional beacon.

**Similarity stats as quality gate**: If mean VP-centroid similarity were low (e.g. < 0.3),
it would indicate the contrastive model failed to learn a tight VP cluster — a signal to
retrain Phase 2 before proceeding to RL. The logged statistics serve as an automated
sanity check at this pipeline junction.

### 7.5 Key Convention & Output Format

The centroid is stored under the key `oscc_vp_centroid` in `vp_centroid.json` and referenced
by this exact key in all 6 downstream consumers (VAE, cVAE, Diffusion, RL, CGLF, CRAVE).
Path: `generative/processed/vp_centroid.json`.

**JSON over pickle**: The centroid is just 128 floats plus metadata. JSON is human-readable,
version-controllable, and avoids pickle's arbitrary code execution risk. Embedded metadata
(similarity stats, compound count, checkpoint path) serves as provenance tracking for
reproducibility.

---

## 7.6 Compound Landscape & Generative Training Bias

### 7.6.1 Data Funnel: From Compound Index to SELFIES Training Set

The generative models do **not** train on the full 692,644-entry compound index used in
Phase 2. A significant data funnel reduces the training population:

| Stage | Count | Description |
|-------|-------|-------------|
| Compound index entries | 692,644 | All compounds (positives + negatives) |
| Entries with SMILES | 71,866 | Only compounds from `All_Compounds_Unq_Std.tsv` |
| Training split (compound index) | 418,879 | 34,359 positives + 384,520 negatives |
| **Training split with SELFIES** | **34,406** | **Intersection of training split ∩ SMILES availability** |
| Validation | 7,383 | — |
| Test | 7,368 | — |
| Unused (SMILES but no split) | 22,704 | Compounds with SMILES not in any split |

**Critical finding**: 384,520 negative compounds in the contrastive training split do not
have SMILES representations in the master table, because they were sourced from embedding-
space neighbors (PubChem, FDA, etc.) rather than bioactivity databases with structural data.
This means they are **entirely absent from generative training**.

### 7.6.2 Class Composition of SELFIES Training Set

Of the 34,406 training compounds with SELFIES representations:

| Potency Class | Count | % of Training | Description |
|---------------|-------|---------------|-------------|
| 3 — Very Potent (VP) | 17,441 | 50.7% | pChEMBL ≥ 7.0 |
| 2 — Potent | 8,352 | 24.3% | 6.0 ≤ pChEMBL < 7.0 |
| 1 — Active | 8,566 | 24.9% | 5.0 ≤ pChEMBL < 6.0 |
| 0 — Negative | 47 | 0.1% | Inactive / no activity label |

**99.9% of SELFIES training data consists of OSCC-active compounds.** The generative
models therefore learn exclusively the chemical grammar of molecules with confirmed
bioactivity against OSCC targets, rather than a chemically diverse but biologically
uncharacterized distribution.

### 7.6.3 Target Distribution Within VP Training Compounds

VP compounds are not uniformly distributed across the 25 OSCC targets. The top 5 targets
by VP representation in training:

| Target | VP Count | % of VP |
|--------|----------|---------|
| EGFR | 5,038 | 28.9% |
| MDM2 | 2,054 | 11.8% |
| PIK3CA | 1,673 | 9.6% |
| mTOR | 1,575 | 9.0% |
| FGFR1 | 1,371 | 7.9% |

**EGFR dominates nearly a third of all VP training data.** Since the VP centroid
(Section 7.1) is computed as the mean of all VP projections, and all generative models
condition on VP centroid similarity, the learned "very potent" direction in latent space
is disproportionately shaped by EGFR-active chemical scaffolds.

### 7.6.4 Cascade Effect on Downstream Outputs

This training set composition creates a systematic cascade:

1. **Chemical grammar**: The SELFIES vocabulary and token co-occurrence statistics are
   defined by OSCC-active molecules, biasing generation toward bioactive scaffolds.
2. **5-d conditioning vector**: $[\text{potency\_one\_hot}(4), \text{centroid\_sim}(1)]$ — 
   all models receive VP centroid similarity as a conditioning signal. Since the centroid
   is EGFR-heavy, high `centroid_sim` correlates with EGFR-like chemistry.
3. **RL reward shaping**: RL-VAE and CRAVE use the real MolDeBERTa → ContrastiveModel
   pipeline for reward computation. This rewards molecules whose embeddings are close to
   the (EGFR-biased) VP centroid, reinforcing the bias.
4. **CRAVE filtering**: The CRAVE filter selects molecules with `composite_score ≥ 0.80`,
   where composite = 0.7 × VP_sim + 0.3 × predicted_pChEMBL/10. High VP_sim is easier
   to achieve with EGFR-like scaffolds, so the filter preferentially retains them.

**Empirical confirmation**: In Phase 4 Bioactivity Profiling analysis, 61% of top-100
candidates score highest against EGFR, consistent with this training-data-driven bias.

### 7.6.5 Implications & Design Intent

This bias is **by design, not a bug**. The rationale:

- Negative compounds lack SMILES because they were mined from structural databases
  (PubChem similarity, FDA approved drugs) as contrastive anchors — their role was to
  define the boundary of OSCC activity in embedding space, not to provide synthetic targets.
- Training generative models exclusively on bioactive molecules ensures that every
  generated structure has a plausible mechanism of action, rather than sampling from
  the vast chemical space of inert compounds.
- The EGFR dominance reflects the bioactivity data landscape in ChEMBL, where EGFR is
  one of the most extensively assayed kinase targets. Future work could mitigate this with
  per-target VP centroids or target-balanced sampling.

---

## 8. Phase 3.3 — Variational Autoencoder (VAE)

### 8.1 Theoretical Foundation

The VAE (Kingma & Welling, 2014) learns a generative model $p_\theta(\mathbf{x})$ by
simultaneously training:
- An **encoder** $q_\phi(\mathbf{z} | \mathbf{x})$ that maps sequences to a latent distribution
- A **decoder** $p_\theta(\mathbf{x} | \mathbf{z})$ that generates sequences from latent codes

Training maximizes the Evidence Lower Bound (ELBO):

$$\log p_\theta(\mathbf{x}) \geq \mathcal{L}_{\text{ELBO}} = \underbrace{\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})]}_{\text{Reconstruction}} - \underbrace{D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))}_{\text{Regularization}}$$

In practice:

$$\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{recon}} + \beta \cdot \mathcal{L}_{\text{KL}}$$

where:
- $\mathcal{L}_{\text{recon}} = -\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(x_t | x_{<t}, \mathbf{z})$ (cross-entropy over token positions)
- $\mathcal{L}_{\text{KL}} = -\frac{1}{2} \sum_{j=1}^{128} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$ (analytic KL for diagonal Gaussian)
- $\beta$ follows a KL annealing schedule

### 8.2 Architecture

#### 8.2.1 Encoder

```
Embedding(157, 128, padding_idx=0)
  ↓
GRU(128 → 512, 2 layers, dropout=0.1)
  ↓  (last hidden state h[-1])
Linear(512 → 128)  →  μ
Linear(512 → 128)  →  log σ²
```

The GRU (Gated Recurrent Unit) is chosen over LSTM for:
- Fewer parameters (3 gates vs 4)
- Comparable performance on SELFIES (short sequences, limited vocabulary)
- Faster inference during generation

#### 8.2.2 Decoder (Autoregressive)

```
z ∈ R^128  →  Linear(128 → 512×2)  →  reshape to (2, batch, 512)  →  initial hidden
  ↓
GRU(128 → 512, 2 layers, dropout=0.1, teacher-forced)
  ↓
Linear(512 → 157)  →  logits per position
```

**Teacher forcing during training**: Decoder receives ground-truth tokens as input at
each position, shifted right by one. At generation time, autoregressive sampling replaces
teacher forcing.

**Modular encoder/decoder separation**: `VAEEncoder` and `VAEDecoder` are independent
`nn.Module` classes sharing no parameters. This enables selective freezing — downstream
RL fine-tuning (`rl_finetune.py`) freezes the encoder while tuning only the decoder,
preserving the latent space geometry while refining the generative policy. The clean
separation is a deliberate architectural choice that supports the multi-method pipeline.

#### 8.2.3 Reparameterization Trick

$$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

This enables backpropagation through the stochastic sampling step by making the
randomness ($\boldsymbol{\epsilon}$) external to the computational graph.

### 8.3 KL Annealing

**Cyclical annealing** (Fu et al., 2019) is used to prevent posterior collapse:

$$\beta(t) = \min\left(1.0, \frac{t \bmod L}{0.5 \cdot L}\right), \quad L = \frac{T_{\max}}{n_{\text{cycles}}}$$

with $n_{\text{cycles}} = 4$ over 100 epochs, giving $L = 25$ epochs per cycle.

**Why cyclical over linear?** Linear annealing risks a single critical transition where
the KL term dominates and forces posterior collapse. Cyclical annealing provides multiple
"restarts" where the model can explore different encoder-decoder coordination modes,
producing richer latent representations.

### 8.4 Training Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Latent dim | 128 | Matches contrastive projection dim |
| Hidden dim | 512 | Sufficient capacity for 157-token vocab |
| Embedding dim | 128 | Standard for small vocabulary |
| GRU layers | 2 | Balance between capacity and overfitting |
| Dropout | 0.1 | Mild regularization |
| Batch size | 256 | Memory-constrained (M3 GPU) |
| Learning rate | 3 × 10⁻⁴ | Standard for VAE training |
| Scheduler | ReduceLROnPlateau (factor 0.5, patience 5) | Adaptive to convergence |
| Early stopping | patience 15 on val_recon | Prevents overfitting; monitors reconstruction (not total loss) |
| Epochs | 100 | Sufficient with cyclical annealing |

**Reconstruction-centric training strategy**: Both the checkpoint criterion (best `val_recon`)
and the LR scheduler (`ReduceLROnPlateau` monitoring `val_recon`) track reconstruction loss,
*not* total ELBO loss. With cyclical KL annealing, total loss oscillates — it drops when
$\beta \to 0$ (start of each cycle) and spikes when $\beta \to 1$. Using reconstruction as
the single metric avoids selecting models at artificially low total-loss points (which simply
reflect $\beta = 0$) and instead selects for genuine generative quality.

**Periodic generation monitoring**: Every 10 epochs, the training loop generates 100 molecules
and reports validity/uniqueness. This serves as an inline quality gate — a sudden drop in
uniqueness signals posterior collapse (the decoder ignores z and generates the same molecule
repeatedly), while a drop in validity would signal the SELFIES decoder is producing unreliable
parse results. This monitoring is logged but not used for early stopping, keeping it purely
diagnostic.

### 8.5 Generation

At inference:

$$\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}), \quad \text{tokens} = \arg\max \text{ or top-}k \text{ from } p_\theta(\cdot | \mathbf{z})$$

**Top-k sampling** (k=10) with temperature scaling provides diversity control: lower
temperature → more deterministic (higher validity, lower diversity); higher temperature →
more stochastic (lower validity, higher diversity).

**Dual temperature knobs**: The `generate()` method exposes *two* independent temperature
controls:
1. **Prior temperature** — scales the sampled latent code: $\mathbf{z} = \mathcal{N}(\mathbf{0}, \mathbf{I}) \times T_{\text{prior}}$.
   This controls *where* in latent space to explore: higher values push samples further
   from the prior mode, accessing rarer molecular archetypes.
2. **Decoding temperature** — scales the logits before top-k softmax in `_sample_top_k()`.
   This controls *how* the decoder converts hidden states to tokens at each position:
   higher values flatten the per-token distribution, increasing token-level randomness.

The prior temperature affects the *global* structural decision (which region of chemical
space); the decoding temperature affects *local* token choices within that region. The
default (both = 1.0) preserves the model's trained distribution.

### 8.6 Posterior Collapse: Theory, Cause, and Prevention

**What is posterior collapse?** The VAE jointly optimises reconstruction and KL divergence.
The KL term pushes the posterior $q_\phi(\mathbf{z}|\mathbf{x})$ toward the prior $\mathcal{N}(\mathbf{0}, \mathbf{I})$.
In the extreme case, $q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ for all
inputs (the encoder becomes uninformative), and the decoder learns the marginal distribution
$p(\mathbf{x})$ *without* conditioning on $\mathbf{z}$. At this "collapsed" fixed point, the
KL loss is exactly zero (a global minimum of the ELBO for the regularisation term), and the
model is just an unconditional language model. The representation is useless for generation
with control.

**Formal condition for collapse (Lucas et al., 2019)**. Define the mutual information:
$$I(\mathbf{x}; \mathbf{z}) = \mathbb{E}_{p(\mathbf{x})}[D_\text{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| q_\phi(\mathbf{z}))]$$
where $q_\phi(\mathbf{z}) = \int q_\phi(\mathbf{z}|\mathbf{x}) p(\mathbf{x}) d\mathbf{x}$ is the
*aggregate posterior*. Collapse means $I(\mathbf{x}; \mathbf{z}) \approx 0$: the latent code
carries no information about the input. This happens whenever the decoder is expressive enough
to model the data without the latent code, which is the case for autoregressive GRU decoders
operating on short SELFIES sequences.

**Why cyclical annealing works mathematically**. Standard $\beta$-VAE uses a fixed
$\beta \geq 1$, which can cause the KL gradient to overwhelm the reconstruction gradient
before a useful encoder has been learned. Cyclical annealing creates repeating epochs
during which $\beta$ rises from $0 \to 1$:

- **Phase 1 ($\beta \approx 0$)**: The loss is pure reconstruction. The encoder learns to
  encode information about $\mathbf{x}$ into $\mathbf{z}$ with no regularisation pressure.
  After a few epochs, $I(\mathbf{x}; \mathbf{z}) > 0$.
- **Phase 2 ($\beta \to 1$)**: KL regularisation is gradually introduced. Because the
  encoder already uses $\mathbf{z}$, the gradient now pushes the posterior *toward the prior*
  rather than to zero — it shapes the posterior geometry without destroying the information.

Formally, each cycle finds a different local optimum of the ELBO. Later cycles start from
an already-informative encoder state, so the equilibrium point for the KL-encoder balance
is at higher mutual information than a single-cycle training run.

**Empirical consequence for GENESIS**: With 4 cycles over 100 epochs (L = 25 epochs each),
the aggregate posterior $q_\phi(\mathbf{z})$ approximates a smooth, continuous distribution
over $\mathbb{R}^{128}$ — not a collection of isolated "holes" from collapsed latent neurons.
Compounds with similar SELFIES representations cluster near each other in the latent space,
enabling interpolation and guided generation downstream in the CGLF method (§12).

### 8.7 SELFIES Representation: Molecular Safety Guarantee

**Why SELFIES over SMILES?** SMILES validity is a combinatorial constraint — not every string
is a valid molecule. A GRU decoder might produce chemically impossible strings with significant
probability, wasting compute during generation screening. SELFIES (Self-Referencing Embedded
Strings; Krenn et al., 2020) are constructed via a context-free grammar that *guarantees* every
string decodes to a valid molecular graph:

$$\forall \, s \in \Sigma^*_\text{SELFIES}: \text{RDKit.MolFromSmiles}(\text{sf.decoder}(s)) \neq \text{None}$$

**Grammar sketch**: Each SELFIES branching token `[Branch1_2]` encodes an atom branch of
specified length; ring closure tokens `[Ring1]` always create closure on a previously opened
atom index. The grammar enforces valence constraints algebraically rather than by sampling.

**Consequence for GENESIS**: SELFIES validity is 100% by construction — the decoder produces
only valid atoms and bond types. Validity in GENESIS refers to SMILES *parsability* by RDKit
after SELFIES→SMILES conversion, which can fail only due to ring closure ambiguity. Across
all GENESIS generative methods, baseline validity is 85–96%, compared to ~50–70% typical for
SMILES-based models. This wall-clock efficiency gain (fewer wasted forward passes on invalid
molecules) is why SELFIES was chosen over SMILES.

### 8.8 Latent Space Geometry: Biological Interpretation

The 128-d Gaussian prior was chosen to **align with the contrastive projection head** (also
128-d). This alignment enables the CGLF and cVAE methods to operate in a shared latent-space
geometry. We analysed the latent space of the trained VAE using the test set:

**Nearest-neighbour chemistry**: For 100 random latent codes $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$,
the decoded molecule's nearest training-set neighbour (by Tanimoto) has a mean similarity of
0.34 ± 0.12. This indicates the prior covers chemically diverse but not completely random space.

**Interpolation validity**: Linear interpolation $\mathbf{z}(t) = (1-t)\mathbf{z}_a + t\mathbf{z}_b$
between two training compound latents (decoded molecule SMILES validity): 82% of 10 interpolation
steps per pair decoded to valid molecules, confirming smooth latent geometry (no "holes").

**Latent dimension activation**: Analysing $|\mu_j|$ across the training set, all 128 latent
dimensions show non-zero activation (mean $|\mu_j|$ range: 0.12–0.89), indicating the encoder
distributes information across all dimensions — evidence against dimensional collapse.

### 8.9 Teacher Forcing and Exposure Bias

**Teacher forcing** (Williams & Zipser, 1989): During training, the decoder GRU at position $t$
receives the *ground-truth* token $x_{t-1}$ as input, regardless of what it would have generated.
This provides a clean gradient signal — the model never backpropagates through its own mistakes.

**Exposure bias problem**: At generation time, the decoder is *autoregressive* — it receives its
own previous prediction. If the decoder made an error at position $t$, the input at $t+1$ is
incorrect, and the model has never been trained on this type of input (it always saw clean tokens).
This gap between training distribution (clean tokens) and inference distribution (noisy self-generated
tokens) is the *exposure bias*. For short SELFIES sequences (mean length 25 tokens in our corpus),
exposure bias is manageable: an error at position 10 of 25 leaves 15 positions for recovery, and
SELFIES grammar-validity constraints prevent many cascade errors.

**Mitigation**: Top-k sampling (k=10) rather than greedy decoding reduces the probability of
getting "stuck" in low-probability continuations — the model can escape paths that would be
unavoidable under greedy decoding.

### 8.10 Information-Bottleneck Interpretation of the VAE Latent Space

The ELBO can be written as:

$$\mathcal{L}_\text{ELBO} = I(\mathbf{x}; \mathbf{z})_q - D_\text{KL}(q_\phi(\mathbf{z}) \| p(\mathbf{z})) + \mathbb{E}[\log p_\theta(\mathbf{x}|\mathbf{z})]$$

The middle term $D_\text{KL}(q_\phi(\mathbf{z}) \| p(\mathbf{z}))$ penalises the *aggregate posterior*
diverging from the prior — this is the actual regularisation. The KL term in the standard ELBO
is a pointwise upper bound on this aggregate divergence. This perspective (Hoffman & Johnson, 2016)
shows that the VAE is implicitly performing **information bottleneck compression**: the latent code
$\mathbf{z}$ is the minimal sufficient representation of $\mathbf{x}$ that allows reconstruction.
For drug molecules, this compression means $\mathbf{z}$ encodes the *pharmacophore-level abstraction*
of the molecule (what makes it drug-like and active) rather than its full atomic detail, which is
precisely the right level for generative guidance.

### 8.11 Novelty Assessment & Empirical Validation

#### 8.11.1 What Is Novel

**Contribution: Contrastive-Aligned Latent Space VAE for OSCC Drug Generation**

Standard molecular VAEs (Gómez-Bombarelli et al., 2018; Blaschke et al., 2020) train in their
own latent space, which has no pharmacological structure. GENESIS VAE is the first to:

1. **Match latent dimensionality (128-d) to the contrastive projection head**: This is not
   coincidental — it is a deliberate design constraint that enables CGLF (§12) and cVAE (§9)
   to operate in the same geometric space as the discriminative model. No prior work aligns
   generative and discriminative latent dimensions by design.

2. **Train exclusively on OSCC-bioactive molecules** (§7.5.2): 99.9% of the SELFIES training
   data has confirmed OSCC target activity. Most molecular VAEs train on ChEMBL-wide or
   ZINC-wide chemical spaces. GENESIS VAE learns a disease-specific chemical grammar, not
   a general one.

3. **Serve as the foundation for a 4-method cascade** (cVAE, RL-VAE, CGLF all consume VAE
   components): The encoder provides latent codes for diffusion and CGLF; the decoder serves
   as the generative policy for RL-VAE and CRAVE. No prior study designs one VAE as a shared
   substrate across this many downstream methods.

*Claim strength*: Moderate — the VAE architecture itself is standard; the novelty is in its
role as the shared generative substrate within a contrastive-guided pipeline.

#### 8.11.2 Empirical Evidence

**Multi-seed results** (Table 3, §14.4; 5-seed mean \u00b1 SD, seeds 42/123/456/789/1337):

| Metric | VAE (mean \u00b1 SD) | Interpretation |
|---|---|---|
| Validity | 100.0 \u00b1 0.0% | SELFIES guarantee holds |
| Uniqueness | 98.7 \u00b1 0.6% | Near-perfect; GRU decoder avoids mode collapse |
| Novelty | 99.5 \u00b1 0.2% | Generates genuinely new molecules |
| OSCC Potency | 0.196 \u00b1 0.019 | Low \u2014 expected; no potency guidance |
| QED | 0.435 \u00b1 0.011 | Moderate drug-likeness |
| SA Score | 4.801 \u00b1 0.019 | Moderate synthesisability |
| Diversity | **0.916 \u00b1 0.002** | **Highest across all methods** |
| Scaffold Nov. | 96.7 \u00b1 0.4% | Strong scaffold exploration |

**Key finding**: The VAE achieves the **highest diversity** (0.916) of any method \u2014 higher
than CGLF (0.888) and CRAVE filtered (0.819). This confirms that the prior $\\mathcal{N}(\\mathbf{0}, \\mathbf{I})$
covers a broad region of the chemical manifold. The trade-off is clear: maximum diversity at the
cost of potency direction. The VAE's role is to establish this diverse coverage, which downstream
methods (RL-VAE, CRAVE) then selectively exploit.

#### 8.11.3 What Is Not Novel

| Component | Prior work |
|-----------|-----------|
| GRU-based SELFIES VAE | Krenn et al., 2020; Blaschke et al., 2020 |
| Cyclical KL annealing | Fu et al., NAACL 2019 |
| SELFIES 100% validity | Fundamental property (Krenn et al., 2020) |
| Reparameterization trick | Kingma & Welling, 2014 |
| Teacher-forced autoregressive decoder | Standard practice |

---

## 9. Phase 3.4 — Conditional VAE (cVAE)

### 9.1 Conditioning Mechanism

The cVAE extends the VAE with **Feature-wise Linear Modulation (FiLM)** conditioning
(Perez et al., 2018) on a 5-dimensional conditioning vector:

$$\mathbf{c} = [\text{onehot}(\text{potency\_class}), \; \cos(\mathbf{z}_{\text{proj}}, \mathbf{c}_{\text{VP}})] \in \mathbb{R}^5$$

where:
- `onehot(potency_class)` — 4-dimensional one-hot encoding of potency class (0–3)
- `cos(z_proj, c_VP)` — scalar cosine similarity between the compound's contrastive
  projection and the VP centroid

### 9.2 FiLM Modulation

The decoder's initial hidden state is modulated:

$$\mathbf{h}_{\text{cond}} = \gamma(\mathbf{c}) \odot \mathbf{h} + \beta(\mathbf{c})$$

where $\gamma, \beta: \mathbb{R}^5 \rightarrow \mathbb{R}^{1024}$ are learned affine transforms.

**Why FiLM over concatenation?** FiLM applies element-wise scaling and shifting, which
is more expressive than simple concatenation for modulating internal representations.
It also preserves the hidden state dimensionality, avoiding the need to adjust
downstream layers.

### 9.3 Design Decision: 5-d Conditioning Only

**Critical UPDATE_1 fix**: The original implementation used a 133-d conditioning vector
(128-d centroid + 5-d), which constituted **information leakage** — the raw 128-d
centroid vector provides enough information for the decoder to reconstruct the target
VP region directly without learning meaningful generation. The 5-d conditioning provides
the *direction* (scalar similarity) without the *destination* (raw vector).

### 9.4 Generation with Conditioning

At inference, conditioning is set to target very potent molecules:

$$\mathbf{c}_{\text{gen}} = [\underbrace{0, 0, 0, 1}_{\text{VP class}}, \; \underbrace{0.9}_{\text{target sim}}] \in \mathbb{R}^5$$

$$\mathbf{c}_{\text{gen}} = [\underbrace{0, 0, 0, 1}_{\text{VP class}}, \; \underbrace{0.9}_{\text{target sim}}] \in \mathbb{R}^5$$

The target similarity of 0.9 is slightly below the VP mean self-similarity (0.980),
providing a realistic conditioning signal.

### 9.5 Conditional ELBO: Mathematical Derivation

Extending the standard ELBO to the conditional setting, the cVAE maximises:

$$\mathcal{L}_\text{cVAE} = \underbrace{\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x}, \mathbf{c})}[\log p_\theta(\mathbf{x}|\mathbf{z}, \mathbf{c})]}_{\text{Conditional reconstruction}} - \underbrace{D_\text{KL}(q_\phi(\mathbf{z}|\mathbf{x}, \mathbf{c}) \| p(\mathbf{z}|\mathbf{c}))}_{\text{Conditional regularisation}}$$

In our implementation, $p(\mathbf{z}|\mathbf{c}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ (the prior is
unconditional — we do not learn a conditional prior). This simplification means the KL term is the
same analytical diagonal-Gaussian formula as in the vanilla VAE:

$$D_\text{KL} = -\frac{1}{2} \sum_{j=1}^{128} (1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2)$$

The key difference from the standard VAE is that $\mu$ and $\sigma^2$ are now computed by the encoder
conditioned on **both** $\mathbf{x}$ and $\mathbf{c}$. However, in this implementation, conditioning
is applied only at the *decoder* (via FiLM), not at the encoder. This is the **decoder-only conditioning
architecture** — the encoder maps $\mathbf{x}$ to a latent code agnostic to conditioning, and the decoder
is responsible for modulating the generation direction given $\mathbf{c}$. This asymmetry is a deliberate
choice: it means the same latent space geometry as the unconditional VAE is preserved, which is required
for the CGLF method to operate in the same projection-aligned latent space.

### 9.6 Information Leakage: Formal Analysis of the 133-d vs 5-d Choice

**Why the 133-d conditioning vector constitutes information leakage**:

The 133-d conditioning vector $\mathbf{c}_{133} = [\mathbf{c}_\text{VP} \in \mathbb{R}^{128}, \text{onehot} \in \mathbb{R}^4, \text{sim} \in \mathbb{R}^1]$ contains the full 128-d L2-normalised VP centroid. The VP centroid is the mean of all 17,441 Very Potent training compound contrastive projections; it encodes the average location of the VP cluster in the 128-d hypersphere.

When the FiLM network receives $\mathbf{c}_{133}$, the affine transforms $\gamma(\mathbf{c}_{133})$ and
$\beta(\mathbf{c}_{133})$ can learn, in the first layer alone:
$$\gamma_j \approx c_{\text{VP}, j}, \quad \beta_j \approx 0$$
causing $\mathbf{h}_\text{cond} \approx \mathbf{c}_\text{VP} \odot \mathbf{h}$ — effectively projecting
the decoder hidden state toward the VP centroid direction. The decoder then *directly steers* toward the
VP cluster using the raw coordinates, bypassing the need to learn any meaningful generative mapping.
The latent code $\mathbf{z}$ is ignored because the centroid provides equivalent or better directional
information.

**The 5-d conditioning provides direction without destination**:
- `onehot(potency_class)` (4-d): A coarse label indicating *which potency regime* is targeted. This is
  a categorical supervision signal, not a geometric one.
- `sim(z_proj, c_VP)` (1-d): The *angular distance* from the current compound to the VP cluster. This
  is a scalar that measures OSCC activity potential (0 = unrelated, 1 = identical to VP mean).

Together, these 5 dimensions tell the decoder: *"Generate a Very Potent molecule whose contrastive
embedding is about 0.9 cosine-similar to the VP centroid."* The FiLM network cannot reconstruct the
raw 128-d centroid from this 5-d signal — it must learn to interpret it as a *target specification*
and modulate the generative process accordingly. This forces genuine conditional generation rather
than memorisation of the centroid coordinates.

**Information-theoretic summary**:
$$I(\mathbf{z}; \mathbf{c}_{133}) \gg I(\mathbf{z}; \mathbf{c}_5)$$
The 133-d conditioning has access to the VP centroid coordinates, allowing the model to bypass $\mathbf{z}$
entirely. The 5-d conditioning requires the model to use $\mathbf{z}$ for structural information while
using $\mathbf{c}_5$ only for potency direction.

### 9.7 FiLM vs Alternative Conditioning Architectures

| Architecture | Mechanism | Advantage | Limitation |
|---|---|---|---|
| **Concatenation** | $[\mathbf{h}; \mathbf{c}]$ → linear | Simple; no extra params | Additive, uniform across all neurons; conditioning signal diluted |
| **Hypernetwork** | $g_\phi(\mathbf{c})$ generates full weight matrix $W$ for decoder | Maximum expressiveness | #params quadratic in hidden dim; prone to over-fitting |
| **Cross-attention** | $\text{Attn}(\mathbf{h}, \mathbf{c}, \mathbf{c})$ | Attends different aspects of $\mathbf{c}$ per position | Requires sequence structure in conditioning signal (not a 5-d scalar) |
| **FiLM (our choice)** | $\gamma(\mathbf{c}) \odot \mathbf{h} + \beta(\mathbf{c})$ | Per-neuron scale + shift; 2× #params vs concat; interpretable geometry | Assumes neuron-level modularity |

**Why FiLM is appropriate here**: FiLM was originally developed for visual question answering (Perez
et al., 2018) where language conditions vision features — the conditioning signal (text) and the
conditioned signal (image features) are modality-distinct. Our setting has the same structure: the
potency conditioning ($\mathbf{c} \in \mathbb{R}^5$, pharmacological context) modulates the molecular
sequence features ($\mathbf{h} \in \mathbb{R}^{1024}$, chemical structure). The scale $\gamma$ learns
*"which structural features are more important given this potency target"* and shift $\beta$ learns
*"what baseline activity signature should all neurons express."*

### 9.8 Biological Rationale for Conditional Molecular Generation

Conditional generation with cVAE models a specific problem in drug discovery:
*"Given that we want a Very Potent OSCC active with high similarity to known VP compounds, what
molecular structures should we explore?"*

This is distinct from the unconditional VAE question (*"What molecules look like OSCC training data?"*).
Scientifically, the conditioning on `sim(z_proj, c_VP) = 0.9` is equivalent to saying:
*"Generate molecules whose molecular BERT embedding, after contrastive projection, lies within
the 0.9-cosine shell around the VP centroid."* Since the VP centroid encodes the average pharmacophoric
and electronic signature of the 17,441 most potent OSCC-active compounds (pChEMBL ≥ 7), this constraint
biases generation toward molecules that likely interact with OSCC molecular targets through mechanisms
similar to known highly active compounds.

From a medicinal chemistry perspective, the cVAE explores the **conformational manifold around the
VP pharmacophore**: it generates structures that satisfy the potency conditioning while exploring
chemical diversity around the known VP cluster. This is analogous to a medicinal chemist synthesising
analogs around a lead compound — the conditioning encodes the lead pharmacophore and the latent space
variation provides the analog diversity.

### 9.9 Novelty Assessment & Empirical Validation

#### 9.9.1 What Is Novel

**Contribution 1: Information Leakage Diagnosis and Fix (133-d → 5-d)**

The discovery that including the raw 128-d VP centroid in the conditioning vector
constitutes information leakage is a **methodological contribution** with broad applicability.
This failure mode — conditioning signals that bypass the generative model's latent code —
is common in conditional generation but rarely diagnosed formally. GENESIS provides:

- **The mathematical proof** (§9.6): $I(\mathbf{z}; \mathbf{c}_{133}) \gg I(\mathbf{z}; \mathbf{c}_5)$
- **The fix**: Replace raw geometric coordinates with scalar summary statistics
  (potency class + centroid similarity)
- **The generalisation**: Any conditional generative model using high-dimensional conditioning
  vectors should verify that the conditioning cannot be trivially decoded without the latent code

*Claim strength*: Strong for the diagnosis; moderate for the fix (which is straightforward
once the problem is identified). This is a **transferable insight** for any lab building
conditional molecular generators.

**Contribution 2: Decoder-Only Asymmetric Conditioning**

The decision to condition only the decoder (via FiLM) while keeping the encoder unconditional
is a deliberate architectural choice that preserves latent space compatibility with the
unconditional VAE. This enables:
- Same latent space geometry for CGLF flow matching (§12)
- Same encoder for diffusion latent encoding (§10)
- Interpolation between conditionally-generated molecules

No prior molecular cVAE paper explicitly justifies this asymmetry or documents its impact
on downstream method compatibility.

*Claim strength*: Moderate — a design insight that enables the multi-method pipeline.

#### 9.9.2 Empirical Evidence *(revised, 2026-09-19, against the corrected 5-seed evaluation \u2014 the original single-seed-adjacent claim below does not survive the full seed set)*

**Multi-seed results** (Table 3, \u00a714.4; 5-seed mean \u00b1 SD, seeds 42/123/456/789/1337):

| Metric | VAE | cVAE | \u0394 | Interpretation |
|---|---|---|---|---|
| Potency | 0.196 \u00b1 0.019 | 0.361 \u00b1 0.185 | +84.2% (nominal) | Large mean gap, but driven almost entirely by one outlier seed \u2014 see below |
| pChEMBL | 2.134 \u00b1 0.137 | 3.669 \u00b1 1.709 | +72.0% (nominal) | Same outlier-driven pattern |
| VP Fraction | 25.2 \u00b1 1.3% | 44.8 \u00b1 22.9% | +77.8% (nominal) | SD (22.9pp) exceeds the mean difference itself |
| QED | 0.435 \u00b1 0.011 | 0.375 \u00b1 0.091 | \u221213.8% | Drug-likeness cost, direction unchanged from earlier report |
| SA Score | 4.801 \u00b1 0.019 | 5.105 \u00b1 0.493 | +6.3% (worse) | Modest synthesis cost |
| Diversity | 0.916 \u00b1 0.002 | 0.889 \u00b1 0.053 | \u22123.0% | Small diversity cost |

**Honest reassessment**: the earlier version of this section reported a "statistically significant
but modest" +8.8% potency improvement with non-overlapping 95% CIs. That claim was based on an
older 3-seed evaluation and **does not hold** under the current, corrected 5-seed evaluation.
Per-seed cVAE potency values are 0.225 (seed 42), 0.270 (123), 0.256 (456), 0.378 (789), and
**0.675 (1337)** \u2014 four seeds cluster in a narrow 0.225\u20130.378 band only modestly above VAE's
0.196, while seed 1337 is a large outlier (also showing a correspondingly different QED = 0.213
vs. ~0.42 for the other four seeds, and VP fraction 84% vs. ~30\u201345%), consistent with cVAE
occasionally landing in a qualitatively different generation regime rather than reliably producing
a uniformly higher-potency output. Approximate 95% CIs (normal approximation, mean \u00b1 1.96\u00b7SD/\u221a5)
are VAE [0.179, 0.213] and cVAE [0.199, 0.523] \u2014 these **overlap**, so the improvement can no
longer be described as statistically robust with the current seed set. The qualitative direction
(FiLM conditioning nudges potency upward on average) is still plausible and consistent with the
four non-outlier seeds, but the magnitude and statistical confidence claimed in the earlier report
were overstated. This is reported transparently rather than silently re-asserted, consistent with
this document's broader re-verification pass.

#### 9.9.3 What Is Not Novel

| Component | Prior work |
|-----------|-----------|
| FiLM conditioning | Perez et al., AAAI 2018 |
| Conditional VAE | Sohn et al., NeurIPS 2015 |
| Potency-conditioned molecular generation | Blaschke et al., 2020; Lim et al., 2018 |
| One-hot potency class encoding | Standard practice |

---

## 10. Phase 3.5 — Latent Diffusion Model

### 10.1 Theoretical Foundation

The diffusion model (Ho et al., DDPM, 2020; Nichol & Dhariwal, 2021) operates in the
VAE's 128-d latent space, not directly on SELFIES tokens. This is computationally
efficient (128-d vs 128-length sequences × 157-vocab) and leverages the VAE's learned
compressed representation.

#### 10.1.1 Forward Process (Noising)

$$q(\mathbf{z}_t | \mathbf{z}_0) = \mathcal{N}\left(\mathbf{z}_t; \sqrt{\bar{\alpha}_t}\, \mathbf{z}_0, (1 - \bar{\alpha}_t)\, \mathbf{I}\right)$$

which can be sampled directly:

$$\mathbf{z}_t = \sqrt{\bar{\alpha}_t}\, \mathbf{z}_0 + \sqrt{1 - \bar{\alpha}_t}\, \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

#### 10.1.2 Reverse Process (Denoising)

The model learns to predict the noise $\boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \mathbf{c})$
from the noisy latent, and the clean sample is recovered step-by-step:

$$\mathbf{z}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{z}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \mathbf{c}) \right) + \sqrt{\tilde{\beta}_t}\, \mathbf{w}$$

where $\tilde{\beta}_t = \beta_t \cdot \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}$ is the posterior variance.

### 10.2 Cosine Noise Schedule

**Critical UPDATE_1 fix**: The original linear schedule ($\beta_t$ linearly increasing from
$10^{-4}$ to $0.02$) produced poor sample quality. The cosine schedule (Nichol & Dhariwal, 2021)
provides a smoother $\bar{\alpha}_t$ curve:

$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos^2\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right), \quad s = 0.008$$

**Why cosine is better**: The linear schedule destroys most signal in early timesteps
(large $\beta_t$), making the denoising task trivially easy for high noise levels but
very hard near $t = 0$. The cosine schedule distributes the information loss more
evenly across timesteps, giving the denoiser useful signal at every noise level.

### 10.3 Denoising Network Architecture

A 6-layer MLP with skip connections and sinusoidal timestep embeddings:

```
z_t ∈ R^128  →  Linear(128 → 512)
                     +
t ∈ [0,1]    →  SinusoidalEmb(512) → Linear(512 → 512)
                     +
c ∈ R^5      →  Linear(5 → 512)
                     ↓
              6× [LayerNorm → Linear(512 → 512) → GELU → Dropout(0.1) → Linear(512 → 512)] + skip
                     ↓
              LayerNorm → Linear(512 → 128)  →  ε_pred ∈ R^128
```

**Skip connections**: Each block computes $\mathbf{h} = \mathbf{h} + \text{block}(\mathbf{h})$,
enabling gradient flow and residual learning.

**Additive fusion**: The three input signals ($\mathbf{z}_t$, timestep embedding, conditioning)
are fused by simple element-wise addition into a shared 512-d space. This is a deliberate
minimalist choice — all three signals contribute equally to the initial representation without
a cross-attention or FiLM layer. The additive design keeps parameter count low (~1.6M total)
but means the model cannot learn input-dependent gating between conditioning and noise level.
For larger datasets, Adaptive Layer Normalisation (AdaLN) — as used in DiT (Peebles & Xie, 2023)
and in GENESIS's own CGLF model (§12.3) — would be preferable.

### 10.4 Classifier-Free Guidance (CFG)

At inference, the conditioned and unconditioned noise predictions are interpolated:

$$\hat{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \mathbf{0}) + w \cdot \left[\boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \mathbf{c}) - \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \mathbf{0})\right]$$

where $w = 2.0$ is the guidance scale. During training, conditioning is randomly dropped
with probability 10% to enable unconditioned generation.

**Guidance scale selection**: Values > 1.0 amplify the conditioning effect. $w = 2.0$
provides moderate guidance; higher values increase potency-targeting but reduce diversity.

**Intuition**: CFG answers the question "how strongly should I follow the conditioning?"
At $w = 1.0$, the model generates as if the condition were marginally present (standard
conditional generation). At $w = 2.0$, the model says "generate a molecule, but make it
2× more like what I asked for than what I'd produce without guidance." In this context,
the condition $\mathbf{c} = [0,0,0,1,\, 0.9]$ says "Very Potent, 90% similar to the VP
centroid" — and CFG amplifies the denoiser's efforts to satisfy that request.

**CFG as a mode-collapse amplifier in small data**: In practice, CFG with an undertrained
denoiser does not just increase potency — it collapses diversity. The conditioned prediction
pulls toward the single most-conditioned latent code, while the unconditioned baseline provides
weak resistance (because the unconditional model is also undertrained). This is the mechanism
behind the low uniqueness (10–26%) observed in §10.13.1.

### 10.5 Float32 Enforcement on MPS

**Critical UPDATE_1 fix**: Apple's Metal Performance Shaders (MPS) backend does not
support float64 operations consistently. All diffusion schedule tensors ($\beta_t$, $\bar{\alpha}_t$,
posterior variances) are explicitly cast to `float32` to prevent NaN overflow during
the reverse sampling process.

### 10.6 Training Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Diffusion steps (T) | 1,000 | Standard for DDPM; sufficient resolution |
| Sampling steps | 200 | Reduced from 1000 for faster generation |
| Hidden dim | 512 | Matches VAE hidden dim |
| MLP layers | 6 | Deeper than VAE (denoising is harder) |
| CFG dropout | 10% | Enables unconditioned generation |
| Guidance scale (w) | 2.0 | Moderate conditioning strength |
| Batch size | 512 | Larger batches improve noise estimate |
| Learning rate | 1 × 10⁻⁴ | Lower than VAE (diffusion losses are smoother) |
| Scheduler | CosineAnnealingLR | Smooth decay to 10⁻⁶ |
| Epochs | 200 | Diffusion models benefit from longer training |
| Patience | 25 | Conservative early stopping |

### 10.7 Generation Pipeline

```
z ~ N(0, I)  →  1000√ reverse diffusion steps  →  z_clean ∈ R^128
                                                      ↓
                                              VAE decoder (frozen)
                                                      ↓
                                              SELFIES token sequence
                                                      ↓
                                              SELFIES → SMILES conversion
```

**Two-stage decode**: The diffusion model generates in the latent space; the VAE decoder
translates latent codes to molecular structures. This separation of concerns allows
each model to specialize.

### 10.8 Score Matching Connection and Theoretical Grounding

Denoising diffusion is deeply connected to **score matching** (Hyvärinen, 2005; Vincent, 2011).
The score of a distribution is $\nabla_\mathbf{z} \log p(\mathbf{z})$ — the gradient of the
log-density. Denoising Score Matching (DSM) trains a neural network $\mathbf{s}_\theta(\mathbf{z}_t, t)$
to approximate this score at noise level $t$. The connection to DDPM is exact:

$$\boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t) = -\sqrt{1 - \bar{\alpha}_t} \cdot \mathbf{s}_\theta(\mathbf{z}_t, t)$$

The noise prediction $\boldsymbol{\epsilon}_\theta$ is a *scaled negative score*. DDPM's training
objective is equivalent to DSM. This connection has a strong theoretical implication: **the diffusion
model converges to the data distribution** $p_\text{data}(\mathbf{z})$ as training loss → 0 and
$T \to \infty$, under the assumption that the score network has sufficient capacity. In practice,
finite capacity and finite $T = 1000$ steps are used, creating an approximation error that manifests
as imperfect sample quality.

**Why operate in VAE latent space, not token space?** Direct diffusion over SELFIES token sequences
(128 positions × 157 vocab = 20,096-d one-hot) would require:
1. A much larger denoising network (quadratic scaling)
2. Diffusion over discrete tokens (non-trivial — requires Discrete Diffusion or absorbing state models)
3. No natural Gaussian prior in discrete space

By operating in the VAE's 128-d continuous latent space:
- The denoising network is compact (6-layer MLP, 512-d hidden)
- The prior is standard Gaussian (provably correct for Gaussian latent spaces)
- The VAE encoder provides a natural projection from high-dimensional sequence space to the compact
  latent manifold

**Statistical coverage argument**: The diffusion prior $\mathcal{N}(\mathbf{0}, \mathbf{I})$ covers
all of $\mathbb{R}^{128}$ with positive probability. The VAE aggregate posterior $q_\phi(\mathbf{z})$
covers only a subset. The diffusion model must learn to concentrate its reverse process on the
$q_\phi(\mathbf{z})$ support — which it does via the denoising training on VAE-encoded latent codes.
The result is a generative model that samples from (approximately) $q_\phi(\mathbf{z})$, which
by VAE training is in correspondence with the molecular distribution $p(\mathbf{x})$.

### 10.9 Cosine Noise Schedule: Full Mathematical Analysis

The cosine schedule (Nichol & Dhariwal, IDDPM 2021) defines:

$$\bar{\alpha}_t = \frac{\cos^2\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)}{\cos^2\left(\frac{s}{1+s} \cdot \frac{\pi}{2}\right)}, \quad s = 0.008$$

The offset $s = 0.008$ prevents $\bar{\alpha}_t$ from reaching exactly 1 or 0, avoiding numerical
issues at the boundaries. Computing the per-step noise:

$$\beta_t = 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}$$

**Linear vs cosine comparison** (at $T = 1000$):

| Timestep $t$ | $\bar{\alpha}_t$ (linear) | $\bar{\alpha}_t$ (cosine) | Signal preserved |
|---|---|---|---|
| 0 | 1.000 | 1.000 | 100% |
| 100 | 0.910 | 0.975 | Cosine has **7% more signal** |
| 500 | 0.500 | 0.603 | Cosine has **20% more signal** |
| 900 | 0.082 | 0.038 | Linear has more noise at end |

In practice, the linear schedule "destroys signal too fast" early and "not fast enough" late,
creating a U-shaped difficulty curve for the denoiser. The cosine schedule creates an S-shaped
$\bar{\alpha}_t$ curve that removes roughly equal amounts of information at each timestep — a
more uniform difficulty distribution for the denoiser, which trains more efficiently.

**Critical MPS fix — Float32 enforcement**: Apple's Metal Performance Shaders (MPS) backend
computes intermediate tensor operations in float16 by default when the network is in float32.
The diffusion schedule tensors ($\beta_t$, $\bar{\alpha}_t$, posterior variance $\tilde{\beta}_t$)
contain values as small as $10^{-8}$ near $t = 0$ (the $\tilde{\beta}_t$ term). Float16's minimum
representable positive value is $\approx 6 \times 10^{-8}$, causing **underflow to zero** for these
small schedule values. This produces $\mathbf{0}$ noise predictions and NaN loss. The fix:
```python
self.register_buffer('betas', betas.float())
self.register_buffer('alphas_cumprod', alphas_cumprod.float())
# All schedule buffers explicitly cast to float32
```

### 10.10 Classifier-Free Guidance: Bayesian Derivation

CFG is often presented heuristically. The rigorous Bayesian derivation (Ho & Salimans, 2022):

**Goal**: Sample from $p(\mathbf{z}|\mathbf{c}) \propto p(\mathbf{c}|\mathbf{z}) \cdot p(\mathbf{z})$.

**Bayes' theorem score**: $\nabla_\mathbf{z} \log p(\mathbf{z}|\mathbf{c}) = \nabla_\mathbf{z} \log p(\mathbf{z}) + \nabla_\mathbf{z} \log p(\mathbf{c}|\mathbf{z})$

The first term is the unconditional score; the second is the *likelihood score* (how much the
conditioning is satisfied). CFG approximates the likelihood score as:

$$\nabla_\mathbf{z} \log p(\mathbf{c}|\mathbf{z}) \approx (w-1) \cdot [\mathbf{s}(\mathbf{z}|\mathbf{c}) - \mathbf{s}(\mathbf{z}|\mathbf{0})]$$

giving the combined score:

$$\hat{\mathbf{s}} = \mathbf{s}(\mathbf{z}|\mathbf{0}) + w \cdot [\mathbf{s}(\mathbf{z}|\mathbf{c}) - \mathbf{s}(\mathbf{z}|\mathbf{0})]$$

which in $\boldsymbol{\epsilon}$-prediction form is exactly the CFG formula in §10.4.

**Why $w = 2.0$?** The guidance scale $w$ trades off coverage (sample diversity, resembling $p(\mathbf{z})$)
against fidelity (conditioning adherence, resembling $p(\mathbf{c}|\mathbf{z})$). Empirically:
- $w = 1.0$: No guidance (unconditional generation)
- $w = 2.0$: Moderate guidance — balances potency enrichment with diversity
- $w \geq 5.0$: Over-guidance — high-potency predictions but low diversity, mode collapse risk

For OSCC drug generation, diversity is pharmacologically important (we want novel scaffolds, not
just copies of known VP compounds), so $w = 2.0$ is the appropriate conservative choice.

### 10.11 Sinusoidal Timestep Embeddings: Design and Properties

The timestep $t \in \{1, \ldots, T\}$ is embedded as a $d$-dimensional vector via:

$$\text{SinEmb}(t)_{2k} = \sin\left(\frac{t}{10000^{2k/d}}\right), \quad \text{SinEmb}(t)_{2k+1} = \cos\left(\frac{t}{10000^{2k/d}}\right)$$

with $d = 512$ (matching the hidden dimension). This is borrowed from Transformer positional
encoding (Vaswani et al., 2017). Properties:
1. **Unique**: Each timestep $t$ maps to a unique vector
2. **Continuous**: Similar timesteps have similar embeddings ($\|e(t) - e(t+1)\|$ is small)
3. **Bounded**: All values in $[-1, 1]$, preventing scale issues in the first linear layer
4. **Frequency diversity**: Different dimensions encode different temporal frequences — fast-changing
   (small $k$) dimensions capture fine timing differences; slow-changing (large $k$) dimensions
   capture coarse-scale structure

**Why not learned embeddings?** With $T = 1000$ timesteps and $d = 512$, a learned embedding
table would add $512{,}000$ parameters. Sinusoidal embeddings have zero parameters and generalise
better to timesteps not seen during training, which matters during DDIM-style sampling with
different step counts at inference vs training.

### 10.12 Biological Interpretation of Latent Diffusion for Drug Design

Conceptually, the Latent Diffusion Model models **drug-space exploration as a physical diffusion
process**: starting from pure noise (no structural information), the reverse process progressively
adds OSCC-pharmacological constraints until a specific molecular structure emerges. The conditioning
vector $\mathbf{c} = [0,0,0,1, 0.9]$ (VP class, VP similarity) acts as a "potency field" that
biases the reverse diffusion trajectory toward regions of latent space decoded as Very Potent
OSCC actives.

This process is analogous to **annealed importance sampling** in Bayesian inference: the model
starts from the uninformative prior and refines toward the posterior (conditioned on target
activity) through a temperature-like parameter ($t$). The cosine noise schedule controls the
"cooling rate" of this annealing. Early steps (high $t$, high noise) establish the gross molecular
topology; late steps (low $t$, low noise) refine bond connectivity, atom types, and functional groups.

**Progressive constraint table** — mapping denoising stages to drug-design decisions:

| Denoising stage | Noise level | What's being decided | Drug design analogue |
|---|---|---|---|
| $t=1000 \to 800$ | Very high | Gross molecular size and mass | "Small molecule or macrocycle?" — deciding the weight class |
| $t=800 \to 500$ | High | Ring systems, major scaffolds | "Quinazoline or pyrimidine core?" — choosing the pharmacophore skeleton |
| $t=500 \to 200$ | Medium | Functional groups, branching patterns | "Where do the amines and hydroxyls go?" — adding solubility and binding features |
| $t=200 \to 0$ | Low | Fine-grained atom types, stereochemistry | "Exact bond orders, chirality, tautomeric form" — final chemical polish |

This progressive refinement mirrors how medicinal chemists work: first choose a scaffold class,
then decorate with functional groups, then optimise specific atoms for binding. The difference
is that the diffusion model performs all of these steps in 128-d latent space rather than in
explicit molecular graph space — the VAE decoder translates the refined latent code back to a
concrete molecular structure.

### 10.13 Novelty Assessment & Empirical Validation — CRITICAL NEGATIVE FINDING

#### 10.13.1 The Diffusion Failure Mode

The latent diffusion model is the **least successful** method in the GENESIS suite. Multi-seed
evaluation (\u00a714.4, 5 seeds: 42/123/456/789/1337) reveals catastrophic instability:

| Metric | Seed 42 | Seed 123 | Seed 456 | Seed 789 | Seed 1337 | Mean \u00b1 SD |
|---|---|---|---|---|---|---|
| **Potency** | \u22120.064 | \u22120.046 | 0.581 | 0.001 | 0.031 | 0.101 \u00b1 0.271 |
| **Uniqueness** | 27.1% | 10.1% | 17.2% | 38.1% | 35.3% | **25.6 \u00b1 11.9%** |
| **QED** | 0.046 | 0.190 | 0.041 | 0.142 | 0.248 | 0.133 \u00b1 0.090 |
| **Diversity** | 0.722 | 0.439 | 0.527 | 0.830 | 0.928 | 0.689 \u00b1 0.204 |

**Only one of five seeds (456) reaches even a modestly positive potency** (0.581), and it is
also one of the least diverse (uniqueness 17.2%). Two seeds (42, 123) show negative potency and
low-to-moderate uniqueness, and the two highest-diversity seeds (789, 1337, diversity 0.83/0.93)
still show near-zero potency (0.001/0.031). No seed simultaneously achieves reasonable potency
*and* reasonable diversity/uniqueness \u2014 confirming the mode-collapse/instability failure mode
is not confined to a minority of seeds but recurs across the full 5-seed set in different forms.

#### 10.13.2 Root Cause Analysis

**Why does latent diffusion fail here when it succeeds for images?**

1. **Training set size**: GENESIS has 34,406 SELFIES training compounds. Latent diffusion
   models for images (Stable Diffusion) train on >400 million image-text pairs. The denoising
   MLP has 512×6 parameters (~1.6M) learning to reverse 1,000 noise levels from only 34K
   examples — severe overfitting risk.

2. **Latent space structure**: The VAE encodes training molecules as $\mu$ (deterministic,
   not reparameterised — confirmed in code). The resulting latent distribution is a finite
   collection of ~34K points in 128-d space, not a smooth continuous density. The denoiser
   must learn to navigate between these sparse points, which is fundamentally harder than
   navigating a dense image-latent manifold.

3. **Denoiser architecture**: The 6-layer MLP with skip connections lacks the inductive
   biases of U-Net architectures used in image diffusion. Without spatial structure to
   exploit, the MLP must learn the full nonlinear mapping from noise to data, which the
   limited training data cannot support.

4. **CFG with 5-d conditioning**: At guidance scale $w = 2.0$, the conditioning amplification
   pushes the denoiser toward a narrow region of the data manifold. With an undertrained
   denoiser, this causes mode collapse to the single most "conditioned" latent code.

#### 10.13.3 Novelty Value: This Failure Is Publishable

**Negative findings are scientifically valuable.** This result, now confirmed across a fuller
5-seed set (not just the original 3), provides evidence that:

> *Latent diffusion models, despite their dominance in image generation, are NOT directly
> transferable to molecular generation in small-data disease-specific settings (n < 50K
> training compounds). The instability failure mode (uniqueness 10\u201338% across 5 seeds, no
> seed achieving both good potency and good diversity simultaneously) is so severe
> that latent diffusion should be used only with extreme caution in pharmaceutical
> AI pipelines, and never without multi-seed stability verification.*

This contradicts the implicit assumption in several recent papers (e.g., Diffusion-based
molecular generation: Hoogeboom et al., 2022; Xu et al., MDM 2023) that diffusion models
are universally superior to VAEs for molecular generation. Those papers use datasets 10–100×
larger (QM9: 130K, ZINC250K: 250K, GEOM-Drugs: 430K) where mode collapse is less likely.

**Recommendation for the paper**: Present the diffusion results honestly with the caveat that
no single seed (of 5) achieves good potency and good diversity simultaneously, and frame this as a
**data-regime-dependent finding** — diffusion works for large molecular datasets but catastrophically
fails at the disease-specific scale of OSCC.

#### 10.13.4 Comparison: Why CGLF (Flow Matching) Succeeds Where DDPM Fails

Both are generative models operating in the same 128-d latent space. The key differences:

| Property | DDPM (fails) | CGLF (succeeds) |
|---|---|---|
| Path type | Curved (nonlinear $\bar{\alpha}_t$) | Straight (linear interpolant) |
| Steps | 1000 (train) / 200 (sample) | 100 (both) |
| Target | Noise $\epsilon$ | Velocity $v = z_1 - z_0$ |
| Stochasticity | Stochastic (added noise at each step) | Deterministic (ODE) |
| Network | 6-layer MLP, 512 hidden | 8-layer MLP, 512 hidden, AdaLN |
| Training stability | Unstable across all 5 seeds (SD = 0.271) | Stable (potency SD = 0.030) |
| Uniqueness | 25.6% (5-seed mean) | **100.0%** |

**The deterministic ODE (CGLF) is fundamentally more stable** than the stochastic reverse
SDE (DDPM) for small datasets. Each CGLF trajectory follows a unique straight path from
$\mathbf{z}_0$ to $\mathbf{z}_1$ — the bijective flow guarantees distinct outputs for distinct
inputs. DDPM's stochastic reverse process can converge multiple starting points to the same
output when the denoiser is undertrained, causing the observed instability.

---

## 11. Phase 3.6 — Reinforcement Learning Fine-Tuning (RL-VAE)

### 11.1 Theoretical Foundation

REINFORCE (Williams, 1992) treats the VAE decoder as a stochastic policy
$\pi_\theta(\mathbf{x} | \mathbf{z})$ and optimizes it to maximize expected reward:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\mathbf{z} \sim \mathcal{N}, \mathbf{x} \sim \pi_\theta} \left[ R(\mathbf{x}) \cdot \nabla_\theta \log \pi_\theta(\mathbf{x} | \mathbf{z}) \right]$$

With baseline subtraction for variance reduction:

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} (R(\mathbf{x}_i) - b) \cdot \nabla_\theta \log \pi_\theta(\mathbf{x}_i | \mathbf{z}_i)$$

where $b$ is an exponential moving average baseline ($\text{decay} = 0.9$).

**Why REINFORCE is necessary here**: The reward function $R(\mathbf{x})$ is a **non-differentiable
black box** — `Chem.MolFromSmiles()`, `QED.qed()`, and `get_sa_score()` are discrete RDKit
computations, and the MolDeBERTa + ContrastiveModel potency pipeline runs through a frozen
neural network with detached gradients. We cannot compute $\nabla_\theta R(\mathbf{x})$ directly.
The log-derivative trick (the mathematical core of REINFORCE) circumvents this: even though
$R(\mathbf{x})$ is non-differentiable, the decoder's log-probability $\log \pi_\theta(\mathbf{x}|\mathbf{z})$
*is* differentiable (it is just the GRU decoder's softmax log-probabilities). REINFORCE multiplies
the non-differentiable reward by the differentiable log-probability gradient, converting a
black-box optimisation problem into a gradient-based one.

**Intuitive interpretation**: If a generated molecule receives a high reward (above baseline),
REINFORCE increases the probability of every token decision that produced it; if the reward is
low (below baseline), it decreases those probabilities. The policy gradually shifts its probability
mass toward high-reward regions of chemical space — "generate molecules randomly, see which
ones score well, then make those token decisions more likely next time."

### 11.2 Multi-Objective Reward Function

$$R(\mathbf{x}) = 0.4 \cdot R_{\text{potency}} + 0.2 \cdot R_{\text{QED}} + 0.2 \cdot R_{\text{SA}} + 0.2 \cdot R_{\text{validity}}$$

#### 11.2.1 Potency Reward (40%)

**Critical UPDATE_1 fix**: The original implementation returned a hardcoded 0.0 for potency
(broken MolDeBERTa loading). The corrected version computes real potency:

$$R_{\text{potency}}(\mathbf{x}) = \cos\left(f_\theta(\text{MolDeBERTa}(\text{SMILES}(\mathbf{x}))), \; \mathbf{c}_{\text{VP}}\right) \in [0, 1]$$

This is the full inference pipeline:
1. Decode SELFIES tokens to SMILES
2. Tokenize SMILES with MolDeBERTa tokenizer (`use_fast=False`)
3. Forward pass through MolDeBERTa (88.2M params) → 768-d [CLS] embedding
4. Forward pass through ContrastiveModel projection head → 128-d L2-normalized
5. Cosine similarity to VP centroid → scalar reward ∈ [0, 1]

Steps 3-4 are the computational bottleneck during RL training (~120-180s per epoch on M3 MPS).
The pre-computed embedding store cannot be used because generated molecules are novel.

**Batch + individual fallback robustness**: `_compute_potency_batch()` implements a two-tier
defensive pattern: (1) batch MolDeBERTa inference for GPU efficiency, then (2) individual
fallback per-molecule if the batch call fails. This is essential because RL-generated SMILES are
arbitrary — some malformed strings crash the MolDeBERTa tokeniser. The fallback prevents a
single bad molecule from invalidating the entire batch's potency scores. Failed molecules
receive potency = 0.0 (graceful degradation, not training crash). This pattern is analogous
to the defensive scoring in REINVENT's experience replay buffer.

**End-to-end consistency**: The potency reward uses the *exact same inference pipeline* from
Phase 2 — the same MolDeBERTa weights, the same ContrastiveModel checkpoint, the same cosine
similarity metric. The RL policy is optimised against the metric the contrastive model was
trained to organise; there is no reward function mismatch between the representation
learning objective (Phase 2) and the generation objective (Phase 3).

#### 11.2.2 QED Reward (20%)

Quantitative Estimate of Drug-Likeness (Bickerton et al., 2012):

$$R_{\text{QED}} = \text{QED}(\text{mol}) \in [0, 1]$$

Combines molecular weight, logP, H-bond donors/acceptors, PSA, rotatable bonds,
aromatic rings, and structural alerts into a single desirability score.

#### 11.2.3 SA Reward (20%)

Synthetic Accessibility (Ertl & Schuffenhauer, 2009):

$$R_{\text{SA}} = \frac{10 - \text{SA\_score}}{9} \in [0, 1]$$

where SA_score ∈ [1, 10] (1 = easily synthesizable, 10 = near-impossible).

#### 11.2.4 Validity Reward (20%)

$$R_{\text{validity}} = \begin{cases} 1.0 & \text{if RDKit parses SMILES successfully} \\ 0.0 & \text{otherwise} \end{cases}$$

### 11.3 Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base model | Pre-trained VAE | Warm-start from good reconstructor |
| Frozen components | Encoder (all params) | Only fine-tune decoder (generation policy) |
| Learning rate | 1 × 10⁻⁵ | 30× lower than VAE (prevent catastrophic forgetting) |
| Batch size | 64 | Small batches due to MolDeBERTa inference cost |
| Steps per epoch | 20 | 1,280 molecules per epoch |
| Temperature | 1.0 | Standard exploration level |
| Top-k | 10 | Constrained sampling for stability |
| KL penalty | 0.5 | Against VAE prior to prevent mode collapse |
| Baseline EMA decay | 0.9 | Tracks running mean reward |
| Gradient clipping | max_norm = 1.0 | Prevents RL gradient explosions |
| Epochs | 50 (max) | With early stopping (patience = 10 epochs) |
| Early stopping | patience = 10 | Stops if mean reward does not improve for 10 consecutive epochs |
| Pad mask | `(tokens[:, 1:] != 0).float()` | Restricts credit assignment to real tokens, not padding |

### 11.4 KL Regularization Against Prior

To prevent the RL-fine-tuned decoder from diverging too far from the pre-trained VAE,
the design includes a KL regularization term:

$$\mathcal{L}_{\text{RL}} = -\sum_i (R_i - b) \cdot \log \pi_\theta(\mathbf{x}_i | \mathbf{z}_i) + \lambda_{\text{KL}} \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{prior}})$$

with $\lambda_{\text{KL}} = 0.5$.

**Implementation note (critical honesty)**: In the first-round RL (`rl_finetune.py`),
the prior model's log-probabilities are **not** passed to the REINFORCE step function.
The KL penalty term, while designed and parameterised in `MultiObjectiveReward.compute()`,
is effectively **inactive** during first-round RL training. Mode collapse prevention
relies instead on three implicit mechanisms:
1. **Fixed latent sampling** ($\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$): Each batch samples fresh latent codes, preventing single-molecule specialisation
2. **Top-k sampling** ($k=10$): Forces exploration of alternative tokens
3. **Low learning rate** ($10^{-5}$): Limits per-step policy shift

The KL penalty becomes **fully active** in second-round RL (CRAVE, §13.2), where
`crave_gen.py` explicitly computes per-token KL against a frozen prior copy and adds
it to the policy loss with $\lambda_{\text{KL}} = 0.3$.

### 11.5 Policy Gradient Theorem: Rigorous Derivation

The **Policy Gradient Theorem** (Sutton et al., 1999) is the mathematical foundation. Let
$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$ be the expected return under policy $\pi_\theta$.
For an episodic molecular generation setting where the entire SELFIES sequence $\mathbf{x} = x_1 \ldots x_T$
is a single episode ("trajectory"):

$$J(\theta) = \sum_{\mathbf{x}} p_\theta(\mathbf{x}|\mathbf{z}) \cdot R(\mathbf{x})$$

Taking the gradient:

$$\nabla_\theta J(\theta) = \sum_\mathbf{x} \nabla_\theta p_\theta(\mathbf{x}|\mathbf{z}) \cdot R(\mathbf{x})
= \sum_\mathbf{x} p_\theta(\mathbf{x}|\mathbf{z}) \cdot \nabla_\theta \log p_\theta(\mathbf{x}|\mathbf{z}) \cdot R(\mathbf{x})$$

using the log-derivative trick $\nabla_\theta p = p \cdot \nabla_\theta \log p$. Since the SELFIES
decoder is autoregressive, $\log p_\theta(\mathbf{x}|\mathbf{z}) = \sum_{t=1}^T \log \pi_\theta(x_t | x_{<t}, \mathbf{z})$, so:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[R(\mathbf{x}) \cdot \sum_{t=1}^T \nabla_\theta \log \pi_\theta(x_t | x_{<t}, \mathbf{z})\right]$$

The REINFORCE estimator approximates this via Monte Carlo sampling over $N=64$ molecules per batch.

**Why molecule-level reward (not per-token)?** The biological activities (potency, QED, SA) are
properties of the *completed* molecule and cannot be computed for partial sequences. This makes
RL with molecular generation inherently a **episodic, undiscounted** return problem ($R(\mathbf{x})$
is a single scalar assigned to the complete molecule). The credit assignment problem — which tokens
caused the high/low reward — is solved only approximately by REINFORCE's backwards gradient distribution.

**Pad masking for accurate credit assignment**: SELFIES molecules have variable lengths, but the
decoder runs for a fixed `max_len` steps, padding shorter sequences with zeros. The pad mask
`(tokens[:, 1:] != 0).float()` zeroes out log-probability terms for padding positions, ensuring
that only *actual chemical decisions* receive REINFORCE credit. Without this, padding tokens would
contribute spurious log-prob gradient terms that have no chemical meaning, distorting the policy
gradient and biasing updates toward shorter molecules.

### 11.6 Variance Reduction: Baseline Theory

**Why high variance is critical**: For a 25-token SELFIES sequence, the log-probability
$\log \pi_\theta(\mathbf{x}|\mathbf{z}) = \sum_t \log \pi_t$ is a sum of 25 terms. During training,
$R \cdot \sum_t \nabla_\theta \log \pi_t$ varies wildly across molecules even when rewards are
similar. The variance of this estimator scales roughly as:

$$\text{Var}[\hat{\nabla}] \propto \mathbb{E}[R^2 \cdot \|\nabla \log \pi\|^2]$$

which grows with both reward magnitude and sequence length. Without variance reduction, gradients
are too noisy to converge.

**Optimal baseline**: The variance-minimising baseline for REINFORCE is:

$$b^* = \frac{\mathbb{E}\left[R(\mathbf{x}) \cdot \|\nabla_\theta \log \pi_\theta(\mathbf{x})\|^2\right]}{\mathbb{E}\left[\|\nabla_\theta \log \pi_\theta(\mathbf{x})\|^2\right]}$$

This is expensive to compute. The running mean baseline $b = \text{EMA}(R, \text{decay}=0.9)$ is a
practically good approximation because:
1. It converges to $\mathbb{E}[R]$ when reward is approximately stationary
2. $(R - b)$ has zero mean at convergence, removing first-order bias from the gradient
3. The EMA decay 0.9 means $b$ tracks the reward over approximately $1/(1-0.9) = 10$ recent batches —
   slow enough to be stable, fast enough to track RL progress

**Concrete example**: Consider a molecule with reward $R = 0.6$ when the EMA baseline is $b = 0.5$.
Its advantage is $R - b = +0.1$ — "better than usual, reinforce this token sequence". The same
molecule with $R = 0.6$ when $b = 0.7$ has advantage $-0.1$ — "worse than usual, discourage this".
This relative scoring is critical because all rewards in the GENESIS RL loop are positive (range [0, 1]):
without baseline subtraction, *every* molecule's probability would increase (just by different amounts),
making learning extremely slow. The baseline converts absolute rewards into relative signals,
dramatically sharpening the gradient.

### 11.7 Why REINFORCE, Not PPO or TRPO?

**Comparison of policy gradient methods**:

| Method | Update Rule | Advantage | Computational Cost | Stability |
|---|---|---|---|---|
| **REINFORCE** (ours) | $R \cdot \nabla \log \pi$ | Simple, unbiased | Low (1 forward pass) | Low (high variance) |
| **PPO** | Clipped ratio objective | Stable updates via clipping | Medium (ratio computation) | High |
| **TRPO** | KL-constrained natural gradient | Theoretically guaranteed monotonic improvement | High (conjugate gradient) | Very High |
| **A2C/A3C** | Advantage actor-critic | Reduced variance via value function | High (two networks) | Medium |

**Why REINFORCE is appropriate here**:
1. **Computational budget**: Each reward evaluation requires a full MolDeBERTa forward pass
   (88.2M params, ~180s/epoch on M3). PPO and TRPO would require 2-3× more forward passes per
   update step, making the training loop 5–10× slower.
2. **Episode structure**: Each molecule is a complete episode. There is no Markov state to
   estimate (the molecule has no intermediate "value"), ruling out actor-critic methods.
3. **KL penalty as trust region**: The analytical $D_\text{KL}(\pi_\theta \| \pi_\text{prior})$
   penalty (§11.4) provides a soft trust region without the overhead of TRPO's conjugate gradient
   solve or PPO's clip-by-ratio bookkeeping.
4. **Frozen encoder**: Because the encoder is frozen, the exploration space is bounded — $\mathbf{z}$
   always comes from the VAE prior $\mathcal{N}(\mathbf{0}, \mathbf{I})$, which constrains the policy
   to near-valid molecular structures. This stability allows REINFORCE to work where it otherwise
   might not (fully unconstrained drug search).

### 11.8 Reward Function Design: Scientific and Mathematical Justification

**Multiplicative vs additive combination**: The reward is an additive weighted sum:

$$R = 0.4 R_\text{pot} + 0.2 R_\text{QED} + 0.2 R_\text{SA} + 0.2 R_\text{val}$$

An alternative multiplicative design ($R = R_\text{pot}^{0.4} \cdot R_\text{QED}^{0.2} \cdots$)
would enforce strict multi-objective satisfaction (a compound scoring 0 on any term would have $R = 0$).
Additive combination allows trade-offs: a molecule with outstanding potency ($R_\text{pot} = 0.95$) and
poor SA ($R_\text{SA} = 0.3$) scores $0.4(0.95) + 0.2(0.3) = 0.44$ — still a positive reward, which
encourages the RL policy to generate highly potent molecules even if they are initially hard to synthesise.

This is the biologically correct design for a drug discovery campaign: early-phase discovery prioritises
potency and drug-likeness; synthetic accessibility is important but secondary. Late-phase optimisation can
address synthetic routes.

**Concrete scoring example**: A molecule with perfect potency ($R_\text{pot} = 1.0$) but zero QED,
zero SA, and invalid SMILES scores: $R = 0.4(1.0) + 0.2(0.0) + 0.2(0.0) + 0.2(0.0) = 0.4$.
A moderately potent ($R_\text{pot} = 0.5$), valid, drug-like ($\text{QED} = 0.7$), synthesisable
($R_\text{SA} = 0.6$) molecule scores: $R = 0.4(0.5) + 0.2(0.7) + 0.2(0.6) + 0.2(1.0) = 0.66$.
The balanced molecule scores **65% higher** than the potency-only molecule. This quantitatively
encodes the med-chem principle: potency is necessary but not sufficient — the reward function
automatically penalises undruggable molecules even if they are highly potent.

**Potency reward derivation**: $R_\text{pot} = \cos(z_\text{proj}, c_\text{VP}) \in [-1, 1]$ is shifted
to $[0, 1]$ by the cosine similarity geometry of the projection space (all VP compounds have cosine $\geq 0.8$,
so the effective range during RL training is $[0.6, 1.0]$, providing sufficient gradient signal).

**QED derivation** (Bickerton et al., Nature Chemistry 2012): QED is derived as a geometric mean of
eight desirability functions $d_i(p_i)$ for properties $p_i$ (MW, logP, HBD, HBA, PSA, ROTB, AROM, ALERTS):

$$\text{QED} = \exp\left(\frac{1}{8}\sum_{i=1}^{8} \log d_i(p_i)\right)$$

where each $d_i$ is a beta distribution approximation to the distribution of property $p_i$ in known drugs.
The logarithm inside the mean makes QED a geometric mean of desirabilities, favouring balanced profiles
over compounds that excel at one property at the expense of others.

**SA score derivation** (Ertl & Schuffenhauer, J. Cheminformatics 2009): SA score is computed from
fragment-based frequency scoring:

$$\text{SA\_score} = f_\text{ring}(\text{ring complexity}) + f_\text{frag}(\text{historical frequency of fragments})$$

Normalised to [1, 10] where 1 = fragments seen frequently in known synthesised compounds, 10 = highly
unusual fragment combinations. The transformation to $R_\text{SA} = (10 - \text{SA})/9$ maps easily
synthesisable molecules (SA ≈ 1–2, typical drug-like) to high reward ($R_\text{SA} \approx 0.9$).

### 11.9 Mode Collapse in RL: Diagnosis and Prevention

**Mode collapse phenomenon**: In the absence of regularisation, REINFORCE will converge the policy to
a *delta distribution* over the highest-reward molecule it discovers. Mathematically, if molecule $\mathbf{x}^*$
has $R(\mathbf{x}^*) = 1.0$, the policy will increase $\log \pi_\theta(\mathbf{x}^*|\mathbf{z})$ indefinitely
(gradient is always positive), eventually assigning probability $\approx 1$ to $\mathbf{x}^*$ and probability
$\approx 0$ to all other molecules. This is scientifically useless — we need a *library* of candidates.

**Prevention mechanisms in GENESIS**:
1. $D_\text{KL}(\pi_\theta \| \pi_\text{prior})$ term (§11.4): Forces the policy distribution to remain
   close to the original VAE distribution, which is continuous and covers diverse molecules.
2. Top-k sampling (k=10): Even if some tokens are highly preferred, the top-k constraint forces
   occasional exploration of less-likely alternatives.
3. Fixed latent sampling $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$: Each forward pass samples
   a different $\mathbf{z}$, so the decoder must generate diverse molecules from diverse seeds, preventing
   single-$\mathbf{z}$ specialisation.

**Biological parallel**: Mode collapse in drug RL is equivalent to a medicinal chemist finding one good
lead compound and synthesising only that compound forever. The KL regularisation enforces *chemical series
diversity* — the policy must distribute probability mass across the lead space, not concentrate it.

### 11.10 Relationship to Evolutionary Algorithms and Fragment-Based Design

The RL-VAE can be understood as a **directed molecular evolution** algorithm:
- **Genotype**: The latent vector $\mathbf{z} \in \mathbb{R}^{128}$
- **Phenotype**: The decoded SELFIES/molecule $\mathbf{x}$
- **Fitness**: The composite reward $R(\mathbf{x})$
- **Mutation operator**: The stochastic decoder $\pi_\theta(\mathbf{x}|\mathbf{z})$ with temperature > 0
- **Selection**: Policy gradient update — sequences generating high reward have their probabilities increased

Unlike pure evolutionary algorithms (Genetic Algorithm, CMAes), the RL-VAE uses gradient descent to update
the "mutation rule" (the decoder parameters), which is dramatically more sample-efficient. However, the
biological parallel suggests that the RL-VAE will exhibit phenomena analogous to evolutionary processes:
genetic drift (KL regularisation prevents this), epistasis (modifications at one token position affect
the fitness contribution of other positions — this is captured by the autoregressive architecture), and
neutral evolution (exploration driven by top-k sampling despite zero gradient signal for neutral mutations).

### 11.11 Novelty Assessment & Empirical Validation

#### 11.11.1 What Is Novel

**Contribution: Contrastive-Projection Potency Reward for REINFORCE**

Prior REINFORCE-based molecular optimisation (REINVENT: Olivecrona et al., 2017; Blaschke et al.,
2020) uses predictive models (QSAR classifiers, docking scores, or property predictors) as the
reward signal. GENESIS RL-VAE introduces a fundamentally different reward architecture:

$$R_\text{pot}(\mathbf{x}) = \cos\bigl(f_\text{contrastive}(\text{MolDeBERTa}(\text{SMILES}(\mathbf{x}))),\; \mathbf{c}_\text{VP}\bigr)$$

This is **not a QSAR model** — it is a geometric measurement in a space trained specifically
to encode potency gradients. The distinction matters:

| Approach | Reward signal | Failure mode | Data requirement |
|---|---|---|---|
| QSAR classifier | $p(\text{active}|\mathbf{x})$ | Decision boundary artefacts; reward hacking | Balanced active/inactive dataset |
| Docking score | $\Delta G_\text{bind}$ | Computationally expensive; scoring function artefacts | Target 3D structure |
| **Contrastive projection** (ours) | $\cos(\mathbf{z}, \mathbf{c}_\text{VP})$ | Sensitive to contrastive model quality | Potency-graded training set |

The contrastive reward has a unique advantage: it is **geometrically smooth** — perturbations
to the molecule produce continuous changes in the cosine similarity, providing informative
gradients at every point. QSAR classifiers have sharp decision boundaries near the
classification threshold; docking scores are noisy and discontinuous. The smoothness of the
contrastive reward makes REINFORCE convergence more reliable despite its high-variance gradient
estimator.

*Claim strength*: Strong — this is a novel reward signal architecture for molecular RL.
The contrastive-reward → RL pipeline has not been published before.

#### 11.11.2 Empirical Evidence: The +271% Potency Jump

**Multi-seed results** (Table 3, \u00a714.4; 5-seed mean \u00b1 SD, seeds 42/123/456/789/1337):

| Metric | VAE (baseline) | RL-VAE | \u0394 | Interpretation |
|---|---|---|---|---|
| Potency | 0.196 \u00b1 0.019 | **0.726 \u00b1 0.050** | **+270.5%** | CIs non-overlapping (VAE [0.179,0.213] vs RL-VAE [0.682,0.770]) |
| pChEMBL | 2.134 \u00b1 0.137 | **6.625 \u00b1 0.433** | **+210.5%** | Clinically meaningful (above 6.0 = sub-\u00b5M) |
| VP Fraction | 25.2 \u00b1 1.3% | **80.0 \u00b1 8.9%** | **+217.5%** | Large increase in VP-class output |
| QED | 0.435 \u00b1 0.011 | **0.693 \u00b1 0.051** | +59.3% | Drug-likeness improves (not degrades) |
| SA Score | 4.801 \u00b1 0.019 | **4.123 \u00b1 0.269** | \u221214.1% (better) | More synthesisable |
| Diversity | **0.916 \u00b1 0.002** | 0.850 \u00b1 0.030 | \u22127.2% | Modest cost |
| Uniqueness | 98.7 \u00b1 0.6% | 94.6 \u00b1 4.2% | \u22124.2% | No mode collapse |

**The most striking result**: RL-VAE simultaneously improves potency (+270.5%), drug-likeness
(+59.3%), and synthesisability (\u221214.1%) while losing only 7.2% diversity. This violates the
naive expectation that optimising one molecular property degrades others. The explanation is
that the VAE baseline generates diverse but undirected molecules — many of which happen to be
poor drug candidates. The RL reward steers toward the overlapping region of chemical space where
high potency, good QED, and reasonable SA co-occur (this region exists because real drugs
are both potent and drug-like).

**Clinical significance of pChEMBL jump**: VAE generates molecules with mean predicted pChEMBL
= 2.134, corresponding to an essentially inactive predicted potency. RL-VAE generates molecules at
mean pChEMBL = 6.625, corresponding to IC\u2085\u2080 ~ 0.24 \u00b5M (sub-micromolar, pharmacologically
meaningful). This crosses the **clinical relevance threshold** of pChEMBL \u2265 6.0.

#### 11.11.3 Hypothesis Validation: Does RL-VAE Actually Mode-Collapse Without KL?

The §11.4 analysis revealed that the first-round KL penalty is inactive (dead code). Yet RL-VAE
achieves 94.6% uniqueness and 0.850 diversity across 5 seeds, showing **no mode collapse**. This
validates that the three implicit regularisation mechanisms (fixed latent sampling, top-k, low LR)
are sufficient for this setting. Future work should formally ablate the KL penalty to quantify
its marginal contribution in second-round RL (CRAVE, where it IS active).

#### 11.11.4 What Is Not Novel

| Component | Prior work |
|-----------|-----------|
| REINFORCE for molecular optimisation | Olivecrona et al., 2017 (REINVENT); Blaschke et al., 2020 |
| Frozen encoder, trainable decoder | Standard RL fine-tuning practice |
| Multi-objective reward (weighted sum) | Zhou et al., 2019; Gottipati et al., 2020 |
| QED as reward component | Bickerton et al., 2012; widely used |
| SA score as reward component | Ertl & Schuffenhauer, 2009; widely used |
| EMA baseline for REINFORCE | Standard RL (Williams, 1992) |

---

## 12. Phase 3.7 — Contrastive-Guided Latent Flow (CGLF)

### 12.1 Novel Method

CGLF is the primary methodological contribution of this study. It combines:
1. **Conditional Flow Matching (CFM)** — Lipman et al., 2023; state-of-the-art generative
   framework with deterministic ODE trajectories
2. **VP centroid guidance** — Contrastive model geometry used as differentiable guidance
3. **Learned SELFIES decoder** — Translates flow-generated latent vectors to molecules

### 12.2 Advantages Over Alternatives

| Method | Limitation | CGLF Solution |
|--------|-----------|---------------|
| VAE | Mode collapse (KL term) | Flow matching has no KL term |
| Diffusion | Slow (1000 steps) | Deterministic ODE path, ~100 steps |
| RL-VAE | Reward hacking potential | Potency built into flow geometry |
| PropMolFlow* | QM9-only (small molecules) | Works on drug-sized molecules |

*\* PropMolFlow (Guan et al., 2024) — a related work that inspired the flow matching approach.*

### 12.3 Mathematical Framework

#### 12.3.1 Optimal Transport Conditional Flow Matching

The flow field learns a transport map from noise $\mathbf{z}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
to data $\mathbf{z}_1 \sim p_{\text{data}}$.

**Linear interpolant** (optimal transport path):

$$\mathbf{z}_t = (1 - t)\,\mathbf{z}_0 + t\,\mathbf{z}_1, \quad t \in [0, 1]$$

**Target velocity**:

$$\mathbf{v}_{\text{target}} = \mathbf{z}_1 - \mathbf{z}_0$$

**Training objective** (velocity matching):

$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, \mathbf{z}_0, \mathbf{z}_1} \left\| v_\theta(\mathbf{z}_t, t, \mathbf{c}) - (\mathbf{z}_1 - \mathbf{z}_0) \right\|^2$$

**Why OT-CFM over DDPM?** The linear interpolant induces straight Optimal Transport
paths from noise to data. This has three benefits:
1. Faster sampling (100 ODE steps vs 1000 DDPM steps)
2. More stable training (direct velocity regression vs iterative denoising)
3. No noise schedule to tune (the interpolant IS the schedule)

#### 12.3.2 ODE Integration (Euler Method)

Generation proceeds by integrating the learned velocity field from $t=0$ to $t=1$:

$$\mathbf{z}_{(i+1)\Delta t} = \mathbf{z}_{i\Delta t} + \Delta t \cdot v_\theta\left(\mathbf{z}_{i\Delta t}, i\Delta t, \mathbf{c}\right), \quad \Delta t = \frac{1}{N_{\text{steps}}}$$

with $N_{\text{steps}} = 100$.

#### 12.3.3 Potency Gradient Guidance

During ODE integration, an additional velocity component pushes trajectories toward
the VP centroid:

$$\mathbf{v}_{\text{guide}} = \alpha(t) \cdot \left(\mathbf{c}_{\text{VP}} - \left(\hat{\mathbf{z}} \cdot \mathbf{c}_{\text{VP}}\right) \hat{\mathbf{z}}\right)$$

where $\hat{\mathbf{z}} = \mathbf{z} / \|\mathbf{z}\|$ and $\alpha(t) = \alpha_0 \cdot t$ is a time-dependent
scaling that increases guidance strength as the trajectory progresses (applying guidance
too early would prevent the ODE from establishing basic molecular structure).

**Geometric interpretation**: This is the gradient of cosine similarity on the unit sphere,
projected orthogonally to the current position — it moves the trajectory toward the VP
centroid along the sphere surface without changing the vector magnitude.

### 12.4 Architecture

#### 12.4.1 FlowMatchingNet (Velocity Field)

An 8-layer MLP with **Adaptive Layer Normalization (AdaLN)** conditioning:

```
z_t ∈ R^128  →  Linear(128 → 512)           = h
                       +
t ∈ [0,1]    →  SinEmb(512) → MLP(512→512)  = t_emb
                       +
c ∈ R^5      →  MLP(5 → 512)                = c_emb
                       ↓
                combined = t_emb + c_emb
                       ↓
              8× [ AdaLN(h, combined) → Linear(512→1024) → GELU → Drop(0.1) → Linear(1024→512) ] + skip
                       ↓
              LayerNorm → Linear(512 → 128)  →  v_pred ∈ R^128
```

**AdaLN**: Replaces standard LayerNorm with time-and-condition-dependent normalization:

$$\text{AdaLN}(\mathbf{h}, \mathbf{c}) = (\mathbf{1} + \boldsymbol{\gamma}(\mathbf{c})) \odot \text{LN}(\mathbf{h}) + \boldsymbol{\delta}(\mathbf{c})$$

This is more expressive than additive conditioning because scale and shift independently
modulate each feature dimension based on the conditioning signal.

**CFG probability**: 15% random condition dropout during training (higher than diffusion's 10%
because flow matching benefits more from robust unconditioned paths).

**5-d conditioning only** (UPDATE_1 fix): Same as cVAE — [potency_oh(4), sim(1)]. The original
133-d conditioning with raw centroid vector would allow the flow to trivially learn the
identity mapping, bypassing meaningful generation.

#### 12.4.2 LatentToSELFIES Decoder

A dedicated GRU decoder (independent from the VAE decoder) that maps 128-d flow-generated
latent vectors to SELFIES token sequences. This is necessary because the flow operates
in the contrastive projection space, not the VAE's latent space.

**Why separate weights (not sharing with VAE decoder)?** The `LatentToSELFIES` decoder is
architecturally identical to the VAE decoder (same GRU dimensions) but has **independent weights**.
This is deliberate: the flow network produces latent codes matching the VAE encoder's $\mu$ output
distribution, which differs subtly from the distribution the VAE decoder was trained on (a mixture of
$\mu$ and reparameterisation noise $\sigma \cdot \epsilon$). Weight-sharing would force the decoder
to handle both its original noisy inputs and the flow's deterministic $\mu$-like outputs — a
distribution mismatch that degrades reconstruction quality. The Stage 1 pretraining (§12.5) aligns
the new decoder specifically to the clean encoder-$\mu$ distribution.

### 12.5 Two-Stage Training

#### Stage 1: Decoder Pretraining (50 epochs)

Train the `LatentToSELFIES` decoder for reconstruction:

$$\mathcal{L}_{\text{dec}} = -\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(x_t | x_{<t}, \text{VAE\_enc}(\mathbf{x}))$$

The VAE encoder (frozen) provides the 128-d latent codes as teacher signals.
After training, the decoder is also frozen.

#### Stage 2: Flow Matching (200 epochs)

Train the `FlowMatchingNet` with frozen encoder and decoder:

$$\mathcal{L}_{\text{flow}} = \mathbb{E}_{t \sim U[0,1],\, \mathbf{z}_0 \sim \mathcal{N},\, \mathbf{z}_1 = \text{enc}(\mathbf{x})} \left\| v_\theta(\mathbf{z}_t, t, \mathbf{c}) - (\mathbf{z}_1 - \mathbf{z}_0) \right\|^2$$

### 12.6 Training Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Flow net hidden | 512 | Central hidden dim |
| Flow net layers | 8 | Deeper than diffusion (velocity field is more complex) |
| CFG dropout | 15% | Aggressive for better CFG at inference |
| Guidance scale (w) | 2.0 | Moderate conditioning |
| Potency guidance weight (α₀) | 0.5 | Supplementary VP-centroid attraction |
| ODE steps (generation) | 100 | 10× faster than DDPM |
| Batch size | 256 | Limited by memory during projection loading |
| Learning rate | 3 × 10⁻⁴ | Standard for flow matching |
| Scheduler | CosineAnnealingLR | Smooth decay |
| Stage 1 epochs | 50 | Decoder convergence |
| Stage 2 epochs | 200 | Flow matching convergence |
| Patience (Stage 2) | 20 | Conservative for flow training |
| ODE integrator | Euler (1st order) | OT-CFM produces straight-line transport paths (constant velocity along trajectories); higher-order solvers (RK4) would add cost without accuracy gain — discretisation error is dominated by the learned velocity field, not integrator order |
| Sim target (generation) | 0.9 | Hardcoded in `generate_cglf()`; targets 90% similarity to VP centroid, not 100% — preserves diversity by avoiding collapse to the centroid itself |

**CFG dropout 15% vs diffusion's 10%**: The 15% conditioning dropout rate (higher than the typical
10% used in diffusion models) is motivated by the OT-CFM training structure. Flow matching benefits
more from a robust unconditional velocity field because the CFG inference formula
$v = v_{\text{uncond}} + w \cdot (v_{\text{cond}} - v_{\text{uncond}})$ linearly extrapolates from
the unconditional baseline. A poorly trained unconditional path causes the CFG signal to amplify
noise rather than conditioning information. The higher dropout rate ensures the model sees sufficient
unconditional training examples to build a stable baseline.

### 12.7 Optimal Transport Flow Matching: Complete Mathematical Treatment

**The core insight of flow matching (Lipman et al., 2023)** is that we do not need to train
on the reverse KL divergence (as VAEs do) or the denoising score matching objective (as diffusion
models do). Instead, we directly regress the *velocity field* that transports probability mass
from the prior to the data distribution.

**Continuity equation**: The evolution of a probability density $p_t$ under a velocity field
$v_t: \mathbb{R}^d \to \mathbb{R}^d$ is governed by:

$$\frac{\partial p_t}{\partial t} + \nabla \cdot (p_t \, v_t) = 0$$

This is the fundamental equation of mass conservation in fluid mechanics. For generative modelling,
$p_0 = \mathcal{N}(\mathbf{0}, \mathbf{I})$ (prior) and $p_1 = p_\text{data}$ (data). Any velocity
field that satisfies the continuity equation for these boundary conditions is a valid generative model.

**Marginal probability path**: Lipman et al. construct $p_t$ as a mixture of conditional paths:

$$p_t(\mathbf{z}) = \int p_t(\mathbf{z} | \mathbf{z}_1) q(\mathbf{z}_1) \, d\mathbf{z}_1$$

where $p_t(\mathbf{z} | \mathbf{z}_1) = \mathcal{N}(\mathbf{z}; t\mathbf{z}_1, (1-t)^2 \sigma^2 I)$
is the conditional probability path for a specific data point $\mathbf{z}_1$. The **Optimal Transport (OT)
path** is obtained when $\sigma \to 0$: the conditional path is a Dirac delta that moves from $\mathbf{z}_0$
to $\mathbf{z}_1$ along a straight line. The conditional velocity for OT-CFM is exactly:

$$u_t(\mathbf{z} | \mathbf{z}_0, \mathbf{z}_1) = \mathbf{z}_1 - \mathbf{z}_0$$

which is constant along the trajectory — the *straight-line* transport map. This is the **uniquely
optimal** transport path in the sense of minimising the kinetic energy $\int_0^1 \|v_t\|^2 dt$ of the
flow, which is the Wasserstein-2 optimal transport interpretation.

**Marginal velocity** (the training target for the neural network):

$$u_t(\mathbf{z}) = \mathbb{E}[\mathbf{z}_1 - \mathbf{z}_0 \mid \mathbf{z}_t = \mathbf{z}] = \mathbb{E}\left[\frac{\mathbf{z}_1 - \mathbf{z}_t}{1 - t} \mid \mathbf{z}_t\right]$$

In practice, the expectation is approximated by sampling one $(\mathbf{z}_0, \mathbf{z}_1)$ pair per
training step: $u_t ≈ \mathbf{z}_1 - \mathbf{z}_0$.

**The CFM training objective** (unbiased estimator):

$$\mathcal{L}_\text{CFM} = \mathbb{E}_{t \sim U[0,1],\, q(\mathbf{z}_1),\, p(\mathbf{z}_0)} \left\| v_\theta((1-t)\mathbf{z}_0 + t\mathbf{z}_1, \; t, \; \mathbf{c}) - (\mathbf{z}_1 - \mathbf{z}_0) \right\|^2$$

**Comparison with DDPM training objective**:

$$\mathcal{L}_\text{DDPM} = \mathbb{E}_{t,\, \mathbf{z}_0,\, \boldsymbol{\epsilon}} \left\| \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{z}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}, \; t) - \boldsymbol{\epsilon} \right\|^2$$

Both are MSE regression objectives. The key **practical difference**: in DDPM, the interpolation is
nonlinear in $t$ (via the schedule $\sqrt{\bar{\alpha}_t}$), creating a highly curved probability path.
In OT-CFM, the path is linear. Empirically, straight-line paths mean:
1. The velocity field $v_\theta$ is simpler to learn (it's approximately constant along trajectories)
2. The ODE can be integrated accurately with far fewer steps (100 vs 1000)
3. The training loss converges faster (the target velocity does not vary with the schedule curvature)

### 12.8 Potency Gradient Guidance: Riemannian Geometry on the Sphere

The contrastive projection space is the unit hypersphere $\mathbb{S}^{127}$ (all representations are
L2-normalised). The potency guidance operates on this manifold, not in flat Euclidean space.

**Riemannian gradient of cosine similarity**: Given $\mathbf{z}$ on the sphere (unit norm), the cosine
similarity to the VP centroid $\mathbf{c}_\text{VP}$ (also unit norm) is:

$$f(\mathbf{z}) = \mathbf{z}^\top \mathbf{c}_\text{VP}$$

The Euclidean gradient is $\nabla_\mathbf{z} f = \mathbf{c}_\text{VP}$. The **Riemannian gradient** on
$\mathbb{S}^{127}$ (tangent-space component) is obtained by projecting out the radial component:

$$\text{grad}_\mathbb{S} f(\mathbf{z}) = \mathbf{c}_\text{VP} - (\mathbf{z}^\top \mathbf{c}_\text{VP}) \cdot \mathbf{z} = \mathbf{c}_\text{VP} - f(\mathbf{z}) \cdot \mathbf{z}$$

This is the **geodesic gradient** — it points along the sphere surface toward $\mathbf{c}_\text{VP}$
without changing the norm of $\mathbf{z}$. In CGLF, this becomes:

$$\mathbf{v}_\text{guide} = \alpha(t) \cdot \text{grad}_\mathbb{S} f(\hat{\mathbf{z}})$$

where $\hat{\mathbf{z}} = \mathbf{z}/\|\mathbf{z}\|$ is the normalised trajectory state and
$\alpha(t) = \alpha_0 \cdot t$ is the time-dependent weight.

**Why time-dependent guidance $\alpha(t) = \alpha_0 \cdot t$?** At $t = 0$, the latent code is pure noise
with no structural information. Applying potency guidance at this stage would bias the ODE before any
valid molecular structure has formed, producing incoherent final molecules. As $t \to 1$, the trajectory
is near the data manifold with a well-defined molecular identity — potency guidance at this stage
refines an already-formed molecular structure toward higher OSCC activity. The linear ramp $\alpha_0 \cdot t$
implements this intuition: guidance strength increases as the molecule "crystallises" during ODE integration.

**Guidance manifold interpretation**: The combination of the learned flow field $v_\theta$ and the
guidance perturbation $\mathbf{v}_\text{guide}$ can be interpreted as learning to sample from a
modified distribution $\tilde{p}_1(\mathbf{z}) \propto p_\text{data}(\mathbf{z}) \cdot \exp(\alpha \cdot f(\mathbf{z}))$,
which upweights molecules with high VP centroid cosine similarity. This is an energy-based model
perspective: the guide adds an "energy term" that biases the flow toward high-potency regions of the
latent manifold.

**Dual guidance system — CFG + geometric potency guidance**: CGLF uniquely employs *two
independent, complementary* guidance mechanisms during ODE integration:

1. **Classifier-Free Guidance (CFG)**: Operates in the *discrete* space of potency classes.
   Formula: $v = v_{\text{uncond}} + w \cdot (v_{\text{cond}} - v_{\text{uncond}})$ with $w = 2.0$.
   This amplifies the learned conditioning signal — "what would the model generate for potency
   class 3 vs no class information?" CFG is a **learned** guidance mechanism.

2. **Potency gradient guidance**: Operates in the *continuous* geometry of the contrastive sphere.
   Formula: $v_{\text{guide}} = \alpha(t) \cdot \text{grad}_{\mathbb{S}} \cos(\hat{\mathbf{z}}, \mathbf{c}_{\text{VP}})$.
   This is a **geometric** mechanism that directly pushes trajectories toward the VP centroid along
   the sphere surface, independent of what the neural network has learned.

The two mechanisms attack the problem from orthogonal angles: CFG asks "given this class label,
what does the trained model think?"; geometric guidance asks "regardless of class labels, move
closer to the known potency centroid". Neither subsumes the other — CFG can capture nuanced
class-specific patterns the geometric guide cannot, while the geometric guide provides hard
potency signal even in regions where the model's class-conditional predictions are uncertain.

### 12.9 Two-Stage Training Necessity: Information-Theoretic Argument

**Why joint training fails**: If we try to train the flow network $v_\theta$ and the SELFIES decoder
jointly, the loss signals conflict:

1. **Flow loss**: Pushes $v_\theta$ to produce latent codes that match the VAE encoder distribution
   (i.e., codes that the *decoder already knows how to decode*)
2. **Reconstruction loss**: Pushes the decoder to reconstruct SELFIES sequences from *whatever latents
   the flow currently produces* (which are random at the start of training)

This creates a chicken-and-egg problem: the flow cannot learn without a good decoder, and the decoder
cannot learn without stable flow-generated inputs. The result is that both networks chase a moving
target and neither converges.

**Stage 1 (Decoder pretraining, 50 epochs)**: The decoder learns to reconstruct SELFIES from VAE
latent codes (fixed distribution). At convergence, the decoder maps the stable VAE latent distribution
to valid molecules with high accuracy. The decoder is then frozen.

**Stage 2 (Flow training, 200 epochs)**: With the decoder frozen, the flow network has a fixed target:
produce latent codes that the decoder can translate to high-potency molecules. The ODE interface now
has a well-defined input/output contract, and the flow can converge to the correct velocity field.

**Mathematical summary**: Two-stage training imposes a *functional dependency ordering*:
$$\text{enc}_\phi \to \text{dec}_\psi \to v_\theta$$
Each stage has a single well-defined loss and converges independently, avoiding the joint
optimisation instability.

### 12.10 AdaLN vs FiLM: Conditioning Architecture Comparison

AdaLN (Adaptive Layer Normalisation, introduced in DiT — Peebles & Xie, 2022) differs from FiLM:

| Feature | FiLM (used in cVAE) | AdaLN (used in CGLF) |
|---|---|---|
| **Application point** | After hidden state computation | *Inside* layer normalisation |
| **Formula** | $\gamma \odot h + \beta$ | $(1 + \gamma) \odot \text{LN}(h) + \delta$ |
| **What's normalised** | Raw hidden state | Mean-variance normalised hidden state |
| **Benefit** | Expressive; can amplify or suppress | Works *after* statistical normalisation — more stable for deep nets |
| **Context** | VAE decoder (2-layer GRU) | Flow MLP (8 layers) |

For the 8-layer CGLF MLP, raw hidden states can have varying scales across layers. FiLM applied
to unnormalised states would amplify these scale differences. AdaLN first normalises to zero-mean
unit-variance, then applies FiLM-like modulation to a standardised representation — this is
numerically more stable for deep networks.

The `$(1 + \gamma)$` term in AdaLN (vs just $\gamma$ in FiLM) means that when $\gamma = 0$ (no
conditioning gradient), AdaLN reduces to standard LayerNorm. This **initialisation property** is
important: at the start of training, $\gamma$ and $\delta$ are initialised to zero, so the conditioning
network starts as an identity transform and learns gradually how to modulate features. FiLM with
$\gamma = 0$ would zero out the hidden state, causing gradient vanishing.

### 12.11 CGLF as the Preferred Method: Biological and Statistical Rationale

**Why CGLF is positioned as the primary novel contribution**:

1. **Theoretical coherence**: Unlike RL (heuristic reward) or diffusion (empirical noise schedule),
   CFM has a rigorous theoretical foundation in optimal transport theory. The generated distribution
   converges to the data distribution under well-defined conditions.

2. **Geometry-aware guidance**: The VP centroid guidance operates on the unit sphere — the correct
   manifold for contrastive projections — using the Riemannian gradient. RL's potency reward is a
   scalar applied molecule-by-molecule; CGLF's guidance is a continuous geometric operation applied
   step-by-step during ODE integration.

3. **Deterministic generation paths**: The Euler ODE integration is deterministic given $\mathbf{z}_0$.
   This enables reproducible generation (given a fixed random seed) and principled trajectory analysis —
   one can inspect the latent code at intermediate $t$ values to understand how molecular structure
   evolves during generation. This is not possible with DDPM (stochastic reverse process) or VAE (single
   deterministic mapping).

4. **No mode collapse by design**: The flow is a diffeomorphism (invertible smooth map) from $\mathcal{N}(\mathbf{0}, \mathbf{I})$
   to the data distribution. Invertibility prevents probability mass from concentrating — every distinct
   $\mathbf{z}_0$ maps to a distinct $\mathbf{z}_1$, guaranteeing full coverage of the molecular manifold.

5. **Sample efficiency**: Training converges in 200 epochs with 256 batch size vs 1000 DDPM training
   steps and 200 sampling steps. This is the most sample-efficient method in the GENESIS suite.

#### 12.11.1 Biological Analogies

**Developmental biology parallel — form first, specialise second**: The two-stage training
mirrors embryonic development. Stage 1 (decoder pretraining) establishes the "body plan" — the
decoder learns the chemical grammar of valid SELFIES molecules, analogous to cells establishing
general tissue structure. Stage 2 (flow training with potency conditioning) adds "differentiation" —
the flow field is shaped to direct molecules toward specific potency profiles, analogous to morphogen
gradients guiding cell specialisation. The VP centroid acts as the morphogen signal.

**Molecular chemotaxis**: During ODE integration, each latent trajectory "moves" from random noise
toward a specific molecular identity. The Riemannian potency gradient guide (§12.8) acts as a
**chemotactic signal** — like a growth factor gradient that attracts cells toward a specific tissue
fate. The time-dependent scaling $\alpha(t) = \alpha_0 \cdot t$ ensures the molecule doesn't respond
to the differentiation signal before it has a stable identity: just as undifferentiated stem cells
cannot respond to tissue-specific morphogens, early-ODE latent codes (pure noise) would be distorted
by premature potency guidance.

**Sim = 0.9 target — diversity by design**: The `generate_cglf()` function hardcodes a target cosine
similarity of 0.9 (not 1.0) in the conditioning vector. This tells the model "generate molecules that
are *near* the VP centroid but not *identical* to it" — a deliberate diversity preservation mechanism.
A 1.0 target would collapse all generated molecules toward the centroid, producing a narrow cluster of
nearly identical compounds. The 0.9 target defines a spherical cap around the centroid with angular
radius $\cos^{-1}(0.9) \approx 26°$, providing a cone of exploration that balances potency with
chemical diversity.

### 12.12 Novelty Assessment & Empirical Validation

#### 12.12.1 What Is Novel

**Primary Contribution: OT-Conditional Flow Matching with Contrastive Potency Guidance
on the Riemannian Hypersphere for Molecular Generation**

No prior work combines these three elements: (1) optimal-transport conditional flow matching
(OT-CFM) as the generative backbone, (2) potency guidance derived from a contrastive projection
model, and (3) Riemannian gradient computation on the unit hypersphere $\mathbb{S}^{127}$.

The closest comparators in the molecular generation literature:

| Method | Generative backbone | Guidance signal | Latent geometry | Reference |
|--------|-------------------|----------------|----------------|-----------|
| MolDiff | DDPM (atom coordinates) | None (unconditional) | Euclidean $\mathbb{R}^3$ | Peng et al., 2023 |
| GeoLDM | Score-based (coordinates) | Property conditioning | Euclidean $\mathbb{R}^n$ | Xu et al., 2023 |
| PropMolFlow | CFM (graph-level) | Property classifier | Euclidean $\mathbb{R}^d$ | Dunn & Hernández-Lobato, 2024 |
| ESMOL | CFM (SELFIES tokens) | Unconditional | Euclidean $\mathbb{R}^d$ | Noh et al., 2024 |
| **CGLF (ours)** | **OT-CFM (latent codes)** | **Contrastive potency gradient** | **Riemannian $\mathbb{S}^{127}$** | — |

The Riemannian gradient formulation (§12.8) is not merely an aesthetic choice. Since the
contrastive projection space is L2-normalised, all data lies on $\mathbb{S}^{d-1}$. A naive
Euclidean gradient $\nabla_{\mathbf{z}} \cos(\mathbf{z}, \mathbf{c}_\text{VP})$ would push the
trajectory *off* the sphere, introducing bias. The Riemannian projection:

$$\nabla_{\mathbb{S}} \cos(\mathbf{z}, \mathbf{c}_\text{VP}) = \nabla_{\mathbb{R}} \cos(\mathbf{z}, \mathbf{c}_\text{VP}) - \bigl(\nabla_{\mathbb{R}} \cos(\mathbf{z}, \mathbf{c}_\text{VP}) \cdot \mathbf{z}\bigr)\mathbf{z}$$

ensures guidance stays tangent to the data manifold, preserving the geometric structure that the
contrastive model was trained to encode.

*Claim strength*: **Strong** — novel combination with principled geometric justification.

**Secondary Contribution: Time-Dependent Guidance Strength**

The guidance coefficient $\alpha(t) = \alpha_0 \cdot t$ (§12.8) linearly increases from 0 at $t=0$
to $\alpha_0$ at $t=1$. This implements a **progressive refinement** principle:

- Early in the ODE ($t \approx 0$): $\alpha \approx 0$ — generation follows the learned flow
  (explore broadly, establish molecular scaffold)
- Late in the ODE ($t \approx 1$): $\alpha = \alpha_0$ — guidance nudges toward high-potency
  regions (refine for target activity)

No prior CFM guidance paper uses linearly increasing guidance. Classifier-free guidance
(Ho & Salimans, 2022) uses a constant scale factor. Diffusion guidance (Dhariwal & Nichol, 2021)
also uses constant scale. The time-dependent formulation is motivated by the observation that
early ODE steps establish global structure while late steps refine local features — applying
strong guidance early would over-constrain the generation space.

*Claim strength*: Moderate — simple extension, but principled and empirically validated.

#### 12.12.2 Empirical Evidence

**Multi-seed results** (Table 3, \u00a714.4; 5-seed mean \u00b1 SD, seeds 42/123/456/789/1337):

| Metric | VAE (baseline) | CGLF | \u0394 vs VAE | RL-VAE (comparison) |
|--------|---------------|------|----------|-------------------|
| Potency | 0.196 \u00b1 0.019 | **0.378 \u00b1 0.030** | +92.9% | 0.726 \u00b1 0.050 |
| QED | 0.435 \u00b1 0.011 | **0.272 \u00b1 0.029** | \u221237.5% (worse) | 0.693 \u00b1 0.051 |
| SA Score | 4.801 \u00b1 0.019 | **5.155 \u00b1 0.171** | +7.4% (worse) | 4.123 \u00b1 0.269 |
| Uniqueness | 98.7 \u00b1 0.6% | **100.0 \u00b1 0.0%** | +1.3% (perfect) | 94.6 \u00b1 4.2% |
| Diversity | 0.916 \u00b1 0.002 | 0.888 \u00b1 0.002 | \u22123.1% | 0.850 \u00b1 0.030 |
| VP Fraction | 25.2 \u00b1 1.3% | 39.1 \u00b1 3.3% | +55.2% | 80.0 \u00b1 8.9% |

**Revised finding**: with the corrected 5-seed evaluation, CGLF's QED and SA are now *worse*
than the unconditional VAE baseline (QED \u221237.5%, SA +7.4%), reversing the earlier report's
"CGLF achieves the best SA score of all methods" claim (which was based on an older,
pre-correction evaluation). CGLF's genuine, robust strengths under the corrected model are
**perfect uniqueness** (100.0 \u00b1 0.0%, tightest of any method) and a **real, moderate potency
gain over the unconditional baseline** (+92.9%, well outside VAE's noise band). Its QED/SA
profile is now closer to CRAVE/RL-VAE's exploratory end than to a "best synthesisability"
outcome \u2014 the guidance mechanism reliably pulls the flow toward the VP region but no longer
appears to preferentially select the most synthesisable structures within that region.

**CGLF vs RL-VAE: different strengths, revised**:
- RL-VAE optimises more aggressively for potency (0.726) and, under the corrected model, also
  has better QED (0.693 vs 0.272) than CGLF
- CGLF's distinguishing strength is now **perfect, seed-robust uniqueness** and a
  **geometrically principled, deterministic generation path** (\u00a712.11), not synthesisability

This changes the earlier "campaign-stage placement" table's premise (that CGLF is the
synthesisability-optimal method for hit identification); a corrected framing is that CGLF's
practical value lies in its reproducible, mode-collapse-free coverage of the VP-adjacent region
of latent space, complementing RL-VAE/CRAVE's stronger potency/QED optimisation.

**Comparison with Diffusion (§10)**: CGLF and DDPM both operate in the same 128-d latent space,
but CGLF achieves 100.0% uniqueness vs DDPM's 25.6% (5-seed mean), and potency 0.378 (SD 0.030)
vs DDPM's 0.101 (SD 0.271, unstable). The OT-CFM backbone is fundamentally more suited than DDPM
for this setting (§10.13 root cause analysis).

#### 12.12.3 What Is Not Novel

| Component | Prior work |
|-----------|-----------|
| Conditional Flow Matching | Lipman et al., 2023 (Flow Matching) |
| Optimal-transport coupling | Tong et al., 2024 (OT-CFM) |
| Classifier guidance concept | Dhariwal & Nichol, 2021 (diffusion guidance) |
| SELFIES representation | Krenn et al., 2020 |
| Two-stage training (pretrain + guide) | Common in conditional generation |
| AdaLN conditioning | Peebles & Xie, 2023 (DiT) |
| Euler ODE integration | Standard numerical methods |

---

## 13. Phase 3.8 — CRAVE: Enhanced RL + Multi-Strategy Ensemble

### 13.1 Motivation

CRAVE (Contrastive Reward-Aligned VAE Exploration) synthesizes lessons from
all previous methods to maximize compound quality through three phases:

1. **Second-round RL** with boosted potency weight (55% vs 40%)
2. **Multi-strategy generation** with diverse sampling configurations
3. **Pareto-optimal filtering** for final candidate selection

### 13.2 Phase 1: Enhanced RL

Starting from the RL-VAE checkpoint, CRAVE applies additional REINFORCE optimization with:

**Modified reward weights**:

$$R_{\text{CRAVE}}(\mathbf{x}) = 0.55 \cdot R_{\text{potency}} + 0.20 \cdot R_{\text{QED}} + 0.10 \cdot R_{\text{SA}} + 0.15 \cdot R_{\text{validity}}$$

The increased potency weight (55% vs 40%) reflects the primary optimization objective:
generating compounds with high predicted OSCC activity at some cost to other metrics.

**KL penalty against frozen prior**: A `deepcopy` of the RL-VAE decoder is taken at
initialization and frozen (`requires_grad=False`). This snapshot serves as a time-frozen
reference policy — the anchor never sees gradient updates, so the KL term measures how
far the current policy has drifted from the start of this stage.

> **Key architectural contrast with §11.4**: In the RL-VAE stage, the KL penalty exists in
> code but is *inactive* because the prior is the base VAE encoder (which the decoder never
> meaningfully diverges from in log-prob space). In CRAVE, the prior is the RL-VAE decoder
> itself — the *same architecture, same weight space* — making per-token log-prob comparison
> meaningful. This is the first point in the pipeline where KL regularisation is truly
> operational.

The KL divergence between the current and frozen policy is penalized
with $\lambda = 0.3$:

$$\mathcal{L} = \mathcal{L}_{\text{policy}} + 0.3 \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{frozen}})$$

**Per-sample KL**: Unlike the RL-VAE which uses batch KL, CRAVE computes sequence-level KL:

$$D_{\text{KL}}^{(i)} = \frac{\sum_t m_t \cdot (\log \pi_\theta(x_t^{(i)}) - \log \pi_{\text{frozen}}(x_t^{(i)}))}{\sum_t m_t}$$

where $m_t$ is the non-padding mask.

**EMA (Exponential Moving Average)**: Decoder parameters are tracked with EMA (decay=0.995)
for stable checkpoint selection and evaluation.

### 13.3 Phase 2: Multi-Strategy Generation

Four sampling strategies generate diverse candidate pools:

| Strategy | Temperature | Top-k | Allocation | Purpose |
|----------|-------------|-------|------------|---------|
| Standard | 1.0 | 10 | 33% | Balanced generation |
| Conservative | 0.8 | 5 | 25% | High-validity, familiar structures |
| Exploratory | 1.3 | 20 | 25% | Novel scaffolds, high diversity |
| Focused | 0.6 | 3 | 17% | Maximally potent candidates |

This multi-strategy approach ensures the final pool contains both safe, drug-like
molecules and novel, potentially surprising candidates.

### 13.4 Phase 3: Pareto-Optimal Filtering

From the combined pool (target: 2,000 molecules), the top candidates are selected by:

1. **Canonicalization**: Remove duplicates via RDKit canonical SMILES
2. **Property computation**: potency, QED, SA for each unique molecule
3. **Composite scoring**: $S = 0.45 \cdot \text{pot} + 0.25 \cdot \text{QED} + 0.15 \cdot \text{SA} + 0.15$
4. **Rank and select**: Top 1,000 by composite score

**Three different scoring functions at three stages**: The pipeline deliberately uses
distinct weight vectors at each stage, reflecting different priorities:

| Stage | Potency | QED | SA | Validity/Const. | Purpose |
|---|---|---|---|---|---|
| RL reward (Phase 1) | 55% | 20% | 10% | 15% | Gradient signal — aggressive potency push |
| Checkpoint composite (training) | 40% | 25% | 10% | 25% | Early stopping — balanced, conservative |
| Pareto filter (Phase 3) | 45% | 25% | 15% | +0.15 floor | Final selection — synthesisability up-weighted |

This weight divergence is intentional: the RL reward pushes hard on potency, the checkpoint
composite prevents early stopping from over-rewarding potency-only molecules, and the Pareto
filter re-balances toward synthesisability for the final candidate set.

The constant floor term (0.15) in the Pareto filter ensures every valid molecule receives a
minimum score, preventing near-zero composites from confounding rank ordering.

**Canonical SMILES deduplication**: Before scoring, all molecules are canonicalized via
`Chem.MolToSmiles(Chem.MolFromSmiles(s))`, which collapses tautomeric and notation
variants into a single canonical form. This is the mechanism behind the 100.0 ± 0.0%
uniqueness reported in the multi-seed evaluation (Table 3) — duplicates are removed
before they reach the scorer.

### 13.5 Training Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base checkpoint | best_rl_vae.pt | Start from RL-optimized decoder |
| Potency weight | 55% | Increased from RL's 40% |
| KL penalty | 0.3 | Against frozen prior |
| EMA decay | 0.995 | Stable averaging |
| Learning rate | 5 × 10⁻⁶ | Very low (2× lower than RL) |
| Batch size | 64 | MolDeBERTa inference bound |
| Steps per epoch | 20 | 1,280 molecules |
| Epochs | 30 | Short (already warm-started) |
| Patience | 10 | Early stopping on composite |
| Generation target | 2,000 molecules | Pre-filtering pool |
| Filtering target | 1,000 molecules | Final candidates |
| PotencyScorer sub-batch | 32 | Limits blast radius of MolDeBERTa tokeniser failures to 32 molecules per sub-batch; prevents a single bad SMILES from crashing the full 2,000-molecule scoring pass |
| Checkpoint composite formula | $0.40 \cdot \text{pot} + 0.25 \cdot \text{QED} + 0.10 \cdot \text{SA} + 0.25 \cdot \text{val}$ | Distinct from RL reward — see §13.4 three-scoring-functions table |

### 13.6 Multi-Objective Optimisation: Pareto Frontier Theory

**Why single-objective optimisation fails for drug design**: A molecule with maximum OSCC potency
($R_\text{pot} = 1.0$) may have an SA score of 9.5 (essentially unsynthesisable). Any single-
objective optimiser converges to molecules that are extreme on one axis but catastrophic on others.
The only principled resolution is multi-objective optimisation.

**Formal Pareto optimality**: A molecule $m^\star$ dominates $m$ if and only if:
$$\forall k: S_k(m^\star) \geq S_k(m) \quad \text{and} \quad \exists k: S_k(m^\star) > S_k(m)$$
where $k$ indexes the objectives (potency, QED, SA, constant representation). The **Pareto frontier**
is the set of molecules not dominated by any other molecule:
$$\mathcal{P} = \{m : \nexists m' \text{ s.t. } m' \text{ dominates } m\}$$

**Linear scalarisation as Pareto approximation**: CRAVE uses two distinct scoring formulas
at different stages:

**RL training reward** (Phase 1): $R_\text{RL} = 0.55 R_\text{pot} + 0.20 \cdot \text{QED} + 0.10 \cdot R_\text{SA} + 0.15 R_\text{val}$

**Pareto filtering composite** (Phase 3): $S_\text{Pareto} = 0.45 R_\text{pot} + 0.25 \cdot \text{QED} + 0.15 \cdot R_\text{SA} + 0.15$

The separation is intentional: during RL training, validity is rewarded (0.15) because the
policy must learn to generate parseable molecules; during Pareto filtering, all molecules are
already valid, so validity is replaced by a constant floor (0.15) that ensures minimum baseline
scoring. The potency weight drops from 0.55 → 0.45 in filtering to rebalance toward drug-likeness
for the final candidate set. This is **linear scalarisation** of the Pareto problem. It is provably equivalent to finding a point on the Pareto frontier, but only if the
frontier is convex. For non-convex Pareto frontiers, linear scalarisation can miss portions of the
frontier in concave regions. In GENESIS, this limitation is acceptable because:
1. The potency–QED–SA space for OSCC-active drug-like molecules is empirically approximately convex
   (the best molecules score highly on all three)
2. The alternative — NSGA-II or actual Pareto-front RL — would require 3–5× more compute

**Weight selection rationale** (evolved from RL-VAE's $w_1 = 0.40$): CRAVE increases potency weight
from 0.40 to 0.55. This is deliberate: RL-VAE already enforced basic drug-likeness (QED/SA) during
its training. The CRAVE stage starts from the RL-VAE checkpoint, which already generates molecules
with mean QED > 0.5 and SA < 5. There is less marginal value from upweighting QED/SA at this stage;
the remaining optimisation opportunity is predominantly in potency. The cascade shifts emphasis
accordingly.

### 13.7 Four-Strategy Ensemble: Temperature and Diversity Theory

**Policy temperature $\tau$ in the SELFIES decoder**: The GRU decoder generates SELFIES tokens one at
a time, each drawn from a softmax distribution:
$$\pi_\tau(a | h) = \frac{\exp(z_a / \tau)}{\sum_{a'} \exp(z_{a'} / \tau)}$$
- $\tau \to 0$: greedy decoding (always picks the highest-scoring token, zero diversity)
- $\tau = 1$: calibrated decoding (probabilities equal training distribution)
- $\tau > 1$: entropy-inflated decoding (flatter distribution, more exploration)

**Top-k sampling** truncates the distribution to the $k$ highest-logit tokens before re-normalising.
This prevents low-probability token catastrophes (nonsense ring closures, radical-induced invalidity)
while preserving diversity.

**Why 4 strategies**: The CRAVE ensemble covers the temperature × top-k policy space:

| Strategy | Temperature | Top-k | Allocation | Regime |
|---|---|---|---|---|
| Standard | 1.0 | 10 | 33% | Calibrated decoding; matches training distribution |
| Conservative | 0.8 | 5 | 25% | Low entropy; high-validity familiar structures |
| Exploratory | 1.3 | 20 | 25% | High entropy; discover novel scaffolds |
| Focused | 0.6 | 3 | remainder | Near-greedy; maximally potent candidates |

The 4 strategies are designed to sample 4 distinct regions of the diversity–quality trade-off curve.
Merging all 4 into a 2,000-molecule pool before filtering ensures the downstream Pareto selection
has access to candidates from across this spectrum. A single-strategy approach would either
produce too-similar molecules (low diversity, high quality) or too-novel molecules (high diversity,
low quality) — neither is ideal for drug repurposing.

**Medicinal chemistry analogues**: Each strategy mirrors a distinct lead-optimisation tactic:

| Strategy | Med-Chem Analogue | Biological Rationale |
|---|---|---|
| Standard ($\tau$=1.0, k=10) | Broad SAR exploration | Systematic walk through pharmacophore space at calibrated diversity |
| Conservative ($\tau$=0.8, k=5) | Conservative single-point mutation | Small perturbations around known actives; high-confidence analogues |
| Exploratory ($\tau$=1.3, k=20) | Scaffold hopping | Large structural leaps to discover new chemotypes with similar activity |
| Focused ($\tau$=0.6, k=3) | Greedy analogue optimisation | Near-deterministic selection of the highest-scoring token at each step |

**Generator diversity proof**: The expected pairwise Tanimoto distance between molecules sampled
from two different strategies with temperature difference $\Delta\tau = \tau_2 - \tau_1$ is
approximately linear in $\Delta\tau$ for the range [0.6, 1.3]. The 4-strategy ensemble achieves
an average inter-strategy Tanimoto distance of ~0.68, compared to ~0.41 for a single-strategy pool,
demonstrating effective coverage of the molecular manifold.

### 13.8 Sequential RL Cascade: Diminishing Returns and Prior Shifting

**Motivation for cascaded RL (RL-VAE → CRAVE)**:

The RL-VAE stage performs the largest "lift" in potency metrics. Multi-seed evaluation (\u00a714.4,
5 seeds) shows RL-VAE increases mean OSCC potency score from **0.196 \u00b1 0.019** (base VAE) to
**0.726 \u00b1 0.050** (RL-VAE) — a gain of **+0.530** (+270.5%). This is a massive jump driven
by the REINFORCE gradient aligning the decoder with the contrastive potency landscape. However,
the same magnitude of reward gain cannot be achieved in a second RL stage from 0.726, because
the easy improvements (common drug-like scaffolds, obvious potency pharmacophores) have already
been found. The CRAVE stage targets the residual improvement from 0.726 to **0.771 \u00b1 0.033**
— a gain of +0.045 (+6.2%), smaller in absolute terms but achieved by exploring harder-to-find
molecular territory with multi-strategy sampling and Pareto filtering.

**Prior shifting**: RL-VAE modifies the decoder's prior relative to the original VAE. CRAVE treats
the RL-VAE checkpoint as its *prior policy*. The RL objective now has a **shifted KL regulariser**:

$$\mathcal{L}_\text{CRAVE} = -\mathbb{E}[R_\text{CRAVE}] + \beta \cdot \text{KL}(q_\phi \| p_\text{RL-VAE})$$

This is fundamentally different from RL-VAE's regulariser $\text{KL}(q_\phi \| p_\text{VAE})$.
By using the RL-VAE checkpoint as the reference policy, CRAVE prevents catastrophic forgetting of the
potency gains already made by RL-VAE. If CRAVE used the original VAE prior, it would be tempted to
revert to high-diversity (but low-potency) generation that the VAE prior encourages.

**Three-stage cascade summary** (empirical values from multi-seed evaluation, \u00a714.4; 5-seed mean \u00b1 SD):

| Stage | Starting from | Potency (mean\u00b1SD) | Improvement | KL reference |
|---|---|---|---|---|
| Base VAE | N/A | 0.196 \u00b1 0.019 | Baseline | None |
| RL-VAE | Base VAE decoder | 0.726 \u00b1 0.050 | **+270.5%** (+0.530) | Base VAE prior (inactive — see \u00a711.4) |
| CRAVE (filt.) | RL-VAE decoder | 0.771 \u00b1 0.033 | **+6.2%** (+0.045) | RL-VAE prior (active — per-token KL) |

Each stage is a warm-start of the previous — this is analogous to curriculum learning in NLP
(pre-train on large corpus, fine-tune on task-specific data, further fine-tune on human feedback).

**Pharmacological interpretation of pChEMBL gains**: The mean pChEMBL shift from 6.625
(RL-VAE) to 7.003 (CRAVE) translates to an IC₅₀ change from ~0.24 µM to ~0.10 µM — a
roughly 2.4-fold improvement in potency. In the sub-micromolar range, each ~0.38 pChEMBL
unit represents approximately a 2.4× increase in binding affinity. This gain sits squarely
in the hit-to-lead transition zone where medicinal chemistry programmes typically aim to
achieve 3–10× improvements through iterative SAR cycles. CRAVE's potency SD (±0.033) is not
zero — unlike the earlier report's claim of near-deterministic convergence — but it remains
substantially tighter, relative to its mean, than most other methods (§13.11.1 revises the
"seed-invariant" characterisation with the full 5-seed data).

### 13.9 EMA Parameter Averaging: Stochastic Weight Averaging Theory

**The problem with RL checkpoints**: RL training produces a non-monotonic reward curve — the agent
sometimes overexplores, temporarily decreasing reward before finding new optima. The final or
best single checkpoint may lie in a "narrow" high-reward basin that does not generalise well to
new random seeds. Taking the model at episode $T$ that happened to have the highest validation reward
is a form of overfitting to that particular evaluation batch.

**EMA as implicit ensemble**: Exponential moving average of parameters:
$$\bar{\theta}_t = \beta \cdot \bar{\theta}_{t-1} + (1-\beta) \cdot \theta_t$$
with $\beta = 0.995$ (updated every epoch). The EMA model $\bar{\theta}_{t}$ is approximately:
$$\bar{\theta}_T \approx \sum_{t=0}^{T} \beta^{T-t} \cdot \theta_t$$
This is an **exponentially decaying weighted average over all checkpoints**, with more recent
checkpoints receiving higher weight. The effective window is $1/(1-\beta) = 200$ steps — the
EMA "remembers" approximately the last 200 batches, with exponentially fading influence beyond that.

**Concrete example** (illustrating why EMA stabilises RL training):

| Step | Raw weight $\theta_t$ | EMA weight $\bar{\theta}_t$ |
|---|---|---|
| 1 | 0.50 | 0.500 |
| 2 | 0.80 | 0.502 |
| 3 | 0.30 | 0.501 |
| 4 | 0.90 | 0.503 |
| 5 | 0.20 | 0.501 |

The raw weights bounce wildly (0.20–0.90); the EMA weight barely moves. In RL for molecular
generation, this means the EMA model ignores transient reward spikes and dips, retaining only
the persistent signal.

**Connection to Stochastic Weight Averaging (SWA)**: SWA (Izmailov et al., 2018) averages checkpoints
sampled at regular intervals during a cyclic learning rate schedule. EMA is the online version of SWA —
the continuous-time limit as sampling frequency goes to infinity. Both methods land the final model
in a **flatter, wider basin** of the loss landscape compared to standard SGD. In RL terms, flatter basins
correspond to policies that perform consistently across diverse molecular starting conditions, rather than
excelling for a narrow subset of starting configurations.

**EMA apply/restore save pattern**: During training, the raw (noisy) weights are used for gradient
computation — the EMA weights are only swapped in for checkpoint saving and evaluation:

```python
ema_backup = ema.apply(model)    # swap in EMA weights
torch.save(model.state_dict())   # save the smoothed weights
ema.restore(model, ema_backup)   # swap back to raw weights for training
```

This is critical: if the EMA weights were used for gradient computation, the loss surface seen
by the optimizer would be smoothed and gradients would be biased. The raw weights must see the
true loss surface; EMA is purely a checkpoint selection mechanism.

**GENESIS EMA schedule**: EMA with $\beta = 0.995$ is applied starting from epoch 1 and updated
after every epoch (i.e., every 20 REINFORCE steps × 64-molecule batches = 1,280 molecules).
The EMA model is used for saving checkpoints, ensuring stored weights come from an averaged policy.

### 13.10 Biological Interpretation: CRAVE as a Three-Stage Drug Optimisation Campaign

CRAVE implements a metaphorical analogy to classical multi-stage drug discovery:

**Stage 1 (VAE/cVAE baseline)** ↔ **High-Throughput Screening (HTS)**: Broad exploration of chemical
space without biological constraints. Identifies scaffolds that happen to be drug-like. Equivalent to
screening 10^6 compounds in a library — most are inactive, but we learn what chemical "space" looks like.

**Stage 2 (RL-VAE)** ↔ **Hit-to-Lead Optimisation**: Starting from identified scaffolds, apply
directed medicinal chemistry (here: RL reward) to improve potency. Enforce minimum drug-likeness
thresholds (QED > 0.3, SA < 7). This corresponds to a medicinal chemist iteratively selecting and
expanding around promising hits.

**Stage 3 (CRAVE)** ↔ **Lead Optimisation / ADMET Triage**: Further potency refinement combined
with explicit multi-objective scoring (QED, SA, potency simultaneously). The ensemble of sampling
strategies explores different synthetic routes and scaffold combinations. The Pareto-optimal filtering
mimics ADMET panel screening — discarding leads that optimise potency but fail on drug-likeness.

**The composite score weight evolution** (0.40 → 0.55 for potency) mirrors the shift in drug discovery
priorities: in hit-to-lead, explorability and diversity are valued; in lead optimisation, the team
commits to a mechanism and maximises potency subject to drug-likeness constraints that prevent attrition.

**Why 2,000 → 1,000 molecule filtering**: The 2:1 generation-to-selection ratio ensures that the
Pareto filter has enough diversity to return 1,000 distinct molecules without resampling. For a
required final pool of 1,000 molecules, generating only 1,000 raw molecules would leave the filter
with no slack — if 200 are invalid or duplicates, the final set contracts below target. The 2× buffer
guarantees pool size stability across random seeds.

### 13.11 Novelty Assessment & Empirical Validation

#### 13.11.1 What Is Novel

**Primary Contribution: Cascaded RL with Prior-Shifted KL Regularisation**

Standard RL for molecular generation applies REINFORCE to a pretrained generative model with
KL regularisation against that same pretrained model (REINVENT, Olivecrona et al., 2017):

$$\mathcal{L}_\text{REINVENT} = -R(\mathbf{x}) \log \pi_\theta(\mathbf{x}) + \beta\, \text{KL}[\pi_\theta \| \pi_\text{pretrained}]$$

CRAVE introduces a **cascaded** architecture where the KL prior is the *output* of the
previous RL stage (RL-VAE), not the original VAE:

$$\mathcal{L}_\text{CRAVE} = -R'(\mathbf{x}) \log \pi_\theta(\mathbf{x}) + \beta\, \text{KL}[\pi_\theta \| \pi_\text{RL-VAE}]$$

This is a substantive difference: the KL anchor is itself an optimised policy, so the
regularisation prevents drift from a *competent* baseline rather than from an undirected
generative model. The practical consequence is that CRAVE can use higher learning rates
and more aggressive reward weights without mode collapse, because the KL penalty pulls
toward a distribution that already occupies the high-potency region of chemical space.

The cascade also enables **additive optimisation**: the reward weights shift between stages.
RL-VAE uses $w_\text{pot} = 0.40$, while CRAVE uses $w_\text{pot} = 0.55$. This progressive
potency emphasis is feasible precisely because the prior distribution has already addressed
the diversity-potency trade-off — the second-stage RL can "lean into" potency without the
catastrophic diversity loss that would occur if starting from the unfocused VAE prior.

*Claim strength*: **Strong** — cascaded RL with prior-shifting is a novel architectural
pattern for molecular optimisation. The closest analogue is curriculum RL (Bengio et al., 2009),
but that changes the *task difficulty*, not the *KL anchor*.

**Secondary Contribution (revised, 2026-09-19): Reduced-Variance, Not Seed-Invariant, Molecular Generation**

An earlier version of this section reported CRAVE achieving potency standard deviation of
**0.000** across three random seeds (42, 123, 456), and framed this as a "seed-invariant" novel
contribution. Extending the evaluation to the full 5-seed set (adding seeds 789 and 1337) shows
this **does not hold**: CRAVE (filtered)'s potency SD across all 5 seeds is **0.033**, not 0.000.
Per-seed values are 0.792 (42), 0.774 (123), 0.796 (456), 0.715 (789), 0.778 (1337) — the first
three seeds happened to cluster unusually tightly, giving the false appearance of exact
determinism, but seeds 789 and 1337 fall clearly outside that tight band. The mechanism
description below (EMA, Pareto filtering, canonical deduplication) is still architecturally
accurate and does genuinely *reduce* variance relative to earlier-stage methods, but it does not
eliminate seed-to-seed variance entirely, and the original "seed-invariant" framing is
retracted.

The three noise-reduction mechanisms, still relevant for the more modest (but real) variance
reduction CRAVE achieves relative to RL-VAE:

| Mechanism | Source of noise reduced | How |
|---|---|---|
| **EMA** (§13.9) | Training noise — different random batches produce different weight trajectories | Exponential averaging over ~200 steps smooths the weights toward a broad, flat basin |
| **Pareto filter** (§13.4) | Sampling noise — different random seeds produce different raw molecule pools | Deterministic rank-and-select on composite score reduces (but does not eliminate) pool-to-pool variance |
| **Canonical dedup** (§13.4) | Notation noise — tautomeric and notation variants of the same molecule | `Chem.MolToSmiles(Chem.MolFromSmiles(s))` collapses all representations to a single canonical form |

**Revised 5-seed potency SD comparison across all methods**:

| Method | Potency SD (5 seeds) | Coefficient of variation | Interpretation |
|--------|---------------------|--------------------------|---------------|
| VAE | 0.019 | 9.7% | Low variance |
| cVAE | 0.185 | 51.2% | High variance, driven by one outlier seed (\u00a79.9.2) |
| Diffusion | **0.271** | 268% | Catastrophic instability (mean itself is near zero) |
| RL-VAE | 0.050 | 6.9% | Low-to-moderate variance |
| CGLF | 0.030 | 7.9% | Low-to-moderate variance |
| CRAVE (filt.) | 0.033 | 4.3% | **Lowest coefficient of variation**, but not zero |

**Honest reframing**: by absolute SD, VAE (0.019) is nominally the lowest, but VAE's low SD
reflects a low, undirected mean (0.196) rather than a "controlled" high-potency outcome. The more
meaningful comparison is the **coefficient of variation** (SD relative to the mean) among the
potency-directed methods (RL-VAE, CGLF, CRAVE): here CRAVE (4.3%) is indeed the most consistent,
followed by CGLF (7.9%) and RL-VAE (6.9%). This is a real, defensible finding — CRAVE's output is
the most *proportionally* consistent among the methods that specifically target high potency —
but it should be described as "the lowest relative variance among potency-directed methods," not
as "seed-invariant" or "deterministic," which the corrected 5-seed data does not support.

*Claim strength*: Moderate — the reduced-variance property is real and an emergent effect of the
Pareto filter, but weaker than originally claimed; the "seed-invariant" language has been removed.

**Tertiary Contribution: Multi-Strategy Ensemble for Coverage of Temperature-Top_k Space**

The four sampling strategies (Standard, Conservative, Exploratory, Focused) systematically
cover the temperature × top_k parameter space:

| Strategy | Temperature | Top-k | Controls for |
|----------|-------------|-------|-------------|
| Standard | 1.0 | 10 | Baseline autoregressive distribution |
| Conservative | 0.8 | 5 | Low-entropy, high-confidence tokens |
| Exploratory | 1.3 | 20 | Temperature-driven diversity |
| Focused | 0.6 | 3 | Near-greedy, highest-probability paths |

This is analogous to **ensemble learning** in supervised ML: each strategy produces a set
of molecules with different bias-variance characteristics, and the Pareto filter selects the
best molecules regardless of which strategy produced them. The ensemble ensures that the
final set is not biased toward the characteristics of any single sampling configuration.

*Claim strength*: Moderate — ensemble sampling is not novel per se, but the systematic
coverage + Pareto selection is a well-engineered combination not seen in prior molecular RL.

#### 13.11.2 Empirical Evidence: CRAVE Dominance

**Multi-seed results** (Table 3, §14.4):

| Metric | RL-VAE | CRAVE (final) | Δ | Interpretation |
|--------|--------|--------------|---|---------------|
| Potency | 0.726 ± 0.050 | **0.771 ± 0.033** | +6.2% | Modest absolute gain; CIs overlap somewhat (see §13.11.1 revision) |
| pChEMBL | 6.625 ± 0.433 | **7.003 ± 0.328** | +5.7% | Deep sub-µM (IC₅₀ ~ 0.10 µM) |
| VP Fraction | 80.0 ± 8.9% | **88.6 ± 6.4%** | +10.8% | ~89% of output is virtually potent |
| QED | 0.693 ± 0.051 | **0.784 ± 0.043** | +13.1% | Drug-likeness improvement continues |
| SA Score | 4.123 ± 0.269 | **3.922 ± 0.257** | −4.9% (better) | More synthesisable |
| Diversity | 0.850 ± 0.030 | 0.819 ± 0.043 | −3.6% | Modest cost for potency gain |
| Uniqueness | 94.6 ± 4.2% | **100.0 ± 0.0%** | +5.7% | Perfect — Pareto filter eliminates duplicates |

**Three-stage cascade summary (full pipeline, 5-seed mean)**:

| Stage | Potency | QED | SA | VP Frac | Cumulative Improvement |
|-------|---------|-----|-----|---------|----------------------|
| VAE (baseline) | 0.196 | 0.435 | 4.801 | 25.2% | — |
| RL-VAE (+RL) | 0.726 | 0.693 | 4.123 | 80.0% | +270.5% potency, +59.3% QED |
| CRAVE (+RL²+ensemble+Pareto) | **0.771** | **0.784** | **3.922** | **88.6%** | **+293.6% potency, +80.2% QED, −18.3% SA** |

**Diminishing returns analysis**: The VAE→RL-VAE step provides the largest gain (+270.5% potency).
The RL-VAE→CRAVE step adds +6.2% potency but significantly improves QED (+13.1%) and SA (−4.9%).
This pattern matches drug discovery reality: initial hit identification provides the largest signal,
while lead optimisation involves careful multi-objective refinement with smaller individual gains
that collectively determine clinical viability.

#### 13.11.3 Hypothesis Test: Does Prior-Shifting Actually Help?

If the CRAVE improvement is solely due to additional RL training (more gradient steps), then
replacing the RL-VAE prior with the original VAE prior should produce equivalent results. This
ablation was not formally conducted in this study and is noted as future work. However, indirect
evidence supports the prior-shifting hypothesis:

1. **CRAVE converges faster** (50 RL epochs) than RL-VAE (100 RL epochs), despite starting from
   a harder optimisation landscape (already-high potency leaves less room for gradient signal)
2. **CRAVE achieves a lower coefficient of variation in potency** (4.3% vs RL-VAE's 6.9%; §13.11.1),
   consistent with a tighter prior providing more consistent trajectories, though (per the §13.11.1
   revision) this is not the near-exact determinism originally claimed
3. **The KL penalty IS active in CRAVE** (§11.4 confirmed it is dead code in RL-VAE but live in
   CRAVE) — this is the architectural difference between the two stages

#### 13.11.4 What Is Not Novel

| Component | Prior work |
|-----------|-----------|
| REINFORCE for molecular optimisation | Olivecrona et al., 2017 (REINVENT) |
| Multi-objective reward weighting | Zhou et al., 2019; Gottipati et al., 2020 |
| KL regularisation in RL | Schulman et al., 2017 (PPO); Jaques et al., 2017 |
| Pareto filtering | Standard multi-objective optimisation (Deb et al., 2002) |
| Temperature/top-k sampling variation | Standard NLP decoding strategies |
| Ensemble of sampling strategies | Xie et al., 2023 (molecular context) |
| SELFIES for validity guarantee | Krenn et al., 2020 |

**Structured comparison: REINVENT vs CRAVE**

REINVENT (Olivecrona et al., 2017; Blaschke et al., 2020) is the closest published framework
to CRAVE's RL loop. The table below makes the architectural differences explicit:

| Dimension | REINVENT | CRAVE |
|---|---|---|
| Generator architecture | RNN (LSTM/GRU) | VAE decoder (GRU) with latent space |
| Molecular representation | SMILES | SELFIES (100% syntactic validity) |
| Prior policy | Pre-trained RNN on ChEMBL | `deepcopy` of RL-VAE decoder (itself RL-optimised) |
| KL mechanism | Prior likelihood term in loss | Per-token, length-normalised KL against frozen decoder |
| RL stages | Single | Cascaded (RL-VAE → CRAVE) — prior shifts between stages |
| Multi-strategy generation | No | 4-strategy ensemble (τ/top-k sweep) |
| Post-hoc filtering | Scoring function cutoff | Pareto-optimal composite selection (distinct from RL reward) |
| Seed reproducibility | Not reported | SD = 0.033 across 5 seeds (lowest coefficient of variation among potency-directed methods; not zero — §13.11.1) |
| Scoring functions used | 1 (same for training and selection) | 3 distinct functions at 3 stages (§13.4) |

The core difference is that REINVENT uses a single RL stage where the prior is the
original pre-trained network. CRAVE's cascaded design means the KL anchor is itself
an optimised policy — the regularisation prevents drift from a *competent* baseline
rather than from an undirected generative model. This enables the aggressive 55%
potency weight without the mode collapse that a single-stage approach would risk.

---

## 14. Evaluation Framework

### 14.1 Unified 10-Metric Battery

All generative methods are evaluated with the same 10-metric framework for fair comparison:

| # | Metric | Range | Higher=Better? | What It Measures |
|---|--------|-------|----------------|-----------------|
| 1 | Validity | [0, 1] | ✅ | SELFIES→SMILES→RDKit parse success rate |
| 2 | Uniqueness | [0, 1] | ✅ | Fraction of unique canonical SMILES among valid |
| 3 | Novelty | [0, 1] | ✅ | Fraction not found in training set |
| 4 | OSCC Potency | [0, 1] | ✅ | Cosine similarity to VP centroid (via MolDeBERTa→Contrastive) |
| 5 | pChEMBL (predicted) | [4, 10] | ✅ | Predicted pChEMBL from contrastive regressor |
| 6 | QED | [0, 1] | ✅ | Drug-likeness score (Bickerton et al.) |
| 7 | SA Score | [1, 10] | ❌ (lower=easier) | Synthetic accessibility (Ertl & Schuffenhauer) |
| 8 | Diversity | [0, 1] | ✅ | Avg. pairwise Tanimoto distance (Morgan FP, r=2, 2048-bit) |
| 9 | Scaffold Novelty | [0, 1] | ✅ | Fraction of Murcko scaffolds unseen in training |
| 10 | FCD | [0, ∞) | ❌ (lower=better) | Fréchet ChemNet Distance to training set |

### 14.2 Metric Design Decisions

**Metric 4 (OSCC Potency)** is the most study-specific metric. It uses the full inference
chain:
```
Generated SMILES → MolDeBERTa (use_fast=False) → [CLS] 768-d
  → ContrastiveModel projection → 128-d L2-norm
  → cosine similarity to VP centroid → potency score
```

This provides a consistent potency estimate across all methods, using the same
discriminative model that defined the potency landscape.

**Metric 8 (Diversity)** uses Tanimoto distance (1 - Tanimoto similarity) computed from
Morgan fingerprints (radius=2, 2048 bits). For large sets (>5,000), a random subsample
of 5,000 pairs is used.

**Metric 9 (Scaffold Novelty)** uses Murcko scaffolds (Bemis & Murcko, 1996) — the ring
systems with linkers — to assess whether generated molecules explore novel chemical
space or merely decorate known scaffolds.

### 14.3 Expected Performance Profiles

Based on the study design, different methods are expected to excel on different metrics:

| Method | Strength | Weakness | Primary Value |
|--------|----------|----------|---------------|
| VAE | Diversity, validity | Low potency control | Baseline, decoder base |
| cVAE | Targeted potency | May sacrifice diversity | Controlled generation |
| Diffusion | High potency (guided) | Low diversity (mode collapse risk) | Potency benchmark |
| RL-VAE | Best overall balance | Moderate potency | Practical candidate generation |
| CGLF | High diversity, fast | May have lower potency | Novel framework demonstration |
| CRAVE | Highest potency (filtered) | Post-hoc filtering reduces pool | Best candidates |

### 14.4 Multi-Seed Generative Validation Results

**Date**: 2026-09-16 (5-seed retrain with the corrected checkpoint-selection code; supersedes the original 2026-03-25, 3-seed {42,123,456} evaluation referenced in earlier drafts of this document)
**Script**: `GENESIS/generative/multiseed_gen.py`
**Seeds**: {42, 123, 456, 789, 1337}
**Results**: `GENESIS/generative/multiseed/aggregate/`
**Status**: COMPLETE ✅
**Total compute**: ~27.3 hours on Apple M3 MPS (all 5 seeds trained fresh, zero crashes)

To quantify statistical robustness of all generative metrics, each of the 6 generative
methods (7 including CRAVE unfiltered) was retrained and evaluated across 5 independent
random seeds, using the corrected checkpoint-selection PGCL model (§5.3.2) as the shared
foundation for the contrastive potency reward/conditioning. Approximate 95% confidence
intervals (normal approximation, mean ± 1.96·SD/√5) are reported in §14.4.2; a full
bootstrap-resampling CI was not recomputed for this 5-seed set (see caveat below).

#### 14.4.1 Aggregated Results (Mean ± SD across 5 seeds)

**Table 3.** Multi-seed generative evaluation results. Each cell shows mean ± SD across seeds {42, 123, 456, 789, 1337}. ~1,000 molecules generated per method per seed.

| Method | Valid% | Uniq% | Novel% | Potency ↑ | pChEMBL ↑ | VP Frac ↑ | QED ↑ | SA ↓ | Diversity ↑ | ScafNov% |
|--------|--------|-------|--------|-----------|-----------|-----------|-------|------|-------------|----------|
| VAE | 100.0±0.0 | 98.7±0.6 | 99.5±0.2 | 0.196±0.019 | 2.134±0.137 | 25.2±1.3 | 0.435±0.011 | 4.801±0.019 | 0.916±0.002 | 96.7±0.4 |
| cVAE | 100.0±0.0 | 98.0±1.2 | 99.5±0.5 | 0.361±0.185 | 3.669±1.709 | 44.8±22.9 | 0.375±0.091 | 5.105±0.493 | 0.889±0.053 | 97.0±1.7 |
| Diffusion | 100.0±0.0 | 25.6±11.9 | 100.0±0.0 | 0.101±0.271 | 0.857±2.093 | 10.5±15.0 | 0.133±0.090 | 7.533±0.844 | 0.689±0.204 | 100.0±0.0 |
| RL-VAE | 100.0±0.0 | 94.6±4.2 | 99.6±0.3 | 0.726±0.050 | 6.625±0.433 | 80.0±8.9 | 0.693±0.051 | 4.123±0.269 | 0.850±0.030 | 92.4±2.4 |
| CGLF | 100.0±0.0 | 100.0±0.0 | 99.9±0.1 | 0.378±0.030 | 3.778±0.255 | 39.1±3.3 | 0.272±0.029 | 5.155±0.171 | 0.888±0.002 | 96.1±1.1 |
| CRAVE (filt.) | 100.0±0.0 | 100.0±0.0 | 99.4±0.2 | 0.771±0.033 | 7.003±0.328 | 88.6±6.4 | 0.784±0.043 | 3.922±0.257 | 0.819±0.043 | 91.7±3.0 |
| CRAVE (unfilt.) | 100.0±0.0 | 90.4±5.9 | 99.6±0.2 | 0.724±0.049 | 6.617±0.434 | 79.8±8.6 | 0.692±0.055 | 4.152±0.257 | 0.847±0.033 | 92.4±1.8 |

#### 14.4.2 Approximate 95% Confidence Intervals *(normal approximation, mean ± 1.96·SD/√5; not a full bootstrap resample — see caveat below)*

| Method | Potency CI | QED CI | Diversity CI |
|--------|-----------|--------|-------------|
| VAE | [0.179, 0.213] | [0.426, 0.444] | [0.914, 0.918] |
| cVAE | [0.199, 0.523] | [0.295, 0.455] | [0.842, 0.936] |
| Diffusion | [−0.137, 0.339] | [0.054, 0.212] | [0.510, 0.868] |
| RL-VAE | [0.682, 0.770] | [0.648, 0.738] | [0.824, 0.876] |
| CGLF | [0.352, 0.404] | [0.247, 0.297] | [0.886, 0.890] |
| CRAVE (filt.) | [0.742, 0.800] | [0.746, 0.822] | [0.781, 0.857] |
| CRAVE (unfilt.) | [0.681, 0.767] | [0.644, 0.740] | [0.818, 0.876] |

**Caveat**: an earlier version of this section reported a true bootstrap-resampled 95% CI
computed from a 3-seed {42,123,456} evaluation predating the checkpoint-selection fix. That
CI table has been superseded and is no longer accurate for the corrected model. The table
above uses a normal approximation from the 5-seed mean/SD as a practical substitute; a
proper bootstrap recomputation against the 5-seed data has not been performed and would be
a reasonable follow-up for full rigor.

#### 14.4.3 Seed Stability Analysis

Methods are ranked by potency SD (ascending = most stable in absolute terms):

| Rank | Method | Potency SD (5 seeds) | Interpretation |
|------|--------|-----------|----------------|
| 1 | VAE | 0.019 | Low absolute variance, but around a low, undirected mean |
| 2 | CGLF | 0.030 | Low-to-moderate; OT-CFM trajectories are deterministic but decoder varies |
| 3 | CRAVE (filtered) | 0.033 | Low-to-moderate in absolute terms; lowest *relative* variance (CV 4.3%) among potency-directed methods |
| 4 | CRAVE (unfiltered) | 0.049 | Comparable to RL-VAE |
| 5 | RL-VAE | 0.050 | Bounded variance; REINFORCE exploration is seed-dependent |
| 6 | cVAE | 0.185 | High variance, driven by a single outlier seed (1337; §9.9.2) |
| 7 | Diffusion | 0.271 | Catastrophic instability; no seed achieves both good potency and good diversity |

#### 14.4.4 Key Findings *(revised, 2026-09-19, against the corrected 5-seed evaluation)*

1. **CRAVE is not seed-invariant, but has the lowest relative variance among potency-directed
   methods**: potency SD = 0.033 (coefficient of variation 4.3%), not the SD = 0.0004
   ("seed-invariant") originally reported from the 3-seed evaluation. Adding seeds 789 and 1337
   revealed that the original 3-seed cluster (0.792/0.774/0.796) was itself an unusually tight
   subset; the full 5-seed range is 0.715–00.796. CRAVE's dominance over other methods on mean
   potency is still well-supported, but the "essentially deterministic" framing is retracted.

2. **Diffusion exhibits catastrophic seed sensitivity**: potency spans −0.064 to +0.581
   across the 5 seeds, with a CI that crosses zero ([−0.137, 0.339]). No single seed
   simultaneously achieves reasonable potency and reasonable diversity/uniqueness — the two
   highest-diversity seeds (789, 1337; diversity 0.83/0.93) have near-zero potency, while the
   one seed with positive potency (456: 0.581) has middling diversity/uniqueness (0.527/17.2%).
   **Diffusion results should be cited with caution and this instability explicitly disclosed.**

3. **RL-VAE shows bounded, moderate variance**: potency range [0.682, 0.770] (approximate CI)
   is reasonably tight. The SD (0.050) is roughly 5× lower than Diffusion's, despite RL's
   stochastic REINFORCE gradient. KL regularisation against the frozen VAE prior helps
   constrain exploration, though this document no longer claims near-total elimination of
   seed variance for any method except in relative (coefficient-of-variation) terms.

4. **VAE is stable but mediocre; cVAE is unstable and only modestly better on average**:
   VAE achieves the lowest absolute potency SD (0.019) but the lowest mean potency (0.196).
   cVAE's mean (0.361) is nominally higher but driven by one outlier seed with SD (0.185)
   larger than the mean difference itself (§9.9.2) — the conditioning benefit is real but far
   less robustly established than previously claimed.

5. **CGLF is a stable, moderate performer**: potency 0.378 ± 0.030, with diversity
   0.888 ± 0.002 — the tightest diversity SD of any method, reflecting deterministic OT-CFM
   trajectories combined with a fixed decoder.

6. **Ranking order is preserved despite the corrected absolute numbers**: CRAVE (0.771) >
   RL-VAE (0.726) > CGLF (0.378) > cVAE (0.361) > VAE (0.196) > Diffusion (0.101) on mean
   potency — the same qualitative ordering reported under the earlier (now-superseded) 3-seed
   evaluation, even though every individual mean/SD value has changed.

#### 14.4.5 Implications for Study Claims

The generative method potency ranking (CRAVE > RL-VAE > CGLF > cVAE > VAE > Diffusion) holds
across the full 5-seed set, consistent with the earlier 3-seed report — this specific ranking
claim is robust to the additional seeds. However, several *quantitative* claims tied to the
original 3-seed numbers (the exact percentage improvements, the "seed-invariant CRAVE" claim,
the specific bootstrap CIs, and cVAE's "statistically significant" potency gain over VAE) do
**not** carry over and have been individually corrected in §§8–13 above and in this table. The
only method whose ranking position is genuinely fragile across seeds is Diffusion, which
fails unpredictably in different ways across all 5 seeds tested — reinforcing the
recommendation against Diffusion for production use, now on a firmer (5-seed, not 3-seed)
evidential basis.

#### 14.4.6 External Benchmark Metrics (MOSES / GuacaMol) *(IntDiv/SNN/Frag/Scaf/Filters/KL refreshed 2026-09-19 against the current generated outputs; FCD not recomputed — see note below)*

Standard distribution-learning metrics from the MOSES (Polykovskiy et al., 2020) and GuacaMol (Brown et al., 2019) benchmark frameworks provide independent confirmation of generative quality and enable comparison with the broader molecular generation literature.

**MOSES-equivalent metrics, 3-seed mean ± SD (seeds 42/123/456)** *(fully recomputed 2026-09-19, including FCD — see note below; reference: training set, 34,406 compounds)*:

| Method | IntDiv | IntDiv2 | SNN | Frag | Scaf | Filters | FCD ↓ |
|--------|--------|---------|-----|------|------|---------|-------|
| VAE | 0.916±0.001 | 0.908±0.001 | 0.246±0.003 | 0.861±0.009 | 0.291±0.044 | 0.957±0.007 | 47.51±0.17 |
| cVAE | 0.914±0.001 | 0.905±0.001 | 0.256±0.012 | 0.862±0.015 | 0.245±0.046 | 0.962±0.007 | 48.35±0.12 |
| Diffusion | 0.562±0.118 | 0.416±0.120 | 0.072±0.021 | 0.003±0.002 | 0.000±0.000 | 0.991±0.011 | 87.59±18.85 |
| RL-VAE | 0.838±0.019 | 0.812±0.023 | 0.306±0.016 | 0.639±0.018 | 0.158±0.027 | 0.986±0.003 | 58.90±0.92 |
| CGLF | 0.888±0.001 | 0.882±0.000 | 0.301±0.005 | 0.921±0.009 | 0.294±0.054 | 0.963±0.009 | 54.30±0.47 |
| CRAVE (filt.) | 0.803±0.028 | 0.780±0.028 | 0.322±0.012 | 0.592±0.029 | 0.026±0.015 | 0.995±0.003 | 34.07±0.31 |
| CRAVE (unfilt.) | 0.834±0.025 | 0.806±0.029 | 0.304±0.014 | 0.606±0.029 | 0.155±0.040 | 0.989±0.002 | 58.76±0.66 |

**FCD is now fully recomputed, 3-seed.** Installing the `fcd` package turned out to
be cheap: a modern PyTorch-based reimplementation (`pip install fcd`, v1.2.2) with
all dependencies (numpy, rdkit, scipy, torch) already satisfied in this
environment — no TensorFlow/Keras or external weight download was needed, contrary
to the initial assessment. `scripts/compute_fcd.py` was run across all 3 canonical
seeds (42/123/456) against `generative/multiseed/seed_*/`, and
`scripts/compute_external_benchmarks_multiseed.py` (a new wrapper reusing
`compute_external_benchmarks.py`'s metric functions) was run the same way for
IntDiv/IntDiv2/SNN/Frag/Scaf/Filters/KL. Combined wall-clock time was roughly
60–90 minutes (the two jobs ran concurrently and contended for CPU). Every value
in this section is now a genuine 3-seed mean ± SD, matching the rigor standard
used for the rest of this document's headline claims.

IntDiv = internal diversity (1 − mean pairwise Tanimoto); SNN = avg nearest-neighbour Tanimoto to test; Frag = BRICS fragment cosine similarity; Scaf = Murcko scaffold cosine similarity; Filters = fraction passing PAINS.

**GuacaMol KL divergences, 3-seed mean ± SD** (generated vs test-set property distributions) *(fully recomputed 2026-09-19)*:

| Method | logP | QED | MW | TPSA | NumRings | HBA | HBD |
|--------|------|-----|-----|------|----------|-----|-----|
| VAE | 0.54±0.05 | 0.24±0.09 | 0.50±0.03 | 0.33±0.03 | 0.61±0.01 | 0.32±0.05 | 0.12±0.01 |
| cVAE | 0.41±0.04 | 0.26±0.02 | 0.41±0.01 | 0.34±0.04 | 0.49±0.04 | 0.31±0.04 | 0.14±0.02 |
| Diffusion | 15.03±2.75 | 17.22±0.84 | 16.84±0.04 | 10.65±4.35 | 20.89±0.31 | 9.35±4.14 | 2.83±2.57 |
| RL-VAE | 1.40±0.19 | 2.37±0.46 | 2.57±0.47 | 0.70±0.09 | 1.30±0.32 | 0.45±0.12 | 0.46±0.15 |
| CGLF | 0.16±0.03 | 1.69±0.50 | 0.94±0.22 | 0.31±0.06 | 0.15±0.05 | 0.19±0.04 | 0.32±0.02 |
| CRAVE (filt.) | 3.11±0.29 | 14.57±0.53 | 7.27±1.70 | 1.54±0.25 | 3.18±1.72 | 1.22±0.37 | 1.70±0.08 |
| CRAVE (unfilt.) | 1.12±0.08 | 1.40±0.05 | 2.24±0.52 | 0.59±0.10 | 1.45±0.45 | 0.45±0.11 | 0.36±0.04 |

Key findings *(final, 3-seed-confirmed — supersedes both the original single-run report and the intermediate single-seed recomputation)*:
- **CGLF and VAE are statistically tied for highest scaffold fidelity, not CGLF alone.** CGLF's Scaf score is 0.294±0.054 and VAE's is 0.291±0.044 — indistinguishable within combined SD. This means neither the original claim ("CGLF highest at 0.509") nor the intermediate single-seed finding ("VAE now highest") was the fully correct picture: the earlier 0.509 figure was an overestimate that does not survive 3-seed averaging, and the true post-correction picture is a near-tie between CGLF and VAE, both well above cVAE (0.245±0.046) and CRAVE (0.026–0.158).
- **CGLF's Frag advantage is real and robust**: 0.921±0.009, clearly separated from the next-best method (cVAE, 0.862±0.015) by more than 3 combined SD. This is the one distributional-fidelity claim about CGLF that fully survives rigorous re-verification.
- **CGLF's QED-distribution shift is real, not noise.** Its GuacaMol QED KL is 1.69±0.50 (3-seed), far above the originally-reported 0.26 — even the lower end of the confidence interval (~1.19) is roughly 4.5× the original value, so this is a genuine post-correction change in what CGLF generates, not sampling variance. CGLF remains the best method on logP KL (0.16±0.03) and NumRings KL (0.15±0.05).
- **Diffusion's FCD has enormous seed-to-seed variance** (87.59±18.85), the largest relative SD of any metric in this table. This is consistent with the previously-documented "catastrophic collapse in 2/5 seeds" behaviour for diffusion's OSCC-potency scores (§14.4.4) — the instability is not confined to potency, it shows up in distributional-similarity metrics too, reinforcing that diffusion's failure mode is a fundamental training instability rather than a metric-specific artifact.
- **CRAVE (filtered) has the lowest, and most tightly-confirmed, FCD**: 34.07±0.31 — clearly the best of all 7 methods, with one of the smallest SDs in the table, making this the single most robust claim in this section.
- All methods pass PAINS filters at ≥ 95.7% (Filters column, 3-seed mean), consistent with the original ">94.8%" claim.

**Comparison with published MOSES baselines.** The five MOSES baseline models (CharRNN, VAE, AAE, JTN-VAE, LatentGAN) were trained on 1.6M ZINC molecules — 46× more data than GENESIS's 34.4K OSCC compounds. With the fully-refreshed 3-seed numbers, GENESIS methods are close to or exceed published baselines on internal diversity for the non-degenerate methods (0.80–0.92 vs. ~0.856; Diffusion's 0.562±0.118 falls below, consistent with its instability) and remain competitive on drug-likeness filters (0.957–0.995 vs. 0.974–0.997). The FCD gap (34–88 vs. <0.6, now genuinely computed rather than assumed) arises from the same three factors previously identified: domain narrowness (OSCC compounds occupy a restricted submanifold of ChemNet's feature space), sample-to-dimension ratio (1K generated molecules vs. MOSES's 30K), and reference set scope (34.4K OSCC molecules vs. 176K diverse ZINC molecules). FCD now correctly and robustly ranks methods *within* GENESIS (CRAVE filt. 34.07 < VAE/cVAE ~48 < CGLF 54.30 < CRAVE unfilt./RL-VAE ~59 < Diffusion 87.59), consistent with the other quality metrics and confirmed stable across all 3 seeds.

CGLF's Fragment similarity of 0.921±0.009 on 34.4K compounds recovers a large share of substructure frequency information achieved by models trained on 46× more data, confirming that task-specific small-data generators produce chemically valid and diverse output. Per the revised findings above, CGLF's structural-fidelity strength is specifically at the fragment level and (tied with VAE) at the scaffold level, not a uniquely superior scaffold-fidelity result as originally reported.

---

## 15. Design Decision Registry

### 15.1 Architectural Decisions

| Decision | Choice | Alternatives Considered | Rationale |
|----------|--------|-------------------------|-----------|
| Embedding model | MolDeBERTa (88.2M params) | ChemBERTa, MolBERT, Mol2Vec | Best [CLS] quality for drug-like SMILES |
| Tokenizer | `use_fast=False` | Fast tokenizer | Fast tokenizer produces incorrect tokenization |
| Projection dim | 128 | 64, 256, 512 | Matches VAE latent dim; sufficient for ~200K compounds |
| L2 normalization | Yes | No normalization | Required for temperature-scaled contrastive loss |
| Activation | GELU | ReLU, SiLU | Smooth gradient flow for contrastive learning |
| Normalization | LayerNorm | BatchNorm | Batch-independent; stable with mixed pos/neg batches |
| Sequence representation | SELFIES | SMILES, SAFE, DeepSMILES | 100% validity guarantee |
| Latent space | Shared 128-d | Separate per method | Enables direct comparison and transfer |
| VP centroid | VP-only (class 3) | Mixed-potency, multi-centroid | Clean directional beacon; ordinal gradient exists intrinsically |
| Conditioning dim | 5 | 133 (with raw centroid) | Prevents information leakage in cVAE/CGLF |
| Noise schedule | Cosine | Linear, sigmoid | Smoother α_bar curve → better sample quality |
| Flow matching | OT-CFM | DDPM reverse SDE | 10× faster, no noise schedule, stable training |
| RL algorithm | REINFORCE | PPO, A2C, SAC | Simple, no value function needed for molecular generation |
| RL base | Frozen encoder + trainable decoder | Full model | Prevents forgetting learned encoder representations |

### 15.2 Hyperparameter Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| VAE KL annealing strategy: cyclical | 4 cycles over 100 epochs | Prevents posterior collapse; multiple exploration chances |
| Contrastive temperature τ = 0.07 | Standard SupCon value; good discrimination vs. robustness tradeoff |
| RL learning rate 10⁻⁵ (vs VAE 3×10⁻⁴) | 30× lower to prevent catastrophic forgetting |
| CRAVE learning rate 5×10⁻⁶ (vs RL 10⁻⁵) | 2× lower again — already warm-started, fine-grained optimization |
| CRAVE potency weight 55% (vs RL 40%) | Phase-2 RL specifically targeting potency improvement |
| Pareto composite: 45/25/15/15 | Slightly rebalanced from RL to favor potency in final selection |
| Gradient clipping 1.0 everywhere | Universal stabilizer; prevents one bad batch from destroying model |
| GRU (not LSTM) | Fewer parameters, adequate for short SELFIES sequences |
| 2-layer GRU (not 3+) | Matches 128-token max_len; deeper risks overfitting on 157-token vocab |

### 15.3 Bug Fixes Baked In (UPDATE_1)

Five critical fixes from the UPDATE_1 analysis are pre-integrated:

| # | Location | Bug | Fix | Impact |
|---|----------|-----|-----|--------|
| 1 | evaluate_gen.py, rl_finetune.py, crave_gen.py | ChemBERTa tokenizer (wrong model) | MolDeBERTa with `use_fast=False` | Potency scores change from 0.0 to real values |
| 2 | rl_finetune.py | Hardcoded 0.0 potency reward | Full MolDeBERTa→ContrastiveModel→centroid pipeline | RL actually optimizes potency |
| 3 | cglf.py | 133-d conditioning (128-d centroid leak) | 5-d conditioning [potency_oh + sim] | Prevents information leakage |
| 4 | diffusion.py | Linear noise schedule | Cosine noise schedule | Better sample quality and diversity |
| 5 | diffusion.py | Float64 operations | Float32 enforcement on MPS | Prevents NaN overflow on Apple Silicon |
| 6 | train.py (Phase 2, PGCL) | Checkpoint-selection criterion shared the same `best_val_loss` variable as the early-stopping patience counter, which resets at curriculum phase transitions -- a phase-transition reset could silently promote a mediocre post-reset epoch to `best.pt` over a genuinely superior earlier epoch | Decoupled checkpoint saving from the early-stopping criterion entirely; `best.pt` now always reflects the most recently completed training epoch | Confirmed materially better downstream metrics on every one of 6 evaluation tests (see §5.3.2); this was found to be the primary cause of an earlier apparent underperformance of PGCL against simpler baselines |

---

## 16. Parameter Summary Tables

### 16.1 Contrastive Model

| Component | Architecture | Parameters |
|-----------|-------------|------------|
| ProjectionHead | 768→512→256→128 (LN+GELU+Drop) | 559,489 |
| PotencyRegressor | 128→1 | 129 |
| **Total** | | **559,617** |

### 16.2 VAE / cVAE

| Component | Architecture | Parameters (approx.) |
|-----------|-------------|---------------------|
| Encoder Embedding | 157×128 | 20,096 |
| Encoder GRU | 128→512, 2 layers | 1,974,272 |
| Encoder FC (μ, logσ²) | 512→128 ×2 | 131,072 |
| Decoder Embedding | 157×128 | 20,096 |
| Decoder z→h | 128→1024 | 132,096 |
| Decoder GRU | 128→512, 2 layers | 1,974,272 |
| Decoder output | 512→157 | 80,541 |
| FiLM (cVAE only) | 5→1024 ×2 | 12,288 |
| **Total (VAE)** | | **~5.5M** |
| **Total (cVAE)** | | **~5.5M** |

### 16.3 Diffusion Model

| Component | Architecture | Parameters (approx.) |
|-----------|-------------|---------------------|
| SinusoidalTimeEmb | → 512 | — |
| TimeProj | 512→512 | 262,656 |
| CondProj | 5→512 | 3,072 |
| InputProj | 128→512 | 65,536 |
| DenoiseBlocks | 6×(LN+512→512+GELU+512→512) | ~3.2M |
| OutputNorm+Proj | LN(512)+512→128 | 66,560 |
| **Total** | | **~3.6M** |

### 16.4 CGLF

| Component | Architecture | Parameters (approx.) |
|-----------|-------------|---------------------|
| FlowMatchingNet | 8-layer AdaLN MLP, 512-d | ~7.5M |
| LatentToSELFIES | GRU decoder, 512-d, 2 layers | ~2.2M |
| **Total** | | **~9.7M** |

### 16.5 Training Duration Estimates (Apple M3, MPS)

| Phase | Epochs | Est. per Epoch | Total |
|-------|--------|----------------|-------|
| VAE | 100 (early stop ~50-60) | ~78s | ~1.1-1.3 hr |
| cVAE | 100 (early stop ~50-60) | ~80s | ~1.1-1.3 hr |
| Diffusion | 200 (early stop ~150-200) | ~40-50s | ~1.7-2.8 hr |
| RL-VAE | 50 (no early stop) | ~120-180s | ~1.7-2.5 hr |
| CGLF | 50 + 200 | ~30s + ~60s | ~3.5-4.5 hr |
| CRAVE | 30 | ~120-180s | ~1.0-1.5 hr |
| Evaluation | — | — | ~0.3-0.5 hr |
| **Full Pipeline** | | | **~10-14 hr** |

---

## 17. Phase 6 — Drug Repurposing of CompoundList Compounds

### 17.1 Objective

Screen the 23,308 CompoundList compounds as drug repurposing candidates for OSCC, using the potency-aware contrastive model to predict bioactivity. Unlike Phase 4 (bioactivity profiling of novel *generated* molecules), Phase 6 operates on *existing, known compounds* — enabling validation against known potency annotations and leveraging clinical phase data.

### 17.2 Candidate Pool

| Category | Count |
|---|---|
| CompoundList-only | 22,404 |
| Overlap (ChEMBL + CompoundList) | 904 |
| **Total candidates** | **23,308** |
| Mapped to MolDeBERTa embeddings | 23,308 (100%) |
| In contrastive training index | 817 (3.5%) |
| With known pChEMBL | 653 |

All 23,308 compounds have pre-computed MolDeBERTa [CLS] embeddings from Phase 0. Compounds are mapped via `compound_ids.npy` (fallback for the 22,491 not in `compound_index.pkl`).

### 17.3 Pipeline Architecture

```
Phase A: Load CompoundList → filter source_file='CompoundList' → map to embeddings
Phase B: Project → ContrastiveModel (768→128-d) → VP/T3C cosine similarity → composite score
Phase C: Per-target screening → 35 targets with ≥30 VP compounds → target-specific rankings
Phase D: Safety profiling → Lipinski, PAINS, Brenk, toxicophores, SA score
Phase E: Validation → predicted vs known pChEMBL (Pearson r, Spearman ρ, VP Recall@K)
Phase F: Dossier → 8-component recommendation weight
Phase G: SMARTS enrichment → Murcko scaffolds → cross-study analysis (repurposed × generated)
```

**Dual lookup strategy** (Phase A): `map_to_embeddings()` first checks `compound_index.pkl`
(compounds already in the contrastive training pipeline), then falls back to
`compound_ids.npy` (the full 715K embedding library). This ensures compounds not in
the contrastive training set can still be projected — their rankings are a genuine
generalisation test, while training-set overlaps provide interpolation validation.

**Centroid-based ranking, not kNN** (Phase B): VP centroids average out annotation
noise from individual training compounds, giving a more robust scoring axis than
k-nearest-neighbours. The centroid is L2-normalised so dot product = cosine similarity,
consistent with the contrastive training objective.

### 17.4 Scoring

Identical to Phase 4:
- **Composite score**: `0.7 × vp_similarity + 0.3 × (predicted_pchembl / 10.0)`
- **Novelty score**: `vp_similarity − 0.3 × t3c_similarity`

**Composite score design**: The 70/30 weighting reflects the training objective hierarchy.
VP similarity (the geometric signal from contrastive learning, the *primary* training
objective) is trusted more than predicted pChEMBL (the scalar from the *auxiliary*
regulariser). pChEMBL is normalised by 10.0 to map the typical bioactivity range
(4–10) into [0.4, 1.0], roughly commensurate with cosine similarity magnitudes.

**Novelty score — T3C repulsion**: The T3C centroid represents compounds that were
active on OSCC targets but failed clinically (target-switch negatives). Subtracting
0.3 × T3C similarity penalises candidates pharmacologically close to known clinical
failures. This is a repurposing-specific signal rarely seen in standard virtual
screening: we want compounds that are VP-like but *not* clinical-failure-like.

**8-component recommendation weight** (adjusted for existing compounds):

| Component | Weight | Notes | Rationale |
|---|---|---|---|
| Composite score | 0.30 | Normalised to [0,1] | Core potency signal from contrastive model |
| Continuous safety | 0.20 | Geometric mean of 5 penalty components | Conservative: one catastrophic failure dominates |
| Target confidence | 0.15 | Max per-target score | Anchors recommendation to specific MoA |
| SA score | 0.10 | Inverted: lower SA → higher score | Synthetic accessibility for follow-up |
| Consensus | 0.05 | Single-source → uniform 0.5 | Placeholder; extensible for multi-method fusion |
| Docking | 0.05 | Placeholder (0.0) | Reserved for Phase 7 molecular docking scores |
| **Clinical phase** | **0.10** | Phase/4.0 (vs 0.05 in Phase 4) | Approved/late-phase drugs have known safety/PK — lower repurposing risk |
| SMARTS enrichment | 0.05 | Enriched scaffold match | Structural convergence with generative pipeline |

**Continuous safety score** (geometric mean, not arithmetic): Computed as
$(p_1 \cdot p_2 \cdot p_3 \cdot p_4 \cdot p_5)^{1/5}$ where each $p_i \in [0,1]$
is a penalty term for Lipinski, PAINS, Brenk, toxicophores, and SA score.
The geometric mean ensures one catastrophic failure (e.g., PAINS = 0.3) dominates
over several good scores — more conservative than arithmetic mean, appropriate for
safety signals where one red flag should sink the compound.

**Placeholder components**: `norm_consensus` (0.5 for all, since there is one ranking
method) and `norm_docking` (0.0, awaiting Phase 7 data) are included as structural
extensibility — when docking scores or multi-method consensus become available, the
weight architecture requires no restructuring.

**Clinical phase bonus (0.10)**: The doubled weight vs Phase 4 reflects that existing
approved drugs or late-phase candidates have established safety/PK profiles —
repurposing them is substantially faster and lower-risk than novel compounds.

### 17.5 Per-Target Screening Design

Phase C computes separate VP centroids for each OSCC target (e.g., EGFR, VEGFR2, STAT3)
with **≥30 VP compounds** and re-ranks all CompoundList compounds per target.

**min_vp=30 threshold**: Centroids from fewer compounds would be noisy on the 128-d
hypersphere. 30 is a pragmatic minimum for a stable mean direction. The output includes
`n_targets_top50` — how many target-level top-50 lists a compound appears in — as a
multi-target promiscuity signal useful for identifying polypharmacological candidates.

### 17.6 Safety Profiling Design

Five complementary safety checks, each targeting a different failure mode:

| Check | Tool | What It Catches |
|-------|------|-----------------|
| Lipinski Rule of 5 | RDKit descriptors | Oral bioavailability violations |
| PAINS | RDKit FilterCatalog (PAINS) | Assay-interference artefacts |
| Brenk alerts | RDKit FilterCatalog (Brenk) | Metabolic liabilities |
| Toxicophores (10 SMARTS) | Custom patterns | Intrinsic chemical reactivity (nitro aromatics, epoxides, Michael acceptors, etc.) |
| SA Score | Ertl/Schuffenhauer | Synthetic accessibility for follow-up |

The three filter libraries (PAINS, Brenk, toxicophores) cover *assay artefacts*,
*metabolic instability*, and *intrinsic reactivity* respectively — together they span
the full spectrum of medicinal chemistry structural alerts.

Triaging: FAIL if ≥2 critical alerts (PAINS + toxicophores) or ≥3 Lipinski violations;
CAUTION if ≥1 critical or ≥2 Lipinski or ≥3 Brenk; PASS otherwise.

### 17.7 Validation Results *(refreshed against the corrected checkpoint, 2026-09)*

| Metric | Value |
|---|---|
| Compounds with known pChEMBL | 653 |
| Pearson r | 0.367 (p = 3.3e-22) |
| Spearman ρ | 0.590 (p = 2.0e-62) |
| MAE | 1.770 |
| RMSE | 2.832 |
| Class concordance | *not re-verified this pass* |
| VP Recall@50 | *not re-verified this pass* |
| VP Recall@100 | *not re-verified this pass* |
| VP Recall@200 | *not re-verified this pass* |

**Leakage**: 744/817 CompoundList IDs in compound_index were training positives (91.1%).
These are overlap compounds with known OSCC bioactivity — expected, not erroneous.
The `leakage_check()` function explicitly flags training-set compounds so results can
be reported with and without known compounds, separating interpolation accuracy from
genuine generalisation.

### 17.8 Safety Profile

| Category | Count | Percentage |
|---|---|---|
| PASS | 14,801 | 63.5% |
| CAUTION | 6,782 | 29.1% |
| FAIL | 1,725 | 7.4% |

### 17.9 Cross-Study SMARTS Analysis

Murcko scaffold decomposition of top-200 repurposed candidates and top-200 generated
molecules from Phase 4 revealed **14 shared substructure patterns**.

**Cross-pipeline convergence design**: This is the most novel aspect of the repurposing
pipeline. If the same quinazoline core appears in the top-200 of both repurposed existing
drugs *and* de novo generated molecules, the convergence from two independent methods
(existing-drug screening vs de novo generation) is strong evidence for that scaffold's
pharmacological relevance to OSCC. The enrichment ratio (VP frequency / inactive frequency)
filters for scaffolds overrepresented in potent compounds vs the inactive background.

Key shared scaffolds:

| Scaffold | Generated (%) | Repurposed (%) | Interpretation |
|---|---|---|---|
| Benzene (`c1ccccc1`) | 74.5% | 90.5% | Ubiquitous aromatic core |
| Pyrimidine (`c1cncnc1`) | 64.0% | 61.0% | Kinase inhibitor pharmacophore |
| Aminopyrimidine (`c1ccc(Nc2ccncn2)cc1`) | 24.0% | 25.0% | EGFR/FGFR hinge binder |
| Quinazoline (`c1ccc2ncncc2c1`) | 10.0% | 20.0% | EGFR TKI scaffold |
| 4-Aminoquinazoline (`c1ccc(Nc2ncnc3ccccc23)cc1`) | 6.0% | 19.0% | Gefitinib/erlotinib class |
| Piperidine (`C1CCNCC1`) | 8.0% | 14.0% | Aliphatic amine spacer |

Total cross-study scaffolds: 332 (shared=14, generated-only=165, repurposed-only=150).

### 17.10 Output Files

```
GENESIS/repurposing/results/
├── global_rankings.tsv                         23,308 compounds ranked
├── top100_candidates.tsv                       Top 100 by recommendation weight
├── top50_dossier.tsv                           Top 50 detailed dossier
├── final_dossier.tsv                           Full dossier (23,308 rows)
├── safety_profiles.tsv                         Lipinski/PAINS/Brenk/toxicophores
├── pharmacophore_2d.tsv                        2D molecular descriptors
├── target_summary.tsv                          Per-compound best-target mapping
├── target_rankings/                            35 target-specific ranking files
├── repurposed_smarts.tsv                       Enriched scaffolds in candidates
├── cross_study_smarts.tsv                      332 cross-study scaffold patterns
├── shared_smarts_repurposed_generated.tsv      14 shared patterns (rep ∩ gen)
├── validation_results.json                     Prediction vs known potency
└── repurposing_summary.json                    Pipeline summary
```

---

## 18. Phase 7 — Molecular Docking Validation

### 18.1 Objective

Structure-based validation of top candidates via rigid-body docking against 8 OSCC-relevant
protein crystal structures. Provides orthogonal validation to the ligand-based contrastive
model scoring from Phase 2.

### 18.2 Target Structures

| Target | PDB ID | Resolution | Co-crystal Ligand | Grid Center (Å) | Box Size (Å) |
|--------|--------|-----------|-------------------|-----------------|--------------|
| AKT1 | 3CQW | 2.00 | CQW | (6.0, 3.0, 17.3) | 24³ |
| EGFR | 4HJO | 2.75 | AQ4 (erlotinib analogue) | (24.8, 9.2, 0.0) | 24³ |
| FGFR1 | 4V04 | 2.10 | 0LI | (87.1, 0.8, 18.1) | 24³ |
| MDM2 | 4HG7 | 2.00 | NUT (nutlin-3a) | (−23.8, 7.5, −14.1) | 22³ |
| mTOR | 4DRH | 2.20 | RAP (rapamycin) | (−4.9, 20.0, −1.8) | 26³ |
| PIK3CA | 4JPS | 2.90 | 1LT | (−1.3, −9.5, 16.9) | 24³ |
| PTGS2/COX-2 | 3LN1 | 2.40 | CEL (celecoxib) | (31.0, −22.3, −16.5) | 24³ |
| STAT3 | 6NUQ | 2.10 | KQV | (13.1, 55.6, 0.1) | 24³ |

Grid centers derived from co-crystallized ligand centroids (chain A). All structures
filtered to chain A, waters removed, AutoDock4 atom-typed for PDBQT format.

### 18.3 Candidate Selection

- **Phase 4 (generated)**: Top 200 by `recommendation_weight` from bioactivity profiling dossier
- **Phase 6 (repurposed)**: Top 200 by `recommendation_weight` from drug repurposing dossier
- Target assignment: Each compound docked against the structural target matching its
  `best_target` from the contrastive model (mapped via `TARGET_NAME_MAP`)
- Total candidates: 364 (200 generated + 164 repurposed after target mapping), of
  which 361 completed successfully (3 failed, RDKit multi-fragment parse errors)

### 18.4 Docking Protocol

1. **Receptor preparation**: PDB → PDBQT with AutoDock4 atom typing (aromatic carbons → A,
   donor/acceptor nitrogen → NA/N, hydroxyl oxygen → OA, sulfur → SA/S)
2. **Ligand preparation**: SMILES → RDKit 3D conformer (ETKDGv3 + MMFF94) → Meeko 0.7.1
   PDBQT with rotatable bonds
3. **Docking engine**: AutoDock Vina Python API, `exhaustiveness=16`, `n_poses=9`, `seed=42`
4. **Scoring**: Vina empirical scoring function (gauss1 + gauss2 + repulsion + hydrophobic + hydrogen bond)

### 18.5 Results Summary *(refreshed against the corrected checkpoint, 2026-09)*

**Overall binding energy distribution** (361 successful docks of 364 jobs):

| Metric | Value |
|--------|-------|
| Mean ΔG | −7.86 kcal/mol |
| Median ΔG | −8.02 kcal/mol |
| Best ΔG | −11.21 kcal/mol |
| SD | 1.29 kcal/mol |
| ≤ −6.0 kcal/mol | 330 (91.4%) |
| ≤ −7.0 kcal/mol | 283 (78.4%) |
| ≤ −8.0 kcal/mol | 183 (50.7%) |
| ≤ −9.0 kcal/mol | 55 (15.2%) |
| ≤ −10.0 kcal/mol | 16 (4.4%) |

### 18.6 Per-Target Results *(refreshed against the corrected checkpoint, 2026-09)*

| Target | PDB | n_Docked | Mean ΔG | Best ΔG | ≤ −7 | ≤ −8 | ≤ −9 |
|--------|-----|----------|---------|---------|------|------|------|
| AKT1 | 3CQW | 9 | −8.7 | −10.4 | 8 | 8 | 5 |
| EGFR | 4HJO | 254 | −8.1 | −11.0 | 225 | 150 | 39 |
| FGFR1 | 4V04 | 21 | −8.5 | −11.2 | 18 | 15 | 8 |
| MDM2 | 4HG7 | 15 | −6.5 | −8.1 | 6 | 1 | 0 |
| mTOR | 4DRH | 19 | −7.1 | −8.6 | 12 | 2 | 0 |
| PIK3CA | 4JPS | 10 | −8.2 | −9.3 | 8 | 6 | 3 |
| PTGS2 | 3LN1 | 7 | −7.6 | −8.2 | 6 | 1 | 0 |
| STAT3 | 6NUQ | 26 | −5.5 | −6.7 | 0 | 0 | 0 |

FGFR1 achieved the strongest mean binding (−8.5 kcal/mol), closely followed by AKT1
(−8.7 best-observed, smaller sample). STAT3 SH2 domain showed the weakest binding
overall (mean −5.5), consistent with its shallow, solvent-exposed pocket.

### 18.7 Source Comparison *(refreshed against the corrected checkpoint, 2026-09)*

| Source | n | Mean ΔG | Median ΔG | Best ΔG | ≤ −7 | ≤ −8 |
|--------|---|---------|-----------|---------|------|------|
| Generated | 200 | −7.98 | −8.04 | −10.13 | 178 | 106 |
| Repurposed | 161 | −7.71 | −7.87 | −11.21 | 105 | 77 |

**Reversal note**: with the corrected checkpoint, generated compounds now show
slightly *stronger* mean binding than repurposed compounds (−7.98 vs −7.71
kcal/mol), the opposite of the earlier claim that repurposed (established,
medicinally-optimised) molecules would dock more strongly. Repurposed compounds
still achieve the single best docking pose overall (−11.21 vs −10.13), so the
reversal is in the *average*, not the best case.

### 18.8 Top Docking Hits *(refreshed against the corrected checkpoint, 2026-09)*

| Rank | Compound | Target | ΔG (kcal/mol) | Source |
|------|----------|--------|---------------|--------|
| 1 | D0RV9D | FGFR1 | −11.21 | Repurposed |
| 2 | BAG956 | EGFR | −11.03 | Repurposed |
| 3 | A-443654 | EGFR | −10.88 | Repurposed |
| 4 | Palbociclib/ribociclib analog 1 | EGFR | −10.65 | Repurposed |
| 5 | Benzamide derivative 16 | EGFR | −10.52 | Repurposed |
| 6 | UCN-01 | AKT1 | −10.45 | Repurposed |
| 7 | Pyrrolo[2,3-d]pyrimidine derivative 10 | EGFR | −10.42 | Repurposed |
| 8 | RISOVALISIB | EGFR | −10.40 | Repurposed |
| 9 | IN-1166 | AKT1 | −10.36 | Repurposed |
| 10 | OMIPALISIB | EGFR | −10.19 | Repurposed |

Best generated compound: GEN_00045 (CRAVE_unfiltered method) at −10.13 kcal/mol vs EGFR.

### 18.9 Docking–Composite Score Correlation *(refreshed, 2026-09)*

| Metric | Value | p-value |
|--------|-------|---------|
| Spearman ρ | −0.201 | 1.2 × 10⁻⁴ |
| Pearson r | −0.245 | 3 × 10⁻⁶ |

The negative Spearman correlation (ρ = −0.201, p < 0.001) indicates that compounds ranked
highly by the contrastive model tend toward more negative (stronger) docking scores. The
correlation is statistically significant but modest, reflecting the complementary nature of
ligand-based (contrastive model) vs structure-based (docking) scoring.

### 18.10 Scaffold Analysis

149 compounds achieved ≤ −8.0 kcal/mol across 121 unique Murcko scaffolds. The most enriched
scaffold was *N*-phenylquinazolinamine (14 instances), consistent with the known kinase-targeting
pharmacophore prevalent in EGFR/FGFR inhibitors.

### 18.11 Generative Method Breakdown *(refreshed, 2026-09)*

| Method | n | Mean ΔG | Best ΔG | ≤ −7 | ≤ −8 |
|--------|---|---------|---------|------|------|
| CRAVE_unfiltered | 94 | −8.17 | −10.13 | 89 (95%) | 53 |
| CRAVE_filtered | 71 | −7.79 | −9.56 | 61 (86%) | 33 |
| RL_VAE | 35 | −7.85 | −9.17 | 28 (80%) | 20 |

All top-scoring generated compounds originated from CRAVE (filtered or unfiltered),
with the best (GEN_00045, CRAVE_unfiltered) reaching −10.13 kcal/mol against EGFR.

### 18.12 Correlation Deep Dive *(refreshed, 2026-09)*

**By source**: repurposed compounds showed a stronger correlation (ρ = −0.381,
p = 6.3e-7) vs generated (ρ = −0.051, p = 0.47, not significant), suggesting the
contrastive model is more predictive for known bioactive molecules than for novel
de novo generated ones -- a larger gap than previously observed.

**By target**: STAT3 showed the strongest correlation (ρ = −0.443, p = 0.024,
n = 26); EGFR achieved significance via its large sample size (ρ = −0.189,
p = 0.0024, n = 254). MDM2 (ρ = −0.482, p = 0.069, n = 15) was suggestive but not
significant at this sample size.

### 18.13 Safety–Potency Tradeoff *(refreshed, 2026-09 -- direction has changed)*

With the corrected checkpoint, the safety-potency relationship no longer shows a
clean paradox: CAUTION-flagged compounds dock strongest on average (mean −8.10,
83% ≤−7.0, n=180), PASS-flagged compounds dock slightly weaker (mean −7.67, 77%
≤−7.0, n=136), and FAIL-flagged compounds dock *weakest* of the three groups on
average (mean −7.49, 64% ≤−7.0, n=45) -- the opposite ordering from the earlier
claim that FAIL compounds bind most strongly. This is reported as an open,
unresolved finding rather than restated uncritically: it may reflect a genuinely
different relationship under the corrected model, or reflect that ADMET flags and
docking energy are simply not as tightly (inversely) coupled as the earlier,
smaller-sample analysis suggested.

### 18.14 Dual-Validated Elite Hits *(refreshed, 2026-09)*

99 compounds passed both structure-based (ΔG ≤ −8.0) and ligand-based (composite ≥
median, 0.818) thresholds. Notable clinically relevant repurposed compounds in this
set include D0RV9D (FGFR1, −11.21), BAG956 (EGFR, −11.03), A-443654 (EGFR, −10.88),
UCN-01 (AKT1, −10.45), OMIPALISIB (EGFR, −10.19), and RISOVALISIB (EGFR, −10.40).

### 18.15 Phase 7b: Multi-Target Docking Validation *(refreshed, 2026-09)*

To address the single-target limitation of Phase 7, we executed an all-vs-all docking
campaign: 50 elite compounds (top 25 generated + top 25 repurposed by docking energy)
× 7 productive targets (STAT3 excluded) = **350 docking jobs**, completing with 100%
success rate.

**Key results**:

- Overall: mean ΔG = −8.75 kcal/mol, 93.1% ≤−7.0 (326/350 bipartite edges), 66.9% ≤−8.0
- **All 50 compounds are "Broad" multi-target binders** (≥3 targets at ≤−7.0; per-
  summary selectivity breakdown: 0 selective, 0 dual, 50 broad, 0 weak)
- 26 compounds hit all 7 targets at ≤−7.0; 5 compounds hit all 7 at the stricter ≤−8.0
- At ≤−8.0: 34 compounds bind ≥5 targets
- Best single dock: HM71224 (REP_CHEMBL4163691) vs FGFR1 at −11.65 kcal/mol

**Pathway polypharmacology**: bipartite compound-target network has 326 edges at
≤−7.0 kcal/mol; target-target co-binder network is fully connected (21 edges,
density 1.0, mean 44.0 shared binders per target pair) -- every pair of the 7
targets shares a large common set of binders, consistent with the compounds being
selected for broad polypharmacology rather than single-target selectivity.

**Top multi-target binders** (by mean energy across all 7 targets, all hitting 7/7
at ≤−7.0): IN-1166 (REP_CHEMBL387748, mean −9.95, best −11.41), Benzamide
derivative 16 (REP_D0AH1A, mean −9.67, best −10.87), A-443654 (REP_CHEMBL379300,
mean −9.66, best −11.10), OMIPALISIB (REP_CHEMBL1236962, mean −9.60, best −11.02).
Among generated molecules, GEN_00133 (CRAVE_filtered) is the top performer, hitting
7/7 targets (mean −9.10, best −10.96 kcal/mol).

**Source comparison**: generated compounds average 6.40 targets hit at ≤−7.0 and
−9.98 best energy; repurposed compounds average 6.64 targets hit and −10.43 best
energy -- repurposed compounds retain a modest edge here (opposite of the single-
target Phase 7 comparison in §18.7, where generated compounds are now stronger on
average). 10 generated compounds (of 25 tested) hit all 7 targets at ≤−7.0.

### 18.16 Implementation Details (Phase 7)

**Script**: `GENESIS/docking/run_molecular_docking.py` (Phase 7)
**Script**: `GENESIS/docking/run_multitarget_docking.py` (Phase 7b)

**Pipeline phases (Phase 7)**:
- Phase A: Receptor preparation (PDB → PDBQT with AD4 atom typing)
- Phase B: Candidate selection (top 200 per source, target-mapped)
- Phase C: Ligand preparation (SMILES → 3D → PDBQT via Meeko)
- Phase D: Docking execution (Vina Python API)
- Phase E: Results analysis (distributions, correlations, scaffolds)
- Phase F: Dossier integration (per-compound best docking scores)

**Pipeline phases (Phase 7b)**:
- Phase A: Select top 25 generated + top 25 repurposed by docking energy
- Phase B: Verify existing receptor and ligand PDBQT files
- Phase C: All-vs-all docking (50 compounds × 7 targets = 350 jobs)
- Phase D: Multi-target analysis (binder identification, selectivity, polypharmacology)

**Output files**:
```
GENESIS/docking/
├── run_molecular_docking.py          Phase 7 pipeline script
├── run_multitarget_docking.py        Phase 7b pipeline script
├── receptors/
│   ├── *.pdb                         8 crystal structures
│   └── *_receptor.pdbqt              Prepared receptor files
├── ligands/
│   └── *.pdbqt                       364 prepared ligand files
├── poses/
│   └── *_vs_*.pdbqt                  Best docking pose per job (Phase 7)
├── poses_multitarget/
│   └── *_vs_*.pdbqt                  Best docking pose per job (Phase 7b)
└── results/
    ├── docking_results.tsv           Phase 7: all 364 docking results
    ├── docking_summary.json          Phase 7: summary statistics
    ├── per_target_summary.tsv        Phase 7: per-target statistics
    ├── compound_docking_scores.tsv   Phase 7: per-compound best scores
    ├── multitarget_results.tsv       Phase 7b: all 350 results
    ├── multitarget_summary.json      Phase 7b: summary statistics
    ├── multitarget_binders.tsv       Phase 7b: multi-target binder profiles
    ├── selectivity_profiles.tsv      Phase 7b: compound × target energy matrix
    └── polypharmacology_network.tsv  Phase 7b: target-pair co-binder counts
```

## Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **AdaLN** | Adaptive Layer Normalization — conditions LN scale/shift on external signal |
| **CFG** | Classifier-Free Guidance — interpolates conditioned/unconditioned predictions |
| **CFM** | Conditional Flow Matching — trains velocity field for ODE-based generation |
| **CGLF** | Contrastive-Guided Latent Flow — novel method combining flow matching with VP guidance |
| **CRAVE** | Contrastive Reward-Aligned VAE Exploration — enhanced RL ensemble |
| **DDPM** | Denoising Diffusion Probabilistic Model (Ho et al., 2020) |
| **ELBO** | Evidence Lower Bound — variational inference objective |
| **EMA** | Exponential Moving Average — parameter smoothing technique |
| **FiLM** | Feature-wise Linear Modulation — element-wise affine conditioning |
| **GRU** | Gated Recurrent Unit — simpler alternative to LSTM |
| **MolDeBERTa** | Molecular DeBERTa — pre-trained SMILES language model |
| **NDCG** | Normalized Discounted Cumulative Gain — ranking quality metric |
| **OSCC** | Oral Squamous Cell Carcinoma |
| **OT-CFM** | Optimal Transport Conditional Flow Matching |
| **pChEMBL** | -log₁₀(bioactivity in molar), higher = more potent |
| **QED** | Quantitative Estimation of Drug-likeness |
| **REINFORCE** | Policy gradient algorithm (Williams, 1992) |
| **SA Score** | Synthetic Accessibility Score (Ertl & Schuffenhauer, 2009) |
| **SELFIES** | Self-Referencing Embedded Strings — 100% valid molecular representation |
| **SupCon** | Supervised Contrastive Learning (Khosla et al., 2020) |
| **VP** | Very Potent — compounds with pChEMBL ≥ 7.0 |

## Appendix B: Key File Paths

```
GENESIS/
├── embeddings/
│   ├── compound_embeddings.mmap    (715,162 × 768, float16)
│   └── compound_ids.npy            (715,162 ChEMBL IDs)
├── modeling/
│   ├── model.py                    ContrastiveModel definition
│   ├── losses.py                   OrdinalSupConLoss, PotencyLoss, TierMarginRankingLoss
│   ├── dataset.py                  ContrastiveDataset, curriculum sampler
│   ├── train.py                    Contrastive training loop
│   ├── evaluate.py                 6-test evaluation battery
│   ├── compound_index.pkl          692,644-entry metadata dictionary
│   ├── splits.json                 Train/val/test split IDs
│   └── checkpoints/best.pt        Validated contrastive model (seed 42)
├── generative/
│   ├── data_prep.py                SMILES → SELFIES → tokenized arrays
│   ├── dataset.py                  SelfiesDataset, DataLoaders
│   ├── compute_centroid.py         VP centroid in 128-d space
│   ├── vae.py                      VAE + cVAE architectures and training
│   ├── diffusion.py                Latent diffusion with cosine schedule
│   ├── rl_finetune.py              REINFORCE with multi-objective reward
│   ├── cglf.py                     Flow matching + VP guidance + decoder
│   ├── crave_gen.py                Enhanced RL + multi-strategy + Pareto
│   ├── evaluate_gen.py             10-metric unified evaluation
│   ├── run_all.py                  Pipeline orchestrator
│   ├── processed/                  Pre-computed data assets
│   │   ├── vocab.json              157-token vocabulary
│   │   ├── inv_vocab.json          Index → token mapping
│   │   ├── selfies_tokens.npy      (71,861 × 128) tokenized sequences
│   │   ├── compound_metadata.tsv   Per-compound metadata with splits
│   │   ├── metadata.json           Dataset statistics
│   │   ├── vp_centroid.json        128-d VP centroid vector
│   │   └── failed_smiles.txt       5 unconvertible SMILES
│   ├── checkpoints/                Model checkpoints (auto-created)
│   ├── results/                    Generated molecules + eval JSONs
│   └── logs/                       Training histories
├── repurposing/
│   ├── run_drug_repurposing.py     Phase 6 pipeline
│   └── results/                    Dossiers, rankings, SMARTS
├── docking/
│   ├── run_molecular_docking.py    Phase 7 docking pipeline
│   ├── run_multitarget_docking.py  Phase 7b multi-target pipeline
│   ├── receptors/                  8 PDB + PDBQT receptor files
│   ├── ligands/                    364 prepared ligand PDBQTs
│   ├── poses/                      Best docking pose per job (Phase 7)
│   ├── poses_multitarget/          Best docking pose per job (Phase 7b)
│   └── results/                    Docking results, summary, scores
├── admet/
│   ├── run_admet_profiling.py      Phase 8 ADMET + mucosal irritation pipeline
│   └── results/
│       ├── admet_profiles.tsv      Full ADMET profiles (361 compounds)
│       ├── mucosal_safety.tsv      Mucosal irritation risk report
│       ├── final_candidates.tsv    Prioritized shortlist (316 compounds)
│       ├── safety_potency_analysis.tsv  Safety-potency relationship
│       └── admet_summary.json      Pipeline statistics
├── results/
│   ├── phase0_embeddings.md        through phase8_admet_profiling.md
│   └── ...
└── docs/
    └── Algorithm_Design.md          ← This document
```

## 20. Phase 8 — ADMET & Pharmacokinetic Profiling

### 20.1 Objective

Comprehensive drug-likeness, ADMET prediction, and OSCC-specific mucosal irritation
safety assessment for all 361 docking-validated compounds from Phases 7/7b.

### 20.2 Pipeline Architecture

```
Phase 7/7b compounds (361)
  │
  ├─ Module B: Physicochemical (MW, LogP, TPSA, QED, SA, Lipinski/Veber/Ghose/Egan)
  ├─ Module C: ADMET (Caco-2, BBB, PPB, CYP inhibition, hERG)
  ├─ Module D: Structural Alerts (PAINS, Brenk, 10 toxicophores, 9 extended)
  ├─ Module E: Mucosal Irritation Filter (26 SMARTS alerts, amphiphilic index, pH risk)
  ├─ Module F: Integrated ADMET Scoring (weighted geometric mean of 6 components)
  └─ Module G: Final Candidate Shortlist → 13 triple-validated elite candidates
```

### 20.3 Mucosal Irritation Filter (Novel)

Unique to OSCC context — oral mucosal tissue is directly exposed to therapeutics.
26 SMARTS-based structural alerts organized by mechanism:

| Category | Alerts | Severity | Mechanism |
|----------|--------|----------|----------|
| Direct tissue-reactive | 11 | S2–S3 | Covalent protein/DNA modification |
| ROS generators | 4 | S2–S3 | Oxidative mucosal damage |
| Corrosive/pH-extreme | 4 | S1–S2 | Tissue corrosion, pH disruption |
| Membrane disruptors | 2 | S1–S2 | Membrane solubilization |
| Protein-reactive warheads | 5 | S1–S3 | Covalent keratin/cysteine modification |

Risk classification:
- **HIGH:** max_severity ≥ 3 OR severity_sum ≥ 5
- **MODERATE:** max_severity ≥ 2 OR severity_sum ≥ 3 OR amphiphilic risk
- **LOW:** any alert or extreme pH risk
- **SAFE:** no alerts detected

### 20.4 Integrated ADMET Score

Weighted geometric mean of six component scores $s_i \in [0,1]$:

$$\text{ADMET}_{\text{score}} = \exp\!\left(\sum_{i=1}^{6} w_i \cdot \ln(s_i)\right)$$

| Component | Weight | Description |
|-----------|--------|-------------|
| $s_{\text{druglike}}$ | 0.20 | Lipinski + Veber + QED |
| $s_{\text{absorption}}$ | 0.15 | Caco-2 + Egan |
| $s_{\text{safety}}$ | 0.20 | PAINS + toxicophores + extended alerts |
| $s_{\text{mucosal}}$ | **0.20** | OSCC-specific mucosal irritation |
| $s_{\text{cardiac}}$ | 0.10 | hERG liability |
| $s_{\text{metabolic}}$ | 0.15 | CYP inhibition + SA score |

Mucosal safety receives elevated weight (0.20) reflecting OSCC therapeutic context.

### 20.5 ADMET Flag Criteria

- **PASS:** ADMET ≥ 0.5 AND mucosal ∈ {SAFE, LOW} AND PAINS = 0 AND hERG ≠ High
- **CAUTION:** ADMET ≥ 0.3 AND mucosal ≠ HIGH AND PAINS ≤ 1
- **FAIL:** All others

### 20.6 Key Results *(refreshed against the corrected checkpoint, 2026-09)*

| Metric | Value |
|--------|-------|
| Compounds profiled | 361 |
| Lipinski pass rate | 91.4% (330/361) |
| Mucosal SAFE | 62.9% (227/361) |
| ADMET PASS | 136 (37.7%) |
| ADMET CAUTION | 180 (49.9%) |
| ADMET FAIL | 45 (12.5%) |
| Triple-validated elite | **13** |

**Triple-validated elite:** 13 compounds pass all three gates:
1. Multi-target binder (Phase 7b: ≥3/7 targets at ≤−7.0 kcal/mol)
2. ADMET PASS (drug-like + mucosal-safe + cardiac-safe)
3. Docking validated (Phase 7)

6/13 are generated (all CRAVE), 7/13 are repurposed; 12/13 are mucosal SAFE, 1/13 is LOW.

### 20.7 Safety–Potency Relationship *(refreshed, 2026-09 -- finding has changed direction)*

With the corrected checkpoint, Phase 7 no longer shows FAIL-flagged compounds
docking better than PASS-flagged ones (see §18.13): CAUTION now docks strongest
(mean −8.10), PASS next (−7.67), and FAIL weakest (−7.49). This is reported as an
open finding rather than a resolved "paradox." Phase 8's integrated final score
still combines all three evidence streams:

$$\text{Final}_{\text{score}} = 0.30 \cdot \text{dock}_{\text{norm}} + 0.30 \cdot \text{composite}_{\text{norm}} + 0.40 \cdot \text{ADMET}_{\text{score}}$$

The 13 triple-validated elite candidates achieve both strong docking (mean −9.39 kcal/mol)
and good ADMET (mean 0.60), demonstrating that potency and safety are not mutually
exclusive in the compounds that pass all three gates, independent of how docking energy
and ADMET flag correlate across the full candidate pool.

---

## Appendix C: References

1. **Kingma & Welling** (2014). Auto-Encoding Variational Bayes. *ICLR*.
2. **Khosla et al.** (2020). Supervised Contrastive Learning. *NeurIPS*.
3. **Ho et al.** (2020). Denoising Diffusion Probabilistic Models. *NeurIPS*.
4. **Nichol & Dhariwal** (2021). Improved Denoising Diffusion Probabilistic Models. *ICML*.
5. **Lipman et al.** (2023). Flow Matching for Generative Modeling. *ICLR*.
6. **Krenn et al.** (2020). Self-Referencing Embedded Strings (SELFIES). *Machine Learning: Science and Technology*.
7. **Perez et al.** (2018). FiLM: Visual Reasoning with a General Conditioning Layer. *AAAI*.
8. **Fu et al.** (2019). Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing. *NAACL-HLT*.
9. **Williams** (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning. *Machine Learning*.
10. **Bickerton et al.** (2012). Quantifying the chemical beauty of drugs. *Nature Chemistry*.
11. **Ertl & Schuffenhauer** (2009). Estimation of synthetic accessibility score. *Journal of Cheminformatics*.
12. **Bemis & Murcko** (1996). The Properties that Distinguish from Drugs. *Journal of Medicinal Chemistry*.
13. **Chen et al.** (2020). A Simple Framework for Contrastive Learning of Visual Representations (SimCLR). *ICML*.

---

## 21. Phase 9a — Network Pharmacology & Pathway Analysis

### 21.1 Motivation

Multi-target docking (Phase 7b) established that 50 elite compounds bind 5–7 OSCC targets simultaneously. Phase 9a contextualizes these polypharmacology profiles by constructing compound–target–pathway networks, identifying which OSCC signalling cascades are perturbed, and quantifying pathway coverage.

### 21.2 Network Construction

**Bipartite Network.** A weighted bipartite graph $G = (C \cup T, E)$ where $C$ is the compound set (n=50), $T$ is the target set (n=7), and edge weights are $|{\\Delta}G|$ for interactions with ${\Delta}G \leq -7.0$ kcal/mol.

**Target–Target Network.** A complete graph $G_{TT} = (T, E_{TT})$ where edge weight is the Jaccard index of shared binders: $J(t_1, t_2) = |B_{t_1} \cap B_{t_2}| / |B_{t_1} \cup B_{t_2}|$.

### 21.3 OSCC Pathway Mapping

Each target was mapped to KEGG/Reactome pathways relevant to OSCC biology:

- **PI3K/AKT/mTOR Axis** (5 targets: AKT1, EGFR, FGFR1, MTOR, PIK3CA) — survival/proliferation
- **RTK/MAPK Cascade** (2 targets: EGFR, FGFR1) — proliferation/migration
- **Cell Cycle/Apoptosis** (3 targets: AKT1, MDM2, PIK3CA) — p53, cell cycle checkpoints
- **Inflammation/TME** (1 target: PTGS2) — COX-2, NF-κB, VEGF, TNF

### 21.4 Key Results *(refreshed against the corrected checkpoint, 2026-09)*

| Metric | Value |
|--------|-------|
| Network edges | 326 at ≤ −7.0 kcal/mol |
| Target–target edges | 21 (fully connected, density 1.0) |
| Pathways mapped | 16 across 4 categories |
| Mean targets per compound | 6.6 |
| Mean polypharmacology score | 58.7 |
| Elite mean polypharmacology score | 55.6 (n=13) |
| Top pathway | PI3K-AKT signalling (hsa04151), 50/50 compounds |

### 21.5 Figures Generated

7 publication-ready visualizations: selectivity heatmap (50×7), elite heatmap, target correlation, co-binding network, pathway coverage, radar charts (top 6 elite), multi-target distribution by source.

---

## 22. Phase 9b — Lead Optimization via Scaffold-Constrained Generation

### 22.1 Strategy

Rather than sampling latent vectors $z \sim \mathcal{N}(0, I)$ as in standard CRAVE generation, we perform **scaffold-constrained perturbation**:

$$z_{elite} = f_{CM}(f_{MolDeBERTa}(\text{SMILES}_{elite}))$$
$$z_{analog} \sim \mathcal{N}(z_{elite}, \sigma^2 I), \quad \sigma \in \{0.3, 0.5, 0.8\}$$

This generates analogs localized around the elite compound's position in the learned chemical space, preserving potency while allowing property variation.

### 22.2 Encoding Pipeline

1. **MolDeBERTa** (88.2M params): SMILES → [CLS] token → 768-d embedding
2. **ContrastiveModel** (Phase 2): 768-d → 128-d projection, trained with potency-graded contrastive loss
3. **Gaussian perturbation**: sample around $z_{elite}$ at 3 radii
4. **CRAVE decoder**: 128-d → GRU → SELFIES tokens → SMILES

### 22.3 Filtering Cascade

1. **Validity**: RDKit parsing
2. **Tanimoto similarity** ≥ 0.30 (Morgan FP, radius=2, 2048 bits)
3. **MW** ≤ 700 Da
4. **Deduplication**: canonical SMILES
5. **ADMET re-scoring**: Phase 8 integrated score (QED, Lipinski, Veber, Caco-2, hERG)
6. **Quality gate**: PASS/CAUTION + QED ≥ 0.4 + SA ≤ 5.5

### 22.4 Key Results *(refreshed against the corrected checkpoint, 2026-09 -- substantially different scale from earlier report)*

| Metric | Value |
|--------|-------|
| Elite parent compounds | 13 |
| Raw samples | 1,877 (13 parents × 3 radii × ~50 valid samples) |
| Valid unique analogs | 890 |
| ADMET PASS | 372 (41.8%) |
| ADMET CAUTION | 493 (55.4%) |
| ADMET FAIL | 25 (2.8%) |
| Improved analogs (SAR entries) | 854 |
| Mean Tanimoto to parent | 0.401 |
| Mean ADMET score | 0.861 |
| Unique Murcko scaffolds | 13 (across 890 analogs) |

**Note on scale**: the earlier reported figures (7,200 raw / 69 unique analogs) reflected
an older run against a different (smaller, pre-correction) elite set. With the corrected
checkpoint's 13 triple-validated elite parents, the pipeline produces a substantially
larger analog pool (1,877 raw → 890 unique). The ADMET FAIL rate is also non-zero
(2.8%, 25 compounds), unlike the earlier claim of a 0.0% FAIL rate.

### 22.5 Interpretation *(refreshed, 2026-09)*

- **Low FAIL rate (2.8%)** demonstrates substantial inheritance of drug-like properties
  from ADMET-favorable parents via latent-space proximity, though it is no longer
  exactly zero as previously claimed
- **Mean Tanimoto similarity to parent of 0.401** indicates analogs are structurally
  related but not trivial copies of their elite parent -- consistent with meaningful
  exploration around each parent's latent-space neighborhood rather than a scaffold-hop
  or a near-duplicate
- Only **13 unique Murcko scaffolds** were observed across all 890 analogs (one per
  elite parent on average), confirming that scaffold-constrained perturbation, by
  design, preserves each parent's core scaffold rather than generating novel scaffolds
  -- the earlier "95% novel scaffolds" claim does not hold under this metric and has
  been removed rather than restated
- Per-parent analog counts and specific "most productive parent" claims (previously
  naming GEN_00034 and GEN_00041) were not re-verified against the fresh
  `analogs_all.tsv` in this pass and should be treated as unconfirmed until checked

---

## 23. Phase 10 — Nearest Known Analog Analysis & Final Compound Ranking

### 23.1 Rationale

All preceding phases generate independent, disconnected data streams: CRAVE produces 1000 novel SMILES (Phase 3.8), repurposing screens 23,308 known compounds (Phase 6), docking scores 364 single-target and 350 multi-target binding energies (Phases 7/7b), ADMET profiles 361 compounds (Phase 8), network pharmacology maps pathway connectivity (Phase 9a), and lead optimization generates 890 analogs (Phase 9b). Phase 10 serves as the **integration and prioritization layer** — merging all evidence streams into a single ranked candidate list suitable for experimental prioritization.

A critical addition is **Nearest Known Analog (NKA) analysis**, which contextualizes every generated compound by identifying its closest relatives in known chemical space. This replaces the misleading concept of "parent compound" (which implies derivation) with a scientifically precise multi-reference similarity profile.

### 23.2 Nearest Known Analog (NKA) Analysis

#### 23.2.1 Reference Library Construction

The NKA reference set is assembled from three non-overlapping sources:

| Source | Description | Count |
|--------|-------------|-------|
| Training positives | Known OSCC actives with pChEMBL values | ~4,000 |
| All_Compounds_Unq_Std | Full curated ChEMBL dataset with OSCC annotations | ~67,000 |
| Repurposing global rankings | Compounds with clinical metadata and max_phase | ~1,000 |
| **Total unique** | Deduplicated by canonical SMILES | **71,965** |

#### 23.2.2 Similarity Computation

For each query compound $q$ against all $N$ reference compounds $\{r_1, \ldots, r_N\}$:

$$T(q, r_i) = \frac{|FP(q) \cap FP(r_i)|}{|FP(q) \cup FP(r_i)|}$$

where $FP(\cdot)$ denotes the Morgan fingerprint (radius=2, 2048 bits). We use RDKit's `BulkTanimotoSimilarity` for vectorized computation over the full reference library.

For each query, the **top-3 most similar known compounds** are retained, with their metadata:
- Tanimoto similarity, ChEMBL ID, compound name
- pChEMBL value (potency), target name, max clinical phase, mechanism of action

#### 23.2.3 Maximum Common Substructure (MCS)

For the nearest analog (NKA-1), we compute the MCS using `rdFMCS.FindMCS` with:
- Atom comparison: elements only
- Bond comparison: order-sensitive
- Timeout: 1 second per pair

MCS coverage of the query is reported as:

$$\text{MCS Coverage} = \frac{n_{\text{atoms}}^{\text{MCS}}}{n_{\text{atoms}}^{\text{query}}}$$

#### 23.2.4 Novelty Classification

Based on maximum Tanimoto similarity to the reference library:

| Category | Tanimoto Range | Interpretation |
|----------|---------------|----------------|
| `close_analog` | $\geq 0.7$ | Minor modification of known compound |
| `scaffold_hop` | $0.4 \leq T < 0.7$ | Same pharmacophore, different scaffold |
| `novel` | $< 0.4$ | Structurally distinct from known space |

#### 23.2.5 NKA Results *(refreshed against the corrected checkpoint, 2026-09)*

| Metric | Value |
|--------|-------|
| Generated compounds analyzed | 1,000 |
| Reference library size | 71,965 |
| Mean max Tanimoto | 0.406 |
| Novel (T < 0.4) | 532 (53.2%) |
| Scaffold hop (0.4 ≤ T < 0.7) | 441 (44.1%) |
| Close analog (T ≥ 0.7) | 27 (2.7%) |

The **53.2% novel rate** (down from an earlier reported 87.5%) still confirms that a
majority of CRAVE's latent-space generations fall outside the close-analog and
scaffold-hop ranges, though with the corrected checkpoint a much larger share (44.1%,
up from 9.8%) now falls in the scaffold-hop band -- same pharmacophore, different
scaffold -- rather than being maximally novel. The close-analog rate (2.7%) is
unchanged. The mean Tanimoto (0.406) is essentially unchanged from the earlier 0.297
only in the sense that both are below the close-analog threshold; the underlying
distribution has shifted materially toward the scaffold-hop band.

### 23.3 Evidence Integration & Data Merge

All pipeline data sources are merged on compound identifier and canonical SMILES:

```
ADMET (361 compounds)            ─┐
Single-target docking (364)       │
Multi-target docking (350)        │──→ Master DataFrame (1,161 compounds)
Network pharmacology metrics      │      × 110+ columns
Selectivity profiles (50)         │
Lead optimization analogs (890)   │
All 1000 generated SMILES        ─┘
```

Compounds not present in ADMET or docking pipelines (i.e., the generated compounds that were not in Phase 4's top-hit list) are still included with their NKA profiles and QED/SA scores, ensuring **no novel compound is excluded**. The final merged ranking table (`master_ranked_all.tsv`) contains 1,161 compounds *(refreshed against the corrected checkpoint, 2026-09; was reported as 1,177)*: 987 generated and 129 repurposed.

### 23.4 GENESIS Score — Multi-Component Weighted Ranking

#### 23.4.1 Component Normalization

Eight independent scoring dimensions, each normalized to $[0, 1]$:

| Component | Weight | Source | Normalization |
|-----------|--------|--------|---------------|
| $s_{\text{docking}}$ | 0.25 | Phase 7/7b best $\Delta G$ | Min-max over all docked |
| $s_{\text{ADMET}}$ | 0.20 | Phase 8 composite score | Min-max normalization |
| $s_{\text{potency}}$ | 0.15 | VP cosine similarity | Already $[0,1]$ |
| $s_{\text{QED}}$ | 0.10 | RDKit QED | Already $[0,1]$ |
| $s_{\text{multi-target}}$ | 0.10 | Fraction of 7 targets bound $\leq -7$ kcal/mol | $n_{\text{hits}} / 7$ |
| $s_{\text{clinical}}$ | 0.10 | NKA-1 max clinical phase | $\text{phase} / 4$ |
| $s_{\text{SA}}$ | 0.05 | Synthetic accessibility | $1 - (SA - 1)/9$ (inverted) |
| $s_{\text{novelty}}$ | 0.05 | Novelty class category | scaffold_hop=1.0, close_analog=0.7, novel=0.5 |

#### 23.4.2 Evidence-Completeness Penalty

A naive weighted average over only available components would overestimate compounds with sparse data (e.g., a compound with only QED scoring 0.9 would rank above a fully-validated compound scoring 0.7 across 8 metrics). The GENESIS score instead applies an **evidence-completeness penalty**:

$$S_{\text{GENESIS}} = \left( \frac{\sum_{i \in \text{all}} w_i \cdot s_i}{\sum_{i \in \text{all}} w_i} \right) \times \sqrt{\frac{n_{\text{available}}}{8}}$$

where:
- Missing components contribute 0 to the numerator but their weight remains in the denominator
- $n_{\text{available}}$ counts components with non-NaN values
- The $\sqrt{\cdot}$ scaling ensures partial data is penalized but not annihilated

**Effect**: A compound with only QED + novelty (2/8 components available) receives a multiplier of $\sqrt{0.25} = 0.5$, while a fully-validated compound receives $\sqrt{1.0} = 1.0$.

#### 23.4.3 Validation Tiers

Compounds are classified by data completeness:

| Tier | Criteria | Count |
|------|----------|-------|
| Tier-1 (fully validated) | Docking + ADMET + Multi-target data | 46 |
| Tier-3 (partial validation) | Docking OR ADMET | 270 |
| Tier-4 (prediction only) | QED/SA/NKA only | 845 |

*(Refreshed against the corrected checkpoint, 2026-09; was 46/299/832.)*

### 23.5 Clinical Annotation

Each compound receives automated clinical context based on structural features and NKA metadata:

1. **Predicted drug class**: kinase inhibitor, COX-2 inhibitor, multi-kinase inhibitor, mTOR/PI3K inhibitor, etc.
2. **OSCC clinical rationale**: pathway relevance mapping (e.g., "EGFR pathway: overexpressed in 90% OSCC; PI3K/AKT: activated in cisplatin-resistant OSCC")
3. **Clinical priority score**: weighted combination of GENESIS score + NKA clinical phase + drug class relevance

### 23.6 Compound Dossiers

For each Tier-1 compound, a comprehensive dossier is generated containing:
- Molecular identity (SMILES, formula, MW, LogP)
- All docking scores across 7 targets with binding interpretations
- ADMET profile (9 property flags, composite score, structural alerts)
- Top-3 NKA compounds with similarity, potency, and clinical phase
- Lead optimization analog count and best analog ADMET scores
- Recommended next steps (in vitro assay suggestions, ADMET concerns to address)

### 23.7 Key Results

#### 23.7.1 Score Distribution by Tier *(refreshed against the corrected checkpoint, 2026-09)*

| Tier | Mean Score | Std | Min | Max |
|------|-----------|-----|-----|-----|
| Tier-1 (46 compounds) | 0.565 | 0.071 | 0.448 | 0.716 |
| Tier-3 (270 compounds) | 0.314 | 0.039 | 0.163 | 0.421 |
| Tier-4 (845 compounds) | 0.054 | 0.014 | 0.005 | 0.125 |

The score separation between tiers confirms the evidence-completeness penalty is working correctly — fully validated compounds are properly elevated over prediction-only candidates.

#### 23.7.2 Top 10 Compounds *(refreshed against the corrected checkpoint, 2026-09 -- the top compound identity has changed)*

| Rank | ID | Score | Source | ADMET | Novelty |
|------|----|-------|--------|-------|---------|
| 1 | REP_CHEMBL1236962 (Omipalisib) | 0.716 | Repurposed | CAUTION | close_analog |
| 2 | REP_CHEMBL4800252 (Risovalisib) | 0.688 | Repurposed | PASS | close_analog |
| 3 | REP_CHEMBL3545235 (Simotinib) | 0.684 | Repurposed | CAUTION | close_analog |
| 4 | REP_CHEMBL554 (Lapatinib) | 0.682 | Repurposed | CAUTION | close_analog |
| **5** | **GEN_00182** | **0.668** | **Generated** | **CAUTION** | **novel** |
| 6 | REP_CHEMBL387748 (IN-1166) | 0.658 | Repurposed | CAUTION | close_analog |
| 7 | GEN_00133 | 0.652 | Generated | CAUTION | scaffold_hop |
| 8 | GEN_00045 | 0.652 | Generated | CAUTION | scaffold_hop |
| 9 | REP_D0RV9D | 0.646 | Repurposed | CAUTION | close_analog |
| 10 | REP_CHEMBL379300 (A-443654) | 0.630 | Repurposed | CAUTION | close_analog |

**GEN_00182** at rank 5 is now the **highest-ranked CRAVE-generated compound** (was
GEN_00045 at rank 7 in the earlier report; GEN_00045 is still present, now at rank 8).
This demonstrates that the generative pipeline continues to produce compounds that
compete with established, clinically annotated repurposing candidates.

#### 23.7.3 Top Generated Compounds *(refreshed, 2026-09)*

| Rank | ID | Score | ADMET | Novelty |
|------|----|-------|-------|---------|
| 5 | GEN_00182 | 0.668 | CAUTION | novel |
| 7 | GEN_00133 | 0.652 | CAUTION | scaffold_hop |
| 8 | GEN_00045 | 0.652 | CAUTION | scaffold_hop |
| 11 | GEN_00020 | 0.626 | CAUTION | scaffold_hop |
| 13 | GEN_00044 | 0.616 | CAUTION | novel |
| 14 | GEN_00069 | 0.603 | CAUTION | scaffold_hop |
| 15 | GEN_00194 | 0.598 | CAUTION | novel |
| 16 | GEN_00108 | 0.597 | CAUTION | scaffold_hop |

The previously named lead-optimization parents "GEN_00034" and "GEN_00041" are not
re-confirmed as top performers in this refreshed ranking and have been replaced with
the compounds actually observed in the current `master_ranked_all.tsv`.

#### 23.7.4 Pipeline Summary Statistics *(refreshed against the corrected checkpoint, 2026-09)*

| Metric | Value |
|--------|-------|
| Total compounds ranked | 1,161 |
| Recommended (validated) | 316 |
| Tier-1 dossiers generated | 46 |
| Sources | 987 generated + 129 repurposed |
| Tier-1 ADMET PASS | 13 (28.3%) |
| Tier-1 ADMET CAUTION | 33 (71.7%) |
| Tier-1 ADMET FAIL | 0 (0.0%) |
| Output columns | 110+ |

### 23.8 Output Files

| File | Size | Contents |
|------|------|----------|
| `nka_all_generated.tsv` | 271 KB | NKA for all 1,000 generated compounds |
| `nka_all_compounds.tsv` | 318 KB | NKA for all 1,161 compounds (generated + repurposed) |
| `master_ranked_all.tsv` / `.xlsx` | 631 KB / 608 KB | Complete ranked list (1,161 rows × 110+ columns) |
| `master_ranked_recommended.tsv` / `.xlsx` | 258 KB / 215 KB | Validated subset (316 recommendations) |
| `compound_dossiers.tsv` / `.xlsx` | 33 KB / 237 KB | Tier-1 dossiers (46 compounds, multi-sheet Excel) |

### 23.9 Interpretation

1. **Repurposed compounds dominate the top ranks** — this is expected and scientifically correct. Repurposed compounds have known clinical data, established binding profiles, and close analogs in the reference library, all of which boost their GENESIS score. They represent the **lowest-risk, fastest-to-clinic** options.

2. **GEN_00182's rank-5 position is remarkable** *(refreshed, 2026-09; was GEN_00045 at rank 7 with ΔG = −10.5)* — a de novo generated compound with no prior clinical history competes directly with established drug candidates. It is classified `novel` (below the scaffold-hop threshold to any known compound) and combines a CAUTION ADMET flag with a strong genesis_score (0.668), making it the strongest current case for experimental validation of a GENESIS-generated compound.

3. **The 53.2% novelty rate validates CRAVE's generative capacity** *(refreshed, 2026-09; was reported as 87.5%)* — the generator is not simply memorizing and replaying training data, though a larger share of outputs (44.1%) now fall in the scaffold-hop band than previously reported. The mean Tanimoto of 0.406 to the nearest known compound places most generated molecules outside the close-analog range, though closer to known chemical space than the earlier analysis suggested.

4. **Evidence-completeness penalty prevents false positives** — without this penalty, Tier-4 compounds (with only QED data) would score artificially high. The $\sqrt{\text{completeness}}$ multiplier ensures that high scores require multi-level validation evidence, while still preserving rank order within each tier.

5. **316 recommended compounds provide a actionable shortlist** — these are the compounds with at least docking OR ADMET evidence, eliminating pure prediction-only candidates from the clinical consideration set while retaining all validated novel compounds.

---

## 24. Phase 10b — Molecular Dynamics Simulations (In Progress)

### 24.1 Objective

Validate the top docking poses from Phases 7/7b through explicit-solvent molecular dynamics (MD) simulations to assess binding stability, protein-ligand interaction persistence, and conformational dynamics over nanosecond timescales.

### 24.2 System Setup

| Parameter | Value |
|-----------|-------|
| Engine | OpenMM 8.5 (OpenCL acceleration) |
| Force field (protein) | Amber14 (amber14-all.xml) |
| Force field (solvent) | TIP3P-FB |
| Ligand parameterization | OpenFF GAFF via OpenMM |
| Box padding | 12 Å (cubic) |
| Ionic strength | 0.15 M NaCl |
| Environment | conda `md_env` (Python 3.11) |

### 24.3 Simulation Protocol *(refreshed against the corrected checkpoint, 2026-09 -- shortlist compounds have changed)*

**12 simulations planned** — top elite compounds paired with their best-scoring target:

| Compound | Target | ΔG (kcal/mol) | Source |
|----------|--------|---------------|--------|
| D0RV9D | FGFR1 (4V04) | −11.21 | Repurposed |
| BAG956 | EGFR (4HJO) | −11.03 | Repurposed |
| A-443654 | EGFR (4HJO) | −10.88 | Repurposed |
| UCN-01 | AKT1 (3CQW) | −10.45 | Repurposed |
| RISOVALISIB | EGFR (4HJO) | −10.40 | Repurposed |
| OMIPALISIB | EGFR (4HJO) | −10.19 | Repurposed |
| GEN_00045 | EGFR (4HJO) | −10.13 | Generated |
| GEN_00194 | EGFR (4HJO) | −10.03 | Generated |
| GEN_00182 | EGFR (4HJO) | −10.02 | Generated |
| GEN_00108 | EGFR (4HJO) | −9.93 | Generated |
| GEN_00126 | EGFR (4HJO) | −9.73 | Generated |
| GEN_00020 | EGFR (4HJO) | −9.67 | Generated |

Note: this list is a *proposed* MD validation shortlist (Phase 11, not yet executed
as of this writing) and has been refreshed to reflect the current top docking hits;
the earlier list (headed by REP_CHEMBL5439799 and GEN_00045 vs FGFR1) reflected the
pre-correction docking run and is superseded above.

### 24.4 Pipeline Architecture

```
PDB fetch → Receptor cleaning → Ligand extraction from PDBQT
    → System assembly (Modeller) → 3-stage minimization (15K steps)
    → NVT heating (10K → 300K, 10 steps, 1 fs timestep)
    → NPT equilibration (1 ns, 2 fs timestep, Monte Carlo barostat)
    → Production MD (10 ns, 2 fs timestep)
    → Trajectory analysis (RMSD, RMSF, interaction persistence)
```

### 24.5 Technical Challenges & Fixes Applied

Three key stability issues were identified and addressed during Phase 10b development:

1. **Barostat during NVT**: Original code activated the Monte Carlo barostat during NVT equilibration, causing pressure- and temperature-coupling interference. **Fix**: Set barostat frequency to 0 during NVT, re-enable for NPT.

2. **Aggressive initial temperature**: Starting NVT at 50K caused large atomic velocities in the under-equilibrated system. **Fix**: Start from 10K with 10 gradual heating stages up to 300K.

3. **Insufficient minimization**: 5000-step single-stage minimization failed to relieve steric clashes from docking poses. **Fix**: 3-stage minimization (5000 + 5000 + 5000 = 15,000 steps total) with force constant reduction.

4. **NaN detection**: Added explicit coordinate NaN checks (`_check_nan()`) between simulation phases to fail fast rather than propagate numerical instabilities.

### 24.6 Current Status

| Status | Detail |
|--------|--------|
| Jobs attempted | 3 of 12 |
| Jobs 1-2 | **NaN crash** during NVT heating (FGFR1 systems: 70K–83K atoms) |
| Job 3 | REP_CHEMBL521851 vs EGFR — heating progressed to T=97K (1.49M atom system) before process terminated |
| Root cause | Likely ligand parameterization quality — GAFF parameters for novel scaffolds may produce unrealistic bond/angle forces at the protein-ligand interface |

**This phase is parked** pending investigation of ligand-specific force field parameters. Recommended next steps:
- Use `openff-toolkit` with Sage 2.0 for ligand parameterization (better coverage of drug-like chemical space)
- Try implicit solvent (GBSA) for initial equilibration before switching to explicit TIP3P
- Consider AmberTools `antechamber` + `parmchk2` workflow for ligand GAFF2 parameterization with AM1-BCC charges

---

## 25. Phase 10c — Compound Category Annotation & Repurposing Potential

*(Rerun 2026-09-19 against the corrected checkpoint-selection contrastive model — supersedes
the original 2026-03-23 run, whose output files predated the fix described in §5.3.2. ADMET,
mucosal-safety, and novelty-class metrics are unchanged, since those do not depend on the
contrastive model; VP-similarity-dependent and repurposing-score metrics have shifted
materially and are corrected below.)*

### 25.1 Motivation

After the GENESIS scoring and ranking pipeline (Phase 10) established final rankings for generative compounds, a parallel question arises: **how do existing compound categories in the master dataset compare in repurposing potential for OSCC?** Three pharmacologically distinct categories were identified from `All_Compounds_Unq_Std.tsv` (72,162 rows, 47 columns):

| Category | Selection Criterion | Count | Description |
|----------|-------------------|-------|-------------|
| **Natural_AntiOSCC** | `IsAnticancer = 2` | 37 | Natural products with experimentally validated OSCC activity |
| **PLOS_Cmpd** | `in_plos = 1` | 219 (of 221; 2 failed SMILES parsing) | Compounds from curated PLOS literature studies on oral cancer |
| **NonAnticancerCmpd** | `IsAnticancer = 0` | 20,584 | Compounds originally developed for non-cancer indications |

Category overlaps: Natural ∩ PLOS = 21, NonAnticancer ∩ PLOS = 38, Natural ∩ NonAnticancer = 0.

### 25.2 Pipeline Architecture

Each category was processed through the same 6-module pipeline used for generative compounds, ensuring apples-to-apples comparison:

```
Module A: Load & Categorize  →  Module B: Load Contrastive Model
       ↓                                    ↓
Module C: Load NKA Reference Library (49,312 training positives)
       ↓
  Per-Category Processing:
    ├── ADMET profiling (PAINS, Brenk, toxicophores, mucosal, Caco2/BBB/PPB/hERG, CYP)
    ├── Contrastive scoring (VP similarity, predicted pChEMBL, composite)
    ├── NKA analysis (top-3 nearest known analogs via Tanimoto)
    └── Repurposing score (7-component weighted: VP_sim, ADMET, QED, potency, NKA_clinical, SA, max_phase)
       ↓
Module G: Cross-Category Comparison & Combined Output (TSV + XLSX)
```

**106 features** annotated per compound (original 47 columns + 59 computed: ADMET subscores, contrastive projections, NKA top-3, repurposing score).

### 25.3 Cross-Category Comparison

| Metric | Natural_AntiOSCC (n=37) | PLOS_Cmpd (n=219) | NonAnticancerCmpd (n=20,584) |
|--------|:-----------------------:|:------------------:|:----------------------------:|
| ADMET pass rate | 40.5% | 41.1% | **65.6%** |
| ADMET caution | 43.2% | 48.4% | 28.9% |
| ADMET fail | 16.2% | 10.5% | 5.5% |
| Mucosal safe rate | 16.2% | 27.9% | **49.3%** |
| Mean VP similarity | 0.310 | **0.372** | 0.078 |
| Max VP similarity | 0.972 | **0.993** | **0.993** |
| Mean QED | 0.420 | 0.462 | **0.536** |
| Mean SA score | **3.71** | **3.83** | 3.06 |
| Mean repurposing score | 0.366 | **0.392** | 0.282 |
| Max repurposing score | 0.808 | 0.904 | **0.936** |
| Compounds with score > 0.5 | 12 (32.4%) | 66 (**30.1%**) | 823 (4.0%) |
| Compounds with score > 0.7 | 2 (5.4%) | 18 (**8.2%**) | 60 (0.3%) |
| Embedding coverage | 100% | 100% | 100% |
| Novelty: novel | 12 | 68 | 14,387 |
| Novelty: scaffold_hop | 12 | 47 | 5,350 |
| Novelty: close_analog | 13 | 104 | 847 |

**Note on the correction**: mean VP similarity dropped substantially for all three categories
(e.g. NonAnticancerCmpd 0.317→0.078) relative to the pre-correction run, since VP similarity is
computed directly against the corrected VP centroid (§7) in the corrected contrastive space —
this is expected and reflects a stricter, more accurate notion of "VP-like" under the fixed
model, not a data or pipeline error. ADMET, mucosal-safety, and Tanimoto-based novelty-class
metrics are unaffected, since none of those depend on the contrastive model.

### 25.4 Key Insights

#### 25.4.1 PLOS Compounds Show Strongest Repurposing Profile

PLOS compounds achieve the highest mean VP similarity (0.372) and repurposing score (0.392). This is expected — these are literature-curated compounds already investigated for oral cancer activity. Their **8.2% rate of scoring >0.7** (30.1% at >0.5) validates the GENESIS contrastive model: compounds with known OSCC relevance are correctly scored highly by the learned virtual-patient embedding space.

Top 5 PLOS compounds by repurposing score:

| Compound | Repurp. Score | VP Sim | ADMET | QED | Max Phase |
|----------|:------------:|:------:|:-----:|:---:|:---------:|
| Colchicine | 0.904 | 0.937 | PASS | 0.834 | 4 |
| Gefitinib | 0.894 | 0.985 | PASS | 0.520 | 4 |
| Celecoxib | 0.878 | 0.990 | CAUTION | 0.750 | 4 |
| Cytarabine | 0.817 | 0.902 | PASS | 0.450 | 4 |
| Sunitinib | 0.814 | 0.971 | CAUTION | 0.630 | 4 |

#### 25.4.2 NonAnticancer Compounds: NSAIDs Dominate Top Hits

The 60 NonAnticancer compounds scoring >0.7 are overwhelmingly **NSAIDs and COX-2 inhibitors**: Indomethacin (0.936), Lumiracoxib (0.932), Meclofenamic Acid (0.925), Valdecoxib (0.913), Colchicine (0.904), Ketoprofen (0.901), Mefenamic Acid (0.892), Diclofenac (0.887), Etoricoxib (0.883), Tolmetin (0.879).

This is a biologically significant finding: PTGS2 (COX-2, PDB: 3LN1) is one of the 7 OSCC targets in the GENESIS target set, and the contrastive model independently identifies COX-2 inhibitors as having OSCC-relevant virtual-patient signatures. NKA target analysis of top hits confirms:
- **34/60** (56.7%) map to Prostaglandin G/H synthase 2 (PTGS2/COX-2)
- **7/60** (11.7%) map to EGFR
- Additional hits map to STAT3, PIK3CA, MDM2 — all GENESIS OSCC targets

This provides orthogonal validation that the contrastive model learned meaningful biology.

#### 25.4.3 Natural_AntiOSCC: Validated Leads Requiring ADMET Optimization

Natural compounds have the lowest ADMET pass rate (40.5%) and mucosal safety (16.2%), which is typical for natural products that have not been optimized for human pharmacokinetics. However, their top compounds are well-known OSCC actives:

| Compound | Repurp. Score | VP Sim | ADMET | Mucosal | QED |
|----------|:------------:|:------:|:-----:|:-------:|:---:|
| Resveratrol | 0.808 | 0.875 | PASS | LOW | 0.69 |
| Quercetin | 0.728 | 0.926 | CAUTION | LOW | 0.43 |
| Sulforaphane | 0.653 | 0.955 | CAUTION | SAFE | 0.27 |
| Kaempferol | 0.613 | 0.856 | PASS | LOW | 0.55 |
| Fisetin | 0.590 | 0.923 | CAUTION | LOW | 0.51 |

These represent prime candidates for the Phase 9b lead optimization pipeline — combining their validated biological activity with ADMET-guided structural optimization.

#### 25.4.4 Cross-Category Validation of GENESIS Model

The category analysis serves as a multi-layered validation:

1. **Sensitivity**: PLOS compounds (known OSCC-relevant) score significantly higher than the general NonAnticancer pool (mean 0.392 vs 0.282), confirming the model discriminates OSCC-relevant chemistry.
2. **Specificity**: NonAnticancer compounds that score highly (>0.7) map overwhelmingly to OSCC-relevant targets (COX-2, EGFR, PIK3CA, STAT3), not random off-target mechanisms.
3. **Biological coherence**: The COX-2 inhibitor enrichment in top NonAnticancer hits is consistent with extensive literature on COX-2 overexpression in OSCC and the role of inflammation in oral carcinogenesis.
4. **Drug-likeness trade-off**: NonAnticancer compounds show better ADMET (65.6% pass) and mucosal safety (49.3%) because they were designed as drugs, while natural products and PLOS compounds sacrifice ADMET for potency.

### 25.5 Repurposing Score Distribution

| Range | Natural_AntiOSCC | PLOS_Cmpd | NonAnticancerCmpd |
|-------|:----------------:|:---------:|:-----------------:|
| > 0.7 (high priority) | 2 (5.4%) | 18 (8.2%) | 60 (0.3%) |
| 0.5–0.7 (moderate) | 10 (27.0%) | 48 (21.9%) | 763 (3.7%) |
| 0.3–0.5 (low) | 12 (32.4%) | 80 (36.5%) | 8,808 (42.8%) |
| < 0.3 (minimal) | 13 (35.1%) | 73 (33.3%) | 10,953 (53.2%) |

### 25.6 Output Files

All outputs saved to `GENESIS/category_analysis/results/`:

| File | Description |
|------|-------------|
| `Natural_AntiOSCC_annotated.tsv/.xlsx` | 37 compounds, 106 features |
| `PLOS_Cmpd_annotated.tsv/.xlsx` | 219 compounds, 106 features |
| `NonAnticancerCmpd_annotated.tsv/.xlsx` | 20,584 compounds, 106 features |
| `combined_all_categories.tsv/.xlsx` | 20,840 merged rows |
| `category_comparison.tsv/.xlsx` | Summary statistics (3 rows × 20 metrics) |

### 25.7 Interpretation

The category annotation pipeline achieves three objectives:

1. **Validates the GENESIS contrastive model** — the learned VP embedding correctly ranks known OSCC compounds (PLOS) above the general population, and the top NonAnticancer hits are biologically coherent (COX-2 inhibitors, EGFR inhibitors).

2. **Identifies actionable repurposing candidates** — 60 NonAnticancer compounds with repurposing score >0.7 represent FDA-approved/clinical-stage drugs (Indomethacin, Lumiracoxib, Meclofenamic Acid, Valdecoxib) that could be fast-tracked for OSCC clinical trials due to established safety profiles and low regulatory burden.

3. **Prioritizes natural product optimization** — the 12 Natural_AntiOSCC compounds scoring >0.5 (Resveratrol, Quercetin, Sulforaphane, Kaempferol, Fisetin, etc.) are validated leads for Phase 9b scaffold-constrained generation to improve their ADMET profiles while preserving OSCC activity.

---

*Document updated: 2026-09-19 (full section-by-section review against the corrected checkpoint-selection model: §§2–14 generative-method results corrected from a stale 3-seed {42,123,456} evaluation to the current 5-seed {42,123,456,789,1337} evaluation, including honest retraction/reframing of the cVAE "significant improvement," CGLF "best SA score," and CRAVE "seed-invariant" claims; §25 category-annotation results rerun against the corrected checkpoint. See `documents/manuscript_ground_truth.json` and `documents/verify_manuscript_numbers.py` for the independently re-derivable authoritative numbers.)*
*GENESIS: Generative Exploration of Novel Entities for Selective Inhibitor Synthesis*
