#!/usr/bin/env python3
"""
GENESIS — Phase 9b: Lead Optimization via Scaffold-Constrained Generation
==========================================================================
Generate focused analogs of the 12 triple-validated elite candidates using:
  1. Scaffold-constrained CRAVE generation (latent-space perturbation)
  2. Murcko scaffold decomposition & R-group enumeration
  3. Rapid ADMET re-scoring (reusing Phase 8 modules)
  4. Structure–Activity Relationship (SAR) analysis
  5. Property improvement tracking vs parent compounds

Strategy:
  Instead of sampling z ~ N(0, I) (as in CRAVE), we:
    a) Encode each elite compound's SMILES → MolDeBERTa [CLS] embedding
    b) Project through ContrastiveModel → 128-d latent
    c) Sample z from N(z_elite, σ²I) with σ = 0.3–0.8 (multiple radii)
    d) Decode via CRAVE's decoder → candidate analogs
    e) Filter for scaffold similarity + ADMET improvement

Dependencies:
    pip install torch rdkit-pypi selfies transformers pandas numpy scipy

Input:
    - GENESIS/admet/results/final_candidates.tsv
    - GENESIS/docking/results/selectivity_profiles.tsv
    - GENESIS/generative/checkpoints/best_crave.pt
    - GENESIS/modeling/checkpoints/best.pt
    - GENESIS/generative/processed/{vocab.json, inv_vocab.json, vp_centroid.json}

Output:
    - GENESIS/leadopt/results/analogs_all.tsv
    - GENESIS/leadopt/results/analogs_improved.tsv
    - GENESIS/leadopt/results/scaffold_analysis.tsv
    - GENESIS/leadopt/results/sar_table.tsv
    - GENESIS/leadopt/results/leadopt_summary.json
"""

import os, sys, json, time, math, hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

import torch
import torch.nn.functional as F

from rdkit import Chem, RDLogger
from rdkit.Chem import (
    QED, Descriptors, rdMolDescriptors, AllChem,
    DataStructs, Draw, rdFMCS
)
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

# SA Score
try:
    from rdkit.Chem import RDConfig
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
    SA_AVAILABLE = True
except Exception:
    SA_AVAILABLE = False

# ── Paths ──────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent         # GENESIS/
LEADOPT_DIR  = ROOT / "leadopt"
RESULTS_DIR  = LEADOPT_DIR / "results"
GEN_DIR      = ROOT / "generative"
MODEL_DIR    = ROOT / "modeling"
ADMET_DIR    = ROOT / "admet" / "results"
DOCK_DIR     = ROOT / "docking" / "results"
CHECKPOINT   = GEN_DIR / "checkpoints" / "best_crave.pt"
CONTRASTIVE  = MODEL_DIR / "checkpoints" / "best.pt"
VOCAB_PATH   = GEN_DIR / "processed" / "vocab.json"
INV_VOCAB    = GEN_DIR / "processed" / "inv_vocab.json"
CENTROID_P   = GEN_DIR / "processed" / "vp_centroid.json"

sys.path.insert(0, str(GEN_DIR))
sys.path.insert(1, str(MODEL_DIR))

# ── Configuration ──────────────────────────────────────────────────────
PERTURBATION_RADII = [0.3, 0.5, 0.8]       # σ for latent-space perturbation
SAMPLES_PER_RADIUS = 200                    # per elite compound per radius
MIN_TANIMOTO       = 0.30                   # minimum similarity to parent scaffold
MAX_MW             = 700                    # upper bound for MW
MAX_SA             = 5.5                    # maximum SA score
DEVICE             = "mps" if torch.backends.mps.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════════
# Module A: Load Elite Compounds & Models
# ══════════════════════════════════════════════════════════════════════

def load_elite_compounds():
    """Load triple-validated elite candidates from Phase 8."""
    candidates = pd.read_csv(ADMET_DIR / "final_candidates.tsv", sep="\t")
    elite = candidates[
        (candidates["in_phase7b"] == True) & (candidates["admet_flag"] == "PASS")
    ].sort_values("final_score", ascending=False).copy()

    # Also load selectivity profiles for docking context
    sel_profiles = pd.read_csv(DOCK_DIR / "selectivity_profiles.tsv", sep="\t")
    return elite, sel_profiles


def load_models(device):
    """Load CRAVE decoder, ContrastiveModel, and MolDeBERTa."""
    from dataset import decode_tokens
    from vae import SELFIES_VAE
    from model import ContrastiveModel
    from transformers import AutoTokenizer, AutoModel

    # Vocab
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)
    with open(INV_VOCAB) as f:
        inv_vocab = json.load(f)

    # Centroid
    with open(CENTROID_P) as f:
        centroid = np.array(json.load(f)["oscc_vp_centroid"], dtype=np.float32)
    centroid_t = torch.tensor(centroid, device=device, dtype=torch.float32)
    centroid_t = centroid_t / (centroid_t.norm() + 1e-12)

    # CRAVE (SELFIES_VAE)
    vae = SELFIES_VAE(vocab_size=len(vocab)).to(device)
    ckpt = torch.load(str(CHECKPOINT), map_location=device, weights_only=False)
    vae.load_state_dict(ckpt["model_state_dict"])
    vae.eval()

    # ContrastiveModel
    cm = ContrastiveModel()
    cm_ckpt = torch.load(str(CONTRASTIVE), map_location="cpu", weights_only=False)
    cm.load_state_dict(cm_ckpt["model_state_dict"])
    cm.eval().to(device)

    # MolDeBERTa
    tokenizer = AutoTokenizer.from_pretrained(
        "SaeedLab/MolDeBERTa-base-123M-mtr", use_fast=False)
    embedder = AutoModel.from_pretrained(
        "SaeedLab/MolDeBERTa-base-123M-mtr").eval().to(device)

    for p in list(cm.parameters()) + list(embedder.parameters()):
        p.requires_grad = False

    return vae, cm, embedder, tokenizer, vocab, inv_vocab, centroid_t


# ══════════════════════════════════════════════════════════════════════
# Module B: Scaffold-Constrained Generation
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def encode_smiles_to_latent(smiles, cm, embedder, tokenizer, device):
    """SMILES → MolDeBERTa [CLS] (768-d) → ContrastiveModel (128-d)."""
    inputs = tokenizer(smiles, return_tensors="pt", padding=True,
                       truncation=True, max_length=128).to(device)
    cls_emb = embedder(**inputs).last_hidden_state[:, 0, :]      # (1, 768)
    z, _ = cm(cls_emb)                                            # (1, 128)
    return z


@torch.no_grad()
def decode_latent_to_smiles(z, vae, inv_vocab, device, temperature=0.8, top_k=5):
    """Decode latent vector z through CRAVE's GRU decoder → SMILES."""
    import selfies as sf
    from dataset import decode_tokens

    decoder = vae.decoder
    n = z.size(0)

    h = decoder.latent_to_hidden(z)
    h = h.view(n, decoder.n_layers, decoder.hidden_dim).permute(1, 0, 2).contiguous()

    input_token = torch.ones(n, 1, dtype=torch.long, device=device)
    all_tokens = [input_token.squeeze(1)]

    for _ in range(decoder.max_len - 1):
        emb = decoder.embedding(input_token)
        output, h = decoder.gru(emb, h)
        logits = decoder.output_proj(output.squeeze(1)) / temperature

        if top_k > 0:
            topk_vals, topk_idx = logits.topk(top_k, dim=-1)
            probs = F.softmax(topk_vals, dim=-1)
            sampled = torch.multinomial(probs, 1)
            tokens = topk_idx.gather(-1, sampled).squeeze(-1)
        else:
            probs = F.softmax(logits, dim=-1)
            tokens = torch.multinomial(probs, 1).squeeze(-1)

        all_tokens.append(tokens)
        input_token = tokens.unsqueeze(1)

    token_tensor = torch.stack(all_tokens, dim=1)  # (n, max_len)

    smiles_list = []
    for i in range(n):
        selfies_str = decode_tokens(token_tensor[i].cpu().numpy(), inv_vocab)
        try:
            smi = sf.decoder(selfies_str)
            smiles_list.append(smi)
        except Exception:
            smiles_list.append(None)

    return smiles_list


def perturb_latent(z_center, n_samples, sigma, device):
    """Sample z ~ N(z_center, σ²I) — Gaussian ball around elite latent."""
    noise = torch.randn(n_samples, z_center.size(-1), device=device) * sigma
    return z_center.expand(n_samples, -1) + noise


def generate_analogs_for_compound(cpd_id, smiles, vae, cm, embedder, tokenizer,
                                   inv_vocab, device):
    """Generate focused analogs for one elite compound."""
    # Encode parent
    z_parent = encode_smiles_to_latent(smiles, cm, embedder, tokenizer, device)

    parent_mol = Chem.MolFromSmiles(smiles)
    parent_fp = AllChem.GetMorganFingerprintAsBitVect(parent_mol, 2, nBits=2048)

    all_analogs = []
    for sigma in PERTURBATION_RADII:
        z_perturbed = perturb_latent(z_parent, SAMPLES_PER_RADIUS, sigma, device)

        # Decode in batches
        batch_size = 64
        for start in range(0, SAMPLES_PER_RADIUS, batch_size):
            end = min(start + batch_size, SAMPLES_PER_RADIUS)
            z_batch = z_perturbed[start:end]
            decoded = decode_latent_to_smiles(z_batch, vae, inv_vocab, device)

            for smi in decoded:
                if smi is None:
                    continue
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue

                # Canonicalize
                canon = Chem.MolToSmiles(mol)

                # Tanimoto similarity to parent
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                tanimoto = DataStructs.TanimotoSimilarity(parent_fp, fp)

                # Basic filters
                mw = Descriptors.MolWt(mol)
                if mw > MAX_MW:
                    continue
                if tanimoto < MIN_TANIMOTO:
                    continue

                all_analogs.append({
                    "parent_id": cpd_id,
                    "parent_smiles": smiles,
                    "analog_smiles": canon,
                    "sigma": sigma,
                    "tanimoto_to_parent": round(tanimoto, 3),
                    "mw": round(mw, 1),
                })

    return all_analogs


# ══════════════════════════════════════════════════════════════════════
# Module C: ADMET Re-Scoring (Lightweight Phase 8 Reuse)
# ══════════════════════════════════════════════════════════════════════

def get_sa_score_safe(mol):
    if SA_AVAILABLE:
        try:
            return round(sascorer.calculateScore(mol), 2)
        except Exception:
            return 5.0
    return 5.0


def admet_score_analog(mol):
    """Lightweight ADMET scoring — consistent with Phase 8."""
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hba  = rdMolDescriptors.CalcNumHBA(mol)
    hbd  = rdMolDescriptors.CalcNumHBD(mol)
    tpsa = Descriptors.TPSA(mol)
    rot  = rdMolDescriptors.CalcNumRotatableBonds(mol)

    try:
        qed_val = QED.qed(mol)
    except Exception:
        qed_val = 0.0

    sa = get_sa_score_safe(mol)
    sa_norm = max(0, (10.0 - sa) / 9.0)

    # Lipinski violations
    lip_v = sum([mw > 500, logp > 5, hba > 10, hbd > 5])

    # Veber violations
    veber_v = sum([tpsa > 140, rot > 10])

    # Caco-2 estimate
    if tpsa <= 90 and 0 <= logp <= 5 and mw <= 500 and hbd <= 3:
        caco2 = "High"
    elif tpsa <= 140 and -1 <= logp <= 6 and mw <= 600:
        caco2 = "Moderate"
    else:
        caco2 = "Low"

    # hERG risk estimate (crude)
    herg = "Low" if logp < 3.5 else ("Moderate" if logp < 5 else "High")

    # Integrated ADMET score (consistent with Phase 8 weighting)
    qed_comp   = qed_val * 0.25
    lip_comp   = max(0, 1 - lip_v * 0.25) * 0.20
    veber_comp = max(0, 1 - veber_v * 0.5) * 0.10
    sa_comp    = sa_norm * 0.15
    caco_comp  = {"High": 1.0, "Moderate": 0.6, "Low": 0.2}[caco2] * 0.15
    herg_comp  = {"Low": 1.0, "Moderate": 0.5, "High": 0.1}[herg] * 0.15
    admet = qed_comp + lip_comp + veber_comp + sa_comp + caco_comp + herg_comp

    # Flag
    if lip_v >= 2 or caco2 == "Low" or herg == "High":
        flag = "FAIL"
    elif lip_v == 1 or caco2 == "Moderate" or herg == "Moderate":
        flag = "CAUTION"
    else:
        flag = "PASS"

    return {
        "mw": round(mw, 1), "logp": round(logp, 2), "hba": hba, "hbd": hbd,
        "tpsa": round(tpsa, 1), "n_rotatable": rot, "qed": round(qed_val, 3),
        "sa_score": round(sa, 2), "lipinski_violations": lip_v,
        "veber_violations": veber_v, "caco2_class": caco2, "herg_risk": herg,
        "admet_score": round(admet, 3), "admet_flag": flag,
    }


# ══════════════════════════════════════════════════════════════════════
# Module D: Scaffold Analysis
# ══════════════════════════════════════════════════════════════════════

def scaffold_analysis(elite, analogs_df):
    """Murcko scaffold decomposition + comparison."""
    print("\n" + "=" * 70)
    print("MODULE D: Scaffold Analysis")
    print("=" * 70)

    scaffolds = []
    for _, row in elite.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            continue
        try:
            scaf = MurckoScaffold.GetScaffoldForMol(mol)
            scaf_smi = Chem.MolToSmiles(scaf)
            generic = MurckoScaffold.MakeScaffoldGeneric(scaf)
            generic_smi = Chem.MolToSmiles(generic)
        except Exception:
            scaf_smi = ""
            generic_smi = ""

        scaffolds.append({
            "cpd_id": row["cpd_id"],
            "smiles": row["smiles"],
            "murcko_scaffold": scaf_smi,
            "generic_scaffold": generic_smi,
            "mw": row.get("mw", Descriptors.MolWt(mol)),
            "source": row.get("source", ""),
        })

    scaf_df = pd.DataFrame(scaffolds)

    # Count unique scaffolds
    n_unique_murcko = scaf_df["murcko_scaffold"].nunique()
    n_unique_generic = scaf_df["generic_scaffold"].nunique()
    print(f"  Elite compounds: {len(scaf_df)}")
    print(f"  Unique Murcko scaffolds: {n_unique_murcko}")
    print(f"  Unique generic scaffolds: {n_unique_generic}")

    # Scaffold frequency in analogs
    if len(analogs_df) > 0:
        analog_scaffolds = []
        for smi in analogs_df["analog_smiles"].unique():
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                scaf = MurckoScaffold.GetScaffoldForMol(mol)
                analog_scaffolds.append(Chem.MolToSmiles(scaf))
            except Exception:
                pass

        parent_scafs = set(scaf_df["murcko_scaffold"].values)
        conserved = sum(1 for s in analog_scaffolds if s in parent_scafs)
        novel = len(analog_scaffolds) - conserved

        print(f"\n  Analog scaffold conservation:")
        print(f"    Total analog scaffolds: {len(analog_scaffolds)}")
        print(f"    Conserved (same as parent): {conserved} ({100*conserved/max(1,len(analog_scaffolds)):.1f}%)")
        print(f"    Novel scaffolds: {novel} ({100*novel/max(1,len(analog_scaffolds)):.1f}%)")

    scaf_df.to_csv(RESULTS_DIR / "scaffold_analysis.tsv", sep="\t", index=False)
    print(f"  Saved: scaffold_analysis.tsv")

    return scaf_df


# ══════════════════════════════════════════════════════════════════════
# Module E: SAR Table
# ══════════════════════════════════════════════════════════════════════

def build_sar_table(elite, analogs_improved, centroid_t, cm, embedder,
                     tokenizer, device):
    """Build Structure-Activity Relationship table comparing parent → analog."""
    print("\n" + "=" * 70)
    print("MODULE E: SAR Table")
    print("=" * 70)

    sar_rows = []

    for parent_id in analogs_improved["parent_id"].unique():
        parent_row = elite[elite["cpd_id"] == parent_id]
        if len(parent_row) == 0:
            continue
        parent_row = parent_row.iloc[0]
        parent_smi = parent_row["smiles"]
        parent_mol = Chem.MolFromSmiles(parent_smi)
        if parent_mol is None:
            continue

        parent_admet = parent_row.get("admet_score", 0)
        parent_qed = parent_row.get("qed", 0)
        parent_sa = parent_row.get("sa_score", 5)
        parent_fs = parent_row.get("final_score", 0)

        p_analogs = analogs_improved[analogs_improved["parent_id"] == parent_id]

        for _, arow in p_analogs.iterrows():
            delta_qed = arow["qed"] - parent_qed
            delta_admet = arow["admet_score"] - parent_admet
            delta_sa = arow["sa_score"] - parent_sa
            delta_mw = arow["mw"] - parent_row.get("mw", 0)

            # Potency (cosine sim to centroid)
            try:
                z_analog = encode_smiles_to_latent(
                    arow["analog_smiles"], cm, embedder, tokenizer, device)
                potency = F.cosine_similarity(
                    z_analog, centroid_t.unsqueeze(0), dim=-1).item()
                potency = max(0.0, potency)
            except Exception:
                potency = 0.0

            sar_rows.append({
                "parent_id": parent_id,
                "analog_smiles": arow["analog_smiles"],
                "tanimoto": arow["tanimoto_to_parent"],
                "potency_sim": round(potency, 3),
                "qed": arow["qed"],
                "delta_qed": round(delta_qed, 3),
                "sa_score": arow["sa_score"],
                "delta_sa": round(delta_sa, 2),
                "admet_score": arow["admet_score"],
                "delta_admet": round(delta_admet, 3),
                "mw": arow["mw"],
                "delta_mw": round(delta_mw, 1),
                "admet_flag": arow["admet_flag"],
                "caco2_class": arow["caco2_class"],
                "herg_risk": arow["herg_risk"],
            })

    sar_df = pd.DataFrame(sar_rows)
    if len(sar_df) > 0:
        sar_df = sar_df.sort_values("potency_sim", ascending=False)
        sar_df.to_csv(RESULTS_DIR / "sar_table.tsv", sep="\t", index=False)
        print(f"  SAR entries: {len(sar_df)}")
        print(f"  Mean potency similarity: {sar_df['potency_sim'].mean():.3f}")
        print(f"  Mean ΔQED: {sar_df['delta_qed'].mean():+.3f}")
        print(f"  Mean ΔSA:  {sar_df['delta_sa'].mean():+.2f}")
        print(f"  Mean ΔADMET: {sar_df['delta_admet'].mean():+.3f}")

        # Top 10
        print(f"\n  Top 10 Analogs (by potency):")
        for i, (_, r) in enumerate(sar_df.head(10).iterrows()):
            print(f"    {i+1:2d}. parent={r['parent_id']:20s}  "
                  f"pot={r['potency_sim']:.3f}  tanm={r['tanimoto']:.2f}  "
                  f"QED={r['qed']:.3f}(Δ{r['delta_qed']:+.3f})  "
                  f"ADMET={r['admet_score']:.3f}(Δ{r['delta_admet']:+.3f})  "
                  f"[{r['admet_flag']}]")
    else:
        print("  No SAR entries (no improved analogs found)")

    return sar_df


# ══════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70)
    print("GENESIS — Phase 9b: Lead Optimization")
    print("=" * 70)
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    print(f"Perturbation radii: {PERTURBATION_RADII}")
    print(f"Samples per radius per compound: {SAMPLES_PER_RADIUS}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(DEVICE)
    torch.manual_seed(42)
    np.random.seed(42)

    # ── Module A: Load data & models ──
    print("\n" + "=" * 70)
    print("MODULE A: Loading Data & Models")
    print("=" * 70)

    elite, sel_profiles = load_elite_compounds()
    print(f"  Elite compounds: {len(elite)}")

    vae, cm, embedder, tokenizer, vocab, inv_vocab, centroid_t = load_models(device)
    print(f"  Models loaded: VAE, ContrastiveModel, MolDeBERTa")

    # ── Module B: Scaffold-Constrained Generation ──
    print("\n" + "=" * 70)
    print("MODULE B: Scaffold-Constrained Generation")
    print("=" * 70)

    n_expected = len(elite) * len(PERTURBATION_RADII) * SAMPLES_PER_RADIUS
    print(f"  Expected samples: {len(elite)} compounds × "
          f"{len(PERTURBATION_RADII)} radii × {SAMPLES_PER_RADIUS} = {n_expected}")

    all_analogs = []
    for idx, (_, row) in enumerate(elite.iterrows()):
        cpd_id = row["cpd_id"]
        smiles = row["smiles"]
        print(f"\n  [{idx+1}/{len(elite)}] {cpd_id}")

        analogs = generate_analogs_for_compound(
            cpd_id, smiles, vae, cm, embedder, tokenizer, inv_vocab, device)
        print(f"    Generated {len(analogs)} valid analogs "
              f"(Tanimoto ≥ {MIN_TANIMOTO}, MW ≤ {MAX_MW})")
        all_analogs.extend(analogs)

    analogs_df = pd.DataFrame(all_analogs)

    # Deduplicate
    if len(analogs_df) > 0:
        n_before = len(analogs_df)
        analogs_df = analogs_df.drop_duplicates(subset=["analog_smiles", "parent_id"])
        # Remove self-generation
        analogs_df = analogs_df[analogs_df["analog_smiles"] != analogs_df["parent_smiles"]]
        print(f"\n  Total analogs: {n_before} → {len(analogs_df)} (after dedup + self-removal)")

    # ── Module C: ADMET Re-Scoring ──
    print("\n" + "=" * 70)
    print("MODULE C: ADMET Re-Scoring")
    print("=" * 70)

    if len(analogs_df) > 0:
        admet_results = []
        for i, (_, row) in enumerate(analogs_df.iterrows()):
            mol = Chem.MolFromSmiles(row["analog_smiles"])
            if mol is None:
                continue
            result = admet_score_analog(mol)
            result["parent_id"] = row["parent_id"]
            result["parent_smiles"] = row["parent_smiles"]
            result["analog_smiles"] = row["analog_smiles"]
            result["sigma"] = row["sigma"]
            result["tanimoto_to_parent"] = row["tanimoto_to_parent"]
            admet_results.append(result)

        analogs_df = pd.DataFrame(admet_results)
        analogs_df.to_csv(RESULTS_DIR / "analogs_all.tsv", sep="\t", index=False)

        print(f"  Scored {len(analogs_df)} analogs")
        print(f"  ADMET distribution:")
        for flag in ["PASS", "CAUTION", "FAIL"]:
            n = (analogs_df["admet_flag"] == flag).sum()
            print(f"    {flag:8s}: {n:5d} ({100*n/len(analogs_df):.1f}%)")

        # Filter for improved analogs: PASS or CAUTION with good QED
        improved = analogs_df[
            (analogs_df["admet_flag"].isin(["PASS", "CAUTION"])) &
            (analogs_df["qed"] >= 0.4) &
            (analogs_df["sa_score"] <= MAX_SA)
        ].copy()
        improved = improved.sort_values("admet_score", ascending=False)
        improved.to_csv(RESULTS_DIR / "analogs_improved.tsv", sep="\t", index=False)

        print(f"\n  Improved analogs (PASS/CAUTION + QED≥0.4 + SA≤{MAX_SA}): {len(improved)}")
    else:
        improved = pd.DataFrame()
        print("  No analogs generated")

    # ── Module D: Scaffold Analysis ──
    scaf_df = scaffold_analysis(elite, analogs_df)

    # ── Module E: SAR Table ──
    if len(improved) > 0:
        sar_df = build_sar_table(elite, improved, centroid_t, cm, embedder,
                                  tokenizer, device)
    else:
        sar_df = pd.DataFrame()
        print("\n  Skipping SAR (no improved analogs)")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("Saving Summary")
    print("=" * 70)

    elapsed = time.time() - t0

    summary = {
        "phase": "9b",
        "description": "Lead Optimization via Scaffold-Constrained Generation",
        "device": DEVICE,
        "elapsed_seconds": round(elapsed, 1),
        "n_elite": len(elite),
        "perturbation_radii": PERTURBATION_RADII,
        "samples_per_radius": SAMPLES_PER_RADIUS,
        "total_raw_analogs": len(all_analogs),
        "total_unique_analogs": len(analogs_df) if len(analogs_df) > 0 else 0,
        "admet_distribution": {
            "PASS": int((analogs_df["admet_flag"] == "PASS").sum()) if len(analogs_df) > 0 else 0,
            "CAUTION": int((analogs_df["admet_flag"] == "CAUTION").sum()) if len(analogs_df) > 0 else 0,
            "FAIL": int((analogs_df["admet_flag"] == "FAIL").sum()) if len(analogs_df) > 0 else 0,
        },
        "improved_analogs": len(improved),
        "sar_entries": len(sar_df),
        "scaffolds": {
            "unique_murcko": int(scaf_df["murcko_scaffold"].nunique()) if len(scaf_df) > 0 else 0,
            "unique_generic": int(scaf_df["generic_scaffold"].nunique()) if len(scaf_df) > 0 else 0,
        },
        "mean_tanimoto": round(analogs_df["tanimoto_to_parent"].mean(), 3) if len(analogs_df) > 0 else 0,
        "mean_admet_score": round(analogs_df["admet_score"].mean(), 3) if len(analogs_df) > 0 else 0,
        "output_files": [
            "analogs_all.tsv", "analogs_improved.tsv",
            "scaffold_analysis.tsv", "sar_table.tsv",
            "leadopt_summary.json",
        ],
    }

    with open(RESULTS_DIR / "leadopt_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Summary saved: leadopt_summary.json")
    print(f"\n{'=' * 70}")
    print(f"Phase 9b COMPLETE in {elapsed:.1f}s")
    print(f"  Elite parents: {len(elite)}")
    print(f"  Total analogs: {summary['total_unique_analogs']}")
    print(f"  Improved analogs: {summary['improved_analogs']}")
    print(f"  SAR entries: {summary['sar_entries']}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
