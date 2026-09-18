#!/usr/bin/env python3
"""
GENESIS — Final Compound Ranking & Nearest Known Analog Analysis
================================================================

Comprehensive integration of ALL pipeline data:
  - 1000 CRAVE-generated compounds (NKA analysis for all)
  - 345 pipeline compounds (189 generated + 156 repurposed) with ADMET, docking
  - 50 multi-target docked compounds with selectivity profiles
  - 69 lead-optimized analogs
  - Network pharmacology metrics
  - 23,308 repurposing-screened compounds

Outputs:
  - nka_all_generated.tsv: NKA analysis for all 1000 generated compounds
  - master_ranked_all.tsv / .xlsx: Full ranked list of all pipeline compounds
  - master_ranked_recommended.tsv / .xlsx: Final recommended compounds for clinical prioritization
  - compound_dossiers.xlsx: Detailed multi-sheet dossier for top compounds
"""

import sys
import json
import time
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, rdFMCS
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════
BASE = Path(__file__).resolve().parent.parent  # GENESIS/
DATA_DIR       = BASE / "data"
GEN_RESULTS    = BASE / "generative" / "results"
ADMET_RESULTS  = BASE / "admet" / "results"
DOCK_RESULTS   = BASE / "docking" / "results"
REPURP_RESULTS = BASE / "repurposing" / "results"
LEADOPT_RESULTS = BASE / "leadopt" / "results"
NETPHARM_RESULTS = BASE / "network_pharmacology" / "results"
MODELING_DIR   = BASE / "modeling"
OUT_DIR        = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# Module A: NKA — Nearest Known Analog Analysis
# ═══════════════════════════════════════════════════════════════

def load_reference_compounds():
    """Load all known compounds as reference set for NKA analysis.
    
    Sources:
      1. Training positives (known OSCC actives with pChEMBL)
      2. All_Compounds_Unq_Std.tsv (full ChEMBL dataset)
      3. Repurposing global_rankings (compounds with clinical metadata)
    """
    print("  Loading reference compound library...")
    
    ref_compounds = {}  # smiles → {metadata}
    
    # 1. Training positives — known OSCC actives
    pos = pd.read_csv(MODELING_DIR / "positives.tsv", sep="\t")
    for _, row in pos.iterrows():
        smi = str(row.get("smiles", ""))
        cid = str(row.get("compound_chembl_id", ""))
        if smi and cid:
            ref_compounds[smi] = {
                "ref_id": cid,
                "ref_name": "",
                "ref_source": "training_positive",
                "ref_pchembl": row.get("pchembl", np.nan),
                "ref_target": str(row.get("target_name", "")),
                "ref_max_phase": row.get("max_phase", np.nan),
                "ref_anticancer": row.get("IsAnticancer", ""),
            }
    
    # 2. All compounds — full metadata
    allcpd = pd.read_csv(DATA_DIR / "All_Compounds_Unq_Std.tsv", sep="\t")
    for _, row in allcpd.iterrows():
        smi = str(row.get("smiles", ""))
        cid = str(row.get("compound_chembl_id", ""))
        if smi and cid:
            ref_compounds[smi] = {
                "ref_id": cid,
                "ref_name": str(row.get("compound_name", "")),
                "ref_source": "chembl_oscc",
                "ref_pchembl": row.get("pchembl", np.nan),
                "ref_target": str(row.get("target_name", "")),
                "ref_max_phase": row.get("max_phase", np.nan),
                "ref_anticancer": str(row.get("IsAnticancer", "")),
                "ref_moa": str(row.get("mechanism_of_action", "")),
                "ref_indication": str(row.get("Indication", "")),
                "ref_origin": str(row.get("compound_origin", "")),
            }

    # 3. Repurposing rankings — additional clinical metadata  
    if (REPURP_RESULTS / "global_rankings.tsv").exists():
        rep = pd.read_csv(REPURP_RESULTS / "global_rankings.tsv", sep="\t",
                          low_memory=False)
        for _, row in rep.iterrows():
            smi = str(row.get("smiles", ""))
            cid = str(row.get("compound_chembl_id", ""))
            if smi and cid and smi not in ref_compounds:
                ref_compounds[smi] = {
                    "ref_id": cid,
                    "ref_name": str(row.get("compound_name", "")),
                    "ref_source": "repurposing_screen",
                    "ref_pchembl": row.get("pchembl", np.nan),
                    "ref_target": str(row.get("target_name", "")),
                    "ref_max_phase": row.get("max_phase", np.nan),
                    "ref_anticancer": str(row.get("IsAnticancer", "")),
                }

    print(f"    Reference library: {len(ref_compounds)} unique compounds")
    return ref_compounds


def compute_fingerprints(smiles_list, radius=2, nbits=2048):
    """Compute Morgan fingerprints for a list of SMILES."""
    fps = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
            fps.append(fp)
            valid_idx.append(i)
        else:
            fps.append(None)
            valid_idx.append(i)
    return fps, valid_idx


def nka_analysis(query_smiles_list, query_ids, ref_compounds, top_k=3):
    """Compute nearest known analogs for each query compound.
    
    Returns DataFrame with:
      - top-3 nearest analogs (Tanimoto, ID, name, metadata)
      - novelty classification
      - MCS analysis for top hit
    """
    print(f"  Running NKA analysis for {len(query_smiles_list)} compounds...")
    
    # Build reference fingerprints
    ref_smiles = list(ref_compounds.keys())
    ref_mols = []
    ref_fps = []
    ref_valid_smi = []
    
    for smi in ref_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            ref_mols.append(mol)
            ref_fps.append(fp)
            ref_valid_smi.append(smi)
    
    print(f"    Reference fingerprints: {len(ref_fps)}")
    
    results = []
    
    for idx, (q_smi, q_id) in enumerate(zip(query_smiles_list, query_ids)):
        if idx % 100 == 0:
            print(f"    Processing {idx}/{len(query_smiles_list)}...", end="\r")
        
        q_mol = Chem.MolFromSmiles(str(q_smi))
        row = {"cpd_id": q_id, "smiles": q_smi}
        
        if q_mol is None:
            row["nka_status"] = "invalid_smiles"
            for k in range(1, top_k + 1):
                row[f"nka{k}_tanimoto"] = np.nan
                row[f"nka{k}_id"] = ""
                row[f"nka{k}_name"] = ""
            row["novelty_class"] = "invalid"
            row["max_tanimoto"] = np.nan
            results.append(row)
            continue
        
        q_fp = AllChem.GetMorganFingerprintAsBitVect(q_mol, 2, nBits=2048)
        
        # Bulk Tanimoto
        sims = DataStructs.BulkTanimotoSimilarity(q_fp, ref_fps)
        
        # Top-k
        top_indices = np.argsort(sims)[-top_k:][::-1]
        
        for k, ti in enumerate(top_indices, 1):
            t_smi = ref_valid_smi[ti]
            meta = ref_compounds[t_smi]
            row[f"nka{k}_tanimoto"] = round(float(sims[ti]), 4)
            row[f"nka{k}_id"] = meta["ref_id"]
            row[f"nka{k}_name"] = meta.get("ref_name", "")
            row[f"nka{k}_pchembl"] = meta.get("ref_pchembl", np.nan)
            row[f"nka{k}_target"] = meta.get("ref_target", "")
            row[f"nka{k}_max_phase"] = meta.get("ref_max_phase", np.nan)
            row[f"nka{k}_moa"] = meta.get("ref_moa", "")
        
        max_tan = float(sims[top_indices[0]])
        row["max_tanimoto"] = round(max_tan, 4)
        
        # Novelty classification
        if max_tan >= 0.7:
            row["novelty_class"] = "close_analog"
        elif max_tan >= 0.4:
            row["novelty_class"] = "scaffold_hop"
        else:
            row["novelty_class"] = "novel"
        
        # MCS with top hit (only for non-trivial similarity)
        if max_tan >= 0.2:
            try:
                ref_mol_top = ref_mols[top_indices[0]]
                mcs = rdFMCS.FindMCS(
                    [q_mol, ref_mol_top],
                    timeout=2,
                    atomCompare=rdFMCS.AtomCompare.CompareElements,
                    bondCompare=rdFMCS.BondCompare.CompareOrder,
                )
                row["mcs_num_atoms"] = mcs.numAtoms
                row["mcs_num_bonds"] = mcs.numBonds
                q_atoms = q_mol.GetNumHeavyAtoms()
                row["mcs_coverage_query"] = round(
                    mcs.numAtoms / q_atoms if q_atoms > 0 else 0, 3)
            except Exception:
                row["mcs_num_atoms"] = np.nan
                row["mcs_num_bonds"] = np.nan
                row["mcs_coverage_query"] = np.nan
        else:
            row["mcs_num_atoms"] = np.nan
            row["mcs_num_bonds"] = np.nan
            row["mcs_coverage_query"] = np.nan
        
        row["nka_status"] = "ok"
        results.append(row)
    
    print(f"    NKA analysis complete for {len(results)} compounds")
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# Module B: Merge all pipeline data
# ═══════════════════════════════════════════════════════════════

def load_all_pipeline_data():
    """Load and merge all GENESIS pipeline outputs into a single DataFrame."""
    print("\n  Loading all pipeline data...")
    
    # ── 1. ADMET + docking (final_candidates.tsv is the most complete) ──
    fc = pd.read_csv(ADMET_RESULTS / "final_candidates.tsv", sep="\t")
    print(f"    final_candidates: {len(fc)} compounds, {len(fc.columns)} cols")
    
    # ── 2. Multi-target selectivity profiles ──
    sp = pd.read_csv(DOCK_RESULTS / "selectivity_profiles.tsv", sep="\t")
    # Merge selectivity data not already in final_candidates
    sp_cols_to_add = [c for c in sp.columns if c not in fc.columns or c == "cpd_id"]
    sp_cols_to_add = list(set(sp_cols_to_add) | {"cpd_id"})
    sp_sub = sp[sp_cols_to_add]
    fc = fc.merge(sp_sub, on="cpd_id", how="left", suffixes=("", "_sp"))
    print(f"    + selectivity profiles: {len(sp)} compounds")
    
    # ── 3. Network pharmacology metrics ──
    if (NETPHARM_RESULTS / "compound_network_metrics.tsv").exists():
        cnm = pd.read_csv(NETPHARM_RESULTS / "compound_network_metrics.tsv", sep="\t")
        cnm_cols = [c for c in cnm.columns if c not in fc.columns or c == "cpd_id"]
        cnm_cols = list(set(cnm_cols) | {"cpd_id"})
        cnm_sub = cnm[cnm_cols]
        fc = fc.merge(cnm_sub, on="cpd_id", how="left", suffixes=("", "_nm"))
        print(f"    + network pharmacology: {len(cnm)} compounds")
    
    # ── 4. Compound docking scores (broader set) ──
    cds = pd.read_csv(DOCK_RESULTS / "compound_docking_scores.tsv", sep="\t")
    # Only add compounds not already in fc
    new_dock = cds[~cds["cpd_id"].isin(fc["cpd_id"])]
    if len(new_dock) > 0:
        fc = pd.concat([fc, new_dock], ignore_index=True, sort=False)
        print(f"    + additional docked compounds: {len(new_dock)}")
    
    # ── 5. Generated compounds not yet in pipeline ──
    # Load all 1000 generated SMILES
    gen_smiles = []
    with open(GEN_RESULTS / "generated_crave.txt") as f:
        gen_smiles = [line.strip() for line in f if line.strip()]
    
    existing_gen_ids = set(fc[fc["cpd_id"].str.startswith("GEN_", na=False)]["cpd_id"])
    new_gen = []
    for i, smi in enumerate(gen_smiles):
        gid = f"GEN_{i:05d}"
        if gid not in existing_gen_ids and gid not in set(r["cpd_id"] for r in new_gen):
            new_gen.append({"cpd_id": gid, "smiles": smi, "source": "generated", "method": "CRAVE"})
    
    if new_gen:
        fc = pd.concat([fc, pd.DataFrame(new_gen)], ignore_index=True, sort=False)
        print(f"    + remaining generated compounds: {len(new_gen)}")
    
    print(f"    Total compounds in master table: {len(fc)}")
    return fc, gen_smiles


def compute_descriptors_for_missing(df):
    """Fill in basic molecular descriptors where missing."""
    print("  Computing descriptors for compounds missing ADMET data...")
    
    desc_cols = ["mw", "logp", "hba", "hbd", "tpsa", "n_rotatable", "qed", "sa_score"]
    
    needs_desc = df[df["mw"].isna() & df["smiles"].notna()].index
    
    count = 0
    for idx in needs_desc:
        smi = str(df.at[idx, "smiles"])
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        
        df.at[idx, "mw"] = round(Descriptors.MolWt(mol), 1)
        df.at[idx, "logp"] = round(Descriptors.MolLogP(mol), 2)
        df.at[idx, "hba"] = Descriptors.NumHAcceptors(mol)
        df.at[idx, "hbd"] = Descriptors.NumHDonors(mol)
        df.at[idx, "tpsa"] = round(Descriptors.TPSA(mol), 1)
        df.at[idx, "n_rotatable"] = Descriptors.NumRotatableBonds(mol)
        df.at[idx, "qed"] = round(Descriptors.qed(mol), 3)
        try:
            from rdkit.Chem import RDConfig
            from rdkit.Contrib.SA_Score import sascorer
            df.at[idx, "sa_score"] = round(sascorer.calculateScore(mol), 2)
        except Exception:
            df.at[idx, "sa_score"] = np.nan
        
        # Lipinski
        violations = 0
        if Descriptors.MolWt(mol) > 500: violations += 1
        if Descriptors.MolLogP(mol) > 5: violations += 1
        if Descriptors.NumHAcceptors(mol) > 10: violations += 1
        if Descriptors.NumHDonors(mol) > 5: violations += 1
        df.at[idx, "lipinski_violations"] = violations
        
        count += 1
    
    print(f"    Computed descriptors for {count} compounds")
    return df


# ═══════════════════════════════════════════════════════════════
# Module C: Comprehensive Ranking
# ═══════════════════════════════════════════════════════════════

def compute_comprehensive_rank(df):
    """Compute multi-criteria ranking for clinical prioritization.
    
    Score components (weighted composite):
      1. Docking affinity (best energy, normalized)         — 25%
      2. ADMET profile (admet_score)                        — 20%
      3. Drug-likeness (QED)                                — 10%
      4. Multi-target coverage (n_hits_le7)                 — 10%
      5. Potency prediction (composite_score / vp_similarity) — 15%
      6. Synthetic accessibility (inverted sa_score)        — 5%
      7. Structural novelty (from NKA → novelty bonus)      — 5% 
      8. Clinical evidence (max_phase of nearest analog)    — 10%
    """
    print("\n  Computing comprehensive ranking scores...")
    
    ranked = df.copy()
    
    # ── Normalize each component to [0, 1] ──
    
    # 1. Docking affinity (lower = better, so negate)
    dock_col = "mt_best_energy"
    if dock_col not in ranked.columns:
        dock_col = "best_dock_energy"
    if dock_col in ranked.columns:
        dock_vals = ranked[dock_col].astype(float)
        valid_dock = dock_vals.dropna()
        if len(valid_dock) > 0:
            dock_min, dock_max = valid_dock.min(), valid_dock.max()
            if dock_max != dock_min:
                ranked["s_docking"] = ((dock_vals - dock_max) / (dock_min - dock_max)).clip(0, 1)
            else:
                ranked["s_docking"] = 0.5
        else:
            ranked["s_docking"] = np.nan
    else:
        ranked["s_docking"] = np.nan
    
    # 2. ADMET score (already 0-1)
    ranked["s_admet"] = ranked.get("admet_score", pd.Series(np.nan, index=ranked.index)).astype(float).clip(0, 1)
    
    # 3. QED (already 0-1)
    ranked["s_qed"] = ranked["qed"].astype(float).clip(0, 1) if "qed" in ranked.columns else np.nan
    
    # 4. Multi-target coverage
    mt_col = "mt_n_hits_le7"
    if mt_col not in ranked.columns:
        mt_col = "n_hits_le7"
    if mt_col in ranked.columns:
        mt_vals = ranked[mt_col].astype(float)
        mt_max = mt_vals.max()
        ranked["s_multitarget"] = (mt_vals / mt_max).clip(0, 1) if mt_max > 0 else np.nan
    else:
        ranked["s_multitarget"] = np.nan
    
    # 5. Potency prediction
    pot_col = "composite_score"
    if pot_col in ranked.columns:
        pot_vals = ranked[pot_col].astype(float)
        valid_pot = pot_vals.dropna()
        if len(valid_pot) > 0:
            pot_min, pot_max = valid_pot.min(), valid_pot.max()
            if pot_max != pot_min:
                ranked["s_potency"] = ((pot_vals - pot_min) / (pot_max - pot_min)).clip(0, 1)
            else:
                ranked["s_potency"] = 0.5
        else:
            ranked["s_potency"] = np.nan
    else:
        ranked["s_potency"] = np.nan
    
    # 6. Synthetic accessibility (lower = easier → invert)
    if "sa_score" in ranked.columns:
        sa_vals = ranked["sa_score"].astype(float)
        ranked["s_sa"] = ((10 - sa_vals) / 10).clip(0, 1)
    else:
        ranked["s_sa"] = np.nan
    
    # 7. Novelty bonus (scaffold_hop > close_analog > novel in terms of balance)
    # Novel compounds get moderate bonus (high risk/reward), close analogs get 
    # high confidence bonus, scaffold hops get best of both worlds
    novelty_map = {"close_analog": 0.7, "scaffold_hop": 1.0, "novel": 0.5}
    if "novelty_class" in ranked.columns:
        ranked["s_novelty"] = ranked["novelty_class"].map(novelty_map).fillna(0.3)
    else:
        ranked["s_novelty"] = 0.5  # default
    
    # 8. Clinical evidence (based on nearest analog max phase)
    if "nka1_max_phase" in ranked.columns:
        phase_vals = ranked["nka1_max_phase"].astype(float)
        ranked["s_clinical_evidence"] = (phase_vals / 4.0).clip(0, 1)
    else:
        ranked["s_clinical_evidence"] = np.nan
    
    # ── Composite score (weighted) ──
    weights = {
        "s_docking": 0.25,
        "s_admet": 0.20,
        "s_potency": 0.15,
        "s_qed": 0.10,
        "s_multitarget": 0.10,
        "s_clinical_evidence": 0.10,
        "s_sa": 0.05,
        "s_novelty": 0.05,
    }
    
    # Weighted average (only over available components) with data-completeness penalty
    # Compounds with more validation evidence get higher scores because
    # missing components are penalized (counted as 0 in numerator but still 
    # contribute to denominator via evidence_weight)
    def calc_weighted_score(row):
        num = 0.0
        den = 0.0
        n_available = 0
        for col, w in weights.items():
            val = row.get(col, np.nan)
            den += w  # always count denominator
            if pd.notna(val):
                num += float(val) * w
                n_available += 1
            # missing components → 0 contribution to numerator
        
        if den == 0:
            return np.nan
        
        raw = num / den
        
        # Evidence completeness multiplier: compounds with all 8 components
        # get full score; compounds with only 2-3 get penalized
        completeness = n_available / len(weights)
        # Apply sqrt scaling so partial data isn't too harsh
        evidence_mult = completeness ** 0.5
        
        return round(raw * evidence_mult, 4)
    
    ranked["genesis_score"] = ranked.apply(calc_weighted_score, axis=1)
    
    # ── Data completeness tier ──
    def completeness_tier(row):
        has_dock = pd.notna(row.get("s_docking", np.nan))
        has_admet = pd.notna(row.get("s_admet", np.nan))
        has_mt = pd.notna(row.get("s_multitarget", np.nan))
        
        if has_dock and has_admet and has_mt:
            return "Tier-1 (fully validated)"
        elif has_dock and has_admet:
            return "Tier-2 (dock + ADMET)"
        elif has_dock or has_admet:
            return "Tier-3 (partial validation)"
        else:
            return "Tier-4 (prediction only)"
    
    ranked["validation_tier"] = ranked.apply(completeness_tier, axis=1)
    
    # ── ADMET verdict ──
    if "admet_flag" not in ranked.columns:
        ranked["admet_flag"] = np.nan
    
    # ── Rank ──
    ranked = ranked.sort_values("genesis_score", ascending=False, na_position="last")
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    
    print(f"    Ranked {len(ranked)} compounds")
    print(f"    Tier distribution:")
    for tier, count in ranked["validation_tier"].value_counts().items():
        print(f"      {tier}: {count}")
    
    return ranked


# ═══════════════════════════════════════════════════════════════
# Module D: External Knowledge Annotations
# ═══════════════════════════════════════════════════════════════

def annotate_clinical_context(df):
    """Add clinical context annotations based on known drug classes and targets.
    
    Uses domain knowledge about OSCC drug development landscape.
    """
    print("\n  Annotating clinical context...")
    
    # Known OSCC-relevant drug classes and their clinical significance
    oscc_drug_classes = {
        "EGFR inhibitor": "EGFR overexpressed in 42-80% OSCC; erlotinib/cetuximab approved for HNSCC",
        "PI3K inhibitor": "PIK3CA mutated in 12-21% OSCC; alpelisib approved for breast cancer with PIK3CA",
        "mTOR inhibitor": "PI3K/AKT/mTOR axis hyperactivated in OSCC; everolimus in clinical trials",
        "AKT inhibitor": "AKT amplified in OSCC; capivasertib approved for breast cancer",
        "FGFR inhibitor": "FGFR1 amplified in subset of OSCC; erdafitinib approved for urothelial",
        "COX-2 inhibitor": "COX-2 overexpressed in OSCC; celecoxib in OSCC prevention trials",
        "MDM2 inhibitor": "TP53 mutated in 62-85% OSCC; MDM2 inhibitors for wt-p53 subset",
        "Multi-kinase inhibitor": "Polypharmacology may overcome OSCC resistance mechanisms",
        "Dual PI3K/mTOR": "Targets both nodes of hyperactivated pathway; higher efficacy potential",
    }
    
    # Target → drug class mapping
    target_to_class = {
        "EGFR": "EGFR inhibitor",
        "PIK3CA": "PI3K inhibitor", 
        "MTOR": "mTOR inhibitor",
        "AKT1": "AKT inhibitor",
        "FGFR1": "FGFR inhibitor",
        "PTGS2": "COX-2 inhibitor",
        "MDM2": "MDM2 inhibitor",
    }
    
    # Annotate based on best target
    def get_drug_class(row):
        target = str(row.get("best_target", row.get("phase7_target", "")))
        mt_hits = row.get("mt_n_hits_le7", row.get("n_hits_le7", 0))
        
        if pd.notna(mt_hits) and float(mt_hits) >= 5:
            return "Multi-kinase inhibitor"
        
        cls = target_to_class.get(target, "")
        
        # Special case: dual PI3K/mTOR
        if target in ("PIK3CA", "MTOR"):
            dock_pik = row.get("dG_PIK3CA", 0)
            dock_mtor = row.get("dG_MTOR", 0)
            if pd.notna(dock_pik) and pd.notna(dock_mtor):
                if float(dock_pik) <= -7 and float(dock_mtor) <= -7:
                    return "Dual PI3K/mTOR"
        
        return cls
    
    df["predicted_drug_class"] = df.apply(get_drug_class, axis=1)
    df["oscc_clinical_rationale"] = df["predicted_drug_class"].map(oscc_drug_classes).fillna("")
    
    # Clinical development priority
    def clinical_priority(row):
        admet = str(row.get("admet_flag", ""))
        tier = str(row.get("validation_tier", ""))
        novelty = str(row.get("novelty_class", ""))
        source = str(row.get("source", ""))
        nka_phase = row.get("nka1_max_phase", 0)
        
        priority_points = 0
        reasons = []
        
        # ADMET status
        if admet == "PASS":
            priority_points += 3
            reasons.append("ADMET_PASS")
        elif admet == "CAUTION":
            priority_points += 1
            reasons.append("ADMET_CAUTION")
        
        # Validation tier
        if "Tier-1" in tier:
            priority_points += 3
            reasons.append("fully_validated")
        elif "Tier-2" in tier:
            priority_points += 2
            reasons.append("dock+admet")
        
        # Novelty value
        if novelty == "scaffold_hop":
            priority_points += 2
            reasons.append("scaffold_hop_novelty")
        elif novelty == "novel":
            priority_points += 1
            reasons.append("novel_scaffold")
        elif novelty == "close_analog":
            priority_points += 1
            reasons.append("known_scaffold")
        
        # Clinical precedent from analog
        if pd.notna(nka_phase) and float(nka_phase) >= 3:
            priority_points += 2
            reasons.append(f"analog_phase{int(float(nka_phase))}")
        
        # Source bonus for repurposed (existing safety data)
        if source == "repurposed":
            priority_points += 1
            reasons.append("existing_drug")
        
        return priority_points, "|".join(reasons)
    
    priorities = df.apply(clinical_priority, axis=1)
    df["clinical_priority_score"] = [p[0] for p in priorities]
    df["priority_reasons"] = [p[1] for p in priorities]
    
    return df


# ═══════════════════════════════════════════════════════════════
# Module E: Lead Optimization Integration
# ═══════════════════════════════════════════════════════════════

def integrate_lead_optimization(df):
    """Add lead optimization context for compounds that have analogs."""
    print("  Integrating lead optimization data...")
    
    if not (LEADOPT_RESULTS / "analogs_all.tsv").exists():
        return df
    
    analogs = pd.read_csv(LEADOPT_RESULTS / "analogs_all.tsv", sep="\t")
    
    # Count analogs per parent
    analog_counts = analogs.groupby("parent_id").agg(
        n_analogs=("analog_smiles", "count"),
        mean_tanimoto=("tanimoto_to_parent", "mean"),
        best_admet_score=("admet_score", "max"),
        n_pass_analogs=("admet_flag", lambda x: (x == "PASS").sum()),
    ).reset_index()
    analog_counts.rename(columns={"parent_id": "cpd_id"}, inplace=True)
    
    df = df.merge(analog_counts, on="cpd_id", how="left")
    
    print(f"    Added lead opt data for {len(analog_counts)} parent compounds")
    return df


# ═══════════════════════════════════════════════════════════════
# Module F: Excel Export with formatting
# ═══════════════════════════════════════════════════════════════

def save_xlsx(df, path, sheet_name="Compounds"):
    """Save DataFrame to Excel with formatting."""
    try:
        import openpyxl
        from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
        from openpyxl.utils import get_column_letter
        
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False, freeze_panes=(1, 2))
            
            ws = writer.sheets[sheet_name]
            
            # Header formatting
            header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
            header_font = Font(color="FFFFFF", bold=True, size=10)
            
            for col_idx in range(1, len(df.columns) + 1):
                cell = ws.cell(row=1, column=col_idx)
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center", wrap_text=True)
            
            # Conditional formatting for ADMET flag
            if "admet_flag" in df.columns:
                col_idx = df.columns.get_loc("admet_flag") + 1
                pass_fill = PatternFill(start_color="C6EFCE", fill_type="solid")
                caution_fill = PatternFill(start_color="FFEB9C", fill_type="solid")
                fail_fill = PatternFill(start_color="FFC7CE", fill_type="solid")
                
                for row_idx in range(2, len(df) + 2):
                    cell = ws.cell(row=row_idx, column=col_idx)
                    if cell.value == "PASS":
                        cell.fill = pass_fill
                    elif cell.value == "CAUTION":
                        cell.fill = caution_fill
                    elif cell.value == "FAIL":
                        cell.fill = fail_fill
            
            # Auto-width columns (capped)
            for col_idx in range(1, min(len(df.columns) + 1, 100)):
                col_letter = get_column_letter(col_idx)
                max_len = max(
                    len(str(ws.cell(row=1, column=col_idx).value or "")),
                    max((len(str(ws.cell(row=r, column=col_idx).value or ""))
                         for r in range(2, min(len(df) + 2, 20))), default=8)
                )
                ws.column_dimensions[col_letter].width = min(max_len + 2, 40)
        
        print(f"    Saved: {path}")
        return True
        
    except ImportError:
        print("    openpyxl not available — skipping .xlsx export")
        return False


# ═══════════════════════════════════════════════════════════════
# Module G: Compound Dossiers
# ═══════════════════════════════════════════════════════════════

def create_dossiers(ranked_df, nka_df, n_top=None):
    """Create detailed dossiers for recommended compounds.
    
    If n_top is None, creates dossiers for ALL Tier-1 and Tier-2 
    compounds with genesis_score > median.
    """
    print("\n  Creating compound dossiers...")
    
    # Select dossier candidates: all Tier-1 + Tier-2 compounds
    mask = ranked_df["validation_tier"].str.contains("Tier-1|Tier-2", na=False)
    candidates = ranked_df[mask].copy()
    
    if n_top and len(candidates) > n_top:
        candidates = candidates.head(n_top)
    
    print(f"    Creating dossiers for {len(candidates)} compounds")
    
    dossiers = []
    
    for _, row in candidates.iterrows():
        d = {}
        
        # ── Identity ──
        d["Compound ID"] = row.get("cpd_id", "")
        d["SMILES"] = row.get("smiles", "")
        d["Source"] = row.get("source", "")
        d["Generation Method"] = row.get("method", "")
        d["Compound Name"] = row.get("compound_name", "")
        
        # ── Ranking ──
        d["GENESIS Rank"] = row.get("rank", "")
        d["GENESIS Score"] = row.get("genesis_score", "")
        d["Validation Tier"] = row.get("validation_tier", "")
        d["Clinical Priority Score"] = row.get("clinical_priority_score", "")
        d["Priority Reasons"] = row.get("priority_reasons", "")
        
        # ── Molecular Properties ──
        d["MW (g/mol)"] = row.get("mw", "")
        d["LogP"] = row.get("logp", "")
        d["HBA"] = row.get("hba", "")
        d["HBD"] = row.get("hbd", "")
        d["TPSA"] = row.get("tpsa", "")
        d["Rotatable Bonds"] = row.get("n_rotatable", "")
        d["QED"] = row.get("qed", "")
        d["SA Score"] = row.get("sa_score", "")
        d["Lipinski Violations"] = row.get("lipinski_violations", "")
        
        # ── Docking ──
        d["Best Docking Energy (kcal/mol)"] = row.get("mt_best_energy", row.get("best_dock_energy", ""))
        d["Best Target"] = row.get("best_target", row.get("phase7_target", ""))
        d["Predicted Drug Class"] = row.get("predicted_drug_class", "")
        d["OSCC Clinical Rationale"] = row.get("oscc_clinical_rationale", "")
        d["N Targets ≤-7 kcal/mol"] = row.get("mt_n_hits_le7", row.get("n_hits_le7", ""))
        d["N Targets ≤-8 kcal/mol"] = row.get("mt_n_hits_le8", row.get("n_hits_le8", ""))
        
        # Per-target docking
        for t in ["AKT1", "EGFR", "FGFR1", "MDM2", "MTOR", "PIK3CA", "PTGS2"]:
            d[f"dG_{t}"] = row.get(f"dG_{t}", "")
        
        d["Selectivity Index"] = row.get("selectivity_index", "")
        
        # ── ADMET Profile ──
        d["ADMET Score"] = row.get("admet_score", "")
        d["ADMET Flag"] = row.get("admet_flag", "")
        d["Caco-2 Class"] = row.get("caco2_class", "")
        d["BBB Class"] = row.get("bbb_class", "")
        d["PPB Class"] = row.get("ppb_class", "")
        d["hERG Risk"] = row.get("herg_risk", "")
        d["PAINS Count"] = row.get("pains_count", "")
        d["Brenk Count"] = row.get("brenk_count", "")
        d["Mucosal Risk"] = row.get("mucosal_risk", "")
        d["Total Structural Alerts"] = row.get("total_structural_alerts", "")
        
        # ── Potency Predictions ──
        d["VP Similarity"] = row.get("vp_similarity", "")
        d["Composite Score"] = row.get("composite_score", "")
        
        # ── NKA (Nearest Known Analog) ──
        d["Max Tanimoto to Known"] = row.get("max_tanimoto", "")
        d["Novelty Classification"] = row.get("novelty_class", "")
        d["MCS Coverage (query)"] = row.get("mcs_coverage_query", "")
        
        for k in range(1, 4):
            d[f"NKA-{k} ID"] = row.get(f"nka{k}_id", "")
            d[f"NKA-{k} Name"] = row.get(f"nka{k}_name", "")
            d[f"NKA-{k} Tanimoto"] = row.get(f"nka{k}_tanimoto", "")
            d[f"NKA-{k} pChEMBL"] = row.get(f"nka{k}_pchembl", "")
            d[f"NKA-{k} Target"] = row.get(f"nka{k}_target", "")
            d[f"NKA-{k} Phase"] = row.get(f"nka{k}_max_phase", "")
            d[f"NKA-{k} MoA"] = row.get(f"nka{k}_moa", "")
        
        # ── Lead Optimization ──
        d["N Lead-Opt Analogs"] = row.get("n_analogs", "")
        d["N PASS Analogs"] = row.get("n_pass_analogs", "")
        d["Best Analog ADMET"] = row.get("best_admet_score", "")
        
        # ── Network Pharmacology ──
        d["Polypharmacology Score"] = row.get("polypharmacology_score", "")
        d["Is Elite"] = row.get("is_elite", "")
        
        dossiers.append(d)
    
    dossier_df = pd.DataFrame(dossiers)
    return dossier_df


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70)
    print("GENESIS — Final Compound Ranking & NKA Analysis")
    print("=" * 70)
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # ═══ A: NKA Analysis for ALL generated compounds ═══
    print("[A] Nearest Known Analog Analysis")
    print("-" * 40)
    
    ref_compounds = load_reference_compounds()
    
    # Load all 1000 generated SMILES
    gen_smiles = []
    with open(GEN_RESULTS / "generated_crave.txt") as f:
        gen_smiles = [line.strip() for line in f if line.strip()]
    gen_ids = [f"GEN_{i:05d}" for i in range(len(gen_smiles))]
    
    print(f"  Generated compounds: {len(gen_smiles)}")
    
    # Run NKA on ALL generated
    nka_gen = nka_analysis(gen_smiles, gen_ids, ref_compounds, top_k=3)
    nka_gen.to_csv(OUT_DIR / "nka_all_generated.tsv", sep="\t", index=False)
    
    # NKA stats  
    print(f"\n  NKA Summary (Generated Compounds):")
    vc = nka_gen["novelty_class"].value_counts()
    for cls, cnt in vc.items():
        print(f"    {cls}: {cnt} ({100*cnt/len(nka_gen):.1f}%)")
    print(f"    Mean max Tanimoto: {nka_gen['max_tanimoto'].mean():.3f}")
    print(f"    Median max Tanimoto: {nka_gen['max_tanimoto'].median():.3f}")
    
    # Also run NKA on repurposed compounds in the pipeline
    print(f"\n  Running NKA for repurposed pipeline compounds...")
    pipeline_data, _ = load_all_pipeline_data()
    
    rep_mask = pipeline_data["source"].fillna("").str.contains("repurposed")
    rep_smiles = pipeline_data.loc[rep_mask, "smiles"].tolist()
    rep_ids = pipeline_data.loc[rep_mask, "cpd_id"].tolist()
    
    nka_rep = nka_analysis(rep_smiles, rep_ids, ref_compounds, top_k=3)
    
    # Combine NKA results
    nka_all = pd.concat([nka_gen, nka_rep], ignore_index=True)
    nka_all.to_csv(OUT_DIR / "nka_all_compounds.tsv", sep="\t", index=False)
    print(f"  Total NKA results: {len(nka_all)} compounds")
    
    # ═══ B: Build master table ═══
    print(f"\n[B] Building Master Compound Table")
    print("-" * 40)
    
    # Reload pipeline data fresh
    master, gen_smiles = load_all_pipeline_data()
    
    # Compute missing descriptors
    master = compute_descriptors_for_missing(master)
    
    # Merge NKA results into master
    nka_merge_cols = ["cpd_id", "max_tanimoto", "novelty_class",
                      "mcs_num_atoms", "mcs_num_bonds", "mcs_coverage_query"]
    for k in range(1, 4):
        nka_merge_cols.extend([
            f"nka{k}_tanimoto", f"nka{k}_id", f"nka{k}_name",
            f"nka{k}_pchembl", f"nka{k}_target", f"nka{k}_max_phase",
            f"nka{k}_moa",
        ])
    
    # Only keep columns that exist
    nka_merge_cols = [c for c in nka_merge_cols if c in nka_all.columns]
    nka_to_merge = nka_all[nka_merge_cols].drop_duplicates(subset=["cpd_id"])
    
    master = master.merge(nka_to_merge, on="cpd_id", how="left")
    
    # ═══ C: Integrate lead optimization ═══
    master = integrate_lead_optimization(master)
    
    # ═══ D: Comprehensive ranking ═══
    print(f"\n[C] Comprehensive Ranking")
    print("-" * 40)
    ranked = compute_comprehensive_rank(master)
    
    # ═══ E: Clinical annotations ═══
    ranked = annotate_clinical_context(ranked)
    
    # Re-sort by genesis_score after annotations
    ranked = ranked.sort_values("genesis_score", ascending=False, na_position="last")
    ranked["rank"] = range(1, len(ranked) + 1)
    
    # ═══ F: Save outputs ═══
    print(f"\n[D] Saving Results")
    print("-" * 40)
    
    # 1. Master ranked list — ALL compounds
    # Select clean column ordering
    id_cols = ["rank", "cpd_id", "smiles", "source", "method", "compound_name"]
    score_cols = ["genesis_score", "validation_tier", "clinical_priority_score",
                  "priority_reasons", "predicted_drug_class", "oscc_clinical_rationale"]
    dock_cols = ["mt_best_energy", "best_dock_energy", "best_target", "phase7_target",
                 "dG_AKT1", "dG_EGFR", "dG_FGFR1", "dG_MDM2", "dG_MTOR",
                 "dG_PIK3CA", "dG_PTGS2", "mt_n_hits_le7", "mt_n_hits_le8",
                 "selectivity_index"]
    admet_cols = ["admet_score", "admet_flag", "mw", "logp", "hba", "hbd", "tpsa",
                  "n_rotatable", "qed", "sa_score", "lipinski_violations",
                  "caco2_class", "bbb_class", "ppb_class", "herg_risk",
                  "pains_count", "brenk_count", "toxicophore_count",
                  "total_structural_alerts", "mucosal_risk"]
    potency_cols = ["composite_score", "vp_similarity"]
    nka_cols = ["max_tanimoto", "novelty_class", "mcs_coverage_query"]
    for k in range(1, 4):
        nka_cols.extend([f"nka{k}_tanimoto", f"nka{k}_id", f"nka{k}_name",
                         f"nka{k}_pchembl", f"nka{k}_target", f"nka{k}_max_phase",
                         f"nka{k}_moa"])
    leadopt_cols = ["n_analogs", "n_pass_analogs", "best_admet_score"]
    network_cols = ["polypharmacology_score", "is_elite", "n_targets_bound"]
    component_cols = ["s_docking", "s_admet", "s_potency", "s_qed",
                      "s_multitarget", "s_clinical_evidence", "s_sa", "s_novelty"]
    
    all_ordered_cols = (id_cols + score_cols + dock_cols + admet_cols +
                        potency_cols + nka_cols + leadopt_cols + network_cols +
                        component_cols)
    
    # Keep only existing columns
    ordered_cols = [c for c in all_ordered_cols if c in ranked.columns]
    # Add any remaining columns not yet included
    remaining = [c for c in ranked.columns if c not in ordered_cols]
    ordered_cols.extend(remaining)
    
    ranked_ordered = ranked[ordered_cols]
    
    # Save ALL compounds
    ranked_ordered.to_csv(OUT_DIR / "master_ranked_all.tsv", sep="\t", index=False)
    print(f"  Saved: master_ranked_all.tsv ({len(ranked_ordered)} compounds)")
    
    # 2. Save RECOMMENDED compounds (exclude Tier-4 with no validation)
    # But include ALL Tier-1, 2, 3 compounds regardless of score
    recommended = ranked_ordered[
        ranked_ordered["validation_tier"] != "Tier-4 (prediction only)"
    ].copy()
    recommended["rank"] = range(1, len(recommended) + 1)
    
    recommended.to_csv(OUT_DIR / "master_ranked_recommended.tsv", sep="\t", index=False)
    print(f"  Saved: master_ranked_recommended.tsv ({len(recommended)} compounds)")
    
    # 3. Try Excel export
    xlsx_ok = save_xlsx(ranked_ordered, OUT_DIR / "master_ranked_all.xlsx",
                        "All_Compounds")
    if xlsx_ok:
        save_xlsx(recommended, OUT_DIR / "master_ranked_recommended.xlsx",
                  "Recommended")
    
    # 4. Compound dossiers
    print(f"\n[E] Compound Dossiers")
    print("-" * 40)
    dossier_df = create_dossiers(ranked, nka_all)
    dossier_df.to_csv(OUT_DIR / "compound_dossiers.tsv", sep="\t", index=False)
    print(f"  Saved: compound_dossiers.tsv ({len(dossier_df)} compounds)")
    
    if xlsx_ok:
        try:
            import openpyxl
            from openpyxl.styles import PatternFill, Font, Alignment
            from openpyxl.utils import get_column_letter
            
            with pd.ExcelWriter(OUT_DIR / "compound_dossiers.xlsx",
                                engine="openpyxl") as writer:
                
                # Sheet 1: Summary table
                dossier_df.to_excel(writer, sheet_name="Dossier_Summary",
                                    index=False, freeze_panes=(1, 2))
                
                # Sheet 2: Generated compounds NKA
                nka_gen.to_excel(writer, sheet_name="NKA_Generated",
                                 index=False, freeze_panes=(1, 1))
                
                # Sheet 3: Tier-1 elite
                tier1 = ranked_ordered[
                    ranked_ordered["validation_tier"].str.contains("Tier-1", na=False)
                ]
                tier1.to_excel(writer, sheet_name="Tier1_Elite",
                               index=False, freeze_panes=(1, 2))
                
                # Sheet 4: Lead optimization analogs
                if (LEADOPT_RESULTS / "analogs_all.tsv").exists():
                    analogs = pd.read_csv(LEADOPT_RESULTS / "analogs_all.tsv", sep="\t")
                    analogs.to_excel(writer, sheet_name="Lead_Opt_Analogs",
                                     index=False, freeze_panes=(1, 1))
                
                # Sheet 5: Network pharmacology
                if (NETPHARM_RESULTS / "pathway_mapping.tsv").exists():
                    pathways = pd.read_csv(NETPHARM_RESULTS / "pathway_mapping.tsv",
                                          sep="\t")
                    pathways.to_excel(writer, sheet_name="Pathways",
                                      index=False)
                
                # Format headers for each sheet
                for sheet_name in writer.sheets:
                    ws = writer.sheets[sheet_name]
                    header_fill = PatternFill(start_color="1F4E79",
                                             end_color="1F4E79", fill_type="solid")
                    header_font = Font(color="FFFFFF", bold=True, size=10)
                    
                    for col_idx in range(1, ws.max_column + 1):
                        cell = ws.cell(row=1, column=col_idx)
                        cell.fill = header_fill
                        cell.font = header_font
                        cell.alignment = Alignment(horizontal="center",
                                                   wrap_text=True)
                
            print(f"  Saved: compound_dossiers.xlsx (multi-sheet)")
        except Exception as e:
            print(f"  Warning: Excel dossier formatting incomplete: {e}")
    
    # ═══ Summary Stats ═══
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")
    
    print(f"\n  Total compounds ranked: {len(ranked_ordered)}")
    print(f"  Recommended (with validation): {len(recommended)}")
    print(f"  Dossiers created: {len(dossier_df)}")
    
    # By source
    print(f"\n  By source:")
    for src, cnt in ranked_ordered.get("source", pd.Series()).value_counts().items():
        print(f"    {src}: {cnt}")
    
    # By tier
    print(f"\n  By validation tier:")
    for tier, cnt in ranked_ordered["validation_tier"].value_counts().items():
        print(f"    {tier}: {cnt}")
    
    # By ADMET
    if "admet_flag" in ranked_ordered.columns:
        print(f"\n  By ADMET flag (validated compounds):")
        for flag, cnt in ranked_ordered["admet_flag"].value_counts(dropna=False).items():
            print(f"    {flag}: {cnt}")
    
    # Top 10
    print(f"\n  Top 10 by GENESIS Score:")
    top10 = ranked_ordered.head(10)
    for _, row in top10.iterrows():
        cpd = row.get("cpd_id", "?")
        score = row.get("genesis_score", 0)
        tier = row.get("validation_tier", "?")
        admet = row.get("admet_flag", "?")
        dock = row.get("mt_best_energy", row.get("best_dock_energy", "?"))
        novelty = row.get("novelty_class", "?")
        drug_cls = row.get("predicted_drug_class", "?")
        print(f"    #{int(row.get('rank', 0)):3d}  {cpd:30s}  "
              f"score={score:.3f}  dock={dock}  "
              f"ADMET={admet}  novelty={novelty}  class={drug_cls}")
    
    # NKA novelty distribution
    print(f"\n  Novelty distribution (all generated):")
    for cls, cnt in nka_gen["novelty_class"].value_counts().items():
        print(f"    {cls}: {cnt} ({100*cnt/len(nka_gen):.1f}%)")
    
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Analysis complete in {elapsed:.1f}s")
    print(f"Results saved to: {OUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
