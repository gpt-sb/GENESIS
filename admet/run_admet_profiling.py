#!/usr/bin/env python3
"""
GENESIS — Phase 8: ADMET & Pharmacokinetic Profiling
=====================================================
Comprehensive drug-likeness, ADMET property prediction, and OSCC-specific
mucosal irritation safety assessment for all compounds validated in Phase 7/7b.

Modules:
  A. Compound Loading — merge Phase 7 single-target + Phase 7b multi-target data
  B. Physicochemical Properties — Lipinski, Veber, Ghose, Egan filters
  C. ADMET Prediction — absorption, metabolism, toxicity (RDKit descriptors)
  D. Structural Alert Screening — PAINS, Brenk, Toxicophores, extended alerts
  E. Mucosal Irritation Filter — OSCC-specific oral mucosal safety (novel)
  F. Integrated Scoring — multi-criteria ADMET score + final prioritization
  G. Final Candidate Shortlist — publication-ready compound table

Dependencies:
    pip install rdkit-pypi pandas numpy scipy

Input:
    - GENESIS/docking/results/docking_results.tsv           (Phase 7)
    - GENESIS/docking/results/selectivity_profiles.tsv       (Phase 7b)
    - GENESIS/docking/results/multitarget_binders.tsv        (Phase 7b)

Output:
    - GENESIS/admet/results/admet_profiles.tsv
    - GENESIS/admet/results/mucosal_safety.tsv
    - GENESIS/admet/results/final_candidates.tsv
    - GENESIS/admet/results/admet_summary.json
    - GENESIS/admet/results/safety_potency_analysis.tsv

Author: GENESIS Phase 8 pipeline
Date: 2026-03-23
"""

import os, sys, time, json, math
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, OrderedDict

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, Lipinski
from rdkit.Chem import FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams
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

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent          # GENESIS/
ADMET_DIR   = ROOT / "admet"
RESULTS_DIR = ADMET_DIR / "results"
DOCK_DIR    = ROOT / "docking" / "results"

# Input files
PHASE7_RESULTS   = DOCK_DIR / "docking_results.tsv"
PHASE7B_PROFILES = DOCK_DIR / "selectivity_profiles.tsv"
PHASE7B_BINDERS  = DOCK_DIR / "multitarget_binders.tsv"

# Output files
ADMET_OUT       = RESULTS_DIR / "admet_profiles.tsv"
MUCOSAL_OUT     = RESULTS_DIR / "mucosal_safety.tsv"
FINAL_OUT       = RESULTS_DIR / "final_candidates.tsv"
SUMMARY_OUT     = RESULTS_DIR / "admet_summary.json"
SAFETY_POT_OUT  = RESULTS_DIR / "safety_potency_analysis.tsv"

# ── Filter Catalogs ───────────────────────────────────────────────────────────
_pains_params = FilterCatalogParams()
_pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
PAINS_CATALOG = FilterCatalog.FilterCatalog(_pains_params)

_brenk_params = FilterCatalogParams()
_brenk_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
BRENK_CATALOG = FilterCatalog.FilterCatalog(_brenk_params)


# ══════════════════════════════════════════════════════════════════════════════
# SMARTS Pattern Libraries
# ══════════════════════════════════════════════════════════════════════════════

# ── Standard Toxicophores (from Phase 4/6) ─────────────────────────────────
TOXICOPHORES = {
    'nitro_aromatic':  '[$(c[N+](=O)[O-])]',
    'aniline':         '[NX3;H2;!$(NC=O)]c1ccccc1',
    'alkyl_halide':    '[CX4][F,Cl,Br,I]',
    'epoxide':         'C1OC1',
    'acyl_halide':     '[CX3](=[OX1])[F,Cl,Br,I]',
    'michael_acceptor': '[CX3]=[CX3][CX3]=[OX1]',
    'aldehyde':        '[CX3H1](=O)[#6]',
    'hydrazine':       '[NX3][NX3]',
    'peroxide':        '[OX2][OX2]',
    'thiol':           '[SX2H]',
}

# ── Extended Structural Alerts ─────────────────────────────────────────────
EXTENDED_ALERTS = {
    'azo_compound':          '[#7]=[#7]',
    'polycyclic_aromatic':   'a1aa2a(a1)a1aaaa3aaaaa1a23',  # PAH ≥3 fused
    'thiocarbonyl':          '[#6](=[SX1])',
    'sulfonyl_halide':       '[SX4](=[OX1])(=[OX1])[F,Cl,Br,I]',
    'phosphonate_ester':     '[PX4](=[OX1])([OX2])([OX2])',
    'iso(thio)cyanate':      '[NX2]=[CX2]=[OX1,SX1]',
    'acyl_cyanide':          '[CX3](=[OX1])[CX2]#[NX1]',
    'disulfide':             '[SX2][SX2]',
    'hydroxamic_acid':       '[CX3](=[OX1])[NX3][OX2H]',
}

# ── Mucosal Irritation Alerts (OSCC-Specific) ─────────────────────────────
# Structural features associated with oral mucosal tissue damage
MUCOSAL_ALERTS = {
    # ── Direct Tissue Reactive (Severity: HIGH) ──
    'aldehyde_reactive':     ('[CX3H1](=O)', 3, 'Schiff base formation with mucosal proteins'),
    'acyl_halide_reactive':  ('[CX3](=[OX1])[F,Cl,Br,I]', 3, 'Acylation of tissue nucleophiles'),
    'epoxide_reactive':      ('C1OC1', 3, 'Alkylation of mucosal epithelium'),
    'michael_acceptor_cr':   ('[CX3]=[CX3][CX3]=[OX1]', 3, 'Covalent protein adducts via conjugate addition'),
    'nitrogen_mustard':      ('ClCCN(CCCl)', 3, 'DNA/protein crosslinking'),
    'sulfur_mustard':        ('ClCCSCCCl', 3, 'Vesicant — severe mucosal damage'),
    'isocyanate':            ('[NX2]=[CX2]=[OX1]', 3, 'Protein crosslinking via isocyanate'),
    'isothiocyanate':        ('[NX2]=[CX2]=[SX1]', 2, 'Protein crosslinking (milder than isocyanate)'),
    'acid_anhydride':        ('[CX3](=[OX1])[OX2][CX3](=[OX1])', 3, 'Rapid acylation of mucosa'),
    'beta_lactone':          ('C1C(=O)OC1', 3, 'Alkylating strained lactone'),
    'aziridine':             ('C1CN1', 3, 'Alkylating strained aziridine'),

    # ── Redox / ROS Generators (Severity: HIGH) ──
    'quinone':               ('O=C1C=CC(=O)C=C1', 2, 'Redox cycling generates ROS in oral tissue'),
    'para_quinone':          ('[#6]1(=[OX1])[#6]=[#6][#6](=[OX1])[#6]=[#6]1', 2, 'Para-quinone redox cycling'),
    'peroxide':              ('[OX2][OX2]', 3, 'Oxidative mucosal damage'),
    'hydroxylamine':         ('[NX3;H1,H2][OX2H]', 2, 'Oxidative stress via nitroso intermediates'),

    # ── Corrosive / pH-Extreme (Severity: MODERATE-HIGH) ──
    'sulfonic_acid':         ('[SX4](=[OX1])(=[OX1])[OX2H]', 2, 'Strong acid — tissue corrosion (pKa ~ −1)'),
    'phosphoric_acid':       ('[PX4](=[OX1])([OX2H])([OX2H])', 1, 'Moderate acid — local pH disruption'),
    'strong_base_guanidine': ('[NX3]C(=[NX2])[NX3]', 1, 'Strong base (pKa ~12.5) — pH-mediated damage'),
    'strong_base_amidine':   ('[NX3]C(=[NX2])[#6]', 1, 'Strong base — local pH disruption'),

    # ── Membrane Disruptors (Severity: MODERATE) ──
    'detergent_sulfate':     ('[OX2H][SX4](=[OX1])(=[OX1])[OX2]', 2, 'Surfactant — membrane solubilization'),
    'long_alkyl_chain':      ('CCCCCCCCCCCC', 1, 'Amphiphilic chain — membrane disruption'),

    # ── Protein-Reactive Warheads (Severity: MODERATE) ──
    'vinyl_sulfone':         ('[SX4](=[OX1])(=[OX1])C=C', 2, 'Covalent cysteine modifier'),
    'acrylamide':            ('[NX3]C(=O)C=C', 1, 'Covalent warhead — reacts with oral keratins'),
    'chloroacetamide':       ('ClCC(=O)[NX3]', 2, 'Alkylating agent targeting thiols'),
    'alpha_haloketone':      ('[F,Cl,Br,I]CC(=O)[#6]', 2, 'Alpha-halogenated carbonyl — electrophilic'),
    'sulfonyl_halide_mr':    ('[SX4](=[OX1])(=[OX1])[F,Cl,Br,I]', 3, 'Highly reactive sulfonylating agent'),
}

# Compile SMARTS patterns
_MUCOSAL_COMPILED = {}
for name, (smarts, severity, desc) in MUCOSAL_ALERTS.items():
    pat = Chem.MolFromSmarts(smarts)
    if pat:
        _MUCOSAL_COMPILED[name] = (pat, severity, desc)

_TOXICO_COMPILED = {}
for name, smarts in TOXICOPHORES.items():
    pat = Chem.MolFromSmarts(smarts)
    if pat:
        _TOXICO_COMPILED[name] = pat

_EXTENDED_COMPILED = {}
for name, smarts in EXTENDED_ALERTS.items():
    pat = Chem.MolFromSmarts(smarts)
    if pat:
        _EXTENDED_COMPILED[name] = pat


# ══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════════════════════

def sa_score(mol):
    """Compute Synthetic Accessibility score."""
    if SA_AVAILABLE:
        try:
            return round(sascorer.calculateScore(mol), 2)
        except Exception:
            return float('nan')
    return float('nan')


def compute_amphiphilic_index(mol):
    """Amphiphilic index: high logP + ionizable groups = membrane disruption risk."""
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    # Amphiphilic molecules: high logP AND significant polar surface
    # True amphiphiles have logP > 3 AND TPSA in 40-90 range
    if logp > 3.0 and 30 < tpsa < 100:
        return round(logp * (tpsa / 60), 2)  # Normalized amphiphilic index
    return 0.0


def estimate_caco2_permeability(mol):
    """Estimate Caco-2 permeability class from descriptors.
    Returns: 'High', 'Moderate', 'Low'
    Based on MW, TPSA, LogP, HBD, rotatable bonds.
    """
    mw = Descriptors.MolWt(mol)
    tpsa = Descriptors.TPSA(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)

    # Egan model approximation: good absorption if TPSA ≤ 131.6 and LogP ≤ 5.88
    if tpsa <= 90 and logp >= 0 and logp <= 5 and mw <= 500 and hbd <= 3:
        return 'High'
    elif tpsa <= 140 and logp >= -1 and logp <= 6 and mw <= 600:
        return 'Moderate'
    return 'Low'


def estimate_bbb_permeability(mol):
    """Estimate BBB permeability (relevant for CNS toxicity assessment).
    BBB+ if: MW ≤ 450, TPSA ≤ 90, HBD ≤ 3, LogP 1-5
    """
    mw = Descriptors.MolWt(mol)
    tpsa = Descriptors.TPSA(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    if mw <= 450 and tpsa <= 90 and hbd <= 3 and 1 <= logp <= 5:
        return 'Permeable'
    elif mw <= 500 and tpsa <= 120 and hbd <= 5:
        return 'Moderate'
    return 'Impermeable'


def estimate_ppb_class(mol):
    """Estimate plasma protein binding from descriptors.
    High PPB (>90%): logP > 3.5/highly lipophilic
    """
    logp = Descriptors.MolLogP(mol)
    if logp > 4.0:
        return 'High (>90%)'
    elif logp > 2.5:
        return 'Moderate (50-90%)'
    return 'Low (<50%)'


CYP_INHIBITOR_SMARTS = {
    'CYP3A4_inhibitor': [
        '[#7]1[#6][#6][#7][#6]1',           # Imidazole — strong CYP3A4 inhibitor
        'c1cnc2[nH]ccc2c1',                 # Benzimidazole
        'c1cc2nc[nH]c2cc1',                 # Benzimidazole variant
    ],
    'CYP2D6_inhibitor': [
        'c1ccc(cc1)[NX3;H1,H2]CCc2ccccc2', # Phenethylamine scaffold
    ],
    'CYP2C9_inhibitor': [
        'c1ccc(-c2cc(=O)c3ccccc3o2)cc1',    # Flavone scaffold
    ],
    'CYP1A2_inhibitor': [
        'c1ccc2c(c1)ccc1ccccc12',            # Naphthalene/PAH
    ],
}

_CYP_COMPILED = {}
for cyp, smarts_list in CYP_INHIBITOR_SMARTS.items():
    pats = []
    for s in smarts_list:
        pat = Chem.MolFromSmarts(s)
        if pat:
            pats.append(pat)
    _CYP_COMPILED[cyp] = pats


def assess_cyp_inhibition(mol):
    """Screen for CYP inhibitor scaffolds. Returns dict of CYP → bool."""
    results = {}
    for cyp, pats in _CYP_COMPILED.items():
        results[cyp] = any(mol.HasSubstructMatch(p) for p in pats)
    return results


# hERG liability SMARTS (simplified structural alerts)
HERG_ALERTS = [
    '[NX3;H0](C)(C)CCc1ccccc1',     # Tertiary amine + phenyl (classic hERG)
    '[NX3;H0](C)(C)CCOc1ccccc1',    # Tertiary amine + phenoxyethyl
]
_HERG_COMPILED = [Chem.MolFromSmarts(s) for s in HERG_ALERTS if Chem.MolFromSmarts(s)]


def assess_herg_risk(mol):
    """Estimate hERG channel liability from structural alerts + descriptors."""
    # Structural alerts
    struct_hit = any(mol.HasSubstructMatch(p) for p in _HERG_COMPILED)

    # Descriptor-based: lipophilic bases are hERG-susceptible
    logp = Descriptors.MolLogP(mol)
    bpa = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7
              and a.GetTotalDegree() >= 3 and a.GetFormalCharge() == 0)  # basic N

    if struct_hit or (logp > 3.5 and bpa >= 1):
        return 'High'
    elif logp > 2.5 and bpa >= 1:
        return 'Moderate'
    return 'Low'


# ══════════════════════════════════════════════════════════════════════════════
# Module A: Compound Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_compounds():
    """Load all compounds from Phase 7 and 7b results."""
    print("\n" + "=" * 70)
    print("MODULE A: Compound Loading")
    print("=" * 70)

    compounds = {}

    # Phase 7 single-target results
    if PHASE7_RESULTS.exists():
        df7 = pd.read_csv(PHASE7_RESULTS, sep="\t")
        valid7 = df7.dropna(subset=["best_energy_kcal"])
        for _, row in valid7.iterrows():
            cid = row["cpd_id"]
            if cid not in compounds:
                compounds[cid] = {
                    "cpd_id": cid,
                    "smiles": row["smiles"],
                    "source": row["source"],
                    "method": row.get("method", ""),
                    "compound_name": row.get("compound_name", ""),
                    "composite_score": row.get("composite_score", 0),
                    "vp_similarity": row.get("vp_similarity", 0),
                    "overall_flag_prev": row.get("overall_flag", ""),
                    "phase7_target": row["target"],
                    "phase7_energy": row["best_energy_kcal"],
                }
        print(f"  Phase 7: {len(compounds)} unique compounds")

    # Phase 7b multi-target selectivity profiles
    mt_data = {}
    if PHASE7B_PROFILES.exists():
        prof = pd.read_csv(PHASE7B_PROFILES, sep="\t")
        for _, row in prof.iterrows():
            cid = row["cpd_id"]
            mt_data[cid] = {
                "mt_best_energy": row.get("best_energy"),
                "mt_mean_energy": row.get("mean_energy"),
                "mt_n_hits_le7": row.get("n_hits_le7", 0),
                "mt_n_hits_le8": row.get("n_hits_le8", 0),
                "in_phase7b": True,
            }
            # Ensure compound exists
            if cid not in compounds:
                compounds[cid] = {
                    "cpd_id": cid,
                    "smiles": row["smiles"],
                    "source": row["source"],
                    "method": row.get("method", ""),
                    "compound_name": row.get("compound_name", ""),
                    "composite_score": row.get("composite_score", 0),
                    "overall_flag_prev": row.get("overall_flag", ""),
                    "phase7_target": "",
                    "phase7_energy": row.get("best_energy", 0),
                }
        print(f"  Phase 7b: {len(mt_data)} multi-target profiles loaded")

    # Merge
    for cid, mt in mt_data.items():
        if cid in compounds:
            compounds[cid].update(mt)

    df = pd.DataFrame(list(compounds.values()))
    # Fill missing mt columns
    for col in ["mt_best_energy", "mt_mean_energy", "mt_n_hits_le7", "mt_n_hits_le8", "in_phase7b"]:
        if col not in df.columns:
            df[col] = None
    df["in_phase7b"] = df["in_phase7b"].fillna(False)

    print(f"  Total compounds: {len(df)}")
    print(f"  In Phase 7b (multi-target): {df['in_phase7b'].sum()}")
    print(f"  Sources: {df['source'].value_counts().to_dict()}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Module B: Physicochemical Properties
# ══════════════════════════════════════════════════════════════════════════════

def compute_physicochemical(df):
    """Compute comprehensive physicochemical properties."""
    print("\n" + "=" * 70)
    print("MODULE B: Physicochemical Properties")
    print("=" * 70)

    props = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        p = {"cpd_id": row["cpd_id"]}

        if mol is None:
            p.update({k: None for k in [
                "mw", "logp", "hba", "hbd", "tpsa", "n_rotatable", "n_rings",
                "n_aromatic_rings", "n_aliphatic_rings", "n_heteroatoms",
                "fsp3", "qed", "sa_score", "molar_refractivity",
                "lipinski_violations", "veber_violations", "ghose_violations",
                "egan_violations",
            ]})
            props.append(p)
            continue

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        tpsa = Descriptors.TPSA(mol)
        rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
        n_rings = rdMolDescriptors.CalcNumRings(mol)
        n_arom = rdMolDescriptors.CalcNumAromaticRings(mol)
        n_ali = rdMolDescriptors.CalcNumAliphaticRings(mol)
        n_het = rdMolDescriptors.CalcNumHeteroatoms(mol)
        fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
        qed_val = Descriptors.qed(mol)
        sa = sa_score(mol)
        mr = Descriptors.MolMR(mol)

        # Lipinski Rule of Five
        lip_v = int(mw > 500) + int(logp > 5) + int(hba > 10) + int(hbd > 5)

        # Veber rules (oral bioavailability): rotatable ≤10, TPSA ≤140
        veber_v = int(rot > 10) + int(tpsa > 140)

        # Ghose filter: 160≤MW≤480, −0.4≤logP≤5.6, 40≤MR≤130, 20≤atoms≤70
        n_atoms = mol.GetNumAtoms()
        ghose_v = (int(mw < 160 or mw > 480) + int(logp < -0.4 or logp > 5.6) +
                   int(mr < 40 or mr > 130) + int(n_atoms < 20 or n_atoms > 70))

        # Egan model (absorption): TPSA ≤131.6, logP ≤5.88
        egan_v = int(tpsa > 131.6) + int(logp > 5.88)

        p.update({
            "mw": round(mw, 1), "logp": round(logp, 2), "hba": hba, "hbd": hbd,
            "tpsa": round(tpsa, 1), "n_rotatable": rot, "n_rings": n_rings,
            "n_aromatic_rings": n_arom, "n_aliphatic_rings": n_ali,
            "n_heteroatoms": n_het, "fsp3": round(fsp3, 3), "qed": round(qed_val, 3),
            "sa_score": sa, "molar_refractivity": round(mr, 1),
            "lipinski_violations": lip_v, "veber_violations": veber_v,
            "ghose_violations": ghose_v, "egan_violations": egan_v,
        })
        props.append(p)

    prop_df = pd.DataFrame(props)
    for col in prop_df.columns:
        if col != "cpd_id":
            df[col] = prop_df[col].values

    # Summary
    print(f"  Drug-likeness filter pass rates:")
    print(f"    Lipinski (≤1 violation): {(df['lipinski_violations'] <= 1).sum()}/{len(df)} "
          f"({(df['lipinski_violations'] <= 1).sum()*100/len(df):.0f}%)")
    print(f"    Veber (0 violations):    {(df['veber_violations'] == 0).sum()}/{len(df)} "
          f"({(df['veber_violations'] == 0).sum()*100/len(df):.0f}%)")
    print(f"    Ghose (0 violations):    {(df['ghose_violations'] == 0).sum()}/{len(df)} "
          f"({(df['ghose_violations'] == 0).sum()*100/len(df):.0f}%)")
    print(f"    Egan (0 violations):     {(df['egan_violations'] == 0).sum()}/{len(df)} "
          f"({(df['egan_violations'] == 0).sum()*100/len(df):.0f}%)")
    print(f"  QED: mean={df['qed'].mean():.3f}, median={df['qed'].median():.3f}")
    print(f"  SA score: mean={df['sa_score'].mean():.2f}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Module C: ADMET Prediction
# ══════════════════════════════════════════════════════════════════════════════

def compute_admet(df):
    """Predict ADMET properties from molecular descriptors."""
    print("\n" + "=" * 70)
    print("MODULE C: ADMET Prediction")
    print("=" * 70)

    admet = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        a = {"cpd_id": row["cpd_id"]}

        if mol is None:
            a.update({
                "caco2_class": None, "bbb_class": None, "ppb_class": None,
                "herg_risk": None, "amphiphilic_index": None,
                "cyp3a4_inhibitor": None, "cyp2d6_inhibitor": None,
                "cyp2c9_inhibitor": None, "cyp1a2_inhibitor": None,
                "n_cyp_flags": 0,
            })
            admet.append(a)
            continue

        # Absorption
        a["caco2_class"] = estimate_caco2_permeability(mol)

        # Distribution
        a["bbb_class"] = estimate_bbb_permeability(mol)
        a["ppb_class"] = estimate_ppb_class(mol)

        # Metabolism — CYP inhibition
        cyp = assess_cyp_inhibition(mol)
        for k, v in cyp.items():
            a[k] = v
        a["n_cyp_flags"] = sum(1 for v in cyp.values() if v)

        # Toxicity — hERG
        a["herg_risk"] = assess_herg_risk(mol)

        # Amphiphilic index
        a["amphiphilic_index"] = compute_amphiphilic_index(mol)

        admet.append(a)

    admet_df = pd.DataFrame(admet)
    for col in admet_df.columns:
        if col != "cpd_id":
            df[col] = admet_df[col].values

    # Summary
    print(f"  Caco-2 permeability: {(df['caco2_class'] == 'High').sum()} High, "
          f"{(df['caco2_class'] == 'Moderate').sum()} Moderate, "
          f"{(df['caco2_class'] == 'Low').sum()} Low")
    print(f"  BBB permeability: {(df['bbb_class'] == 'Permeable').sum()} Permeable, "
          f"{(df['bbb_class'] == 'Moderate').sum()} Moderate, "
          f"{(df['bbb_class'] == 'Impermeable').sum()} Impermeable")
    print(f"  hERG risk: {(df['herg_risk'] == 'High').sum()} High, "
          f"{(df['herg_risk'] == 'Moderate').sum()} Moderate, "
          f"{(df['herg_risk'] == 'Low').sum()} Low")
    print(f"  CYP inhibitor flags: mean={df['n_cyp_flags'].mean():.1f}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Module D: Structural Alert Screening
# ══════════════════════════════════════════════════════════════════════════════

def screen_structural_alerts(df):
    """Screen for PAINS, Brenk, toxicophores, and extended alerts."""
    print("\n" + "=" * 70)
    print("MODULE D: Structural Alert Screening")
    print("=" * 70)

    alerts = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        a = {"cpd_id": row["cpd_id"]}

        if mol is None:
            a.update({
                "pains_count": 0, "pains_names": "",
                "brenk_count": 0, "brenk_names": "",
                "toxicophore_count": 0, "toxicophore_names": "",
                "extended_alert_count": 0, "extended_alert_names": "",
                "total_structural_alerts": 0,
            })
            alerts.append(a)
            continue

        # PAINS
        pains_m = PAINS_CATALOG.GetMatches(mol)
        a["pains_count"] = len(pains_m)
        a["pains_names"] = ";".join(m.GetDescription() for m in pains_m)

        # Brenk
        brenk_m = BRENK_CATALOG.GetMatches(mol)
        a["brenk_count"] = len(brenk_m)
        a["brenk_names"] = ";".join(m.GetDescription() for m in brenk_m)

        # Toxicophores
        toxico_hits = []
        for tname, tpat in _TOXICO_COMPILED.items():
            if mol.HasSubstructMatch(tpat):
                toxico_hits.append(tname)
        a["toxicophore_count"] = len(toxico_hits)
        a["toxicophore_names"] = ";".join(toxico_hits)

        # Extended alerts
        ext_hits = []
        for ename, epat in _EXTENDED_COMPILED.items():
            if mol.HasSubstructMatch(epat):
                ext_hits.append(ename)
        a["extended_alert_count"] = len(ext_hits)
        a["extended_alert_names"] = ";".join(ext_hits)

        a["total_structural_alerts"] = (a["pains_count"] + a["toxicophore_count"]
                                        + a["extended_alert_count"])

        alerts.append(a)

    alert_df = pd.DataFrame(alerts)
    for col in alert_df.columns:
        if col != "cpd_id":
            df[col] = alert_df[col].values

    print(f"  PAINS alerts: {(df['pains_count'] > 0).sum()}/{len(df)} compounds flagged")
    print(f"  Brenk alerts: {(df['brenk_count'] > 0).sum()}/{len(df)} compounds flagged")
    print(f"  Toxicophores: {(df['toxicophore_count'] > 0).sum()}/{len(df)} compounds flagged")
    print(f"  Extended alerts: {(df['extended_alert_count'] > 0).sum()}/{len(df)} compounds flagged")
    print(f"  Clean (0 structural alerts): {(df['total_structural_alerts'] == 0).sum()}/{len(df)}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Module E: Mucosal Irritation Filter (OSCC-Specific)
# ══════════════════════════════════════════════════════════════════════════════

def assess_mucosal_safety(df):
    """OSCC-specific mucosal irritation risk assessment."""
    print("\n" + "=" * 70)
    print("MODULE E: Mucosal Irritation Filter (OSCC-Specific)")
    print("=" * 70)

    mucosal = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        m = {"cpd_id": row["cpd_id"]}

        if mol is None:
            m.update({
                "mucosal_alerts": "", "mucosal_alert_count": 0,
                "mucosal_max_severity": 0, "mucosal_severity_sum": 0,
                "amphiphilic_risk": False, "extreme_ph_risk": False,
                "mucosal_risk": "UNKNOWN",
            })
            mucosal.append(m)
            continue

        # ── SMARTS-based mucosal alerts ──
        alert_hits = []
        max_severity = 0
        severity_sum = 0
        for aname, (apat, sev, desc) in _MUCOSAL_COMPILED.items():
            if mol.HasSubstructMatch(apat):
                alert_hits.append(f"{aname}(S{sev})")
                max_severity = max(max_severity, sev)
                severity_sum += sev

        m["mucosal_alerts"] = ";".join(alert_hits)
        m["mucosal_alert_count"] = len(alert_hits)
        m["mucosal_max_severity"] = max_severity
        m["mucosal_severity_sum"] = severity_sum

        # ── Amphiphilic risk (physicochemical) ──
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        # Amphiphilic molecules disrupt oral mucosal membranes
        amphiphilic = logp > 4.0 and 30 < tpsa < 90
        m["amphiphilic_risk"] = amphiphilic

        # ── Extreme pH risk (need pH < 2 or > 10 for dissolution) ──
        # Detect strong acid/base functional groups
        has_strong_acid = any(
            mol.HasSubstructMatch(p) for p in [
                Chem.MolFromSmarts('[SX4](=[OX1])(=[OX1])[OX2H]'),  # sulfonic acid
            ] if p
        )
        has_strong_base = any(
            mol.HasSubstructMatch(p) for p in [
                Chem.MolFromSmarts('[NX3]C(=[NX2])[NX3]'),  # guanidine
            ] if p
        )
        m["extreme_ph_risk"] = has_strong_acid or has_strong_base

        # ── Integrated mucosal risk classification ──
        if max_severity >= 3 or severity_sum >= 5:
            m["mucosal_risk"] = "HIGH"
        elif max_severity >= 2 or severity_sum >= 3 or amphiphilic:
            m["mucosal_risk"] = "MODERATE"
        elif len(alert_hits) > 0 or m["extreme_ph_risk"]:
            m["mucosal_risk"] = "LOW"
        else:
            m["mucosal_risk"] = "SAFE"

        mucosal.append(m)

    mucosal_df = pd.DataFrame(mucosal)
    for col in mucosal_df.columns:
        if col != "cpd_id":
            df[col] = mucosal_df[col].values

    # Save detailed mucosal safety report
    mucosal_df.to_csv(MUCOSAL_OUT, sep="\t", index=False)

    # Summary
    print(f"  Mucosal risk distribution:")
    for risk in ["SAFE", "LOW", "MODERATE", "HIGH"]:
        n = (df["mucosal_risk"] == risk).sum()
        pct = n * 100 / len(df) if len(df) > 0 else 0
        print(f"    {risk:10s}: {n:4d} ({pct:.1f}%)")

    flagged = df[df["mucosal_alert_count"] > 0]
    if len(flagged) > 0:
        print(f"\n  Top mucosal alerts across all compounds:")
        all_alerts = []
        for s in flagged["mucosal_alerts"]:
            all_alerts.extend(s.split(";"))
        from collections import Counter
        for alert, cnt in Counter(all_alerts).most_common(10):
            print(f"    {cnt:3d}x  {alert}")

    print(f"\n  Amphiphilic risk: {df['amphiphilic_risk'].sum()} compounds")
    print(f"  Extreme pH risk: {df['extreme_ph_risk'].sum()} compounds")
    print(f"  Saved: {MUCOSAL_OUT.name}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Module F: Integrated ADMET Scoring & Prioritization
# ══════════════════════════════════════════════════════════════════════════════

def compute_integrated_score(df):
    """Compute integrated ADMET score combining all modules."""
    print("\n" + "=" * 70)
    print("MODULE F: Integrated ADMET Scoring")
    print("=" * 70)

    scores = []
    for _, row in df.iterrows():
        # ── Component scores (each 0-1, higher = better) ──

        # 1. Drug-likeness (Lipinski + Veber + QED)
        lip = float(row.get("lipinski_violations", 3) or 3)
        veb = float(row.get("veber_violations", 2) or 2)
        qed = float(row.get("qed", 0) or 0)
        s_druglike = max(0, 1.0 - 0.2 * lip) * max(0, 1.0 - 0.3 * veb) * max(0.1, qed)
        s_druglike = min(1.0, s_druglike ** 0.5)  # Geometric mean softening

        # 2. Absorption (Caco-2 + Egan)
        caco2_map = {"High": 1.0, "Moderate": 0.6, "Low": 0.2, None: 0.2}
        egan = float(row.get("egan_violations", 2) or 2)
        s_absorption = caco2_map.get(row.get("caco2_class"), 0.2) * max(0, 1.0 - 0.4 * egan)

        # 3. Safety (structural alerts)
        pains = float(row.get("pains_count", 0) or 0)
        toxico = float(row.get("toxicophore_count", 0) or 0)
        brenk = float(row.get("brenk_count", 0) or 0)
        ext = float(row.get("extended_alert_count", 0) or 0)
        s_safety = max(0, 1.0 - 0.30 * pains - 0.25 * toxico
                       - 0.10 * min(brenk, 2) - 0.15 * min(ext, 2))

        # 4. Mucosal safety (OSCC-specific)
        mucosal_map = {"SAFE": 1.0, "LOW": 0.8, "MODERATE": 0.4, "HIGH": 0.1, "UNKNOWN": 0.2}
        s_mucosal = mucosal_map.get(row.get("mucosal_risk", "UNKNOWN"), 0.2)

        # 5. Cardiac safety (hERG)
        herg_map = {"Low": 1.0, "Moderate": 0.6, "High": 0.2, None: 0.5}
        s_cardiac = herg_map.get(row.get("herg_risk"), 0.5)

        # 6. Metabolic (CYP + SA)
        n_cyp = float(row.get("n_cyp_flags", 0) or 0)
        sa = float(row.get("sa_score", 5) or 5)
        s_sa = 1.0 if sa <= 4 else (max(0, (7 - sa) / 3 * 0.5 + 0.5) if sa <= 7 else 0.0)
        s_metabolic = max(0, 1.0 - 0.2 * n_cyp) * s_sa

        # ── Integrated ADMET score (weighted geometric mean) ──
        # Weights emphasize mucosal safety for OSCC context
        components = [
            (s_druglike, 0.20),
            (s_absorption, 0.15),
            (s_safety, 0.20),
            (s_mucosal, 0.20),    # Elevated weight for OSCC
            (s_cardiac, 0.10),
            (s_metabolic, 0.15),
        ]

        # Weighted geometric mean
        log_sum = sum(w * math.log(max(s, 1e-6)) for s, w in components)
        admet_score = math.exp(log_sum)

        scores.append({
            "cpd_id": row["cpd_id"],
            "s_druglike": round(s_druglike, 3),
            "s_absorption": round(s_absorption, 3),
            "s_safety": round(s_safety, 3),
            "s_mucosal": round(s_mucosal, 3),
            "s_cardiac": round(s_cardiac, 3),
            "s_metabolic": round(s_metabolic, 3),
            "admet_score": round(admet_score, 4),
        })

    score_df = pd.DataFrame(scores)
    for col in score_df.columns:
        if col != "cpd_id":
            df[col] = score_df[col].values

    # Summary
    print(f"  ADMET score distribution:")
    print(f"    Mean: {df['admet_score'].mean():.3f}")
    print(f"    Median: {df['admet_score'].median():.3f}")
    print(f"    ≥0.5 (favorable): {(df['admet_score'] >= 0.5).sum()} ({(df['admet_score'] >= 0.5).sum()*100/len(df):.0f}%)")
    print(f"    ≥0.6 (good):      {(df['admet_score'] >= 0.6).sum()} ({(df['admet_score'] >= 0.6).sum()*100/len(df):.0f}%)")
    print(f"    ≥0.7 (excellent): {(df['admet_score'] >= 0.7).sum()} ({(df['admet_score'] >= 0.7).sum()*100/len(df):.0f}%)")

    print(f"\n  Component score means:")
    for comp in ["s_druglike", "s_absorption", "s_safety", "s_mucosal", "s_cardiac", "s_metabolic"]:
        print(f"    {comp:15s}: {df[comp].mean():.3f}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Module G: Final Candidate Shortlist
# ══════════════════════════════════════════════════════════════════════════════

def generate_final_shortlist(df):
    """Generate prioritized final candidate table."""
    print("\n" + "=" * 70)
    print("MODULE G: Final Candidate Shortlist")
    print("=" * 70)

    # ── Combined score: docking + contrastive + ADMET ──
    # Normalize docking energy to 0-1 (more negative = better)
    dock_col = "mt_best_energy" if "mt_best_energy" in df.columns else "phase7_energy"
    dock_vals = df[dock_col].fillna(0)
    dock_min, dock_max = dock_vals.min(), dock_vals.max()
    if dock_min != dock_max:
        df["dock_norm"] = (dock_vals - dock_max) / (dock_min - dock_max)
    else:
        df["dock_norm"] = 0.5

    # Normalize composite score to 0-1
    comp_vals = df["composite_score"].fillna(0)
    comp_min, comp_max = comp_vals.min(), comp_vals.max()
    if comp_min != comp_max:
        df["composite_norm"] = (comp_vals - comp_min) / (comp_max - comp_min)
    else:
        df["composite_norm"] = 0.5

    # Final integrated score
    df["final_score"] = (
        0.30 * df["dock_norm"] +
        0.30 * df["composite_norm"] +
        0.40 * df["admet_score"]
    )

    # ── ADMET flag ──
    def admet_flag(row):
        if (row.get("admet_score", 0) >= 0.5 and
            row.get("mucosal_risk") in ("SAFE", "LOW") and
            row.get("pains_count", 0) == 0 and
            row.get("herg_risk") != "High"):
            return "PASS"
        elif (row.get("admet_score", 0) >= 0.3 and
              row.get("mucosal_risk") != "HIGH" and
              row.get("pains_count", 0) <= 1):
            return "CAUTION"
        return "FAIL"

    df["admet_flag"] = df.apply(admet_flag, axis=1)

    # Summary
    n_pass = (df["admet_flag"] == "PASS").sum()
    n_caution = (df["admet_flag"] == "CAUTION").sum()
    n_fail = (df["admet_flag"] == "FAIL").sum()
    print(f"  ADMET flag: PASS={n_pass}, CAUTION={n_caution}, FAIL={n_fail}")

    # ── Final shortlist: PASS or CAUTION, sorted by final_score ──
    shortlist = df[df["admet_flag"].isin(["PASS", "CAUTION"])].sort_values(
        "final_score", ascending=False
    )

    # Select columns for final table
    final_cols = [
        "cpd_id", "smiles", "source", "method", "compound_name",
        "phase7_energy", "phase7_target",
        "mt_best_energy", "mt_n_hits_le7", "mt_n_hits_le8", "in_phase7b",
        "composite_score", "vp_similarity",
        "mw", "logp", "hba", "hbd", "tpsa", "n_rotatable", "qed", "sa_score",
        "lipinski_violations", "veber_violations",
        "caco2_class", "bbb_class", "ppb_class", "herg_risk", "n_cyp_flags",
        "pains_count", "brenk_count", "toxicophore_count", "total_structural_alerts",
        "mucosal_risk", "mucosal_alert_count", "mucosal_severity_sum",
        "amphiphilic_risk", "extreme_ph_risk",
        "s_druglike", "s_absorption", "s_safety", "s_mucosal", "s_cardiac", "s_metabolic",
        "admet_score", "admet_flag",
        "dock_norm", "composite_norm", "final_score",
    ]
    existing_cols = [c for c in final_cols if c in df.columns]

    shortlist_out = shortlist[existing_cols]
    shortlist_out.to_csv(FINAL_OUT, sep="\t", index=False)

    print(f"\n  Final shortlist: {len(shortlist_out)} candidates")
    print(f"    PASS: {(shortlist_out['admet_flag'] == 'PASS').sum()}")
    print(f"    CAUTION: {(shortlist_out['admet_flag'] == 'CAUTION').sum()}")

    # Top 20
    print(f"\n  === Top 20 Final Candidates ===")
    for i, (_, r) in enumerate(shortlist_out.head(20).iterrows()):
        name = r.get("compound_name", "") if pd.notna(r.get("compound_name", "")) else r.get("method", "")
        mt_info = f"MT:{int(r['mt_n_hits_le7'])}/7" if pd.notna(r.get("mt_n_hits_le7")) else ""
        print(f"  {i+1:2d}. {r['cpd_id']:30s} [{r['source']:10s}] "
              f"final={r['final_score']:.3f} ADMET={r['admet_score']:.3f} "
              f"dock={r.get('mt_best_energy', r.get('phase7_energy', '?')):.1f} "
              f"mucosal={r['mucosal_risk']:8s} {r['admet_flag']:7s} "
              f"{mt_info} {name}")

    # ── Safety-potency analysis ──
    print(f"\n  === Safety-Potency Analysis ===")
    sp = df[["cpd_id", "source", "admet_flag", "admet_score", "mucosal_risk",
             "phase7_energy", "composite_score", "final_score"]].copy()
    if "mt_best_energy" in df.columns:
        sp["mt_best_energy"] = df["mt_best_energy"]
        sp["mt_n_hits_le7"] = df["mt_n_hits_le7"]
    sp.to_csv(SAFETY_POT_OUT, sep="\t", index=False)

    for flag in ["PASS", "CAUTION", "FAIL"]:
        sub = df[df["admet_flag"] == flag]
        if len(sub) > 0:
            mean_dock = sub["phase7_energy"].mean()
            mean_admet = sub["admet_score"].mean()
            mean_final = sub["final_score"].mean()
            print(f"  {flag:7s}: n={len(sub):3d}, mean_dock={mean_dock:.1f}, "
                  f"mean_admet={mean_admet:.3f}, mean_final={mean_final:.3f}")

    # ── Phase 7b elite intersection ──
    if "in_phase7b" in df.columns:
        mt_elite = df[(df["in_phase7b"] == True) & (df["admet_flag"] == "PASS")]
        print(f"\n  === Triple-Validated Elite Candidates ===")
        print(f"  (Multi-target binder + ADMET PASS + docking validated)")
        print(f"  Total: {len(mt_elite)}")
        if len(mt_elite) > 0:
            top_elite = mt_elite.sort_values("final_score", ascending=False)
            for _, r in top_elite.head(10).iterrows():
                name = r.get("compound_name", "") if pd.notna(r.get("compound_name", "")) else r.get("method", "")
                print(f"    {r['cpd_id']:30s} final={r['final_score']:.3f} "
                      f"dock={r.get('mt_best_energy', '?'):.1f} "
                      f"ADMET={r['admet_score']:.3f} mucosal={r['mucosal_risk']} "
                      f"MT:{int(r.get('mt_n_hits_le7', 0))}/7  {name}")

    return df, shortlist_out


# ══════════════════════════════════════════════════════════════════════════════
# Summary & Output
# ══════════════════════════════════════════════════════════════════════════════

def save_summary(df, shortlist):
    """Save comprehensive summary JSON."""
    print("\n" + "=" * 70)
    print("Saving Results")
    print("=" * 70)

    # Save full profiles
    df.to_csv(ADMET_OUT, sep="\t", index=False)
    print(f"  Full ADMET profiles: {ADMET_OUT.name} ({len(df)} compounds)")

    summary = {
        "phase": 8,
        "description": "ADMET & Pharmacokinetic Profiling with Mucosal Irritation Filter",
        "n_compounds": len(df),
        "n_phase7b_compounds": int(df["in_phase7b"].sum()) if "in_phase7b" in df.columns else 0,

        "physicochemical": {
            "lipinski_pass": int((df["lipinski_violations"] <= 1).sum()),
            "veber_pass": int((df["veber_violations"] == 0).sum()),
            "ghose_pass": int((df["ghose_violations"] == 0).sum()),
            "egan_pass": int((df["egan_violations"] == 0).sum()),
            "mean_qed": round(df["qed"].mean(), 3),
            "mean_mw": round(df["mw"].mean(), 1),
            "mean_logp": round(df["logp"].mean(), 2),
        },

        "admet": {
            "caco2_high": int((df["caco2_class"] == "High").sum()),
            "caco2_moderate": int((df["caco2_class"] == "Moderate").sum()),
            "caco2_low": int((df["caco2_class"] == "Low").sum()),
            "bbb_permeable": int((df["bbb_class"] == "Permeable").sum()),
            "herg_high": int((df["herg_risk"] == "High").sum()),
            "herg_moderate": int((df["herg_risk"] == "Moderate").sum()),
            "herg_low": int((df["herg_risk"] == "Low").sum()),
            "mean_cyp_flags": round(df["n_cyp_flags"].mean(), 2),
        },

        "structural_alerts": {
            "pains_flagged": int((df["pains_count"] > 0).sum()),
            "brenk_flagged": int((df["brenk_count"] > 0).sum()),
            "toxicophore_flagged": int((df["toxicophore_count"] > 0).sum()),
            "clean": int((df["total_structural_alerts"] == 0).sum()),
        },

        "mucosal_safety": {
            "safe": int((df["mucosal_risk"] == "SAFE").sum()),
            "low": int((df["mucosal_risk"] == "LOW").sum()),
            "moderate": int((df["mucosal_risk"] == "MODERATE").sum()),
            "high": int((df["mucosal_risk"] == "HIGH").sum()),
            "amphiphilic_risk": int(df["amphiphilic_risk"].sum()),
            "extreme_ph_risk": int(df["extreme_ph_risk"].sum()),
        },

        "scoring": {
            "mean_admet_score": round(df["admet_score"].mean(), 4),
            "median_admet_score": round(df["admet_score"].median(), 4),
            "mean_final_score": round(df["final_score"].mean(), 4),
        },

        "final_candidates": {
            "admet_pass": int((df["admet_flag"] == "PASS").sum()),
            "admet_caution": int((df["admet_flag"] == "CAUTION").sum()),
            "admet_fail": int((df["admet_flag"] == "FAIL").sum()),
            "shortlist_size": len(shortlist),
        },
    }

    if "in_phase7b" in df.columns:
        triple_validated = df[(df["in_phase7b"] == True) & (df["admet_flag"] == "PASS")]
        summary["triple_validated_elite"] = {
            "count": len(triple_validated),
            "mean_final_score": round(triple_validated["final_score"].mean(), 4) if len(triple_validated) > 0 else None,
        }

    with open(SUMMARY_OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {SUMMARY_OUT.name}")

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70)
    print("GENESIS — Phase 8: ADMET & Pharmacokinetic Profiling")
    print("with Mucosal Irritation Filter (OSCC-Specific)")
    print("=" * 70)
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Module A: Load compounds
    df = load_compounds()

    # Module B: Physicochemical properties
    df = compute_physicochemical(df)

    # Module C: ADMET prediction
    df = compute_admet(df)

    # Module D: Structural alert screening
    df = screen_structural_alerts(df)

    # Module E: Mucosal irritation filter
    df = assess_mucosal_safety(df)

    # Module F: Integrated scoring
    df = compute_integrated_score(df)

    # Module G: Final candidate shortlist
    df, shortlist = generate_final_shortlist(df)

    # Save summary
    summary = save_summary(df, shortlist)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Phase 8 COMPLETE in {elapsed:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
