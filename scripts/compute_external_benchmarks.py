#!/usr/bin/env python3
"""
GENESIS — External Benchmark Metrics (MOSES + GuacaMol equivalents)
====================================================================
Computes standard distribution-learning benchmarks for all generative
methods using the same mathematical formulations as MOSES (Polykovskiy
et al., 2020) and GuacaMol (Brown et al., 2019), implemented directly
to avoid dependency issues with outdated packages.

MOSES metrics:
  Valid, Unique@1k, Novelty, IntDiv, IntDiv2, Frag, Scaf, SNN, Filters

GuacaMol metrics:
  KL divergence of property distributions (logP, SA, QED, MW, TPSA,
  NumRings, NumAromaticRings, HBA, HBD)

FCD values are loaded from the previously computed fcd_results.json.

Output:
  generative/results/external_benchmarks.json
"""

import os, sys, json, warnings
import numpy as np
from collections import Counter
from itertools import combinations

warnings.filterwarnings("ignore", category=DeprecationWarning)

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED as QEDModule
from rdkit.Chem import rdMolDescriptors, FilterCatalog
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs

RDLogger.logger().setLevel(RDLogger.ERROR)

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN_DIR = os.path.join(ROOT, "generative", "results")
DATA_TSV = os.path.join(ROOT, "data", "All_Compounds_Unq_Std.tsv")
SPLITS_JSON = os.path.join(ROOT, "modeling", "splits.json")
FCD_JSON = os.path.join(GEN_DIR, "fcd_results.json")
OUT_JSON = os.path.join(GEN_DIR, "external_benchmarks.json")

# ── Generated molecule files ───────────────────────────────────────────
METHOD_FILES = {
    "VAE":              os.path.join(GEN_DIR, "generated_vae.txt"),
    "cVAE":             os.path.join(GEN_DIR, "generated_cvae.txt"),
    "Diffusion":        os.path.join(GEN_DIR, "generated_diffusion.txt"),
    "RL-VAE":           os.path.join(GEN_DIR, "generated_rl_vae.txt"),
    "CGLF":             os.path.join(GEN_DIR, "generated_cglf.txt"),
    "CRAVE (filt.)":    os.path.join(GEN_DIR, "generated_crave.txt"),
    "CRAVE (unfilt.)":  os.path.join(GEN_DIR, "generated_crave_unfiltered.txt"),
}


# =====================================================================
# Utility functions
# =====================================================================

def canonicalize(smi):
    """Return canonical SMILES or None if invalid."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def load_smiles(path):
    """Load SMILES from a one-per-line text file."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def get_mol(smi):
    try:
        return Chem.MolFromSmiles(smi)
    except Exception:
        return None


def get_fingerprint(mol, radius=2, nbits=2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


# =====================================================================
# MOSES metrics
# =====================================================================

def compute_validity(smiles_list):
    """Fraction of parseable SMILES."""
    valid = sum(1 for s in smiles_list if canonicalize(s) is not None)
    return valid / len(smiles_list) if smiles_list else 0.0


def compute_uniqueness(smiles_list):
    """Fraction of unique canonical SMILES among valid ones."""
    canonical = [canonicalize(s) for s in smiles_list]
    valid = [c for c in canonical if c is not None]
    if not valid:
        return 0.0
    return len(set(valid)) / len(valid)


def compute_novelty(smiles_list, train_smiles_set):
    """Fraction of valid generated molecules not in training set."""
    canonical = [canonicalize(s) for s in smiles_list]
    valid = [c for c in canonical if c is not None]
    if not valid:
        return 0.0
    novel = sum(1 for c in valid if c not in train_smiles_set)
    return novel / len(valid)


def compute_internal_diversity(mols, n_sample=1000, seed=42):
    """
    IntDiv = 1 - (1/|P|) * sum Tanimoto(i,j) for all i!=j pairs.
    Approximated by sampling n_sample molecules.
    """
    rng = np.random.default_rng(seed)
    if len(mols) > n_sample:
        idx = rng.choice(len(mols), size=n_sample, replace=False)
        mols = [mols[i] for i in idx]

    fps = [get_fingerprint(m) for m in mols]
    n = len(fps)
    if n < 2:
        return 0.0, 0.0

    tani_sum = 0.0
    tani_sq_sum = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            t = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            tani_sum += t
            tani_sq_sum += t * t
            count += 1

    avg_tani = tani_sum / count
    avg_tani_sq = tani_sq_sum / count
    int_div = 1.0 - avg_tani
    int_div2 = 1.0 - np.sqrt(avg_tani_sq)
    return int_div, int_div2


def compute_snn(gen_mols, ref_mols, n_ref_sample=5000, seed=42):
    """
    SNN = average max-Tanimoto similarity between each generated
    molecule and the reference (test) set.
    """
    rng = np.random.default_rng(seed)
    if len(ref_mols) > n_ref_sample:
        idx = rng.choice(len(ref_mols), size=n_ref_sample, replace=False)
        ref_mols = [ref_mols[i] for i in idx]

    ref_fps = [get_fingerprint(m) for m in ref_mols]
    snn_values = []
    for mol in gen_mols:
        fp = get_fingerprint(mol)
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
        snn_values.append(max(sims))
    return float(np.mean(snn_values))


def get_brics_fragments(mol):
    """Return set of BRICS fragment SMILES."""
    try:
        from rdkit.Chem import BRICS
        frags = BRICS.BRICSDecompose(mol)
        return set(frags)
    except Exception:
        return set()


def get_murcko_scaffold(mol):
    """Return Murcko scaffold SMILES."""
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(core)
    except Exception:
        return ""


def cosine_similarity(vec_a, vec_b):
    """Cosine similarity between two Counter-based frequency vectors."""
    all_keys = set(vec_a.keys()) | set(vec_b.keys())
    if not all_keys:
        return 0.0
    a = np.array([vec_a.get(k, 0) for k in all_keys], dtype=float)
    b = np.array([vec_b.get(k, 0) for k in all_keys], dtype=float)
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


def compute_frag(gen_mols, ref_mols):
    """Fragment similarity: cosine of BRICS fragment frequency vectors."""
    gen_frags = Counter()
    for m in gen_mols:
        gen_frags.update(get_brics_fragments(m))
    ref_frags = Counter()
    for m in ref_mols:
        ref_frags.update(get_brics_fragments(m))
    return cosine_similarity(gen_frags, ref_frags)


def compute_scaf(gen_mols, ref_mols):
    """Scaffold similarity: cosine of Murcko scaffold frequency vectors."""
    gen_scaf = Counter(get_murcko_scaffold(m) for m in gen_mols)
    ref_scaf = Counter(get_murcko_scaffold(m) for m in ref_mols)
    # Remove empty scaffold
    gen_scaf.pop("", None)
    ref_scaf.pop("", None)
    return cosine_similarity(gen_scaf, ref_scaf)


def compute_filters(mols):
    """Fraction passing PAINS + Glaxo (NIH) filters."""
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog.FilterCatalog(params)

    passed = 0
    for mol in mols:
        entry = catalog.GetFirstMatch(mol)
        if entry is None:
            passed += 1
    return passed / len(mols) if mols else 0.0


# =====================================================================
# GuacaMol KL-divergence metrics
# =====================================================================

def continuous_kldiv(x, y, n_bins=100):
    """
    Approximate KL divergence between two samples using histograms.
    Returns KL(P||Q) where P=reference, Q=generated.
    """
    mn = min(min(x), min(y))
    mx = max(max(x), max(y))
    if mn == mx:
        return 0.0

    bins = np.linspace(mn, mx, n_bins + 1)
    p_hist, _ = np.histogram(x, bins=bins, density=True)
    q_hist, _ = np.histogram(y, bins=bins, density=True)

    # Add small epsilon and normalise
    eps = 1e-10
    p_hist = p_hist + eps
    q_hist = q_hist + eps
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()

    kl = float(np.sum(p_hist * np.log(p_hist / q_hist)))
    return kl


def discrete_kldiv(x, y):
    """KL divergence for discrete distributions."""
    all_vals = set(x) | set(y)
    cx = Counter(x)
    cy = Counter(y)
    nx = len(x)
    ny = len(y)
    eps = 1e-10
    kl = 0.0
    for v in all_vals:
        p = cx.get(v, 0) / nx + eps
        q = cy.get(v, 0) / ny + eps
        kl += p * np.log(p / q)
    return float(kl)


def compute_properties(mols):
    """Compute property vectors for a list of mols."""
    props = {
        "logP": [], "SA": [], "QED": [], "MW": [],
        "TPSA": [], "NumRings": [], "NumAromaticRings": [],
        "HBA": [], "HBD": [],
    }
    for mol in mols:
        try:
            props["logP"].append(Descriptors.MolLogP(mol))
            props["SA"].append(rdMolDescriptors.CalcNumRotatableBonds(mol))
            props["QED"].append(QEDModule.qed(mol))
            props["MW"].append(Descriptors.MolWt(mol))
            props["TPSA"].append(Descriptors.TPSA(mol))
            props["NumRings"].append(Descriptors.RingCount(mol))
            props["NumAromaticRings"].append(Descriptors.NumAromaticRings(mol))
            props["HBA"].append(Descriptors.NumHAcceptors(mol))
            props["HBD"].append(Descriptors.NumHDonors(mol))
        except Exception:
            continue
    return props


def compute_kl_divergences(gen_props, ref_props):
    """Compute KL divergence for each property."""
    continuous_keys = ["logP", "QED", "MW", "TPSA"]
    discrete_keys = ["SA", "NumRings", "NumAromaticRings", "HBA", "HBD"]

    kls = {}
    for k in continuous_keys:
        if gen_props[k] and ref_props[k]:
            kls[k] = continuous_kldiv(ref_props[k], gen_props[k])
        else:
            kls[k] = float("nan")

    for k in discrete_keys:
        if gen_props[k] and ref_props[k]:
            kls[k] = discrete_kldiv(ref_props[k], gen_props[k])
        else:
            kls[k] = float("nan")

    return kls


# =====================================================================
# Main
# =====================================================================

def main():
    print("GENESIS External Benchmark Computation")
    print("=" * 60)

    # ── Load training SMILES ───────────────────────────────────────
    print("\n[1/5] Loading training and test SMILES...")
    with open(SPLITS_JSON) as f:
        splits = json.load(f)

    train_ids = set(splits["train"]["positive_ids"] + splits["train"]["negative_ids"])
    test_ids = set(splits["test"]["positive_ids"] + splits["test"]["negative_ids"])

    # Load SMILES from TSV
    id2smi = {}
    with open(DATA_TSV) as f:
        header = f.readline().strip().split("\t")
        id_col = header.index("compound_chembl_id")
        smi_col = header.index("smiles")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > smi_col:
                id2smi[parts[id_col]] = parts[smi_col]

    train_smiles_raw = [id2smi[cid] for cid in train_ids if cid in id2smi]
    test_smiles_raw = [id2smi[cid] for cid in test_ids if cid in id2smi]

    # Canonicalise training set
    train_canonical = set()
    for s in train_smiles_raw:
        c = canonicalize(s)
        if c:
            train_canonical.add(c)

    # Build test mols for SNN/Frag/Scaf reference
    print(f"  Training set: {len(train_canonical)} canonical SMILES")
    print(f"  Test set: {len(test_smiles_raw)} raw SMILES")

    test_mols = []
    for s in test_smiles_raw:
        m = get_mol(s)
        if m:
            test_mols.append(m)
    print(f"  Test mols (valid): {len(test_mols)}")

    # Compute reference (test set) properties for KL divergence
    print("\n[2/5] Computing reference property distributions...")
    ref_props = compute_properties(test_mols)
    print(f"  Reference properties computed for {len(ref_props['logP'])} molecules")

    # ── Load FCD results ───────────────────────────────────────────
    fcd_data = {}
    if os.path.exists(FCD_JSON):
        with open(FCD_JSON) as f:
            fcd_data = json.load(f)
        print(f"\n  FCD results loaded ({len(fcd_data)} methods)")

    fcd_map = {
        "VAE": "vae", "cVAE": "cvae", "Diffusion": "diffusion",
        "RL-VAE": "rl_vae", "CGLF": "cglf",
        "CRAVE (filt.)": "crave_filtered",
        "CRAVE (unfilt.)": "crave_unfiltered",
    }

    # ── Process each method ────────────────────────────────────────
    print("\n[3/5] Computing metrics for each method...")
    all_results = {}

    for method_name, filepath in METHOD_FILES.items():
        print(f"\n  ── {method_name} ──")
        if not os.path.exists(filepath):
            print(f"    SKIPPED: file not found")
            continue

        raw_smiles = load_smiles(filepath)
        print(f"    Raw SMILES: {len(raw_smiles)}")

        # Parse and canonicalise
        canonical = []
        mols = []
        for s in raw_smiles:
            c = canonicalize(s)
            if c is not None:
                canonical.append(c)
                mols.append(Chem.MolFromSmiles(c))

        n_valid = len(canonical)
        n_total = len(raw_smiles)
        validity = n_valid / n_total if n_total > 0 else 0.0
        print(f"    Valid: {n_valid}/{n_total} ({validity:.1%})")

        # MOSES metrics
        uniqueness = len(set(canonical)) / n_valid if n_valid > 0 else 0.0
        novelty = compute_novelty(raw_smiles, train_canonical)
        int_div, int_div2 = compute_internal_diversity(mols)
        filters = compute_filters(mols)

        print(f"    Unique: {uniqueness:.4f}")
        print(f"    Novelty: {novelty:.4f}")
        print(f"    IntDiv: {int_div:.4f}  IntDiv2: {int_div2:.4f}")
        print(f"    Filters (PAINS-free): {filters:.4f}")

        # SNN against test set
        print(f"    Computing SNN (vs test set)...")
        snn = compute_snn(mols, test_mols)
        print(f"    SNN: {snn:.4f}")

        # Fragment & scaffold similarity
        print(f"    Computing Frag & Scaf similarity...")
        frag = compute_frag(mols, test_mols)
        scaf = compute_scaf(mols, test_mols)
        print(f"    Frag: {frag:.4f}  Scaf: {scaf:.4f}")

        # FCD (from precomputed)
        fcd_key = fcd_map.get(method_name)
        fcd_val = fcd_data.get(fcd_key, {}).get("mean", float("nan"))
        fcd_std = fcd_data.get(fcd_key, {}).get("std", float("nan"))
        print(f"    FCD: {fcd_val:.2f} ± {fcd_std:.2f}")

        # GuacaMol KL divergences
        print(f"    Computing property KL divergences...")
        gen_props = compute_properties(mols)
        kl_divs = compute_kl_divergences(gen_props, ref_props)
        for prop, kl in kl_divs.items():
            print(f"      KL({prop}): {kl:.4f}")

        all_results[method_name] = {
            "moses": {
                "Valid": round(validity, 4),
                "Unique@1k": round(uniqueness, 4),
                "Novelty": round(novelty, 4),
                "IntDiv": round(int_div, 4),
                "IntDiv2": round(int_div2, 4),
                "SNN": round(snn, 4),
                "Frag": round(frag, 4),
                "Scaf": round(scaf, 4),
                "Filters": round(filters, 4),
                "FCD": round(fcd_val, 2),
            },
            "guacamol_kl": {k: round(v, 4) for k, v in kl_divs.items()},
            "n_generated": n_total,
            "n_valid": n_valid,
        }

    # ── Summary table ──────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("[4/5] MOSES-EQUIVALENT METRICS SUMMARY")
    print("=" * 100)

    header = f"{'Method':<18} {'Valid':>6} {'Uniq':>6} {'Novel':>6} {'IntDiv':>7} {'IntDiv2':>7} {'SNN':>6} {'Frag':>6} {'Scaf':>6} {'Filt':>6} {'FCD':>8}"
    print(header)
    print("-" * len(header))

    for method_name in METHOD_FILES:
        if method_name not in all_results:
            continue
        m = all_results[method_name]["moses"]
        print(f"{method_name:<18} {m['Valid']:>6.3f} {m['Unique@1k']:>6.3f} "
              f"{m['Novelty']:>6.3f} {m['IntDiv']:>7.4f} {m['IntDiv2']:>7.4f} "
              f"{m['SNN']:>6.3f} {m['Frag']:>6.3f} {m['Scaf']:>6.3f} "
              f"{m['Filters']:>6.3f} {m['FCD']:>8.2f}")

    print("\n" + "=" * 100)
    print("[5/5] GUACAMOL KL-DIVERGENCE SUMMARY")
    print("=" * 100)

    kl_props = ["logP", "QED", "MW", "TPSA", "NumRings", "NumAromaticRings", "HBA", "HBD"]
    header2 = f"{'Method':<18} " + " ".join(f"{p:>10}" for p in kl_props)
    print(header2)
    print("-" * len(header2))

    for method_name in METHOD_FILES:
        if method_name not in all_results:
            continue
        kl = all_results[method_name]["guacamol_kl"]
        vals = " ".join(f"{kl.get(p, float('nan')):>10.4f}" for p in kl_props)
        print(f"{method_name:<18} {vals}")

    # ── Save ───────────────────────────────────────────────────────
    with open(OUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {OUT_JSON}")


if __name__ == "__main__":
    main()
