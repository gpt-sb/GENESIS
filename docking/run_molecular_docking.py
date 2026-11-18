#!/usr/bin/env python3
"""
GENESIS — Phase 7: Molecular Docking Validation
================================================
Structure-based validation of top candidates from Phase 4 (generated molecules)
and Phase 6 (repurposed compounds) via AutoDock Vina rigid-body docking against
8 OSCC-relevant protein crystal structures.

Provides orthogonal validation to the ligand-based contrastive model scoring.

Dependencies:
    pip install vina meeko rdkit-pypi requests pandas numpy scipy

Input:
    - results/phase4_bioactivity_profiling/top100_candidates.tsv   (Phase 4)
    - repurposing/results/top100_candidates.tsv                     (Phase 6)
    - PDB structures: 3CQW, 4HJO, 4V04, 4HG7, 4DRH, 4JPS, 3LN1, 6NUQ

Note: the original paper run drew candidates from larger, internal per-method
dossiers (~1,000 molecules/method) that are not included in this repository;
the committed top100_candidates.tsv files (top 100 by recommendation_weight)
are a strict subset, so a from-scratch rerun selects from a smaller pool than
the paper's exact 364-candidate selection and will not be bit-identical to
docking/results/docking_results.tsv. Use that committed file plus
documents/verify_manuscript_numbers.py to verify the reported docking numbers
without rerunning this script.

Output:
    - GENESIS/docking/results/docking_results.tsv
    - GENESIS/docking/results/docking_summary.json
    - GENESIS/docking/results/multitarget_binders.tsv
    - GENESIS/docking/results/per_target_summary.tsv
    - GENESIS/docking/receptors/*.pdbqt
    - GENESIS/docking/ligands/*.pdbqt

Author: GENESIS Phase 7 pipeline
Date: 2026-03-23
"""

import os, sys, time, json, hashlib
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from collections import defaultdict
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent          # GENESIS/
DOCK_DIR    = ROOT / "docking"
REC_DIR     = DOCK_DIR / "receptors"
LIG_DIR     = DOCK_DIR / "ligands"
RESULTS_DIR = DOCK_DIR / "results"
POSES_DIR   = DOCK_DIR / "poses"

# Source PDB files (from previous docking run, or freshly downloaded)
PDB_CACHE   = Path(__file__).resolve().parent.parent.parent / "repurposing" / "docking" / "receptors"

# Input dossiers -- top100_candidates.tsv (not the paper's full internal
# dossier, see module docstring) are the largest candidate pools tracked in
# this repository.
PHASE4_DOSSIER = ROOT / "results" / "phase4_bioactivity_profiling" / "top100_candidates.tsv"
PHASE6_DOSSIER = ROOT / "repurposing" / "results" / "top100_candidates.tsv"

# Output files
DOCK_RESULTS    = RESULTS_DIR / "docking_results.tsv"
DOCK_SUMMARY    = RESULTS_DIR / "docking_summary.json"
MULTITARGET_OUT = RESULTS_DIR / "multitarget_binders.tsv"
PER_TARGET_OUT  = RESULTS_DIR / "per_target_summary.tsv"

# ── Docking Parameters ────────────────────────────────────────────────────────
TOP_N_GEN     = 200   # top compounds from Phase 4 (generated)
TOP_N_REP     = 200   # top compounds from Phase 6 (repurposed)
EXHAUSTIVENESS = 16
N_MODES        = 9
SEED           = 42

# ── Target Definitions ────────────────────────────────────────────────────────
# PDB structures with corrected grid centers from co-crystallized ligand centroids
TARGETS = {
    "AKT1": {
        "pdb_id": "3CQW",
        "chain": "A",
        "center": (6.0, 3.0, 17.3),
        "size": (24, 24, 24),
        "co_crystal": "CQW",
        "resolution": 2.00,
        "description": "AKT1 allosteric pocket",
    },
    "EGFR": {
        "pdb_id": "4HJO",
        "chain": "A",
        "center": (24.8, 9.2, 0.0),
        "size": (24, 24, 24),
        "co_crystal": "AQ4",
        "resolution": 2.75,
        "description": "EGFR kinase domain (erlotinib-bound)",
    },
    "FGFR1": {
        "pdb_id": "4V04",
        "chain": "A",
        "center": (87.1, 0.8, 18.1),
        "size": (24, 24, 24),
        "co_crystal": "0LI",
        "resolution": 2.10,
        "description": "FGFR1 kinase domain",
    },
    "MDM2": {
        "pdb_id": "4HG7",
        "chain": "A",
        "center": (-23.8, 7.5, -14.1),
        "size": (22, 22, 22),
        "co_crystal": "NUT",
        "resolution": 2.00,
        "description": "MDM2 p53-binding pocket (nutlin-3a bound)",
    },
    "MTOR": {
        "pdb_id": "4DRH",
        "chain": "A",
        "center": (-4.9, 20.0, -1.8),
        "size": (26, 26, 26),
        "co_crystal": "RAP",
        "resolution": 2.20,
        "description": "mTOR kinase domain (rapamycin-bound)",
    },
    "PIK3CA": {
        "pdb_id": "4JPS",
        "chain": "A",
        "center": (-1.3, -9.5, 16.9),
        "size": (24, 24, 24),
        "co_crystal": "1LT",
        "resolution": 2.90,
        "description": "PI3Kα catalytic site",
    },
    "PTGS2": {
        "pdb_id": "3LN1",
        "chain": "A",
        "center": (31.0, -22.3, -16.5),
        "size": (24, 24, 24),
        "co_crystal": "CEL",
        "resolution": 2.40,
        "description": "COX-2 active site (celecoxib-bound)",
    },
    "STAT3": {
        "pdb_id": "6NUQ",
        "chain": "A",
        "center": (13.1, 55.6, 0.1),
        "size": (24, 24, 24),
        "co_crystal": "KQV",
        "resolution": 2.10,
        "description": "STAT3 SH2 domain",
    },
}

# Map target names from dossier columns → our target keys
TARGET_NAME_MAP = {
    "Epidermal growth factor receptor": "EGFR",
    "Epidermal growth factor receptor and ErbB2 (HER1 and HER2)": "EGFR",
    "Epidermal growth factor receptor erbB1": "EGFR",
    "E3 ubiquitin-protein ligase Mdm2": "MDM2",
    "Tumour suppressor p53/oncoprotein Mdm2": "MDM2",
    "Protein Mdm4": "MDM2",
    "Fibroblast growth factor receptor 1": "FGFR1",
    "Fibroblast growth factor receptor 3": "FGFR1",  # dock against FGFR1 structure
    "Fibroblast growth factor receptor": "FGFR1",
    "Phosphatidylinositol 4,5-bisphosphate 3-kinase catalytic subunit alpha isoform": "PIK3CA",
    "PI3-kinase p110-alpha/p85-alpha": "PIK3CA",
    "PI3-kinase class I": "PIK3CA",
    "Serine/threonine-protein kinase mTOR": "MTOR",
    "mTORC1": "MTOR",
    "mTORC2": "MTOR",
    "RAC-alpha serine/threonine-protein kinase": "AKT1",
    "Prostaglandin G/H synthase 2": "PTGS2",
    "Cyclooxygenase": "PTGS2",
    "Signal transducer and activator of transcription 3": "STAT3",
    "CDK6/cyclin D1": "AKT1",          # no CDK structure; dock against AKT1 as kinase proxy
    "Cyclin-dependent kinase 4/cyclin D1": "AKT1",
    "Caspase-8": "MDM2",               # apoptosis pathway; dock against MDM2
    "GTPase HRas": "AKT1",             # RAS-AKT pathway
    "Myc proto-oncogene protein": "STAT3",  # transcription factor pathway
    "Neurogenic locus notch homolog protein 1": "STAT3",
}

# AutoDock4 atom type mapping for protein atoms → PDBQT format
AD4_ATOM_TYPES = {
    ("*", "N"):   "NA", ("*", "CA"):  "C",  ("*", "C"):   "C",
    ("*", "O"):   "OA", ("*", "OXT"): "OA", ("*", "CB"):  "C",
    ("*", "H"):   "HD", ("*", "HA"):  "H",  ("*", "HN"):  "HD",
    ("*", "CG"):  "C",  ("*", "CG1"): "C",  ("*", "CG2"): "C",
    ("*", "CD"):  "C",  ("*", "CD1"): "C",  ("*", "CD2"): "C",
    ("*", "CE"):  "C",  ("*", "CE1"): "C",  ("*", "CE2"): "C",
    ("*", "CE3"): "C",  ("*", "CZ"):  "C",  ("*", "CZ2"): "C",
    ("*", "CZ3"): "C",  ("*", "CH2"): "C",
    # Aromatic carbons
    ("PHE", "CG"):  "A", ("PHE", "CD1"): "A", ("PHE", "CD2"): "A",
    ("PHE", "CE1"): "A", ("PHE", "CE2"): "A", ("PHE", "CZ"):  "A",
    ("TYR", "CG"):  "A", ("TYR", "CD1"): "A", ("TYR", "CD2"): "A",
    ("TYR", "CE1"): "A", ("TYR", "CE2"): "A", ("TYR", "CZ"):  "A",
    ("TRP", "CG"):  "A", ("TRP", "CD1"): "A", ("TRP", "CD2"): "A",
    ("TRP", "CE2"): "A", ("TRP", "CE3"): "A", ("TRP", "CZ2"): "A",
    ("TRP", "CZ3"): "A", ("TRP", "CH2"): "A",
    ("HIS", "CG"):  "A", ("HIS", "CD2"): "A", ("HIS", "CE1"): "A",
    # Nitrogen donors/acceptors
    ("LYS", "NZ"):  "N",  ("ARG", "NE"):  "NA", ("ARG", "NH1"): "N",
    ("ARG", "NH2"): "N",  ("HIS", "ND1"): "NA", ("HIS", "NE2"): "NA",
    ("TRP", "NE1"): "NA", ("ASN", "ND2"): "N",  ("GLN", "NE2"): "N",
    # Oxygen acceptors
    ("SER", "OG"):  "OA", ("THR", "OG1"): "OA", ("TYR", "OH"):  "OA",
    ("ASP", "OD1"): "OA", ("ASP", "OD2"): "OA",
    ("GLU", "OE1"): "OA", ("GLU", "OE2"): "OA",
    ("ASN", "OD1"): "OA", ("GLN", "OE1"): "OA",
    # Sulfur
    ("CYS", "SG"):  "SA", ("MET", "SD"):  "S",
}


def get_ad4_type(resname, atomname, element):
    """Get AutoDock4 atom type for a protein atom."""
    resname = resname.strip()
    atomname = atomname.strip()
    element = element.strip().upper() if element else ""
    key = (resname, atomname)
    if key in AD4_ATOM_TYPES:
        return AD4_ATOM_TYPES[key]
    key = ("*", atomname)
    if key in AD4_ATOM_TYPES:
        return AD4_ATOM_TYPES[key]
    elem_map = {
        "C": "C", "N": "NA", "O": "OA", "S": "SA", "H": "H",
        "F": "F", "CL": "Cl", "BR": "Br", "I": "I", "P": "P",
        "FE": "Fe", "ZN": "Zn", "MG": "Mg", "MN": "Mn",
        "CA": "Ca", "CU": "Cu", "SE": "Se",
    }
    return elem_map.get(element, "C")


# ══════════════════════════════════════════════════════════════════════════════
# Phase A: Receptor Preparation
# ══════════════════════════════════════════════════════════════════════════════

def download_pdb(pdb_id: str, outpath: Path) -> bool:
    """Download PDB file from RCSB."""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    resp = requests.get(url, timeout=30)
    if resp.status_code == 200:
        outpath.write_text(resp.text)
        return True
    return False


def pdb_to_pdbqt_receptor(pdb_file: Path, pdbqt_file: Path, chain: str = "A") -> int:
    """Convert PDB → PDBQT for rigid receptor. Returns atom count."""
    lines_out = []
    with open(pdb_file) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            resname = line[17:20].strip()
            if resname in ("HOH", "WAT", "DOD"):
                continue
            rec_chain = line[21]
            if chain and rec_chain != chain:
                continue

            atomname = line[12:16].strip()
            element = line[76:78].strip() if len(line) > 76 else ""
            if not element:
                element = atomname[0] if atomname else "C"
            ad4_type = get_ad4_type(resname, atomname, element)

            pdb_66 = line[:66].rstrip().ljust(66)
            charge_str = f"{0.000:10.3f}"
            type_str = f" {ad4_type:<2s}"
            lines_out.append(f"{pdb_66}{charge_str}{type_str}\n")

    pdbqt_file.write_text("".join(lines_out))
    return len(lines_out)


def prepare_all_receptors():
    """Download PDB files and convert to PDBQT."""
    print("\n" + "=" * 70)
    print("PHASE A: Receptor Preparation")
    print("=" * 70)

    REC_DIR.mkdir(parents=True, exist_ok=True)
    receptor_map = {}

    for target, info in TARGETS.items():
        pdb_id = info["pdb_id"]
        chain = info["chain"]

        pdb_path = REC_DIR / f"{pdb_id}.pdb"
        pdbqt_path = REC_DIR / f"{pdb_id}_receptor.pdbqt"

        # Get PDB file
        if not pdb_path.exists():
            # Try cache from previous docking run
            cached = PDB_CACHE / f"{pdb_id}.pdb"
            if cached.exists():
                import shutil
                shutil.copy2(cached, pdb_path)
                print(f"  [{target}] Copied {pdb_id}.pdb from cache")
            else:
                print(f"  [{target}] Downloading {pdb_id} from RCSB...")
                if not download_pdb(pdb_id, pdb_path):
                    print(f"    ERROR: Failed to download {pdb_id}")
                    continue

        # Convert to PDBQT
        n_atoms = pdb_to_pdbqt_receptor(pdb_path, pdbqt_path, chain)
        receptor_map[target] = str(pdbqt_path)
        print(f"  [{target}] {pdb_id} chain {chain} → {n_atoms} atoms → {pdbqt_path.name}")

    return receptor_map


# ══════════════════════════════════════════════════════════════════════════════
# Phase B: Candidate Selection & Ligand Preparation
# ══════════════════════════════════════════════════════════════════════════════

def load_candidates():
    """Load top candidates from Phase 4 and Phase 6."""
    print("\n" + "=" * 70)
    print("PHASE B: Candidate Selection")
    print("=" * 70)

    for dossier in (PHASE4_DOSSIER, PHASE6_DOSSIER):
        if not dossier.exists():
            raise FileNotFoundError(
                f"Missing candidate dossier: {dossier}\n"
                "Run Phase 4 (generative/evaluate_gen.py) and Phase 6 "
                "(repurposing/run_drug_repurposing.py) first, or use the "
                "committed docking/results/docking_results.tsv directly -- "
                "see README.md 'Reproducing the Paper Results'."
            )

    candidates = []

    # Phase 4: generated molecules
    gen = pd.read_csv(PHASE4_DOSSIER, sep='\t')
    top_gen = gen.nlargest(TOP_N_GEN, 'recommendation_weight')
    for i, row in top_gen.iterrows():
        best_target = row.get('best_target', '')
        dock_target = TARGET_NAME_MAP.get(best_target)
        if not dock_target:
            continue
        cpd_id = f"GEN_{i:05d}"
        candidates.append({
            'cpd_id': cpd_id,
            'smiles': row['smiles'],
            'source': 'generated',
            'method': row.get('method', ''),
            'composite_score': row.get('composite_score', 0),
            'vp_similarity': row.get('vp_similarity', 0),
            'recommendation_weight': row.get('recommendation_weight', 0),
            'overall_flag': row.get('overall_flag', ''),
            'best_target_name': best_target,
            'dock_target': dock_target,
        })
    print(f"  Phase 4 (generated): {len([c for c in candidates if c['source']=='generated'])} candidates")

    # Phase 6: repurposed compounds
    rep = pd.read_csv(PHASE6_DOSSIER, sep='\t', low_memory=False)
    top_rep = rep.nlargest(TOP_N_REP, 'recommendation_weight')
    for i, row in top_rep.iterrows():
        best_target = row.get('best_target', '')
        dock_target = TARGET_NAME_MAP.get(best_target)
        if not dock_target:
            continue
        cid = row.get('compound_chembl_id', f'REP_{i:05d}')
        cpd_id = f"REP_{cid}" if isinstance(cid, str) else f"REP_{i:05d}"
        candidates.append({
            'cpd_id': cpd_id,
            'smiles': row['smiles'],
            'source': 'repurposed',
            'method': '',
            'composite_score': row.get('composite_score', 0),
            'vp_similarity': row.get('vp_similarity', 0),
            'recommendation_weight': row.get('recommendation_weight', 0),
            'overall_flag': row.get('overall_flag', ''),
            'compound_name': row.get('compound_name', ''),
            'best_target_name': best_target,
            'dock_target': dock_target,
        })
    n_rep = len([c for c in candidates if c['source'] == 'repurposed'])
    print(f"  Phase 6 (repurposed): {n_rep} candidates")
    print(f"  Total docking candidates: {len(candidates)}")

    # Count per target
    target_counts = defaultdict(int)
    for c in candidates:
        target_counts[c['dock_target']] += 1
    print("  Per-target job counts:")
    for t in sorted(target_counts):
        print(f"    {t:8s}: {target_counts[t]}")

    return candidates


def prepare_ligand_meeko(smiles: str, pdbqt_path: Path) -> tuple:
    """Convert SMILES → 3D conformer → PDBQT via Meeko."""
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "Invalid SMILES"

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = SEED
    result = AllChem.EmbedMolecule(mol, params)
    if result != 0:
        params2 = AllChem.ETKDGv3()
        params2.useRandomCoords = True
        params2.randomSeed = SEED
        result = AllChem.EmbedMolecule(mol, params2)
        if result != 0:
            return False, "3D embedding failed"

    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass

    try:
        prep = MoleculePreparation()
        mol_setups = prep.prepare(mol)
        for setup in mol_setups:
            pdbqt_str, is_ok, err_msg = PDBQTWriterLegacy.write_string(setup)
            if is_ok:
                pdbqt_path.write_text(pdbqt_str)
                return True, "OK"
            else:
                return False, f"Meeko error: {err_msg}"
        return False, "No setups from Meeko"
    except Exception as e:
        return False, str(e)


def prepare_all_ligands(candidates):
    """Prepare PDBQT files for all candidate ligands."""
    print("\n" + "=" * 70)
    print("PHASE C: Ligand Preparation")
    print("=" * 70)

    LIG_DIR.mkdir(parents=True, exist_ok=True)

    # Deduplicate by SMILES (same compound may appear in both phases)
    unique = {}
    for c in candidates:
        smi = c['smiles']
        if smi not in unique:
            unique[smi] = c['cpd_id']

    n_ok, n_fail = 0, 0
    ligand_map = {}  # cpd_id → pdbqt_path
    smiles_to_pdbqt = {}  # smiles → pdbqt_path (for dedup)

    total = len(unique)
    report_every = max(1, total // 10)

    for idx, (smi, cpd_id) in enumerate(unique.items()):
        if idx % report_every == 0:
            print(f"  [{idx}/{total}] preparing ligands...")

        pdbqt_path = LIG_DIR / f"{cpd_id}.pdbqt"

        if pdbqt_path.exists() and pdbqt_path.stat().st_size > 50:
            smiles_to_pdbqt[smi] = pdbqt_path
            n_ok += 1
            continue

        ok, msg = prepare_ligand_meeko(smi, pdbqt_path)
        if ok:
            smiles_to_pdbqt[smi] = pdbqt_path
            n_ok += 1
        else:
            n_fail += 1
            if n_fail <= 5:
                print(f"    FAIL {cpd_id}: {msg}")

    # Map all candidates to their pdbqt paths
    for c in candidates:
        pdbqt = smiles_to_pdbqt.get(c['smiles'])
        if pdbqt:
            ligand_map[c['cpd_id']] = str(pdbqt)

    print(f"  Ligand preparation: {n_ok} OK, {n_fail} failed, {len(ligand_map)} mapped")
    return ligand_map


# ══════════════════════════════════════════════════════════════════════════════
# Phase D: Docking Execution
# ══════════════════════════════════════════════════════════════════════════════

def run_docking(candidates, receptor_map, ligand_map):
    """Run AutoDock Vina docking for all compound-target pairs."""
    from vina import Vina

    print("\n" + "=" * 70)
    print("PHASE D: Docking Execution")
    print("=" * 70)

    POSES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build job list
    jobs = []
    for c in candidates:
        target = c['dock_target']
        cpd_id = c['cpd_id']
        if target not in receptor_map:
            continue
        if cpd_id not in ligand_map:
            continue
        jobs.append({
            'cpd_id': cpd_id,
            'smiles': c['smiles'],
            'source': c['source'],
            'target': target,
            'receptor_pdbqt': receptor_map[target],
            'ligand_pdbqt': ligand_map[cpd_id],
            'center': TARGETS[target]['center'],
            'size': TARGETS[target]['size'],
            'composite_score': c['composite_score'],
            'vp_similarity': c['vp_similarity'],
            'recommendation_weight': c['recommendation_weight'],
            'overall_flag': c.get('overall_flag', ''),
            'compound_name': c.get('compound_name', ''),
            'method': c.get('method', ''),
        })

    print(f"  Total docking jobs: {len(jobs)}")

    all_results = []
    n_success, n_fail = 0, 0
    t_start = time.time()
    report_every = max(1, len(jobs) // 20)

    for ji, job in enumerate(jobs):
        if ji % report_every == 0:
            elapsed = time.time() - t_start
            rate = max(ji, 1) / elapsed if elapsed > 0 else 1
            remaining = (len(jobs) - ji) / rate if rate > 0 else 0
            print(f"  [{ji+1}/{len(jobs)}] ({ji*100//max(1,len(jobs))}%) "
                  f"[{elapsed:.0f}s elapsed, ~{remaining:.0f}s left] "
                  f"{job['cpd_id']} vs {job['target']}")

        try:
            v = Vina(sf_name="vina", verbosity=0)
            v.set_receptor(job['receptor_pdbqt'])
            v.set_ligand_from_file(job['ligand_pdbqt'])
            v.compute_vina_maps(center=list(job['center']),
                                box_size=list(job['size']))
            v.dock(exhaustiveness=EXHAUSTIVENESS, n_poses=N_MODES)
            energies = v.energies()

            best_energy = round(float(energies[0][0]), 2)
            all_energies = "|".join(f"{float(row[0]):.1f}" for row in energies)
            n_poses = len(energies)

            # Save best pose
            pose_file = POSES_DIR / f"{job['cpd_id']}_vs_{job['target']}.pdbqt"
            v.write_poses(str(pose_file), n_poses=1, overwrite=True)

            n_success += 1

        except Exception as e:
            best_energy = None
            all_energies = ""
            n_poses = 0
            n_fail += 1
            if n_fail <= 5:
                print(f"    DOCK FAIL: {job['cpd_id']} vs {job['target']}: {e}")

        all_results.append({
            'cpd_id': job['cpd_id'],
            'smiles': job['smiles'],
            'source': job['source'],
            'method': job['method'],
            'compound_name': job.get('compound_name', ''),
            'target': job['target'],
            'pdb_id': TARGETS[job['target']]['pdb_id'],
            'best_energy_kcal': best_energy,
            'all_energies': all_energies,
            'n_poses': n_poses,
            'composite_score': job['composite_score'],
            'vp_similarity': job['vp_similarity'],
            'recommendation_weight': job['recommendation_weight'],
            'overall_flag': job['overall_flag'],
        })

    elapsed = time.time() - t_start
    print(f"\n  Docking complete: {n_success} OK, {n_fail} failed in {elapsed:.1f}s")

    return pd.DataFrame(all_results)


# ══════════════════════════════════════════════════════════════════════════════
# Phase E: Results Analysis
# ══════════════════════════════════════════════════════════════════════════════

def analyze_results(df_dock):
    """Comprehensive analysis of docking results."""
    from scipy.stats import spearmanr, pearsonr

    print("\n" + "=" * 70)
    print("PHASE E: Results Analysis")
    print("=" * 70)

    valid = df_dock.dropna(subset=['best_energy_kcal'])
    print(f"\n  Valid docking results: {len(valid)}/{len(df_dock)}")

    if len(valid) == 0:
        print("  No valid docking results. Aborting analysis.")
        return {}

    # ── 1. Overall energy distribution ──
    energies = valid['best_energy_kcal']
    print(f"\n  === Binding Energy Distribution ===")
    print(f"  Mean:   {energies.mean():.1f} kcal/mol")
    print(f"  Median: {energies.median():.1f}")
    print(f"  Best:   {energies.min():.1f}")
    print(f"  Worst:  {energies.max():.1f}")
    print(f"  SD:     {energies.std():.1f}")

    thresholds = {}
    for thresh in [-6, -7, -8, -9, -10]:
        n = (energies <= thresh).sum()
        pct = n * 100 / len(energies) if len(energies) > 0 else 0
        thresholds[f"le_{abs(thresh)}"] = int(n)
        print(f"  ≤ {thresh}: {n} ({pct:.1f}%)")

    # ── 2. Per-target summary ──
    print(f"\n  === Per-Target Summary ===")
    target_stats = []
    for target in sorted(valid['target'].unique()):
        tgt_df = valid[valid['target'] == target]
        tgt_e = tgt_df['best_energy_kcal']
        n_le7 = (tgt_e <= -7).sum()
        n_le8 = (tgt_e <= -8).sum()
        n_le9 = (tgt_e <= -9).sum()
        stats = {
            'target': target,
            'pdb_id': TARGETS[target]['pdb_id'],
            'n_docked': len(tgt_e),
            'mean_energy': round(tgt_e.mean(), 1),
            'median_energy': round(tgt_e.median(), 1),
            'best_energy': round(tgt_e.min(), 1),
            'sd': round(tgt_e.std(), 1),
            'n_le_minus7': int(n_le7),
            'n_le_minus8': int(n_le8),
            'n_le_minus9': int(n_le9),
        }
        target_stats.append(stats)
        print(f"  {target:8s} ({TARGETS[target]['pdb_id']}): "
              f"n={len(tgt_e):3d}, mean={tgt_e.mean():.1f}, "
              f"best={tgt_e.min():.1f}, ≤-7: {n_le7}, ≤-8: {n_le8}, ≤-9: {n_le9}")

    pd.DataFrame(target_stats).to_csv(PER_TARGET_OUT, sep='\t', index=False)

    # ── 3. Source comparison (generated vs repurposed) ──
    print(f"\n  === Source Comparison ===")
    for src in ['generated', 'repurposed']:
        sub = valid[valid['source'] == src]
        if len(sub) > 0:
            se = sub['best_energy_kcal']
            print(f"  {src:12s}: n={len(sub):3d}, mean={se.mean():.1f}, "
                  f"median={se.median():.1f}, best={se.min():.1f}, "
                  f"≤-7: {(se<=-7).sum()}, ≤-8: {(se<=-8).sum()}")

    # ── 4. Top 20 docking hits ──
    print(f"\n  === Top 20 Docking Hits ===")
    top20 = valid.nsmallest(20, 'best_energy_kcal')
    for _, r in top20.iterrows():
        name = r.get('compound_name', '') or r.get('method', '')
        print(f"  {r['cpd_id']:20s} vs {r['target']:8s}: "
              f"{r['best_energy_kcal']:6.1f} kcal/mol  "
              f"[{r['source']}, {name}]")

    # ── 5. Multi-target binders ──
    print(f"\n  === Multi-Target Binders (≤-7.0 kcal/mol) ===")
    strong = valid[valid['best_energy_kcal'] <= -7.0]
    cpd_targets = strong.groupby('cpd_id').agg(
        n_targets=('target', 'nunique'),
        targets=('target', lambda x: ', '.join(sorted(x.unique()))),
        best_energy=('best_energy_kcal', 'min'),
        source=('source', 'first'),
        smiles=('smiles', 'first'),
    ).reset_index()
    multi = cpd_targets[cpd_targets['n_targets'] >= 2].sort_values('best_energy')

    if len(multi) > 0:
        print(f"  {len(multi)} compounds bind ≤-7.0 to ≥2 targets:")
        for _, r in multi.head(15).iterrows():
            print(f"    {r['cpd_id']:20s}: {r['n_targets']} targets "
                  f"({r['targets']}), best={r['best_energy']:.1f}")
        multi.to_csv(MULTITARGET_OUT, sep='\t', index=False)
    else:
        print("  No multi-target binders found at ≤-7.0 threshold")
        # Try at -6.0
        strong6 = valid[valid['best_energy_kcal'] <= -6.0]
        cpd_targets6 = strong6.groupby('cpd_id').agg(
            n_targets=('target', 'nunique'),
            targets=('target', lambda x: ', '.join(sorted(x.unique()))),
            best_energy=('best_energy_kcal', 'min'),
        ).reset_index()
        multi6 = cpd_targets6[cpd_targets6['n_targets'] >= 2]
        print(f"  At ≤-6.0 threshold: {len(multi6)} multi-target binders")

    # ── 6. Docking–composite score correlation ──
    print(f"\n  === Docking vs Composite Score Correlation ===")
    corr_df = valid[['composite_score', 'best_energy_kcal']].dropna()
    if len(corr_df) >= 10:
        rho, p_spear = spearmanr(corr_df['composite_score'], corr_df['best_energy_kcal'])
        r_pear, p_pear = pearsonr(corr_df['composite_score'], corr_df['best_energy_kcal'])
        print(f"  Spearman ρ = {rho:.3f} (p = {p_spear:.4f})")
        print(f"  Pearson r  = {r_pear:.3f} (p = {p_pear:.4f})")
    else:
        rho, p_spear, r_pear, p_pear = 0, 1, 0, 1
        print("  Too few data points for correlation")

    # ── 7. Scaffold analysis of strong binders ──
    print(f"\n  === Scaffold Analysis (≤-8.0 kcal/mol) ===")
    strong8 = valid[valid['best_energy_kcal'] <= -8.0]
    if len(strong8) > 0:
        from rdkit.Chem.Scaffolds import MurckoScaffold
        scaffolds = defaultdict(int)
        for smi in strong8['smiles'].unique():
            mol = Chem.MolFromSmiles(smi)
            if mol:
                try:
                    core = MurckoScaffold.GetScaffoldForMol(mol)
                    scaffolds[Chem.MolToSmiles(core)] += 1
                except Exception:
                    pass
        print(f"  {len(strong8)} hits with ≤-8.0 across {len(strong8['smiles'].unique())} unique compounds")
        print(f"  {len(scaffolds)} unique Murcko scaffolds")
        for smi, cnt in sorted(scaffolds.items(), key=lambda x: -x[1])[:10]:
            print(f"    {cnt:3d}x  {smi[:60]}")
    else:
        scaffolds = {}
        print("  No compounds at ≤-8.0 threshold")

    # Build summary
    summary = {
        'total_jobs': len(df_dock),
        'successful_docks': len(valid),
        'failed_docks': len(df_dock) - len(valid),
        'energy_stats': {
            'mean': round(energies.mean(), 2),
            'median': round(energies.median(), 2),
            'best': round(energies.min(), 2),
            'worst': round(energies.max(), 2),
            'sd': round(energies.std(), 2),
        },
        'thresholds': thresholds,
        'n_targets': len(valid['target'].unique()),
        'targets': sorted(valid['target'].unique()),
        'n_multitarget_binders': len(multi) if len(multi) > 0 else 0,
        'correlation': {
            'spearman_rho': round(rho, 4),
            'spearman_p': round(p_spear, 6),
            'pearson_r': round(r_pear, 4),
            'pearson_p': round(p_pear, 6),
        },
        'source_breakdown': {
            src: {
                'n': int(len(valid[valid['source']==src])),
                'mean_energy': round(valid[valid['source']==src]['best_energy_kcal'].mean(), 2)
                    if len(valid[valid['source']==src]) > 0 else None,
            }
            for src in ['generated', 'repurposed']
        },
        'n_scaffolds_le8': len(scaffolds),
        'per_target': target_stats,
    }

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# Phase F: Update Dossiers with Docking Scores
# ══════════════════════════════════════════════════════════════════════════════

def update_dossiers(df_dock):
    """Create a merged view with docking scores for downstream integration."""
    print("\n" + "=" * 70)
    print("PHASE F: Dossier Integration")
    print("=" * 70)

    # Per-compound best docking score (across all targets)
    valid = df_dock.dropna(subset=['best_energy_kcal'])
    if len(valid) == 0:
        print("  No valid results to integrate")
        return

    best_per_cpd = valid.groupby('cpd_id').agg(
        best_dock_energy=('best_energy_kcal', 'min'),
        best_dock_target=('target', lambda x: valid.loc[valid.loc[x.index, 'best_energy_kcal'].idxmin(), 'target']),
        n_targets_docked=('target', 'nunique'),
        mean_dock_energy=('best_energy_kcal', 'mean'),
    ).reset_index()

    out_path = RESULTS_DIR / "compound_docking_scores.tsv"
    best_per_cpd.to_csv(out_path, sep='\t', index=False)
    print(f"  Saved per-compound docking scores: {out_path.name} ({len(best_per_cpd)} compounds)")

    # Summary stats
    bde = best_per_cpd['best_dock_energy']
    print(f"  Per-compound best docking energy:")
    print(f"    Mean: {bde.mean():.1f}, Median: {bde.median():.1f}, Best: {bde.min():.1f}")


# ══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70)
    print("GENESIS — Phase 7: Molecular Docking Validation")
    print("=" * 70)
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Exhaustiveness: {EXHAUSTIVENESS}, Poses: {N_MODES}")
    print(f"Top-N generated: {TOP_N_GEN}, Top-N repurposed: {TOP_N_REP}")

    # Phase A: Prepare receptors
    receptor_map = prepare_all_receptors()

    # Phase B: Select and load candidates
    candidates = load_candidates()

    # Phase C: Prepare ligands
    ligand_map = prepare_all_ligands(candidates)

    # Phase D: Run docking
    df_dock = run_docking(candidates, receptor_map, ligand_map)

    # Save raw results
    df_dock.to_csv(DOCK_RESULTS, sep='\t', index=False)
    print(f"\n  Saved raw results: {DOCK_RESULTS}")

    # Phase E: Analyze
    summary = analyze_results(df_dock)

    # Phase F: Update dossiers
    update_dossiers(df_dock)

    # Save summary
    summary['runtime_seconds'] = round(time.time() - t0, 1)
    summary['parameters'] = {
        'exhaustiveness': EXHAUSTIVENESS,
        'n_modes': N_MODES,
        'top_n_gen': TOP_N_GEN,
        'top_n_rep': TOP_N_REP,
        'seed': SEED,
    }
    with open(DOCK_SUMMARY, 'w') as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Phase 7 complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Results: {RESULTS_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
