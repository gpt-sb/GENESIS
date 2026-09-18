#!/usr/bin/env python3
"""
Phase 1.4 — Scaffold and target-family assignment.

Computes Murcko scaffolds (RDKit) and assigns compounds to target-family
groups for leakage-safe splitting. These two grouping artifacts are consumed
by create_splits.py to produce scaffold-disjoint, pathway-aware train/val/test
splits — the gold standard for molecular ML.

Pipeline position:
  prepare_positives.py → positives.tsv ┐
  prepare_negatives.py → negatives.tsv ┤
                                         ▼
                          compute_scaffolds.py  ← THIS FILE
                                         │
                             scaffold_groups.pkl
                             target_families.pkl
                                         │
                                         ▼
                           create_splits.py → splits.json

Output:
  GENESIS/modeling/scaffold_groups.pkl  — {compound_to_scaffold, scaffold_to_compounds}
  GENESIS/modeling/target_families.pkl  — {compound_chembl_id → pathway_family}
"""
from pathlib import Path
import pandas as pd
import pickle
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent        # GENESIS/modeling/
GENESIS_DIR = SCRIPT_DIR.parent                     # GENESIS/

POS_FILE = SCRIPT_DIR / 'positives.tsv'
NEG_FILE = SCRIPT_DIR / 'negatives.tsv'
OUTPUT_SCAFFOLDS = SCRIPT_DIR / 'scaffold_groups.pkl'
OUTPUT_FAMILIES = SCRIPT_DIR / 'target_families.pkl'

# Target-family mapping: gene symbols → one of 16 OSCC-relevant pathway families.
# This curated lookup table groups ~70 gene targets into pathway-level families
# so that create_splits.py can enforce pathway-aware splitting. Without this,
# a model could memorize "EGFR inhibitors are active" from training-set EGFR
# compounds and appear to generalise on val-set EGFR compounds (pathway leakage).
FAMILY_MAP = {
    # EGFR/ERBB family
    'EGFR': 'EGFR_ERBB', 'ERBB2': 'EGFR_ERBB', 'ERBB3': 'EGFR_ERBB', 'ERBB4': 'EGFR_ERBB',
    # PI3K/AKT/MTOR
    'PIK3CA': 'PI3K_AKT_MTOR', 'PIK3CB': 'PI3K_AKT_MTOR', 'PIK3CG': 'PI3K_AKT_MTOR',
    'PIK3CD': 'PI3K_AKT_MTOR', 'AKT1': 'PI3K_AKT_MTOR', 'AKT2': 'PI3K_AKT_MTOR',
    'MTOR': 'PI3K_AKT_MTOR',
    # RAS/RAF/MEK/ERK
    'KRAS': 'RAS_MAPK', 'NRAS': 'RAS_MAPK', 'HRAS': 'RAS_MAPK',
    'BRAF': 'RAS_MAPK', 'RAF1': 'RAS_MAPK',
    'MAP2K1': 'RAS_MAPK', 'MAP2K2': 'RAS_MAPK',
    'MAPK1': 'RAS_MAPK', 'MAPK3': 'RAS_MAPK',
    # Apoptosis
    'CASP3': 'APOPTOSIS', 'CASP8': 'APOPTOSIS', 'CASP9': 'APOPTOSIS',
    'BCL2': 'APOPTOSIS', 'BCL2L1': 'APOPTOSIS', 'BAX': 'APOPTOSIS',
    'MCL1': 'APOPTOSIS',
    # CDK/cell cycle
    'CDK1': 'CELL_CYCLE', 'CDK2': 'CELL_CYCLE', 'CDK4': 'CELL_CYCLE', 'CDK6': 'CELL_CYCLE',
    'CCND1': 'CELL_CYCLE', 'CCNB1': 'CELL_CYCLE', 'CCNA2': 'CELL_CYCLE',
    'CDKN1B': 'CELL_CYCLE', 'CDKN2A': 'CELL_CYCLE',
    # TP53 pathway
    'TP53': 'TP53', 'MDM2': 'TP53', 'MDM4': 'TP53',
    # NOTCH
    'NOTCH1': 'NOTCH', 'NOTCH2': 'NOTCH', 'NOTCH3': 'NOTCH',
    # VEGF/angiogenesis
    'VEGFA': 'VEGF_ANGIO', 'KDR': 'VEGF_ANGIO', 'FLT1': 'VEGF_ANGIO', 'FLT4': 'VEGF_ANGIO',
    # HDAC/epigenetic
    'HDAC1': 'EPIGENETIC', 'HDAC2': 'EPIGENETIC', 'HDAC3': 'EPIGENETIC',
    'HDAC6': 'EPIGENETIC', 'DNMT1': 'EPIGENETIC', 'EZH2': 'EPIGENETIC',
    # JAK/STAT
    'JAK1': 'JAK_STAT', 'JAK2': 'JAK_STAT', 'STAT3': 'JAK_STAT',
    # PARP/DNA repair
    'PARP1': 'DNA_REPAIR', 'PARP2': 'DNA_REPAIR', 'ATR': 'DNA_REPAIR', 'ATM': 'DNA_REPAIR',
    # Proteasome
    'PSMB5': 'PROTEASOME', 'PSMB1': 'PROTEASOME',
    # Tubulin/cytoskeleton
    'TUBB': 'TUBULIN', 'TUBA1A': 'TUBULIN',
    # NFkB
    'NFKB1': 'NFKB', 'RELA': 'NFKB', 'IKBKB': 'NFKB',
    # COX
    'PTGS2': 'COX', 'PTGS1': 'COX',
    # SRC family
    'SRC': 'SRC_FAMILY', 'ABL1': 'SRC_FAMILY', 'FYN': 'SRC_FAMILY',
    # MMP
    'MMP2': 'MMP', 'MMP9': 'MMP', 'MMP14': 'MMP',
}


def get_murcko_scaffold(smiles):
    """Compute Murcko scaffold for a SMILES string.

    The Murcko scaffold is the core ring system with linkers, stripping all
    side chains. Compounds sharing the same scaffold are structurally related;
    splitting them across train/val would allow the model to memorize
    scaffold-level patterns instead of learning genuine potency signals.

    Edge cases return sentinel strings (not None) so no compound is dropped
    from the pipeline — sentinels form their own scaffold group in the split.
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    if not smiles or pd.isna(smiles):
        return 'NO_SMILES'
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 'INVALID_SMILES'
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(core)
    except Exception:
        return 'SCAFFOLD_ERROR'


def assign_target_family(gene_symbols_str):
    """Map a compound's gene symbols string to its pathway family.

    Handles semicolon- and pipe-delimited gene lists (e.g. 'EGFR;ERBB2').
    Multi-target compounds get a joined label (e.g. 'APOPTOSIS|RAS_MAPK').
    Returns 'OTHER' for known genes not in FAMILY_MAP, 'UNKNOWN' for missing.
    """
    if not gene_symbols_str or pd.isna(gene_symbols_str):
        return 'UNKNOWN'
    genes = [g.strip() for g in str(gene_symbols_str).replace('||', ';').split(';')]
    families = set()
    for g in genes:
        g_upper = g.upper().strip()
        if g_upper in FAMILY_MAP:
            families.add(FAMILY_MAP[g_upper])
    if not families:
        return 'OTHER'
    if len(families) == 1:
        return families.pop()
    return '|'.join(sorted(families))


def main():
    # Load positive (OSCC-active) and negative (tiered) compounds
    pos = pd.read_csv(POS_FILE, sep='\t', low_memory=False)
    neg = pd.read_csv(NEG_FILE, sep='\t', low_memory=False)
    log.info('Positives: %d, Negatives: %d', len(pos), len(neg))

    # Deduplicate: a compound may appear in both files or multiple rows;
    # we need exactly one scaffold assignment per compound.
    all_compounds = pd.concat([
        pos[['compound_chembl_id', 'smiles', 'gene_symbols']],
        neg[['compound_chembl_id', 'smiles', 'gene_symbols']],
    ], ignore_index=True).drop_duplicates(subset='compound_chembl_id')
    log.info('Unique compounds for scaffold computation: %d', len(all_compounds))

    # Compute scaffolds (batch, with progress)
    log.info('Computing Murcko scaffolds...')
    scaffolds = {}
    for i, (_, row) in enumerate(all_compounds.iterrows()):
        scaffolds[row['compound_chembl_id']] = get_murcko_scaffold(row['smiles'])
        if (i + 1) % 50000 == 0:
            log.info('  Scaffolds computed: %d / %d', i + 1, len(all_compounds))

    log.info('Scaffolds computed: %d', len(scaffolds))

    # Group by scaffold: compounds sharing a scaffold MUST stay in the same
    # split to prevent structural leakage. create_splits.py treats each
    # scaffold group as an atomic unit during partitioning.
    scaffold_groups = defaultdict(list)
    for cid, scaffold in scaffolds.items():
        scaffold_groups[scaffold].append(cid)

    log.info('Unique scaffolds: %d', len(scaffold_groups))
    sizes = [len(v) for v in scaffold_groups.values()]
    log.info('Scaffold group sizes: mean=%.1f, median=%.0f, max=%d',
             sum(sizes)/len(sizes), sorted(sizes)[len(sizes)//2], max(sizes))

    # Save both directions: compound→scaffold (for per-compound lookup) and
    # scaffold→compounds (for group-level splitting).
    with open(OUTPUT_SCAFFOLDS, 'wb') as f:
        pickle.dump({
            'compound_to_scaffold': scaffolds,
            'scaffold_to_compounds': dict(scaffold_groups),
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info('Saved → %s', OUTPUT_SCAFFOLDS)

    # Assign target families — used alongside scaffolds by create_splits.py
    # to ensure pathway balance across splits (no single pathway dominates
    # any split, preventing pathway-level leakage).
    log.info('Assigning target families...')
    target_families = {}
    for _, row in all_compounds.iterrows():
        target_families[row['compound_chembl_id']] = assign_target_family(row['gene_symbols'])

    family_dist = defaultdict(int)
    for fam in target_families.values():
        family_dist[fam] += 1

    log.info('--- Target family distribution ---')
    for fam, count in sorted(family_dist.items(), key=lambda x: -x[1])[:15]:
        log.info('  %s: %d', fam, count)

    with open(OUTPUT_FAMILIES, 'wb') as f:
        pickle.dump(target_families, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info('Saved → %s', OUTPUT_FAMILIES)


if __name__ == '__main__':
    main()
