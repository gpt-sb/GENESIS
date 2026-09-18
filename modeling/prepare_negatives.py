#!/usr/bin/env python3
"""
Phase 1.2 — Build unified negative table with tier confidence tags.

Reads all 6 negative tier files, harmonizes columns, resolves
positive/negative collisions, and outputs GENESIS/modeling/negatives.tsv.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent        # GENESIS/modeling/
GENESIS_DIR = SCRIPT_DIR.parent                     # GENESIS/

NEG_DIR = GENESIS_DIR / 'negatives'
POS_FILE = SCRIPT_DIR / 'positives.tsv'
OUTPUT = SCRIPT_DIR / 'negatives.tsv'

HARDNESS_MAP = {
    'T1': 'hard',
    'T2': 'medium',
    'T3A_inactive': 'gold',
    'T3B': 'soft',
    'T3C': 'hard',
    'T4': 'easy',
}

UNIFIED_COLS = [
    'compound_chembl_id', 'smiles', 'neg_tier', 'neg_hardness',
    'ic50_nM', 'pchembl', 'target_name', 'gene_symbols',
]


def load_tier1():
    df = pd.read_csv(NEG_DIR / 'tier1_confirmed_inactives.tsv', sep='\t', low_memory=False)
    return pd.DataFrame({
        'compound_chembl_id': df['compound_chembl_id'],
        'smiles': df['smiles'],
        'neg_tier': 'T1',
        'neg_hardness': 'hard',
        'ic50_nM': pd.to_numeric(df.get('ic50_nM'), errors='coerce'),
        'pchembl': pd.to_numeric(df.get('pchembl'), errors='coerce'),
        'target_name': df.get('target_name'),
        'gene_symbols': df.get('gene_symbols'),
    })


def load_tier2():
    df = pd.read_csv(NEG_DIR / 'tier2_other_cancer_actives.tsv', sep='\t', low_memory=False)
    return pd.DataFrame({
        'compound_chembl_id': df['compound_chembl_id'],
        'smiles': df['smiles'],
        'neg_tier': 'T2',
        'neg_hardness': 'medium',
        'ic50_nM': pd.to_numeric(df.get('best_ic50_nM'), errors='coerce'),
        'pchembl': pd.to_numeric(df.get('best_pchembl'), errors='coerce'),
        'target_name': df.get('target_name'),
        'gene_symbols': df.get('gene_symbols'),
    })


def load_tier3a():
    """Extract only the INACTIVE side of activity cliff pairs."""
    df = pd.read_csv(NEG_DIR / 'tier3a_activity_cliff_pairs.tsv', sep='\t', low_memory=False)
    return pd.DataFrame({
        'compound_chembl_id': df['inactive_chembl_id'],
        'smiles': df['inactive_smiles'],
        'neg_tier': 'T3A_inactive',
        'neg_hardness': 'gold',
        'ic50_nM': pd.to_numeric(df.get('inactive_ic50_nM'), errors='coerce'),
        'pchembl': np.nan,
        'target_name': df.get('shared_targets'),
        'gene_symbols': np.nan,
    })


def load_tier3b():
    df = pd.read_csv(NEG_DIR / 'tier3b_scaffold_matched.tsv', sep='\t', low_memory=False)
    return pd.DataFrame({
        'compound_chembl_id': df['compound_chembl_id'],
        'smiles': df['smiles'],
        'neg_tier': 'T3B',
        'neg_hardness': 'soft',
        'ic50_nM': np.nan,
        'pchembl': np.nan,
        'target_name': np.nan,
        'gene_symbols': np.nan,
    })


def load_tier3c():
    df = pd.read_csv(NEG_DIR / 'tier3c_target_switching.tsv', sep='\t', low_memory=False)
    return pd.DataFrame({
        'compound_chembl_id': df['compound_chembl_id'],
        'smiles': df['smiles'],
        'neg_tier': 'T3C',
        'neg_hardness': 'hard',
        'ic50_nM': pd.to_numeric(df.get('best_ic50_nM'), errors='coerce'),
        'pchembl': pd.to_numeric(df.get('best_pchembl'), errors='coerce'),
        'target_name': np.nan,
        'gene_symbols': df.get('gene_symbols'),
    })


def load_tier4():
    df = pd.read_csv(NEG_DIR / 'tier4_background.tsv', sep='\t', low_memory=False)
    return pd.DataFrame({
        'compound_chembl_id': df['compound_chembl_id'],
        'smiles': df['smiles'],
        'neg_tier': 'T4',
        'neg_hardness': 'easy',
        'ic50_nM': np.nan,
        'pchembl': np.nan,
        'target_name': np.nan,
        'gene_symbols': np.nan,
    })


def main():
    # Load all tiers
    loaders = [
        ('T1', load_tier1),
        ('T2', load_tier2),
        ('T3A', load_tier3a),
        ('T3B', load_tier3b),
        ('T3C', load_tier3c),
        ('T4', load_tier4),
    ]

    dfs = []
    for name, loader in loaders:
        df = loader()
        log.info('Loaded %s: %d rows', name, len(df))
        dfs.append(df)

    neg = pd.concat(dfs, ignore_index=True)
    log.info('Total negatives (raw): %d', len(neg))

    # Deduplicate: keep the hardest tier for duplicates
    # Priority: gold > hard > medium > soft > easy
    tier_priority = {'gold': 0, 'hard': 1, 'medium': 2, 'soft': 3, 'easy': 4}
    neg['_priority'] = neg['neg_hardness'].map(tier_priority)
    neg = neg.sort_values('_priority').drop_duplicates(subset='compound_chembl_id', keep='first')
    neg = neg.drop(columns='_priority')
    log.info('After dedup (keep hardest): %d unique compounds', len(neg))

    # Remove compounds that appear in positive set
    pos = pd.read_csv(POS_FILE, sep='\t', usecols=['compound_chembl_id'])
    pos_ids = set(pos['compound_chembl_id'].dropna())
    before = len(neg)
    neg = neg[~neg['compound_chembl_id'].isin(pos_ids)]
    log.info('Removed %d positive-overlap compounds from negatives', before - len(neg))
    log.info('Final negatives: %d', len(neg))

    # Log tier distribution
    log.info('--- Tier distribution ---')
    for tier, count in neg['neg_tier'].value_counts().sort_index().items():
        log.info('  %s: %d', tier, count)

    # Save
    neg[UNIFIED_COLS].to_csv(OUTPUT, sep='\t', index=False)
    log.info('Saved → %s', OUTPUT)


if __name__ == '__main__':
    main()
