#!/usr/bin/env python3
"""
Phase 1.3 — Embedding index mapping.

Maps every compound in positives and negatives to its row index in the
embedding memmap. Outputs GENESIS/modeling/compound_index.pkl.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent        # GENESIS/modeling/
GENESIS_DIR = SCRIPT_DIR.parent                     # GENESIS/

POS_FILE = SCRIPT_DIR / 'positives.tsv'
NEG_FILE = SCRIPT_DIR / 'negatives.tsv'
EMB_IDS = GENESIS_DIR / 'embeddings' / 'compound_ids.npy'
EMB_MMAP = GENESIS_DIR / 'embeddings' / 'compound_embeddings.mmap'
OUTPUT = SCRIPT_DIR / 'compound_index.pkl'


def main():
    # Load embedding ID index
    log.info('Loading embedding IDs...')
    emb_ids = np.load(EMB_IDS, allow_pickle=True)
    id_to_idx = {cid: i for i, cid in enumerate(emb_ids)}
    log.info('Embedding store: %d compounds', len(id_to_idx))

    # Verify memmap is accessible
    mmap = np.memmap(EMB_MMAP, dtype='float16', mode='r', shape=(len(emb_ids), 768))
    log.info('Memmap shape: %s, dtype: %s', mmap.shape, mmap.dtype)

    # Load positives
    pos = pd.read_csv(POS_FILE, sep='\t', low_memory=False)
    log.info('Positives: %d', len(pos))

    # Load negatives
    neg = pd.read_csv(NEG_FILE, sep='\t', low_memory=False)
    log.info('Negatives: %d', len(neg))

    # Build compound index
    compound_index = {}
    missing_pos = 0
    missing_neg = 0
    zero_emb = 0

    # Process positives
    for _, row in pos.iterrows():
        cid = row['compound_chembl_id']
        if cid not in id_to_idx:
            missing_pos += 1
            continue
        idx = id_to_idx[cid]
        emb = mmap[idx]
        if np.all(emb == 0):
            zero_emb += 1
        compound_index[cid] = {
            'emb_idx': idx,
            'label': 'positive',
            'potency_class_num': int(row['potency_class_num']) if pd.notna(row['potency_class_num']) else None,
            'pchembl_filled': float(row['pchembl_filled']) if pd.notna(row.get('pchembl_filled')) else None,
            'neg_tier': None,
            'target_name': row.get('target_name'),
            'gene_symbols': row.get('gene_symbols'),
            'in_plos': int(row.get('in_plos', 0)) if pd.notna(row.get('in_plos')) else 0,
            'max_phase': float(row.get('max_phase', 0)) if pd.notna(row.get('max_phase')) else 0,
        }

    # Process negatives
    for _, row in neg.iterrows():
        cid = row['compound_chembl_id']
        if cid not in id_to_idx:
            missing_neg += 1
            continue
        if cid in compound_index:
            continue
        idx = id_to_idx[cid]
        emb = mmap[idx]
        if np.all(emb == 0):
            zero_emb += 1
        compound_index[cid] = {
            'emb_idx': idx,
            'label': 'negative',
            'potency_class_num': None,
            'pchembl_filled': None,
            'neg_tier': row['neg_tier'],
            'target_name': row.get('target_name') if pd.notna(row.get('target_name')) else None,
            'gene_symbols': row.get('gene_symbols') if pd.notna(row.get('gene_symbols')) else None,
            'in_plos': 0,
            'max_phase': 0,
        }

    log.info('--- Compound index stats ---')
    log.info('Total indexed: %d', len(compound_index))
    n_pos = sum(1 for v in compound_index.values() if v['label'] == 'positive')
    n_neg = sum(1 for v in compound_index.values() if v['label'] == 'negative')
    log.info('Positives: %d', n_pos)
    log.info('Negatives: %d', n_neg)
    log.info('Missing from embeddings (pos): %d', missing_pos)
    log.info('Missing from embeddings (neg): %d', missing_neg)
    log.info('Zero-vector embeddings: %d', zero_emb)

    # Save
    with open(OUTPUT, 'wb') as f:
        pickle.dump(compound_index, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info('Saved → %s', OUTPUT)


if __name__ == '__main__':
    main()
