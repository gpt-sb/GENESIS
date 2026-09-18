#!/usr/bin/env python3
"""
Phase 3, Step 1 — Data Preparation for Generative Models.

Converts SMILES → SELFIES, builds vocabulary, tokenizes,
and saves processed arrays for fast training.

Usage:
    python generative/data_prep.py [--workers 4]

Outputs (in generative/processed/):
    vocab.json          — token-to-index mapping
    inv_vocab.json      — index-to-token mapping
    selfies_tokens.npy  — (N, max_len) int32 padded token arrays
    compound_metadata.tsv — per-compound metadata with splits
    metadata.json       — dataset statistics
    failed_smiles.txt   — SMILES that couldn't be converted
"""
import os
import sys
import json
import logging
import argparse
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
import selfies as sf
from rdkit import Chem

# GENESIS root is two levels up from this file (GENESIS/generative/data_prep.py)
GENESIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELING_DIR = os.path.join(GENESIS_DIR, 'modeling')
OUTPUT_DIR = os.path.join(GENESIS_DIR, 'generative', 'processed')

# Source data lives in the Research root (one level above GENESIS)
RESEARCH_DIR = os.path.dirname(GENESIS_DIR)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# Special tokens
PAD_TOKEN = '[PAD]'
BOS_TOKEN = '[BOS]'
EOS_TOKEN = '[EOS]'
UNK_TOKEN = '[UNK]'
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


def canonicalize_smiles(smi):
    """Canonicalize SMILES string using RDKit."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def smiles_to_selfies(smi):
    """Convert a single SMILES to SELFIES. Returns (selfies_str, error_msg)."""
    canon = canonicalize_smiles(smi)
    if canon is None:
        return None, 'invalid_smiles'
    try:
        sel = sf.encoder(canon)
        if sel is None:
            return None, 'selfies_encode_failed'
        # Verify roundtrip
        decoded = sf.decoder(sel)
        if decoded is None:
            return None, 'selfies_roundtrip_failed'
        return sel, None
    except Exception as e:
        return None, str(e)


def process_chunk(args):
    """Process a chunk of (idx, smiles) pairs. Returns list of (idx, selfies_str, error)."""
    chunk = args
    results = []
    for idx, smi in chunk:
        sel, err = smiles_to_selfies(smi)
        results.append((idx, sel, err))
    return results


def build_vocabulary(selfies_list, min_count=2):
    """Build token vocabulary from list of SELFIES strings."""
    token_counter = Counter()
    for sel in selfies_list:
        tokens = list(sf.split_selfies(sel))
        token_counter.update(tokens)

    # Filter by minimum count
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    idx = len(SPECIAL_TOKENS)
    for tok, count in token_counter.most_common():
        if count >= min_count:
            vocab[tok] = idx
            idx += 1

    return vocab, token_counter


def tokenize_selfies(selfies_str, vocab, max_len):
    """Tokenize SELFIES string to integer array with BOS/EOS."""
    tokens = list(sf.split_selfies(selfies_str))
    unk_idx = vocab[UNK_TOKEN]

    # BOS + tokens + EOS, truncated to max_len
    ids = [vocab[BOS_TOKEN]]
    for tok in tokens[:max_len - 2]:  # leave room for BOS + EOS
        ids.append(vocab.get(tok, unk_idx))
    ids.append(vocab[EOS_TOKEN])

    # Pad
    while len(ids) < max_len:
        ids.append(vocab[PAD_TOKEN])

    return ids


def main():
    parser = argparse.ArgumentParser(description='SELFIES data preparation')
    parser.add_argument('--workers', type=int, default=4, help='Parallel workers')
    parser.add_argument('--max-len', type=int, default=128, help='Max SELFIES token length (including BOS/EOS)')
    parser.add_argument('--min-count', type=int, default=2, help='Min token frequency for vocab')
    parser.add_argument('--percentile', type=float, default=99.5, help='Length percentile for max_len auto')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load compound data ──────────────────────────────────────────
    tsv_path = os.path.join(RESEARCH_DIR, 'All_Compounds_Unq_Std.tsv')
    log.info('Loading compounds from %s', tsv_path)
    df = pd.read_csv(tsv_path, sep='\t', usecols=[
        'compound_chembl_id', 'smiles', 'potency_class_num', 'pchembl', 'target_name'
    ])
    df = df.dropna(subset=['smiles']).drop_duplicates(subset=['compound_chembl_id'])
    log.info('Loaded %d unique compounds with SMILES', len(df))

    # ── Load splits (from GENESIS/modeling/) ─────────────────────────
    with open(os.path.join(MODELING_DIR, 'splits.json')) as f:
        splits = json.load(f)

    train_ids = set(splits['train']['positive_ids'] + splits['train']['negative_ids'])
    val_ids = set(splits['val']['positive_ids'] + splits['val']['negative_ids'])
    test_ids = set(splits['test']['positive_ids'] + splits['test']['negative_ids'])

    def assign_split(cid):
        if cid in train_ids:
            return 'train'
        elif cid in val_ids:
            return 'val'
        elif cid in test_ids:
            return 'test'
        return 'unused'

    df['split'] = df['compound_chembl_id'].apply(assign_split)
    split_counts = df['split'].value_counts()
    log.info('Split distribution:\n%s', split_counts.to_string())

    # ── Convert SMILES → SELFIES (parallel) ──────────────────────────
    log.info('Converting SMILES → SELFIES with %d workers ...', args.workers)
    pairs = list(zip(range(len(df)), df['smiles'].values))

    chunk_size = max(1, len(pairs) // (args.workers * 10))
    chunks = [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]

    all_results = []
    with Pool(args.workers) as pool:
        for batch_results in pool.imap_unordered(process_chunk, chunks):
            all_results.extend(batch_results)

    # Sort back to original order
    all_results.sort(key=lambda x: x[0])

    selfies_strs = [None] * len(df)
    errors = []
    for idx, sel, err in all_results:
        if sel is not None:
            selfies_strs[idx] = sel
        else:
            errors.append((int(idx), df.iloc[idx]['compound_chembl_id'], df.iloc[idx]['smiles'], err))

    n_success = sum(1 for s in selfies_strs if s is not None)
    n_fail = len(errors)
    log.info('SELFIES conversion: %d success, %d failed (%.2f%% success rate)',
             n_success, n_fail, 100 * n_success / len(df))

    # Save failed SMILES
    if errors:
        fail_path = os.path.join(OUTPUT_DIR, 'failed_smiles.txt')
        with open(fail_path, 'w') as f:
            f.write('idx\tcompound_chembl_id\tsmiles\terror\n')
            for idx, cid, smi, err in errors:
                f.write(f'{idx}\t{cid}\t{smi}\t{err}\n')
        log.info('Failed SMILES → %s', fail_path)

    # ── Build mask for successful conversions ────────────────────────
    valid_mask = [s is not None for s in selfies_strs]
    valid_selfies = [s for s in selfies_strs if s is not None]
    df_valid = df[valid_mask].copy()
    df_valid['selfies'] = valid_selfies

    # ── Compute length distribution ──────────────────────────────────
    lengths = [len(list(sf.split_selfies(s))) for s in valid_selfies]
    lengths_arr = np.array(lengths)

    p50 = int(np.percentile(lengths_arr, 50))
    p90 = int(np.percentile(lengths_arr, 90))
    p95 = int(np.percentile(lengths_arr, 95))
    p99 = int(np.percentile(lengths_arr, 99))
    p995 = int(np.percentile(lengths_arr, 99.5))
    p_target = int(np.percentile(lengths_arr, args.percentile))

    log.info('SELFIES length distribution:')
    log.info('  min=%d, max=%d, mean=%.1f, median=%d', lengths_arr.min(), lengths_arr.max(),
             lengths_arr.mean(), p50)
    log.info('  p90=%d, p95=%d, p99=%d, p99.5=%d', p90, p95, p99, p995)

    # +2 for BOS/EOS tokens
    auto_max_len = p_target + 2
    max_len = min(args.max_len, auto_max_len) if args.max_len else auto_max_len
    log.info('Using max_len=%d (p%.1f=%d + 2 for BOS/EOS)', max_len, args.percentile, p_target)

    # ── Build vocabulary ──────────────────────────────────────────────
    log.info('Building vocabulary ...')
    vocab, token_counter = build_vocabulary(valid_selfies, min_count=args.min_count)
    log.info('Vocabulary size: %d tokens (min_count=%d)', len(vocab), args.min_count)
    log.info('Top-20 tokens: %s', token_counter.most_common(20))

    # ── Tokenize ─────────────────────────────────────────────────────
    log.info('Tokenizing %d SELFIES (max_len=%d) ...', len(valid_selfies), max_len)
    token_arrays = np.zeros((len(valid_selfies), max_len), dtype=np.int32)
    n_truncated = 0
    for i, sel in enumerate(valid_selfies):
        tokens = list(sf.split_selfies(sel))
        if len(tokens) > max_len - 2:
            n_truncated += 1
        token_arrays[i] = tokenize_selfies(sel, vocab, max_len)
    log.info('Truncated %d sequences (%.2f%%)', n_truncated, 100 * n_truncated / len(valid_selfies))

    # ── Save outputs ─────────────────────────────────────────────────
    # 1. Tokenized arrays
    np.save(os.path.join(OUTPUT_DIR, 'selfies_tokens.npy'), token_arrays)
    log.info('Tokenized arrays → selfies_tokens.npy  shape=%s', token_arrays.shape)

    # 2. Vocabulary
    vocab_path = os.path.join(OUTPUT_DIR, 'vocab.json')
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, indent=2)
    log.info('Vocabulary → vocab.json (%d tokens)', len(vocab))

    # 3. Compound metadata (compound_id, split, potency, target, selfies_length)
    df_valid = df_valid.copy()
    df_valid['selfies_len'] = lengths
    meta_cols = ['compound_chembl_id', 'smiles', 'selfies', 'split',
                 'potency_class_num', 'pchembl', 'target_name', 'selfies_len']
    meta_path = os.path.join(OUTPUT_DIR, 'compound_metadata.tsv')
    df_valid[meta_cols].to_csv(meta_path, sep='\t', index=False)
    log.info('Compound metadata → compound_metadata.tsv (%d rows)', len(df_valid))

    # 4. Summary statistics
    stats = {
        'n_input_compounds': int(len(df)),
        'n_valid_selfies': int(n_success),
        'n_failed': int(n_fail),
        'success_rate': float(n_success / len(df)),
        'vocab_size': int(len(vocab)),
        'max_len': int(max_len),
        'length_stats': {
            'min': int(lengths_arr.min()),
            'max': int(lengths_arr.max()),
            'mean': float(lengths_arr.mean()),
            'median': int(p50),
            'p90': int(p90),
            'p95': int(p95),
            'p99': int(p99),
            'p99.5': int(p995),
        },
        'split_counts': {
            'train': int((df_valid['split'] == 'train').sum()),
            'val': int((df_valid['split'] == 'val').sum()),
            'test': int((df_valid['split'] == 'test').sum()),
            'unused': int((df_valid['split'] == 'unused').sum()),
        },
        'potency_distribution': df_valid['potency_class_num'].value_counts().to_dict(),
        'n_truncated': int(n_truncated),
        'special_tokens': SPECIAL_TOKENS,
    }
    stats_path = os.path.join(OUTPUT_DIR, 'metadata.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    log.info('Statistics → metadata.json')

    # 5. Inverse vocab for decoding
    inv_vocab = {v: k for k, v in vocab.items()}
    inv_path = os.path.join(OUTPUT_DIR, 'inv_vocab.json')
    with open(inv_path, 'w') as f:
        json.dump(inv_vocab, f, indent=2)

    log.info('Data preparation complete.')


if __name__ == '__main__':
    main()
