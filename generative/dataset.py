#!/usr/bin/env python3
"""
Phase 3, Step 2 — PyTorch Dataset for Generative Models.

Loads pre-tokenized SELFIES arrays and provides:
- Token sequences for autoregressive / VAE training
- Potency class labels and pChEMBL values for conditioning
- 128-d contrastive projections for CGLF conditioning
- Scaffold-aware train/val/test splits (reuses modeling splits)

Usage:
    from generative.dataset import get_dataloaders
    train_dl, val_dl, test_dl = get_dataloaders(batch_size=256)
"""
import os
import json
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

GENESIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN_DIR = os.path.join(GENESIS_DIR, 'generative', 'processed')
MODELING_DIR = os.path.join(GENESIS_DIR, 'modeling')
RESEARCH_DIR = os.path.dirname(GENESIS_DIR)


class SelfiesDataset(Dataset):
    """
    Dataset returning tokenized SELFIES + conditioning labels.

    Each item:
        tokens:         (max_len,) int64 — BOS ... tokens ... EOS PAD*
        potency_class:  int — 0 (inactive), 1 (active), 2 (potent), 3 (very_potent)
        pchembl:        float — measured pChEMBL value (NaN → 0.0)
        proj_128:       (128,) float32 — contrastive projection (if available, else zeros)
    """

    def __init__(self, split='train', load_projections=False, device='cpu'):
        # Load tokenized SELFIES
        self.tokens = np.load(os.path.join(GEN_DIR, 'selfies_tokens.npy'), mmap_mode='r')

        # Load compound metadata
        meta = pd.read_csv(os.path.join(GEN_DIR, 'compound_metadata.tsv'), sep='\t')

        # Filter by split
        mask = meta['split'] == split
        self.indices = np.where(mask.values)[0]
        self.meta = meta.iloc[self.indices].reset_index(drop=True)

        # Conditioning labels
        self.potency_class = self.meta['potency_class_num'].fillna(0).astype(int).values
        self.pchembl = self.meta['pchembl'].fillna(0.0).astype(np.float32).values
        self.compound_ids = self.meta['compound_chembl_id'].values

        # Optional: load 128-d projections from contrastive model
        self.proj_128 = None
        if load_projections:
            self._load_projections(device)

        # Load vocab for decoding
        with open(os.path.join(GEN_DIR, 'vocab.json')) as f:
            self.vocab = json.load(f)
        self.vocab_size = len(self.vocab)
        self.pad_idx = self.vocab['[PAD]']
        self.bos_idx = self.vocab['[BOS]']
        self.eos_idx = self.vocab['[EOS]']
        self.max_len = self.tokens.shape[1]

    def _load_projections(self, device):
        """Compute 128-d projections using the trained contrastive model."""
        import sys
        sys.path.insert(0, MODELING_DIR)
        from model import ContrastiveModel

        # Load model
        ckpt_path = os.path.join(MODELING_DIR, 'checkpoints', 'best.pt')
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model = ContrastiveModel()
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        # Load compound index + embeddings (from GENESIS paths)
        with open(os.path.join(MODELING_DIR, 'compound_index.pkl'), 'rb') as f:
            compound_index = pickle.load(f)

        emb_ids = np.load(
            os.path.join(GENESIS_DIR, 'embeddings', 'compound_ids.npy'), allow_pickle=True)
        n_compounds = len(emb_ids)
        emb_mmap = np.memmap(
            os.path.join(GENESIS_DIR, 'embeddings', 'compound_embeddings.mmap'),
            dtype='float16', mode='r', shape=(n_compounds, 768))

        # Project each compound
        projections = np.zeros((len(self.compound_ids), 128), dtype=np.float32)
        batch_size = 512
        model = model.to(device)

        for start in range(0, len(self.compound_ids), batch_size):
            end = min(start + batch_size, len(self.compound_ids))
            batch_embs = []
            for cid in self.compound_ids[start:end]:
                if cid in compound_index:
                    emb_idx = compound_index[cid]['emb_idx']
                    batch_embs.append(emb_mmap[emb_idx].astype(np.float32))
                else:
                    batch_embs.append(np.zeros(768, dtype=np.float32))
            embs_t = torch.tensor(np.stack(batch_embs), device=device)
            with torch.no_grad():
                z, _ = model(embs_t)
            projections[start:end] = z.cpu().numpy()

        self.proj_128 = projections

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        tokens = torch.tensor(self.tokens[self.indices[idx]], dtype=torch.long)
        potency = torch.tensor(self.potency_class[idx], dtype=torch.long)
        pchembl = torch.tensor(self.pchembl[idx], dtype=torch.float32)

        if self.proj_128 is not None:
            proj = torch.tensor(self.proj_128[idx], dtype=torch.float32)
        else:
            proj = torch.zeros(128, dtype=torch.float32)

        return {
            'tokens': tokens,
            'potency_class': potency,
            'pchembl': pchembl,
            'proj_128': proj,
        }


def get_dataloaders(batch_size=256, load_projections=False, device='cpu', num_workers=0):
    """Create train/val/test DataLoaders."""
    train_ds = SelfiesDataset('train', load_projections=load_projections, device=device)
    val_ds = SelfiesDataset('val', load_projections=load_projections, device=device)
    test_ds = SelfiesDataset('test', load_projections=load_projections, device=device)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=False, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=False)

    return train_dl, val_dl, test_dl


def decode_tokens(token_ids, inv_vocab):
    """Convert token IDs back to SELFIES string."""
    tokens = []
    for tid in token_ids:
        tid = int(tid)
        tok = inv_vocab.get(str(tid), '[UNK]')
        if tok == '[EOS]':
            break
        if tok not in ('[PAD]', '[BOS]'):
            tokens.append(tok)
    return ''.join(tokens)


if __name__ == '__main__':
    import time
    t0 = time.time()
    ds = SelfiesDataset('train')
    print(f'Train dataset: {len(ds)} samples, vocab_size={ds.vocab_size}, max_len={ds.max_len}')
    print(f'Load time: {time.time() - t0:.2f}s')

    sample = ds[0]
    print(f'Sample tokens shape: {sample["tokens"].shape}')
    print(f'Sample potency_class: {sample["potency_class"]}')
    print(f'Sample pchembl: {sample["pchembl"]:.2f}')

    train_dl, val_dl, test_dl = get_dataloaders(batch_size=64)
    batch = next(iter(train_dl))
    print(f'\nBatch shapes:')
    for k, v in batch.items():
        print(f'  {k}: {v.shape}')
