#!/usr/bin/env python3
"""
MolDeBERTa Embedding Extraction Script for GENESIS

- Reads unique SMILES from GENESIS/data/CompoundList.tsv
- Uses HuggingFace checkpoint: SaeedLab/MolDeBERTa-base-123M-mtr
- Saves embeddings (float16, shape [N, 768]) as memmap in GENESIS/embeddings/compound_embeddings.mmap
- Saves compound IDs as GENESIS/embeddings/compound_ids.npy
- Logs meta and progress as JSON
- Batches for memory efficiency, logs time for major steps

Author: [Your Name]
Date: 2026-03-21
"""

import os, csv, time, json
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# Config
INPUT_TSV  = os.path.join(os.path.dirname(__file__), '../data/CompoundList.tsv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../embeddings')
MODEL_NAME = "SaeedLab/MolDeBERTa-base-123M-mtr"
BATCH_SIZE = 64
EMB_DIM    = 768
EMB_DTYPE  = np.float16
MAX_LENGTH = 128
LOG_EVERY  = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output paths
mmap_path     = os.path.join(OUTPUT_DIR, "compound_embeddings.mmap")
ids_path      = os.path.join(OUTPUT_DIR, "compound_ids.npy")
meta_path     = os.path.join(OUTPUT_DIR, "meta.json")
progress_path = os.path.join(OUTPUT_DIR, "progress.json")

# Device
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Device: {device}")

# Load model
print(f"Loading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()
print(f"Model ready: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

# Read SMILES and IDs
print("Reading CompoundList.tsv ...")
compound_ids, smiles_list = [], []
with open(INPUT_TSV) as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        cid = row.get('ChEMBL_ID', '').strip()
        smi = row.get('SMILES', '').strip()
        if cid and smi:
            compound_ids.append(cid)
            smiles_list.append(smi)
N = len(compound_ids)
print(f"Total compounds: {N}")

# Prepare output memmap
embeddings = np.memmap(mmap_path, dtype=EMB_DTYPE, mode='w+', shape=(N, EMB_DIM))
np.save(ids_path, np.array(compound_ids))
with open(meta_path, 'w') as f:
    json.dump({'model': MODEL_NAME, 'date': time.strftime('%Y-%m-%d %H:%M:%S'), 'N': N, 'emb_dim': EMB_DIM}, f, indent=2)

# Embedding extraction loop
start_time = time.time()
for i in range(0, N, BATCH_SIZE):
    batch_smiles = smiles_list[i:i+BATCH_SIZE]
    batch_ids = compound_ids[i:i+BATCH_SIZE]
    enc = tokenizer(batch_smiles, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='pt')
    for k in enc:
        enc[k] = enc[k].to(device)
    with torch.no_grad():
        out = model(**enc)
        batch_emb = out.last_hidden_state[:, 0, :].cpu().numpy().astype(EMB_DTYPE)  # [CLS] token
    embeddings[i:i+len(batch_emb)] = batch_emb
    if (i // BATCH_SIZE) % LOG_EVERY == 0:
        elapsed = time.time() - start_time
        print(f"Processed {i+len(batch_emb)}/{N} compounds ({100*(i+len(batch_emb))/N:.1f}%), elapsed: {elapsed/60:.1f} min")
        with open(progress_path, 'w') as f:
            json.dump({'done': i+len(batch_emb), 'total': N, 'elapsed_sec': elapsed}, f)
embeddings.flush()
total_time = time.time() - start_time
print(f"Embedding extraction complete. Total time: {total_time/60:.1f} min")

# Final progress log
with open(progress_path, 'w') as f:
    json.dump({'done': N, 'total': N, 'elapsed_sec': total_time}, f)
