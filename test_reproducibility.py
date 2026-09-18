#!/usr/bin/env python3
"""
GENESIS - Reproducibility Verification Test
Run from repo root: python test_reproducibility.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

P, F, W = "\033[92m\u2713\033[0m", "\033[91m\u2717\033[0m", "\033[93m\u26a0\033[0m"
failures, warnings = [], []


def check(ok, label, detail=""):
    if ok:
        print(f"  {P} {label}")
    else:
        print(f"  {F} {label}" + (f"\n      -> {detail}" if detail else ""))
        failures.append(label)
    return ok


def warn(ok, label, detail=""):
    if not ok:
        print(f"  {W} {label}" + (f"\n      -> {detail}" if detail else ""))
        warnings.append(label)
    else:
        print(f"  {P} {label}")
    return ok


def section(t):
    print(f"\n{'-' * 60}\n  {t}\n{'-' * 60}")


section("1. Required code files")
for rel in [
    "modeling/model.py", "modeling/losses.py", "modeling/train.py",
    "modeling/evaluate.py", "modeling/dataset.py",
    "modeling/prepare_positives.py", "modeling/prepare_negatives.py",
    "modeling/create_splits.py", "modeling/build_index.py",
    "generative/vae.py", "generative/crave_gen.py", "generative/run_all.py",
    "scripts/moldeberta_embedding_extract.py",
    "scripts/download_embeddings.py",
    "docking/run_molecular_docking.py",
    "admet/run_admet_profiling.py",
    "repurposing/run_drug_repurposing.py",
    "final_ranking/run_final_ranking.py",
    "documents/manuscript_ground_truth.json",
    "documents/verify_manuscript_numbers.py",
    "README.md", "requirements.txt", "REPRODUCTION_PLAN.md", "LICENSE",
]:
    check((ROOT / rel).exists(), rel, f"Missing: {ROOT / rel}")

section("2. Small tracked checkpoints (should be present, no download needed)")
for rel in [
    "modeling/checkpoints/best.pt",
    "modeling/validation/multiseed/checkpoints/best_seed42.pt",
    "modeling/validation/multiseed/checkpoints/best_seed123.pt",
    "modeling/validation/multiseed/checkpoints/best_seed456.pt",
]:
    check((ROOT / rel).exists(), rel, f"Missing: {ROOT / rel}")

section("3. Zenodo-hosted large files (python scripts/download_embeddings.py)")
for rel, desc in {
    "embeddings/compound_embeddings.mmap": "MolDeBERTa embeddings (~1.1 GB)",
    "generative/checkpoints/best_vae.pt": "VAE checkpoint",
    "generative/checkpoints/best_crave.pt": "CRAVE checkpoint",
}.items():
    warn((ROOT / rel).exists(), f"{rel}  [{desc}]",
         "Not found -- run: python scripts/download_embeddings.py")

section("4. No hardcoded credentials")
import re
cred_re = re.compile(r"password\s*=\s*['\"][^'\"]+['\"]", re.IGNORECASE)
hits = []
for py in ROOT.rglob("*.py"):
    if "__pycache__" in py.parts:
        continue
    try:
        text = py.read_text(errors="replace")
    except Exception:
        continue
    if cred_re.search(text):
        hits.append(str(py.relative_to(ROOT)))
check(len(hits) == 0, "No hardcoded credentials in any Python script",
      f"Found in: {hits}")

section("5. Python module imports")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "modeling"))
for mod, sym in [
    ("model", "ContrastiveModel"),
    ("losses", "OrdinalSupConLoss"),
]:
    try:
        m = __import__(mod)
        check(hasattr(m, sym), f"{mod}.{sym} importable")
    except Exception as e:
        check(False, f"{mod}.{sym} importable", str(e))

print(f"\n{'=' * 60}")
if not failures:
    print(f"  {P} All checks passed ({len(warnings)} warning(s))")
    for w in warnings:
        print(f"    {W}  {w}")
    print("  Warnings are non-blocking (Zenodo files not yet downloaded).")
else:
    print(f"  \033[91mFAILED: {len(failures)} error(s), {len(warnings)} warning(s)\033[0m")
    for f in failures:
        print(f"    {F}  {f}")
print(f"{'=' * 60}\n")
sys.exit(0 if not failures else 1)
