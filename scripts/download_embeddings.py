#!/usr/bin/env python3
"""
Download pre-computed MolDeBERTa embeddings and generative-model checkpoints
from Zenodo.

These files are not tracked in git (see .gitignore) because of their size:
  - embeddings/compound_embeddings.mmap + compound_ids.npy  (~1.4 GB total)
  - generative/checkpoints/*.pt                             (~325 MB total)

Without them, the pipeline can still run from scratch:
  - Phase 0 (scripts/moldeberta_embedding_extract.py, ~45 min) regenerates
    the embeddings.
  - Phase 3 (generative/run_all.py, several hours per seed) regenerates the
    generative-model checkpoints.

The small PGCL checkpoints (modeling/checkpoints/best.pt and the 3 multiseed
checkpoints, all <10MB) ARE tracked directly in this repository and do not
need to be downloaded separately -- this script does not re-fetch them. They
are, however, also archived on the same Zenodo record for permanent, citable
access independent of GitHub, since the PGCL checkpoint is the single most
important artifact in the paper.

Zenodo DOI: 10.5281/zenodo.23340896

Usage:
    python scripts/download_embeddings.py
"""
import sys
import urllib.request
from pathlib import Path

ZENODO_DOI = "10.5281/zenodo.23340896"
ZENODO_BASE = "https://zenodo.org/records/23340896/files"

FILES = {
    "embeddings/compound_embeddings.mmap": f"{ZENODO_BASE}/compound_embeddings.mmap",
    "embeddings/compound_ids.npy": f"{ZENODO_BASE}/compound_ids.npy",
    "embeddings/meta.json": f"{ZENODO_BASE}/meta.json",
    "embeddings/progress.json": f"{ZENODO_BASE}/progress.json",
    "generative/checkpoints/best_vae.pt": f"{ZENODO_BASE}/best_vae.pt",
    "generative/checkpoints/best_cvae.pt": f"{ZENODO_BASE}/best_cvae.pt",
    "generative/checkpoints/best_diffusion.pt": f"{ZENODO_BASE}/best_diffusion.pt",
    "generative/checkpoints/best_rl_vae.pt": f"{ZENODO_BASE}/best_rl_vae.pt",
    "generative/checkpoints/best_cglf.pt": f"{ZENODO_BASE}/best_cglf.pt",
    "generative/checkpoints/best_crave.pt": f"{ZENODO_BASE}/best_crave.pt",
    "generative/checkpoints/cglf_decoder.pt": f"{ZENODO_BASE}/cglf_decoder.pt",
}

REPO_ROOT = Path(__file__).resolve().parents[1]


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {dest.relative_to(REPO_ROOT)} ...")
    urllib.request.urlretrieve(url, dest)


def main() -> None:
    if "[TBD]" in ZENODO_DOI:
        print("ERROR: Zenodo DOI not yet filled in.")
        print(f"  Please visit: https://doi.org/{ZENODO_DOI}")
        print("  and update ZENODO_DOI / ZENODO_BASE in this script.")
        print()
        print("Alternatively, regenerate the files locally:")
        print("  Phase 0: python scripts/moldeberta_embedding_extract.py")
        print("  Phase 3: python generative/run_all.py --seed 42")
        sys.exit(1)

    print(f"Downloading GENESIS large files from Zenodo ({ZENODO_DOI})")
    print(f"Destination: {REPO_ROOT}")
    print()

    for rel, url in FILES.items():
        dest = REPO_ROOT / rel
        if dest.exists():
            print(f"  Skipping {rel} (already present)")
            continue
        download(url, dest)

    print()
    print("All files downloaded. The full pipeline can now be run end-to-end,")
    print("or entered at Phase 4+ using the pre-trained checkpoints.")


if __name__ == "__main__":
    main()
