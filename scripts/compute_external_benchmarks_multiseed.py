#!/usr/bin/env python3
"""
Wrapper: run compute_external_benchmarks.py's per-method metrics across the
3 canonical seeds (42, 123, 456), reading from generative/multiseed/seed_X/,
then aggregate mean +/- SD per method/metric.

Output: generative/results/external_benchmarks_3seed.json
"""
import copy
import json
import os

import compute_external_benchmarks as ceb

ROOT = ceb.ROOT
MULTISEED_DIR = os.path.join(ROOT, "generative", "multiseed")
OUT_JSON = os.path.join(ROOT, "generative", "results", "external_benchmarks_3seed.json")

SEEDS = [42, 123, 456]
METHOD_KEYS = ["VAE", "cVAE", "Diffusion", "RL-VAE", "CGLF", "CRAVE (filt.)", "CRAVE (unfilt.)"]
FILE_STEMS = {
    "VAE": "generated_vae.txt",
    "cVAE": "generated_cvae.txt",
    "Diffusion": "generated_diffusion.txt",
    "RL-VAE": "generated_rl_vae.txt",
    "CGLF": "generated_cglf.txt",
    "CRAVE (filt.)": "generated_crave.txt",
    "CRAVE (unfilt.)": "generated_crave_unfiltered.txt",
}


def main():
    per_seed_results = {}

    for seed in SEEDS:
        print(f"\n{'='*70}\nSEED {seed}\n{'='*70}")
        seed_dir = os.path.join(MULTISEED_DIR, f"seed_{seed}")
        ceb.METHOD_FILES = {name: os.path.join(seed_dir, stem) for name, stem in FILE_STEMS.items()}
        ceb.OUT_JSON = os.path.join(ROOT, "generative", "results", f"external_benchmarks_seed{seed}.json")

        ceb.main()

        with open(ceb.OUT_JSON) as f:
            per_seed_results[seed] = json.load(f)

    # ── Aggregate mean +/- SD across seeds ─────────────────────────────
    print(f"\n{'='*70}\nAGGREGATING ACROSS {len(SEEDS)} SEEDS\n{'='*70}")
    import numpy as np

    aggregated = {}
    for method in METHOD_KEYS:
        moses_keys = list(per_seed_results[SEEDS[0]][method]["moses"].keys())
        kl_keys = list(per_seed_results[SEEDS[0]][method]["guacamol_kl"].keys())

        moses_agg = {}
        for k in moses_keys:
            vals = [per_seed_results[s][method]["moses"][k] for s in SEEDS if method in per_seed_results[s]]
            moses_agg[k] = {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4), "per_seed": vals}

        kl_agg = {}
        for k in kl_keys:
            vals = [per_seed_results[s][method]["guacamol_kl"][k] for s in SEEDS if method in per_seed_results[s]]
            kl_agg[k] = {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4), "per_seed": vals}

        aggregated[method] = {"moses": moses_agg, "guacamol_kl": kl_agg}

    with open(OUT_JSON, "w") as f:
        json.dump({"seeds": SEEDS, "per_method": aggregated}, f, indent=2)
    print(f"\nAggregated 3-seed results -> {OUT_JSON}")

    # Summary table
    print(f"\n{'Method':<18s} {'IntDiv':>14s} {'SNN':>14s} {'Frag':>14s} {'Scaf':>14s}")
    print("-" * 76)
    for method in METHOD_KEYS:
        m = aggregated[method]["moses"]
        print(f"{method:<18s} "
              f"{m['IntDiv']['mean']:.3f}\u00b1{m['IntDiv']['std']:.3f}  "
              f"{m['SNN']['mean']:.3f}\u00b1{m['SNN']['std']:.3f}  "
              f"{m['Frag']['mean']:.3f}\u00b1{m['Frag']['std']:.3f}  "
              f"{m['Scaf']['mean']:.3f}\u00b1{m['Scaf']['std']:.3f}")


if __name__ == "__main__":
    main()
