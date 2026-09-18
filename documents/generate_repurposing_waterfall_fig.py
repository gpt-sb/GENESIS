#!/usr/bin/env python3
"""Fig8: Drug repurposing waterfall - top-10 candidates by recommendation weight.
Values read directly from repurposing/results/repurposing_summary.json (fresh 2026-09-17 run).
Output: documents/images/Fig8_Repurposing_Waterfall.png
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
GENESIS = os.path.dirname(BASE)
OUT = os.path.join(BASE, "images", "Fig8_Repurposing_Waterfall.png")

summ = json.load(open(os.path.join(GENESIS, "repurposing/results/repurposing_summary.json")))
top10 = summ["top10"]

names = [c["compound_name"].title() if c.get("compound_name") else c["compound_chembl_id"] for c in top10]
weights = [c["recommendation_weight"] for c in top10]
flags = [c["overall_flag"] for c in top10]
FLAG_C = {"PASS": "#2ca02c", "CAUTION": "#ff7f0e", "FAIL": "#d62728"}
colors = [FLAG_C.get(f, "#888") for f in flags]

fig, ax = plt.subplots(figsize=(9, 6))
y = range(len(names))
bars = ax.barh(y, weights, color=colors, edgecolor="white", linewidth=0.6, zorder=3)
ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=9)
ax.invert_yaxis()
for b, w, c in zip(bars, weights, top10):
    ax.text(w + 0.005, b.get_y() + b.get_height() / 2,
            f"{w:.4f} ({c.get('clinical_phase', 'n/a')})",
            va="center", fontsize=7.5)
ax.set_xlabel("Recommendation weight")
ax.set_xlim(0, 1.0)
ax.set_title(f"Top-10 drug repurposing candidates\n"
             f"(from {summ['n_total_compounds']:,} compounds screened against "
             f"{summ['n_targets_scored']} OSCC targets)", fontsize=10.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FLAG_C.values()]
ax.legend(handles, FLAG_C.keys(), loc="lower right", fontsize=8, title="Safety flag")

fig.tight_layout()
fig.savefig(OUT, dpi=300, facecolor="white")
print(f"Saved -> {OUT}")
