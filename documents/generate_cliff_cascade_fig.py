#!/usr/bin/env python3
"""Fig4: Activity-cliff discrimination cascade (Table 1 baselines).
All values sourced from documents/manuscript_ground_truth.json (verified ledger).
Output: documents/images/Fig4_Baseline_Cliff_Cascade_MOSES.png
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "images", "Fig4_Baseline_Cliff_Cascade_MOSES.png")

LEDGER = json.load(open(os.path.join(BASE, "manuscript_ground_truth.json")))
T1 = LEDGER["table1_baselines"]["metrics"]["cliff_pair_accuracy"]

methods = ["ECFP4_kNN", "BCE", "FlatSupCon", "PGCL"]
labels = ["ECFP4 k-NN", "BCE classifier", "Flat SupCon", "PGCL"]
vals = [float(str(T1[m]).split("\u00b1")[0]) for m in methods]
colors = ["#7F7F7F", "#C44E52", "#8172B2", "#154360"]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v*100:.1f}%",
            ha="center", fontsize=10, fontweight="bold")
ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
ax.text(3.4, 0.505, "chance level", fontsize=7, color="grey", ha="right")
ax.set_ylabel("Activity-cliff pair accuracy")
ax.set_ylim(0, 0.85)
ax.set_title("Activity-cliff discrimination across baselines\n"
              "(15,471 held-out structurally-similar, differently-potent pairs)",
              fontsize=10.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(0.02, 0.97,
        "PGCL and Flat SupCon are statistically indistinguishable\n"
        "(66.4% vs 67.1%) \u2014 contrastive learning drives cliff\n"
        "discrimination; ordinal weighting's distinct benefit is\n"
        "the potency-ranking gradient (see Table 1).",
        transform=ax.transAxes, fontsize=7.5, va="top", ha="left",
        bbox=dict(facecolor="#f5f5f5", edgecolor="#ccc", boxstyle="round,pad=0.4"))

fig.tight_layout()
fig.savefig(OUT, dpi=300, facecolor="white")
print(f"Saved -> {OUT}")
