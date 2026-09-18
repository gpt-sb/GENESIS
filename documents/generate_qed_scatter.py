#!/usr/bin/env python3
"""
Panel (a): VP-centroid cosine similarity vs QED scatter (345 final candidates)
Panel (b): Existing Fig6_qed_vs_composite.png
Output: documents/images/Fig_QED_Elite_Combined.png
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image

BASE    = os.path.dirname(os.path.abspath(__file__))
GENESIS = os.path.dirname(BASE)
OUT     = os.path.join(BASE, "images", "Fig_QED_Elite_Combined.png")
os.makedirs(os.path.join(BASE, "images"), exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(
    os.path.join(GENESIS, "final_ranking/results/master_ranked_recommended.tsv"),
    sep="\t", low_memory=False
)

ADMET_C = {"PASS": "#2ca02c", "CAUTION": "#ff7f0e", "FAIL": "#d62728"}
ORDER   = ["FAIL", "CAUTION", "PASS"]   # draw PASS on top

# ── Panel (a) ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 7.5), facecolor="white")
ax_a = fig.add_axes([0.05, 0.12, 0.40, 0.76])   # left panel

for flag in ORDER:
    sub = df[df["admet_flag"] == flag]
    ax_a.scatter(
        sub["vp_similarity"], sub["qed"],
        c=ADMET_C.get(flag, "#888"),
        s=40, alpha=0.72, edgecolors="white", linewidths=0.3,
        label=f"{flag} (n={len(sub)})", zorder=ORDER.index(flag)+2
    )

# Threshold lines
ax_a.axvline(0.95, color="#333", lw=1.4, ls="--", alpha=0.7)
ax_a.axhline(0.60, color="#333", lw=1.4, ls="--", alpha=0.7)

# Elite zone annotation
elite = df[(df["vp_similarity"] > 0.95) & (df["admet_flag"] == "PASS")]
ax_a.add_patch(plt.Rectangle(
    (0.95, 0.60), 0.05, 0.40,
    linewidth=1.5, edgecolor="#2ca02c", facecolor="#2ca02c", alpha=0.08, zorder=1
))
ax_a.text(0.952, 0.615,
    f"Elite zone\nn={len(elite)}\nmean QED={elite['qed'].mean():.3f}",
    fontsize=8.5, color="#1a6b1a", va="bottom", ha="left",
    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2))

# Threshold labels
ax_a.text(0.951, 0.02, "VP = 0.95", fontsize=8, color="#333", rotation=90, va="bottom")
ax_a.text(0.02,  0.61, "QED = 0.60", fontsize=8, color="#333", va="bottom")

ax_a.set_xlabel("PGCL VP-centroid cosine similarity", fontsize=12)
ax_a.set_ylabel("QED (Quantitative Estimate of Drug-likeness)", fontsize=12)
ax_a.set_xlim(0.82, 1.005)
ax_a.set_ylim(-0.02, 1.02)
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)
ax_a.tick_params(labelsize=10)

# Legend
handles = [
    mpatches.Patch(color=ADMET_C["PASS"],    label=f"ADMET PASS  (n={len(df[df['admet_flag']=='PASS'])})"),
    mpatches.Patch(color=ADMET_C["CAUTION"], label=f"ADMET CAUTION (n={len(df[df['admet_flag']=='CAUTION'])})"),
    Line2D([0],[0], color="#333", lw=1.4, ls="--", label="VP=0.95 / QED=0.60 thresholds"),
]
ax_a.legend(handles=handles, fontsize=9, framealpha=0.9, loc="lower left")

# Panel label
ax_a.text(-0.04, 1.04, "(a)", transform=ax_a.transAxes,
          fontsize=16, fontweight="bold", color="#1C2833")
ax_a.set_title(
    "VP-centroid cosine similarity vs QED\n(316 GENESIS-ranked compounds; "
    f"{len(df[df['source']=='generated'])} generated + {len(df[df['source']=='repurposed'])} repurposed)",
    fontsize=10.5, pad=8
)

# ── Panel (b): QED vs composite score (regenerated fresh, no stale dependency) ──
ax_b = fig.add_axes([0.55, 0.12, 0.42, 0.76])
for flag in ORDER:
    sub = df[df["admet_flag"] == flag]
    ax_b.scatter(
        sub["composite_score"], sub["qed"],
        c=ADMET_C.get(flag, "#888"),
        s=40, alpha=0.72, edgecolors="white", linewidths=0.3,
        zorder=ORDER.index(flag) + 2
    )
ax_b.set_xlabel("GENESIS composite score", fontsize=12)
ax_b.set_ylabel("QED", fontsize=12)
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)
ax_b.tick_params(labelsize=10)
ax_b.text(-0.04, 1.04, "(b)", transform=ax_b.transAxes,
          fontsize=16, fontweight="bold", color="#1C2833")

# ── Divider ───────────────────────────────────────────────────────────────
fig.add_artist(plt.Line2D([0.495, 0.495], [0.05, 0.95],
    transform=fig.transFigure, color="#ddd", lw=0.8))

# ── Shared title ──────────────────────────────────────────────────────────
fig.suptitle(
    "Drug-likeness and composite score distribution of GENESIS-ranked candidates",
    fontsize=13, fontweight="bold", y=0.99
)

plt.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

from PIL import Image as PILImage
final = PILImage.open(OUT)
print(f"Saved: {OUT}")
print(f"Size: {final.size[0]}x{final.size[1]} px")
import os as _os
print(f"File: {_os.path.getsize(OUT)/1024:.0f} KB")
