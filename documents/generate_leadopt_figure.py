#!/usr/bin/env python3
"""
GENESIS — Phase 9b Lead Optimization Figure
Panel (a): ADMET flag distribution — Phase 8 candidates vs Phase 9b analogs
Panel (b): Parent ADMET score → mean analog ADMET score (one point per parent)
Output: documents/images/Fig_LeadOptimization.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

BASE    = os.path.dirname(os.path.abspath(__file__))
GENESIS = os.path.dirname(BASE)
OUT     = os.path.join(BASE, "images", "Fig_LeadOptimization.png")
os.makedirs(os.path.join(BASE, "images"), exist_ok=True)

# ── Colour palette (matches rest of manuscript figures) ──────────────────
ADMET_C = {"PASS": "#2ca02c", "CAUTION": "#ff7f0e", "FAIL": "#d62728"}
SRC_C   = {"repurposed": "#1f77b4", "generated": "#e45f3c"}

# ── Load data ────────────────────────────────────────────────────────────
phase8 = pd.read_csv(
    os.path.join(GENESIS, "admet", "results", "admet_profiles.tsv"),
    sep="\t", low_memory=False
)

analogs = pd.read_csv(
    os.path.join(GENESIS, "leadopt", "results", "analogs_all.tsv"),
    sep="\t", low_memory=False
)

master = pd.read_csv(
    os.path.join(GENESIS, "final_ranking", "results", "master_ranked_recommended.tsv"),
    sep="\t", low_memory=False
)

# ── ADMET distributions ──────────────────────────────────────────────────
flags  = ["PASS", "CAUTION", "FAIL"]
p8_tot = len(phase8)
p9_tot = len(analogs)

p8_n   = {f: int(phase8["admet_flag"].eq(f).sum()) for f in flags}
p9_n   = {f: int(analogs["admet_flag"].eq(f).sum()) for f in flags}
p8_pct = {f: p8_n[f] / p8_tot * 100 for f in flags}
p9_pct = {f: p9_n[f] / p9_tot * 100 for f in flags}

print("Phase 8  distribution:", {f: f"{p8_n[f]} ({p8_pct[f]:.1f}%)" for f in flags})
print("Phase 9b distribution:", {f: f"{p9_n[f]} ({p9_pct[f]:.1f}%)" for f in flags})

# ── Scatter: per-parent improvement ─────────────────────────────────────
parent_info = (
    master[["cpd_id", "admet_score", "source"]]
    .rename(columns={"cpd_id": "parent_id", "admet_score": "parent_admet"})
)

analog_grp = (
    analogs.groupby("parent_id")
    .agg(mean_analog_admet=("admet_score", "mean"),
         n_analogs=("admet_score", "count"))
    .reset_index()
)

scatter = analog_grp.merge(parent_info, on="parent_id", how="left")
scatter["src_group"] = scatter["source"].apply(
    lambda s: "repurposed" if str(s).lower().startswith("repur") else "generated"
)

# Per-analog Δ ADMET (for annotation — should match +0.276 from pipeline)
analogs_j = analogs.merge(
    master[["cpd_id", "admet_score"]].rename(
        columns={"cpd_id": "parent_id", "admet_score": "parent_admet"}
    ),
    on="parent_id", how="left"
)
mean_delta  = (analogs_j["admet_score"] - analogs_j["parent_admet"]).mean()
n_improved  = (analogs_j["admet_score"] > analogs_j["parent_admet"]).sum()

print(f"Scatter: {len(scatter)} parent points, mean Δ ADMET = {mean_delta:+.3f}")
print(f"Analogs with improvement: {n_improved}/{p9_tot}")

# ── Figure ───────────────────────────────────────────────────────────────
fig, (ax_a, ax_b) = plt.subplots(
    1, 2, figsize=(15, 6.5),
    gridspec_kw={"width_ratios": [1.0, 1.15]},
    facecolor="white"
)
fig.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.11, wspace=0.30)

# ── Panel (a) ─────────────────────────────────────────────────────────────
x     = np.arange(len(flags))
width = 0.35

bars_p8 = ax_a.bar(
    x - width / 2, [p8_pct[f] for f in flags], width,
    color=[ADMET_C[f] for f in flags], alpha=0.55,
    edgecolor="#555", linewidth=0.7, hatch="///",
    label=f"Phase 8 candidates (n={p8_tot})"
)
bars_p9 = ax_a.bar(
    x + width / 2, [p9_pct[f] for f in flags], width,
    color=[ADMET_C[f] for f in flags], alpha=0.92,
    edgecolor="#555", linewidth=0.7,
    label=f"Phase 9b analogs (n={p9_tot})"
)

# Annotate Phase 8 bar values
for bar, f in zip(bars_p8, flags):
    h = bar.get_height()
    if h > 0.5:
        ax_a.text(bar.get_x() + bar.get_width() / 2, h + 0.9,
                  f"{h:.1f}%", ha="center", va="bottom", fontsize=8, color="#444")

# Annotate Phase 9b bar values (bold for FAIL to highlight 0%)
for bar, f in zip(bars_p9, flags):
    h = bar.get_height()
    ax_a.text(bar.get_x() + bar.get_width() / 2, h + 0.9,
              f"{h:.1f}%" if h > 0.2 else "0%",
              ha="center", va="bottom", fontsize=8.5,
              fontweight="bold" if f == "FAIL" else "normal",
              color="#d62728" if f == "FAIL" else "#444")

ax_a.set_xticks(x)
ax_a.set_xticklabels(flags, fontsize=10.5)
ax_a.set_ylabel("Percentage of compounds (%)", fontsize=9.5)
ymax = max(p8_pct["PASS"], p9_pct["PASS"]) * 1.25
ax_a.set_ylim(0, ymax)
ax_a.set_title("(a) ADMET Safety Profile: Phase 8 vs Phase 9b", fontsize=10.5, pad=9)
ax_a.spines[["top", "right"]].set_visible(False)
ax_a.yaxis.grid(True, alpha=0.30, lw=0.6)
ax_a.set_axisbelow(True)

leg_p8 = mpatches.Patch(facecolor="#aaa", hatch="///", edgecolor="#555",
                         alpha=0.55, label=f"Phase 8 (n={p8_tot})")
leg_p9 = mpatches.Patch(facecolor="#aaa", edgecolor="#555",
                         alpha=0.92, label=f"Phase 9b analogs (n={p9_tot})")
ax_a.legend(handles=[leg_p8, leg_p9], fontsize=8.5, loc="upper right",
            frameon=True, framealpha=0.85)

# ── Panel (b): Dumbbell plot — parent → mean analog ADMET per parent ────
valid = scatter.dropna(subset=["parent_admet", "mean_analog_admet"]).copy()
valid = valid.sort_values("parent_admet").reset_index(drop=True)

def short_label(pid):
    if pid.startswith("GEN_"):
        return pid
    if pid.startswith("REP_CHEMBL"):
        return "REP_" + pid[10:17]
    return pid[:12]

valid["label"] = valid["parent_id"].apply(short_label)
y_pos = np.arange(len(valid))

# Connecting lines
for i, row in valid.iterrows():
    ax_b.hlines(i, row["parent_admet"], row["mean_analog_admet"],
                color="#ccc", lw=2.0, zorder=1)

# Parent dots (coloured by source)
for i, row in valid.iterrows():
    c = SRC_C.get(row["src_group"], "#888")
    ax_b.scatter(row["parent_admet"], i, s=110, c=c,
                 edgecolors="white", linewidths=0.8, zorder=3)

# Analog diamonds (green, sized by n_analogs) -- modest linear scaling so
# markers stay well within row spacing and never touch neighbouring rows
for i, row in valid.iterrows():
    size = 90 + int(row["n_analogs"]) * 2
    ax_b.scatter(row["mean_analog_admet"], i, s=size,
                 c="#2ca02c", marker="D",
                 edgecolors="white", linewidths=0.8, zorder=3)

# Δ and n annotation on each row
for i, row in valid.iterrows():
    delta = row["mean_analog_admet"] - row["parent_admet"]
    ax_b.text(row["mean_analog_admet"] + 0.007, i,
              f"Δ={delta:+.2f}  n={int(row['n_analogs'])}",
              va="center", ha="left", fontsize=7.5, color="#444")

ax_b.set_yticks(y_pos)
ax_b.set_yticklabels(valid["label"], fontsize=8.5)
ax_b.set_xlabel("ADMET Score", fontsize=9.5)
ax_b.set_ylim(-0.8, len(valid) - 0.2)
ax_b.set_xlim(valid["parent_admet"].min() - 0.05,
              valid["mean_analog_admet"].max() + 0.20)
ax_b.set_title(
    "(b) ADMET Improvement per Elite Parent\n"
    "(circle = parent; diamond = mean analog; size ∝ n analogs)",
    fontsize=10.5, pad=9
)
ax_b.spines[["top", "right"]].set_visible(False)
ax_b.xaxis.grid(True, alpha=0.25, lw=0.6)
ax_b.set_axisbelow(True)

ax_b.text(0.98, 0.96,
          f"Mean \u0394 ADMET = {mean_delta:+.3f}\n"
          f"{n_improved}/{p9_tot} analogs improved",
          transform=ax_b.transAxes, ha="right", va="top", fontsize=8.5,
          color="#2ca02c", fontweight="bold",
          bbox=dict(facecolor="white", alpha=0.90,
                    edgecolor="#2ca02c", boxstyle="round,pad=0.35"))

src_handles = [
    mpatches.Patch(color=SRC_C["repurposed"], label="Repurposed parent"),
    mpatches.Patch(color=SRC_C["generated"],  label="AI-generated parent"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor="#2ca02c",
           markersize=7, label="Mean analog ADMET"),
]
ax_b.legend(handles=src_handles, fontsize=8, loc="lower left",
            frameon=True, framealpha=0.85)

# ── Save ─────────────────────────────────────────────────────────────────
plt.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved → {OUT}")
