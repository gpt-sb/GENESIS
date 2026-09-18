#!/usr/bin/env python3
"""
GENESIS Pipeline Figure — clean, impactful, minimal clutter
Boustrophedon: Row 1 L→R (Ph0–Ph6), Row 2 R→L (Ph7–Ph10)
Each box: Phase number badge + Phase name (large) + 1 key output
PGCL Phase 2 highlighted as the central unifying innovation.
"""
import matplotlib, os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

BASE    = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(BASE, "images", "Fig1_Pipeline_Overview_v2.png")
os.makedirs(os.path.join(BASE, "images"), exist_ok=True)

# ── Layout ───────────────────────────────────────────────────────────────
FW, FH  = 24.0, 11.2
ROW1_Y  = 7.90
ROW2_Y  = 3.40
MARGIN  = 0.30
BH      = 2.65     # box height
N1, N2  = 7, 4
GAP1    = 0.34
GAP2    = 0.78

span    = FW - 2*MARGIN
BW1     = (span - (N1-1)*GAP1) / N1   # ≈3.11"
BW2     = (span - (N2-1)*GAP2) / N2   # ≈5.30"

R1_XS = [MARGIN + i*(BW1+GAP1) for i in range(N1)]
R2_XS = list(reversed([MARGIN + i*(BW2+GAP2) for i in range(N2)]))
# R2_XS[0] = Phase 7 (rightmost), R2_XS[3] = Phase 10 (leftmost)

AC = "#444444"

# ── Phase data (1 key output each; lean and impactful) ───────────────────
R1_PHASES = [
    {"num":"0","name":"Molecular\nEmbedding",
     "tag":"768-d CLS · 715K compounds",
     "key":"MolDeBERTa 123M · 263 cpds/sec · 2.1 h",
     "color":"#5C4FC7","tc":"white","icon":"⬢"},
    {"num":"1","name":"Data Prep &\nNeg Mining",
     "tag":"Six-tier neg taxonomy",
     "key":"49,312 pos · 643K neg · 6 tiers",
     "color":"#7B2D8B","tc":"white","icon":"◼"},
    {"num":"2","name":"PGCL\nTraining",
     "tag":"★  Core Innovation  ★",
     "key":"768→128-d · ρ=0.555 · AUC=0.967",
     "color":"#154360","tc":"white","icon":"★","star":True},
    {"num":"3","name":"Generative\nDesign",
     "tag":"6 architectures · 5 seeds",
     "key":"5,416 unique mols · CRAVE 0.771±0.033",
     "color":"#0B6B5A","tc":"white","icon":"⬡"},
    {"num":"4/5","name":"Profiling &\nValidation",
     "tag":"Bioactivity · Ablation · Reproducibility",
     "key":"8 ablations · 3-seed PGCL · 5-seed generative",
     "color":"#0E8A6D","tc":"white","icon":"✔"},
    {"num":"6","name":"Drug\nRepurposing",
     "tag":"CompoundList · 23,308 candidates",
     "key":"Erlotinib HCl top hit · RW=0.872",
     "color":"#B03A0A","tc":"white","icon":"💊"},
]

R2_PHASES = [   # [0]=Ph7(rightmost)…[3]=Ph10(leftmost)
    {"num":"7","name":"Molecular\nDocking",
     "tag":"AutoDock Vina · 8 OSCC structures",
     "key":"77.7% ≤−7.0 kcal/mol · best −11.2 FGFR1",
     "color":"#1B4F72","tc":"white","icon":"🔩"},
    {"num":"8","name":"ADMET\nProfiling",
     "tag":"Novel oral mucosal safety filter",
     "key":"91.4% Ro5 · 62.9% mucosal SAFE · 13 elite",
     "color":"#1E6E3B","tc":"white","icon":"🛡"},
    {"num":"9a/b","name":"Network Pharm\n& Lead Opt",
     "tag":"Pathway analysis · Scaffold optimisation",
     "key":"PI3K-AKT top pathway · 890 analogs · 2.8% FAIL",
     "color":"#0F7B68","tc":"white","icon":"🔬"},
    {"num":"10","name":"Final\nRanking",
     "tag":"8-component GENESIS score",
     "key":"316 recs · 46 Tier-1 · 13 elite",
     "color":"#1C2833","tc":"white","icon":"🏆"},
]

# ── Box drawing (minimal: badge + icon + name + tag + key metric) ─────────
def draw_box(ax, cx, cy, bw, bh, ph):
    ec = "#F4D03F" if ph.get("star") else "none"
    lw = 3.5      if ph.get("star") else 0
    # box
    ax.add_patch(FancyBboxPatch((cx-bw/2, cy-bh/2), bw, bh,
        boxstyle="round,pad=0.18", fc=ph["color"], ec=ec, lw=lw, zorder=2))
    tc = ph["tc"]
    y_top = cy + bh/2

    # Phase badge (top-left)
    badge_r = 0.23
    ax.add_patch(plt.Circle((cx-bw/2+0.32, y_top-0.28), badge_r,
        fc="white", ec="none", alpha=0.25, zorder=3))
    ax.text(cx-bw/2+0.32, y_top-0.28, ph["num"],
            ha="center", va="center", fontsize=8.5, fontweight="bold",
            color="white", zorder=4)

    # Phase name (large, centered top)
    ax.text(cx, cy+0.56, ph["name"],
            ha="center", va="center", fontsize=12.5, fontweight="bold",
            color=tc, zorder=3, linespacing=1.20)

    # Tag (italic, medium)
    ax.text(cx, cy-0.22, ph["tag"],
            ha="center", va="center", fontsize=8.8, color=tc,
            alpha=0.85, zorder=3, fontstyle="italic")

    # Key metric (small, bottom strip)
    strip_y = cy - bh/2 + 0.32
    ax.add_patch(FancyBboxPatch((cx-bw/2+0.06, strip_y-0.24),
        bw-0.12, 0.48, boxstyle="round,pad=0.06",
        fc="black", ec="none", alpha=0.22, zorder=3))
    ax.text(cx, strip_y, ph["key"],
            ha="center", va="center", fontsize=8.3, color=tc,
            alpha=0.96, zorder=4, fontweight="bold")

def harrow(ax, x1, y, x2, label="", above=True):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
        arrowprops=dict(arrowstyle="-|>", color=AC, lw=1.8, mutation_scale=15))
    if label:
        yl = y+BH/2+0.36 if above else y-BH/2-0.32
        va = "bottom" if above else "top"
        ax.text((x1+x2)/2, yl, label, ha="center", va=va,
                fontsize=8.2, color="#777", fontstyle="italic")

def varrow(ax, x, y1, y2):
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
        arrowprops=dict(arrowstyle="-|>", color=AC, lw=2.2, mutation_scale=16))

# ── Build figure ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FW, FH), facecolor="white")
ax.set_xlim(-0.5, FW-0.3)
ax.set_ylim(0.1, FH)
ax.set_aspect("equal")
ax.axis("off")

# Title bar
ax.add_patch(FancyBboxPatch((-0.1, FH-1.15), FW-0.3, 1.0,
    boxstyle="round,pad=0.05", fc="#1A1A2E", ec="none", zorder=0))
ax.text(FW/2-0.4, FH-0.45,
        "GENESIS  ·  Generative Exploration of Novel Entities for Selective Inhibitor Synthesis",
        ha="center", va="center", fontsize=15, fontweight="bold", color="white", zorder=1)
ax.text(FW/2-0.4, FH-0.90,
        "End-to-end AI drug discovery pipeline for OSCC  ·  Phases 0 – 10  ·"
        "  MolDeBERTa → PGCL → 6 Generative Methods → Docking → ADMET → 316 Recommendations",
        ha="center", va="center", fontsize=9.5, color="#AAB7C4", fontstyle="italic", zorder=1)

# Row 1 boxes
for i, ph in enumerate(R1_PHASES):
    cx = R1_XS[i] + BW1/2
    draw_box(ax, cx, ROW1_Y, BW1, BH, ph)

# Row 1 arrows
lbl1 = ["768-d","neg pool","128-d","molecules","candidates",""]
for i in range(len(R1_PHASES)-1):  # 5 arrows between 6 phases (no stray)
    harrow(ax, R1_XS[i]+BW1, ROW1_Y, R1_XS[i+1], label=lbl1[i], above=True)

# Vertical turn Ph6 → Ph7
# Ph6→Ph7: smooth arc connector (boustrophedon turn)
# Phase 6 bottom-center → Phase 7 top-center via arc curving right
x_p6_bc = R1_XS[5] + BW1/2          # Ph6 bottom center
x_p7_tc = R2_XS[0] + BW2/2          # Ph7 top center
ax.annotate('',
    xy     = (x_p7_tc, ROW2_Y + BH/2 + 0.06),
    xytext = (x_p6_bc, ROW1_Y - BH/2 - 0.06),
    arrowprops=dict(
        arrowstyle    = '-|>',
        color         = AC,
        lw            = 2.4,
        mutation_scale= 18,
        connectionstyle='arc3,rad=-0.35'
    ), zorder=5)

# Row 2 boxes (Ph7 rightmost … Ph10 leftmost)
for i, ph in enumerate(R2_PHASES):
    cx = R2_XS[i] + BW2/2
    draw_box(ax, cx, ROW2_Y, BW2, BH, ph)

# Row 2 arrows (right → left)
lbl2 = ["docked hits","ADMET-filtered","analogs + NP"]
for i in range(len(R2_PHASES)-1):
    x1, x2 = R2_XS[i], R2_XS[i+1]+BW2
    harrow(ax, x1, ROW2_Y, x2, label=lbl2[i], above=False)

# ── PGCL unified-space strip ──────────────────────────────────────────────
ann_cx  = FW/2 - 0.3
ann_cy  = (ROW1_Y + ROW2_Y) / 2 + 0.04
ann_w, ann_h = 11.2, 0.94

ax.add_patch(FancyBboxPatch((ann_cx-ann_w/2, ann_cy-ann_h/2), ann_w, ann_h,
    boxstyle="round,pad=0.10", fc="#EBF5FB", ec="#154360", lw=2.0, zorder=4))

uses = [("①","RL reward","cosine(z, c_VP)","#E59866"),
        ("②","Conditional input","c_VP → decoder","#76D7C4"),
        ("③","Retrieval metric","k-NN similarity","#A569BD")]

for j, (num, role, desc, col) in enumerate(uses):
    ux = ann_cx - 3.4 + j*3.4
    ax.add_patch(plt.Circle((ux-1.20, ann_cy), 0.21,
        fc=col, ec="none", alpha=0.90, zorder=5))
    ax.text(ux-1.20, ann_cy, num, ha="center", va="center",
            fontsize=9, fontweight="bold", color="white", zorder=6)
    ax.text(ux-0.82, ann_cy+0.17, role,
            ha="left", va="center", fontsize=9.5, fontweight="bold",
            color="#1A5276", zorder=5)
    ax.text(ux-0.82, ann_cy-0.18, desc,
            ha="left", va="center", fontsize=8.5, color="#5D6D7E",
            fontstyle="italic", zorder=5)

ax.text(ann_cx+ann_w/2-0.30, ann_cy,
        "PGCL 128-d\nfrozen space",
        ha="right", va="center", fontsize=9.0, fontweight="bold",
        color="#154360", zorder=5)

# Dotted connector from Ph2 bottom to annotation
pgcl_cx = R1_XS[2] + BW1/2
ax.plot([pgcl_cx, ann_cx-ann_w/2+0.4],
        [ROW1_Y-BH/2, ann_cy],
        color="#2471A3", lw=1.2, ls="dotted", zorder=4)

# ── Phase-type legend ─────────────────────────────────────────────────────
legend_items = [
    ("#5C4FC7","Data / Embedding"),
    ("#154360","Representation (PGCL ★)"),
    ("#0B6B5A","Generative"),
    ("#B03A0A","Repurposing"),
    ("#1B4F72","Docking / ADMET"),
    ("#1C2833","Final Ranking"),
]
lx0 = 0.3
for i, (col, lbl) in enumerate(legend_items):
    lx = lx0 + i*3.90
    ax.add_patch(FancyBboxPatch((lx, 0.54), 0.30, 0.28,
        boxstyle="round,pad=0.04", fc=col, ec="none", zorder=3))
    ax.text(lx+0.42, 0.67, lbl, ha="left", va="center",
            fontsize=8.5, color="#333", zorder=3)

# Hardware + summary line
ax.text(FW/2-0.3, 0.22,
        "Hardware: Apple M3 Pro · 16-core · 128 GB unified memory · "
        "PyTorch 2.4 MPS (float32) · Total runtime: 8.6 h",
        ha="center", va="center", fontsize=9.0, color="#666", fontstyle="italic")

plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {OUT_PNG}")
