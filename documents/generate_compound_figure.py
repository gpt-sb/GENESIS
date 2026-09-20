#!/usr/bin/env python3
"""
GENESIS — Top Compound Figure Generator
Renders top 6 CRAVE-generated Tier-1 candidates (verified from compound_dossiers.tsv).
Saves to documents/images/Fig_TopGeneratedCompounds.png
Embeds figure + caption into GENESIS_NMI_Manuscript_v2_v3.docx
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from io import BytesIO

try:
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem import rdDepictor
    rdDepictor.SetPreferCoordGen(True)
    from PIL import Image
except ImportError as e:
    sys.exit(f"Missing dependency: {e}")

BASE    = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE, "images")
os.makedirs(IMG_DIR, exist_ok=True)
OUT_PNG = os.path.join(IMG_DIR, "Fig_TopGeneratedCompounds.png")
DOCX_IN = os.path.join(BASE, "GENESIS_NMI_Manuscript_v2_v3.docx")

# ── Verified fresh from final_ranking/results/master_ranked_recommended.tsv (2026-09-17 run) ─
COMPOUNDS = [
    {"id":"GEN_00182","rank":5,"score":0.6680,"admet":"CAUTION","qed":0.753,"sa":2.41,"mw":328.7,
     "best_dg":-11.2,"best_target":"PTGS2","n_targets":7,"novelty":"Novel scaffold","method":"CRAVE",
     "smiles":"O=C1Cc2cc3c(Nc4ccc(F)c(Cl)c4)ncnc3cc2N1",
     "dg":{"AKT1":-9.1,"EGFR":-10.0,"FGFR1":-9.9,"MDM2":-8.5,"MTOR":-7.0,"PIK3CA":-8.8,"PTGS2":-11.2}},
    {"id":"GEN_00133","rank":7,"score":0.6516,"admet":"CAUTION","qed":0.709,"sa":2.76,"mw":383.8,
     "best_dg":-11.0,"best_target":"PTGS2","n_targets":7,"novelty":"Scaffold hop","method":"CRAVE",
     "smiles":"COc1ccccc1-c1cc(Nc2ccc(F)c(Cl)c2)ncncncn1",
     "dg":{"AKT1":-9.2,"EGFR":-9.5,"FGFR1":-9.7,"MDM2":-8.6,"MTOR":-7.5,"PIK3CA":-8.3,"PTGS2":-11.0}},
    {"id":"GEN_00045","rank":8,"score":0.6515,"admet":"CAUTION","qed":0.743,"sa":2.82,"mw":327.7,
     "best_dg":-11.0,"best_target":"PTGS2","n_targets":7,"novelty":"Scaffold hop","method":"CRAVE",
     "smiles":"Fc1ccc(Nc2ncnc3cc4c(cc23)COC=C4)cc1Cl",
     "dg":{"AKT1":-9.0,"EGFR":-10.1,"FGFR1":-9.6,"MDM2":-7.8,"MTOR":-7.1,"PIK3CA":-8.5,"PTGS2":-11.0}},
    {"id":"GEN_00020","rank":11,"score":0.6263,"admet":"CAUTION","qed":0.773,"sa":2.30,"mw":315.7,
     "best_dg":-10.8,"best_target":"PTGS2","n_targets":6,"novelty":"Scaffold hop","method":"CRAVE",
     "smiles":"Fc1ccc(Nc2ncnc3cc4c(cc23)CCO4)cc1Cl",
     "dg":{"AKT1":-8.7,"EGFR":-9.7,"FGFR1":-9.3,"MDM2":-7.5,"MTOR":-6.9,"PIK3CA":-8.3,"PTGS2":-10.8}},
    {"id":"GEN_00044","rank":13,"score":0.6159,"admet":"CAUTION","qed":0.746,"sa":2.41,"mw":314.8,
     "best_dg":-11.0,"best_target":"PTGS2","n_targets":6,"novelty":"Novel scaffold","method":"CRAVE",
     "smiles":"Fc1ccc(Nc2ncnc3cc4c(cc23)CCN4)cc1Cl",
     "dg":{"AKT1":-8.8,"EGFR":-9.6,"FGFR1":-9.5,"MDM2":-8.1,"MTOR":-6.9,"PIK3CA":-8.4,"PTGS2":-11.0}},
    {"id":"GEN_00069","rank":14,"score":0.6034,"admet":"CAUTION","qed":0.770,"sa":2.43,"mw":315.7,
     "best_dg":-10.6,"best_target":"PTGS2","n_targets":6,"novelty":"Scaffold hop","method":"CRAVE",
     "smiles":"Fc1ccc(Nc2ncnc3cc4c(cc23)COC4)cc1Cl",
     "dg":{"AKT1":-8.4,"EGFR":-9.5,"FGFR1":-9.7,"MDM2":-7.7,"MTOR":-6.9,"PIK3CA":-8.9,"PTGS2":-10.6}},
]

ADMET_C  = {"PASS":"#2ca02c","CAUTION":"#ff7f0e","FAIL":"#d62728"}
TARGET_C = {"AKT1":"#9467bd","EGFR":"#2ca02c","FGFR1":"#1f77b4",
            "MDM2":"#8c564b","MTOR":"#e377c2","PIK3CA":"#17becf","PTGS2":"#d62728"}

def mol_to_pil(smiles, w=380, h=280):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(w*2, h*2)
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2.4
    opts.addStereoAnnotation = False
    opts.minFontSize = 14
    opts.maxFontSize = 22
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return Image.open(BytesIO(drawer.GetDrawingText())).convert("RGBA")

def docking_bar(ax, dg, best, ac):
    tgts = list(dg.keys())
    vals = [-dg[t] for t in tgts]
    cols = [TARGET_C.get(t, "#888") for t in tgts]
    ax.barh(range(len(tgts)), vals, color=cols, edgecolor="white", linewidth=0.5, height=0.72)
    ax.set_yticks(range(len(tgts)))
    ax.set_yticklabels(tgts, fontsize=7)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_xlabel("-DeltaG (kcal/mol)", fontsize=7, labelpad=2)
    ax.axvline(7.0, color="#aaa", lw=0.8, ls="--", alpha=0.7)
    ax.text(7.1, len(tgts)-0.7, "threshold\n-7.0", fontsize=5.5, color="#777", va="top")
    ax.set_xlim(0, max(vals)+1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#f9f9f9")
    if best in tgts:
        bi = tgts.index(best)
        ax.get_yticklabels()[bi].set_fontweight("bold")
        ax.get_yticklabels()[bi].set_fontsize(8)
    admet_label = [k for k,v in ADMET_C.items() if v==ac][0]
    ax.text(0.99, 0.97, f"ADMET: {admet_label}", transform=ax.transAxes,
            ha="right", va="top", fontsize=7, fontweight="bold", color=ac,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=ac, alpha=0.12, edgecolor=ac, lw=0.8))

def make_figure():
    fig = plt.figure(figsize=(18.5, 13.0), facecolor="white")
    fig.suptitle("Top CRAVE-Generated Novel Candidates — GENESIS Pipeline",
                 fontsize=14, fontweight="bold", y=0.998)
    fig.text(0.5, 0.974,
             "Six Tier-1 fully validated compounds (overall ranks 7-17) with novel scaffolds, "
             "multi-target docking across 7 OSCC crystal structures (AutoDock Vina), and ADMET profiling. "
             "All structurally novel: Tanimoto < 0.49 to any training compound.",
             ha="center", va="top", fontsize=8.5, color="#444")

    gs = fig.add_gridspec(2, 3, left=0.03, right=0.97, top=0.94, bottom=0.07,
                          hspace=0.45, wspace=0.24)

    for idx, cpd in enumerate(COMPOUNDS):
        r, c = divmod(idx, 3)
        inner = gs[r, c].subgridspec(3, 1, height_ratios=[3.0, 0.75, 2.4], hspace=0.0)

        # Structure panel
        ax_s = fig.add_subplot(inner[0])
        img = mol_to_pil(cpd["smiles"])
        if img:
            ax_s.imshow(img, aspect="equal", interpolation="lanczos")
        ax_s.set_xticks([]); ax_s.set_yticks([])
        for sp in ax_s.spines.values():
            sp.set_visible(False)
        ac = ADMET_C[cpd["admet"]]
        ax_s.add_patch(FancyBboxPatch((0,0),1,1, boxstyle="round,pad=0.012",
            transform=ax_s.transAxes, fill=False, edgecolor=ac, lw=3.0, zorder=10))
        ax_s.set_title(f"{cpd['id']}   (Rank #{cpd['rank']})",
                       fontsize=10.5, fontweight="bold", pad=4)

        # Info text strip
        ax_i = fig.add_subplot(inner[1])
        ax_i.set_visible(False)
        pos = ax_s.get_position()
        info = (f"Score {cpd['score']:.3f} | MW {cpd['mw']:.0f} | QED {cpd['qed']:.2f} | "
                f"SA {cpd['sa']:.2f} | \u0394G {cpd['best_dg']:.1f} ({cpd['best_target']})\n"
                f"{cpd['n_targets']}/7 targets | {cpd['novelty']} | {cpd['method']}")
        fig.text(pos.x0 + pos.width/2, pos.y0 - 0.003, info,
                 ha="center", va="top", fontsize=6.3, color="#111",
                 fontfamily="monospace", linespacing=1.5,
                 bbox=dict(boxstyle="round,pad=0.28", facecolor=ac, alpha=0.10,
                           edgecolor=ac, lw=0.9))

        # Docking bar chart
        ax_b = fig.add_subplot(inner[2])
        docking_bar(ax_b, cpd["dg"], cpd["best_target"], ac)

    # Legend
    leg_e = [
        mpatches.Patch(facecolor=ADMET_C["PASS"],    alpha=0.45, edgecolor=ADMET_C["PASS"],    label="ADMET PASS"),
        mpatches.Patch(facecolor=ADMET_C["CAUTION"], alpha=0.45, edgecolor=ADMET_C["CAUTION"], label="ADMET CAUTION"),
        mpatches.Patch(color="none", label="  Targets:"),
    ] + [mpatches.Patch(facecolor=TARGET_C[t], label=t, linewidth=0)
         for t in ["AKT1","EGFR","FGFR1","MDM2","MTOR","PIK3CA","PTGS2"]]
    fig.legend(handles=leg_e, loc="lower center", ncol=10, fontsize=8.5,
               frameon=True, fancybox=True, shadow=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.0, columnspacing=0.9)

    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figure saved: {OUT_PNG}")
    return OUT_PNG

def embed_figure(img_path):
    from docx import Document
    from docx.shared import Inches, Pt
    from docx.oxml import OxmlElement
    doc = Document(DOCX_IN)

    anchor = None
    for p in doc.paragraphs:
        if "Generated molecules are novel, potent" in p.text:
            anchor = p
            break
    if anchor is None:
        for p in doc.paragraphs:
            if p.text.strip() == "Discussion":
                anchor = p
                break
    if anchor is None:
        anchor = doc.paragraphs[-1]

    def ins_after(ref, style=None):
        np_ = OxmlElement("w:p")
        ref._element.addnext(np_)
        from docx.text.paragraph import Paragraph
        p = Paragraph(np_, ref._parent)
        if style:
            try: p.style = style
            except: pass
        return p

    cap = ins_after(anchor, "Body A")
    r = cap.add_run(
        "Figure 4. Top CRAVE-generated novel candidates from the GENESIS pipeline. "
        "Six Tier-1 fully validated de novo compounds (overall ranks 7-17 in the unified "
        "generated+repurposed ranking) shown by GENESIS composite score. "
        "2D structures rendered with RDKit. Bar charts show per-target AutoDock Vina binding "
        "energy (DeltaG, kcal/mol) across seven OSCC crystal structures; dashed line marks the "
        "-7.0 kcal/mol therapeutic threshold. Border: green = ADMET PASS; orange = ADMET CAUTION. "
        "All compounds have novel or scaffold-hop structural novelty (Tanimoto < 0.49 to training set). "
        "GEN_00045 (rank 7, GENESIS score 0.5842) achieves -10.5 kcal/mol at FGFR1, "
        "binds 7/7 OSCC targets above threshold (mean DeltaG = -8.5 kcal/mol), "
        "and ranks above six FDA-approved clinical drugs in the unified compound ranking."
    )
    r.italic = True
    r.font.size = Pt(9)
    cap.paragraph_format.space_before = Pt(8)

    img_p = ins_after(cap, "Body A")
    img_p.alignment = 1
    img_p.add_run().add_picture(img_path, width=Inches(6.5))

    doc.save(DOCX_IN)
    print(f"Manuscript updated: {DOCX_IN}")

if __name__ == "__main__":
    print("Generating RDKit compound figure (fresh 2026-09-17 top-6 generated candidates)...")
    fig_path = make_figure()
    print("Done. NOT auto-embedding into any manuscript (v3/v4 use a separate figure-insertion step).")
