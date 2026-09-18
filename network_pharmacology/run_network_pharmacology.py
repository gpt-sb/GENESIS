#!/usr/bin/env python3
"""
GENESIS — Phase 9a: Network Pharmacology & Pathway Analysis
=============================================================
Build compound–target–pathway networks from multi-target docking data.
Compute network centrality, identify key polypharmacology hubs, map
targets to OSCC-relevant signalling pathways, and produce publication-
ready heatmaps and network statistics.

Modules:
  A. Data Loading — selectivity profiles + ADMET elite list
  B. Bipartite Compound–Target Network — weighted by binding affinity
  C. Target–Target Co-Binding Network — shared binder analysis
  D. OSCC Pathway Mapping — targets → KEGG/Reactome pathways (curated)
  E. Compound Selectivity Heatmap — per-target energy matrix
  F. Network Metrics & Hub Identification
  G. Pathway Enrichment Analysis — which pathways are multi-hit
  H. Visualization — heatmaps, network layouts, radar charts

Dependencies: pip install networkx matplotlib seaborn pandas numpy scipy

Input:
    - GENESIS/docking/results/selectivity_profiles.tsv
    - GENESIS/docking/results/polypharmacology_network.tsv
    - GENESIS/admet/results/final_candidates.tsv
    - GENESIS/admet/results/admet_profiles.tsv

Output:
    - GENESIS/network_pharmacology/results/*.tsv   (data tables)
    - GENESIS/network_pharmacology/results/*.png   (figures)
    - GENESIS/network_pharmacology/results/*.json  (summary)
"""

import os, json, time, math
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# ── Paths ──────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
NP_DIR      = ROOT / "network_pharmacology"
RESULTS_DIR = NP_DIR / "results"
DOCK_DIR    = ROOT / "docking" / "results"
ADMET_DIR   = ROOT / "admet" / "results"

# ══════════════════════════════════════════════════════════════════════
# Curated OSCC Target–Pathway Knowledge Base
# ══════════════════════════════════════════════════════════════════════

TARGET_INFO = {
    "AKT1":   {"full_name": "RAC-alpha serine/threonine-protein kinase",
               "gene": "AKT1", "uniprot": "P31749",
               "pathways": ["PI3K-AKT signalling (hsa04151)",
                            "mTOR signalling (hsa04150)",
                            "Apoptosis (hsa04210)",
                            "FOXO signalling (hsa04068)"],
               "cancer_role": "Survival/Proliferation",
               "oscc_relevance": "Constitutively active in 50-70% OSCC; promotes cell survival and chemo-resistance"},

    "EGFR":   {"full_name": "Epidermal Growth Factor Receptor",
               "gene": "EGFR", "uniprot": "P00533",
               "pathways": ["ErbB signalling (hsa04012)",
                            "MAPK signalling (hsa04010)",
                            "PI3K-AKT signalling (hsa04151)",
                            "Ras signalling (hsa04014)"],
               "cancer_role": "Proliferation/Migration",
               "oscc_relevance": "Overexpressed in 80-90% OSCC; cetuximab is FDA-approved for HNSCC"},

    "FGFR1":  {"full_name": "Fibroblast Growth Factor Receptor 1",
               "gene": "FGFR1", "uniprot": "P11362",
               "pathways": ["MAPK signalling (hsa04010)",
                            "PI3K-AKT signalling (hsa04151)",
                            "Ras signalling (hsa04014)",
                            "Regulation of actin cytoskeleton (hsa04810)"],
               "cancer_role": "Proliferation/Angiogenesis",
               "oscc_relevance": "Amplified in 10-17% OSCC; correlates with lymph node metastasis"},

    "MDM2":   {"full_name": "E3 ubiquitin-protein ligase Mdm2",
               "gene": "MDM2", "uniprot": "Q00987",
               "pathways": ["p53 signalling (hsa04115)",
                            "Cell cycle (hsa04110)",
                            "Apoptosis (hsa04210)"],
               "cancer_role": "Tumour Suppressor Evasion",
               "oscc_relevance": "Overexpressed in 40-60% OSCC; degrades p53 tumour suppressor"},

    "MTOR":   {"full_name": "Serine/threonine-protein kinase mTOR",
               "gene": "MTOR", "uniprot": "P42345",
               "pathways": ["mTOR signalling (hsa04150)",
                            "PI3K-AKT signalling (hsa04151)",
                            "Autophagy (hsa04140)",
                            "AMPK signalling (hsa04152)"],
               "cancer_role": "Metabolism/Growth",
               "oscc_relevance": "Hyperactivated via PI3K/AKT; drives metabolic reprogramming in OSCC"},

    "PIK3CA": {"full_name": "PI3-kinase catalytic subunit alpha",
               "gene": "PIK3CA", "uniprot": "P42336",
               "pathways": ["PI3K-AKT signalling (hsa04151)",
                            "mTOR signalling (hsa04150)",
                            "FOXO signalling (hsa04068)",
                            "Apoptosis (hsa04210)"],
               "cancer_role": "Survival/Proliferation",
               "oscc_relevance": "Mutated in 20-30% OSCC (hotspot E545K, H1047R); activates AKT cascade"},

    "PTGS2":  {"full_name": "Prostaglandin-endoperoxide synthase 2 (COX-2)",
               "gene": "PTGS2", "uniprot": "P35354",
               "pathways": ["Arachidonic acid metabolism (hsa00590)",
                            "NF-kappa B signalling (hsa04064)",
                            "VEGF signalling (hsa04370)",
                            "TNF signalling (hsa04668)"],
               "cancer_role": "Inflammation/Angiogenesis",
               "oscc_relevance": "Overexpressed in 80%+ OSCC; promotes angiogenesis and immune evasion"},
}

# Pathway → category mapping for enrichment
PATHWAY_CATEGORIES = {
    "PI3K-AKT signalling (hsa04151)": "PI3K/AKT/mTOR Axis",
    "mTOR signalling (hsa04150)":     "PI3K/AKT/mTOR Axis",
    "FOXO signalling (hsa04068)":     "PI3K/AKT/mTOR Axis",
    "AMPK signalling (hsa04152)":     "PI3K/AKT/mTOR Axis",
    "Autophagy (hsa04140)":           "PI3K/AKT/mTOR Axis",
    "MAPK signalling (hsa04010)":     "RTK/MAPK Cascade",
    "Ras signalling (hsa04014)":      "RTK/MAPK Cascade",
    "ErbB signalling (hsa04012)":     "RTK/MAPK Cascade",
    "Regulation of actin cytoskeleton (hsa04810)": "RTK/MAPK Cascade",
    "p53 signalling (hsa04115)":      "Cell Cycle/Apoptosis",
    "Cell cycle (hsa04110)":          "Cell Cycle/Apoptosis",
    "Apoptosis (hsa04210)":           "Cell Cycle/Apoptosis",
    "Arachidonic acid metabolism (hsa00590)": "Inflammation/TME",
    "NF-kappa B signalling (hsa04064)":       "Inflammation/TME",
    "VEGF signalling (hsa04370)":             "Inflammation/TME",
    "TNF signalling (hsa04668)":              "Inflammation/TME",
}


# ══════════════════════════════════════════════════════════════════════
# Module A: Data Loading
# ══════════════════════════════════════════════════════════════════════

def load_data():
    print("\n" + "=" * 70)
    print("MODULE A: Data Loading")
    print("=" * 70)

    profiles = pd.read_csv(DOCK_DIR / "selectivity_profiles.tsv", sep="\t")
    print(f"  Selectivity profiles: {len(profiles)} compounds × 7 targets")

    polynet = pd.read_csv(DOCK_DIR / "polypharmacology_network.tsv", sep="\t")
    print(f"  Polypharmacology network: {len(polynet)} target pairs")

    admet = pd.read_csv(ADMET_DIR / "admet_profiles.tsv", sep="\t")
    print(f"  ADMET profiles: {len(admet)} compounds")

    # Identify elite compounds
    candidates = pd.read_csv(ADMET_DIR / "final_candidates.tsv", sep="\t")
    elite = candidates[
        (candidates["in_phase7b"] == True) & (candidates["admet_flag"] == "PASS")
    ].sort_values("final_score", ascending=False)
    print(f"  Triple-validated elite: {len(elite)} compounds")

    # Extract docking energy columns for the 50 multi-target compounds
    target_cols = [c for c in profiles.columns if c.startswith("dG_")]
    targets = [c.replace("dG_", "") for c in target_cols]
    print(f"  Targets: {', '.join(targets)}")

    return profiles, polynet, admet, elite, target_cols, targets


# ══════════════════════════════════════════════════════════════════════
# Module B: Bipartite Compound–Target Network
# ══════════════════════════════════════════════════════════════════════

def build_bipartite_network(profiles, target_cols, targets, elite):
    print("\n" + "=" * 70)
    print("MODULE B: Bipartite Compound–Target Network")
    print("=" * 70)

    G = nx.Graph()

    # Add target nodes
    for t in targets:
        G.add_node(t, bipartite=0, node_type="target",
                   full_name=TARGET_INFO[t]["full_name"],
                   cancer_role=TARGET_INFO[t]["cancer_role"])

    # Add compound nodes with edges weighted by binding affinity
    elite_ids = set(elite["cpd_id"].values) if len(elite) > 0 else set()

    for _, row in profiles.iterrows():
        cpd_id = row["cpd_id"]
        is_elite = cpd_id in elite_ids
        G.add_node(cpd_id, bipartite=1, node_type="compound",
                   source=row.get("source", ""), is_elite=is_elite,
                   compound_name=row.get("compound_name", ""))

        for tc, t in zip(target_cols, targets):
            dg = row[tc]
            if pd.notna(dg) and dg <= -7.0:
                # Edge weight: absolute binding energy (higher = stronger)
                G.add_edge(cpd_id, t, weight=abs(dg), dG=dg)

    n_cpd = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "compound")
    n_edges = G.number_of_edges()
    print(f"  Nodes: {len(targets)} targets + {n_cpd} compounds = {G.number_of_nodes()}")
    print(f"  Edges: {n_edges} (binding affinity ≤ −7.0 kcal/mol)")
    print(f"  Density: {nx.density(G):.3f}")

    # Target degree (number of bound compounds)
    print(f"\n  Target degree (compounds bound at ≤−7.0):")
    for t in sorted(targets):
        deg = G.degree(t)
        avg_w = np.mean([G[t][n]["weight"] for n in G.neighbors(t)]) if deg > 0 else 0
        print(f"    {t:8s}: {deg:3d} compounds, mean |ΔG|={avg_w:.1f}")

    return G


# ══════════════════════════════════════════════════════════════════════
# Module C: Target–Target Co-Binding Network
# ══════════════════════════════════════════════════════════════════════

def build_target_network(profiles, target_cols, targets):
    print("\n" + "=" * 70)
    print("MODULE C: Target–Target Co-Binding Network")
    print("=" * 70)

    G_tt = nx.Graph()
    for t in targets:
        G_tt.add_node(t, **TARGET_INFO[t])

    # For each pair of targets, count compounds that bind both at ≤−7.0
    for i, t1 in enumerate(targets):
        for j, t2 in enumerate(targets):
            if j <= i:
                continue
            col1, col2 = f"dG_{t1}", f"dG_{t2}"
            both = profiles[(profiles[col1] <= -7.0) & (profiles[col2] <= -7.0)]
            if len(both) > 0:
                jaccard = len(both) / len(profiles[(profiles[col1] <= -7.0) |
                                                    (profiles[col2] <= -7.0)])
                mean_e1 = both[col1].mean()
                mean_e2 = both[col2].mean()
                G_tt.add_edge(t1, t2, n_shared=len(both), jaccard=round(jaccard, 3),
                              mean_dG_1=round(mean_e1, 1), mean_dG_2=round(mean_e2, 1))

    print(f"  Target–target edges: {G_tt.number_of_edges()}")
    print(f"\n  Co-binding matrix (n compounds binding both at ≤−7.0):")

    # Print co-binding matrix
    header = f"{'':8s}" + "".join(f"{t:>8s}" for t in targets)
    print(f"  {header}")
    for t1 in targets:
        row_str = f"  {t1:8s}"
        for t2 in targets:
            if t1 == t2:
                n = len(profiles[profiles[f"dG_{t1}"] <= -7.0])
                row_str += f"{n:>8d}"
            elif G_tt.has_edge(t1, t2):
                row_str += f"{G_tt[t1][t2]['n_shared']:>8d}"
            else:
                row_str += f"{'0':>8s}"
        print(row_str)

    # Network metrics
    print(f"\n  Target network centrality:")
    degree_cent = nx.degree_centrality(G_tt)
    between_cent = nx.betweenness_centrality(G_tt, weight="n_shared")
    close_cent = nx.closeness_centrality(G_tt)

    centrality_data = []
    for t in targets:
        centrality_data.append({
            "target": t,
            "degree_centrality": round(degree_cent.get(t, 0), 3),
            "betweenness_centrality": round(between_cent.get(t, 0), 3),
            "closeness_centrality": round(close_cent.get(t, 0), 3),
            "n_neighbours": G_tt.degree(t),
        })
        print(f"    {t:8s}  deg={degree_cent[t]:.3f}  "
              f"betw={between_cent[t]:.3f}  close={close_cent[t]:.3f}")

    centrality_df = pd.DataFrame(centrality_data)
    centrality_df.to_csv(RESULTS_DIR / "target_centrality.tsv", sep="\t", index=False)

    # Identify pathway-linked target clusters
    if G_tt.number_of_edges() > 0:
        communities = list(nx.community.greedy_modularity_communities(G_tt))
        print(f"\n  Detected target communities (modularity-based): {len(communities)}")
        for i, comm in enumerate(communities):
            print(f"    Cluster {i+1}: {', '.join(sorted(comm))}")

    return G_tt, centrality_df


# ══════════════════════════════════════════════════════════════════════
# Module D: OSCC Pathway Mapping
# ══════════════════════════════════════════════════════════════════════

def map_pathways(targets, profiles, target_cols):
    print("\n" + "=" * 70)
    print("MODULE D: OSCC Pathway Mapping")
    print("=" * 70)

    # Build pathway → targets mapping
    pathway_targets = defaultdict(list)
    for t in targets:
        for pw in TARGET_INFO[t]["pathways"]:
            pathway_targets[pw].append(t)

    # Count compounds hitting each pathway (via target binding)
    pathway_compound_counts = {}
    for pw, pw_targets in pathway_targets.items():
        # A compound "hits" a pathway if it binds ≥1 target in that pathway at ≤−7.0
        mask = pd.Series(False, index=profiles.index)
        for t in pw_targets:
            mask = mask | (profiles[f"dG_{t}"] <= -7.0)
        pathway_compound_counts[pw] = mask.sum()

    rows = []
    print(f"\n  Pathway coverage by multi-target binders:")
    for pw in sorted(pathway_targets.keys(), key=lambda x: pathway_compound_counts[x], reverse=True):
        cat = PATHWAY_CATEGORIES.get(pw, "Other")
        n_targets = len(pathway_targets[pw])
        n_cpds = pathway_compound_counts[pw]
        rows.append({
            "pathway": pw,
            "category": cat,
            "n_targets": n_targets,
            "targets": ", ".join(pathway_targets[pw]),
            "n_compounds_hitting": n_cpds,
            "pct_compounds": round(n_cpds * 100 / len(profiles), 1),
        })
        print(f"    {pw:50s}  targets={n_targets}  cpds={n_cpds:3d} ({n_cpds*100/len(profiles):.0f}%)  [{cat}]")

    pw_df = pd.DataFrame(rows)
    pw_df.to_csv(RESULTS_DIR / "pathway_mapping.tsv", sep="\t", index=False)

    # Category-level summary
    print(f"\n  Pathway category summary:")
    cat_data = []
    for cat in sorted(set(PATHWAY_CATEGORIES.values())):
        cat_pathways = [pw for pw, c in PATHWAY_CATEGORIES.items() if c == cat]
        cat_targets = set()
        for pw in cat_pathways:
            cat_targets.update(pathway_targets.get(pw, []))
        # Compounds hitting ANY target in this category
        mask = pd.Series(False, index=profiles.index)
        for t in cat_targets:
            col = f"dG_{t}"
            if col in profiles.columns:
                mask = mask | (profiles[col] <= -7.0)
        n = mask.sum()
        cat_data.append({
            "category": cat,
            "n_pathways": len(cat_pathways),
            "n_targets": len(cat_targets),
            "targets": ", ".join(sorted(cat_targets)),
            "n_compounds": n,
        })
        print(f"    {cat:25s}  pathways={len(cat_pathways)}  targets={len(cat_targets)}  cpds={n}")

    cat_df = pd.DataFrame(cat_data)
    cat_df.to_csv(RESULTS_DIR / "pathway_categories.tsv", sep="\t", index=False)

    return pw_df, pathway_targets


# ══════════════════════════════════════════════════════════════════════
# Module E: Selectivity Heatmap
# ══════════════════════════════════════════════════════════════════════

def generate_selectivity_heatmap(profiles, target_cols, targets, elite):
    print("\n" + "=" * 70)
    print("MODULE E: Selectivity Heatmap")
    print("=" * 70)

    elite_ids = set(elite["cpd_id"].values) if len(elite) > 0 else set()

    # Sort compounds by total binding strength
    energy_matrix = profiles[target_cols].copy()
    energy_matrix.index = profiles["cpd_id"]
    energy_matrix.columns = targets

    # Sort by mean energy (best binders first)
    mean_energy = energy_matrix.mean(axis=1)
    energy_matrix = energy_matrix.loc[mean_energy.sort_values().index]

    # ── Full heatmap (50 compounds × 7 targets) ──
    fig, ax = plt.subplots(figsize=(10, 16))
    cmap = LinearSegmentedColormap.from_list("dock",
        ["#dc3545", "#fd7e14", "#ffc107", "#28a745", "#0d6efd"], N=256)

    im = ax.imshow(energy_matrix.values, aspect='auto', cmap=cmap,
                   vmin=-14, vmax=-5)

    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, fontsize=10, rotation=45, ha='right')
    ax.set_yticks(range(len(energy_matrix)))

    # Label with compound names, highlight elite
    ylabels = []
    for cpd_id in energy_matrix.index:
        row = profiles[profiles["cpd_id"] == cpd_id].iloc[0]
        name = row.get("compound_name", "")
        if pd.isna(name) or name == "":
            name = cpd_id
        if cpd_id in elite_ids:
            name = f"★ {name}"
        ylabels.append(name)

    ax.set_yticklabels(ylabels, fontsize=7)

    # Annotate cells with energy values
    for i in range(len(energy_matrix)):
        for j in range(len(targets)):
            val = energy_matrix.iloc[i, j]
            if pd.notna(val):
                color = "white" if val <= -10 else "black"
                ax.text(j, i, f"{val:.1f}", ha='center', va='center',
                        fontsize=5.5, color=color)

    plt.colorbar(im, ax=ax, label="Binding Energy (kcal/mol)", shrink=0.6)
    ax.set_title("Multi-Target Selectivity Profiles\n(50 Elite Compounds × 7 OSCC Targets)",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("Target", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "selectivity_heatmap.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: selectivity_heatmap.png")

    # ── Elite-only heatmap ──
    if len(elite) > 0:
        elite_matrix = energy_matrix.loc[energy_matrix.index.isin(elite_ids)]

        fig2, ax2 = plt.subplots(figsize=(10, max(4, len(elite_matrix) * 0.5)))
        im2 = ax2.imshow(elite_matrix.values, aspect='auto', cmap=cmap,
                         vmin=-14, vmax=-5)
        ax2.set_xticks(range(len(targets)))
        ax2.set_xticklabels(targets, fontsize=11, rotation=45, ha='right')
        ax2.set_yticks(range(len(elite_matrix)))

        elite_labels = []
        for cpd_id in elite_matrix.index:
            row = profiles[profiles["cpd_id"] == cpd_id].iloc[0]
            name = row.get("compound_name", "")
            if pd.isna(name) or name == "":
                name = row.get("method", cpd_id)
            elite_labels.append(f"{cpd_id}  ({name})")
        ax2.set_yticklabels(elite_labels, fontsize=9)

        for i in range(len(elite_matrix)):
            for j in range(len(targets)):
                val = elite_matrix.iloc[i, j]
                if pd.notna(val):
                    color = "white" if val <= -10 else "black"
                    ax2.text(j, i, f"{val:.1f}", ha='center', va='center',
                             fontsize=9, color=color, fontweight='bold')

        plt.colorbar(im2, ax=ax2, label="ΔG (kcal/mol)", shrink=0.8)
        ax2.set_title("Triple-Validated Elite Candidates — Binding Profile",
                      fontsize=13, fontweight='bold')
        fig2.tight_layout()
        fig2.savefig(RESULTS_DIR / "elite_selectivity_heatmap.png", dpi=200,
                     bbox_inches='tight')
        plt.close(fig2)
        print(f"  Saved: elite_selectivity_heatmap.png")

    # ── Target correlation heatmap ──
    corr = energy_matrix.corr(method='spearman')
    fig3, ax3 = plt.subplots(figsize=(8, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
                vmin=-1, vmax=1, center=0, ax=ax3,
                linewidths=0.5, square=True)
    ax3.set_title("Target–Target Binding Correlation\n(Spearman ρ across 50 compounds)",
                  fontsize=12, fontweight='bold')
    fig3.tight_layout()
    fig3.savefig(RESULTS_DIR / "target_correlation_heatmap.png", dpi=200,
                 bbox_inches='tight')
    plt.close(fig3)
    print(f"  Saved: target_correlation_heatmap.png")

    return energy_matrix


# ══════════════════════════════════════════════════════════════════════
# Module F: Network Metrics & Hub Identification
# ══════════════════════════════════════════════════════════════════════

def compute_network_metrics(G_bipartite, profiles, targets, elite):
    print("\n" + "=" * 70)
    print("MODULE F: Network Metrics & Hub Identification")
    print("=" * 70)

    compounds = [n for n, d in G_bipartite.nodes(data=True)
                 if d.get("node_type") == "compound"]

    elite_ids = set(elite["cpd_id"].values)

    # Compound-level metrics
    cpd_metrics = []
    for cpd in compounds:
        neighbors = list(G_bipartite.neighbors(cpd))
        target_neighbors = [n for n in neighbors
                           if G_bipartite.nodes[n].get("node_type") == "target"]
        n_targets = len(target_neighbors)
        mean_affinity = np.mean([G_bipartite[cpd][t]["weight"]
                                 for t in target_neighbors]) if n_targets > 0 else 0
        max_affinity = max([G_bipartite[cpd][t]["weight"]
                           for t in target_neighbors]) if n_targets > 0 else 0

        # Polypharmacology score: n_targets × mean_affinity
        poly_score = n_targets * mean_affinity

        cpd_metrics.append({
            "cpd_id": cpd,
            "is_elite": cpd in elite_ids,
            "n_targets_bound": n_targets,
            "mean_affinity": round(mean_affinity, 2),
            "max_affinity": round(max_affinity, 2),
            "polypharmacology_score": round(poly_score, 1),
            "targets_bound": ", ".join(sorted(target_neighbors)),
            "source": G_bipartite.nodes[cpd].get("source", ""),
        })

    cpd_df = pd.DataFrame(cpd_metrics).sort_values("polypharmacology_score",
                                                     ascending=False)
    cpd_df.to_csv(RESULTS_DIR / "compound_network_metrics.tsv", sep="\t", index=False)

    print(f"  Compound network metrics:")
    print(f"    Total compounds in network: {len(cpd_df)}")
    print(f"    Mean targets bound: {cpd_df['n_targets_bound'].mean():.1f}")
    print(f"    Mean polypharmacology score: {cpd_df['polypharmacology_score'].mean():.1f}")

    # Top polypharmacology hubs
    print(f"\n  === Top 15 Polypharmacology Hubs ===")
    for i, (_, row) in enumerate(cpd_df.head(15).iterrows()):
        elite_mark = "★" if row["is_elite"] else " "
        print(f"  {elite_mark} {i+1:2d}. {row['cpd_id']:30s}  "
              f"targets={row['n_targets_bound']}  mean|ΔG|={row['mean_affinity']:.1f}  "
              f"poly_score={row['polypharmacology_score']:.1f}  "
              f"[{row['source']}]")

    # Elite vs non-elite comparison
    if len(elite_ids) > 0:
        elite_metrics = cpd_df[cpd_df["is_elite"]]
        non_elite_metrics = cpd_df[~cpd_df["is_elite"]]
        print(f"\n  Elite vs Non-Elite:")
        print(f"    {'Metric':<25s}  {'Elite (n={})'.format(len(elite_metrics)):>18s}  "
              f"{'Non-Elite (n={})'.format(len(non_elite_metrics)):>18s}")
        for col in ["n_targets_bound", "mean_affinity", "polypharmacology_score"]:
            e_val = elite_metrics[col].mean()
            ne_val = non_elite_metrics[col].mean() if len(non_elite_metrics) > 0 else 0
            print(f"    {col:<25s}  {e_val:>18.2f}  {ne_val:>18.2f}")

    return cpd_df


# ══════════════════════════════════════════════════════════════════════
# Module G: Pathway Enrichment Analysis
# ══════════════════════════════════════════════════════════════════════

def pathway_enrichment(profiles, targets, elite, pathway_targets):
    print("\n" + "=" * 70)
    print("MODULE G: Pathway Enrichment Analysis")
    print("=" * 70)

    elite_ids = set(elite["cpd_id"].values)
    elite_profiles = profiles[profiles["cpd_id"].isin(elite_ids)]

    # For each pathway, count elite compounds hitting it vs background rate
    enrichment = []
    n_elite = len(elite_profiles)
    n_total = len(profiles)

    for pw, pw_targets in pathway_targets.items():
        # Background: fraction of all compounds hitting this pathway
        bg_mask = pd.Series(False, index=profiles.index)
        for t in pw_targets:
            col = f"dG_{t}"
            if col in profiles.columns:
                bg_mask = bg_mask | (profiles[col] <= -7.0)
        bg_count = bg_mask.sum()
        bg_rate = bg_count / n_total if n_total > 0 else 0

        # Elite: fraction hitting this pathway
        el_mask = pd.Series(False, index=elite_profiles.index)
        for t in pw_targets:
            col = f"dG_{t}"
            if col in elite_profiles.columns:
                el_mask = el_mask | (elite_profiles[col] <= -7.0)
        el_count = el_mask.sum()
        el_rate = el_count / n_elite if n_elite > 0 else 0

        fold = el_rate / bg_rate if bg_rate > 0 else float('inf')

        enrichment.append({
            "pathway": pw,
            "category": PATHWAY_CATEGORIES.get(pw, "Other"),
            "n_elite_hits": el_count,
            "elite_rate": round(el_rate, 3),
            "n_background_hits": bg_count,
            "background_rate": round(bg_rate, 3),
            "fold_enrichment": round(fold, 2),
        })

    enrich_df = pd.DataFrame(enrichment).sort_values("fold_enrichment", ascending=False)
    enrich_df.to_csv(RESULTS_DIR / "pathway_enrichment.tsv", sep="\t", index=False)

    print(f"  Pathway enrichment (elite vs background):")
    for _, row in enrich_df.iterrows():
        indicator = "▲" if row["fold_enrichment"] > 1 else "▼"
        print(f"    {indicator} {row['pathway']:50s}  "
              f"elite={row['n_elite_hits']}/{n_elite} ({row['elite_rate']:.0%})  "
              f"bg={row['n_background_hits']}/{n_total} ({row['background_rate']:.0%})  "
              f"fold={row['fold_enrichment']:.2f}x")

    return enrich_df


# ══════════════════════════════════════════════════════════════════════
# Module H: Visualization — Network Layouts & Radar Charts
# ══════════════════════════════════════════════════════════════════════

def generate_visualizations(G_tt, profiles, targets, elite, pw_df, energy_matrix):
    print("\n" + "=" * 70)
    print("MODULE H: Visualization")
    print("=" * 70)

    # ── 1. Target–Target Network Graph ──
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G_tt, seed=42, k=2)

    # Node sizes by degree
    node_sizes = [600 + G_tt.degree(t, weight="n_shared") * 20 for t in G_tt.nodes()]

    # Node colours by cancer role
    role_colours = {
        "Survival/Proliferation": "#e74c3c",
        "Proliferation/Migration": "#3498db",
        "Proliferation/Angiogenesis": "#2ecc71",
        "Tumour Suppressor Evasion": "#9b59b6",
        "Metabolism/Growth": "#f39c12",
        "Inflammation/Angiogenesis": "#e67e22",
    }
    node_colors = [role_colours.get(TARGET_INFO[t]["cancer_role"], "#95a5a6")
                   for t in G_tt.nodes()]

    # Edge widths by shared binder count
    edge_widths = [G_tt[u][v].get("n_shared", 1) / 10 for u, v in G_tt.edges()]

    nx.draw_networkx_nodes(G_tt, pos, ax=ax, node_size=node_sizes,
                           node_color=node_colors, edgecolors='black',
                           linewidths=1.5, alpha=0.9)
    nx.draw_networkx_labels(G_tt, pos, ax=ax, font_size=11, font_weight='bold')
    nx.draw_networkx_edges(G_tt, pos, ax=ax, width=edge_widths,
                           alpha=0.6, edge_color='#555555')

    # Edge labels (n_shared)
    edge_labels = {(u, v): str(G_tt[u][v]["n_shared"]) for u, v in G_tt.edges()}
    nx.draw_networkx_edge_labels(G_tt, pos, edge_labels=edge_labels,
                                  ax=ax, font_size=8, font_color='#333')

    # Legend
    patches = [mpatches.Patch(color=c, label=r) for r, c in role_colours.items()]
    ax.legend(handles=patches, loc='lower left', fontsize=8, title="Cancer Role",
              title_fontsize=9)
    ax.set_title("OSCC Target Co-Binding Network\n(edge width ∝ shared binders; "
                 "label = n compounds binding both)", fontsize=12, fontweight='bold')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "target_network.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: target_network.png")

    # ── 2. Pathway Coverage Bar Chart ──
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    pw_sorted = pw_df.sort_values("n_compounds_hitting", ascending=True)

    colors_cat = {
        "PI3K/AKT/mTOR Axis": "#e74c3c",
        "RTK/MAPK Cascade": "#3498db",
        "Cell Cycle/Apoptosis": "#9b59b6",
        "Inflammation/TME": "#e67e22",
    }
    bar_colors = [colors_cat.get(cat, "#95a5a6") for cat in pw_sorted["category"]]

    bars = ax2.barh(range(len(pw_sorted)), pw_sorted["n_compounds_hitting"],
                    color=bar_colors, edgecolor='black', linewidth=0.5)
    ax2.set_yticks(range(len(pw_sorted)))
    ax2.set_yticklabels(pw_sorted["pathway"], fontsize=8)
    ax2.set_xlabel("Number of Compounds Hitting Pathway", fontsize=11)
    ax2.set_title("OSCC Pathway Coverage by Multi-Target Binders\n(via target-mediated binding)",
                  fontsize=12, fontweight='bold')

    # Annotate bars
    for bar, n in zip(bars, pw_sorted["n_compounds_hitting"]):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 str(n), va='center', fontsize=9, fontweight='bold')

    # Legend
    patches2 = [mpatches.Patch(color=c, label=r) for r, c in colors_cat.items()]
    ax2.legend(handles=patches2, loc='lower right', fontsize=9, title="Pathway Category")
    fig2.tight_layout()
    fig2.savefig(RESULTS_DIR / "pathway_coverage.png", dpi=200, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved: pathway_coverage.png")

    # ── 3. Elite Compound Radar Charts ──
    if len(elite) > 0:
        elite_ids = list(elite["cpd_id"].values[:6])  # Top 6

        fig3, axes = plt.subplots(2, 3, figsize=(15, 10),
                                   subplot_kw=dict(projection='polar'))
        axes = axes.flatten()

        angles = np.linspace(0, 2 * np.pi, len(targets), endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon

        for idx, cpd_id in enumerate(elite_ids):
            ax_r = axes[idx]
            row = profiles[profiles["cpd_id"] == cpd_id].iloc[0]
            values = [abs(row.get(f"dG_{t}", 0)) for t in targets]
            values += values[:1]

            ax_r.fill(angles, values, alpha=0.25, color=f"C{idx}")
            ax_r.plot(angles, values, 'o-', linewidth=2, color=f"C{idx}", markersize=4)
            ax_r.set_xticks(angles[:-1])
            ax_r.set_xticklabels(targets, fontsize=8)
            ax_r.set_ylim(0, 15)
            ax_r.set_yticks([5, 8, 10, 12])
            ax_r.set_yticklabels(["5", "8", "10", "12"], fontsize=6, color='grey')

            name = row.get("compound_name", "")
            if pd.isna(name) or name == "":
                name = row.get("method", cpd_id)
            ax_r.set_title(f"{cpd_id}\n({name})", fontsize=9, fontweight='bold', pad=15)

        fig3.suptitle("Elite Compound Binding Profiles (|ΔG| kcal/mol per target)",
                      fontsize=13, fontweight='bold', y=1.02)
        fig3.tight_layout()
        fig3.savefig(RESULTS_DIR / "elite_radar_charts.png", dpi=200, bbox_inches='tight')
        plt.close(fig3)
        print(f"  Saved: elite_radar_charts.png")

    # ── 4. Compound Source × Multi-Target Hits Distribution ──
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    for src in ["repurposed", "generated"]:
        sub = profiles[profiles["source"] == src]
        counts = sub["n_hits_le7"].value_counts().sort_index()
        ax4.bar(counts.index + (-0.2 if src == "repurposed" else 0.2),
                counts.values, width=0.35,
                label=src.capitalize(), alpha=0.85,
                color="#3498db" if src == "repurposed" else "#e74c3c",
                edgecolor='black', linewidth=0.5)
    ax4.set_xlabel("Number of Targets Bound (≤−7.0 kcal/mol)", fontsize=11)
    ax4.set_ylabel("Number of Compounds", fontsize=11)
    ax4.set_title("Multi-Target Binding Distribution by Source", fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.set_xticks(range(int(profiles["n_hits_le7"].min()),
                         int(profiles["n_hits_le7"].max()) + 1))
    fig4.tight_layout()
    fig4.savefig(RESULTS_DIR / "multitarget_distribution.png", dpi=200, bbox_inches='tight')
    plt.close(fig4)
    print(f"  Saved: multitarget_distribution.png")

    print(f"\n  Total figures generated: 5")


# ══════════════════════════════════════════════════════════════════════
# Summary & Output
# ══════════════════════════════════════════════════════════════════════

def save_summary(profiles, targets, elite, G_tt, cpd_metrics, pw_df, enrich_df):
    print("\n" + "=" * 70)
    print("Saving Summary")
    print("=" * 70)

    elite_ids = set(elite["cpd_id"].values)

    summary = {
        "phase": "9a",
        "description": "Network Pharmacology & Pathway Analysis",
        "n_compounds": len(profiles),
        "n_targets": len(targets),
        "n_elite": len(elite),
        "n_pathways_mapped": len(pw_df),
        "n_pathway_categories": len(set(PATHWAY_CATEGORIES.values())),

        "target_network": {
            "n_edges": G_tt.number_of_edges(),
            "density": round(nx.density(G_tt), 3),
            "mean_shared_binders": round(
                np.mean([G_tt[u][v]["n_shared"] for u, v in G_tt.edges()]), 1),
        },

        "polypharmacology": {
            "mean_targets_per_compound": round(
                cpd_metrics["n_targets_bound"].mean(), 1),
            "mean_polypharmacology_score": round(
                cpd_metrics["polypharmacology_score"].mean(), 1),
            "elite_mean_poly_score": round(
                cpd_metrics[cpd_metrics["is_elite"]]["polypharmacology_score"].mean(),
                1) if len(elite) > 0 else None,
        },

        "pathway_coverage": {
            "top_pathway": pw_df.sort_values("n_compounds_hitting",
                                              ascending=False).iloc[0]["pathway"],
            "top_pathway_hits": int(pw_df.sort_values("n_compounds_hitting",
                                                       ascending=False).iloc[0]["n_compounds_hitting"]),
        },

        "output_files": {
            "data": ["target_centrality.tsv", "pathway_mapping.tsv",
                     "pathway_categories.tsv", "compound_network_metrics.tsv",
                     "pathway_enrichment.tsv"],
            "figures": ["selectivity_heatmap.png", "elite_selectivity_heatmap.png",
                       "target_correlation_heatmap.png", "target_network.png",
                       "pathway_coverage.png", "elite_radar_charts.png",
                       "multitarget_distribution.png"],
        },
    }

    with open(RESULTS_DIR / "network_pharmacology_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Summary saved: network_pharmacology_summary.json")
    print(f"  Data tables: {len(summary['output_files']['data'])}")
    print(f"  Figures: {len(summary['output_files']['figures'])}")


# ══════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70)
    print("GENESIS — Phase 9a: Network Pharmacology & Pathway Analysis")
    print("=" * 70)
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Module A: Load data
    profiles, polynet, admet, elite, target_cols, targets = load_data()

    # Module B: Bipartite network
    G_bipartite = build_bipartite_network(profiles, target_cols, targets, elite)

    # Module C: Target–Target co-binding network
    G_tt, centrality_df = build_target_network(profiles, target_cols, targets)

    # Module D: Pathway mapping
    pw_df, pathway_targets = map_pathways(targets, profiles, target_cols)

    # Module E: Selectivity heatmap
    energy_matrix = generate_selectivity_heatmap(profiles, target_cols, targets, elite)

    # Module F: Network metrics
    cpd_metrics = compute_network_metrics(G_bipartite, profiles, targets, elite)

    # Module G: Pathway enrichment
    enrich_df = pathway_enrichment(profiles, targets, elite, pathway_targets)

    # Module H: Visualization
    generate_visualizations(G_tt, profiles, targets, elite, pw_df, energy_matrix)

    # Summary
    save_summary(profiles, targets, elite, G_tt, cpd_metrics, pw_df, enrich_df)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Phase 9a COMPLETE in {elapsed:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
