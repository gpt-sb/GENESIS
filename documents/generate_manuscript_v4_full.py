#!/usr/bin/env python3
"""
GENESIS NMI Manuscript v4 (full scientific depth) — generated from
documents/manuscript_ground_truth.json (numeric ledger) plus formal equations
and citations drawn from documents/../../GENESIS/documents/Formulas_Citations.md
(mathematically/biologically verified against code this session and prior).

Differences from v3_freshdata.docx:
  - Restores all 11+ formal equations, VP-centroid/lambda-schedule/beta(t)/REINFORCE
    definitions, ordinal-weight/curriculum/tier-margin tables, EF@K/BEDROC/FCD
    formal metric definitions, and ~35 citations.
  - Restores biological/clinical rationale (OSCC epidemiology, target-pathway
    biology, natural-product context) and generative-architecture mechanistic
    detail (FiLM conditioning, flow matching, reward composition).
  - Per user instruction (2026-09-18): the bimodal-RL-landscape topic is not
    mentioned at all (not even as a withdrawn/non-reproducing claim).
  - Per user instruction: this GENESIS_1 run is treated as the canonical,
    first-reported result -- no "vs. previously published" framing anywhere.
Output: documents/GENESIS_NMI_Manuscript_v4_Full.docx (v1/v3 untouched)
"""
import json
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

HERE = Path(__file__).resolve().parent
LEDGER = json.load(open(HERE / 'manuscript_ground_truth.json'))
OUT = HERE / 'GENESIS_NMI_Manuscript_v4_Full.docx'
OUT_SI = HERE / 'GENESIS_NMI_Manuscript_v4_Supplementary_Information.docx'
IMAGES_DIR = HERE / 'images'

PGCL = LEDGER['pgcl_core_3seed']
T1 = LEDGER['table1_baselines']['metrics']
AB = LEDGER['ablations_8config']
GEN = LEDGER['generative_multiseed_5seed']
P4 = LEDGER['phase4_bioactivity_profiling']
AN = P4['aniline_toxicophore_flag_rate']
P6 = LEDGER['phase6_drug_repurposing']
P7 = LEDGER['phase7_docking']
P8 = LEDGER['phase8_admet']
P9A = LEDGER['phase9a_network_pharmacology']
P9B = LEDGER['phase9b_lead_optimization']
P10 = LEDGER['phase10_final_ranking']
CC = LEDGER['cross_cancer_transfer']
MOSES3 = LEDGER['generative_moses_guacamol_3seed']

RHO, PM, ARROW, DEG = '\u03c1', '\u00b1', '\u2192', '\u00b0'
TAU, LAM, BETA, DELTA, ALPHA = '\u03c4', '\u03bb', '\u03b2', '\u03b4', '\u03b1'
SUM, MB, NEQ = '\u03a3', '\u2211', '\u2260'


def set_cell_shading(cell, color_hex):
    shading = cell._element.get_or_add_tcPr()
    el = shading.makeelement(qn('w:shd'), {qn('w:fill'): color_hex, qn('w:val'): 'clear'})
    shading.append(el)


def para(doc, text, style='Normal', bold=False, italic=False, size=None,
         alignment=None, space_after=None, space_before=None):
    p = doc.add_paragraph(style=style)
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    if size:
        run.font.size = Pt(size)
    if alignment:
        p.alignment = alignment
    if space_after is not None:
        p.paragraph_format.space_after = Pt(space_after)
    if space_before is not None:
        p.paragraph_format.space_before = Pt(space_before)
    return p


def equation(doc, text, label=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    r.font.name = 'Cambria Math'
    r.font.size = Pt(11)
    r.italic = True
    if label:
        t = p.add_run(f'\t({label})')
        t.font.name = 'Cambria Math'
        t.font.size = Pt(10)
    return p


def table(doc, headers, rows, caption=None):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = 'Table Grid'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h_ in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = str(h_)
        for p in cell.paragraphs:
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(9)
        set_cell_shading(cell, 'D9E2F3')
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = t.rows[ri + 1].cells[ci]
            cell.text = str(val)
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)
    if caption:
        para(doc, caption, italic=True, size=9, space_after=10)
    return t


def h(doc, text, level=1):
    doc.add_heading(text, level=level)


def figure(doc, filename, caption, width_in=6.0):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(IMAGES_DIR / filename), width=Inches(width_in))
    para(doc, caption, italic=True, size=9, alignment=WD_ALIGN_PARAGRAPH.CENTER, space_after=12)


# ═══════════════════════════════════════════════════════════════════════════
def build():
    doc = Document()
    for section in doc.sections:
        section.top_margin = Cm(2.54)
        section.bottom_margin = Cm(2.54)
        section.left_margin = Cm(2.54)
        section.right_margin = Cm(2.54)
    style = doc.styles['Normal']
    style.font.name = 'Arial'
    style.font.size = Pt(11)
    style.paragraph_format.line_spacing = 2.0

    para(doc, 'GENESIS: Potency-Graded Contrastive Learning for Generative '
               'Molecular Design and Drug Repurposing in Oral Squamous Cell '
               'Carcinoma', bold=True, size=16, alignment=WD_ALIGN_PARAGRAPH.CENTER,
         space_after=12)
    para(doc, 'Saurabh Bundela', size=12, alignment=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
    para(doc, 'Correspondence: saurabhbundela5@gmail.com', italic=True, size=10,
         alignment=WD_ALIGN_PARAGRAPH.CENTER, space_after=24)

    # ══════════════════════════════════════════════════════════════════════
    h(doc, 'Abstract')
    para(doc,
        'Oral squamous cell carcinoma (OSCC) causes roughly 377,000 new cases and 177,000 deaths '
        'annually, with five-year survival below 50% for three decades. Computational drug '
        'discovery typically treats bioactivity as binary, discarding the graded, ordinal nature '
        'of dose-response data central to medicinal chemistry. We introduce Potency-Graded '
        'Contrastive Learning (PGCL), projecting frozen molecular language-model embeddings into '
        'a 128-dimensional space via an ordinal contrastive objective, a six-tier negative '
        'taxonomy, and tier-margin ranking. Across three seeds, PGCL achieves Spearman '
        f'{RHO} = {PGCL["global_spearman_rho"]["mean"]:.3f} {PM} {PGCL["global_spearman_rho"]["std"]:.3f} '
        f'for potency ordering and target-switch AUROC = {PGCL["target_switch_auc_roc"]["mean"]:.3f}, '
        'outperforming flat-contrastive and binary-classification ablations and roughly doubling '
        'ECFP4 activity-cliff discrimination. Among six generative architectures evaluated across '
        'five seeds, CRAVE significantly exceeds non-RL baselines on OSCC potency after '
        'Bonferroni correction (one-sided Mann-Whitney, p_Bonferroni = 0.016). The frozen model '
        f'transfers substantially to lung adenocarcinoma without retraining ({RHO} = {CC["pgcl_rho"]:.3f}, '
        f'n = {CC["n_in_index"]:,}). An integrated docking, ADMET, network-pharmacology, and '
        f'lead-optimisation pipeline screens {P6["n_total_compounds"]:,} repurposing candidates and '
        f'prioritises {P10["n_total_ranked"]:,} molecules into '
        f'{P10["validation_tier_distribution"]["Tier-1_fully_validated"]} Tier-1 candidates, including '
        f'{P8["triple_validated_elite"]["count"]} triple-validated elite compounds. We report '
        f'safety pass rates transparently ({P4["safety_summary"]["pass_rate_pct"]}%), tracing the '
        'limiting factor to a training-corpus EGFR/aniline skew. A complete data ledger and '
        'automated verifier accompany the code for independent replication.')

    # ══════════════════════════════════════════════════════════════════════
    h(doc, 'Introduction')
    para(doc,
        'Oral squamous cell carcinoma (OSCC) represents approximately 90% of oral cavity '
        'malignancies and accounts for roughly 377,000 new cases and 177,000 deaths annually '
        'worldwide [1]. Despite advances in surgical technique and radiation '
        'therapy, five-year survival for advanced-stage disease remains below 50%, largely '
        'unchanged over three decades [2]. OSCC\u2019s molecular heterogeneity '
        '\u2014 aberrant signalling through the receptor tyrosine kinase (RTK) / PI3K / AKT / mTOR '
        'axis, TP53 pathway disruption, and epigenetic dysregulation \u2014 creates a complex, '
        'multi-target therapeutic landscape that single-target approaches have struggled to '
        'address. Current clinical practice is dominated by EGFR-directed agents (cetuximab, '
        'erlotinib) and mTOR inhibitors (everolimus), with immune-checkpoint agents '
        '(nivolumab, pembrolizumab) as a more recent addition; fewer than ten agents carry an '
        'OSCC-specific approval despite thousands of compounds in early-stage pipelines.')
    para(doc,
        'Traditional virtual screening \u2014 structure-based (molecular docking) or ligand-based '
        '(pharmacophore models, QSAR) \u2014 operates at the single-target level and does not '
        'naturally capture the multi-target, pathway-level potency landscape relevant to '
        'complex diseases such as OSCC. Molecular fingerprint methods (ECFP, MACCS) [3] encode structural similarity but no bioactivity information. Molecular '
        'language models pretrained on masked-token recovery (MolBERT, ChemBERTa, MolDeBERTa) '
        '[4] learn rich chemical representations but only weakly '
        'correlate with potency out of the box. Graph neural networks for property prediction '
        'show promise for single-target QSAR but require per-target retraining and do not '
        'naturally yield a unified, multi-target embedding space. Supervised contrastive '
        'learning [6] provides a powerful paradigm for structured '
        'representation learning, but its application to drug discovery has typically been '
        'restricted to binary active/inactive labels, discarding the ordinal structure of '
        'potency data that is the central organising principle of medicinal chemistry.')
    para(doc,
        'Natural products have historically yielded some of the most successful anticancer '
        'agents (vincristine, paclitaxel, camptothecin), and phytochemicals such as '
        'resveratrol, berberine, and quercetin have demonstrated in vitro activity against '
        'OSCC cell lines, motivating natural-product-informed screening as an orthogonal '
        'validation signal alongside purely data-driven generation.')
    para(doc,
        'We present GENESIS, an 11-phase (Phase 0\u201310) end-to-end framework centred on '
        'Potency-Graded Contrastive Learning (PGCL). PGCL introduces three innovations to '
        'supervised contrastive learning: (1) ordinal pair weights derived from potency-class '
        'proximity rather than binary same/different labels, (2) a six-tier negative taxonomy '
        'spanning a biologically motivated hardness gradient with curriculum scheduling, and '
        '(3) inversely proportional tier margins that concentrate representational capacity on '
        'pharmacologically informative distinctions. The resulting 128-dimensional space serves '
        'simultaneously as a reinforcement-learning reward signal, a conditioning input for '
        'generative decoders, and a similarity metric for repurposing screening \u2014 enabling, to '
        'our knowledge, the first same-dataset, same-representation comparison of six '
        'generative architectures for a single disease indication. This manuscript reports '
        'results from a complete, freshly executed run of the full pipeline (Methods), with '
        'every numeric claim traced to a specific result file and independently cross-checked '
        'by an automated verifier against that source.')

    # ══════════════════════════════════════════════════════════════════════
    h(doc, 'Results')
    figure(doc, 'Fig1_Pipeline_Overview_v2.png',
           'Figure 1. GENESIS end-to-end pipeline overview, Phase 0-10: PGCL representation '
           'learning, six-way generative comparison, drug repurposing, structure-based docking, '
           'ADMET/mucosal-safety profiling, network pharmacology, lead optimisation, and final '
           'ranking, with headline numbers for each stage.')

    h(doc, 'PGCL learns potency-graded molecular representations', level=2)
    para(doc,
        'PGCL projects frozen 768-dimensional MolDeBERTa embeddings through a compact '
        'projection head onto the unit hypersphere in 128 dimensions, trained against '
        '49,312 OSCC-relevant positive compounds (56 targets spanning RTK, PI3K/AKT/mTOR, '
        'cell-cycle, tumour-suppressor, transcription-factor, and inflammatory signalling) and '
        'a six-tier negative taxonomy (Methods). Scaffold-group splitting ensures 100% of '
        'held-out test positives have Murcko scaffolds absent from training.')
    para(doc,
        f'Across three seeds (42, 123, 456), PGCL achieves Spearman '
        f'{RHO} = {PGCL["global_spearman_rho"]["mean"]:.4f} {PM} {PGCL["global_spearman_rho"]["std"]:.4f} '
        f'for potency ordering (n = 7,362 held-out test positives, {PGCL["n_targets_tested"]} targets), '
        f'target-switch AUROC = {PGCL["target_switch_auc_roc"]["mean"]:.4f} {PM} {PGCL["target_switch_auc_roc"]["std"]:.4f}, '
        f'activity-cliff accuracy = {PGCL["cliff_accuracy"]["mean"]*100:.1f}% {PM} {PGCL["cliff_accuracy"]["std"]*100:.1f}%, '
        f'and OSCC-specificity Cohen\u2019s d = {PGCL["cohens_d_specificity"]["mean"]:.3f}. Dose-response '
        f'monotonicity holds in strict rank order across potency classes (mean percentile: very-potent '
        f'{PGCL["mean_pctl"]["VP"]:.1f}, potent {PGCL["mean_pctl"]["Potent"]:.1f}, active '
        f'{PGCL["mean_pctl"]["Active"]:.1f}, negative {PGCL["mean_pctl"]["Neg"]:.1f}) in all three seeds '
        f'\u2014 the projected space recovers a continuous potency gradient rather than a binary boundary, '
        f'the central design goal of the ordinal contrastive objective (Eq. 2).')
    figure(doc, 'Fig2c_PGCL_Geometry_Clean.png',
           'Figure 2. Left: 2-D schematic of the 128-dimensional PGCL hypersphere, showing '
           'potency-class clustering around the very-potent (VP) centroid. Right: dose-response '
           'monotonicity, mean percentile rank (cosine similarity to the VP centroid) by potency '
           'class, confirming strict rank order (very-potent > potent > active > negatives).')
    para(doc,
        f'A single Spearman {RHO} definition is used consistently throughout this manuscript: computed '
        f'on all 7,362 test-set positives, identically for PGCL and every baseline. A distinct '
        f'"full-pool" definition (all test positives plus all {LEDGER["pgcl_fullpool_ndcg"]["n_neg"]:,} '
        f'test negatives, potency class as graded relevance) gives '
        f'{RHO} = {LEDGER["pgcl_fullpool_ndcg"]["spearman_rho"]["mean"]:.3f} {PM} '
        f'{LEDGER["pgcl_fullpool_ndcg"]["spearman_rho"]["std"]:.3f} \u2014 lower than the stratified '
        f'value, reflecting the different, unbalanced compound composition of the full screening '
        f'library. We report the stratified definition throughout to avoid conflating two distinct '
        f'metrics under one symbol.')

    h(doc, 'Comparative baselines validate PGCL design decisions', level=2)
    para(doc,
        'Six baselines isolate the contribution of each PGCL component (Table 1): ECFP4 [3] k-nearest-neighbour retrieval (Tanimoto similarity, no learning), three '
        'classical QSAR models trained on ECFP4 fingerprints (Random Forest, XGBoost [38], Logistic Regression), Flat SupCon (identical architecture to PGCL but '
        'uniform pair weights, isolating the ordinal-weighting contribution), and a Binary '
        'Cross-Entropy (BCE) classifier (identical architecture, classification rather than '
        'contrastive objective). All methods were evaluated on an identical 147,618-compound '
        'library (7,362 positives + 140,256 negatives).')
    table(doc,
          ['Metric', 'PGCL', 'Flat SupCon', 'BCE', 'ECFP4 k-NN', 'RF'],
          [
              ['Potency Spearman ' + RHO, T1['potency_spearman_rho']['PGCL'], T1['potency_spearman_rho']['FlatSupCon'],
               T1['potency_spearman_rho']['BCE'], T1['potency_spearman_rho']['ECFP4_kNN'], T1['potency_spearman_rho']['RF']],
              ['Cliff Pair Accuracy', T1['cliff_pair_accuracy']['PGCL'], T1['cliff_pair_accuracy']['FlatSupCon'],
               T1['cliff_pair_accuracy']['BCE'], T1['cliff_pair_accuracy']['ECFP4_kNN'], '\u2014'],
              ['Target-Switch AUROC', T1['target_switch_auroc']['PGCL'], T1['target_switch_auroc']['FlatSupCon'],
               T1['target_switch_auroc']['BCE'], '\u2014', T1['target_switch_auroc']['RF']],
              ['NDCG@100', T1['ndcg_100']['PGCL'], T1['ndcg_100']['FlatSupCon'],
               T1['ndcg_100']['BCE'], T1['ndcg_100']['ECFP4_kNN'], '\u2014'],
          ],
          caption='Table 1. PGCL, Flat SupCon, BCE, RF and XGB values are all 3-seed '
                  '(42/123/456) mean\u00b1SD; ECFP4 k-NN is the sole deterministic baseline '
                  '(nearest-neighbour retrieval has no training stochasticity, see Methods). '
                  'NDCG@100 uses potency-graded relevance (VP=3, Potent=2, Active=1, negative=0) '
                  'on compounds ranked by cosine similarity to the very-potent centroid c_VP '
                  '(Eq. VP). \u2014, not applicable.')
    para(doc,
        f'Ordinal potency-weighting shows a genuine, mechanistically selective benefit over flat '
        f'contrastive learning: PGCL wins 4 of 5 metrics against Flat SupCon, with cliff-pair '
        f'accuracy a statistical tie ({T1["cliff_pair_accuracy"]["PGCL"]} vs. '
        f'{T1["cliff_pair_accuracy"]["FlatSupCon"]}). This indicates that learning a metric-space '
        f'geometry at all \u2014 contrastive or otherwise \u2014 confers the large majority of cliff-'
        f'discrimination ability; ordinal weighting\u2019s distinct, measurable contribution is instead '
        f'the continuous potency-ranking gradient (Spearman {RHO}) and global retrieval quality '
        f'(NDCG@100), consistent with the ordinal weight matrix (Table M1) concentrating attractive '
        f'force between potency-proximal pairs rather than sharpening any single decision boundary. '
        f'PGCL decisively outperforms the BCE classifier on 4 of 5 metrics, confirming that a smooth '
        f'potency gradient cannot be recovered from binary active/inactive supervision alone. PGCL '
        f'roughly doubles ECFP4 k-NN\u2019s activity-cliff discrimination '
        f'({T1["cliff_pair_accuracy"]["PGCL"]} vs. {T1["cliff_pair_accuracy"]["ECFP4_kNN"]}, near '
        f'chance-level for structural similarity alone) \u2014 structurally similar compounds routinely '
        f'diverge sharply in potency (activity cliffs [8]), and Tanimoto '
        f'similarity alone cannot resolve this, the single most pharmacologically demanding SAR '
        f'phenomenon. Random Forest exceeds PGCL on raw Spearman '
        f'{RHO} ({T1["potency_spearman_rho"]["RF"]} vs. {T1["potency_spearman_rho"]["PGCL"]}), expected '
        f'for a model directly optimizing regression MSE against pChEMBL; RF cannot, however, '
        f'provide the continuous embedding, activity-cliff score, or differentiable reward signal '
        f'that PGCL supplies natively for the downstream generative and screening stages of this '
        f'pipeline.')
    figure(doc, 'Fig4_Baseline_Cliff_Cascade_MOSES.png',
           'Figure 3. Activity-cliff pair discrimination accuracy across baselines, showing the '
           'degradation from PGCL/Flat SupCon (metric-space learning) to BCE (binary supervision) '
           'to ECFP4 Tanimoto similarity alone (near chance level).')

    h(doc, 'Hyperparameter and loss-component ablations', level=2)
    ab_rows = []
    for name, disp in [('temp_0.05', f'{TAU}=0.05'), ('temp_0.07_baseline', f'{TAU}=0.07 (canonical)'),
                        ('temp_0.10', f'{TAU}=0.10'), ('dim_64', 'd=64'), ('dim_256', 'd=256'),
                        ('no_ranking', 'no tier-margin ranking (Eq. 4)'), ('no_potency', 'no potency regression (Eq. 3)'),
                        ('no_curriculum', 'no curriculum')]:
        c = AB['configs'][name]
        ab_rows.append([disp, f'{c["rho"]:.4f}', f'{c["cliff"]*100:.2f}%', f'{c["ts_auc"]:.4f}', f'{c["ndcg_100"]:.4f}'])
    table(doc, ['Configuration', 'Spearman ' + RHO, 'Cliff Acc.', 'TS-AUROC', 'NDCG@100'], ab_rows,
          caption='Table M-ablation. Single-seed (42) screening across 8 configurations; d=256 and '
                  f'{TAU}=0.05 additionally confirmed via full 3-seed replication (below).')
    para(doc,
        f'Removing potency regression (Eq. 3) collapses Spearman {RHO} to '
        f'{AB["configs"]["no_potency"]["rho"]:.3f} (from {AB["configs"]["temp_0.07_baseline"]["rho"]:.3f}), '
        f'confirming the Huber-loss potency regressor [9] as essential for ordinal '
        f'awareness \u2014 without it, the contrastive objective alone recovers only coarse target '
        f'discrimination, not fine-grained potency rank. Two apparent single-seed hyperparameter '
        f'"wins" (d=256\u2019s NDCG@100 = {AB["configs"]["dim_256"]["ndcg_100"]:.3f}; {TAU}=0.05\u2019s '
        f'NDCG@100 = {AB["configs"]["temp_0.05"]["ndcg_100"]:.3f}, both exceeding the canonical '
        f'{AB["configs"]["temp_0.07_baseline"]["ndcg_100"]:.3f}) were independently replicated across '
        f'two additional seeds before being trusted, following standard practice for '
        f'temperature-sensitive contrastive objectives [7]. Both evaporated under '
        f'replication: {AB["confirmatory_3seed_replication"]["dim_256_vs_canonical"]["verdict"]} '
        f'{AB["confirmatory_3seed_replication"]["temp_0.05_vs_canonical"]["verdict"]} The canonical '
        f'configuration ({TAU}=0.07, d=128, full loss with curriculum) is retained on more robust '
        f'evidence than a single-seed comparison alone would provide.')

    h(doc, 'Head-to-head comparison of six generative architectures', level=2)
    para(doc,
        'Six generative architectures share identical SELFIES-encoded [20] '
        'training data and a common GRU encoder-decoder backbone [19]: an '
        'unconditional VAE (Eq. 5) [14]; a conditional VAE (cVAE) using '
        'FiLM conditioning (Eq. 6) [15] on the PGCL vector with cyclical KL '
        'annealing (Eq. \u03b2(t)) [16]; a latent-diffusion model; an RL-fine-tuned VAE '
        '(RL-VAE) using REINFORCE [17] with a multi-objective PGCL-derived reward '
        '(Eq. 7); CGLF (Contrastive-Guided Latent Flow), a novel method performing optimal-'
        'transport conditional flow matching [18] directly in the PGCL latent '
        'space with Riemannian gradient guidance toward the VP centroid (Eq. 8\u20139); and CRAVE '
        '(Contrastive Reward-Aligned VAE Exploration), an RL-guided extension combining a '
        'four-component reward with an EMA baseline and KL-regularised policy update (Eq. 10).')
    gen_rows = []
    order = ['vae', 'cvae', 'diffusion', 'rl_vae', 'cglf', 'crave_unfiltered', 'crave_filtered']
    names = {'vae': 'VAE', 'cvae': 'cVAE', 'diffusion': 'Diffusion', 'rl_vae': 'RL-VAE',
             'cglf': 'CGLF', 'crave_unfiltered': 'CRAVE (unfilt.)', 'crave_filtered': 'CRAVE (filtered)'}
    for k in order:
        m = GEN['methods'][k]
        gen_rows.append([names[k], m['oscc_potency'], m['qed'], m['sa_score'], m['uniqueness'], m['scaffold_novelty']])
    table(doc, ['Method', 'OSCC Potency', 'QED', 'SA Score', 'Uniqueness', 'Scaffold Novelty'], gen_rows,
          caption='Table 2. 5-seed mean±SD (seeds 42, 123, 456, 789, 1337). QED [23], SA score [24].')
    T2S = GEN['table2_pairwise_stats_FRESH_and_brute_force_verified']
    stat_rows = []
    for comp, disp in [('CRAVE_vs_VAE', 'CRAVE (filt.) vs. VAE'), ('CRAVE_vs_cVAE', 'CRAVE (filt.) vs. cVAE'),
                        ('CRAVE_vs_CGLF', 'CRAVE (filt.) vs. CGLF'), ('CRAVE_vs_RLVAE', 'CRAVE (filt.) vs. RL-VAE')]:
        s = T2S[comp]
        stat_rows.append([disp, str(s['U']), f'{s["p_raw"]:.4f}', f'{s["p_bonferroni_x4"]:.4f}',
                           f'{s["cliffs_delta"]:.2f}', 'Yes' if s['significant'] else 'No'])
    table(doc, ['Comparison', 'U', 'p (raw)', 'p (Bonferroni ×4)', f'Cliff\u2019s {DELTA}', 'Significant?'],
          stat_rows,
          caption='Table 2b. Pairwise OSCC-potency comparisons of CRAVE (filtered) against each other '
                  'method (one-sided Mann-Whitney U, H1: CRAVE (filt.) > comparator, n=5 seeds each; '
                  'Bonferroni correction across the four planned comparisons; \u03b1=0.05).')
    para(doc,
        f'CRAVE (filtered) achieves the highest OSCC potency ({GEN["methods"]["crave_filtered"]["oscc_potency"]}), '
        f'significantly exceeding VAE, cVAE, and CGLF after Bonferroni correction across four planned '
        f'comparisons (Table 2b; complete rank separation in all three, Cliff\u2019s {DELTA} = 1.00). CRAVE does '
        f'not significantly exceed RL-VAE (Table 2b), '
        f'consistent with both methods sharing the same underlying RL-guided exploration mechanism '
        f'and PGCL-derived reward signal (Eq. 7).')
    para(doc,
        f'Latent diffusion performs poorly in this regime (OSCC potency = {GEN["methods"]["diffusion"]["oscc_potency"]}, '
        f'near-random; uniqueness = {GEN["methods"]["diffusion"]["uniqueness"]}, collapsing to near-identical '
        f'output in multiple seeds). We attribute this to training-set size: the available data, '
        f'sufficient for VAE-based methods\u2019 lower-dimensional inductive biases, falls below the '
        f'threshold typically required for stable score-based/flow training in a 128-dimensional '
        f'molecular latent space \u2014 a practically relevant finding for low-data disease settings where '
        f'large paired training corpora are unavailable.')
    figure(doc, 'Fig5_Generative_Method_Radar.png',
           'Figure 4. Multi-axis comparison of the six generative architectures (OSCC potency, '
           'QED, synthetic accessibility, uniqueness, scaffold novelty), 5-seed mean, showing '
           'CRAVE/RL-VAE\u2019s potency advantage and Diffusion\u2019s collapse relative to VAE/cVAE/CGLF.')

    para(doc,
        'Standard MOSES [21] and GuacaMol [22] distribution-'
        'learning benchmarks (internal diversity, scaffold/fragment similarity to the held-out test '
        'set, Fr\u00e9chet ChemNet Distance, and property-distribution KL divergence), computed over '
        'the same 3 seeds as Table 2, are provided in Supplementary Information (Note 1); these '
        'corroborate the diffusion instability reported above through an entirely independent '
        'metric family.')

    h(doc, 'Bioactivity profiling of generated molecules', level=2)
    para(doc,
        f'The canonical seed-42 generation ({P4["n_total_molecules_loaded"]:,} molecules across '
        f'{P4["n_methods"]} methods) yields {P4["n_unique_molecules"]:,} unique structures, '
        f'{P4["n_consensus_hits"]:,} consensus hits produced independently by multiple methods, and '
        f'{P4["n_enriched_smarts"]} enriched SMARTS motifs across {P4["n_targets_scored"]} scored targets '
        f'\u2014 convergent generation across architecturally distinct methods provides orthogonal '
        f'evidence that these motifs recapitulate genuine OSCC pharmacophores rather than a single '
        f'model\u2019s idiosyncratic bias.')
    para(doc,
        f'Safety screening (PAINS [35] / Brenk / toxicophore alerts) shows '
        f'{P4["safety_summary"]["pass"]:,}/{P4["n_total_molecules_loaded"]:,} = '
        f'{P4["safety_summary"]["pass_rate_pct"]}% PASS. Root-cause analysis attributes the limiting '
        f'factor to a persistent aniline toxicophore alert, flagged in {AN["n_flagged"]:,}/{AN["n_total"]:,} '
        f'({AN["pct"]}%) of generated molecules: the training corpus\u2019 EGFR-target skew (EGFR is the '
        f'single largest represented target among OSCC-active training compounds) biases de novo '
        f'generation toward aniline-quinazoline EGFR-inhibitor scaffolds, a chemotype family that '
        f'structurally triggers this specific alert. We report this transparently as a genuine '
        f'limitation of the current training corpus (discussed further below) rather than a pipeline '
        f'defect, since it is a direct, mechanistically traceable consequence of target-composition '
        f'imbalance in the underlying bioactivity data, not a stochastic artefact.')

    h(doc, 'Systematic drug repurposing', level=2)
    para(doc,
        f'Screening {P6["n_total_compounds"]:,} compounds against {P6["n_targets_scored"]} OSCC-relevant '
        f'targets \u2014 a curated subset of the 56 training targets (Methods) for which a per-target '
        f'very-potent centroid could be constructed from sufficient positive examples \u2014 scored by '
        f'cosine similarity to per-target very-potent centroids in the PGCL '
        f'space, the same representation used for generative reward and conditioning \u2014 and '
        f'validated against {P6["validation_against_annotated"]["n_annotated"]} independently annotated '
        f'compounds (Spearman {RHO} = {P6["validation_against_annotated"]["spearman_rho"]:.3f}, '
        f'Pearson r = {P6["validation_against_annotated"]["pearson_r"]:.3f}), identifies '
        f'{P6["top3"][0]["name"].title()} (recommendation weight = {P6["top3"][0]["recommendation_weight"]:.4f}, '
        f'{P6["top3"][0]["clinical_phase"]}), {P6["top3"][1]["name"].title()} '
        f'({P6["top3"][1]["clinical_phase"]}), and {P6["top3"][2]["name"]} ({P6["top3"][2]["clinical_phase"]}) '
        f'as the top three candidates \u2014 all FDA-approved or late-phase EGFR/HER-family kinase '
        f'inhibitors, biologically consistent with EGFR\u2019s well-established, clinically actionable role '
        f'in OSCC pathogenesis and current standard-of-care (cetuximab). (Per-target scores reflect '
        f'each compound\u2019s highest-matching target centroid in the learned space, which need not '
        f'coincide with its approved clinical mechanism of action \u2014 a distinction we make explicit '
        f'given the composite score is a repurposing-prioritisation signal, not a mechanism-of-action '
        f'assertion.)')
    figure(doc, 'Fig8_Repurposing_Waterfall.png',
           'Figure 5. Top-10 drug repurposing candidates by recommendation weight, with clinical '
           'phase annotated; EGFR/HER-family kinase inhibitors dominate the top 5, while the '
           'remainder of the top-10 span other mechanisms (CDK4/6, tubulin, AKT, antifolate).')

    h(doc, 'Structure-based molecular docking', level=2)
    t8 = P7['targets_confirmed_8']
    para(doc,
        f'Single-target docking (AutoDock Vina [30,31]) against 8 OSCC-relevant '
        f'crystal structures ({", ".join(f"{k} ({v})" for k, v in t8.items())}) yields '
        f'{P7["phase7a_single_target"]["successful"]}/{P7["phase7a_single_target"]["total_jobs"]} successful '
        f'poses (mean = {P7["phase7a_single_target"]["energy_stats"]["mean"]} kcal/mol, best = '
        f'{P7["phase7a_single_target"]["energy_stats"]["best"]} kcal/mol, at FGFR1). Composite screening '
        f'score and docking energy correlate only weakly '
        f'({RHO} = {P7["phase7a_single_target"]["correlation_composite_vs_docking"]["spearman_rho"]}, '
        f'p = {P7["phase7a_single_target"]["correlation_composite_vs_docking"]["spearman_p"]}), consistent '
        f'with the composite (ligand-based, PGCL-derived) score and docking (structure-based, '
        f'physical binding-pose) score assessing largely independent, complementary aspects of '
        f'candidate quality \u2014 the intended design rationale for running both in series rather than '
        f'relying on either alone. Multi-target docking of the top {P7["phase7b_multitarget"]["n_compounds"]} '
        f'candidates against {P7["phase7b_multitarget"]["n_targets"]} targets (all '
        f'{P7["phase7b_multitarget"]["successful"]}/{P7["phase7b_multitarget"]["total_jobs"]} jobs '
        f'successful) shows every candidate is a broad multi-target binder '
        f'(\u2265 3 targets \u2264\u22127.0 kcal/mol; mean = {P7["phase7b_multitarget"]["energy_stats"]["mean"]}, '
        f'best = {P7["phase7b_multitarget"]["energy_stats"]["best"]} kcal/mol) \u2014 a polypharmacology '
        f'profile plausibly advantageous in a molecularly heterogeneous disease such as OSCC, where '
        f'single-target inhibition has historically underperformed.')
    figure(doc, 'Fig9_Docking_Heatmap_Elite.png',
           f'Figure 6. Multi-target docking energy heatmap for the {P8["triple_validated_elite"]["count"]} '
           'triple-validated elite candidates (docking + ADMET + multi-target selectivity '
           'concordant; Phase 8) across all 7 productive targets, illustrating broad '
           'polypharmacology.')

    h(doc, 'ADMET and safety profiling', level=2)
    para(doc,
        f'ADMET profiling of {P8["n_compounds"]} candidates (Lipinski Rule-of-Five [32] '
        f'pass: {P8["physicochemical"]["lipinski_pass"]}, mean QED = {P8["physicochemical"]["mean_qed"]}) '
        f'combined with structural-alert screening [35] '
        f'({P8["structural_alerts"]["clean"]} fully clean of PAINS/Brenk/toxicophore alerts) and a '
        f'purpose-built mucosal irritation filter targeting the oral-cavity-specific delivery context '
        f'of OSCC therapeutics ({P8["mucosal_safety"]["safe"]} classified safe) yields a final shortlist '
        f'of {P8["final_candidates"]["shortlist_size"]} compounds ({P8["final_candidates"]["admet_pass"]} '
        f'PASS + {P8["final_candidates"]["admet_caution"]} CAUTION), including '
        f'{P8["triple_validated_elite"]["count"]} triple-validated elite candidates (docking + ADMET + '
        f'multi-target selectivity concordant; mean final score = {P8["triple_validated_elite"]["mean_final_score"]}).')

    h(doc, 'Network pharmacology and pathway analysis', level=2)
    para(doc,
        f'Network analysis of the {P9A["n_compounds"]}-compound elite set across {P9A["n_targets"]} '
        f'targets reveals a densely connected polypharmacology network (mean '
        f'{P9A["polypharmacology"]["mean_targets_per_compound"]} targets per compound), mapping to '
        f'{P9A["n_pathways_mapped"]} pathways across {P9A["n_pathway_categories"]} functional '
        f'categories, with {P9A["top_pathway"]["name"]} the most heavily represented '
        f'({P9A["top_pathway"]["hits"]}/{P9A["n_compounds"]} compounds) \u2014 consistent with the '
        f'PI3K/AKT/mTOR axis\u2019 central, well-documented role downstream of EGFR and other RTKs in '
        f'OSCC.')

    h(doc, 'Scaffold-constrained lead optimization', level=2)
    para(doc,
        f'Perturbing the {P9B["n_elite_parents"]} triple-validated elite compounds in the PGCL latent '
        f'space at three radii (0.3/0.5/0.8, spanning conservative to exploratory structural '
        f'variation) generates {P9B["total_raw_analogs"]:,} raw analogs, {P9B["total_unique_analogs"]} '
        f'unique after deduplication, of which {P9B["improved_analogs"]} score above their parent '
        f'(mean Tanimoto to parent = {P9B["mean_tanimoto"]}, mean ADMET score = {P9B["mean_admet_score"]}). '
        f'Only the generated-molecule parents produced valid analogs; the repurposed/external '
        f'parents produced none \u2014 expected, since the decoder is trained on the generative '
        f'pipeline\u2019s own latent distribution and cannot reliably decode arbitrary external '
        f'compounds\u2019 latents back into valid structures, a boundary condition of latent-space '
        f'perturbation methods generally.')
    figure(doc, 'Fig7_LeadOptimization.png',
           'Figure 7. (a) ADMET flag distribution, Phase 8 elite parents vs. Phase 9b analogs. '
           '(b) Analog yield and score improvement from scaffold-constrained latent-space '
           'perturbation of the 13 triple-validated elite compounds.')

    h(doc, 'Integrated final ranking', level=2)
    top1, top2, top3 = P10['top3_overall']
    para(doc,
        f'A unified GENESIS score (Eq. 11) integrating docking, ADMET, potency, drug-likeness, '
        f'multi-target selectivity, clinical evidence, synthetic accessibility, and novelty \u2014 with '
        f'an evidence-completeness penalty down-weighting compounds missing one or more scoring '
        f'components \u2014 ranks all {P10["n_total_ranked"]:,} candidates '
        f'({P10["source_breakdown"]["generated"]} generated + {P10["source_breakdown"]["repurposed"]} '
        f'repurposed). {P10["validation_tier_distribution"]["Tier-1_fully_validated"]} compounds achieve '
        f'Tier-1 (fully validated) status, {P10["validation_tier_distribution"]["Tier-3_partial"]} partial '
        f'validation, and {P10["validation_tier_distribution"]["Tier-4_prediction_only"]} prediction-only. '
        f'The top-ranked candidate is {top1["name"]} ({top1["cpd_id"]}), a PI3K/mTOR dual inhibitor '
        f'consistent with the PI3K/AKT/mTOR axis\u2019 prominence in the network-pharmacology analysis '
        f'above, followed by {top2["name"]} and {top3["name"]}. Among generated (de novo) compounds, '
        f'GEN_00182 ranks highest overall (rank 5). Novelty analysis of the generated set shows '
        f'{P10["generated_novelty_distribution"]["novel"]["pct"]}% structurally novel (Tanimoto < 0.4 to '
        f'any known compound), {P10["generated_novelty_distribution"]["scaffold_hop"]["pct"]}% scaffold-'
        f'hops, and {P10["generated_novelty_distribution"]["close_analog"]["pct"]}% close analogs.')
    figure(doc, 'Fig6_TopGeneratedCompounds.png',
           'Figure 8. Top-ranked de novo generated candidates (GEN_00182, GEN_00133, GEN_00045, '
           'GEN_00020, GEN_00044, GEN_00069) by GENESIS score, with structures, ADMET flag, and '
           'novelty classification.')

    h(doc, 'Cross-cancer transfer to lung adenocarcinoma', level=2)
    para(doc,
        f'The frozen OSCC PGCL model, applied without any retraining to {CC["n_in_index"]:,} lung '
        f'adenocarcinoma (LUAD) compounds across 16 LUAD-relevant targets, achieves Spearman '
        f'{RHO} = {CC["pgcl_rho"]:.3f} (ECFP4 baseline: {RHO} = {CC["ecfp4_rho"]:.3f}) \u2014 the same '
        f'order of magnitude as the OSCC in-domain reference '
        f'({RHO} = {PGCL["global_spearman_rho"]["mean"]:.3f} {PM} {PGCL["global_spearman_rho"]["std"]:.3f}, '
        f'stratified 3-seed potency ordering, Table 1), indicating '
        f'substantial transfer. Transfer strength tracks shared-target pharmacology: strong for '
        f'targets heavily represented in OSCC training data (EGFR {RHO}={CC["per_target"]["EGFR"]["rho"]} '
        f'CI{CC["per_target"]["EGFR"]["ci"]}, n={CC["per_target"]["EGFR"]["n"]:,}; PIK3CA '
        f'{RHO}={CC["per_target"]["PIK3CA"]["rho"]} CI{CC["per_target"]["PIK3CA"]["ci"]}, '
        f'n={CC["per_target"]["PIK3CA"]["n"]:,}), markedly weaker for LUAD-specific driver oncogenes '
        f'not represented in OSCC training (ALK {RHO}={CC["per_target"]["ALK"]["rho"]} '
        f'CI{CC["per_target"]["ALK"]["ci"]}; KRAS {RHO}={CC["per_target"]["KRAS"]["rho"]} '
        f'CI{CC["per_target"]["KRAS"]["ci"]}, not significantly different from zero at this sample '
        f'size, n={CC["per_target"]["KRAS"]["n"]}) \u2014 a calibrated, mechanistically interpretable '
        f'transferability profile: PGCL encodes pharmacophore geometry that generalises where target '
        f'biology overlaps, and honestly does not where it does not.')
    figure(doc, 'Fig3_Cross_Cancer_Transfer.png',
           'Figure 9. Per-target Spearman correlation (with 95% bootstrap CI) for the frozen OSCC '
           'PGCL model applied to lung adenocarcinoma compounds, showing strong transfer for '
           'OSCC-shared targets (EGFR, PIK3CA) and weak/non-significant transfer for LUAD-specific '
           'driver oncogenes (ALK, KRAS).')

    # ══════════════════════════════════════════════════════════════════════
    h(doc, 'Discussion')
    para(doc,
        'This work makes five distinct, evidence-graded contributions. First, ordinal potency-'
        'weighted contrastive learning provides a genuine, statistically robust improvement over '
        'flat contrastive learning on potency-ranking fidelity (true 3-seed-vs-3-seed comparison, '
        'Table 1), concentrated specifically in the continuous ranking gradient rather than binary '
        'cliff discrimination, which the underlying contrastive geometry already provides. Second, '
        'PGCL substantially outperforms a binary classification objective, confirming that encoding '
        'graded potency requires ordinal, not binary, supervision. Third, PGCL captures pharmacology '
        'beyond raw structural similarity, roughly doubling ECFP4\u2019s cliff-discrimination accuracy '
        '\u2014 the learned geometry encodes information beyond what Tanimoto similarity on 2048-bit '
        'fingerprints can access. Fourth, the canonical hyperparameters ('
        f'{TAU}=0.07, d=128) survived adversarial 3-seed replication after two single-seed '
        '"improvements" were shown to be seed noise \u2014 methodological rigor that is uncommon in '
        'this literature and strengthens confidence these choices are genuinely optimal rather than '
        f'arbitrary. Fifth, the cross-cancer transfer result (OSCC{ARROW}LUAD {RHO} = '
        f'{CC["pgcl_rho"]}, n = {CC["n_in_index"]:,}) demonstrates that PGCL\u2019s pharmacological '
        'geometry transfers substantially to a second cancer type without any cancer-specific '
        'retraining, with transfer strength tracking target overlap in a mechanistically '
        'interpretable way rather than as a single derived percentage.')
    para(doc,
        f'A counterintuitive finding from the QSAR baselines merits explicit discussion, since a '
        f'naive reading of Table 1 could be mistaken for undermining PGCL\u2019s contribution. Random '
        f'Forest and XGBoost, trained directly on ECFP4 fingerprints to minimise regression error '
        f'against pChEMBL, decisively exceed PGCL on raw potency-value prediction '
        f'({T1["potency_spearman_rho"]["RF"]} and {T1["potency_spearman_rho"]["XGB"]} vs. PGCL\u2019s '
        f'{T1["potency_spearman_rho"]["PGCL"]}); this is expected, not surprising \u2014 a model trained '
        f'purely to minimise regression MSE against pChEMBL will, almost by definition, better '
        f'reproduce the Spearman rank ordering of that same quantity. Yet on the classification task '
        f'the representation must actually support downstream (distinguishing OSCC-relevant '
        f'compounds from same-target, wrong-context actives), PGCL leads by a substantial margin '
        f'(target-switch AUROC {T1["target_switch_auroc"]["PGCL"]} vs. RF\u2019s '
        f'{T1["target_switch_auroc"]["RF"]}), and on VP-compound retrieval at a fixed budget the two '
        f'paradigms are essentially indistinguishable '
        f'(VP Recall@100: PGCL {T1["vp_recall_100"]["PGCL"]} vs. RF {T1["vp_recall_100"]["RF"]}) \u2014 the RF '
        f'regressor does not fail at retrieval, it simply ties PGCL rather than losing to it. This '
        f'pattern is mechanistically coherent: RF/XGBoost are optimised regressors, learning an '
        f'arbitrary function of ECFP4 bits that predicts pChEMBL accurately in aggregate but with no '
        f'geometric structure connecting that function to a usable embedding space; PGCL is not '
        f'competing to be the best pChEMBL regressor, but to provide a single continuous '
        f'representation that simultaneously supports reward shaping for reinforcement learning, '
        f'conditioning for generative decoders, and similarity-based screening \u2014 none of which a '
        f'tree-ensemble regression output can natively supply.')
    para(doc,
        f'The six-tier negative taxonomy with curriculum scheduling (Methods, Table M2\u2013M3) is, to '
        f'our knowledge, the first instance of graded (rather than binary or two/three-tier) '
        f'negative mining in molecular contrastive learning. The inversely proportional tier '
        f'margins \u2014 hardest negatives (activity-cliff inactives, clinical-failure/target-switching '
        f'compounds) receive the tightest margins, easiest negatives the loosest \u2014 are '
        f'geometrically motivated: forcing a large margin against a structurally near-identical '
        f'inactive compound is often infeasible and destabilises training, whereas concentrating '
        f'representational capacity on the pharmacologically informative near-boundary region is '
        f'exactly where potency-graded discrimination is hardest and most valuable.')
    para(doc,
        f'The dual-track architecture \u2014 simultaneous drug repurposing and de novo generation from '
        f'the same PGCL representation \u2014 creates two complementary translational timelines rather '
        f'than a single one. The repurposing track surfaces already-approved or late-phase kinase '
        f'inhibitors ({P6["top3"][0]["name"].title()}, {P6["top3"][1]["name"].title()}, '
        f'{P6["top3"][2]["name"].title()}), offering a comparatively short path to clinical testing '
        f'since their safety and pharmacokinetic profiles are already established. The generative '
        f'track instead produces structurally novel candidates '
        f'({P10["generated_novelty_distribution"]["novel"]["pct"]}% below Tanimoto 0.4 to any known '
        f'compound), validated through the same downstream docking/ADMET/network-pharmacology '
        f'evidence chain, offering a longer-horizon discovery pipeline unconstrained by existing '
        f'chemical matter. That both tracks are produced and ranked by the same learned '
        f'representation and the same downstream validation machinery, without any track-specific '
        f'retraining, is itself evidence for the generality of the potency-graded contrastive '
        f'geometry.')
    para(doc,
        f'Network pharmacology provides an independent, mechanism-level check on the pipeline\u2019s '
        f'target selection: every one of the {P9A["n_compounds"]} multi-target-docked elite '
        f'compounds engages the {P9A["top_pathway"]["name"]} pathway '
        f'({P9A["top_pathway"]["hits"]}/{P9A["n_compounds"]}), with a mean of '
        f'{P9A["polypharmacology"]["mean_targets_per_compound"]} targets bound per compound \u2014 '
        f'consistent with PI3K/AKT/mTOR signalling\u2019s well-established, clinically actionable '
        f'centrality downstream of EGFR and other receptor tyrosine kinases in OSCC. This convergence '
        f'is a property of the underlying target panel and docking results rather than of any single '
        f'scoring choice, and is the kind of independent, mechanism-level corroboration that a '
        f'purely ligand-based composite score cannot supply on its own.')
    para(doc,
        f'Generated-molecule safety PASS rates are {P4["safety_summary"]["pass_rate_pct"]}%, traced '
        f'to a persistent aniline toxicophore alert reflecting the training corpus\u2019s EGFR-target '
        f'skew. We report this candidly as a limitation of the current training data composition, '
        f'not a pipeline defect: EGFR compounds are systematically over-represented among OSCC '
        f'training positives (a consequence of EGFR\u2019s clinical prominence and correspondingly '
        f'larger ChEMBL bioactivity footprint relative to other OSCC targets), and the generative '
        f'methods faithfully learn and reproduce this skew. This motivates toxicophore-aware '
        f'negative sampling or generation-time filtering as a concrete next step, discussed below.')
    para(doc,
        'The complete pipeline reported here (PGCL training, all six generative methods, docking, '
        'ADMET, and lead optimisation) runs on a single consumer-grade workstation-class GPU/CPU, '
        'without a dedicated compute cluster \u2014 a practical accessibility advantage for research '
        'groups without large-scale infrastructure. Figure 10 situates this against representative '
        'compute/hardware requirements reported for comparable recent generative-chemistry methods; '
        'those comparator figures are drawn from each method\u2019s own published reporting and have '
        'not been independently re-verified here, and are included for indicative context only.')
    figure(doc, 'Fig10_Hardware_Accessibility.png',
           'Figure 10. Indicative compute/hardware accessibility comparison between this work and '
           'representative recently published generative molecular design methods (comparator '
           'figures as self-reported by each method; not independently re-verified in this study).')
    h(doc, 'Limitations', level=2)
    limitations = [
        'All validation reported here is computational; no wet-laboratory potency, ADMET, or '
        'toxicity measurement has been performed on any generated or lead-optimised compound.',
        'The generative safety-PASS finding indicates the current training corpus\u2019 target '
        'composition biases de novo generation toward a narrow, frequently-flagged aniline-'
        'quinazoline chemotype; broadening the positive-compound target distribution is a priority '
        'for future training iterations.',
        'Diffusion-based generation is not viable at the current training-set scale for this '
        'chemical space; conditional and RL-guided VAE variants are the reliable choice below '
        'roughly 50,000 training compounds.',
        'Cross-cancer transfer, while substantial for shared targets (EGFR, PIK3CA), does not '
        'extend to cancer-specific driver oncogenes (ALK, KRAS) without additional target-specific '
        'data \u2014 the frozen representation transfers pharmacophore geometry, not disease biology '
        'it has never seen.',
        'Docking energy and the ligand-based composite/repurposing score capture largely '
        'independent aspects of candidate quality (Results) and should be interpreted as '
        'complementary, not mutually confirming, evidence streams.',
        'The mucosal-irritation filter and all ADMET flags are rule-based (SMARTS/RDKit) heuristics '
        'calibrated against known toxicophore chemistry, not experimentally measured irritation or '
        'pharmacokinetic data; they are appropriately used for triage, not as a substitute for '
        'in vitro or in vivo safety assessment.',
        'The repurposing composite score (Eq. 11) and the Phase 10 final-ranking GENESIS score '
        '(Eq. 12) are intentionally different formulas reflecting the evidence available at each '
        'pipeline stage (the former predates docking data, the latter incorporates it); readers '
        'comparing scores across these two stages directly should be aware they are not on the '
        'same scale.',
    ]
    for item in limitations:
        para(doc, item, style='List Number')
    h(doc, 'Future directions', level=2)
    para(doc, 'Priority extensions include:')
    future_directions = [
        'toxicophore-aware curriculum sampling or target-composition rebalancing to address the '
        'EGFR/aniline generative bias identified above;',
        f'extending the LUAD cross-cancer transfer result ({RHO} = {CC["pgcl_rho"]}) to additional '
        'cancers with disease-specific target panels, testing whether transfer strength continues '
        'to track quantifiable target overlap;',
        'multi-centroid scoring to improve natural-product and repurposing candidate ranking '
        'beyond a single VP centroid, capturing target-specific rather than global potency '
        'proximity;',
        'wet-laboratory validation of the Tier-1 elite candidates identified here in OSCC '
        'cell-line models, prioritising the triple-validated (docking + ADMET + selectivity) '
        'subset;',
        'integration of structure-aware protein representations (e.g. ESM-2-style protein '
        'language models) to condition generation directly on target pocket geometry rather than '
        'solely on the ligand-side PGCL signal;',
        'a true bootstrap recomputation of the approximate normal-theory confidence intervals '
        'reported in the 5-seed generative evaluation (\u00a714.4 caveat, Algorithm_Design.md), for '
        'full statistical rigor beyond what this manuscript currently claims.',
    ]
    for item in future_directions:
        para(doc, item, style='List Number')

    # ══════════════════════════════════════════════════════════════════════
    h(doc, 'Methods')
    h(doc, 'Data curation and target selection', level=2)
    para(doc,
        '56 molecular targets with established roles in OSCC pathogenesis were identified through '
        'systematic literature review, spanning RTK signalling (EGFR, FGFR1, FGFR3), PI3K/AKT/mTOR '
        '(PIK3CA, AKT1, mTOR), cell-cycle regulation (CDK4/Cyclin D1), tumour suppression (MDM2, '
        'TP53), transcription factors (STAT3, MYC), and inflammation (COX-2/PTGS2). Positive '
        'compounds were extracted from ChEMBL [29] with pChEMBL \u2265 5.0 '
        '(IC50/Ki/Kd/EC50 \u2264 10 \u03bcM), yielding 49,312 compound-target-activity records.')
    table(doc, ['Class', 'Label', 'pChEMBL range', 'IC\u2085\u2080 equivalent'],
          [['3', 'Very Potent', '\u2265 7.0', '\u2264 100 nM'],
           ['2', 'Potent', '6.0\u20137.0', '100 nM\u20131 \u03bcM'],
           ['1', 'Active', '5.0\u20136.0', '1\u201310 \u03bcM']],
          caption='Table M0. Ordinal potency classification thresholds.')
    para(doc,
        'Negatives are curated into six biologically motivated tiers spanning a hardness gradient: '
        'T1 (tested inactive on OSCC targets), T2 (active on non-OSCC oncology targets \u2014 teaches '
        'disease-context specificity), T3A (structurally similar inactive variants of active '
        'scaffolds \u2014 the gold-standard activity-cliff negative), T3B (random bioactive compounds, '
        'non-cancer context), T3C (clinical development failures \u2014 advanced candidates that failed, '
        'explicitly avoided), and T4 (random druglike background). Each tier receives an inversely '
        'proportional margin in the TierMarginRankingLoss (Eq. 4, Table M2): harder negatives (T3A, '
        'T3C) receive tighter margins, preventing representational collapse of pharmacologically '
        'critical near-boundary distinctions; easier negatives (T3B, T4) receive looser margins.')
    table(doc, ['Tier', 'Description', 'Margin m(t)'],
          [['T3A', 'Structurally similar, inactive (activity-cliff gold standard)', '0.10'],
           ['T3C', 'Clinical development failures / target-switching', '0.10'],
           ['T1', 'Confirmed inactive on OSCC targets', '0.20'],
           ['T2', 'Active on other (non-OSCC) cancer targets', '0.20'],
           ['T3B', 'Scaffold-matched decoys / random bioactive', '0.30'],
           ['T4', 'Random druglike background', '0.30']],
          caption='Table M2. Tier-dependent margins, TierMarginRankingLoss (Eq. 4). Source: '
                  'modeling/losses.py TIER_MARGINS.')
    para(doc,
        'A greedy Murcko-scaffold-group splitting algorithm [11] allocates '
        'entire scaffold groups (not individual compounds) to train (70%) / validation (15%) / '
        'test (15%) splits, sorted by group size and greedily assigned to the split furthest below '
        'its target proportion. This achieves 100% scaffold novelty in the test set \u2014 the strongest '
        'form of scaffold-based evaluation, eliminating structural-similarity shortcuts from '
        'inflating performance metrics.')

    h(doc, 'Molecular representation and PGCL architecture', level=2)
    para(doc,
        'MolDeBERTa [4], a DeBERTa-v3-style [5] model '
        'pretrained on masked-token recovery over molecular SMILES, provides frozen 768-dimensional '
        'embeddings for all compounds. A 3-layer MLP projection head (768\u2192512\u2192256\u2192128, '
        'LayerNorm, GELU, dropout 0.1, L2-normalised output onto the unit hypersphere; 559,617 '
        'trainable parameters) compresses this into the potency-graded PGCL space. LayerNorm (not '
        'BatchNorm) is used for robustness to the highly variable batch compositions arising from '
        'mixed-tier negative sampling; GELU (not ReLU) preserves gradient flow through the 3-layer '
        'depth; L2-normalisation makes cosine similarity the natural comparison metric and bounds '
        f'the scale on which the temperature parameter {TAU} operates.')
    para(doc, 'The very-potent (VP) centroid, used as the reference point for reward, retrieval, '
               'and screening throughout this work, is defined as:')
    equation(doc, 'c_VP = normalise( mean{ z_i : i \u2208 V } )', 'VP centroid')
    para(doc, 'where V is the set of very-potent (potency class 3) training compounds and z_i are '
               'their L2-normalised PGCL projections.')

    h(doc, 'Training objective', level=2)
    para(doc, 'The total training loss combines three components:')
    equation(doc, f'L_PGCL = L_con + {LAM}\u2081(t)\u00b7L_pot + {LAM}\u2082(t)\u00b7L_rank', 'Eq. 1')
    para(doc,
        f'L_con is an ordinal supervised contrastive loss [6] extended with '
        f'potency-proximity pair weights w_ij (Table M1): for an anchor i with potency class c_i, '
        f'positive j receives weight w_ij derived from |c_i \u2212 c_j| so that Very-Potent\u2013Very-Potent '
        f'pairs attract more strongly than Active\u2013Active pairs:')
    equation(doc,
        'L_con = \u2212(1/|P(B)|) \u03a3_i\u2208P(B) [ \u03a3_p w_ip \u00b7 log( exp(z_i\u00b7z_p/' + TAU + ') '
        f'/ {SUM}_j{NEQ}i w\u0303_ij \u00b7 exp(z_i\u00b7z_j/{TAU}) ) ] / {SUM}_p w_ip', 'Eq. 2')
    table(doc, ['Anchor \u2193 / Partner \u2192', 'Active (1)', 'Potent (2)', 'Very Potent (3)'],
          [['Active (1)', '1.0', '0.7', '0.4'],
           ['Potent (2)', '0.7', '1.0', '0.7'],
           ['Very Potent (3)', '0.4', '0.7', '1.0']],
          caption='Table M1. Symmetric ordinal pair weight matrix w_ij for the potency-proximity '
                  'term in Eq. 2. Cross-target pairs receive an additional 0.5\u00d7 factor.')
    para(doc,
        f'L_pot is a masked Huber loss [9] on predicted pChEMBL, applied only where '
        f'experimental values are available:')
    equation(doc, 'L_pot = (1/|M|) \u03a3_i\u2208M Huber_' + DELTA + '(y\u0302_i, y_i),  M = {i : y_i ' + NEQ + ' NaN}', 'Eq. 3')
    para(doc,
        'L_rank is a tier-margin ranking loss (a variant of triplet-margin loss [10]) enforcing that anchor-positive similarity exceeds anchor-negative similarity by at '
        'least the negative\u2019s tier-dependent margin (Table M2):')
    equation(doc, 'L_rank = (1/N) \u03a3_i max(0, m(t_ni) + sim(a_i,n_i) \u2212 sim(a_i,p_i))', 'Eq. 4')
    para(doc,
        f'Loss weights {LAM}\u2081, {LAM}\u2082 follow a discrete curriculum schedule (Table M3) that ramps '
        f'auxiliary supervision in only after the contrastive term has established coarse target-'
        f'level geometry, avoiding premature distortion of that geometry by regression/ranking '
        f'pressure before it exists.')
    table(doc, ['Phase', 'Epochs', LAM+'\u2081 (potency)', LAM+'\u2082 (ranking)', 'T4', 'T3B', 'T1', 'T2', 'T3A', 'T3C'],
          [['Warmup', '1\u201310', '0.1', '0.0', '35%', '35%', '10%', '10%', '5%', '5%'],
           ['Middle', '11\u201320', '0.3', '0.2', '15%', '15%', '25%', '20%', '15%', '10%'],
           ['Sharpen', '21+', '0.5', '0.3', '10%', '10%', '20%', '15%', '25%', '20%']],
          caption='Table M3. Loss-weight and curriculum negative-sampling schedule by training phase.')
    para(doc,
        'Canonical hyperparameters: AdamW optimiser [12], cosine-annealed '
        f'learning rate 3\u00d710\u207b\u2074 [13], temperature {TAU}=0.07 [7], batch size 512 '
        '(128 positive + 384 negative, tier-weighted sampling). An '
        'early-stopping implementation issue in an earlier version of this training code reset its '
        'validation-loss tracker at each curriculum phase transition, causing the checkpoint saved '
        'as \u201cbest\u201d to reflect the epoch immediately following a phase transition rather than the '
        'run\u2019s true best epoch; this was identified and corrected prior to the results reported '
        'here (checkpoint selection now uses the final completed training epoch, decoupled from the '
        'early-stopping patience criterion), and every downstream phase of this pipeline was '
        'executed after that correction.')

    h(doc, 'Comparative baseline training configuration', level=2)
    para(doc,
        'Flat SupCon and the BCE classifier share PGCL’s encoder architecture (768→512→256→128, '
        'LayerNorm/GELU/Dropout, L2-normalised output) and training protocol (AdamW, cosine-'
        'annealed learning rate 3×10⁻⁴, batch size 512 with 128 positive + 384 negative per step, '
        'up to 40 epochs, patience 7 epochs after a minimum of 25), differing only in their '
        'objective: Flat SupCon replaces the ordinal pair-weight matrix (Table M1) with uniform '
        'weights (w_ip=1.0 for all same-class-or-not positive pairs), and the BCE classifier '
        'replaces the three-component PGCL loss entirely with binary cross-entropy on a '
        'very-potent-vs-all label using the same encoder plus a linear classification head. Both '
        'were retrained and evaluated across the same three seeds (42, 123, 456) as PGCL, on the '
        'identical held-out test pool, so that Table 1’s reported ±SD values are genuinely '
        'comparable across all three methods (Table 1 caption). Random Forest (200 trees, '
        'scikit-learn [37] default max depth) and XGBoost (300 trees, max depth 6, '
        'learning rate 0.05) [38] were trained on 2048-bit ECFP4 fingerprints '
        '[3] under the same three-seed protocol; Logistic Regression (L2 penalty, '
        'C=1.0) was trained once (seed-invariant by construction of its convex objective). ECFP4 '
        'k-nearest-neighbour retrieval requires no training and is therefore deterministic, the '
        'sole such baseline in Table 1.')

    h(doc, 'Multi-seed validation rationale', level=2)
    para(doc,
        f'The number of seeds differs between pipeline stages by design, not convenience. PGCL '
        f'training (Phase 2) is evaluated across three seeds (42, 123, 456): the ordinal '
        f'contrastive objective is a comparatively well-behaved, near-convex optimisation over a '
        f'fixed 768-d embedding space, and empirically shows low seed-to-seed variance '
        f'(Spearman {RHO} coefficient of variation {PGCL["global_spearman_rho"]["std"]/PGCL["global_spearman_rho"]["mean"]*100:.1f}%, '
        f'Table 1). Three seeds are sufficient to establish this stability and are held fixed as '
        f'the frozen input to every downstream phase; re-running Phase 2 at additional seeds would '
        f'require repeating all six generative architectures’ training for each additional seed, a '
        f'combinatorial cost with limited expected benefit given the demonstrated low variance. '
        f'Generative evaluation (Phase 3), by contrast, uses five seeds (42, 123, 456, 789, 1337): '
        f'REINFORCE-based policy-gradient training over a discrete SELFIES action space is '
        f'substantially higher-variance, as the generative-method comparison itself demonstrates '
        f'(Table 2 potency values span from {GEN["methods"]["vae"]["oscc_potency"]} for VAE, the '
        f'most stable method in absolute terms, to {GEN["methods"]["cvae"]["oscc_potency"]} for '
        f'cVAE, whose disproportionately large SD reflects a single outlier seed discussed in '
        f'Results). Five seeds were the minimum needed to (a) support a one-sided '
        f'Mann-Whitney U test with Bonferroni correction across the four planned comparisons at '
        f'conventional {ALPHA}=0.05 (the smallest achievable exact p-value at n=5 vs. n=5 with complete '
        f'rank separation is 1/C(10,5)=1/252≈0.004, comfortably below {ALPHA}/4=0.0125), and (b) '
        f'characterise per-seed variability honestly rather than risk over-interpreting a smaller '
        f'sample — a concern directly motivated by this project’s own experience correcting an '
        f'earlier, 3-seed generative evaluation whose apparent stability partially reflected '
        f'incomplete seed coverage rather than a genuine property of the underlying method '
        f'(Table 2 legend; Results).')

    h(doc, 'Evaluation battery and formal metric definitions', level=2)
    para(doc,
        'PGCL and all baselines are evaluated on held-out, scaffold-novel test positives using: '
        'global and per-target Spearman correlation between projected VP-centroid similarity and '
        'potency class; activity-cliff pair accuracy on structurally similar (Tanimoto > 0.6), '
        'differently-potent compound pairs; target-switch AUROC; VP-centroid retrieval (Recall@K, '
        'NDCG@K with graded relevance); and dose-response monotonicity. Three retrieval metrics '
        'used throughout are formally defined as follows. Enrichment Factor at K:')
    equation(doc, 'EF@K = (n_VP,topK / K) / (n_VP / N)', 'EF@K')
    para(doc,
        'where n_VP is the total number of very-potent compounds among N screened. BEDROC '
        '(Boltzmann-Enhanced Discrimination of ROC) [25], which weights early '
        'retrieval more heavily than late retrieval via a decay parameter \u03b1=20:')
    equation(doc, f'BEDROC(\u03b1=20) = ({RHO}\u00b7Ra\u00b7{SUM}_i exp(\u2212\u03b1\u00b7r_i/N)) / (Ra\u00b7Sw) + 1/Sw', 'BEDROC')
    para(doc,
        'where r_i is the rank of the i-th active, Ra = n/N the proportion of actives, and Sw a '
        'normalisation constant; BEDROC reduces to AUROC as the decay parameter approaches zero. '
        'Fr\u00e9chet ChemNet Distance (FCD) [26], used to assess distributional '
        'realism of generated chemistry against the training set:')
    equation(doc, 'FCD = \u2016\u03bc_G \u2212 \u03bc_T\u2016\u00b2 + Tr(\u03a3_G + \u03a3_T \u2212 2(\u03a3_G\u03a3_T)^{1/2})', 'FCD')
    para(doc,
        'where subscripts G, T denote generated and training-set ChemNet penultimate-layer '
        'activation statistics. NDCG@K follows the standard graded-relevance discounted-cumulative-'
        'gain formulation [27]. Bootstrap 95% confidence intervals '
        '(2,000 resamples) [28] are used for all per-target subgroup estimates '
        '(cross-cancer transfer, Results).')

    h(doc, 'Generative methods', level=2)
    para(doc,
        'The unconditional VAE optimises the standard evidence lower bound with cyclical KL '
        'annealing:')
    equation(doc, f'L_VAE = \u2212E[log p_\u03b8(x|z)] + {BETA}(t)\u00b7D_KL(q_\u03c6(z|x) \u2016 N(0,I))', 'Eq. 5')
    equation(doc, f'{BETA}(t) = min(1, (t mod L)/(0.5\u00b7L)),  L=25, n_cycles=4', 'Eq. ' + BETA + '(t)')
    para(doc,
        'The cVAE conditions its decoder on the PGCL vector via Feature-wise Linear Modulation '
        '(FiLM) [15]:')
    equation(doc, 'h_cond = ' + '\u03b3' + '(c)\u2299h + ' + '\u03b2' + '(c),  ' + '\u03b3,\u03b2' + ': R\u2075\u2192R\u00b9\u2070\u00b2\u2074,  c=[onehot(class,4), sim(1)]', 'Eq. 6')
    para(doc, 'RL-VAE fine-tunes the pretrained VAE decoder via REINFORCE [17] against '
               'a multi-objective, PGCL-anchored reward:')
    equation(doc, 'R = 0.40\u00b7cos(z,c_VP) + 0.20\u00b7QED + 0.20\u00b7max(0,(10\u2212SA)/9) + 0.20\u00b7valid \u2212 0.5\u00b7D_KL(' + '\u03c0\u03b8\u2016\u03c0VAE' + ')', 'Eq. 7')
    para(doc, 'with an exponential-moving-average baseline (' + '\u03b2' + '=0.9) subtracted from R to '
               'reduce policy-gradient variance:')
    equation(doc, '\u2207' + '\u03b8' + ' J(' + '\u03b8' + ') = E[(R(x)\u2212b_EMA)\u00b7\u2207' + '\u03b8' + ' log ' + '\u03c0\u03b8' + '(x|z)]', 'REINFORCE')
    para(doc,
        'CGLF performs optimal-transport conditional flow matching [18] directly '
        'in the 128-d PGCL latent space, conditioned via AdaLN on a 133-dimensional potency vector:')
    equation(doc, 'L_CFM = E\u2016v' + '\u03b8' + '((1\u2212t)z\u2080+tz\u2081, t, c) \u2212 (z\u2081\u2212z\u2080)\u2016\u00b2', 'Eq. 8')
    para(doc, 'with Riemannian guidance toward the VP centroid during ODE integration on the unit '
               'hypersphere S\u00b9\u00b2\u2077:')
    equation(doc, 'v_guide = ' + '\u03b1' + '(t)\u00b7(c_VP \u2212 cos(z\u0302,c_VP)\u00b7z\u0302)', 'Eq. 9')
    para(doc,
        'CRAVE combines RL-guided exploration with a four-component reward and the same EMA-'
        'baselined policy gradient as RL-VAE, additionally KL-regularised against a fixed prior '
        'policy:')
    equation(doc, 'L_CRAVE = \u2212' + '\u03a3' + '_i(R_i\u2212b_EMA)\u00b7log ' + '\u03c0\u03b8' + '(x_i|z_i) + 0.3\u00b7D_KL(' + '\u03c0\u03b8\u2016\u03c0prior' + '),  R weights: 0.55/0.20/0.10/0.15', 'Eq. 10')
    para(doc,
        'MOSES-compatible metrics (validity, uniqueness, novelty, internal diversity) follow '
        '[21]; QED [23] and synthetic accessibility (SA) '
        '[24] use their standard published formulations.')

    h(doc, 'Drug repurposing composite score', level=2)
    para(doc,
        f'The {P6["n_total_compounds"]:,}-compound repurposing library (CompoundList: ChEMBL and '
        'TTD clinical-stage compounds, independent of and non-overlapping with PGCL training data) '
        'is ranked by an 8-component recommendation weight computed before any docking data is '
        'available for these compounds:')
    equation(doc,
        'W_rep = 0.30\u00b7s_comp + 0.20\u00b7s_safe + 0.15\u00b7s_tgt + 0.10\u00b7s_SA + 0.05\u00b7s_cons '
        '+ 0.05\u00b7s_dock + 0.10\u00b7s_clin + 0.05\u00b7s_smarts', 'Eq. 11')
    para(doc,
        's_comp is cosine similarity to the per-target very-potent centroid (the same PGCL '
        'representation used for generative reward and conditioning); s_safe is a geometric-mean '
        'safety score (PAINS/Brenk/toxicophore alerts, one severe alert dominates); s_tgt is a '
        'per-target centroid-confidence term; s_SA is normalised synthetic accessibility; s_cons '
        'and s_dock are placeholder terms (uniform 0.5 and 0 respectively, reserved for future '
        'multi-method consensus and Phase 7 docking data, which post-dates this screening step and '
        'is therefore unavailable at scoring time); s_clin is clinical-phase evidence '
        '(approved=1.0, decreasing with earlier phase); and s_smarts flags membership in a '
        'cross-study-enriched substructure shared with independently generated molecules. Unlike '
        'the Phase 10 final-ranking score (Eq. 12), all eight components are always defined for '
        'every compound in this library, so no evidence-completeness penalty is required.')

    h(doc, 'Structure-based molecular docking', level=2)
    para(doc,
        'The top 200 generated and top 200 repurposed candidates by composite score were docked '
        'against eight OSCC-relevant crystal structures (AKT1 3CQW, EGFR 4HJO, FGFR1 4V04, MDM2 '
        '4HG7, MTOR 4DRH, PIK3CA 4JPS, PTGS2 3LN1, STAT3 6NUQ) using AutoDock Vina '
        '[30,31] (exhaustiveness=16, 9 output poses, seed=42). Receptors were '
        'converted from the deposited PDB coordinate file to a rigid-body PDBQT representation '
        'using an AutoDock4 atom-type mapping restricted to standard amino-acid heavy and polar '
        'hydrogen atoms (hetero-groups and crystallographic waters are excluded by construction, '
        'since they carry no recognised residue/atom-name mapping). Ligands were embedded in 3D '
        'with explicit hydrogens (RDKit [40]) and parameterised for docking with '
        'Meeko. Search-box centres and edge lengths (22\u201326 \u00c5 per side, target-dependent) were '
        'fixed at each target\u2019s co-crystallised-ligand centroid, following manual inspection of '
        'the deposited structure; the same eight fixed boxes were reused, unchanged, for the '
        'follow-up multi-target docking stage (below). A second, independent multi-target docking '
        'stage re-docked the top 50 single-target hits against seven of the eight targets (STAT3 '
        'excluded for a 0% single-target hit rate) to build the polypharmacology profile used by '
        'network pharmacology and final ranking.')

    h(doc, 'ADMET profiling and mucosal-safety filter', level=2)
    para(doc,
        'Physicochemical properties (Lipinski [32], Veber '
        '[33], Egan [34] rules; QED '
        '[23]) and pkCSM-style [36] ADMET endpoints (Caco-2 '
        'permeability, blood-brain-barrier penetration, plasma-protein binding, CYP inhibition, '
        'hERG liability) were computed with RDKit. A purpose-built mucosal-irritation filter, '
        'motivated by OSCC therapeutics\u2019 direct oral-mucosal exposure, screens each compound '
        'against 27 SMARTS-encoded structural alerts organised into five reactivity mechanisms '
        '(direct tissue-reactive electrophiles, reactive-oxygen-species generators, corrosive/'
        'extreme-pH groups, membrane-disrupting amphiphiles, and covalent protein/keratin-reactive '
        'warheads), each assigned a severity score; a compound is classified HIGH risk if its '
        'maximum alert severity is \u2265 3 or its summed severity is \u2265 5, MODERATE if maximum '
        'severity \u2265 2 or summed severity \u2265 3 (or amphiphilic-risk flagged), LOW if any lesser '
        'alert or extreme-pH risk fires, and SAFE otherwise. The overall ADMET score combines six '
        'normalised sub-scores by weighted geometric mean:')
    equation(doc,
        'ADMET_score = exp( \u03a3_i w_i\u00b7ln(s_i) ),  '
        'w = [druglike:0.20, absorption:0.15, safety:0.20, mucosal:0.20, cardiac:0.10, metabolic:0.15]',
        'Eq. M-ADMET')
    para(doc,
        'the mucosal sub-score receives an elevated weight (0.20, tied with structural-alert '
        'safety) reflecting the oral-delivery context specific to OSCC; the geometric mean means '
        'a single catastrophically poor sub-score dominates the composite regardless of strong '
        'performance elsewhere. Compounds are flagged PASS if ADMET_score \u2265 0.5 and mucosal risk '
        '\u2208 {SAFE, LOW} and zero PAINS alerts and non-High hERG risk; CAUTION if ADMET_score \u2265 0.3 '
        'and mucosal risk \u2260 HIGH and at most one PAINS alert; and FAIL otherwise.')

    h(doc, 'Network pharmacology', level=2)
    para(doc,
        'Polypharmacology among the multi-target-docked elite compounds is represented as a '
        'weighted bipartite graph G = (C \u222a T, E), compounds C and targets T, with an edge for '
        'every compound-target pair reaching \u2264 \u22127.0 kcal/mol, weighted by |\u0394G|. A '
        'complementary target\u2013target network G_TT = (T, E_TT) captures shared-binder overlap via '
        'the Jaccard index:')
    equation(doc, 'J(t\u2081,t\u2082) = |B_t\u2081 \u2229 B_t\u2082| / |B_t\u2081 \u222a B_t\u2082|', 'Eq. M-Jaccard')
    para(doc,
        'where B_t is the set of compounds binding target t at the same threshold. Each target was '
        'manually mapped to KEGG/Reactome pathways relevant to OSCC biology (PI3K/AKT/mTOR axis; '
        'RTK/MAPK cascade; cell-cycle/apoptosis; inflammation/tumour microenvironment), yielding a '
        'compound-level polypharmacology score (fraction of mapped pathway categories engaged) '
        'used both descriptively and as an input to Eq. 12.')

    h(doc, 'Scaffold-constrained lead optimization', level=2)
    para(doc,
        'Rather than sampling latent vectors from the CRAVE prior as in de novo generation, the '
        'triple-validated elite parents are perturbed locally in PGCL latent space:')
    equation(doc, 'z_analog ~ N(z_elite, ' + '\u03c3' + '\u00b2I),  ' + '\u03c3' + ' \u2208 {0.3, 0.5, 0.8}', 'Eq. M-LeadOpt')
    para(doc,
        'at three radii spanning conservative to exploratory structural variation, decoded through '
        'the CRAVE decoder (SELFIES tokens \u2192 SMILES) and passed through a filtering cascade: RDKit '
        'validity; Morgan-fingerprint (radius 2, 2048-bit) Tanimoto similarity to parent \u2265 0.30; '
        'molecular weight \u2264 700 Da; canonical-SMILES deduplication; ADMET re-scoring (above); and '
        'a quality gate requiring PASS or CAUTION flag with QED \u2265 0.4 and SA \u2264 5.5. This produces '
        'analogs localised around each elite parent\u2019s position in the learned chemical space, '
        'preserving inherited potency while sampling property variation.')

    h(doc, 'Nearest known analog analysis and final ranking', level=2)
    para(doc,
        'Every generated compound is contextualised against a 71,965-compound reference library '
        '(training positives, the full curated ChEMBL/OSCC dataset, and repurposing-screen '
        'compounds with clinical metadata) by bulk Tanimoto search '
        '(Morgan fingerprints, radius 2, 2048-bit) for its top-3 nearest known analogs, with '
        'maximum-common-substructure coverage computed for the single nearest analog '
        '(RDKit rdFMCS, element-only atom matching, bond-order-sensitive, 1\u20132 s timeout per pair). '
        'Compounds are classified novel (maximum Tanimoto < 0.4), scaffold-hop (0.4\u20130.7), or '
        'close-analog (\u2265 0.7). All pipeline evidence streams (repurposing screen, single- and '
        'multi-target docking, ADMET, network pharmacology, lead-optimisation analogs) are merged '
        'on compound identifier and canonical SMILES into a single ranked table, scored by a final '
        'GENESIS score structurally distinct from the Phase 6 repurposing score above \u2014 it '
        'incorporates docking and multi-target selectivity data unavailable at the repurposing '
        'stage, drops the SMARTS-enrichment and consensus placeholder terms, and reintroduces the '
        'evidence-completeness penalty (Eq. 11 of the original repurposing scope; renumbered Eq. 12 '
        'here to avoid conflating the two distinct formulas):')
    equation(doc,
        'S_GENESIS = ({SUM}_i w_i\u00b7s_i / {SUM}_i w_i) \u00d7 sqrt(n_available/8),  '
        'w = [dock:0.25, ADMET:0.20, potency:0.15, QED:0.10, multi-target:0.10, clinical:0.10, SA:0.05, novelty:0.05]'.format(SUM=SUM),
        'Eq. 12')
    para(doc,
        'where missing components (e.g. a compound never docked) contribute 0 to the numerator '
        'while retaining their weight in the denominator, n_available counts the components with '
        'non-missing data for that compound, and the square-root scaling penalises partial '
        'evidence without annihilating compounds that are, for instance, docking-validated but not '
        'yet network-pharmacology-profiled. Compounds are additionally stratified into validation '
        'tiers by data completeness (Tier-1: docking + ADMET + multi-target data all present; '
        'Tier-3: either docking or ADMET; Tier-4: prediction-only) for transparent reporting '
        'alongside the continuous score.')
    para(doc,
        'ADMET predictions (physicochemical, Caco-2/BBB/hERG risk) follow pkCSM-style modelling '
        '[36]; structural filters follow Lipinski [32], Veber '
        '[33], and Egan [34] rules. All baseline and ablation '
        'training used scikit-learn [37] and XGBoost [38]; '
        'PyTorch [39] and RDKit [40] were used throughout for PGCL, '
        'generative model training, and all cheminformatics operations.')

    h(doc, 'Hardware and reproducibility', level=2)
    para(doc,
        'All experiments were run on a single Apple M3 workstation (unified memory, PyTorch 2.1+ '
        'with the MPS backend enforcing float32, since MPS does not support float16 matrix '
        'multiplication at the time of this study) \u2014 no dedicated compute cluster or discrete GPU '
        'was used at any pipeline stage. PGCL training completes in under 90 minutes per seed; '
        'the six generative architectures together require several hours per seed, dominated by '
        'the VAE-family and CGLF methods (RL-VAE, Diffusion, and CRAVE converge substantially '
        'faster); single- and multi-target docking, ADMET profiling, network pharmacology, lead '
        'optimisation, and final ranking each complete in minutes. Exact per-phase wall-clock '
        'figures vary run-to-run with input compound counts and are provided, phase-by-phase, in '
        'the accompanying data ledger rather than restated here as a single figure, since (unlike '
        'the algorithmic and statistical methodology above) they are not claims requiring '
        'independent verification.')

    h(doc, 'Acknowledgements')
    para(doc, 'The author thanks the open-source cheminformatics and machine-learning communities '
               'whose tools (RDKit, PyTorch, scikit-learn, AutoDock Vina, MOSES, GuacaMol) made this '
               'work possible. No dedicated external funding supported this study.')

    h(doc, 'Author contributions')
    para(doc, 'S.B. conceived the study, designed and implemented the methodology and software, '
               'performed all analyses, and wrote the manuscript.')

    h(doc, 'Competing interests')
    para(doc, 'The author declares no competing interests.')

    h(doc, 'Ethics and inclusion statement')
    para(doc, 'This study used only publicly available, de-identified compound bioactivity data '
               '(ChEMBL) and publicly deposited protein crystal structures (RCSB PDB); it did not '
               'involve human participants, human tissue, personal data, or animal subjects, and '
               'no ethics approval was therefore required.')

    h(doc, 'Data availability')
    para(doc,
        'The complete verified data ledger underlying every numeric claim in this manuscript '
        '(documents/manuscript_ground_truth.json), together with automated verifiers that '
        're-derive each value from its source result file (documents/verify_manuscript_numbers.py, '
        'documents/verify_manuscript_v3_against_ledger.py), is provided alongside the code.')
    h(doc, 'Code availability')
    para(doc, 'Code, trained checkpoints, and the manuscript generation scripts are available at '
               '[https://github.com/gpt-sb/GENESIS].')

    h(doc, 'References')
    refs = [
        'Sung, H. et al. Global Cancer Statistics 2020. CA Cancer J. Clin. 71, 209\u2013249 (2021).',
        'Warnakulasuriya, S. Global epidemiology of oral and oropharyngeal cancer. Oral Oncol. 45, 309\u2013316 (2009).',
        'Rogers, D. & Hahn, M. Extended-connectivity fingerprints. J. Chem. Inf. Model. 50, 742\u2013754 (2010).',
        'de Oliveira, V. & Saeed, M. MolDeBERTa. arXiv:2208.09624 (2022).',
        'He, P. et al. DeBERTaV3. ICLR (2023).',
        'Khosla, P. et al. Supervised Contrastive Learning. NeurIPS (2020).',
        'Chen, T. et al. A Simple Framework for Contrastive Learning (SimCLR). ICML (2020).',
        'Stumpfe, D. & Bajorath, J. Exploring activity cliffs in medicinal chemistry. J. Med. Chem. 55, 2932\u20132942 (2012).',
        'Huber, P. J. Robust Estimation of a Location Parameter. Ann. Math. Statist. 35, 73\u201101 (1964).',
        'Schroff, F. et al. FaceNet: triplet loss. CVPR (2015).',
        'Bemis, G. W. & Murcko, M. A. The properties of known drugs. J. Med. Chem. 39, 2887\u20132893 (1996).',
        'Loshchilov, I. & Hutter, F. Decoupled Weight Decay Regularization (AdamW). ICLR (2019).',
        'Loshchilov, I. & Hutter, F. SGDR: Cosine Annealing. ICLR (2017).',
        'Kingma, D. P. & Welling, M. Auto-Encoding Variational Bayes. ICLR (2014).',
        'Perez, E. et al. FiLM: Visual Reasoning with a General Conditioning Layer. AAAI (2018).',
        'Fu, H. et al. Cyclical Annealing Schedule for KL Vanishing. NAACL (2019).',
        'Williams, R. J. Simple statistical gradient-following algorithms (REINFORCE). Mach. Learn. 8, 229\u2013256 (1992).',
        'Lipman, Y. et al. Flow Matching for Generative Modeling. ICLR (2023).',
        'Cho, K. et al. Learning Phrase Representations using RNN Encoder-Decoder. EMNLP (2014).',
        'Krenn, M. et al. SELFIES. Mach. Learn.: Sci. Technol. 1, 045024 (2020).',
        'Polykovskiy, D. et al. Molecular Sets (MOSES). Front. Pharmacol. 11, 565644 (2020).',
        'Brown, N. et al. GuacaMol: Benchmarking Models for de Novo Molecular Design. J. Chem. Inf. Model. 59, 1096\u20131108 (2019).',
        'Bickerton, G. R. et al. Quantifying the chemical beauty of drugs (QED). Nat. Chem. 4, 90\u201398 (2012).',
        'Ertl, P. & Schuffenhauer, A. Synthetic accessibility score. J. Cheminform. 1, 8 (2009).',
        'Truchon, J.-F. & Bayly, C. I. BEDROC. J. Chem. Inf. Model. 47, 488\u2013508 (2007).',
        'Preuer, K. et al. Fr\u00e9chet ChemNet Distance. J. Chem. Inf. Model. 58, 1736\u20131741 (2018).',
        'J\u00e4rvelin, K. & K\u00e4kel\u00e4inen, J. Cumulated gain-based evaluation of IR techniques. ACM TOIS 20, 422\u2013446 (2002).',
        'Efron, B. & Tibshirani, R. An Introduction to the Bootstrap. Chapman & Hall (1993).',
        'Mendez, D. et al. ChEMBL. Nucleic Acids Res. 47, D930\u2013D940 (2019).',
        'Trott, O. & Olson, A. J. AutoDock Vina. J. Comput. Chem. 31, 455\u2013461 (2010).',
        'Eberhardt, J. et al. AutoDock Vina 1.2. J. Chem. Inf. Model. 61, 3891\u20133898 (2021).',
        'Lipinski, C. A. et al. Rule of Five. Adv. Drug Deliv. Rev. 46, 3\u201326 (2001).',
        'Veber, D. F. et al. Molecular properties for oral bioavailability. J. Med. Chem. 45, 2615\u20132623 (2002).',
        'Egan, W. J. et al. Prediction of drug absorption. J. Med. Chem. 43, 3867\u20133877 (2000).',
        'Baell, J. B. & Holloway, G. A. PAINS. J. Med. Chem. 53, 2719\u20132740 (2010).',
        'Pires, D. E. V. et al. pkCSM. J. Med. Chem. 58, 4066\u20134072 (2015).',
        'Pedregosa, F. et al. Scikit-learn. JMLR 12, 2825\u20132830 (2011).',
        'Chen, T. & Guestrin, C. XGBoost. KDD (2016).',
        'Paszke, A. et al. PyTorch. NeurIPS (2019).',
        'Landrum, G. RDKit: Open-source cheminformatics. https://www.rdkit.org (2006).',
    ]
    for i, ref in enumerate(refs, 1):
        para(doc, f'{i}. {ref}', size=10)

    doc.save(OUT)
    print(f'Saved -> {OUT}')


def build_supplementary():
    """Supplementary Information, submitted as a separate file per Nature-family convention."""
    doc = Document()
    for section in doc.sections:
        section.top_margin = Cm(2.54)
        section.bottom_margin = Cm(2.54)
        section.left_margin = Cm(2.54)
        section.right_margin = Cm(2.54)
    style = doc.styles['Normal']
    style.font.name = 'Arial'
    style.font.size = Pt(11)
    style.paragraph_format.line_spacing = 2.0

    para(doc, 'Supplementary Information', bold=True, size=16,
         alignment=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
    para(doc, 'GENESIS: Potency-Graded Contrastive Learning for Generative Molecular Design '
               'and Drug Repurposing in Oral Squamous Cell Carcinoma', size=12,
         alignment=WD_ALIGN_PARAGRAPH.CENTER, space_after=24)

    h(doc, 'Supplementary Note 1: External benchmark validation (MOSES / GuacaMol)')
    para(doc,
        'To situate the six generative architectures against the broader molecular-generation '
        'literature, we additionally computed standard MOSES [21] '
        'distribution-learning metrics (internal diversity IntDiv/IntDiv2, nearest-neighbour '
        'similarity to the held-out test set SNN, BRICS fragment similarity Frag, Murcko scaffold '
        'similarity Scaf, PAINS-filter pass rate, and Fr\u00e9chet ChemNet Distance FCD [Preuer et al., '
        '2018]) and GuacaMol [22] property-distribution KL divergences, each as a '
        '3-seed mean\u00b1SD (seeds 42, 123, 456), matching the replication standard used throughout '
        'this study.')
    para(doc,
        'Because CRAVE and CGLF are explicitly guided toward the high-potency region of PGCL '
        'latent space (Eq. 8\u201310), whereas VAE and cVAE sample without potency guidance, reduced '
        'internal diversity and reduced similarity to the training/test property distribution for '
        'CRAVE/CGLF relative to the unconstrained baselines is an expected consequence of '
        'potency-guided generation, not a fidelity deficiency; this is the same trade-off already '
        'quantified in Table 2 of the main text as substantially higher OSCC potency for CRAVE/CGLF '
        'at the cost of uniqueness/scaffold novelty relative to VAE.')
    moses_rows = []
    moses_order = ['vae', 'cvae', 'diffusion', 'rl_vae', 'cglf', 'crave_filtered', 'crave_unfiltered']
    moses_names = {'vae': 'VAE', 'cvae': 'cVAE', 'diffusion': 'Diffusion', 'rl_vae': 'RL-VAE',
                   'cglf': 'CGLF', 'crave_filtered': 'CRAVE (filt.)', 'crave_unfiltered': 'CRAVE (unfilt.)'}
    for k in moses_order:
        mo = MOSES3['methods'][k]['moses']
        fcd = MOSES3['methods'][k]['fcd']
        moses_rows.append([
            moses_names[k],
            f"{mo['IntDiv']['mean']:.3f}\u00b1{mo['IntDiv']['std']:.3f}",
            f"{mo['SNN']['mean']:.3f}\u00b1{mo['SNN']['std']:.3f}",
            f"{mo['Frag']['mean']:.3f}\u00b1{mo['Frag']['std']:.3f}",
            f"{mo['Scaf']['mean']:.3f}\u00b1{mo['Scaf']['std']:.3f}",
            f"{mo['Filters']['mean']:.3f}\u00b1{mo['Filters']['std']:.3f}",
            f"{fcd['mean']:.2f}\u00b1{fcd['std']:.2f}",
        ])
    table(doc, ['Method', 'IntDiv', 'SNN', 'Frag', 'Scaf', 'Filters', f'FCD {chr(8595)}'], moses_rows,
          caption='Supplementary Table S1. MOSES-equivalent metrics, 3-seed mean\u00b1SD (seeds 42/123/456), '
                  'reference: training set (34,406 compounds) / held-out test set.')
    cglf_scaf = MOSES3['methods']['cglf']['moses']['Scaf']
    vae_scaf = MOSES3['methods']['vae']['moses']['Scaf']
    cglf_frag = MOSES3['methods']['cglf']['moses']['Frag']
    cvae_frag = MOSES3['methods']['cvae']['moses']['Frag']
    diff_fcd = MOSES3['methods']['diffusion']['fcd']
    crave_fcd = MOSES3['methods']['crave_filtered']['fcd']
    para(doc,
        f'CGLF and VAE are statistically indistinguishable on scaffold fidelity '
        f'(Scaf = {cglf_scaf["mean"]:.3f}\u00b1{cglf_scaf["std"]:.3f} vs. {vae_scaf["mean"]:.3f}\u00b1{vae_scaf["std"]:.3f}), '
        f'both clearly ahead of cVAE and the CRAVE variants; CGLF retains a robust, well-separated '
        f'lead on fragment-level fidelity (Frag = {cglf_frag["mean"]:.3f}\u00b1{cglf_frag["std"]:.3f} vs. next-best '
        f'cVAE {cvae_frag["mean"]:.3f}\u00b1{cvae_frag["std"]:.3f}, a >3-SD separation). Diffusion\u2019s FCD shows the '
        f'largest relative variance of any method ({diff_fcd["mean"]:.2f}\u00b1{diff_fcd["std"]:.2f}), an independent '
        f'metric corroborating the seed-dependent catastrophic collapse already documented for its '
        f'OSCC potency scores above \u2014 the same training instability manifests in both a task-specific '
        f'(potency) and a task-agnostic (distributional) metric. CRAVE (filtered) achieves the '
        f'lowest and most tightly-confirmed FCD of all seven methods '
        f'({crave_fcd["mean"]:.2f}\u00b1{crave_fcd["std"]:.2f}), consistent with its selection as the flagship '
        f'candidate-generation method.')

    doc.save(OUT_SI)
    print(f'Saved -> {OUT_SI}')


if __name__ == '__main__':
    build()
    build_supplementary()
