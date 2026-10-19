# Result File Column Reference

Short descriptions for the less self-explanatory columns across the result
tables in this repository. Standard cheminformatics fields (`smiles`, `mw`,
`logp`, `hba`/`hbd`, `tpsa`, `qed`, `sa_score`, `pains_count`, `brenk_count`,
`lipinski_violations`, `veber_violations`) are not repeated per-file below
since their meaning is constant throughout.

## `results/elite_candidates.tsv`
The 13 triple-validated elite compounds (docking + ADMET + multi-target
selectivity all concordant; `final_ranking/results/master_ranked_all.tsv`
filtered on `is_elite == True`), sorted by `genesis_score` descending.

## `final_ranking/results/master_ranked_all.tsv` / `master_ranked_recommended.tsv`
The complete integrated ranking (1,161 compounds / 316 recommended subset).
- `cpd_id` — `GEN_*` = de novo generated, `REP_*` = repurposed
- `genesis_score` — final composite score (Eq. 12 in the manuscript)
- `validation_tier` — Tier-1 (fully validated) / Tier-3 (partial) / Tier-4 (prediction-only)
- `is_elite` — triple-validated elite flag (docking + ADMET + selectivity concordant, 13 compounds)
- `dG_<TARGET>` — docking binding energy (kcal/mol) against that specific target
- `mt_n_hits_le7` / `mt_n_hits_le8` — number of targets bound at ≤ −7.0 / ≤ −8.0 kcal/mol (multi-target docking)
- `selectivity_index` — polypharmacology/selectivity composite from network pharmacology
- `s_docking`, `s_admet`, `s_potency`, `s_qed`, `s_multitarget`, `s_clinical_evidence`, `s_sa`, `s_novelty` — the 8 normalised sub-scores combined into `genesis_score`
- `nka1_*`/`nka2_*`/`nka3_*` — top-3 nearest known analogs (Tanimoto similarity, ChEMBL ID, name, pChEMBL, target, max clinical phase, mechanism of action)
- `novelty_class` — novel (Tanimoto < 0.4) / scaffold-hop (0.4–0.7) / close-analog (≥ 0.7) to nearest known compound
- `vp_similarity` — cosine similarity to the PGCL very-potent centroid
- `clinical_priority_score`, `priority_reasons`, `predicted_drug_class`, `oscc_clinical_rationale` — translational-priority annotations

## `admet/results/final_candidates.tsv`
ADMET-screened shortlist (316 compounds: PASS + CAUTION).
- `admet_score` — weighted geometric mean of 6 sub-scores (see Eq. M-ADMET in the manuscript)
- `admet_flag` — PASS / CAUTION / FAIL
- `mucosal_risk` — SAFE / LOW / MODERATE / HIGH (OSCC-specific oral-mucosa irritation filter)
- `mucosal_alert_count`, `mucosal_severity_sum`, `amphiphilic_risk`, `extreme_ph_risk` — mucosal-filter sub-components
- `s_druglike`, `s_absorption`, `s_safety`, `s_mucosal`, `s_cardiac`, `s_metabolic` — the 6 ADMET sub-scores
- `caco2_class`, `bbb_class`, `ppb_class`, `herg_risk` — pkCSM-style predicted ADMET endpoints
- `phase7_energy`, `mt_best_energy`, `in_phase7b` — docking carry-over fields

## `admet/results/admet_summary.json`
Aggregate counts: Lipinski pass rate, structural-alert-clean count, mucosal-safe count, final shortlist breakdown, and `triple_validated_elite` (count + mean final score).

## `docking/results/docking_results.tsv` (single-target) / `multitarget_results.tsv`
- `best_energy_kcal` — lowest (most favourable) Vina binding energy across poses
- `all_energies` — full list of per-pose energies
- `n_poses` — number of docking poses generated
- `original_target` / `original_energy` (multi-target only) — the Phase 7a target/energy this compound was originally selected from before multi-target re-docking
- `overall_flag` — combined docking+composite-score triage flag

## `repurposing/results/global_rankings.tsv` / `top100_candidates.tsv`
The 23,308-compound repurposing library, ranked by `recommendation_weight` (Eq. 11, 8-component composite — see `norm_*` columns for each normalised sub-component).
- `compound_origin`, `natural_product`, `np_likeness_score` — provenance flags
- `potency_class` / `potency_class_num` — Active/Potent/Very Potent ordinal class and its 1/2/3 numeric code
- `vp_similarity`, `t3c_similarity` — cosine similarity to the very-potent centroid / target-switching-negative centroid
- `enriched_smarts_hits` — count of enriched substructure motifs matched (within-library enrichment, see manuscript s_smarts definition)
- `in_training_positives` — whether this compound was part of PGCL's training data (should be excluded from novel-discovery claims if true)
- `top100_candidates.tsv` additionally includes `*_pharm` columns (pharmacophore-derived descriptors) and `best_target`/`max_target_score`/`n_targets_top50` (per-target scoring breakdown)

## `leadopt/results/analogs_improved.tsv`
Scaffold-constrained analogs of the 13 elite parents that scored above their parent compound.
- `parent_id`, `parent_smiles` — the elite compound this analog was generated from
- `analog_smiles` — the generated analog structure
- `sigma` — latent-space perturbation radius used (0.3 / 0.5 / 0.8)
- `tanimoto_to_parent` — structural similarity retained to the parent

## `network_pharmacology/results/`
- `compound_network_metrics.tsv` — per-compound polypharmacology profile (`targets_bound`, `n_targets_bound`, `mean_affinity`/`max_affinity`, `polypharmacology_score`)
- `pathway_enrichment.tsv` — pathway-level enrichment of the elite set vs. background (`fold_enrichment`, `elite_rate` vs. `background_rate`)
- `pathway_mapping.tsv`, `pathway_categories.tsv`, `target_centrality.tsv` — supporting target→pathway and network-centrality tables

## `results/phase4_bioactivity_profiling/`
- `profiling_summary.json` — aggregate counts (unique molecules, consensus hits, safety PASS rate, enriched SMARTS)
- `consensus_compounds.tsv` — molecules independently generated by ≥ 2 of the 6 methods
- `top100_candidates.tsv` — top-scoring generated molecules by composite score

## `generative/outputs/seed_42/generated_<method>.txt`
Raw canonical-SMILES output, one molecule per line, for each of the 6 generative architectures (`vae`, `cvae`, `diffusion`, `rl_vae`, `cglf`, `crave`) from the canonical seed-42 run, plus `generated_crave_unfiltered.txt` (pre-safety-filter CRAVE output, for comparison). Seed 42 only is included here (the manuscript's 5-seed statistics are in `generative/multiseed/aggregate/`); re-run `generative/run_all.py --seed <N>` for other seeds.

## `.xlsx` files
`final_ranking/results/*.xlsx` are formatted spreadsheet versions of the corresponding `.tsv` files (identical data), provided for convenience when opening in Excel/Numbers/Google Sheets rather than a text editor.
