# Table source map

Workspace-relative paths below identify the original files. The archived prior ledger and its source snapshots are under **data/prior_display_inputs/** in this repository. Its original table numbers are intentional. Paths beginning with **02_Replication/** refer to the original archive hierarchy. The corresponding files are now under **code/** and **data/** at the repository root.

| Current table | Original model lineage | Authoritative output inputs |
|---|---|---|
| 1 | Frozen full-risk sample and standardization from **anchor_amplification_exploration_20261106/code/baseline.py** | **anchor_amplification_exploration_20261106/00_baseline_frozen/replication/{parent,direct}/sample.json** and the component JSONs in those folders |
| 2 | Original benchmark plus **analysis_extensions/theory_evidence_alignment/run.py**, specification **a1_unified** | **unified_specification_results.csv**, **fits/{mapping}/a1_unified/**, and frozen originals under **frozen_table3/** |
| 3 | **02_Replication/code/reestimate_table3.py**, then **replay_table3.py** | **02_Replication/data/table3/{mapping}_a1_moments.npz**, timing/fit metadata and **table3_estimates.csv** |
| 4 | **analysis_extensions/theory_evidence_alignment/{placebo,conditional_placebo,collect}.py** | **diagnostics/placebo_mean_contrasts.csv** and **diagnostics/coefficient_contrasts.csv** under the extension |
| 5 | **temporal_productivity_validation_20261104/code/{crossfit_propensity,proportional,ppml_stata,ppml_certified}.py** | **results/{mapping}/{proportional,home_proportional}/restricted_benchmark.json** and **data/PPML_coefficients.csv**, **data/PPML_formal_relative_rate_contrasts.csv** |
| 6 | **temporal_productivity_validation_20261104/code/temporal.py** | **data/all_LPM_coefficients.csv** and **data/temporal_formal_differences.csv** under the temporal project |
| 7 | **portfolio_A_theory_driven_exploration_20261103/code/{verify_architecture_lead,collect_verification}.py** | **data/lead_validity_all_covariances.csv**, **data/lead_validity_counts.csv** and **verification/recent_dyad/{mapping}/validity/** under Portfolio A |

The prior display pipeline is archived with its numeric inputs under **02_Replication/data/prior_display_inputs/code/**. It verifies the exact source for every star and regenerates the prior table layouts. The final audit compares those cells to the reorganized main exhibits. Table 3 is generated directly from its new aggregate fit.
