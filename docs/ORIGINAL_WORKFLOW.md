# Original workspace workflow

The scripts under `code/original_workspace/` retain their original directory structure. They refer to other files in the research workspace and are supplied as source snapshots. The complete dependency inventory and checksums are in [original_workspace_inventory.csv](original_workspace_inventory.csv) and [external_analysis_inputs_sha256.csv](external_analysis_inputs_sha256.csv). Use a copy of the original workspace for re-estimation.

1. The OpenAlex scan begins with `expanded_sample_reassessment_20261102/01_sample_construction/compact_catalog.py`. Its upstream checks depend on the prior membership and baseline benchmark files. `build_global_histories.py` and `build_populations.py` produce author-field histories and eligible cores. `build_core_data.py` constructs joint works and the dated expertise measures.
2. The reference-profile inputs trace to `wp1_knowledge_input/audit_raw.py` and `validate_windows.py`. `build_frozen_matrices.py` forms the knowledge gap, distance and frozen Class C support. `portfolio_A_theory_driven_exploration_20261103/code/build_architecture.py` builds individual, shared and distributed coverage from frozen member histories and the reference matrix. `expanded_sample_reassessment_20261102/08_reproducibility/panel_api.py` joins exposures and builds the mapping-specific risk sets.
3. `analysis_extensions/theory_evidence_alignment/engine.py` handles the Unified Knowledge Specification. `run.py`, `placebo.py`, and `conditional_placebo.py` are the original entry points for the Table 2 and Table 4 specifications. `temporal_productivity_validation_20261104/code/` contains the original Table 5 and Table 6 scripts. The optimizer and historical paths have separate prerequisites documented in that project's archived README. `portfolio_A_theory_driven_exploration_20261103/code/verify_architecture_lead.py` constructs the outcome and measurement partitions in Table 7.

To reconstruct the outcome cross-products behind Table 3, the workspace needs the original risk loader, work and author histories, and locked A1 regressor moments and validation fits. From this repository, run the two mappings separately using the original workspace path:

```bash
python -B code/reestimate_table3.py parent --workspace /path/to/OpenAlex --out generated/workspace_table3
python -B code/reestimate_table3.py direct --workspace /path/to/OpenAlex --out generated/workspace_table3
```

The wrapper reads `analysis_extensions/theory_evidence_alignment/cache/{mapping}/unified.npz`, the same project's `fits/{mapping}/a1_unified/fit.npz`, and the original `portfolio_A_theory_driven_exploration_20261103/verification/recent_dyad/{mapping}/validity/` fits. Copies of the locked moment and A1 fit arrays are in `data/table3/` as `{mapping}_original_unified_moments.npz` and `{mapping}_original_a1_fit.npz`. The table replay in the root README starts after this stage and uses the final `{mapping}_a1_moments.npz`. The workspace wrapper uses pandas, DuckDB, and the workspace's PyArrow and scientific Python dependencies in addition to the portable table environment.
