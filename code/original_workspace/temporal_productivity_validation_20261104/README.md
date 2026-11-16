# Temporal comparability and proportional-productivity validation

Primary output: [concise research memo](Exploratory_Validation_Temporal_and_Productivity.md). Supporting material: [technical appendix](Detailed_Methods_and_Results.md), [comparison-table index](tables/INDEX.md), [all LPM coefficients](data/all_LPM_coefficients.csv), and [PPML mode contrasts](data/PPML_formal_relative_rate_contrasts.csv).

This focused extension evaluates two questions using the existing Organization of Expertise project: whether the recent-dyad HomeGain x distributed-coverage association has a comparable historical counterpart, and whether a common proportional response interacting with heterogeneous historical entry probabilities explains the signed LPM gradient.

The original project and all source datasets/results remain unchanged. Main populations are the complete recent-dyad frozen2021 C candidate universes under the existing Parent and Direct mapping/Home-support rules. All absent events remain zeros, including after a first entry. This is fixed-universe first incidence, not a surviving-risk hazard.

## Outputs

- `Exploratory_Validation_Temporal_and_Productivity.md`: research memo and direct assessment.
- `tables/`: compact comparison tables, Markdown and LaTeX.
- `data/`: every coefficient, covariance alternative, formal difference, failure and model-support statistic in CSV.
- `results/`: new LPM fits, full covariance matrices, joint period/outcome covariances, and restricted heterogeneous-propensity benchmarks.
- `baseline_propensities_crossfit/`: historical-only out-of-core-fold predictions and validation. The older `baseline_propensities/` own-history attempt is retained as a superseded construction, not used in the main test.
- `relative_rate/` and `relative_rate_accelerated/`: official PPMLHDFE attempts, exact likelihood-support accounting, separation failures, source data, scripts and covariance matrices. The accelerated namespace changes only numerical nuisance preconditioning.
- `relative_rate_certified*`: covariance refits of the same likelihood after a completed official separation/full-rank certificate. Acceptance requires reproduction of the original AI coefficients, capability covariance, likelihood, N and FE degrees of freedom. These are numerical implementations, not additional empirical specifications. Interrupted numerical approaches remain archived.
- `diagnostics/`: source hashes, independent algebra/estimator checks, historical coverage, timing quality and failed attempts.

## Fixed design

Original benchmark: July2025-May2026, eleven months. Equal-length comparison: January-October2022 versus July2025-April2026, ten months each. All knowledge primitives, mapping and scales remain fixed. See `ANALYSIS_REGISTRY.md` for pre-estimate decisions and subsequent construct/numerical corrections.

Time-invariant knowledge distribution and future-average marker interactions are absorbed by pair FE. The absolute-gradient comparison therefore uses identical core-month/destination-month effects and other modifier-month slopes, while its pooled change model retains pair FE. The comparable sorting marker replays the existing future capability trajectory by relative month and preserves the full original FE design. It is not historical AI exposure or a parallel-trends test.

The proportional benchmark uses full-risk historical destination/Home-field/architecture surfaces with empirical-Bayes smoothing. Each core's prediction excludes its own historical entry labels; core activity predictors are frozen2021 publications and breadth, trained on other cores. Partialling all original nuisance terms gives the exact implied signed LPM projection and its covariance-aware residual. Predictions are fixed in downstream inference, so first-stage uncertainty is not fully included.

Full-pair-FE PPML is attempted, not assumed feasible. Offset alternatives explicitly replace pair FE and compare type-specific versus common core-month opportunity factors. PPML profiles analytically zero FE groups; complete eligible populations and informative likelihood support are reported separately. Five one-way cluster fits form the original five covariance families by inclusion-exclusion outside Stata, preserving negative variances without PSD correction. Stata nested-FE small-sample conventions differ from the original Python full-FE CR1.

## Reproduction

Analysis Python: `C:/Python313/python.exe`. Stata: `C:/Program Files/Stata17/StataMP-64.exe`. NumPy, pandas, SciPy and DuckDB are required. The official project-local PPMLHDFE source and the missing official reghdfe legacy Mata compatibility file are recorded by hashes in `dependencies/ppmlhdfe`. Global Stata packages are unchanged. The public dependency download required sandbox network authorization once; cached sources suffice for reproduction.

Run each script from this directory or with its full path. Scripts resolve paths from their own location and reuse completion markers. Existing measurement caches in the sibling `portfolio_A_theory_driven_exploration_20261103` and full populations in `expanded_sample_reassessment_20261102` are required.

1. `audit.py`, `validate_restrictions.py`, `replicate.py parent`, `replicate.py direct`.
2. `temporal.py MAPPING gradient`, `temporal.py MAPPING trajectory_marker` for each mapping.
3. `propensity.py MAPPING` constructs reusable historical cell counts. `crossfit_propensity.py MAPPING` produces the main probabilities. `proportional.py MAPPING` estimates the specified focal-X restricted benchmark and mode-specific responses. `proportional.py MAPPING --home-only` estimates the economically distinct general HomeGain counterbenchmark, using the same full risk and historical predictions. Both are retained.
4. `ppml_stata.py MAPPING`, `ppml_stata.py MAPPING --shared-only`. Optional `ppml_accelerated.py MAPPING` estimates exactly the same offset models with an independently validated invertible nuisance preconditioner. No source is selected by its coefficient or p-value.
   When an original model has a converged cap fit with zero non-FE separation and full AI rank, `ppml_certified.py MAPPING REGIME --reuse-rank` reuses those certificates to avoid repeated expensive checks. It validates the unchanged likelihood/capability covariance before accepting the supplementary covariance results. Run `ppml_certified.py direct historical_offset --smoke --reuse-rank` for the independent estimator check. Supplementary covariance recomputation has a fixed 45-minute total budget; incomplete results do not invalidate the original full-check capability covariance and are not presented as five-scheme PPML robustness. The minimum numerical budgets for first mean fits are 90 minutes for original attempts and 45 minutes for equivalent accelerators without a first IRLS iteration; progressing jobs have a three-hour ceiling. `ppml_supervise.py` is the recorded session-specific process controller and is not a fresh reproduction entry point. Normal scripts implement budgets in `ppml_runtime.py`.
5. `risk_diagnostics.py`, `expertise_timing_diagnostics.py`, `verify_partial_certificates.py` (after bounded certificate/rank-reuse attempts close), `collect.py`, `diagnose_outputs.py`, `validate_outputs.py`, `figures.py`, `build_report.py`, `make_manifest.py`. The report builder requires the completed model-validation receipt and the authored substantive assessment in `interpretation.json`; it never automatically selects a story from p-values. Markdown and LaTeX comparison tables are generated together; TeX fragments require `booktabs` and `graphicx`.

Large computations are resumable. Preserve separate logs for resumed attempts. Before restarting a module, check running processes/completion markers to avoid duplicate work. The optional task queues are execution helpers, not prerequisites. No significance-oriented lag/window/subgroup/path search is performed.

## Final covariance validity note

Individual contrast variance and whole-matrix PSD status are recorded separately. The per-outcome LPM audit flags41 indefinite matrices among320, all in the Home/capability composite; two temporal variances are negative. Joint temporal composite matrices can also be indefinite. The Direct/shared PPML capability/core composite is indefinite despite a positive focal variance. These results remain available as diagnostics with no PSD repair. Main compact tables use capability clustering; the report describes the completed one-way PPML alternatives and the four PSD benchmark covariance families. Do not interpret the five saved LPM schemes as five uniformly valid matrices or claim complete five-scheme PPML robustness.
