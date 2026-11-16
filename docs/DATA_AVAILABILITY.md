# Data availability and provenance

## Repository inputs

The repository contains model-level aggregate estimates, capability-path regression cross-products, table source data, reference shares by subfield, and OpenAlex-to-MMLU-Pro field mappings. The portable replay reads these files without downloading OpenAlex or accessing the original research workspace. Workspace-specific prefixes in archived source ledgers have been removed. Numerical values and the relative paths used to resolve their archive pointers were preserved.

The OpenAlex source used by the catalog builder is the Parquet snapshot dated **23 September 2026**, as identified by its local manifest. The snapshot date, format and manifest checksum are recorded in [openalex_snapshot.json](openalex_snapshot.json). The provider page documents how OpenAlex distributes snapshots. Use the date and checksum to identify the original version. [OpenAlex snapshot documentation](https://help.openalex.org/access/snapshot/)

The repository includes the destination-by-input citation share matrix for 2010–2014. `data/citation_profiles/subfield_index.csv` assigns the same zero-based subfield order to rows and columns. `pre2014_domain_counts.csv` records classified citation counts and coverage for that window. These files contain discipline-level aggregates. Author roster data, author publication histories, work records, and the full candidate panel remain in the original workspace. The [original input hash list](external_analysis_inputs_sha256.csv) identifies many of those dependencies.

## Benchmark score inputs

The archived `field_monthly_scores.csv` begins with supplied domain-by-month scores. `field_month_proxies.csv` is the corresponding frozen regression grid. The exact upstream model-level evaluations and protocol-specific scoring records are unavailable. The project archive does not establish the redistribution terms for the two input CSVs. Both remain in the authors' existing measurement archive. External readers need access to the exact inputs or a provider-approved copy to rerun the separate measurement replay. The supplied mixed-protocol series is distinct from the public question set and the protocol described in the benchmark paper. [MMLU-Pro benchmark paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ad236edc564f3e3156e1b2feafb99a24-Abstract-Datasets_and_Benchmarks_Track.html)

## Submission data statement

This archive demonstrates table reproduction at the level stated in the README. Raw-data reconstruction, historical propensity estimation, donor refits, and the offset PPML optimizer have separate data dependencies. For the submission Data Availability Statement, identify any accessible deposition of those analysis dependencies and state the verified access terms for third-party inputs. PLOS ONE evaluates access to the data needed to replicate the reported findings. [PLOS ONE data availability policy](https://journals.plos.org/plosone/s/data-availability)
