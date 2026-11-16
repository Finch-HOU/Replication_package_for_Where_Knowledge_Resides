# Where knowledge resides

Replication materials for **Where knowledge resides: Generative AI and the organization of innovation**, by Fangzhuo HOU. The table numbering and printed values follow the PLOS ONE manuscript.

## Reproduce the tables

Use Python 3.13. The commands below were tested with Python 3.13.5. Run them from the repository root, preferably in a virtual environment.

```bash
python -m pip install -r requirements.txt
python -B reproduce.py
```

The script re-estimates Table 3 from the archived capability-path cross-products for the Parent and Direct mappings. It recomputes the component estimates and their joint contrast, including the restricted path-score bootstrap. It regenerates the historical displays underlying Tables 1, 2, and 4–7, checks their reported cells against the archived estimates, and recalculates the four proportional-productivity benchmarks in Table 5, Panel A. It checks numeric cells and significance marks against the final PLOS ONE table source.

The script writes the verified LaTeX exhibits to `generated/tables/` and the PLOS ONE LaTeX fragments to `generated/plos_tables/`. The Table 3 coefficients and inference are written to `generated/table3_estimates.csv` and `generated/table3_results.json`. Use `--out <directory>` to choose another output location.

| Location | Contents |
| --- | --- |
| `data/table3/` | Capability-path cross-products, fit metadata and stored Table 3 estimates. |
| `data/benchmark/` | Archived estimates and path scores used to recalculate Table 5, Panel A. |
| `data/prior_display_inputs/` | Aggregate outputs, display code and cell-by-cell source ledgers. Some ledger labels precede the final renumbering. |
| `data/citation_profiles/` | Historical subfield-to-subfield citation shares, counts and matrix index. |
| `measurement/` | Parent and Direct crosswalks, capability construction code, input hashes and audit summaries. |
| `code/original_workspace/` | Data construction and estimation snapshots from the original OpenAlex workspace. |
| `reference/` | Published-layout table fragments and their archived numeric references. |

## Rebuild the capability paths with the archived score inputs

This step uses two external files with the hashes recorded in `measurement/reference/aggregate_reconciliation.json`. Place the exact `field_monthly_scores.csv` and `field_month_proxies.csv` in a local input directory. The measurement audit begins with domain-by-month aggregates and checks the frozen proxy grid.

```bash
python -m pip install -r measurement/requirements.txt
python -B code/reproduce_capability.py --benchmark-inputs local_inputs
```

The command checks the source hashes and reconstructs the 110 path-month values used by the main analysis. It writes the series, mapping and missingness diagnostics, summary tables, and trajectory figure to `generated/measurement/`. See [data availability](docs/DATA_AVAILABILITY.md) for the status of the score inputs and the missing upstream evaluation records.

## Re-estimate on the original OpenAlex workspace

The original builder and optimizer snapshots document upstream estimation. A full replay requires the original paths, author and work records, frozen histories, and other files listed in the [workspace inventory](docs/original_workspace_inventory.csv). Some historical scripts take Windows filesystem paths or rely on a local Stata installation. See the [original workflow](docs/ORIGINAL_WORKFLOW.md), [table source map](docs/TABLE_SOURCE_MAP.md), [data dictionary](docs/data_dictionary.md), and [validation record](docs/VALIDATION.md). The `SHA256SUMS` file records the packaged data inputs and reference exhibits.
