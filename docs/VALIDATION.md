# Validation record

The aggregate replay and the optional measurement replay were tested in an isolated directory with Python 3.13.5, NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.6 and Matplotlib 3.11.2. The main table replay requires only NumPy and SciPy.

| Check | Result |
| --- | --- |
| Table 3 replay | Parent and Direct produced the archived table and prose values, including the joint contrast and all 1,024 restricted bootstrap draws. |
| Tables 1, 2 and 4–7 | Numeric rows and significance markers matched the archived display pipeline. |
| Table 5, Panel A | Recalculated the four restricted comparisons from aggregate primitive estimates and aligned capability-path scores. |
| PLOS ONE layout | The numeric cells and stars in all seven tables matched the current submission source in their original row and column order. |
| Citation profiles | The 252-by-252 matrix and its field index reproduced the frozen estimation matrix exactly after ordering by the estimation subfield index. |
| Benchmark replay with the author's archived inputs | Reconstructed 110 main path-month gains with maximum absolute discrepancy of approximately $8.33\times10^{-17}$ across the archived gain series. Confirmed 186 Parent and 131 Direct mapped destinations. |
| Original analysis units | The complete risk sets, inference conventions, and historical baselines were inherited from the original fits. |

Full raw-data optimization, historical propensity fitting and donor refitting were outside this portable check. Table 5, Panel B uses stored estimates and diagnostics from the offset PPML fit. The original full-risk Table 3 construction has a separate workspace-dependent entry point in `code/reestimate_table3.py`.
