# Data dictionary

| Item | Meaning and provenance |
|---|---|
| Core | Persistent dyad with at least two qualifying exact-roster publications during 2018–2021, constructed from pre-freeze membership. |
| Candidate | OpenAlex subfield supported by the respective mapping and frozen Class C for the dyad. Both members have zero recorded destination publications at end-2021. The retained core has valid frozen Home support. |
| Candidate-month | A Class C pair and one of the eleven months July 2025–May 2026. The fixed prospective universe retains zero rows after first entry. |
| Historical input weights | Reference shares from destination publications in 2010–2014. Collective, shared and best-member coverage use the frozen author histories. |
| G, D, U, Sh, L | Knowledge gap, minimum citation-profile distance, distributed coverage beyond the best member, shared coverage, and directional applicability. A1 uses both gain interactions and month slopes for each, and adds DestinationGain interactions with log breadth and log productivity. |
| HomeGain and DestinationGain | Mixed-protocol 12-month benchmark change lagged six months. HomeGain averages over the pooled frozen mapped publication counts. DestinationGain uses the candidate mapping. |
| First-entry bundle | The works of a core in its first observed joint publication month in the candidate. A bundle is Specialist if any work includes an additional author with recorded prior-year destination expertise. |
| Table 3 neither / at_least_one | Two full-risk indicator outcomes. Each requires a No-Specialist bundle and the indicated core expertise state. All other observations have outcome zero. |
| prior-year core status | One or both core members have an author-field first publication year strictly smaller than the outcome calendar year. This can include post-2021 experience. The source is the complete author-field history joined to the original membership. |
| Q_by_path, xy_by_path | Anonymous projected regressor cross-products and two outcome cross-products by capability path. The 62 columns comprise twelve gain interactions and ten orthonormal time deviations for each of five modifiers after the original three fixed-effect sets are projected out. |
| b, se, t, p, ci_lo, ci_hi | Coefficient, clustered standard error, t statistic, two-sided t(9) probability and 95% limits. The manuscript multiplies LPM coefficients, standard errors and interval limits by 10,000. PPML slopes stay in log-rate units. |
| difference | Neither minus at least one on the common candidate denominator. It is the coefficient from the difference of the two outcome equations, with covariance from aligned path scores. |
| wild_score_p | Exhaustive restricted-null Rademacher path-score bootstrap using the same design and CR1 studentization. Its interpretation relies on independence and symmetry across path scores, which correlated benchmark trajectories may challenge. |
