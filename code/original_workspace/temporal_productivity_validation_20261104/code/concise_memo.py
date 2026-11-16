"""Reader-facing summary; complete tables and technical appendix stay intact."""
from common import *

def write_memo(tables):
 rows=[]
 for file,label in [('heterogeneous_proportional_benchmark','Common response to X'),('home_general_productivity_benchmark','Common Home-only response')]:
  a=pd.read_csv(F/'data'/(file+'.csv'));a=a[a.cluster.eq('capability')].set_index(['mapping','statistic'])
  for mp in ['parent','direct']:
   obs=a.loc[(mp,'signed_observed')];imp=a.loc[(mp,'signed_proportional')];r=a.loc[(mp,'signed_residual')]
   rows.append(f'| {mp.title()} | {label} | {obs.b*1e4:.4f} | {imp.b*1e4:.4f} | {r.b*1e4:.4f} | {r.se*1e4:.4f} | [{r.lo*1e4:.4f}, {r.hi*1e4:.4f}] | {r.p:.6f} | {imp.b/obs.b*100:.2f}% |')
 benchmark='| Mapping | Benchmark | Observed | Implied | Residual | Residual SE | Residual 95% CI | Raw p | Implied / observed |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n'+'\n'.join(rows)
 text=rf'''# Exploratory Validation: Temporal Comparability and General Productivity

4 November 2026. Focused exploratory validation; no working-paper revision or significance-oriented specification search.

## 1. Research questions and assessment

The original positive HomeGain x Distributed Input Coverage coefficient in Specialist-minus-No-Specialist entry is reproduced. **Neither of two estimated common proportional productivity benchmarks explains it.** Three offset PPML models also show different relative-rate responses across entry modes.

**Temporal distinctiveness is considerably less secure.** The matched historical point estimates differ, but the formal period difference is borderline for Parent and imprecise for Direct. These findings support a conditional knowledge-architecture/entry-mode association; they do not identify an AI-induced change in organizational coordination or expertise sourcing.

This memo summarizes the completed evidence. [The technical appendix](Detailed_Methods_and_Results.md), [16 complete comparison tables](tables/INDEX.md), [LPM coefficients](data/all_LPM_coefficients.csv), [definitions](DATA_DICTIONARY.md) and [execution instructions](README.md) preserve all components, alternative inference, null results and failures.

## 2. Data and sample construction

The sample preserves recent dyads, frozen-2021 Class C candidate destinations, original Parent/Direct mappings, gain12_l6, standardization and equal candidate-month weights. Parent includes **602,919 cores and 102,614,232 candidate pairs**; Direct includes **597,077 cores and 70,422,255 pairs**. No current-period survival or successful-entry restriction is imposed on the LPM.

The pre-ChatGPT window is **1 January-31 October 2022**. The equal current window is **1 July 2025-30 April 2026**. Both have ten months and 304 days. The original **1 July 2025-31 May 2026**, eleven-month specification remains the reference. December 2021 would violate the frozen-2021 timing rule; full November 2022 would include ChatGPT's release. Windows were not selected using coefficients.

The first joint-domain month is the event. If any paper in that month has an additional collaborator with strictly prior-year destination experience, the bundle is Specialist (S); otherwise it is No Specialist (N). Categories are mutually exclusive, Total=S+N and Signed=S-N. All later first-incidence outcomes are zero while the pair stays in the panel. This is not a surviving-risk hazard. No Specialist includes more than exact-core publications and does not establish absence of human expertise.

## 3. Exact specifications

Let i denote a fixed core, j a candidate destination, t a month, and o an outcome. U is collective reference-input coverage beyond the best single member; G is the uncovered input share, D expertise distance, and Sh shared input coverage. All are predetermined and standardized with original complete-risk scales. H denotes HomeGain from the frozen Home publication basket; A denotes DestinationGain. The original model is

$$y^o_{{ijt}}=\alpha^o_{{ij}}+\eta^o_{{it}}+\gamma^o_{{jt}}+\sum_{{k\in\{{G,D,U,Sh\}}}}\lambda^o_{{kt}}z^k_{{ij}}+\sum_k\left[\beta^o_{{kH}}z^k_{{ij}}H_{{it}}+\beta^o_{{kA}}z^k_{{ij}}A_{{jt}}\right]+W'_{{ijt}}\pi^o+\varepsilon^o_{{ijt}}.$$

W contains frozen log breadth and log productivity x DestinationGain. Pair, core-month and destination-month effects and all four modifier-month slopes are retained. Main capability effects are absorbed. The focal interaction is **X=zU x HomeGain**; its signed coefficient is a conditional destination-entry gradient, not an aggregate AI productivity effect.

Frozen U is pair-absorbed. Absolute U gradients therefore use core-month/destination-month effects and G/D/Sh month slopes; their pooled period-change model adds pair effects. Including unrestricted U-month slopes would absorb that change. A static future-average gain x U is also pair-absorbed.

The FE-preserving historical sorting test instead replays the **same first ten observed future capability-path months** onto the ten historical months by relative month. Historical/current equations have identical X, masks and scales, period-specific nuisance effects, and joint cross-period covariance. This is future sorting information, not historical AI exposure or a parallel-trends test. No genuine pre-ChatGPT capability series is available in the inspected inputs.

## 4. Main results and temporal comparability

All four original outcomes and all five original covariance calculations were freshly reproduced. Below, coefficient/SE/CI values are multiplied by 10,000: first events per 100,000 candidate-months for a one-SD U contrast and 0.10 gain. Events count nonzero cells; Signed has one for every Total event. Compact tables use capability clustering with ten paths and a t9 reference.

{tables['primary']}

The positive Signed result comes mainly from the negative No-Specialist gradient. Specialist entry does not show a precisely established increase, and the Total gradient is also negative. This is neither evidence of greater specialist integration nor a pure unchanged-total mode conversion. Total's conditional gradient is not an aggregate production decline.

The comparable ten-month historical/current results are:

{tables['temporal']}

The **formal current-minus-historical Signed differences** are 0.1520 (SE 0.0672, p=0.049995) for Parent and 0.1610 (SE 0.0982, p=0.135656) for Direct, in the same scaled units. Parent's interval is approximately [0.0000043, 0.3039]; its lower bound is exceptionally close to zero. Core clustering gives p=0.075361. Direct is imprecise under the completed one-way schemes. All component differences and exact intervals are in [Table 3](tables/03_formal_temporal_differences.md).

The unweighted pooled Signed U-gradient change is imprecise: p=0.795246/0.697383. Total's U-gradient change is positive, with p=0.021603/0.168056; that is a different, non-AI estimand. [Tables 4-5](tables/05_direct_pooled_architecture_change.md) retain these results.

Comparability remains incomplete. About **19.8-20.2%** of matched current first-entry bundles already have a core member with strictly prior-year destination experience, versus zero historically. Frozen-2021 C does not imply current unfamiliarity. Jan1 bundle shares also differ, approximately 20.6% versus 9.0-9.3%. No observations are dropped on either basis. Cohort aging, expertise accumulation, calendar alignment and publication dating remain alternatives to an AI interpretation.

## 5. General proportional productivity alternative

Baseline monthly probabilities p0_S and p0_N are trained only on January-October 2022. A Gamma-Poisson empirical-Bayes hierarchy uses destination, frozen Home field, and frozen-risk G/U quantile cells, collapsing repeated boundaries. Activity predictions use frozen log publication count and breadth. Five fixed core folds exclude each target core's labels from **all** fitted parameters; all-zero cores retain positive predictions. Historical own-label activity predictions were superseded before benchmark coefficients were inspected because irreversible first entry can generate mechanical mean reversion. The interrupted attempt remains archived.

The first restricted model replaces the bare focal X term with delta*p0_o*X, sharing delta across S/N while preserving outcome-specific original FE and the other nine AI controls. The economically distinct second model uses delta*p0_o*HomeGain, allowing a general Home-driven productivity response **without structural U moderation**. It was declared after the first benchmark but before its own estimates; both remain reported.

If r is X residualized on all original nuisance terms, the implied Signed slope is delta*[r'(p0_S F)-r'(p0_N F)]/(r'r), with F=X or HomeGain. Thus the calculation uses the actual conditional projection, not unconditional mean rates. Observed, implied and residual slopes use joint covariance-aware scores. Historical probabilities are treated as fixed; their estimation uncertainty is **not included** in these intervals.

{benchmark}

Coefficient/SE/CI columns use the same x10,000 scale. The first benchmark implies approximately -10% of the observed Signed slope, and the more general Home-only benchmark approximately -2%. A negative ratio means opposite direction, not a negative fraction of variance explained. Ratios are untruncated; [Fieller intervals](tables/07_signed_accounted_fraction.md) and [Home-only intervals](tables/14_general_Home_accounted_fraction.md) retain ratio uncertainty.

The residuals remain positive under the **four PSD benchmark covariance families**: capability, Home field, core and capability/core. All four give p<0.013 for the focal-X benchmark and p<0.018 for Home-only. The indefinite Home/capability composites remain diagnostic. Separate proportional-response coefficients, every component and the labeled unconditional-rate approximation remain in Tables 6-8 and 11-15 and the underlying CSVs.

These tests contradict the two fitted common-factor approximations. They do not exclude every general opportunity process. Historical calibration need not remain valid for older cores, and a binned probability surface can miss economically relevant heterogeneity.

## 6. Relative-rate specifications and competing explanations

Original pair-FE PPML fails in both mappings through separation/collinearity (return code 2001); it cannot supply a same-FE relative-rate replication. Offset alternatives estimate

$$\log\mu^o_{{ijt}}=\log p^o_{{ij,0}}+a^o_{{it}}+c^o_{{jt}}+\sum_k\theta^o_{{kt}}z^k_{{ij}}+Z'_{{ijt}}b^o,$$

where Z contains the same ten AI interactions. One model has type-specific core-month effects; the other imposes a **shared** core-month effect across S/N, so a common multiplicative opportunity factor cancels in their log-rate contrast. Both replace unrestricted pair FE with a historical offset, a substantive change. Total's separately fitted nonlinear coefficient is not constrained to equal the sum of component coefficients.

{tables['ratecontrast']}

These contrasts imply illustrative S/N rate-ratio changes of about 7.0%, 12.8% and 7.9%, respectively, per 0.10 HomeGain x one SD of U. Specialist coefficients remain imprecise; the negative No-Specialist component again drives the difference. Direct's type-specific Total coefficient is -0.2581 (p=0.106737), not evidence of a zero total response. [Table 9](tables/09_relative_rate_components.md) reports all component coefficients, likelihood N and retained events.

Three models converge with all their positive input events retained. Zero FE groups are analytically profiled, so likelihood support is smaller than the declared full universe; it is not silently labeled the full LPM sample. Parent's type-specific offset remains computationally unresolved. Capability and Home-field covariances are completed for all three models; Direct/shared also has core covariance. The focal contrasts stay positive under these one-way alternatives (raw p=0.005152-0.027267). Its additional capability/core composite is indefinite and only diagnostic. No PPML model has all five usable schemes.

All attempts, time limits and numerical verification are in [Table 16](tables/16_PPML_all_numerical_attempts.md). Same-likelihood accelerators must reproduce the original strict-fit points, capability covariance, N, likelihood and FE degrees of freedom. Partial covariance availability does not imply a zero effect or MLE nonexistence. Stata's nested-FE finite-sample convention differs from the original Python full-FE correction.

| Diagnostic | General proportional account | Architecture/mode interpretation |
| --- | --- | --- |
| Specialist | No clear increase. | Does not establish greater specialist integration. |
| No Specialist | Negative gradient could reflect opportunity changes. | Main source of the positive Signed association. |
| Total | Pure unchanged-total conversion is incomplete. | Conditional allocation gradient; not aggregate production. |
| Signed | Baseline rate differences can mechanically produce it. | Reproduced, but insufficient alone. |
| Common-factor benchmarks | Neither fitted benchmark accounts for it. | Positive conditional residual survives four PSD families. |
| Relative rates | Original pair FE fail; offset assumptions differ. | Completed one-way contrasts differ even with a shared opportunity factor. |
| Temporal comparison | Aging, expertise and dating remain alternatives. | Distinctive AI timing is not firmly established. |

## 7. Measurement, covariance and identification limits

The mappings are dependent variants; ten correlated capability paths do not create independently assigned treatment. Capability gains do not measure actual AI use. Coauthorship does not identify knowledge integration, and specialist absence does not establish substitution. Fixed cores may publish with additional authors and do not establish an unchanged organizational workforce.

At fixed G, higher U is mechanically lower best-member coverage. This contrast cannot separately identify benefits of distribution and consequences of weaker best-member coverage; the equivalent parameterization is not independent corroboration. Historical probability uncertainty, author overlap across folds and mode-specific opportunity changes remain unresolved.

Whole-matrix validity is separate from positive individual variance. **41 of 320 full per-outcome LPM coefficient covariance matrices are indefinite**, all Home/capability composites; two auxiliary temporal variances are negative. Joint temporal-marker capability/core and Home/capability matrices are also indefinite. Direct/shared PPML's capability/core composite has minimum eigenvalue about -0.000914 despite a positive focal variance. These results are preserved as diagnostics, without PSD repair; negative-variance intervals remain unavailable. Completed one-way PPML covariances are PSD up to numerical rounding. CSVs separately flag contrast variance and matrix PSD status, using a scale-relative 1e-8 eigenvalue tolerance.

All limited tests and raw p-values remain exploratory. No subgroup, lag, window or capability path was chosen to improve significance. The [technical appendix](Detailed_Methods_and_Results.md) and [matrix diagnostics](diagnostics/LPM_covariance_diagnostics.csv) provide the full audit.

## 8. Direct assessment and implications

- **Temporally distinctive?** Suggestive, not established. Historical point estimates differ, but the formal difference is borderline for Parent, imprecise for Direct, and sensitive to inference. This is not parallel trends.
- **Explained by general proportional productivity?** Neither estimated benchmark explains the positive Signed gradient. Offset PPML also finds different proportional mode responses under its assumptions. More general opportunity, maturation and measurement processes remain possible.
- **Remaining architecture evidence?** Yes: a conditional distribution-of-prior-knowledge/relative-entry-mode association, driven largely by No-Specialist entry. There is no precisely established increase in specialist-assisted entry.
- **Reasonable claim?** Historical knowledge distribution is associated with different first-entry patterns along Home capability trajectories. Actual AI adoption, causal coordination-cost changes, expertise substitution and knowledge integration remain unidentified. Retain the narrow exploratory association; the stronger AI-induced organization-of-expertise mechanism remains a hypothesis.

Figures: [temporal comparison](figures/01_comparable_temporal_marker.png) and [both productivity benchmarks](figures/03_two_productivity_benchmarks.png). All code, estimates, nulls, failed approaches and source checks accompany this memo. No revised working paper is produced.
'''
 (F/'Exploratory_Validation_Temporal_and_Productivity.md').write_text(text,encoding='utf8')
