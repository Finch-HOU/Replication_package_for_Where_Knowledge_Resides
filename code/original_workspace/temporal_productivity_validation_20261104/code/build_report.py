from common import *
import html,re,argparse

def read(name):
 p=F/'data'/(name+'.csv');return pd.read_csv(p)if p.exists()and p.stat().st_size>2 else pd.DataFrame()
def num(x,n=4):return 'n.a.'if pd.isna(x)else f'{x:.2g}'if x!=0 and abs(x)<10**(-n)else f'{x:,.{n}f}'
def prob(x):return 'n.a.'if pd.isna(x)else'<0.000001'if x<1e-6 else f'{x:.6f}'
def interval(r,m):return '['+num(r.lo*m)+', '+num(r.hi*m)+']'
def md(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(str(x).replace('|','/').replace('\n',' ')for x in row)+' |'for row in rows)+'\n'
def write_table(name,headers,rows):
 o=F/'tables';o.mkdir(exist_ok=True);s=md(headers,rows);(o/(name+'.md')).write_text(s,encoding='utf8')
 def esc(v):return str(v).replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&').replace('<',r'$<$')
 tex=[r'\begin{table}[!htbp]\centering\footnotesize',r'\caption{'+esc(name.replace('_',' ').title())+'}',r'\resizebox{\linewidth}{!}{\begin{tabular}{'+'l'*len(headers)+'}',r'\toprule',' & '.join(esc(x)for x in headers)+r' \\',r'\midrule'];tex +=[' & '.join(esc(x)for x in row)+r' \\'for row in rows];tex +=[r'\bottomrule\end{tabular}}',r'\end{table}'];(o/(name+'.tex')).write_text('\n'.join(tex),encoding='utf8');return s

def main(tables_only=False):
 status=load(F/'diagnostics/numerical_validation.json');notes=load(F/'interpretation.json')
 if not tables_only:assert status['all_PPML_attempts_finished']and notes['status']=='final_assessment'
 a=read('all_LPM_coefficients');d=read('temporal_formal_differences');b=read('heterogeneous_proportional_benchmark');r=read('proportional_accounted_fraction');hb=read('home_general_productivity_benchmark');hr=read('home_general_productivity_accounted_fraction');pp=read('PPML_coefficients');pc=read('PPML_formal_relative_rate_contrasts');tables={}
 q=a[a.stage.eq('replication')&a.term.eq('Distributed_Home')&a.cluster.eq('capability')];rows=[]
 for mp in ['parent','direct']:
  for out in OUTCOMES:
   x=q[q.mapping.eq(mp)&q.model.eq(out)].iloc[0];rows.append([mp.title(),out,num(x.b*1e4),num(x.se*1e4),interval(x,1e4),prob(x.p),f'{int(x.N):,}',f'{int(x.nonzero_cells):,}',num(x.R2,6),f'{x.within_R2:.3g}'])
 tables['primary']=write_table('01_original_eleven_month_reproduction',['Mapping','Outcome','b x10k','SE x10k','95% CI x10k','Raw p','N','Events','R2','Within R2'],rows)
 q=a[a.stage.eq('trajectory_marker')&a.term.eq('Distributed_Home')&a.cluster.eq('capability')];rows=[]
 for mp in ['parent','direct']:
  for per in ['pre','current']:
   for out in OUTCOMES:
    x=q[q.mapping.eq(mp)&q.model.eq(per+'_'+out)].iloc[0];rows.append([mp.title(),per+' 10 months',out,num(x.b*1e4),num(x.se*1e4),interval(x,1e4),prob(x.p),f'{int(x.N):,}',f'{int(x.nonzero_cells):,}'])
 tables['temporal']=write_table('02_comparable_future_trajectory_marker',['Mapping','Period','Outcome','b x10k','SE x10k','95% CI x10k','Raw p','N per period','Events'],rows)
 q=d[d.comparison.eq('trajectory_marker_current_minus_pre')&d.cluster.eq('capability')];tables['differences']=write_table('03_formal_temporal_differences',['Mapping','Outcome','Current minus pre x10k','SE x10k','95% CI x10k','Raw p'],[[x.mapping.title(),x.outcome,num(x.b*1e4),num(x.se*1e4),interval(x,1e4),prob(x.p)]for x in q.itertuples()])
 q=a[a.stage.eq('gradient')&a.term.eq('Distributed')&a.cluster.eq('capability')];rows=[]
 for mp in ['parent','direct']:
  for per in ['pre','current']:
   for out in OUTCOMES:
    x=q[q.mapping.eq(mp)&q.model.eq('levels__'+per+'_'+out)].iloc[0];rows.append([mp.title(),per,out,num(x.b*1e5),num(x.se*1e5),interval(x,1e5),prob(x.p)])
 tables['levels']=write_table('04_monthly_unweighted_architecture_gradients',['Mapping','Period','Outcome','b /100k /SD','SE','95% CI','Raw p'],rows)
 q=d[d.comparison.eq('gradient_pooled_change')&d.cluster.eq('capability')];tables['pooled']=write_table('05_direct_pooled_architecture_change',['Mapping','Outcome','Change /100k /SD','SE','95% CI','Raw p','N pooled'],[[x.mapping.title(),x.outcome,num(x.b*1e5),num(x.se*1e5),interval(x,1e5),prob(x.p),f'{int(x.N_pooled):,}']for x in q.itertuples()])
 rows=[]
 for mp in ['parent','direct']:
  for out in ['specialist','no_specialist','total','signed']:
   qo='signed_observed'if out=='signed'else out;qp='signed_proportional'if out=='signed'else out+'_proportional';qr='signed_residual'if out=='signed'else out+'_residual';sub=b[b.mapping.eq(mp)&b.cluster.eq('capability')].set_index('statistic');obs,pr,res=sub.loc[qo],sub.loc[qp],sub.loc[qr];rows.append([mp.title(),out,num(obs.b*1e4),num(pr.b*1e4),num(res.b*1e4),num(res.se*1e4),interval(res,1e4),prob(res.p)])
 tables['proportional']=write_table('06_heterogeneous_common_proportional_benchmark',['Mapping','Outcome','Observed x10k','Implied common x10k','Residual x10k','Residual SE','Residual 95% CI','Residual raw p'],rows)
 q=r[r.cluster.eq('capability')];tables['fraction']=write_table('07_signed_accounted_fraction',['Mapping','Signed benchmark / observed','95% Fieller set','Lower','Upper'],[[x.mapping.title(),num(x.accounted_fraction),x.Fieller_set,num(x.Fieller_low),num(x.Fieller_high)]for x in q.itertuples()])
 q=b[b.cluster.eq('capability')&b.statistic.str.contains('delta')];tables['delta']=write_table('08_propensity_scaled_linear_responses',['Mapping','Statistic','b','SE','95% CI','Raw p'],[[x.mapping.title(),x.statistic,num(x.b),num(x.se),interval(x,1),prob(x.p)]for x in q.itertuples()])
 rows=[]
 if len(pp):
  q=pp[pp.term.eq('Distributed_Home')&pp.cluster.eq('capability')]
  for x in q.itertuples():rows.append([x.mapping.title(),x.regime,x.outcome,num(x.b),num(x.se),interval(x,1),prob(x.p),str(int(x.likelihood_support_N))if pd.notna(x.likelihood_support_N)else'n.a.',str(int(x.likelihood_support_events))if pd.notna(x.likelihood_support_events)else'n.a.'])
 tables['ppml']=write_table('09_relative_rate_components',['Mapping','Nuisance structure','Outcome','Log-rate b','SE','95% CI','Raw p','Likelihood N','Retained events'],rows or[['Both','Original pair FE','Not estimable','n.a.','n.a.','n.a.','n.a.','n.a.','n.a.']])
 rows=[]
 if len(pc):
  for x in pc[pc.term.eq('Distributed_Home')&pc.cluster.eq('capability')].itertuples():rows.append([x.mapping.title(),x.regime,num(x.b),num(x.se),interval(x,1),prob(x.p),str(x.variance_valid)])
 tables['ratecontrast']=write_table('10_stacked_relative_rate_contrasts',['Mapping','Nuisance structure','S minus No log-rate slope','SE','95% CI','Raw p','Valid variance'],rows or[['Both','Original pair FE','Not estimable','n.a.','n.a.','n.a.','False']])
 q=read('unconditional_rate_approximation');q=q[q.cluster.eq('capability')];tables['means']=write_table('11_unconditional_rate_approximation',['Mapping','Approximation statistic','b','SE','Raw p'],[[x.mapping.title(),x.statistic,num(x.b*(1e4 if x.statistic!='homogeneous_delta'else 1)),num(x.se*(1e4 if x.statistic!='homogeneous_delta'else 1)),prob(x.p)]for x in q.itertuples()])
 support=[]
 for mp in ['parent','direct']:
  c=load(F/'baseline_propensities_crossfit'/mp/'diagnostics.json');support.append([mp.title(),f"{c['sample']['cores']:,}",f"{c['sample']['candidate_pairs']:,}",f"{int(c['observed_pre_events'][0]):,} / {int(c['observed_pre_events'][1]):,}",num(c['out_of_fold_predicted_pre_events'][0],1)+' / '+num(c['out_of_fold_predicted_pre_events'][1],1),num(sum(x['log_score_improvement']for x in c['cross_validation']),1)])
 tables['support']=write_table('12_baseline_propensity_support',['Mapping','Cores','Frozen C pairs','Pre S / No events','OOF predicted events','OOF log-score gain vs global'],support)
 rows=[]
 for mp in ['parent','direct']:
  sub=hb[hb.mapping.eq(mp)&hb.cluster.eq('capability')].set_index('statistic')
  for out in ['specialist','no_specialist','total','signed']:
   obs=sub.loc['signed_observed'if out=='signed'else out];pr=sub.loc[out+'_proportional'];res=sub.loc[out+'_residual'];rows.append([mp.title(),out,num(obs.b*1e4),num(pr.b*1e4),num(res.b*1e4),num(res.se*1e4),interval(res,1e4),prob(res.p)])
 tables['home_proportional']=write_table('13_general_Home_productivity_counterbenchmark',['Mapping','Outcome','Observed x10k','Home-only implied x10k','Residual x10k','Residual SE','Residual 95% CI','Residual raw p'],rows)
 q=hr[hr.cluster.eq('capability')];tables['home_fraction']=write_table('14_general_Home_accounted_fraction',['Mapping','Implied / observed signed','95% Fieller set','Lower','Upper'],[[x.mapping.title(),num(x.accounted_fraction),x.Fieller_set,num(x.Fieller_low),num(x.Fieller_high)]for x in q.itertuples()])
 q=hb[hb.cluster.eq('capability')&hb.statistic.str.contains('delta')];tables['home_delta']=write_table('15_general_Home_proportional_responses',['Mapping','Statistic','b','SE','95% CI','Raw p'],[[x.mapping.title(),x.statistic,num(x.b),num(x.se),interval(x,1),prob(x.p)]for x in q.itertuples()])
 pa=read('PPML_all_numerical_attempts');tables['attempts']=write_table('16_PPML_all_numerical_attempts',['Mapping','Regime','Implementation','Minutes','Return status','Coefficient sets','Base input N'],[[x.mapping.title(),x.regime,('equivalent preconditioner' if x.namespace.endswith('accelerated') else 'certified same likelihood' if 'certified' in x.namespace else 'original'),num(x.elapsed_seconds/60,1),str(x.cap_returncode),int(x.exported_coefficient_sets),f'{int(x.profiled_input_rows):,}']for x in pa.itertuples()])
 index=['# Comparison tables','', '| Table | Markdown | LaTeX |','| --- | --- | --- |']
 for fp in sorted((F/'tables').glob('[0-9]*.md')):index.append('| '+fp.stem.replace('_',' ')+' | [View]('+fp.name+') | [Source]('+fp.with_suffix('.tex').name+') |')
 (F/'tables/INDEX.md').write_text('\n'.join(index)+'\n',encoding='utf8')
 if tables_only:log('COMPARISON TABLE PREFLIGHT COMPLETE; final memo remains gated');return
 contents=rf'''# Exploratory Validation: Temporal Comparability and General Productivity

Final research memo, 4 November 2026. This is a focused diagnostic exercise, not a revised working paper or a confirmatory causal analysis.

## 1. Research questions and assessment

{notes['overview']}

The two questions are whether the architecture-conditioned Home capability association differs from a comparable historical sorting pattern, and whether a common proportional response combined with heterogeneous baseline incidence can generate the positive specialist-minus-no-specialist LPM slope. All specified outcomes and both dependent mapping variants are retained.

## 2. Data, risk sets and timing

Recent dyads and unfamiliar destinations use the original frozen2021 C universe, mapped-Home gate and complete candidate sets. Membership, reference-input requirements and standardization are unchanged. Every original component and all five covariance matrices were freshly reproduced. The maximum coefficient error and covariance checks are archived in `results/*/replication/replication_checks.json`.

The complete C candidate-month risks use equal observation weights. There is no sampled-control or inverse-sampling expansion in this module.

The pre-period is **1 January-31 October2022**, ten months. The equal current window is **1 July2025-30 April2026**, ten months. Both contain304 calendar days. The original **1 July2025-31 May2026** eleven-month window remains the reference. No period was selected using coefficients. Adding December2021 would use knowledge observed after its outcome; full November2022 would include ChatGPT's release. Neither was used to manufacture an eleven-month pre-window.

Pairs remain in the frozen universe after entry, with all later first-incidence outcomes zero. This is not a surviving-risk hazard or aggregate publication production. Specialist status requires demonstrated destination experience in a strictly earlier publication year. A first-month bundle with any additional specialist has specialist priority. S and No are disjoint; total=S+No and signed=S-No. No does not establish absence of expertise or AI substitution.

### Reproduced eleven-month HomeGain x distributed-coverage results

Units below are first events per100,000 candidate-months for one original-risk SD of distributed coverage and0.10 capability gain. Events count nonzero cells; signed has one nonzero cell for every total event.

{tables['primary']}

## 3. Specifications and what is identifiable

Let i be a frozen research core, j a candidate destination, t a publication month, and o an outcome. U is distributed input coverage: collective destination-input coverage beyond the best single member. G is missing historical input coverage, D the original expertise distance, and S_h shared input coverage. All continuous primitives use the **same original risk-set SDs**. HomeGain is the frozen historical knowledge-basket exposure; DestinationGain is destination capability progress.

The original monthly model is

$$y^o_{{ijt}}=a^o_{{ij}}+b^o_{{it}}+c^o_{{jt}}+\sum_{{k\in\{{G,D,U,S_h\}}}}\eta^o_{{kt}}z^k_{{ij}}+\sum_k[\beta^o_{{kH}}z^k_{{ij}}HomeGain_{{it}}+\beta^o_{{kA}}z^k_{{ij}}DestinationGain_{{jt}}]+W^o_{{ijt}}+e^o_{{ijt}}.$$

W contains frozen log breadth and productivity x DestinationGain. Both gain sources enter jointly. Outcome-specific pair/core-month/destination-month effects absorb AI main effects. These are relative conditional gradients, not aggregate AI impacts.

Frozen U and future-window average gain x U are absorbed by pair FE. Including U x every month also absorbs U x period. These are structural identification failures, not estimated zero coefficients. Absolute monthly U gradients therefore use identical core-month/destination-month effects and G/D/shared-coverage month slopes, without pair FE or U-month slopes. The pooled20-month comparison adds pair FE and directly estimates U x current period; its point estimate equals the difference of separate-period gradients. All component and joint-covariance identities pass.

Detailed levels and direct pooled period contrasts are in [Table 4](tables/04_monthly_unweighted_architecture_gradients.md) and [Table 5](tables/05_direct_pooled_architecture_change.md).

{notes['static_gradient']}

## 4. Comparable historical future-marker exercise

The first ten observed post-period capability-path months are replayed onto the ten historical months by **relative month**, retaining the same frozen Home basket, candidate mask, gains, controls, scales and original pair/core-month/destination-month/modifier-month effects. Historical and current outcomes are fitted jointly with period-specific nuisance effects. Their formal difference uses the saved cross-period covariance, including overlap in cores and capability paths. This is equivalent to a fully interacted stacked-period model with period-specific pair effects.

This marker is future information applied to historical outcomes. It is **not a historical AI series**. Existing primary gain inputs have no valid pre-ChatGPT observations; the inspected upstream capability file begins in2024. No historical capability measurements are backfilled. The earlier cumulative-average marker remains a reference with different FE/units, not an equivalent placebo.

{tables['temporal']}

{tables['differences']}

{notes['temporal']}

This is not a parallel-trends test. Period length and model construction are comparable, while cohort age, accumulated expertise, calendar alignment and publication timing can differ. Jan1 first-bundle shares are about20.6% historically versus9.0-9.3% in the matched current window. Jan1 is not automatically an invalid date; no date-based observations are dropped. The common modifier-month controls absorb general monthly U gradients, but do not resolve every core-specific dating or opportunity change.

## 5. Heterogeneous historical baseline probabilities

Only January-October2022 outcomes train the probability surface. Empirical-Bayes Gamma-Poisson hierarchies combine destination, the existing frozen Home-field classification, and frozen-risk quartiles of G and U. Duplicate quantile boundaries are collapsed. Smoothing strengths maximize pre-period marginal likelihood. Core activity uses frozen2021 log production and breadth. Every target core is excluded from all fitted parameters for its prediction through five fixed core folds. Thus its own first-entry label never increases its predicted future first-incidence probability. All-zero cores remain eligible and have positive predictions. Core folds do not eliminate author overlap.

The initial own-history activity construction was stopped before interpreting new benchmark coefficients because irreversible first entry can generate mechanical mean reversion. Its predictions and interrupted attempts are preserved under diagnostics. The final benchmark uses only the cross-fitted construction. There is no post-outcome training or significance-based model choice.

Historical event counts, eligible cores/pairs and out-of-fold calibration are in [Table 12](tables/12_baseline_propensity_support.md).

The binned conditional surface is a transparent approximation, not a validated structural model of research attempts. Predictions are positive, calibrated out of fold in aggregate, and improve historical log scores over training-global-rate predictions in every fold. The maximum combined monthly probability is below0.06 and eleven times that maximum remains below one. Future first-incidence depletion is diagnosed without changing the sample or predictors: already-entered pairs are0.065-0.075% of frozen risks and carry roughly0.3-0.4% of predicted probability mass. Cohort aging and structural opportunity changes still remain possible.

## 6. Common proportional benchmark and relative rates

Let X=zU x HomeGain and p0_o be the cross-fitted historical monthly incidence probability. The restricted benchmark replaces the bare focal X term by delta*p0_o*X, with **one common delta across S and No**, while every original FE and other AI slope remains outcome-specific. Separate delta_S/delta_No estimates are also reported. This is a first-order proportional-gradient benchmark, not a causal productivity estimate or an exact log-link model.

Let r be X residualized on all original FE and the other nine AI interactions, and pi_o=r'(p0_o X)/(r'r). The implied signed LPM coefficient is delta*(pi_S-pi_No). The implementation uses these actual conditional projections, not unconditional mean rates. The residual is the observed signed coefficient minus its implied value. Covariance-aware cluster scores jointly cover the observed, common and mode-specific estimates; they reproduce the original component covariances under all five schemes. Baseline predictions are treated as fixed, so first-stage estimation uncertainty is not included.

{tables['proportional']}

Common versus mode-specific proportional responses are in [Table 8](tables/08_propensity_scaled_linear_responses.md). Untruncated signed accounted fractions and Fieller sets are in [Table 7](tables/07_signed_accounted_fraction.md).

A negative accounted fraction means the benchmark points in the opposite direction; it is not a negative fraction of variance explained. Fractions are not truncated to0-100%. Fieller sets retain ratio uncertainty. The unconditional historical-rate approximation below is separately labeled; it cannot replace the heterogeneous test. Signed coefficients/residuals are scaled x10,000; its delta is unscaled.

The explicitly labeled unconditional-rate approximation is retained in [Table 11](tables/11_unconditional_rate_approximation.md).

{notes['proportional']}

### General Home-driven production counterbenchmark

A common productivity response need not itself depend on distributed coverage. A change proportional to HomeGain alone can also project onto X if historical incidence differs across U. Therefore the second benchmark uses delta*p0_o*HomeGain, **with no structural U moderation**, and otherwise the identical nuisance effects, nine other AI controls, full candidate risks, restriction engine and covariance-aware projections. Now pi_o=r'(p0_o HomeGain)/(r'r). This economically distinct counterbenchmark was declared after the specified focal-X results but before observing any Home-only estimate; both are reported, with neither selected by significance. Its residual is not automatically a structural architecture effect.

{tables['home_proportional']}

Accounted fractions and common/mode-specific responses are in [Table 14](tables/14_general_Home_accounted_fraction.md) and [Table 15](tables/15_general_Home_proportional_responses.md).

{notes['home_proportional']}

### Log-link/PPML checks

The exact original pair-FE PPML fails under both mappings: after zero-margin/singleton profiling and separation, all30 AI regressors are collinear and the remaining observations are separated (return code2001). This is preserved as a failed estimator. It does not invalidate the LPM, and it prevents claiming an equivalent original-FE relative-rate replication.

Historical-offset alternatives replace unrestricted pair FE by log(p0_o). One retains type-specific core-month effects; another imposes a **shared core-month factor** across S/No, directly matching a common production-opportunity explanation. Both retain type-specific destination-month and all modifier-month slopes and both capability sources. The latter keeps both modes and every destination in every likelihood-informative total-core-month, including mode-specific zeros. They are different nuisance assumptions and are not substitutes for the original pair controls.

The type-specific log-mean specification is

$$\log\mu^o_{{ijt}}=\log p^o_{{ij,0}}+a^o_{{it}}+c^o_{{jt}}+\sum_k\eta^o_{{kt}}z^k_{{ij}}+Z'_{{ijt}}\beta^o,$$

where Z contains the same ten original AI interactions. The shared-factor model replaces a^o_it with a_it, so the common core-month opportunity factor cancels in log(mu_S)-log(mu_No). Its S-minus-No coefficient contrast is a proportional mode comparison under that restriction. The separate Total equation uses p0_S+p0_No as its offset; nonlinear Total coefficients are not constrained to equal the sum of the component coefficients.

Full zero-risk populations remain the declared universe. PPML analytically profiles zero FE groups, and official separation checks can remove more uninformative cells. Likelihood-support N and retained events below therefore differ from the full candidate denominator. Neither a successful-entry sample nor N after profiling is silently called the complete LPM sample. Signed outcomes can be negative and are not fitted by PPML; the formal S-minus-No **log-rate slope contrast** is the relative-rate diagnostic.

{tables['ppml']}

{tables['ratecontrast']}

{notes['relative_rate']}

All numerical attempts, elapsed time, convergence/separation return codes and partial outputs are reported in [Table 16](tables/16_PPML_all_numerical_attempts.md) and [the complete attempt CSV](data/PPML_all_numerical_attempts.csv). A computation time limit leaves the relative-rate question unresolved; it is not an estimated zero or evidence of nonexistence of an offset-model MLE.

Official PPMLHDFE is pinned in the project dependencies. An independently validated invertible nuisance preconditioner changes no model column space or original AI coefficient units. Further same-likelihood implementations reuse a **completed original** separation/full-rank certificate; acceptance requires matching its AI coefficients, capability covariance, likelihood, N and FE degrees of freedom. These numerical attempts are not additional empirical specifications. Supplementary covariance recomputation has a fixed 45-minute budget. Valid original capability-cluster fits remain reportable even if all supplementary schemes cannot be completed. Other one-way covariances and inclusion-exclusion composites are reported only where their inputs are available and verified. No five-scheme PPML robustness is inferred from capability clustering alone. Negative variances and non-PSD matrices remain diagnostic failures, never clipped. Stata nested-FE finite-sample corrections differ from the Python full-FE CR1; the small-path t reference is explicitly retained. The official separation guide and estimator source are archived with dependency hashes.

## 7. Competing explanations and limitations

{notes['competing_table']}

The two mappings are dependent variants and ten correlated capability paths do not create independently assigned treatment. First-stage probability uncertainty, noisy demonstrated knowledge, evolving prior-year specialist histories, publication lags and unobserved activity opportunities remain. Coauthorship does not identify integration, specialist absence does not identify substitution, and capability proxies do not measure actual AI use. No claim is made that the total LPM gradient is an aggregate decline in innovation production.

Because U equals collective coverage minus best-member coverage, holding G fixed makes higher U equivalent to lower best-member coverage. This contrast cannot separately identify benefits of knowledge distribution and consequences of weaker coverage by the strongest member. The equivalent parameterization is not counted as independent corroboration. Fixed core membership also allows additional authors on outcome publications; it does not establish an unchanged organizational workforce.

The HomeGain range is approximately0.036-0.661 and the candidate-weighted within-core temporal SD is about0.162-0.165, so the illustrative0.10-gain units have support. Distributed coverage has an original-risk SD of0.01473 (Parent) or0.01394 (Direct); a one-SD contrast is not one percentage point. The squared cluster-score concentration diagnostics are descriptive and do not increase the number of independent shocks. Historical propensity calibration need not remain valid for the older current-period cores; treating those predictions as fixed understates first-stage uncertainty.

Covariance validity is assessed separately from an individual contrast's positive variance. **41 of 320 full LPM coefficient covariance matrices are indefinite**, all under the Home/capability composite. Two auxiliary temporal coefficients have negative individual variances: Parent pooled total-entry architecture change and Parent historical-marker Specialist entry. Their SEs, intervals and p-values remain unavailable. The full joint temporal-marker covariance is also indefinite under the capability/core and Home/capability composites. These composite results are retained as diagnostics, without PSD repair; positive contrast variances do not make their full matrices valid. The compact tables use the original capability covariance. The benchmark residual patterns remain under the four PSD covariance families: capability, Home field, core, and capability/core.

For PPML, the Direct/shared capability/core composite is indefinite (minimum eigenvalue approximately -0.000914). Its focal contrast variance is positive and its calculated p-value is preserved as a diagnostic, not counted as clean additional inference. The completed one-way PPML covariances are PSD up to numerical rounding. CSV files carry separate individual-variance and whole-matrix diagnostic flags; the eigenvalue tolerance is relative to matrix scale at 1e-8. See [covariance diagnostics](diagnostics/LPM_covariance_diagnostics.csv), [PPML contrasts](data/PPML_formal_relative_rate_contrasts.csv), [expertise-label comparability](diagnostics/temporal_expertise_label_comparability.csv), [gain variation](diagnostics/HomeGain_units_and_variation.csv) and [capability score concentration](diagnostics/capability_score_concentration.csv).

All five **LPM** covariance alternatives, available PPML covariance schemes, every component, omitted variable, failure and convergence record are retained in CSV/JSON. No path, window, lag or subgroup was chosen for significance. The narrow tests are exploratory; raw p-values are not confirmatory research-wide error control. The unweighted temporal gradient, replayed future marker, propensity-scaled linear response and PPML log-rate coefficient have different estimands and units.

## 8. Implications for the current story

{notes['decision']}

**Temporal distinctiveness:** {notes['answer_temporal']}

**General proportional productivity:** {notes['answer_productivity']}

**Knowledge architecture:** {notes['answer_architecture']}

**Reasonable claim and remaining identification:** {notes['answer_claim']}

The complete source-linked estimates, exact risk/scale metadata, covariance identities, convergence failures and decision registry accompany this memo. No revised working paper is produced.

Diagnostic figures: [comparable temporal marker](figures/01_comparable_temporal_marker.png), [component decomposition](figures/02_proportional_benchmark.png), and [both common-productivity benchmarks](figures/03_two_productivity_benchmarks.png). Definitions and execution instructions are in [the data dictionary](DATA_DICTIONARY.md) and [README](README.md).
'''
 contents=contents.replace('frozen2021','frozen 2021').replace('per100','per 100').replace('and0.','and 0.').replace('in2024','in 2024').replace('pooled20','pooled 20').replace('all30','all 30').replace('to0-','to 0-')
 contents=re.sub(r'\b(through|in|versus|about|with|and|below|per|SE|than|both|contain)(?=[0-9])',r'\1 ',contents)
 (F/'Detailed_Methods_and_Results.md').write_text(contents,encoding='utf8')
 from concise_memo import write_memo
 write_memo(tables);log('CONCISE MEMO, TECHNICAL APPENDIX AND COMPARISON TABLES BUILT')

if __name__=='__main__':a=argparse.ArgumentParser();a.add_argument('--tables-only',action='store_true');x=a.parse_args();main(x.tables_only)
