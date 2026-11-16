"""Descriptive timing comparability; no new outcomes, regressions or selection."""
from common import *

def main():
 rows=[];gain=[]
 for mp in ['parent','direct']:
  p=Panel('recent','dyad',mp,classes='C')
  w=p.c.execute(f"SELECT w.* FROM read_parquet('{(p.u/'analysis_works.parquet').as_posix()}') w SEMI JOIN selected USING(g) WHERE (w.mo BETWEEN 744 AND 753 OR w.mo BETWEEN 786 AND 796) AND w.mo=w.first_mo").fetchdf();w=p.localize(w)
  w['additional_specialist']=w.current_experts>w.current_core_experts
  w['core_expert']=w.current_core_experts>0
  e=w.groupby(['i','j0','mo']).agg(specialist=('additional_specialist','max'),any_core_prior_year_expert=('core_expert','max'),all_papers_core_prior_year_expert=('core_expert','all'),papers=('work_id','size')).reset_index()
  for name,lo,hi,T in [('pre10',744,753,10),('current10',786,795,10),('current11',786,796,11)]:
   a=e[e.mo.between(lo,hi)]
   for out in ['total','specialist','no_specialist']:
    b=a if out=='total'else a[a.specialist]if out=='specialist'else a[~a.specialist]
    rows.append({'mapping':mp,'window':name,'outcome':out,'N_candidate_months':int(p.mask.sum())*T,'first_events':len(b),'events_per_100k':len(b)/(p.mask.sum()*T)*1e5,'specialist_share_of_all_first_entries':float(a.specialist.mean()),'events_with_any_core_prior_year_destination_expert':int(b.any_core_prior_year_expert.sum()),'core_prior_year_expert_share':float(b.any_core_prior_year_expert.mean()),'baseline_C_does_not_imply_current_core_has_no_expert':True,'descriptive_only_no_outcome_conditioned_primary_model':True})
  weights=p.mask.sum(1).astype(float);hg=p.ha
  mean=np.average(hg,axis=0,weights=weights)
  within=hg-hg.mean(1,keepdims=True)
  gain.append({'mapping':mp,'gain_min':float(hg.min()),'gain_max':float(hg.max()),'candidate_weighted_HomeGain_mean':float(mean.mean()),'candidate_weighted_temporal_SD':float(np.sqrt(np.average((within**2).mean(1),weights=weights))),'HomeGain_main_effect_absorbed_by_core_month':True,'illustrative_table_gain_change':.10,'capability_paths':len(p.paths),'focal_primitive_SD':load(F/'results'/mp/'replication/sample.json')['standardization']['Distributed']['sd']})
  p.c.close();log('EXPERTISE TIMING DIAGNOSTIC COMPLETE',mp)
 pd.DataFrame(rows).to_csv(F/'diagnostics/temporal_expertise_label_comparability.csv',index=False)
 pd.DataFrame(gain).to_csv(F/'diagnostics/HomeGain_units_and_variation.csv',index=False)

if __name__=='__main__':main()
