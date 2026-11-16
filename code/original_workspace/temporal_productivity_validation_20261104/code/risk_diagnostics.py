from common import *

def main():
 rows=[];depletion=[]
 for mp in ['parent','direct']:
  p=Panel('recent','dyad',mp,classes='C');pred=np.load(F/'baseline_propensities_crossfit'/mp/'p0.npy',mmap_mode='r');tot=np.zeros(2)
  for a in range(0,p.I,8192):
   b=min(a+8192,p.I);tot+=(pred[a:b]*p.mask[a:b,:,None]).sum((0,1))
  w=p.c.execute(f"SELECT w.* FROM read_parquet('{(p.u/'analysis_works.parquet').as_posix()}') w SEMI JOIN selected USING(g) WHERE w.mo BETWEEN 744 AND 796 AND w.mo=w.first_mo").fetchdf();w=p.localize(w);w['S']=w.current_experts>w.current_core_experts;e=w.groupby(['i','j0','mo']).agg(specialist=('S','max'),any_jan1=('jan1','max'),all_jan1=('jan1','all'),papers=('work_id','size')).reset_index()
  prior=e[e.mo<786];assert not prior.duplicated(['i','j0']).any();mass=pred[prior.i.to_numpy(),prior.j0.to_numpy()].sum(0);depletion.append({'mapping':mp,'candidate_pairs':int(p.mask.sum()),'pairs_entered_Jan2022_Jun2025':len(prior),'prior_entry_share_of_frozen_pairs':len(prior)/p.mask.sum(),'share_of_specialist_p0_mass_on_already_entered_pairs':mass[0]/tot[0],'share_of_no_specialist_p0_mass_on_already_entered_pairs':mass[1]/tot[1],'diagnostic_only_not_a_sample_or_propensity_adjustment':True,'pre_AI_predictions_not_updated_with_post_2022_labels':True})
  for name,start,end in [('pre10',744,753),('current10',786,795),('current11',786,796)]:
   q=e[e.mo.between(start,end)]
   for out in ['specialist','no_specialist','total']:
    a=q[q.specialist]if out=='specialist'else q[~q.specialist]if out=='no_specialist'else q;rows.append({'mapping':mp,'window':name,'outcome':out,'first_events':len(a),'any_first_bundle_Jan1':int(a.any_jan1.sum()),'all_first_bundle_Jan1':int(a.all_jan1.sum()),'Jan1_share':float(a.any_jan1.mean()),'multi_paper_first_bundles':int(a.papers.gt(1).sum()),'calendar_January_event_share':float((a.mo%12==0).mean()),'Jan1_is_not_automatically_an_invalid_date':True})
  p.c.close()
 (F/'diagnostics').mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(F/'diagnostics/temporal_event_date_quality.csv',index=False);pd.DataFrame(depletion).to_csv(F/'diagnostics/frozen_risk_depletion.csv',index=False);log('DATE AND DEPLETION DIAGNOSTICS COMPLETE')

if __name__=='__main__':main()
