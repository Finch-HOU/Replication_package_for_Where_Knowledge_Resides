from common import *
from scipy.stats import t as td
from restriction_tools import linear_statistics

def uncertainty(b,var,ng):
 ok=bool(np.isfinite(var)and var>0);se=float(np.sqrt(var))if ok else None;c=float(td.ppf(.975,ng-1))if ng>1 else np.nan;return {'b':float(b),'se':se,'lo':float(b-c*se)if ok else None,'hi':float(b+c*se)if ok else None,'p':float(2*td.sf(abs(b/se),ng-1))if ok else None,'variance_valid':ok,'clusters':ng}

def export(name,rows):
 d=pd.DataFrame(rows)
 if len(d)and name in ['heterogeneous_proportional_benchmark','home_general_productivity_benchmark','proportional_accounted_fraction','home_general_productivity_accounted_fraction','unconditional_rate_approximation']:
  for out in ['specialist','no_specialist','total']:
   counts={mp:load(F/'results'/mp/'replication'/(out+'.json'))['meta']['nonzero_cells']for mp in ['parent','direct']};d['current_'+out+'_first_events']=d.mapping.map(counts)
 if len(d)and name=='temporal_formal_differences':
  for period in ['pre','current']:
   d[period+'_nonzero_events']=[load(F/'results'/r.mapping/'trajectory_marker'/(period+'_'+r.outcome+'.json'))['meta']['nonzero_cells']for r in d.itertuples()]
 (F/'data').mkdir(exist_ok=True);d.to_csv(F/'data'/(name+'.csv'),index=False);return d

def fieller(num,den,V,ng):
 c=td.ppf(.975,ng-1);aa=den*den-c*c*V[1,1];bb=-2*(num*den-c*c*V[0,1]);cc=num*num-c*c*V[0,0];disc=bb*bb-4*aa*cc;point=num/den if den else None
 if disc<0:return {'accounted_fraction':point,'Fieller_set':'all real numbers'if aa<0 else'empty diagnostic set','Fieller_low':None,'Fieller_high':None}
 roots=sorted([(-bb-np.sqrt(disc))/(2*aa),(-bb+np.sqrt(disc))/(2*aa)])if aa else[None,None]
 return {'accounted_fraction':point,'Fieller_set':'bounded interval'if aa>0 else'unbounded outside interval','Fieller_low':roots[0],'Fieller_high':roots[1]}

def main():
 allrows=[];changes=[];bench=[];ratios=[];meanbench=[];models=[]
 for mp in ['parent','direct']:
  for stage in ['replication','gradient','trajectory_marker','proportional','home_proportional']:
   root=F/'results'/mp/stage
   for pp in sorted(root.glob('*.json')):
    r=load(pp)
    if 'meta'not in r or 'status'not in r:continue
    mm=r['meta'];models.append({'mapping':mp,'stage':stage,'model':pp.stem,'status':r['status'],**{k:mm.get(k)for k in ['N','teams','periods','rank','condition','R2','within_R2','nonzero_cells','mean_outcome','df_FE']},'source':str(pp)})
    for c0 in r.get('coefficients',[]):
     c=c0.copy()
     c['variance_valid']=bool(pd.notna(c.get('se'))and c.get('se',0)>0)
     c.update(covariance_diagnostic(r['covariance'][c['cluster']]))
     if stage=='home_proportional':c['source_term']=c['term'];c['term']=c['term'].replace('_x_focal_','_x_HomeGain_')
     allrows.append({'mapping':mp,'stage':stage,'model':pp.stem,**c,**{k:mm.get(k)for k in ['N','teams','periods','R2','within_R2','nonzero_cells','mean_outcome','df_FE']},'source':str(pp),'auxiliary_basis_not_an_additional_primary_hypothesis':stage in ['proportional','home_proportional']})
  for stage,prefix,term in [('gradient','levels__','Distributed'),('trajectory_marker','','Distributed_Home')]:
   root=F/'results'/mp/stage;pp=root/(prefix+'joint_metadata.json')
   if not pp.exists():continue
   md=load(pp);j=np.load(root/(prefix+'joint_covariances.npz'));K=len(md['regressors']);k=md['regressors'].index(term);yn=md['outcomes']
   for out in OUTCOMES:
    L=np.zeros(K*len(yn));L[yn.index('current_'+out)*K+k]=1;L[yn.index('pre_'+out)*K+k]=-1;b=float(L@j['beta'].reshape(-1))
    for cl in SCHEMES:
     ng=md['home_clusters']if cl=='homefield'else md['teams']if cl=='team'else md['paths'];changes.append({'mapping':mp,'comparison':stage+'_current_minus_pre','outcome':out,'term':term,'cluster':cl,**uncertainty(b,float(L@j[cl]@L),ng),**covariance_diagnostic(j[cl]),'N_per_period':md['N'],'pre_start':'2022-01-01','pre_end':'2022-10-31','current_start':'2025-07-01','current_end':'2026-04-30','months_per_period':10,'uses_joint_cross_period_covariance':True,'source':str(root/(prefix+'joint_covariances.npz'))})
  root=F/'results'/mp/'gradient'
  for out in OUTCOMES:
   pp=root/('pooled_change__'+out+'.json')
   if not pp.exists():continue
   rr=load(pp)
   for cc in rr.get('coefficients',[]):changes.append({'mapping':mp,'comparison':'gradient_pooled_change','outcome':out,**cc,**covariance_diagnostic(rr['covariance'][cc['cluster']]),'N_per_period':rr['meta']['N']//2,'N_pooled':rr['meta']['N'],'pre_start':'2022-01-01','pre_end':'2022-10-31','current_start':'2025-07-01','current_end':'2026-04-30','months_per_period':10,'uses_direct_pooled_period_interaction':True,'source':str(pp)})
  root=F/'results'/mp/'proportional';pp=root/'restricted_benchmark.json'
  if pp.exists():
   r=load(pp);pt=np.array(r['primitive_points']);pi=r['projection_loading'];linear=linear_statistics(pi);qq=np.array(load(root/'basis__signed.json')['sufficient_statistics']['Q']);loadings=np.linalg.solve(qq[:10,:10],qq[:10,10:])[5]
   for nm,k in [('specialist',0),('no_specialist',1)]:linear[nm+'_proportional']=np.array([0,0,loadings[k],0,0.]);linear[nm+'_residual']=linear[nm]-linear[nm+'_proportional']
   linear['total_proportional']=linear['specialist_proportional']+linear['no_specialist_proportional'];linear['total_residual']=linear['total']-linear['total_proportional'];assert abs(loadings[0]-loadings[1]-pi)<1e-10
   for cl,V in r['covariance'].items():
    V=np.array(V);ng=r['home_clusters']if cl=='homefield'else r['cores']if cl=='team'else r['capability_paths']
    for name,L in linear.items():bench.append({'mapping':mp,'statistic':name,'cluster':cl,**uncertainty(float(L@pt),float(L@V@L),ng),**covariance_diagnostic(V),'N_per_outcome':r['N_per_outcome'],'projection_loading':pi,'first_stage_uncertainty_included':False,'baseline_prediction':'core_crossfit_2022_with_frozen2021_core_covariates','source':str(pp)})
    L1=linear['signed_proportional'];L2=linear['signed_observed'];ll=np.vstack([L1,L2]);ratios.append({'mapping':mp,'cluster':cl,**fieller(float(L1@pt),float(L2@pt),ll@V@ll.T,ng),**covariance_diagnostic(V),'fraction_not_truncated_to_zero_one':True,'first_stage_predictions_fixed':True,'source':str(pp)})
    # Labeled unconditional-rate approximation, never the main heterogeneous test.
    hist=load(F/'baseline_propensities_crossfit'/mp/'diagnostics.json');rates=np.array(hist['observed_pre_events'])/(hist['sample']['candidate_pairs']*10);VV=V[:2,:2]
    if np.linalg.eigvalsh(VV).min()>0:
     iv=np.linalg.inv(VV);ww=(iv@rates)/(rates@iv@rates);dd=float(ww@pt[:2]);load0=float(rates[0]-rates[1]);Lc=np.array([1.,-1.])-load0*ww;Lg=load0*ww
     for name,L in [('homogeneous_implied_signed',Lg),('homogeneous_residual_signed',Lc),('homogeneous_delta',ww)]:meanbench.append({'mapping':mp,'statistic':name,'cluster':cl,**uncertainty(float(L@pt[:2]),float(L@VV@L),ng),'baseline_specialist_mean_rate':rates[0],'baseline_no_specialist_mean_rate':rates[1],'approximation':'Unconditional historical rates with GLS weighting; not a heterogeneous propensity test','source':str(pp)})
    else:meanbench.append({'mapping':mp,'cluster':cl,'status':'nonpositive_GLS_covariance_preserved','source':str(pp)})
 D=export('all_LPM_coefficients',allrows);export('LPM_model_status',models);export('temporal_formal_differences',changes);export('heterogeneous_proportional_benchmark',bench);export('proportional_accounted_fraction',ratios);export('unconditional_rate_approximation',meanbench)
 if len(D):
  q=D[D.term.isin(['Distributed_Home','Distributed','Distributed_x_CurrentPeriod'])&D.stage.isin(['replication','gradient','trajectory_marker'])].copy();q['event_count']=q.nonzero_cells;export('temporal_comparison_all_covariances',q.to_dict('records'))
 collect_home();collect_ppml();save(F/'logs/collection_status.json',{'replications_complete':sum((F/'results'/mp/'replication/COMPLETE').exists()for mp in ['parent','direct']),'temporal_modules_complete':sum((F/'results'/mp/st/'COMPLETE').exists()for mp in ['parent','direct']for st in ['gradient','trajectory_marker']),'proportional_modules_complete':sum((F/'results'/mp/'proportional/COMPLETE').exists()for mp in ['parent','direct']),'home_general_modules_complete':sum((F/'results'/mp/'home_proportional/COMPLETE').exists()for mp in ['parent','direct']),'PPML_attempts_complete':sum((ppml_root(mp,st)/'COMPLETE').exists()for mp in ['parent','direct']for st in ['pair','historical_offset','shared_coremonth'])});log('COLLECTION',len(allrows),'LPM rows',len(bench),'benchmark rows')

def collect_home():
 rows=[];rat=[]
 for mp in ['parent','direct']:
  root=F/'results'/mp/'home_proportional';p=root/'restricted_benchmark.json'
  if not p.exists():continue
  r=load(p);pt=np.array(r['primitive_points']);pi=r['projection_loading'];lin=linear_statistics(pi);q=np.array(load(root/'basis__signed.json')['sufficient_statistics']['Q']);ll=np.linalg.solve(q[:10,:10],q[:10,10:])[5]
  for name,k in [('specialist',0),('no_specialist',1)]:lin[name+'_proportional']=np.array([0,0,ll[k],0,0.]);lin[name+'_residual']=lin[name]-lin[name+'_proportional']
  lin['total_proportional']=lin['specialist_proportional']+lin['no_specialist_proportional'];lin['total_residual']=lin['total']-lin['total_proportional']
  for cl,v in r['covariance'].items():
   v=np.array(v);ng=r['home_clusters']if cl=='homefield'else r['cores']if cl=='team'else r['capability_paths']
   for name,L in lin.items():rows.append({'mapping':mp,'statistic':name,'cluster':cl,**uncertainty(float(L@pt),float(L@v@L),ng),**covariance_diagnostic(v),'N_per_outcome':r['N_per_outcome'],'projection_loading':pi,'counterfactor':'HomeGain only; no direct U moderation','first_stage_uncertainty_included':False,'new_exploratory_improvement_not_original_hypothesis_replication':True,'source':str(p)})
   a=lin['signed_proportional'];b=lin['signed_observed'];A=np.vstack([a,b]);rat.append({'mapping':mp,'cluster':cl,**fieller(float(a@pt),float(b@pt),A@v@A.T,ng),**covariance_diagnostic(v),'fraction_not_truncated':True,'source':str(p)})
 export('home_general_productivity_benchmark',rows);export('home_general_productivity_accounted_fraction',rat)

def collect_ppml():
 rows=[];contr=[];status=[];attempts=[]
 for namespace in ['relative_rate','relative_rate_accelerated','relative_rate_certified','relative_rate_certified_rankreuse']:
  for mp in ['parent','direct']:
   for regime in ['pair','historical_offset','shared_coremonth']:
    rt=F/namespace/mp/regime
    if not(rt/'execution.json').exists():continue
    ex=load(rt/'execution.json');sp=load(rt/'support.json');st=ex.get('statuses',{});cap=st.get('cap_status.csv',{});budget=st.get('computational_budget',{})
    attempts.append({'mapping':mp,'regime':regime,'namespace':namespace,'elapsed_seconds':ex.get('seconds'),'process_returncode':ex.get('process_returncode'),'cap_returncode':cap.get('return_code',budget.get('return_code',ex.get('process_returncode'))),'cap_converged':cap.get('converged'),'computationally_unresolved':ex.get('computationally_unresolved',False),'exported_coefficient_sets':len(list(rt.glob('*_coefficients.csv'))),'full_candidate_months_per_outcome':sp['type_support'][0]['full_candidate_months'],'all_input_first_events_by_type':json.dumps({x['outcome']:x['full_first_events']for x in sp['type_support']}),'profiled_input_rows':sp['input_rows'],'no_PSD_repair':True,'complete_marker':(rt/'COMPLETE').exists(),'source':str(rt/'execution.json'),'all_statuses':json.dumps(st)})
 scheme={'capability':'cap','homefield':'home','team':'core'}
 for mp in ['parent','direct']:
  for regime in ['pair','historical_offset','shared_coremonth']:
   root=ppml_root(mp,regime)
   if not(root/'COMPLETE').exists():continue
   ex=load(root/'execution.json');supp=load(root/'support.json')
   for k,v in ex['statuses'].items():status.append({'mapping':mp,'regime':regime,'covariance_fit':k,**v,'source':str(root/'execution.json')})
   if not ppml_cap_converged(root):continue
   cp=root/'cap_coefficients.csv'
   if not cp.exists():continue
   cs=pd.read_csv(cp);names=cs.term.tolist();b=cs.b.to_numpy();om=cs.omitted.to_numpy();mats={};md={}
   allowed=load(root/'certification.json').get('verified_covariance_keys',ppml_covariance_keys(root))if(root/'certification.json').exists()else ppml_covariance_keys(root)
   for key in allowed:
    pp=root/(key+'_covariance.csv');ms=root/(key+'_status.csv')
    if pp.exists():mats[key]=pd.read_csv(pp,header=None).apply(pd.to_numeric,errors='coerce').to_numpy()
    if ms.exists():tab=pd.read_csv(ms);md[key]={r.key:pd.to_numeric(r.value,errors='coerce')for r in tab.itertuples()}
   cov={cl:mats[key]for cl,key in scheme.items()if key in mats}
   if all(k in mats for k in ['cap','core','corepath']):cov['capability_team']=mats['cap']+mats['core']-mats['corepath']
   if all(k in mats for k in ['cap','home','homepath']):cov['homefield_capability']=mats['cap']+mats['home']-mats['homepath']
   for cl,V in cov.items():
    key='home'if cl=='homefield'else'core'if cl=='team'else'cap';ng=int(md[key]['N_clust']);min_eig=float(np.linalg.eigvalsh(V).min())if np.isfinite(V).all()else None
    for k,nm in enumerate(names):
     if not(nm.startswith(('sx','nx','tx'))and nm[2:].isdigit()):continue
     typ={'s':1,'n':2,'t':3}[nm[0]];nonidentified=bool(om[k]);row={'mapping':mp,'regime':regime,'outcome':['specialist','no_specialist','total'][typ-1],'term':NAMES[int(nm[2:])],'cluster':cl,**uncertainty(b[k],float(V[k,k])if not nonidentified else np.nan,ng),'omitted_or_not_identified':nonidentified,**covariance_diagnostic(V),'covariance_minimum_eigenvalue':min_eig,'PSD_repair_applied':False,'full_candidate_months':supp['type_support'][typ-1]['full_candidate_months'],'full_events':supp['type_support'][typ-1]['full_first_events'],'likelihood_support_N':md[key].get('N_type'+str(typ)),'likelihood_support_events':md[key].get('events_type'+str(typ)),'total_stacked_N':md[key]['N'],'PPML_pseudo_R2':md[key].get('r2_p'),'source':str(cp)};rows.append(row)
    for k,nm in enumerate(NAMES):
     s='sx'+str(k);n='nx'+str(k)
     if s not in names or n not in names:continue
     L=np.zeros(len(b));L[names.index(s)]=1;L[names.index(n)]=-1;valid=not(om[names.index(s)]or om[names.index(n)]);contr.append({'mapping':mp,'regime':regime,'contrast':'specialist_minus_no_relative_rate_coefficient','full_candidate_months_per_outcome':supp['type_support'][0]['full_candidate_months'],'specialist_likelihood_N':md[key].get('N_type1'),'no_specialist_likelihood_N':md[key].get('N_type2'),'specialist_events':md[key].get('events_type1'),'no_specialist_events':md[key].get('events_type2'),'total_stacked_N':md[key]['N'],'term':nm,'cluster':cl,**uncertainty(float(L@b),float(L@V@L)if valid else np.nan,ng),'omitted_or_not_identified':not valid,**covariance_diagnostic(V),'covariance_aware_stacked_equations':True,'PSD_repair_applied':False,'source':str(root)})
 export('PPML_coefficients',rows);export('PPML_formal_relative_rate_contrasts',contr);export('PPML_status_and_convergence',status);export('PPML_all_numerical_attempts',attempts)
 economics=[]
 for r in rows+contr:
  if r['term']!='Distributed_Home':continue
  economics.append({**r,'illustrative_gain_change':.1,'distributed_coverage_change_in_original_SD':1,'rate_ratio_change':float(np.expm1(.1*r['b'])),'rate_ratio_CI_low':float(np.expm1(.1*r['lo']))if r['lo']is not None else None,'rate_ratio_CI_high':float(np.expm1(.1*r['hi']))if r['hi']is not None else None,'not_a_causal_AI_use_effect':True})
 export('PPML_illustrative_economic_contrasts',economics)

if __name__=='__main__':main()
