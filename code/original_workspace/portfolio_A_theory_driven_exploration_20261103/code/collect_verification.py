from common import *
from collect import rows,export
import numpy as np,pandas as pd
from scipy.stats import t as td
from group_contrasts import compare
def main():
 pathrows=[];status=[];valid=[];counts=[];contr=[];checks=[];augrows=[];augdiff=[];duplicates=[];unique_paths=0
 for mp in ['parent','direct']:
  root=F/'verification/recent_dyad'/mp
  for family in ['risk','input']:
   pathnames=load(F/'results/dyad/recent'/mp/'foundation/sample.json')['paths']
   for path in pathnames:
    original=root/'paths'/family/path;accelerated=root/'accelerated_paths'/family/path
    available=[x for x in [original,accelerated]if(x/'COMPLETE').exists()]
    if not available:continue
    o=available[0];unique_paths+=1
    if len(available)==2:
     for outcome in ['total','specialist','no_specialist','signed']:
      aa,bb=[load(x/(outcome+'.json'))for x in available];assert aa['status']==bb['status']
      if aa['status']in['estimated','zero_outcome']:
       assert np.allclose(aa['sufficient_statistics']['beta'],bb['sufficient_statistics']['beta'],atol=1e-11,rtol=1e-7);assert np.allclose(aa['covariance']['capability'],bb['covariance']['capability'],atol=1e-12,rtol=1e-6)
     duplicates.append({'mapping':mp,'family':family,'path':path,'same_specification_agreement_verified':True,'selection':'Original namespace first if complete, otherwise accelerated; never coefficient/p selection'})
    for outcome in ['total','specialist','no_specialist','signed']:
     pp=o/(outcome+'.json');r=load(pp);meta={'mapping':mp,'family':family,'omitted_path':o.name,'outcome':outcome,'label':'targeted post-discovery exact path sensitivity','population':'dyad','cohort':'recent'};status.append({**meta,'status':r['status'],'N':r['meta']['N'],'rank':r['meta']['rank'],'source':str(pp)});pathrows+=rows(pp,meta,1e4)
    rs={k:load(o/(k+'.json'))for k in ['total','specialist','no_specialist','signed']}
    if all(r['status']in['estimated','zero_outcome']for r in rs.values()):
     for key in ['beta','scores_by_path']:
      s={k:np.array(r['sufficient_statistics'][key])for k,r in rs.items()};assert np.allclose(s['total'],s['specialist']+s['no_specialist'],rtol=1e-7,atol=1e-10);assert np.allclose(s['signed'],s['specialist']-s['no_specialist'],rtol=1e-7,atol=1e-10)
     main=load(F/'results/dyad/recent'/mp/'architecture/architecture_estimable__signed.json')
     if family=='input':assert rs['signed']['meta']['N']==main['meta']['N']and rs['signed']['meta']['nonzero_cells']==main['meta']['nonzero_cells']
  o=root/'validity'
  if not(o/'COMPLETE').exists():continue
  md=load(o/'joint_metadata.json');j=np.load(o/'joint_covariances.npz');names=md['regressors'];yn=md['outcomes'];K=len(names);B=j['beta'];ix=lambda x:yn.index(x);refroot=F/'results/dyad/recent'/mp/'architecture';refno=load(refroot/'architecture_estimable__no_specialist.json');refcur=load(refroot/'architecture_estimable__signed.json')
  identities=[({'exact_no':1,'expanded_no':1},refno,'exact/expanded sum to primary no'),({'no_current_core_expert':1,'with_current_core_expert':1},refno,'current-core partitions sum to primary no'),({'complete_no':1,'uncertain_no':1},refno,'history partitions sum to primary no'),({'specialist':1,'exact_no':-1,'expanded_no':-1},refcur,'specialist/no partitions reproduce primary signed')]
  for weights,ref,label in identities:
   L=np.zeros((K,len(yn)*K));bb=np.zeros(K)
   for nm,w in weights.items():L[:,ix(nm)*K:(ix(nm)+1)*K]=w*np.eye(K);bb+=w*B[ix(nm)]
   assert np.allclose(bb,ref['sufficient_statistics']['beta'],atol=1e-11,rtol=1e-7)
   for cl in ref['covariance']:assert np.allclose(L@j[cl]@L.T,ref['covariance'][cl],rtol=1e-6,atol=1e-12)
   checks.append({'mapping':mp,'identity':label,'all_five_joint_covariance_identities_passed':True})
  for nm in yn:
   pp=o/(nm+'.json');r=load(pp);valid+=rows(pp,{'mapping':mp,'outcome':nm,'population':'dyad','cohort':'recent','label':'targeted post-discovery outcome/construct validation'},1e4);counts.append({'mapping':mp,'outcome':nm,'nonzero_events':r['meta']['nonzero_cells'],'outcome_sum':float(j['outcome_sum'][ix(nm)]),'N':r['meta']['N'],'source':str(pp)})
  for label,wts in [('complete_minus_uncertain',{'complete_no':1,'uncertain_no':-1}),('without_minus_with_current_core_expert',{'no_current_core_expert':1,'with_current_core_expert':-1})]:
   for term in ['Distributed_Home','Distributed_Destination']:
    k=names.index(term);L=np.zeros(len(yn)*K)
    for nm,w in wts.items():L[ix(nm)*K+k]=w
    b=float(L@B.reshape(-1))
    for cl in ['capability','homefield','team','capability_team','homefield_capability']:
     ng=md['home_clusters']if cl=='homefield'else md['teams']if cl=='team'else md['paths'];var=float(L@j[cl]@L);ok=var>0 and np.isfinite(var);se=np.sqrt(var)if ok else np.nan;crit=td.ppf(.975,ng-1);contr.append({'mapping':mp,'outcome_contrast':label,'term':term,'cluster':cl,'b':b,'se':se,'lo':b-crit*se,'hi':b+crit*se,'p':float(2*td.sf(abs(b/se),ng-1))if ok else np.nan,'variance_valid':ok,'clusters':ng,'magnitude_multiplier':1e4,'covariance_aware':True,'source':str(o/'joint_covariances.npz')})
  ao=root/'augmented_controls'
  if(ao/'COMPLETE').exists():
   for outcome in ['total','specialist','no_specialist','signed']:
    pp=ao/(outcome+'.json');r=load(pp);augrows+=rows(pp,{'mapping':mp,'outcome':outcome,'population':'dyad','cohort':'recent','label':'post-discovery applicability/access control sensitivity'},1e4)
    basefile=refroot/('architecture_estimable__'+outcome+'.json');br=load(basefile);assert r['meta']['N']==br['meta']['N']and r['meta'].get('nonzero_cells')==br['meta']['nonzero_cells']
   rr=load(ao/'signed.json');base=load(refroot/'architecture_estimable__signed.json')
   if rr['status']=='estimated':
    for term in ['Distributed_Home','Distributed_Destination']:augdiff.append(compare((refroot/'architecture_estimable__signed.json',base,load(refroot/'scaling.json')),(ao/'signed.json',rr,load(ao/'scaling.json')),term,{'mapping':mp,'population':'dyad','cohort':'recent','comparison':'augmented minus primary, signed gradient in raw coverage units'}))
 D=export('lead_exact_path_coefficients',pathrows);export('lead_exact_path_status',status);export('lead_validity_all_covariances',valid);export('lead_validity_counts',counts);export('lead_validity_formal_contrasts',contr);export('lead_augmented_controls',augrows);export('lead_augmented_minus_primary',augdiff);summary=[]
 for(mp,fam,out,term),q in D[D.term.str.startswith('Distributed_')].groupby(['mapping','family','outcome','term']):
  ref=load(F/'results/dyad/recent'/mp/'architecture'/('architecture_estimable__'+out+'.json'));base=next(x['b']for x in ref['coefficients']if x['term']==term and x['cluster']=='capability');summary.append({'mapping':mp,'family':fam,'outcome':out,'term':term,'folds':len(q),'full_b':base,'b_min':q.b.min(),'b_max':q.b.max(),'same_sign_folds':int((np.sign(q.b)==np.sign(base)).sum()),'p_min':q.p.min(),'p_max':q.p.max(),'not_exogeneity_test':True,'source':'data/lead_exact_path_coefficients.csv'})
 export('lead_exact_path_summary',summary)
 npaths=unique_paths;nval=len(list((F/'verification').glob('*/*/validity/COMPLETE')));naug=len(list((F/'verification').glob('*/*/augmented_controls/COMPLETE')));assert npaths==40 and nval==2 and naug==2,(npaths,nval,naug)
 save(F/'qa/lead_verification.json',{'path_refits':npaths,'validity_refits':nval,'augmented_control_refits':naug,'expected_total':44,'complete':True,'validity_identity_checks':checks,'duplicate_specifications_checked':duplicates,'path_rank_or_zero_outcomes_retained':[x for x in status if x['status']!='estimated'],'new_measurement_does_not_restore_original_H3_by_relabeling':True});log('LEAD VERIFICATION COMPLETE',npaths,nval,naug)
if __name__=='__main__':main()
