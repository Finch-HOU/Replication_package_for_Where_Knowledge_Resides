from common import *
from restriction_tools import linear_statistics

def main():
 checks=[]
 for mp in ['parent','direct']:
  assert(F/'results'/mp/'replication/COMPLETE').exists()
  for stage,prefix in [('replication',''),('gradient','levels__'),('gradient','pooled_change__'),('trajectory_marker',''),('proportional','basis__'),('home_proportional','basis__')]:
   o=F/'results'/mp/stage;md=load(o/(prefix+'joint_metadata.json'));d=np.load(o/(prefix+'joint_covariances.npz'));yn=md['outcomes'];K=len(md['regressors']);groups=[['pre_'+x for x in OUTCOMES],['current_'+x for x in OUTCOMES]]if len(yn)==8 else[OUTCOMES]
   for group in groups:
    ix={out:yn.index(nm)for out,nm in zip(OUTCOMES,group)};b=d['beta'];assert np.allclose(b[ix['total']],b[ix['specialist']]+b[ix['no_specialist']],rtol=1e-6,atol=1e-11);assert np.allclose(b[ix['signed']],b[ix['specialist']]-b[ix['no_specialist']],rtol=1e-6,atol=1e-11)
    A=np.zeros((K,len(yn)*K));A[:,ix['specialist']*K:(ix['specialist']+1)*K]=np.eye(K);A[:,ix['no_specialist']*K:(ix['no_specialist']+1)*K]=-np.eye(K);i=ix['signed']
    for cl in SCHEMES:assert np.allclose(A@d[cl]@A.T,d[cl][i*K:(i+1)*K,i*K:(i+1)*K],rtol=2e-5,atol=1e-12),(mp,stage,cl)
   checks.append({'mapping':mp,'stage':stage,'prefix':prefix,'coefficient_and_five_covariance_component_identities':True,'N':md['N']})
  for stage in ['proportional','home_proportional']:
   b=load(F/'results'/mp/stage/'restricted_benchmark.json');assert all(x['five_covariances_reproduced']for x in b['coefficient_and_covariance_checks'])
   pt=np.array(b['primitive_points']);pi=b['projection_loading'];lin=linear_statistics(pi)
   assert abs(lin['signed_residual']@pt+lin['signed_proportional']@pt-lin['signed_observed']@pt)<1e-14
   assert abs(lin['total']@pt-lin['specialist']@pt-lin['no_specialist']@pt)<1e-14
   assert np.max(np.abs(b['score_sum']))<1e-7
   checks.append({'mapping':mp,'stage':stage,'restricted_common_response_and_original_covariances_verified':True,'N':b['N_per_outcome']})
  p0=load(F/'baseline_propensities_crossfit'/mp/'diagnostics.json');assert p0['exact_heldout_core_labels_excluded']and not p0['historical_own_core_first_entry_count_used_in_own_prediction'];assert all(fit['numerical_convergence']for ff in p0['hyperparameters_and_convergence']for fit in ff['frozen_core_activity']);assert p0['maximum_p0_total']*11<1
 for r in load(F/'diagnostics/original_source_hashes.json'):assert hashlib.file_digest(Path(r['path']).open('rb'),'sha256').hexdigest()==r['sha256'],r['path']
 ppml=[]
 for mp in ['parent','direct']:
  for regime in ['pair','historical_offset','shared_coremonth']:
   pending=F/'relative_rate_certified_rankreuse'/mp/regime
   if pending.exists()and not(pending/'COMPLETE').exists():ppml.append({'mapping':mp,'regime':regime,'status':'pending','remaining_task':'same_likelihood_covariance_refits'});continue
   o=ppml_root(mp,regime)
   if not(o/'COMPLETE').exists():ppml.append({'mapping':mp,'regime':regime,'status':'pending'});continue
   ex=load(o/'execution.json');keys=load(o/'certification.json').get('verified_covariance_keys',ppml_covariance_keys(o))if(o/'certification.json').exists()else ppml_covariance_keys(o);files=[o/(k+'_coefficients.csv')for k in keys];present=[p for p in files if p.exists()]
   if present and ppml_cap_converged(o):
    a=pd.read_csv(present[0]);errs=[]
    for p in present[1:]:
     b=pd.read_csv(p);assert a.term.tolist()==b.term.tolist();err=float(np.max(np.abs(a.b-b.b)));errs.append(err)
     # Optimization is re-run for covariance variants with the same likelihood.
     assert np.allclose(a.b,b.b,rtol=2e-4,atol=2e-4),(mp,regime,err)
    ppml.append({'mapping':mp,'regime':regime,'status':'estimated'if len(present)==5 else'estimated_with_partial_covariance_availability','valid_one_way_covariance_fits':len(present),'all_five_covariance_fits_available':len(present)==5,'maximum_covariance_refit_beta_error':max(errs)if errs else 0,'covariance_repair_not_applied':True})
   else:ppml.append({'mapping':mp,'regime':regime,'status':'failed_or_incomplete_estimator_preserved','partial_covariance_fits':len(present),'execution':ex})
 budget=load(F/'diagnostics/PPML_running_budget.json');all_closed=all(x['finished']for x in budget)
 save(F/'diagnostics/numerical_validation.json',{'LPM_identity_checks':checks,'original_source_hashes_unchanged':True,'original_primary_coefficients_and_all_five_covariances_reproduced':True,'historical_predictions_crossfitted_without_own_labels':True,'crossfitted_probabilities_allow_at_most_one_total_event_over_11_months':True,'relative_rate_status':ppml,'all_PPML_attempts_finished':all_closed and not any(x['status']=='pending'for x in ppml),'invalid_covariances_not_repaired':True});log('OUTPUT VALIDATION COMPLETE',len(checks),ppml)

if __name__=='__main__':main()
