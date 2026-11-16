"""Verify every completed covariance subset after bounded computation.

No additional fit or significance-based choice. A partial numerical attempt can
contribute completed covariances only after reproducing the original strict-fit
points, capability covariance, sample, likelihood and FE degrees of freedom.
"""
from common import *

def kv(p):
 a=pd.read_csv(p);return {r.key:pd.to_numeric(r.value,errors='coerce')for r in a.itertuples()}

def main():
 rows=[]
 for mp,reg in [('direct','historical_offset'),('parent','shared_coremonth'),('direct','shared_coremonth')]:
  o=F/'relative_rate_certified_rankreuse'/mp/reg;assert(o/'COMPLETE').exists()
  src=F/'relative_rate'/mp/reg
  if not ppml_cap_converged(o):rows.append({'mapping':mp,'regime':reg,'status':'No completed new capability fit; original valid covariance retained'});continue
  a=pd.read_csv(src/'cap_coefficients.csv');b=pd.read_csv(o/'cap_coefficients.csv');s=kv(src/'cap_status.csv');z=kv(o/'cap_status.csv');assert a.term.tolist()==b.term.tolist();ix=a.term.str.match(r'^[snt]x[0-9]+$').to_numpy()
  v0=pd.read_csv(src/'cap_covariance.csv',header=None).to_numpy();v1=pd.read_csv(o/'cap_covariance.csv',header=None).to_numpy();be=float(np.max(np.abs(a.b.to_numpy()[ix]-b.b.to_numpy()[ix])));ve=float(np.max(np.abs(v0[np.ix_(ix,ix)]-v1[np.ix_(ix,ix)])))
  valid=bool(be<1e-5 and ve<1e-5 and s['N']==z['N']and s['df_a']==z['df_a']and abs(s['ll']-z['ll'])<1e-4)
  keys=[];errors=[]
  for k in ppml_covariance_keys(o):
   try:c=pd.read_csv(o/(k+'_coefficients.csv'));ss=kv(o/(k+'_status.csv'));vv=pd.read_csv(o/(k+'_covariance.csv'),header=None).to_numpy();er=float(np.max(np.abs(c.b.to_numpy()[ix]-a.b.to_numpy()[ix])))
   except Exception as e:errors.append({'scheme':k,'complete_covariance_verified':False,'partial_file_diagnostic':str(e)});continue
   ok=bool(c.term.tolist()==a.term.tolist()and er<2e-4 and ss['N']==s['N']and vv.shape==v0.shape and np.isfinite(vv).all())
   errors.append({'scheme':k,'maximum_AI_beta_error':er,'complete_covariance_verified':ok})
   if ok:keys.append(k)
  save(o/'certification.json',{'validated_exact_likelihood_and_cap_covariance':valid,'verified_covariance_keys':keys if valid else[],'all_five_covariance_fits_available':valid and len(keys)==5,'accepted_only_completed_verified_covariance_subsets':True,'maximum_AI_beta_error':be,'maximum_AI_cap_covariance_error':ve,'likelihood_N_and_FE_df_checked':True,'per_covariance_checks':errors,'original_full_separation_and_rank_source':str(src),'no_model_or_significance_selection':True})
  rows.append({'mapping':mp,'regime':reg,'same_likelihood_and_cap_covariance_verified':valid,'verified_covariances':','.join(keys)if valid else'','maximum_AI_beta_error':be,'maximum_AI_cap_covariance_error':ve})
 pd.DataFrame(rows).to_csv(F/'diagnostics/PPML_partial_covariance_verification.csv',index=False);log('COMPLETED COVARIANCE SUBSETS VERIFIED',rows)

if __name__=='__main__':main()
