"""Same-likelihood covariance refits after an official full-separation certificate.

Never skips the first separation/existence check. Requires a converged official
fit reporting zero non-FE separated rows, reconstructs its exact profiled support,
then verifies original point estimates AND capability covariance after refitting.
"""
from common import *
from ppml_stata import make_do
from ppml_runtime import run_official,read_statuses
import argparse

def readkv(p):
 a=pd.read_csv(p);return {r.key:pd.to_numeric(r.value,errors='coerce')for r in a.itertuples()}

def main(mp,regime,smoke=False,reuse_rank=False):
 source=F/'diagnostics/ppml_smoke_test'if smoke else F/'relative_rate'/mp/regime
 out=F/'diagnostics'/('ppml_certified_rank_validation'if reuse_rank else'ppml_certified_validation')if smoke else F/('relative_rate_certified_rankreuse'if reuse_rank else'relative_rate_certified')/mp/regime
 out.mkdir(parents=True,exist_ok=True)
 if(out/'COMPLETE').exists():return
 st=readkv(source/'cap_status.csv');assert st['return_code']==0 and st['converged']==1 and st['num_separated']==0
 if reuse_rank:
  cc=pd.read_csv(source/'cap_coefficients.csv');assert not cc.omitted.any();assert int(st['rank'])==int(st['df_m'])
 # The first successful fit retains every positive event, so zero-FE profiling
 # is exactly the intersection of positive cmm/dmm/mm margins; no label is
 # used for main-sample selection. Likelihood-zero groups are profiled only.
 data=pd.read_stata(source/'support.dta');shared=regime=='shared_coremonth'
 data['cmm']=data.core*11+data.t if shared else data.core*33+(data['type']-1)*11+data.t
 data['dmm']=data.dest*33+(data['type']-1)*11+data.t;data['mm']=(data['type']-1)*11+data.t
 keep=np.ones(len(data),bool)
 for group in ['cmm','dmm','mm']:keep&=data.groupby(group).y.transform('sum').to_numpy()>0
 assert int(keep.sum())==int(st['N']),(keep.sum(),st['N'])
 for k in [1,2,3]:
  ix=keep&data['type'].eq(k).to_numpy();assert int(ix.sum())==int(st['N_type'+str(k)]);assert int(data.loc[ix,'y'].sum())==int(st['events_type'+str(k)])
 del data
 make_do(out,regime);dd=out/'estimate.do';s=dd.read_text();s=s.replace((out/'support.dta').as_posix(),(source/'support.dta').as_posix())
 injection='''
* Exact support certified by the preceding converged full-separation fit.
foreach gr in cmm dmm mm {
 bysort `gr': egen double __pos=total(y)
 keep if __pos>0
 drop __pos
}
assert _N=='''+str(int(st['N']))+'\n'
 s=s.replace('egen long corepath=',injection+'\negen long corepath=')
 s=s.replace('maxiterations(500)','maxiterations(500) separation(none) verbose(1)')
 if reuse_rank:s=s.replace('separation(none)','separation(none) remove_collinear_variables(0)')
 dd.write_text(s,encoding='ascii')
 runtime=run_official(out,2700,maximum_seconds=2700);ss=read_statuses(out,runtime)
 save(out/'execution.json',{**runtime,'statuses':ss,'same_likelihood_certified_by_original_full_separation':True,'full_rank_certificate_reused':reuse_rank,'original_fit_source':str(source),'not_an_alternative_empirical_specification':True})
 if not smoke:save(out/'support.json',load(source/'support.json'))
 valid=False;checks={}
 if(out/'cap_coefficients.csv').exists():
  a=pd.read_csv(source/'cap_coefficients.csv');b=pd.read_csv(out/'cap_coefficients.csv');assert a.term.tolist()==b.term.tolist()
  ix=a.term.str.match(r'^[snt]x[0-9]+$').to_numpy();be=float(np.max(np.abs(a.b.to_numpy()[ix]-b.b.to_numpy()[ix])))
  V0=pd.read_csv(source/'cap_covariance.csv',header=None).to_numpy();V1=pd.read_csv(out/'cap_covariance.csv',header=None).to_numpy();ve=float(np.max(np.abs(V0[np.ix_(ix,ix)]-V1[np.ix_(ix,ix)])))
  ns=readkv(out/'cap_status.csv');checks={'maximum_AI_beta_error':be,'maximum_AI_cap_covariance_error':ve,'likelihood_N_same':ns['N']==st['N'],'df_a_same':ns['df_a']==st['df_a'],'log_likelihood_error':float(abs(ns['ll']-st['ll']))}
  valid=bool(be<1e-5 and ve<1e-5 and checks['likelihood_N_same']and checks['df_a_same']and checks['log_likelihood_error']<1e-4)
 save(out/'certification.json',{'validated_exact_likelihood_and_cap_covariance':valid,'checks':checks,'source_fit_has_zero_non_FE_separation':True,'official_full_separation_check_was_not_skipped':True,'requires_five_converged_covariance_fits_for_complete_inference':True,'source':str(source)})
 (out/'COMPLETE').write_text('Certified same-likelihood attempt ended; validation result retained.')
 if smoke:assert valid,checks
 log('CERTIFIED PPML COMPLETE',mp,regime,valid,checks)

if __name__=='__main__':a=argparse.ArgumentParser();a.add_argument('mapping');a.add_argument('regime');a.add_argument('--smoke',action='store_true');a.add_argument('--reuse-rank',action='store_true');x=a.parse_args();main(x.mapping,x.regime,x.smoke,x.reuse_rank)
