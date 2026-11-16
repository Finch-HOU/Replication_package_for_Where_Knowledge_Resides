"""Same PPML specifications with an invertible nuisance-space preconditioner.

Original attempts remain separate. Completion namespace, never coefficient/p,
selects which completed fit is collected. Added intercepts/rotations change no
fitted model column space. Official PPMLHDFE remains the estimator.
"""
from common import *
from ppml_stata import make_do
import subprocess,argparse
from ppml_runtime import run_official,read_statuses

PRECONDITION='''
* Remove existing FE components, then whiten nuisance slopes within mode-month.
* All rotations and rescalings are invertible; no covariate direction is dropped.
local qn=0
foreach zz in Gap Distance Distributed Shared {
 local qn=`qn'+1
 quietly reghdfe `zz', absorb(cmm dmm) keepsingletons residuals(rz`qn') tolerance(1e-10)
}
sort mm
generate double q1=.
generate double q2=.
generate double q3=.
generate double q4=.
mata:
R=st_data(.,("rz1","rz2","rz3","rz4"));Q=R;info=panelsetup(st_data(.,"mm"),1);worst=0;V=J(4,4,0);L=J(1,4,0)
for(g=1;g<=rows(info);g++) {
 aa=info[g,1];bb=info[g,2];rr=R[|aa,1\\bb,4|];C=quadcross(rr,rr)
 symeigensystem(C,V,L)
 den=sqrt(abs(L):/rows(rr))
 for(k=1;k<=cols(den);k++) if(den[k]<1e-10) den[k]=1
 qq=(rr*V'):/den
 back=(qq:*den)*V
 worst=max((worst,max(abs(back-rr))))
 Q[|aa,1\\bb,4|]=qq
}
st_store(.,("q1","q2","q3","q4"),Q)
st_numscalar("precond_error",worst)
end
display "INVERTIBLE_PRECONDITION_MAX_RECONSTRUCTION_ERROR " precond_error
assert precond_error<1e-8
'''

def prepare_do(o,source,regime,one_cov=False):
 o.mkdir(parents=True,exist_ok=True);make_do(o,regime);p=o/'estimate.do';s=p.read_text();s=s.replace((o/'support.dta').as_posix(),(source/'support.dta').as_posix()).replace('mata:\nvoid dump_estimate',PRECONDITION+'\nmata:\nvoid dump_estimate').replace('mm##c.Gap mm##c.Distance mm##c.Distributed mm##c.Shared','mm##c.q1 mm##c.q2 mm##c.q3 mm##c.q4')
 if one_cov:s=s.replace('foreach scheme in cap home core corepath homepath','foreach scheme in cap')
 p.write_text(s,encoding='ascii')

def main(mp,smoke=False):
 regimes=['smoke']if smoke else['historical_offset','shared_coremonth']
 for regime in regimes:
  source=F/'diagnostics/ppml_smoke_test'if smoke else F/'relative_rate'/mp/regime;o=F/'diagnostics/ppml_precondition_validation'if smoke else F/'relative_rate_accelerated'/mp/regime
  if(o/'COMPLETE').exists():continue
  prepare_do(o,source,'historical_offset'if smoke else regime,smoke);runtime=run_official(o,2700);statuses=read_statuses(o,runtime)
  if smoke:
   a=pd.read_csv(source/'cap_coefficients.csv');b=pd.read_csv(o/'cap_coefficients.csv');err=float(np.max(np.abs(a.b-b.b)));assert err<1e-5,err;save(o/'validation.json',{'same_specification_max_beta_error':err,'invertible_nuisance_rotation':True,'covariate_directions_dropped':False,'official_estimator':True})
  else:save(o/'support.json',load(source/'support.json'))
  save(o/'execution.json',{**runtime,'statuses':statuses,'same_specification_as_original_namespace':True,'nuisance_preconditioned_by_invertible_within_mode_month_rotation':True,'source_support':str(source/'support.dta'),'covariance_plan':'Five one-way fits plus unmodified inclusion-exclusion; no PSD repair','CR1_note':'Stata nested-FE finite-sample adjustments, not original Python full-FE CR1'});(o/'COMPLETE').write_text('Same-model numerically preconditioned official PPML attempt completed.');log('PRECONDITIONED PPML COMPLETE',mp,regime,statuses)

if __name__=='__main__':ap=argparse.ArgumentParser();ap.add_argument('mapping');ap.add_argument('--smoke',action='store_true');a=ap.parse_args();main(a.mapping,a.smoke)
