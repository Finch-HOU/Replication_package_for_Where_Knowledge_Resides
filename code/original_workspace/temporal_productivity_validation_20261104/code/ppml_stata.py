"""Established PPMLHDFE estimator; stacked types and unmodified covariance signs.

Only analytically zero-likelihood FE groups are pre-profiled for data size. Every
unselected destination of an informative core-month is retained. Pair-FE models
start with all eleven months of every positive-margin pair. The full prospective
risk denominator and every profile/drop count are reported separately.
"""
from common import *
import subprocess,argparse,gc
from ppml_runtime import run_official,read_statuses

def raw_x(p,z,ii,jj,tt):
 return np.column_stack([z[k][ii,jj]*(p.A[jj,tt]if s=='Destination'else p.ha[ii,tt])for k in TERMS for s in ['Destination','Home']]+[z[k][ii,jj]*p.A[jj,tt]for k in EXTRAS])

def prepare(mp,regime):
 o=F/'relative_rate'/mp/regime;o.mkdir(parents=True,exist_ok=True)
 if(o/'DATA_COMPLETE').exists():return o
 p,z,sc=setup(mp);e,v=events(p,786,796);pred=np.load(F/'baseline_propensities_crossfit'/mp/'p0.npy',mmap_mode='r')if regime!='pair'else None;frames=[];support=[]
 for typ,k in ([(1,1),(2,2)]if regime=='shared_coremonth'else[(1,1),(2,2),(3,0)]):
  ev=e[v[:,k]>0].copy();pieces=[]
  if regime=='pair':
   ii=np.repeat(ev.i.to_numpy(),11);jj=np.repeat(ev.j0.to_numpy(),11);tt=np.tile(np.arange(11),len(ev));ys=(tt==np.repeat(ev.mo.to_numpy()-786,11)).astype(float);pieces=[(ii,jj,tt,ys)]
  else:
   keys=(ev.i.to_numpy()*p.J+ev.j0.to_numpy())*11+ev.mo.to_numpy()-786;keys=np.sort(keys)
   for t in range(11):
    source=e if regime=='shared_coremonth'else ev;cores=np.unique(source.loc[source.mo.eq(t+786),'i'].to_numpy());ai,jj=np.where(p.mask[cores]);ii=cores[ai];tt=np.full(len(ii),t,int);kk=(ii*p.J+jj)*11+t;ix=np.searchsorted(keys,kk);ys=((ix<len(keys))&(keys[np.minimum(ix,len(keys)-1)]==kk)).astype(float)if len(keys)else np.zeros(len(ii));pieces.append((ii,jj,tt,ys))
  n=0;nnz=0
  for ii,jj,tt,ys in pieces:
   x=raw_x(p,z,ii,jj,tt);q={'core':ii.astype(np.int32),'dest':jj.astype(np.int16),'t':tt.astype(np.int8),'type':np.full(len(ii),typ,np.int8),'cap':p.fc[jj].astype(np.int16),'home':p.home[ii].astype(np.int32),'pair':(ii*p.J+jj).astype(np.int32),'y':ys}
   for k0 in range(10):q['x'+str(k0)]=x[:,k0]
   for nm in TERMS:q[nm]=z[nm][ii,jj]
   if regime!='pair':
    prob=pred[ii,jj,typ-1]if typ<3 else pred[ii,jj].sum(1);assert np.all(prob>0);q['ln_p0']=np.log(prob)
   frames.append(pd.DataFrame(q));n+=len(ii);nnz+=int(ys.sum());del x,q
  assert nnz==len(ev);support.append({'type':typ,'outcome':['specialist','no_specialist','total'][typ-1],'full_candidate_months':int(p.mask.sum())*11,'full_first_events':len(ev),'analytically_profiled_input_rows':n,'preprofile_rule':'All-zero pair margins'if regime=='pair'else'All-zero total-core-month margins shared by both modes'if regime=='shared_coremonth'else'All-zero type-core-month margins','unselected_destinations_retained_on_likelihood_support':True});log('PPML INPUT TYPE',mp,regime,typ,n,nnz)
 d=pd.concat(frames,ignore_index=True);d.to_stata(o/'support.dta',write_index=False,version=118);save(o/'support.json',{'sample':sample(p,sc),'regime':regime,'type_support':support,'input_rows':len(d),'full_risk_is_not_input_support_N':True,'official_estimator_will_further_check_separation_and_singletons':True,'pair_FE_removed_in_offset_alternative':regime!='pair','historical_offset_source':str(F/'baseline_propensities_crossfit'/mp/'diagnostics.json')if regime!='pair'else None});(o/'DATA_COMPLETE').write_text('Exact zero-FE profiling, not a selected-entry primary sample.');p.c.close();del d,frames;gc.collect();return o

def make_do(o,regime):
 dep=(F/'dependencies/ppmlhdfe').as_posix();out=o.as_posix();types=['s','n']if regime=='shared_coremonth'else['s','n','t'];xs=' '.join(pre+'x'+str(k)for pre in types for k in range(10));absorb=('pairmode 'if regime=='pair'else'')+'cmm dmm mm##c.Gap mm##c.Distance mm##c.Distributed mm##c.Shared';offset='offset(ln_p0)'if regime!='pair'else'';cmm='core*11+t'if regime=='shared_coremonth'else'core*33 + (type-1)*11 + t'
 s=f'''clear all
set more off
set processors 16
capture log close _all
log using "{out}/estimation_detail.log", text replace
adopath ++ "{dep}"
use "{out}/support.dta", clear
if _N<500000 set processors 4
generate long cmm = {cmm}
generate int dmm = dest*33 + (type-1)*11 + t
generate byte mm = (type-1)*11+t
generate long pairmode=pair*3+type
egen long corepath=group(core cap)
egen int homepath=group(home cap)
forvalues k=0/9 {{
 generate double sx`k'=x`k'*(type==1)
 generate double nx`k'=x`k'*(type==2)
 generate double tx`k'=x`k'*(type==3)
}}
mata:
void dump_estimate(string scalar prefix) {{
 real matrix b,V,omit
 string matrix stripe
 real scalar f,i
 b=st_matrix("e(b)");V=st_matrix("e(V)");omit=st_matrix("omitted");stripe=st_matrixcolstripe("e(b)")
 f=fopen(prefix+"_coefficients.csv","w");fput(f,"term,b,omitted")
 for(i=1;i<=cols(b);i++) fput(f,stripe[i,2]+","+strofreal(b[1,i],"%21.16g")+","+strofreal(omit[1,i],"%9.0g"))
 fclose(f);f=fopen(prefix+"_covariance.csv","w")
 for(i=1;i<=rows(V);i++) fput(f,invtokens(strofreal(V[i,.],"%21.16g"),","))
 fclose(f)
}}
end
local warm=0
foreach scheme in cap home core corepath homepath {{
 capture drop __dfe
 local guessopt ""
 if `warm' local guessopt "guess(variable __mu0)"
 capture noisily ppmlhdfe y {xs}, absorb({absorb}) {offset} vce(cluster `scheme') d(__dfe) tolerance(1e-9) maxiterations(500) `guessopt'
 local fitrc=_rc
 file open f using "{out}/`scheme'_status.csv", write replace
 file write f "key,value" _n "return_code,`fitrc'" _n
 if `fitrc'==0 {{
  foreach stat in N N_full num_separated num_singletons df_a df_m df_r converged ic ll r2_p N_clust rank {{
   tempname vv
   scalar `vv'=.
   capture scalar `vv'=e(`stat')
   file write f "`stat'," %21.16g (`vv') _n
  }}
  forvalues oo=1/3 {{
   quietly summarize y if e(sample) & type==`oo', meanonly
   file write f "events_type`oo'," %21.0g (r(sum)) _n
   quietly count if e(sample) & type==`oo'
   file write f "N_type`oo'," %21.0g (r(N)) _n
  }}
  _ms_omit_info e(b)
  matrix omitted=r(omit)
  mata: dump_estimate("{out}/`scheme'")
  capture drop __mu0
  quietly predict double __mu0 if e(sample), mu
  quietly replace __mu0=.1 if missing(__mu0)
  local warm=1
 }}
 file close f
 if `fitrc'!=0 continue, break
}}
log close
exit, clear
''';(o/'estimate.do').write_text(s,encoding='ascii')

def main(mp,shared_only=False):
 assert(F/'baseline_propensities_crossfit'/mp/'COMPLETE').exists()
 for regime in (['shared_coremonth']if shared_only else['pair','historical_offset']):
  o=prepare(mp,regime)
  if(o/'COMPLETE').exists():continue
  make_do(o,regime);runtime=run_official(o,5400);statuses=read_statuses(o,runtime)
  save(o/'execution.json',{**runtime,'statuses':statuses,'covariance_plan':'Read five one-way covariances; form the two two-way matrices by inclusion-exclusion outside Stata, preserving negative variances. No PSD repair.','CR1_note':'Stata cluster-dependent nested-FE small-sample conventions, recorded per fit, differ from original complete-risk full-FE correction.','official_source_manifest':str(F/'dependencies/ppmlhdfe/source_manifest.json')});(o/'COMPLETE').write_text('Official PPMLHDFE attempt finished; success, separation, omitted regressors and failures preserved.');log('PPML ATTEMPT COMPLETE',mp,regime,statuses)

if __name__=='__main__':ap=argparse.ArgumentParser();ap.add_argument('mapping');ap.add_argument('--shared-only',action='store_true');a=ap.parse_args();main(a.mapping,a.shared_only)
