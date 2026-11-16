from common import *
import numpy as np,pandas as pd,gc
from panel_api import Panel

def constructs(p,needed=None):
 m=F/'measurement';raw={'Gap':p.G,'Distance':p.D,'C':p.cy};h=p.hist;R=np.load(m/'R.npy');sub=p.s;old=pd.read_csv(H/'08_reproducibility/inputs/pre2014_domain_counts.csv').set_index('subfield_id');cnt=old.classified_citations.reindex(sub.subfield_id).to_numpy();base=(cnt[:,None]*R).sum(0)/cnt.sum();raw['Applicability']=np.log1p((h/h.sum(1,keepdims=True).clip(1))@(R/base.clip(1e-12)).T)[:,p.js]
 def read(name):return np.load(m/(name+'.npy'),mmap_mode='r')[p.g][:,p.js].astype(float)
 if needed is None or 'Access'in needed:raw['Access']=np.log1p(np.load(p.u/'network/network.npy',mmap_mode='r')[p.g][:,p.js])
 if needed is None or 'InternalShare'in needed:raw['InternalShare']=read('carriers')/p.mi.core_size.to_numpy()[:,None]
 if needed is None or 'Depth'in needed:raw['Depth']=np.log1p(h[:,p.js])
 for k,nm in [('Distributed','distributed_input'),('Shared','shared_input'),('Top5Gap','top5_gap')]:
  if needed is None or k in needed:raw[k]=read(nm)
 br=(h>0).sum(1);raw['LogSize']=np.broadcast_to(np.log(p.mi.core_size.to_numpy())[:,None],p.mask.shape);raw['LogBreadth']=np.broadcast_to(np.log1p(br)[:,None],p.mask.shape);raw['LogProductivity']=np.broadcast_to(np.log1p(p.mi.n_pre.to_numpy())[:,None],p.mask.shape)
 if needed is not None:raw={k:v for k,v in raw.items()if k in needed}
 z={};sc={}
 for k,x in raw.items():
  if k=='C':z[k]=x;sc[k]={'mean':0,'sd':1,'unit':'Class C minus B'}
  else:
   # Zero support is retained and explicitly diagnosed; never divide by zero.
   mu=float(x[p.mask].mean());sd=float(x[p.mask].std());z[k]=(x-mu)/(sd if sd>1e-12 else 1);sc[k]={'mean':mu,'sd':sd,'unit':'one risk-set standard deviation','zero_support':sd<=1e-12}
 for nm,a,b in [('GapApplicability','Gap','Applicability'),('GapAccess','Gap','Access'),('ApplicabilityAccess','Applicability','Access'),('CGap','C','Gap'),('CApplicability','C','Applicability'),('CAccess','C','Access'),('GapLogSize','Gap','LogSize'),('GapLogBreadth','Gap','LogBreadth'),('GapLogProductivity','Gap','LogProductivity'),('Top5GapApplicability','Top5Gap','Applicability'),('Top5GapAccess','Top5Gap','Access')]:
  if a in z and b in z and(needed is None or nm in needed):z[nm]=z[a]*z[b]
 if needed is None:z['GapSquared']=z['Gap']**2
 return raw,z,sc

def first_events(p,pre=False):
 w=p.c.execute(f"SELECT w.* FROM read_parquet('{(p.u/'analysis_works.parquet').as_posix()}') w SEMI JOIN selected USING(g) WHERE w.mo BETWEEN 744 AND 753").fetchdf() if pre else p.ww
 if pre:w=p.localize(w)
 w=w[w.mo==w.first_mo].copy();w['specialist']=w.current_experts>w.current_core_experts
 # Any specialist in the first-month publication bundle receives priority.
 e=w.groupby(['i','j0','mo'],sort=True).agg(specialist=('specialist','max'),papers=('work_id','size')).reset_index();assert not e.duplicated(['i','j0']).any()
 s=e.specialist.to_numpy(float);vals=np.column_stack([np.ones(len(e)),s,1-s,2*s-1]);return e,vals,['total','specialist','no_specialist','signed']

def diagnostic(p,raw,z,sc,out):
 out.mkdir(parents=True,exist_ok=True);names=['Gap','Distance','Applicability','Access','InternalShare','Depth','Distributed','Shared','Top5Gap'];s=np.zeros(len(names));Q=np.zeros((len(names),len(names)));n=0;dist=[]
 for a in range(0,p.I,10000):
  b=min(a+10000,p.I);mk=p.mask[a:b];x=np.column_stack([raw[k][a:b][mk]for k in names]);s+=x.sum(0);Q+=x.T@x;n+=len(x)
 cov=Q/n-np.outer(s/n,s/n);sd=np.sqrt(np.maximum(np.diag(cov),0));corr=cov/np.outer(sd,sd).clip(1e-30);rows=[]
 for k,nm in enumerate(names):
  others=[names.index(x)for x in ['Gap','Distance'] if x!=nm];partial=cov[k,k]-cov[k,others]@np.linalg.pinv(cov[np.ix_(others,others)])@cov[others,k];rows.append({'construct':nm,'mean':s[k]/n,'sd':sd[k],'residual_variance_after_Gap_Distance':max(partial,0),'fraction_variance_beyond_Gap_Distance':max(partial,0)/max(cov[k,k],1e-30),'definition_is_predetermined':True})
 pd.DataFrame(corr,index=names,columns=names).to_csv(out/'correlations.csv');pd.DataFrame(rows).to_csv(out/'construct_support.csv',index=False);save(out/'scaling.json',sc)
 # Complete-risk raw rates, plus support by class; these are descriptive.
 for pre in [True,False]:
  e,v,yn=first_events(p,pre);evg=p.G[e.i.to_numpy(),e.j0.to_numpy()];rows=[]
  for cl in [1,2]:
   em=p.z[e.i.to_numpy(),e.j0.to_numpy()]==cl
   for lo,hi in zip([0,.2,.4,.6,.8],[.2,.4,.6,.8,1.000001]):
    den=0
    for a in range(0,p.I,25000):
     b=min(a+25000,p.I);den+=int((p.mask[a:b]&(p.z[a:b]==cl)&(p.G[a:b]>=lo)&(p.G[a:b]<hi)).sum())
    sel=em&(evg>=lo)&(evg<hi)
    rows.append({'class':'B'if cl==1 else'C','Gap_low':lo,'Gap_high':min(hi,1),'candidate_pairs':den,'months':10 if pre else 11,**{k:int(v[sel,l].sum()) for l,k in enumerate(yn)},'window':'pre2022'if pre else'AI2025_26'})
  pd.DataFrame(rows).to_csv(out/('raw_rates_pre.csv'if pre else'raw_rates_AI.csv'),index=False)
 save(out/'sample.json',{'cores':p.I,'pairs':int(p.mask.sum()),'BC_B_pairs':int((p.mask&(p.z==1)).sum()),'BC_C_pairs':int((p.mask&(p.z==2)).sum()),'destinations':p.J,'paths':p.paths.tolist(),'cohort':p.co,'population':p.pop,'mapping':p.mapping,'freeze':2021,'original_eligibility_and_home_gate_preserved':True})
