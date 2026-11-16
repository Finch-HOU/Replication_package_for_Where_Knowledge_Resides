from pathlib import Path
import os,sys,json,time,hashlib
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='8'
F=Path(__file__).resolve().parents[1];ROOT=F.parent;OLD=ROOT/'portfolio_A_theory_driven_exploration_20261103';H=ROOT/'expanded_sample_reassessment_20261102'
sys.path.insert(1,str(H/'08_reproducibility'));sys.path.insert(2,str(H))
import numpy as np,pandas as pd
from panel_api import Panel
from streaming_fe import StreamingDesign,SparseBatch

def log(*x):print(time.strftime('%Y-%m-%d %H:%M:%S'),*x,flush=True)
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,default=str),encoding='utf8');q.replace(p)
def load(p):return json.loads(Path(p).read_text(encoding='utf8'))

def covariance_diagnostic(V):
 v=np.array(V,float);ev=np.linalg.eigvalsh((v+v.T)/2);sc=max(float(np.max(np.abs(ev))),1e-30);ok=bool(ev.min()>=-1e-8*sc)
 return {'covariance_minimum_eigenvalue':float(ev.min()),'covariance_maximum_eigenvalue':float(ev.max()),'covariance_PSD_relative_tolerance':1e-8,'covariance_matrix_PSD_within_tolerance':ok,'inference_diagnostic_only_due_to_non_PSD_covariance':not ok,'PSD_repair_applied':False}
def month(m):return f'{1960+m//12:04d}-{m%12+1:02d}'
TERMS=['Gap','Distance','Distributed','Shared'];EXTRAS=['LogBreadth','LogProductivity'];NAMES=[k+'_'+s for k in TERMS for s in ['Destination','Home']]+[k+'_Destination'for k in EXTRAS];OUTCOMES=['total','specialist','no_specialist','signed'];SCHEMES=['capability','homefield','team','capability_team','homefield_capability']

def setup(mp):
 p=Panel('recent','dyad',mp,classes='C');raw={'Gap':p.G,'Distance':p.D}
 for k,n in [('Distributed','distributed_input'),('Shared','shared_input')]:raw[k]=np.load(OLD/'measurement'/(n+'.npy'),mmap_mode='r')[p.g][:,p.js].astype(float)
 raw['LogBreadth']=np.broadcast_to(np.log1p((p.hist>0).sum(1))[:,None],p.mask.shape);raw['LogProductivity']=np.broadcast_to(np.log1p(p.mi.n_pre.to_numpy())[:,None],p.mask.shape)
 z={};sc={};ref=load(OLD/'results/dyad/recent'/mp/'architecture/scaling.json')
 for k,x in raw.items():
  mu=float(x[p.mask].mean());sd=float(x[p.mask].std());assert abs(mu-ref[k]['mean'])<1e-12 and abs(sd-ref[k]['sd'])<1e-12;z[k]=(x-mu)/sd;sc[k]={'mean':mu,'sd':sd}
 return p,z,sc

def events(p,start,end):
 w=p.c.execute(f"SELECT w.* FROM read_parquet('{(p.u/'analysis_works.parquet').as_posix()}') w SEMI JOIN selected USING(g) WHERE w.mo BETWEEN {start} AND {end} AND w.mo=w.first_mo").fetchdf();w=p.localize(w);w['S']=w.current_experts>w.current_core_experts;e=w.groupby(['i','j0','mo'],sort=True).agg(specialist=('S','max'),papers=('work_id','size')).reset_index();assert not e.duplicated(['i','j0']).any();s=e.specialist.to_numpy(float);v=np.column_stack([np.ones(len(e)),s,1-s,2*s-1]);return e,v

def factory(p,z,T=None):
 T=p.T if T is None else T
 def x(a,b):return np.stack([z[k][a:b,:,None]*(p.A[None,:,:T]if src=='Destination'else p.ha[a:b,None,:T])for k in TERMS for src in ['Destination','Home']]+[z[k][a:b,:,None]*p.A[None,:,:T]for k in EXTRAS],axis=3)
 def c(a,b):return np.stack([z[k][a:b]for k in TERMS],axis=2)
 return x,c

class ProgressDesign(StreamingDesign):
 def blocks(self):
  self.pass_n=getattr(self,'pass_n',0)+1;mark=0
  for a,b in super().blocks():
   if a>=mark:log('FE pass',self.pass_n,f'{a:,}/{self.I:,} cores');mark=a+max(self.I//4,self.chunk)
   yield a,b

def sample(p,sc):return {'cohort':'recent','construction':'dyad','classes':'C','cores':p.I,'candidate_pairs':int(p.mask.sum()),'destinations':p.J,'capability_paths':p.paths.tolist(),'freeze':2021,'standardization':sc,'zero_inclusive':True,'after_entry_outcomes_remain_zero':True,'mapping':p.mapping}

def ppml_root(mp,regime):
 r=F/'relative_rate_certified_rankreuse'/mp/regime
 original=F/'relative_rate'/mp/regime
 if(r/'COMPLETE').exists()and(r/'certification.json').exists():
  cert=load(r/'certification.json')
  if cert.get('validated_exact_likelihood_and_cap_covariance')is True and ppml_cap_converged(r):
   keys=cert.get('verified_covariance_keys',ppml_covariance_keys(r)if ppml_all_covariances_converged(r)else[])
   if len(keys)>len(ppml_covariance_keys(original)):return r
 c=F/'relative_rate_certified'/mp/regime
 if(c/'COMPLETE').exists()and(c/'certification.json').exists()and load(c/'certification.json').get('validated_exact_likelihood_and_cap_covariance')is True and ppml_all_covariances_converged(c):return c
 a=F/'relative_rate'/mp/regime;b=F/'relative_rate_accelerated'/mp/regime
 if not(b/'COMPLETE').exists():return a
 if not(a/'COMPLETE').exists():return b
 def converged(o):
  s=load(o/'execution.json').get('statuses',{}).get('cap_status.csv',{});return str(s.get('return_code')).strip()in['0','0.0']and str(s.get('converged','')).strip()in['1','1.0']
 return a if converged(a)or not converged(b)else b

def ppml_all_covariances_converged(o):
 ex=load(o/'execution.json');ss=ex.get('statuses',{})
 for k in ['cap','home','core','corepath','homepath']:
  x=ss.get(k+'_status.csv',{})
  if str(x.get('return_code')).strip()not in ['0','0.0']or str(x.get('converged')).strip()not in ['1','1.0']:return False
  if not all((o/(k+s)).exists()for s in ['_coefficients.csv','_covariance.csv']):return False
 return True

def ppml_cap_converged(o):
 s=load(o/'execution.json').get('statuses',{}).get('cap_status.csv',{})
 return str(s.get('return_code')).strip()in ['0','0.0']and str(s.get('converged')).strip()in ['1','1.0']and(o/'cap_coefficients.csv').exists()and(o/'cap_covariance.csv').exists()

def ppml_covariance_keys(o):
 if not(o/'execution.json').exists():return[]
 ss=load(o/'execution.json').get('statuses',{});out=[]
 for k in ['cap','home','core','corepath','homepath']:
  z=ss.get(k+'_status.csv',{})
  if str(z.get('return_code')).strip()in ['0','0.0']and str(z.get('converged')).strip()in ['1','1.0']and all((o/(k+s)).exists()for s in ['_coefficients.csv','_covariance.csv']):out.append(k)
 return out
