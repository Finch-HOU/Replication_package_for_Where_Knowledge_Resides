"""Fixed-protocol follow-up. Original research projects are read-only inputs."""
from pathlib import Path
import os,sys,json,time,hashlib,gc
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='8'
F=Path(__file__).resolve().parents[1];ROOT=F.parent
H=ROOT/'expanded_sample_reassessment_20261102';P=ROOT/'portfolio_A_theory_driven_exploration_20261103';V=ROOT/'temporal_productivity_validation_20261104';W=ROOT/'knowledge_architecture_working_paper_20261104'
sys.path.insert(0,str(H/'08_reproducibility'));sys.path.insert(1,str(H))
import numpy as np,pandas as pd
from panel_api import Panel
from streaming_fe import StreamingDesign,SparseBatch
from scipy.stats import t as td
TERMS=['Gap','Distance','Distributed','Shared'];EXTRAS=['LogBreadth','LogProductivity']
NAMES=[k+'_'+s for k in TERMS for s in ['Destination','Home']]+[k+'_Destination'for k in EXTRAS]
OUTCOMES=['total','specialist','no_specialist','signed'];SCHEMES=['capability','homefield','team','capability_team','homefield_capability']
EXPECTED={'parent':{'cores':602919,'pairs':102614232,'N':1128756552,'events':[9065,3386,5679,9065]},'direct':{'cores':597077,'pairs':70422255,'N':774644805,'events':[7188,2711,4477,7188]}}
def log(*x):print(time.strftime('%Y-%m-%d %H:%M:%S'),*x,flush=True)
def save(p,x):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,default=str),encoding='utf8');q.replace(p)
def load(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as s:
  for b in iter(lambda:s.read(16*1024*1024),b''):h.update(b)
 return h.hexdigest()
def setup(mp):
 p=Panel('recent','dyad',mp,classes='C');raw={'Gap':p.G,'Distance':p.D}
 for k,n in [('Distributed','distributed_input'),('Shared','shared_input')]:raw[k]=np.load(P/'measurement'/(n+'.npy'),mmap_mode='r')[p.g][:,p.js].astype(float)
 raw['LogBreadth']=np.broadcast_to(np.log1p((p.hist>0).sum(1))[:,None],p.mask.shape)
 raw['LogProductivity']=np.broadcast_to(np.log1p(p.mi.n_pre.to_numpy())[:,None],p.mask.shape)
 z={};sc={};ref=load(P/'results/dyad/recent'/mp/'architecture/scaling.json')
 for k,x in raw.items():
  mu=float(x[p.mask].mean());sd=float(x[p.mask].std());assert abs(mu-ref[k]['mean'])<1e-12 and abs(sd-ref[k]['sd'])<1e-12,(mp,k)
  z[k]=(x-mu)/sd;sc[k]={'mean':mu,'sd':sd}
 e=EXPECTED[mp];assert p.I==e['cores']and int(p.mask.sum())==e['pairs']and p.tm.tolist()==list(range(786,797))
 return p,z,sc
def events(p):
 w=p.ww[p.ww.mo==p.ww.first_mo].copy();w['S']=w.current_experts>w.current_core_experts
 e=w.groupby(['i','j0','mo'],sort=True).agg(specialist=('S','max'),papers=('work_id','size')).reset_index()
 assert not e.duplicated(['i','j0']).any();s=e.specialist.to_numpy(float)
 v=np.column_stack([np.ones(len(e)),s,1-s,2*s-1]);assert np.count_nonzero(v,axis=0).tolist()==EXPECTED[p.mapping]['events']
 return e,v
def factory(p,z,extra=None,mods=None):
 extra=[]if extra is None else extra;mods=TERMS if mods is None else mods
 nn=NAMES+[k+'_'+src for k in extra for src in ['Destination','Home']]
 def x(a,b):return np.stack([z[k][a:b,:,None]*(p.A[None,:,:]if src=='Destination'else p.ha[a:b,None,:])for k in TERMS for src in ['Destination','Home']]+[z[k][a:b,:,None]*p.A[None,:,:]for k in EXTRAS]+[z[k][a:b,:,None]*(p.A[None,:,:]if src=='Destination'else p.ha[a:b,None,:])for k in extra for src in ['Destination','Home']],axis=3)
 def c(a,b):return np.stack([z[k][a:b]for k in mods],axis=2)
 return nn,x,c
class ProgressDesign(StreamingDesign):
 def blocks(self):
  self.pass_n=getattr(self,'pass_n',0)+1;mark=0
  for a,b in super().blocks():
   if a>=mark:log('FE pass',self.pass_n,f'{a:,}/{self.I:,} cores');mark=a+max(self.I//4,self.chunk)
   yield a,b
def PSD(V):
 v=np.array(V,float);ev=np.linalg.eigvalsh((v+v.T)/2);scale=max(float(np.abs(ev).max()),1e-30);ok=bool(ev.min()>=-1e-8*scale)
 return {'covariance_min_eigenvalue':float(ev.min()),'covariance_max_abs_eigenvalue':scale,'covariance_PSD':ok,'diagnostic_only_due_to_indefinite_covariance':not ok,'PSD_repair_applied':False}
def gate():
 assert (F/'00_baseline_frozen/BASELINE_RECONCILED').exists(),'Both full-population baselines must replicate before mechanism estimation.'
