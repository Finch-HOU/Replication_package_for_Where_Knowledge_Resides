"""Shared complete-risk panel access for expanded archival specifications."""
from pathlib import Path
import sys,os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='8'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import *
import numpy as np,pandas as pd,duckdb
from streaming_fe import StreamingDesign,SparseBatch

class Panel:
 def __init__(self,co,pop,mapping,classes='BC',size='full',require_home=True,size_group='all'):
  self.co,self.pop,self.mapping=co,pop,mapping;self.size=size;self.u=H/'02_dyad_full_population/universal';f=self.u/'frozen';self.s=pd.read_parquet(f/'sub.parquet').sort_values('j');self.c=duckdb.connect();self.c.execute('SET threads=8');self.c.execute('PRAGMA disable_progress_bar')
  fp=H/('02_dyad_full_population' if pop=='dyad' else '03_multimember_cores/methodA_exact_all')/co;mi=pd.read_parquet(fp/'global_core_index.parquet')
  if size!='full':mi=mi[mi.hash_rank<=int(size)]
  if size_group=='history_quality':
   cv=pd.read_parquet(self.u/'content/coverage.parquet');ok=(cv.baseline_works_refs5>=5)&(cv.baseline_prior_refs/cv.baseline_recorded_refs.replace(0,np.nan)>=.5);mi=mi[mi.g.isin(cv.loc[ok,'g'])]
  elif size_group!='all':mi=mi[mi.core_size.eq(3) if size_group=='3' else mi.core_size.between(4,5) if size_group=='4-5' else mi.core_size.ge(6)]
  mi=mi.sort_values('team_id');maps=pd.read_csv(MAPPING);m=maps[maps.mapping==mapping].set_index('candidate_subfield_id');self.js=np.flatnonzero(self.s.subfield_id.isin(m.index));fs=m.field.reindex(self.s.subfield_id.iloc[self.js]).to_numpy();self.paths,self.fc=np.unique(fs,return_inverse=True);cap=pd.read_csv(CAPABILITY).query("protocol=='mix'");a=cap.pivot(index='mo',columns='field',values='gain12_l6').reindex(columns=fs).dropna();a=a[(a.index>=768)&(a.index<=797)];self.tm=a.index.to_numpy(int);self.A=a.to_numpy().T;assert self.tm.tolist()==list(range(786,797))
  g=mi.g.to_numpy();h=np.load(f/'history.npy',mmap_mode='r')[g].astype(float);z=np.load(f/'classes.npy',mmap_mode='r')[g][:,self.js];valid=h[:,self.js].sum(1)>0;mask=(z==2 if classes=='C' else z>0 if classes=='BC' else np.ones(z.shape,bool))
  if require_home:mask&=valid[:,None]
  keep=mask.any(1);self.mi=mi.iloc[np.flatnonzero(keep)].copy();self.g=self.mi.g.to_numpy();self.hist=h[keep];self.z=z[keep];self.mask=mask[keep];self.G=np.load(f/'G.npy',mmap_mode='r')[self.g][:,self.js];self.D=np.load(f/'D.npy',mmap_mode='r')[self.g][:,self.js];self.home=np.load(f/'home.npy',mmap_mode='r')[self.g];self.I,self.J=self.mask.shape;self.T=len(self.tm);self.cy=(self.z==2).astype(float);hh=self.hist[:,self.js];self.ha=(hh/hh.sum(1,keepdims=True).clip(1))@self.A;self.loc=np.full(len(np.load(f/'home.npy',mmap_mode='r')),-1,np.int32);self.loc[self.g]=np.arange(self.I);self.jloc=np.full(252,-1,np.int16);self.jloc[self.js]=np.arange(self.J);self.c.register('selected',pd.DataFrame({'g':self.g}));self.ww=self.c.execute(f"SELECT w.* FROM read_parquet('{q(self.u/'analysis_works.parquet')}') w SEMI JOIN selected USING(g) WHERE w.mo BETWEEN 786 AND 796").fetchdf();self.ww=self.localize(self.ww)
 def localize(self,w):
  w=w[w.j.isin(self.js)&w.g.isin(self.g)].copy();w['i']=self.loc[w.g.to_numpy()];w['j0']=self.jloc[w.j.to_numpy()];return w[self.mask[w.i.to_numpy(),w.j0.to_numpy()]].copy()
 def std(self,v):
  mu=float(v[self.mask].mean());sd=float(v[self.mask].std());return (v-mu)/sd,{'mean':mu,'sd':sd}
 def aggregate(self,w,values,names):
  return SparseBatch(w.i,w.j0,w.mo-int(self.tm.min()),values,names,self.I,self.J,self.T,self.mask)
 def fit(self,folder,prefix,names,factory,control,w,values,outcomes,save_team=False):
  folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
  if all((folder/(prefix+k+'.json')).exists() for k in outcomes):return
  data=self.aggregate(w,values,outcomes);d=StreamingDesign(self.mask,self.fc,self.home,names,factory,control,True,1024 if len(names)>4 else 2048);result=d.fit_many(data,folder,prefix,save_team);del d;return result
 def h2_factory(self,a,b):return np.stack([self.cy[a:b,:,None]*self.A[None,:,:],self.cy[a:b,:,None]*self.ha[a:b,None,:]],axis=3)
 def h2_control(self,a,b):return self.cy[a:b]
 def firsts(self):return self.ww[self.ww.mo==self.ww.first_mo].copy()
 def grouped(self,w=None):
  w=self.ww if w is None else w
  return w.assign(spec=w.current_experts>w.current_core_experts).groupby(['i','j0','mo'],sort=True).agg(first_mo=('first_mo','first'),n=('work_id','size'),spec=('spec','max')).reset_index()
 def archive(self):
  if self.size!='full':
   o=H/'08_reproducibility/extension_benchmark'/self.pop/self.co/self.mapping;o.mkdir(parents=True,exist_ok=True);return o
  return H/'04_reestimated_tables'/self.pop/self.co/'all'/'full'/self.mapping/'archive'
