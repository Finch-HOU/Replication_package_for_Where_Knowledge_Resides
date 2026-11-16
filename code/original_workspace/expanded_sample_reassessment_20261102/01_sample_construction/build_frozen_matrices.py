from pathlib import Path
import sys,os,time,json,argparse
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='16'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import *
import numpy as np,pandas as pd,pyarrow.parquet as pq
from numba import njit,prange,set_num_threads

@njit(parallel=True)
def nearest(known,S,out):
 for i in prange(len(known)):
  for j in range(252):out[i,j]=1.
  for k in range(252):
   if known[i,k]:
    for j in range(252):
     d=max(0.,1-S[k,j])
     if d<out[i,j]:out[i,j]=d

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--method',choices=['A','B'],default='A');args=ap.parse_args();p=H/('02_dyad_full_population/universal' if args.method=='A' else '03_multimember_cores/methodB_network/universal');files=['cores.parquet','member_first.parquet','joint_pre.parquet']
 for f in files:pq.read_metadata(p/f)
 n=pq.read_metadata(p/'cores.parquet').num_rows;shape=(n,252);out=p/'frozen';out.mkdir(exist_ok=True)
 def matrix(name,dtype):
  q0=out/(name+'.npy')
  if q0.exists():return np.load(q0,mmap_mode='r+')
  a=np.lib.format.open_memmap(q0,mode='w+',dtype=dtype,shape=shape);a[:]=0;return a
 if not (out/'COUNTS_COMPLETE').exists():
  hist=matrix('history',np.uint32);h18=matrix('history2018',np.uint32);joint=matrix('joint',np.uint32);a=pd.read_parquet(p/'member_first.parquet');assert a.n_history.max()<2**32;hist[a.g,a.j]=a.n_history.astype('uint32');h18[a.g,a.j]=a.n_history2018.astype('uint32');b=pd.read_parquet(p/'joint_pre.parquet');joint[b.g,b.j]=b.n.astype('uint32');hist.flush();h18.flush();joint.flush();(out/'COUNTS_COMPLETE').write_text('Global pre-2022 counts and prior joint fields.');log('FROZEN COUNTS COMPLETE',n)
 hist=np.load(out/'history.npy',mmap_mode='r');joint=np.load(out/'joint.npy',mmap_mode='r');known=hist>0;assert np.all(known.sum(1)>0);assert np.all((joint==0)|known)
 s=pd.read_parquet(OLD/'output/sub.parquet').sort_values('j');old=pd.read_csv(DOMAIN_INDEX).subfield_id.tolist();perm=[old.index(v) for v in s.subfield_id];R=np.load(REQUIREMENTS)[np.ix_(perm,perm)];S=R@R.T;S/=np.sqrt(np.diag(S))[:,None]*np.sqrt(np.diag(S))[None,:]
 if not (out/'G_COMPLETE').exists():
  G=matrix('G',np.float64)
  for start in range(0,n,25000):G[start:start+25000]=np.clip(1-known[start:start+25000].astype(float)@R.T,0,1)
  G.flush();(out/'G_COMPLETE').write_text('Same frozen reference-input Gap as foundation.');log('GAP COMPLETE')
 if not (out/'D_COMPLETE').exists():
  D=matrix('D',np.float64);set_num_threads(64)
  for start in range(0,n,50000):
   nearest(known[start:start+50000],S,D[start:start+50000])
   if start%250000==0:log('DISTANCE',start,'/',n)
  D.flush();(out/'D_COMPLETE').write_text('Same nearest-known citation cosine distance.');log('DISTANCE COMPLETE')
 if not (out/'classes.npy').exists():
  cl=matrix('classes',np.int8)
  for start in range(0,n,50000):cl[start:start+50000]=np.where(joint[start:start+50000]>0,0,np.where(known[start:start+50000],1,2))
  cl.flush()
 fields=pd.factorize(s.candidate_field_id if 'candidate_field_id' in s else s.field_id)[0];home=np.zeros(n,np.int16);onehot=np.eye(len(np.unique(fields)),dtype=np.uint32)[fields]
 for start in range(0,n,50000):home[start:start+50000]=(joint[start:start+50000].astype(float)@onehot.astype(float)).argmax(1)
 np.save(out/'home.npy',home);s.to_parquet(out/'sub.parquet',index=False)
 checks=[]
 for tag,z in COHORTS.items():
  if args.method!='A':continue
  m=pd.read_parquet(H/'02_dyad_full_population'/tag/'global_core_index.parquet');m=m[m.hash_rank<=30000].sort_values('team_id');old=np.load(EXT/'output'/z['old_dir']/'frozen_constructs_primary.npz');indices=m.g.to_numpy();diffs={}
  for name,srcname in [('history','history'),('G','G'),('D','D'),('classes','classes')]:
   new=np.load(out/(name+'.npy'),mmap_mode='r')[indices];ref=old[srcname];err=float(np.max(np.abs(new.astype(float)-ref)));assert err<1e-10,(tag,name,err);diffs[name]=err
  assert np.array_equal(home[indices],old['home']);checks.append({'cohort':tag,'all_original_30k_constructs_match':True,'maximum_absolute_errors':diffs});log('FULL RECONSTRUCTION MATCHES 30K',tag,diffs)
 write(out/'original_construct_validation.json',checks);(out/'FROZEN_COMPLETE').write_text('Same constructs reconstructed from full raw cache; A includes exact original 30k validation.');log('FROZEN MATRIX COMPLETE')
if __name__=='__main__':main()
