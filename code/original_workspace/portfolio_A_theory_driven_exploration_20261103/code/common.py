from pathlib import Path
import sys,os,json,time,hashlib
F=Path(__file__).resolve().parents[1];H=F.parent/'expanded_sample_reassessment_20261102';PRE=F.parent/'empirical_feasibility_assessment_20261103'
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='4'
sys.path.insert(0,str(H));sys.path.insert(0,str(H/'08_reproducibility'))
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,default=str),encoding='utf-8');t.replace(p)
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def log(*x):print(time.strftime('%Y-%m-%d %H:%M:%S'),*x,flush=True)
def primary(pop,co,mp):return H/'04_reestimated_tables'/pop/co/'all/full'/mp

def fit_complete(out,prefix):
 p=out/(prefix+'signed.json')
 if not p.exists():return False
 r=load(p)
 return r.get('status')=='rank_deficient' or all((out/(prefix+x)).exists()for x in ['joint_covariances.npz','joint_metadata.json'])
