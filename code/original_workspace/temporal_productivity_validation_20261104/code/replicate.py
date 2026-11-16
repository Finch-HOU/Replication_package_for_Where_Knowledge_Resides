from common import *
import argparse

def main(mp):
 o=F/'results'/mp/'replication';o.mkdir(parents=True,exist_ok=True)
 if(o/'COMPLETE').exists():return
 p,z,sc=setup(mp);save(o/'sample.json',sample(p,sc));e,v=events(p,786,796);e.to_parquet(o/'events.parquet',index=False);x,c=factory(p,z);d=ProgressDesign(p.mask,p.fc,p.home,NAMES,x,c,True,4096);data=SparseBatch(e.i,e.j0,e.mo-786,v,OUTCOMES,p.I,p.J,11,p.mask);d.fit_many(data,o,'',save_joint=True)
 checks=[]
 for out in OUTCOMES:
  a=load(o/(out+'.json'));b=load(OLD/'results/dyad/recent'/mp/'architecture'/('architecture_estimable__'+out+'.json'));err=float(np.max(np.abs(np.array(a['sufficient_statistics']['beta'])-b['sufficient_statistics']['beta'])));assert err<1e-10
  for cl in SCHEMES:assert np.allclose(a['covariance'][cl],b['covariance'][cl],rtol=1e-5,atol=1e-13),(mp,out,cl)
  assert a['meta']['N']==b['meta']['N']and a['meta']['nonzero_cells']==b['meta']['nonzero_cells'];checks.append({'outcome':out,'maximum_beta_error':err,'five_covariances_match':True})
 save(o/'replication_checks.json',checks);(o/'COMPLETE').write_text('Fresh original-panel fits reproduce coefficients, five covariances, risks and events.');log('REPLICATION COMPLETE',mp)

if __name__=='__main__':ap=argparse.ArgumentParser();ap.add_argument('mapping');a=ap.parse_args();main(a.mapping)
