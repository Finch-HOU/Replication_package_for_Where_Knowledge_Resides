from common import *
import argparse
def main(mp):
 o=F/'00_baseline_frozen/replication'/mp;o.mkdir(exist_ok=True,parents=True)
 assert (F/'00_baseline_frozen/FREEZE_COMPLETE').exists()
 if(o/'RECONCILED').exists():return
 p,z,sc=setup(mp);e,v=events(p);save(o/'sample.json',{'mapping':mp,**EXPECTED[mp],'standardization':sc,'months':p.tm.tolist(),'five_original_covariance_families':True})
 e.to_parquet(o/'events.parquet',index=False);nn,x,c=factory(p,z)
 d=ProgressDesign(p.mask,p.fc,p.home,nn,x,c,True,4096)
 data=SparseBatch(e.i,e.j0,e.mo-786,v,OUTCOMES,p.I,p.J,11,p.mask)
 d.fit_many(data,o,'',save_joint=True)
 rows=[]
 for out in OUTCOMES:
  a=load(o/(out+'.json'));b=load(V/'results'/mp/'replication'/(out+'.json'));assert a['status']=='estimated'
  err=np.max(np.abs(np.array(a['sufficient_statistics']['beta'])-b['sufficient_statistics']['beta']));assert err<1e-11,(mp,out,err)
  assert a['meta']['N']==b['meta']['N']and a['meta']['nonzero_cells']==b['meta']['nonzero_cells']
  for cl in SCHEMES:
   assert np.allclose(a['covariance'][cl],b['covariance'][cl],rtol=1e-6,atol=1e-14),(mp,out,cl)
   ra=next(q for q in a['coefficients']if q['term']=='Distributed_Home'and q['cluster']==cl);rb=next(q for q in b['coefficients']if q['term']=='Distributed_Home'and q['cluster']==cl)
   for k in ['b','se','p','lo','hi']:assert np.isclose(ra[k],rb[k],rtol=1e-6,atol=1e-12),(out,cl,k)
   rows.append({'mapping':mp,'outcome':out,'cluster':cl,'new_b':ra['b'],'archived_b':rb['b'],'b_x10000':ra['b']*1e4,'new_p':ra['p'],'archived_p':rb['p'],'N':a['meta']['N'],'events':a['meta']['nonzero_cells'],'max_all_coefficient_error':float(err),'passed':True})
 j=np.load(o/'joint_covariances.npz');b=j['beta'];assert np.max(np.abs(b[0]-b[1]-b[2]))<1e-12;assert np.max(np.abs(b[3]-b[1]+b[2]))<1e-12
 for cl in SCHEMES:assert np.allclose(j[cl],np.load(V/'results'/mp/'replication/joint_covariances.npz')[cl],rtol=1e-6,atol=1e-14)
 pd.DataFrame(rows).to_csv(o/'reconciliation.csv',index=False);(o/'RECONCILED').write_text('Actual full-population re-estimation matches full-precision reference, every coefficient/covariance, all risks/events/identities.')
 if all((F/'00_baseline_frozen/replication'/m/'RECONCILED').exists()for m in ['parent','direct']):(F/'00_baseline_frozen/BASELINE_RECONCILED').write_text('Both mappings freshly reproduced. Mechanism estimation gate open.')
 log('BASELINE RECONCILED',mp)
if __name__=='__main__':ap=argparse.ArgumentParser();ap.add_argument('mapping',choices=['parent','direct']);main(ap.parse_args().mapping)
