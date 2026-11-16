"""One cached author-field read, shared by all cohorts and mappings."""
from common import *
import numpy as np,pandas as pd,duckdb,pyarrow.parquet as pq,gc
from numba import njit,prange,set_num_threads
@njit(parallel=True)
def aggregate(ptr,aa,counts,profile,carrier,max_count,best,hist,errors):
 for g in prange(len(ptr)-1):
  total=np.zeros(252,np.uint64)
  for z in range(ptr[g],ptr[g+1]):
   a=aa[z]
   for k in range(252):
    n=counts[a,k]
    total[k]+=n
    if n>0:carrier[g,k]+=1
    if n>max_count[g,k]:max_count[g,k]=n
    if profile[a,k]>best[g,k]:best[g,k]=profile[a,k]
  for k in range(252):
   if total[k]!=hist[g,k]:errors[g]+=1
def main():
 o=F/'measurement';o.mkdir(parents=True,exist_ok=True)
 if (o/'ARCHITECTURE_V2_COMPLETE').exists():return
 u=H/'02_dyad_full_population/universal';sub=pd.read_parquet(u/'frozen/sub.parquet').sort_values('j');oldidx=pd.read_csv(H/'08_reproducibility/inputs/domain_knowledge_input.csv').subfield_id.tolist();perm=[oldidx.index(x)for x in sub.subfield_id];R=np.load(H/'08_reproducibility/inputs/reference_weights_2010_2014_temporal.npy')[np.ix_(perm,perm)];np.save(o/'R.npy',R)
 authors=pd.read_parquet(u/'core_authors.parquet').sort_values('author_id');ids=authors.author_id.to_numpy();np.save(o/'author_ids.npy',ids);n=len(ids);U=pq.read_metadata(u/'cores.parquet').num_rows
 if not (o/'AUTHOR_COUNTS_COMPLETE').exists():
  log('AUTHOR FIELD CACHE',n);c=duckdb.connect();c.execute('SET threads=24');c.register('selected',authors);c.register('sub',pd.DataFrame({'sf':sub.subfield_id.str.extract(r'([0-9]+)$')[0].astype(int),'j':np.arange(252)}));arr=np.lib.format.open_memmap(o/'author_counts.npy',mode='w+',dtype=np.uint32,shape=(n,252));arr[:]=0
  batch=c.execute(f"SELECT a.author_id,s.j,a.n_pre2021 FROM read_parquet('{(H/'01_sample_construction/author_field.parquet').as_posix()}') a JOIN sub s USING(sf) SEMI JOIN selected USING(author_id) WHERE a.n_pre2021>0").fetch_record_batch(500000)
  for i,b in enumerate(batch):
   d=b.to_pandas();ar=np.searchsorted(ids,d.author_id.to_numpy());arr[ar,d.j.to_numpy()]=d.n_pre2021.to_numpy();
   if i%20==0:log('AUTHOR CACHE BATCH',i)
  arr.flush();c.close()
 counts=np.load(o/'author_counts.npy',mmap_mode='r')
 if not (o/'AUTHOR_PROFILES_COMPLETE').exists():
  arr=np.lib.format.open_memmap(o/'author_input_coverage.npy',mode='w+',dtype=np.float32,shape=(n,252))
  for a in range(0,n,50000):arr[a:a+50000]=(counts[a:a+50000]>0).astype(np.float32)@R.T.astype(np.float32)
  arr.flush();(o/'AUTHOR_PROFILES_COMPLETE').write_text('Fully flushed, not inferred from file existence.');log('AUTHOR INPUT PROFILES COMPLETE')
 profiles=np.load(o/'author_input_coverage.npy',mmap_mode='r');sample=np.linspace(0,n-1,1000,dtype=int);assert np.allclose(profiles[sample],(counts[sample]>0).astype(float)@R.T,atol=2e-6);assert profiles[sample].max()>0
 members=pd.read_parquet(u/'members.parquet',columns=['g','author_id']).sort_values('g');sz=np.bincount(members.g,minlength=U);ptr=np.r_[0,np.cumsum(sz)].astype(np.int64);aa=np.searchsorted(ids,members.author_id.to_numpy());assert (ids[aa]==members.author_id.to_numpy()).all();np.save(o/'core_size.npy',sz)
 def mm(name,dtype):
  z=np.lib.format.open_memmap(o/(name+'.npy'),mode='w+',dtype=dtype,shape=(U,252));z[:]=0;return z
 carrier=mm('carriers',np.uint16);maxc=mm('max_member_records',np.uint32);best=mm('best_member_input_coverage',np.float32);set_num_threads(24);hist=np.load(u/'frozen/history.npy',mmap_mode='r');count_errors=np.zeros(U,np.int16);aggregate(ptr,aa,counts,profiles,carrier,maxc,best,hist,count_errors);assert count_errors.sum()==0;(o/'AUTHOR_COUNTS_COMPLETE').write_text('Exact author count sums match all original frozen histories.');log('EXACT AUTHOR COUNT SUMS MATCH ALL ORIGINAL HISTORY CELLS')
 hist=np.load(u/'frozen/history.npy',mmap_mode='r');gap=np.load(u/'frozen/G.npy',mmap_mode='r');shared=mm('shared_input',np.float32);distributed=mm('distributed_input',np.float32);rare=mm('top5_gap',np.float32);top=np.zeros_like(R)
 for j in range(252):ix=np.argsort(-R[j],kind='stable')[:5];top[j,ix]=R[j,ix];top[j]/=max(top[j].sum(),1e-12)
 errors=[];summ=[]
 for a in range(0,U,25000):
  b=min(a+25000,U);known=hist[a:b]>0;assert np.array_equal(carrier[a:b]>0,known);assert (maxc[a:b]<=hist[a:b]).all();union=1-gap[a:b];shared[a:b]=(carrier[a:b]>=2).astype(float)@R.T;dist=union-best[a:b];errors.append(float((-dist).max()));assert dist.min()>-2e-6;distributed[a:b]=np.clip(dist,0,1);rare[a:b]=np.clip(1-known.astype(float)@top.T,0,1);claims=carrier[a:b].sum(1);breadth=known.sum(1);summ.extend([{'g':g,'core_size':int(sz[g]),'union_breadth':int(br),'member_field_claims':int(cl),'redundancy':float(1-br/max(cl,1)),'mean_individual_breadth':float(cl/sz[g])}for g,br,cl in zip(range(a,b),breadth,claims)])
  if a%250000==0:log('ARCHITECTURE',a,'/',U)
 for v in [carrier,maxc,best,shared,distributed,rare]:v.flush()
 pd.DataFrame(summ).to_parquet(o/'core_architecture.parquet',index=False);save(o/'validation.json',{'cores':U,'authors':n,'member_links':len(members),'union_equals_original_known_all_cells':True,'member_count_sums_equal_original_all_cells':True,'author_profile_direct_multiply_1000_checks':True,'max_member_records_bounded_by_original_history':True,'max_float_union_minus_best_violation':max(errors),'freeze':2021,'author_counts_source':str(H/'01_sample_construction/author_field.parquet'),'no_focal_or_future_information':True,'best_member_profile_is_input_coverage_not_actual_method_skills':True});(o/'ARCHITECTURE_V2_COMPLETE').write_text('All count sums, binary unions, sampled input profiles and bounds verified.');log('ARCHITECTURE COMPLETE V2')
if __name__=='__main__':main()
