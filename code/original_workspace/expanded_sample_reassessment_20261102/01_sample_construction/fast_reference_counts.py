"""Exact file-wise reference classification with sorted immutable lookup arrays.

Avoids a multi-billion-edge grouped hash table. Each citing work is independent;
deduplicated reference IDs are classified against the ORIGINAL reference lookup,
using strictly prior publication years. Every catalog shard checkpoints.
"""
from pathlib import Path
import sys,time,json,os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import *
import numpy as np,duckdb,pyarrow as pa,pyarrow.parquet as pq,pyarrow.compute as pc
from numba import njit,prange,set_num_threads
from concurrent.futures import ProcessPoolExecutor,as_completed
F=H/'02_dyad_full_population/universal/content/fast_counts';F.mkdir(exist_ok=True);PART=F/'parts';PART.mkdir(exist_ok=True)

@njit(cache=True,parallel=True)
def select_wanted(ids,wanted):
 out=np.zeros(len(ids),np.bool_)
 for n in prange(len(ids)):
  i=np.searchsorted(wanted,ids[n]);out[n]=i<len(wanted) and wanted[i]==ids[n]
 return out
@njit(cache=True,parallel=True)
def count_refs(off,refs,y,keys,years,fields):
 out=np.zeros((len(y),252),np.uint32)
 for n in prange(len(y)):
  rr=np.sort(refs[off[n]:off[n+1]].copy());last=np.uint64(0)
  for r in rr:
   if r==0 or r==last:continue
   last=r;k=np.searchsorted(keys,r)
   if k<len(keys) and keys[k]==r and years[k]<y[n]:out[n,fields[k]]+=np.uint32(1)
 return out
def init_worker():
 global KEYS,YEARS,FIELDS,WANTED
 set_num_threads(2);KEYS=np.load(F/'lookup_id.npy',mmap_mode='r');YEARS=np.load(F/'lookup_year.npy',mmap_mode='r');FIELDS=np.load(F/'lookup_field.npy',mmap_mode='r');WANTED=np.load(F/'wanted_id.npy',mmap_mode='r')
def worker(job):
 n,src=job;src=Path(src);out=PART/f'{n:05d}.parquet';audit=out.with_suffix('.json')
 if out.exists() and audit.exists():pq.read_metadata(out);return json.loads(audit.read_text())
 ar=pq.read_table(src,columns=['work_id','y','refs'],use_threads=False);ids=ar['work_id'].combine_chunks().to_numpy();hit=np.flatnonzero(select_wanted(ids,WANTED));schema=pa.schema([('work_id',pa.uint64()),('rj',pa.int16()),('n',pa.uint32())]);tmp=out.with_suffix('.tmp');wr=pq.ParquetWriter(tmp,schema,compression='zstd');valid=0;cells=0
 for start in range(0,len(hit),100000):
  part=ar.take(pa.array(hit[start:start+100000]));ww=part['work_id'].combine_chunks().to_numpy();yy=part['y'].combine_chunks().fill_null(2147483647).to_numpy();rr=part['refs'].combine_chunks();val=rr.values.fill_null(0).to_numpy();off=rr.offsets.to_numpy();counts=count_refs(off,val,yy,KEYS,YEARS,FIELDS);i,j=np.nonzero(counts);values=counts[i,j];valid+=int(values.astype(np.uint64).sum());cells+=len(i)
  if len(i):wr.write_table(pa.Table.from_arrays([pa.array(ww[i],type=pa.uint64()),pa.array(j.astype('int16')),pa.array(values,type=pa.uint32())],schema=schema),row_group_size=1000000)
 wr.close();os.replace(tmp,out);z={'source':str(src),'source_size':src.stat().st_size,'wanted_works':len(hit),'prior_classified_refs':valid,'field_cells':cells};write(audit,z);return z

def main():
 o=F.parent;assert (o/'ORIGINAL_REFERENCE_LOOKUP_COMPLETE').exists();c=duckdb.connect(str(o/'content.duckdb'),read_only=True);c.execute('SET threads=16');c.execute("SET memory_limit='100GB'");c.execute('PRAGMA disable_progress_bar');c.execute(f"SET temp_directory='{q(F/'spill')}'")
 if not (F/'LOOKUP_ARRAYS_COMPLETE').exists():
  N=c.execute('SELECT count(*) FROM lookup').fetchone()[0];keys=np.lib.format.open_memmap(F/'lookup_id.npy',mode='w+',dtype='uint64',shape=(N,));years=np.lib.format.open_memmap(F/'lookup_year.npy',mode='w+',dtype='int32',shape=(N,));fields=np.lib.format.open_memmap(F/'lookup_field.npy',mode='w+',dtype='int16',shape=(N,));off=0;prev=None
  for batch in c.execute('SELECT refid,ry,rj FROM lookup ORDER BY refid').fetch_record_batch(2000000):
   kk=batch.column('refid').to_numpy();assert len(kk)==0 or ((prev is None or kk[0]>prev) and np.all(kk[1:]>kk[:-1])),'Nonunique original reference IDs';keys[off:off+len(kk)]=kk;years[off:off+len(kk)]=batch.column('ry').fill_null(2147483647).to_numpy();fields[off:off+len(kk)]=batch.column('rj').to_numpy();off+=len(kk);prev=kk[-1]
  assert off==N;keys.flush();years.flush();fields.flush();(F/'LOOKUP_ARRAYS_COMPLETE').write_text(str(N));log('REFERENCE LOOKUP ARRAYS',N)
 if not (F/'WANTED_ARRAY_COMPLETE').exists():
  N=c.execute('SELECT count(*) FROM wanted').fetchone()[0];want=np.lib.format.open_memmap(F/'wanted_id.npy',mode='w+',dtype='uint64',shape=(N,));off=0
  for batch in c.execute('SELECT work_id FROM wanted ORDER BY work_id').fetch_record_batch(2000000):v=batch.column('work_id').to_numpy();want[off:off+len(v)]=v;off+=len(v)
  assert off==N;want.flush();(F/'WANTED_ARRAY_COMPLETE').write_text(str(N));log('REFERENCE WANTED ARRAY',N)
 c.close();files=sorted((H/'01_sample_construction/global_catalog').glob('*.parquet'));start=time.time();stats=[]
 with ProcessPoolExecutor(max_workers=8,initializer=init_worker) as ex:
  fs=[ex.submit(worker,z) for z in enumerate(files)]
  for n,f in enumerate(as_completed(fs),1):
   stats.append(f.result())
   if n%100==0:log('EXACT REFERENCE PARTS',n,'/',len(files),'seconds',round(time.time()-start,1))
 n=sum(s['wanted_works'] for s in stats);expected=int((F/'WANTED_ARRAY_COMPLETE').read_text());assert n==expected,(n,expected,'Missing or duplicate target work IDs');write(F/'audit.json',{'wanted_works':n,'prior_classified_reference_edges':sum(s['prior_classified_refs'] for s in stats),'nonzero_work_field_cells':sum(s['field_cells'] for s in stats),'files':len(files),'elapsed_seconds':time.time()-start,'lookup':'Original BASE work_domain_lookup, not a changed reference-eligibility scope','strict_prior_year':True,'reference_IDs_deduplicated':True});(F/'FAST_COUNTS_COMPLETE').write_text('Every original target work classified, every source shard checkpointed.');log('EXACT REFERENCE COUNTS COMPLETE')
if __name__=='__main__':main()
