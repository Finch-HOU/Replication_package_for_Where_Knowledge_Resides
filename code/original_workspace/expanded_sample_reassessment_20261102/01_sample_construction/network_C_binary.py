"""Exact C-only access by bitset unions; internal members cannot know a C field.

This does not replace full distinct-contact counts. For Class C only, including
internal contacts in a binary union is algebraically irrelevant because no core
member has baseline destination experience. Full counts continue separately.
"""
from pathlib import Path
import sys,json,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import *
import duckdb,numpy as np,pandas as pd,pyarrow.parquet as pq
def main():
 u=H/'02_dyad_full_population/universal';p=u/'network';pq.read_metadata(p/'small_ties.parquet');c=duckdb.connect(str(p/'binary_access.duckdb'));c.execute('SET threads=32');c.execute("SET memory_limit='150GB'");c.execute('PRAGMA disable_progress_bar')
 def stage(name,sql):
  if c.execute('SELECT count(*) FROM information_schema.tables WHERE table_name=?',[name]).fetchone()[0]:return
  log('C ACCESS BUILD',name);t=time.time();c.execute('CREATE TABLE '+name+' AS '+sql);log('C ACCESS DONE',name,c.execute('SELECT count(*) FROM '+name).fetchone()[0],round(time.time()-t,1))
 expr=','.join(f"bit_or(CASE WHEN s.j BETWEEN {k*64} AND {min(k*64+63,251)} THEN 1::UBIGINT<<(s.j-{k*64}) ELSE 0::UBIGINT END) b{k}" for k in range(4))
 stage('author_bits',f"SELECT a.author_id,{expr} FROM read_parquet('{q(H/'01_sample_construction/author_field.parquet')}') a JOIN read_parquet('{q(OLD/'output/sub.parquet')}') s ON a.sf=try_cast(regexp_extract(s.subfield_id,'([0-9]+)$',1) AS INTEGER) WHERE a.n_pre2021>0 GROUP BY 1")
 agg=','.join(f'bit_or(b{k}) b{k}' for k in range(4));stage('contact_bits',f"SELECT t.core_author,{agg} FROM read_parquet('{q(p/'small_ties.parquet')}') t JOIN author_bits a ON a.author_id=t.collaborator WHERE t.first_small_y<=2021 GROUP BY 1");stage('core_bits',f"SELECT m.g,{agg} FROM read_parquet('{q(u/'members.parquet')}') m JOIN contact_bits c ON m.author_id=c.core_author GROUP BY 1")
 n=pq.read_metadata(u/'cores.parquet').num_rows;out=np.lib.format.open_memmap(p/'network_C_binary.npy',mode='w+',dtype='bool',shape=(n,252));out[:]=False;cl=np.load(u/'frozen/classes.npy',mmap_mode='r');r=c.execute('SELECT * FROM core_bits ORDER BY g').fetch_record_batch(50000)
 for batch in r:
  d=batch.to_pandas();gi=d.g.to_numpy();v=np.zeros((len(d),252),bool)
  for k in range(4):
   width=min(64,252-k*64);bits=d['b'+str(k)].to_numpy(dtype='uint64');v[:,k*64:k*64+width]=((bits[:,None]>>np.arange(width,dtype='uint64')[None,:])&1).astype(bool)
  out[gi]=v&(cl[gi]==2)
 out.flush();checks=[]
 for co,z in COHORTS.items():
  m=pd.read_parquet(H/'02_dyad_full_population'/co/'global_core_index.parquet');m=m[m.hash_rank<=30000].sort_values('team_id');old=np.load(EXT/'output'/z['old_dir']/'network_access.npz')['count'];origcl=np.load(EXT/'output'/z['old_dir']/'frozen_constructs_primary.npz')['classes'];a=out[m.g];mask=origcl==2;err=int(np.count_nonzero(a[mask]!=(old[mask]>0)));assert err==0,(co,err);checks.append({'cohort':co,'mismatches_on_every_original_C_pair':err,'identity':'Internal members have zero pre-2022 destination expertise in C; binary union therefore exactly equals the outside-core contact union on C.'});log('C ACCESS VERIFIED',co)
 write(p/'binary_access_validation.json',checks);(p/'NETWORK_C_BINARY_COMPLETE').write_text('Exact binary access on C; all original C-pair values verified. A/B values intentionally not used.');c.close();log('C ACCESS COMPLETE')
if __name__=='__main__':main()
