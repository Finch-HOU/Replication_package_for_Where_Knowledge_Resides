"""Global binary contact knowledge, exact on unfamiliar Class C destinations.

Union knowledge over all coauthors on each pre-2022 <=20-author publication,
then over an author's publications. Own knowledge is harmless on C destinations
and avoids explicit quadratic collaborator pairs. No new exposure definition.
"""
from pathlib import Path
import sys,time,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import *
import duckdb,numpy as np,pandas as pd,pyarrow.parquet as pq
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--method',default='A',choices=['A','B']);args=ap.parse_args();o=H/'01_sample_construction/global_contact_knowledge';o.mkdir(exist_ok=True);c=duckdb.connect(str(o/'contact_knowledge.duckdb'));c.execute('SET threads=24');c.execute("SET memory_limit='100GB'");c.execute('PRAGMA disable_progress_bar');c.execute(f"ATTACH '{q(H/'02_dyad_full_population/universal/network/binary_access.duckdb')}' AS baseline_bits (READ_ONLY)")
 def stage(name,sql):
  if c.execute('SELECT count(*) FROM information_schema.tables WHERE table_name=?',[name]).fetchone()[0]:return
  log('GLOBAL CONTACT BUILD',name);t=time.time();c.execute('CREATE TABLE '+name+' AS '+sql);log('GLOBAL CONTACT DONE',name,c.execute('SELECT count(*) FROM '+name).fetchone()[0],round(time.time()-t,1))
 glob=q(H/'01_sample_construction/global_catalog/*.parquet');agg=','.join(f'bit_or(a.b{k}) b{k}' for k in range(4));stage('pub_authors',f"SELECT work_id,UNNEST(ids) author_id FROM read_parquet('{glob}') WHERE sf IS NOT NULL AND y<=2021 AND len(ids)<=20");stage('paper_bits',f'SELECT p.work_id,{agg} FROM pub_authors p JOIN baseline_bits.author_bits a USING(author_id) GROUP BY 1');stage('author_contact_bits',f'SELECT p.author_id,{agg} FROM pub_authors p JOIN paper_bits a USING(work_id) GROUP BY 1')
 path=o/'author_contact_bits.parquet'
 if not path.exists():c.execute(f"COPY author_contact_bits TO '{q(path)}' (FORMAT PARQUET,COMPRESSION ZSTD)")
 u=H/('02_dyad_full_population/universal' if args.method=='A' else '03_multimember_cores/methodB_network/universal');assert (u/'frozen/FROZEN_COMPLETE').exists();p=u/'network';p.mkdir(exist_ok=True);name='core_bits_'+args.method;stage(name,f"SELECT m.g,{agg} FROM read_parquet('{q(u/'members.parquet')}') m JOIN author_contact_bits a USING(author_id) GROUP BY 1");n=pq.read_metadata(u/'cores.parquet').num_rows;cl=np.load(u/'frozen/classes.npy',mmap_mode='r');outpath=p/('global_C_binary.npy' if args.method=='A' else 'network_C_binary.npy');out=np.lib.format.open_memmap(outpath,mode='w+',dtype='bool',shape=(n,252));out[:]=False
 for batch in c.execute('SELECT g,b0::UBIGINT b0,b1::UBIGINT b1,b2::UBIGINT b2,b3::UBIGINT b3 FROM '+name+' ORDER BY g').fetch_record_batch(50000):
  g=batch.column('g').to_numpy();v=np.zeros((len(g),252),bool)
  for k in range(4):
   width=min(64,252-k*64);bb=batch.column('b'+str(k)).to_numpy();v[:,k*64:k*64+width]=((bb[:,None]>>np.arange(width,dtype='uint64'))&1)>0
  out[g]=v&(cl[g]==2)
 out.flush();checks={'method':args.method,'identity':'On baseline C, own and other core members have no destination knowledge; the all-coauthor knowledge union equals the outside-core contact knowledge union.','include_only_actual_pre2022_small_paper_contacts':True}
 if args.method=='A':
  old=np.load(p/'network_C_binary.npy',mmap_mode='r');err=0
  for i in range(0,n,50000):err+=int(np.count_nonzero(old[i:i+50000]!=out[i:i+50000]))
  assert err==0;checks['all_A_C_pair_mismatches']=err
 else:(p/'NETWORK_C_BINARY_COMPLETE').write_text('Global contact knowledge; exact C identity validated on every primary A C pair.')
 c.close();write(p/'global_contact_knowledge_validation.json',checks);log('GLOBAL CONTACT COMPLETE',args.method,checks)
if __name__=='__main__':main()
