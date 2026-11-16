"""Exact distinct-contact field counts without billion-row GROUP BY state.

The already deduplicated core-contact relation is joined to four knowledge
words per collaborator. Each set field bit increments the dense core-field
count once. This is identical to joining author-field records and grouping.
"""
from pathlib import Path
import sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import *
import duckdb,numpy as np,pandas as pd,pyarrow.parquet as pq
from numba import njit
@njit
def trailing_zero(x):
 b=0
 if (x&np.uint64(0xffffffff))==0:x>>=np.uint64(32);b+=32
 if (x&np.uint64(0xffff))==0:x>>=np.uint64(16);b+=16
 if (x&np.uint64(0xff))==0:x>>=np.uint64(8);b+=8
 if (x&np.uint64(0xf))==0:x>>=np.uint64(4);b+=4
 if (x&np.uint64(0x3))==0:x>>=np.uint64(2);b+=2
 if (x&np.uint64(0x1))==0:b+=1
 return b
@njit
def add_counts(out,g,words):
 for i in range(len(g)):
  for k in range(4):
   w=words[i,k]
   while w!=0:
    j=k*64+trailing_zero(w)
    if j<252:out[g[i],j]+=np.uint32(1)
    w&=w-np.uint64(1)
def main():
 u=H/'02_dyad_full_population/universal';o=u/'network';c=duckdb.connect();c.execute('SET threads=16');c.execute("SET memory_limit='80GB'");c.execute('PRAGMA disable_progress_bar');c.execute(f"ATTACH '{q(o/'binary_access.duckdb')}' AS bits (READ_ONLY)");n=pq.read_metadata(u/'cores.parquet').num_rows;dest=o/'network.npy';out=np.lib.format.open_memmap(dest,mode='w+',dtype='uint32',shape=(n,252));out[:]=0;start=time.time();processed=0
 query=f"SELECT n.g,a.b0::UBIGINT b0,a.b1::UBIGINT b1,a.b2::UBIGINT b2,a.b3::UBIGINT b3 FROM read_parquet('{q(o/'core_contacts.parquet')}') n JOIN bits.author_bits a ON a.author_id=n.collaborator"
 for batch in c.execute(query).fetch_record_batch(2000000):
  g=batch.column('g').to_numpy(zero_copy_only=False);words=np.column_stack([batch.column('b'+str(k)).to_numpy(zero_copy_only=False) for k in range(4)]).astype('uint64',copy=False);add_counts(out,g,words);processed+=len(g)
  if processed%50000000==0:log('EXACT NETWORK DENSE COUNTS',processed,'seconds',round(time.time()-start,1))
 out.flush();checks=[]
 for co,z in COHORTS.items():
  m=pd.read_parquet(H/'02_dyad_full_population'/co/'global_core_index.parquet');m=m[m.hash_rank<=30000].sort_values('team_id');old=np.load(EXT/'output'/z['old_dir']/'network_access.npz')['count'];err=int(np.count_nonzero(out[m.g.to_numpy()]!=old));assert err==0,(co,err);checks.append({'cohort':co,'all_252_original_field_counts_mismatches':err});log('EXACT FULL NETWORK COUNTS VERIFIED',co)
 oldC=np.load(o/'network_C_binary.npy',mmap_mode='r');cl=np.load(u/'frozen/classes.npy',mmap_mode='r');errC=0
 for i in range(0,n,50000):mask=cl[i:i+50000]==2;errC+=int(np.count_nonzero((out[i:i+50000]>0)[mask]!=oldC[i:i+50000][mask]))
 assert errC==0;write(o/'dense_counts_validation.json',{'checks':checks,'all_full_population_C_binary_mismatches':errC,'deduplicated_core_contacts_with_mapped_expertise':processed,'elapsed_seconds':time.time()-start,'algorithm':'Exact set-bit summation on a pre-deduplicated contact relation; no contact-level approximation.'});(o/'NETWORK_MATRIX_COMPLETE').write_text('Full distinct-contact counts, all original 252-field values verified.');(o/'NETWORK_COMPLETE').write_text('Exact dense matrix is the canonical count checkpoint; sparse grouped export unnecessary.');c.close();log('EXACT NETWORK DENSE COMPLETE')
if __name__=='__main__':main()
