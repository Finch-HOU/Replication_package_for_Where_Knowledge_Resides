from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import *
import numpy as np,pandas as pd,pyarrow.parquet as pq
def main():
 p0=H/'02_dyad_full_population/universal/network'
 if (p0/'NETWORK_MATRIX_COMPLETE').exists():log('NETWORK MATRIX ALREADY COMPLETE AND VERIFIED');return
 u=H/'02_dyad_full_population/universal';p=u/'network';assert (p/'NETWORK_COMPLETE').exists();n=pq.read_metadata(u/'cores.parquet').num_rows;a=np.lib.format.open_memmap(p/'network.npy',mode='w+',dtype='uint32',shape=(n,252));a[:]=0;f=pq.ParquetFile(p/'network_access.parquet')
 for b in f.iter_batches(batch_size=1000000):
  d=b.to_pandas();assert d.n_specialist_contacts.max()<2**32;a[d.g,d.j]=d.n_specialist_contacts.astype('uint32')
 a.flush();checks=[]
 for tag,z in COHORTS.items():
  m=pd.read_parquet(H/'02_dyad_full_population'/tag/'global_core_index.parquet');m=m[m.hash_rank<=30000].sort_values('team_id');old=np.load(EXT/'output'/z['old_dir']/'network_access.npz')['count'];err=float(np.abs(a[m.g].astype(float)-old).max());assert err==0,(tag,err);checks.append({'cohort':tag,'all_30k_prior_contact_counts_match':True,'max_error':err});log('NETWORK 30K MATCH',tag)
 write(p/'original_network_validation.json',checks);(p/'NETWORK_MATRIX_COMPLETE').write_text('Exact original distinct-contact construct verified on all original cores.');log('NETWORK MATRIX COMPLETE')
if __name__=='__main__':main()
