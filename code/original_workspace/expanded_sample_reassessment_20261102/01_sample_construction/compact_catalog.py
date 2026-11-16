"""One resumable numeric bibliometric scan shared by all population definitions."""
from pathlib import Path
import sys,os,time,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import *
import duckdb,pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor,as_completed
OUT=H/'01_sample_construction/global_catalog';OUT.mkdir(exist_ok=True)

def worker(src):
 src=Path(src);name=src.parent.name+'__'+src.stem;p=OUT/(name+'.parquet');a=OUT/(name+'.json')
 if p.exists() and a.exists():pq.read_metadata(p);return json.loads(a.read_text())
 start=time.time();cols=['id','publication_year','publication_date','type','primary_topic.subfield.id','is_retracted','is_paratext','authorships.list.element.author.id','referenced_works','title','doi'];ar=pq.ParquetFile(src).read(columns=cols,use_threads=False);c=duckdb.connect();c.execute('SET threads=2');c.execute("SET memory_limit='12GB'");c.execute('PRAGMA disable_progress_bar');c.register('raw',ar)
 c.execute("CREATE TEMP TABLE cat AS SELECT try_cast(regexp_extract(id,'W([0-9]+)$',1) AS UBIGINT) work_id,publication_year::INTEGER y,try_cast(publication_date AS DATE) dt,type work_type,try_cast(regexp_extract(primary_topic.subfield.id,'([0-9]+)$',1) AS INTEGER) sf,list_sort(list_distinct(list_transform(authorships, a->try_cast(regexp_extract(a.author.id,'A([0-9]+)$',1) AS UBIGINT)))) ids,list_transform(referenced_works,x->try_cast(regexp_extract(x,'W([0-9]+)$',1) AS UBIGINT)) refs,title,lower(doi) doi,len(list_distinct(list_transform(authorships,a->a.author.id))) source_authors FROM raw WHERE NOT coalesce(is_retracted,false) AND NOT coalesce(is_paratext,false)")
 bad=c.execute('SELECT coalesce(count_if(work_id IS NULL),0),coalesce(count_if(len(ids)<>source_authors),0) FROM cat').fetchone();assert bad==(0,0),(src,bad)
 meta={'source':str(src),'source_bytes':src.stat().st_size,'source_mtime_ns':src.stat().st_mtime_ns,'rows':c.execute('SELECT count(*) FROM cat').fetchone()[0],'numeric_id_failures':bad,'seconds':time.time()-start,'all_available_work_types_preserved':True,'filter':'Only retracted/paratext excluded; missing subfields/years retained for measurement diagnostics'};tmp=p.with_suffix('.tmp');c.execute(f"COPY (SELECT * EXCLUDE(source_authors) FROM cat) TO '{q(tmp)}' (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 131072)");os.replace(tmp,p);write(a,meta);c.close();return meta

def main():
 # Expansion is gated on actual-data reproduction, not just source hashes.
 for co in ['stable_core','stable_core_recent2021']:
  for ma in ['parent','direct']:
   check=json.loads((H/'08_reproducibility'/f'benchmark_{co}_{ma}.json').read_text());assert check and all(v['passed'] for v in check)
  check=json.loads((H/'08_reproducibility'/f'H1_benchmark_{co}.json').read_text());assert all(v['passed'] for v in check)
 assert all(v['exact_original_membership'] for v in json.loads((H/'08_reproducibility/original_membership_audit.json').read_text()))
 files=sorted(RAW.glob('updated_date=*/part_*.parquet'));rows=[];start=time.time();log('SHARED RAW CATALOG START',len(files),'files; 24 workers')
 with ProcessPoolExecutor(max_workers=int(os.environ.get('EXP_SCAN_WORKERS','24'))) as ex:
  jobs={ex.submit(worker,p):p for p in files}
  try:
   for n,f in enumerate(as_completed(jobs),1):
    rows.append(f.result())
    if n%25==0 or n==len(files):log('GLOBAL CATALOG',n,'/',len(files),'rows',sum(r['rows'] for r in rows),'seconds',round(time.time()-start,1))
  except BaseException:
   for f in jobs:f.cancel()
   for proc in list(ex._processes.values()):proc.terminate()
   raise
 write(H/'01_sample_construction/global_catalog_manifest.json',{'files':rows,'total_rows':sum(r['rows'] for r in rows),'seconds':time.time()-start,'source_projects_modified':False,'zero_risks':'Catalog is shared source infrastructure; regression risk sets are constructed separately and include all candidate zeros.'});(H/'01_sample_construction/CATALOG_COMPLETE').write_text('All original raw partitions processed without failures.');log('GLOBAL CATALOG COMPLETE')
if __name__=='__main__':main()
