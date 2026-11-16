"""Exact pre-2022 small-team contact unions and independent target expertise."""
from pathlib import Path
import sys,time,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import *
import duckdb,pyarrow.parquet as pq

def main():
 u=H/'02_dyad_full_population/universal';out=u/'network';out.mkdir(exist_ok=True)
 for name in ['members.parquet','core_authors.parquet']:pq.read_metadata(u/name)
 c=duckdb.connect(str(out/'network.duckdb'));c.execute('SET threads=48');c.execute("SET memory_limit='250GB'");c.execute('PRAGMA disable_progress_bar');c.execute(f"SET temp_directory='{q(out/'spill')}'")
 def stage(name,sql):
  if c.execute('SELECT count(*) FROM information_schema.tables WHERE table_name=?',[name]).fetchone()[0]:return
  t=time.time();log('BASELINE NETWORK BUILD',name);c.execute('CREATE TABLE '+name+' AS '+sql);log('BASELINE NETWORK DONE',name,c.execute('SELECT count(*) FROM '+name).fetchone()[0],round(time.time()-t,1))
 def export(name):
  p=out/(name+'.parquet')
  if not p.exists():c.execute(f"COPY {name} TO '{q(p)}' (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 262144)")
 glob=q(H/'01_sample_construction/global_catalog/*.parquet');hist=q(H/'01_sample_construction/author_field.parquet');members=q(u/'members.parquet');authors=q(u/'core_authors.parquet')
 stage('small_ties',f"WITH p AS (SELECT y,dt,ids,UNNEST(ids) core_author FROM read_parquet('{glob}') WHERE sf IS NOT NULL AND y<=2026 AND len(ids)<=20), selected AS (SELECT * FROM p SEMI JOIN read_parquet('{authors}') a ON a.author_id=p.core_author), expanded AS (SELECT core_author,y,dt,UNNEST(ids) collaborator FROM selected) SELECT core_author,collaborator,min(y) first_small_y,min(dt) first_small_dt FROM expanded WHERE collaborator<>core_author GROUP BY 1,2");export('small_ties')
 # List membership is equivalent to the prior correlated anti-join, but avoids
 # DuckDB's serial delimiter-join bottleneck on billions of intermediate links.
 stage('core_contacts',f"SELECT DISTINCT m.g,t.collaborator FROM small_ties t JOIN read_parquet('{members}') m ON m.author_id=t.core_author JOIN read_parquet('{q(u/'cores.parquet')}') b ON b.g=m.g WHERE t.first_small_y<=2021 AND NOT list_contains(b.ids,t.collaborator)");export('core_contacts')
 # Full join/group remains documented above in history but the preferred exact
 # implementation avoids enormous grouped state and stores the dense count
 # matrix directly. Completed contact checkpoints remain reusable.
 c.close()
 import subprocess
 subprocess.run([sys.executable,'-X','utf8',str(H/'01_sample_construction/network_dense_counts.py')],check=True,cwd=H)
 return
 write(out/'audit.json',{'small_author_ties':c.execute('SELECT count(*) FROM small_ties').fetchone()[0],'core_contact_links':c.execute('SELECT count(*) FROM core_contacts').fetchone()[0],'access_cells':c.execute('SELECT count(*) FROM network_access').fetchone()[0],'baseline_cutoff':2021,'team_size_limit_for_contact':20,'independent_contact_expertise':True,'baseline_core_members_excluded':True,'all_work_types_in_histories':True});c.close();(out/'NETWORK_COMPLETE').write_text('Exact distinct-contact counts, not summed member-level approximations.');log('BASELINE NETWORK COMPLETE')
if __name__=='__main__':main()
