"""Current-year specialist identities and strictly historical sourcing routes.

Small-paper ties are reused, but unrestricted prior collaboration also checks
larger papers. First observed coauthorship is not project formation or hiring.
"""
from pathlib import Path
import sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import *
import duckdb

def main():
 u=H/'02_dyad_full_population/universal';o=u/'sourcing';o.mkdir(exist_ok=True)
 c=duckdb.connect(str(o/'sourcing.duckdb'));c.execute('SET threads=32');c.execute("SET memory_limit='140GB'");c.execute('PRAGMA disable_progress_bar')
 def stage(name,sql):
  if c.execute('SELECT count(*) FROM information_schema.tables WHERE table_name=?',[name]).fetchone()[0]:return
  log('SOURCING BUILD',name);t=time.time();c.execute('CREATE TABLE '+name+' AS '+sql);log('SOURCING DONE',name,c.execute('SELECT count(*) FROM '+name).fetchone()[0],round(time.time()-t,1))
 def export(name):
  p=o/(name+'.parquet')
  if not p.exists():c.execute(f"COPY {name} TO '{q(p)}' (FORMAT PARQUET,COMPRESSION ZSTD)")
 af=q(H/'01_sample_construction/author_field.parquet');aa=q(H/'01_sample_construction/author_any.parquet');sub=q(u/'frozen/sub.parquet');glob=q(H/'01_sample_construction/global_catalog/*.parquet')
 stage('window_extra',f"WITH w AS (SELECT w.g,w.work_id,w.j,w.y,w.dt,w.mo,UNNEST(list_filter(j.ids,x->NOT list_contains(b.ids,x))) aid FROM read_parquet('{q(u/'analysis_works.parquet')}') w JOIN read_parquet('{q(u/'joint_works.parquet')}') j USING(g,work_id) JOIN read_parquet('{q(u/'cores.parquet')}') b USING(g) WHERE w.mo BETWEEN 786 AND 796), a AS (SELECT f.*,s.j FROM read_parquet('{af}') f JOIN read_parquet('{sub}') s ON f.sf=try_cast(regexp_extract(s.subfield_id,'([0-9]+)$',1) AS INTEGER)) SELECT w.*,coalesce(a.first_y<w.y,false) is_spec,coalesce(a.n_pre2021>0,false) frozen_spec,coalesce(h.first_any_y<w.y,false) has_prior_record,coalesce(h.n_pre2021>0,false) frozen_any FROM w LEFT JOIN a ON a.author_id=w.aid AND a.j=w.j LEFT JOIN read_parquet('{aa}') h ON h.author_id=w.aid")
 export('window_extra')
 stage('focal_pairs',f"SELECT DISTINCT m.author_id core_author,w.aid collaborator FROM window_extra w JOIN read_parquet('{q(u/'members.parquet')}') m USING(g) WHERE w.is_spec")
 stage('small_history',f"SELECT p.*,s.first_small_y,s.first_small_dt FROM focal_pairs p LEFT JOIN read_parquet('{q(u/'network/small_ties.parquet')}') s USING(core_author,collaborator)")
 # Anchor focal pair at a baseline core member: only larger-paper histories are
 # scanned here, since <=20 histories have already been extracted exactly.
 stage('large_history',f"WITH a AS (SELECT DISTINCT core_author FROM focal_pairs), w AS (SELECT work_id,y,dt,ids,UNNEST(ids) core_author FROM read_parquet('{glob}') WHERE sf IS NOT NULL AND y<=2026 AND len(ids)>20), selected AS (SELECT w.* FROM w SEMI JOIN a USING(core_author)) SELECT p.core_author,p.collaborator,min(w.y) first_large_y,min(w.dt) first_large_dt FROM selected w JOIN focal_pairs p ON p.core_author=w.core_author AND list_contains(w.ids,p.collaborator) GROUP BY 1,2")
 stage('pair_history',"SELECT s.*,least(s.first_small_y,l.first_large_y) first_y,least(s.first_small_dt,l.first_large_dt) first_dt FROM small_history s LEFT JOIN large_history l USING(core_author,collaborator)")
 stage('expert_routes',f"SELECT w.g,w.work_id,w.aid,max(coalesce(h.first_y<w.y,false)) old_year,max(coalesce(h.first_dt<w.dt,false)) old_date,max(coalesce(h.first_small_y<w.y,false)) old_small FROM window_extra w JOIN read_parquet('{q(u/'members.parquet')}') m USING(g) JOIN pair_history h ON h.core_author=m.author_id AND h.collaborator=w.aid WHERE w.is_spec GROUP BY 1,2,3")
 stage('specialist_sources',"SELECT g,work_id,count_if(old_year)::INTEGER n_existing,count_if(NOT old_year)::INTEGER n_new,count_if(old_date)::INTEGER n_existing_date,count_if(NOT old_date)::INTEGER n_new_date,count_if(old_small)::INTEGER n_existing_small,count_if(NOT old_small)::INTEGER n_new_small FROM expert_routes GROUP BY 1,2");export('specialist_sources')
 stage('frozen_extra_counts',"SELECT g,work_id,count_if(frozen_spec)::INTEGER extra_specialists,count_if(frozen_any)::INTEGER extra_prior_records,count(*)::INTEGER additional_authors,count_if(is_spec)::INTEGER current_extra_specialists FROM window_extra GROUP BY 1,2");export('frozen_extra_counts')
 write(o/'audit.json',{'window_extra_records':c.execute('SELECT count(*) FROM window_extra').fetchone()[0],'specialist_identity_records':c.execute('SELECT count(*) FROM expert_routes').fetchone()[0],'focal_pair_count':c.execute('SELECT count(*) FROM focal_pairs').fetchone()[0],'unrestricted_large_paper_history_checked':True,'strict_prior_year_primary':True,'first_observed_not_collaboration_start':True});c.close();(o/'SOURCING_COMPLETE').write_text('All routes and frozen extra-author counts complete.');log('SOURCING COMPLETE')
if __name__=='__main__':main()
