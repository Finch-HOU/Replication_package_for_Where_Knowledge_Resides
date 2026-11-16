"""Shared complete-roster joint output, baseline histories and current expertise."""
from pathlib import Path
import sys,time,json,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import *
import duckdb,pandas as pd, numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--method',choices=['A','B'],default='A');args=ap.parse_args();method=args.method
 assert (H/'01_sample_construction/HISTORIES_AND_ROSTERS_COMPLETE').exists()
 out=H/('02_dyad_full_population/universal' if method=='A' else '03_multimember_cores/methodB_network/universal');out.mkdir(exist_ok=True);c=duckdb.connect(str(out/'core_data.duckdb'));c.execute('SET threads=48');c.execute("SET memory_limit='240GB'");c.execute('PRAGMA disable_progress_bar');c.execute(f"SET temp_directory='{q(out/'spill')}'")
 def stage(name,sql):
  if c.execute('SELECT count(*) FROM information_schema.tables WHERE table_name=?',[name]).fetchone()[0]:return
  t=time.time();log('CORE DATA BUILD',name);c.execute('CREATE TABLE '+name+' AS '+sql);log('CORE DATA DONE',name,c.execute('SELECT count(*) FROM '+name).fetchone()[0],round(time.time()-t,1))
 def export(name,namefile=None):
  p=out/((namefile or name)+'.parquet')
  if not p.exists():c.execute(f"COPY {name} TO '{q(p)}' (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 262144)")
 roots=[]
 for tag,z in COHORTS.items():roots += [H/'02_dyad_full_population'/tag/'cores.parquet',H/'03_multimember_cores/methodA_exact_all'/tag/'membership.parquet'] if method=='A' else [H/'03_multimember_cores/methodB_network'/tag/'membership.parquet']
 union=' UNION ALL '.join(f"SELECT team_id,ids,core_size FROM read_parquet('{q(p)}')" for p in roots)
 stage('cores','SELECT *, (row_number() OVER(ORDER BY team_id)-1)::INTEGER g FROM (SELECT DISTINCT team_id,ids,core_size FROM ('+union+'))');export('cores');stage('members','SELECT g,team_id,core_size,UNNEST(ids) author_id FROM cores');export('members');stage('core_authors','SELECT DISTINCT author_id FROM members');export('core_authors')
 hist=H/'01_sample_construction/author_field.parquet';anyhist=H/'01_sample_construction/author_any.parquet'
 stage('anchor',f"SELECT m.g,arg_min(m.author_id,struct_pack(n:=a.n_pre2021,id:=m.author_id)) anchor FROM members m JOIN read_parquet('{q(anyhist)}') a USING(author_id) GROUP BY 1")
 for tag,z in COHORTS.items():
  for kind,p in ([('dyad',H/'02_dyad_full_population'/tag),('multimember',H/'03_multimember_cores/methodA_exact_all'/tag)] if method=='A' else [('multimember_network',H/'03_multimember_cores/methodB_network'/tag)]):
   dest=p/'global_core_index.parquet'
   if not dest.exists():c.execute(f"COPY (SELECT m.*,c.g FROM read_parquet('{q(p/'membership.parquet')}') m JOIN cores c USING(team_id)) TO '{q(dest)}' (FORMAT PARQUET,COMPRESSION ZSTD)")
 stage('sub',f"SELECT *,try_cast(regexp_extract(subfield_id,'([0-9]+)$',1) AS INTEGER) sf FROM read_parquet('{q(OLD/'output/sub.parquet')}')")
 stage('baseline_author_fields',f"SELECT a.* FROM read_parquet('{q(hist)}') a SEMI JOIN core_authors b USING(author_id) WHERE a.n_pre2021>0")
 stage('member_first','SELECT m.g,s.j,min(a.first_y) first_y,sum(a.n_pre2021) n_history,sum(a.n_pre2018) n_history2018 FROM members m JOIN baseline_author_fields a USING(author_id) JOIN sub s USING(sf) GROUP BY 1,2');export('member_first')
 glob=q(H/'01_sample_construction/global_catalog/*.parquet')
 stage('joint_works',f"WITH w AS (SELECT work_id,y,dt,work_type,sf,ids,len(ids)::INTEGER authors_n,UNNEST(ids) aid FROM read_parquet('{glob}') WHERE y<=2026 AND sf IS NOT NULL AND work_type IN ('article','review','conference-paper','preprint')) SELECT k.g,w.work_id,w.y,w.dt,w.work_type,w.sf,s.j,w.authors_n,w.ids FROM w JOIN anchor a ON a.anchor=w.aid JOIN cores k USING(g) JOIN sub s ON s.sf=w.sf WHERE w.authors_n>=k.core_size AND list_has_all(w.ids,k.ids)")
 export('joint_works');stage('joint_pre','SELECT g,j,count(*) n FROM joint_works WHERE y<=2021 GROUP BY 1,2');export('joint_pre');stage('work_first',"SELECT g,j,min(date_diff('month',DATE '1960-01-01',dt))::INTEGER first_mo FROM joint_works GROUP BY 1,2");export('work_first')
 # Only observed core outputs need author-by-destination classification; no
 # candidate rows are removed by limiting this computation to output records.
 stage('pub_authors','SELECT g,work_id,y,dt,sf,j,authors_n,UNNEST(ids) author_id FROM joint_works WHERE y>=2022')
 stage('current_experience',f"SELECT p.g,p.work_id,count(*) authors_n,count_if(coalesce(a.first_y<p.y,false)) current_experts,count_if(coalesce(a.first_y<p.y,false) AND m.author_id IS NOT NULL) current_core_experts,count_if(coalesce(anyf.first_any_y<p.y,false)) authors_with_prior_records FROM pub_authors p LEFT JOIN read_parquet('{q(hist)}') a ON a.author_id=p.author_id AND a.sf=p.sf LEFT JOIN members m ON m.g=p.g AND m.author_id=p.author_id LEFT JOIN read_parquet('{q(anyhist)}') anyf ON anyf.author_id=p.author_id GROUP BY 1,2");export('current_experience')
 stage('analysis_works',"SELECT w.g,w.work_id,w.y,w.dt,w.j,w.work_type,w.authors_n,month(w.dt)=1 AND day(w.dt)=1 jan1,f.first_mo,date_diff('month',DATE '1960-01-01',w.dt)::INTEGER mo,c.current_experts,c.current_core_experts,c.authors_with_prior_records FROM joint_works w JOIN work_first f USING(g,j) LEFT JOIN current_experience c USING(g,work_id)");export('analysis_works')
 stage('first_entry_types',"SELECT g,j,mo,count(*) n_papers,bool_or(authors_n>c.core_size) expanded,bool_or(current_experts>current_core_experts) specialist,bool_or(authors_with_prior_records<authors_n) AS \"unknown\",count_if(current_experts>current_core_experts) n_specialist FROM analysis_works w JOIN cores c USING(g) WHERE mo=first_mo AND mo BETWEEN 786 AND 796 GROUP BY 1,2,3");export('first_entry_types')
 audit={'cores':c.execute('SELECT count(*) FROM cores').fetchone()[0],'unique_authors':c.execute('SELECT count(*) FROM core_authors').fetchone()[0],'joint_work_occurrences':c.execute('SELECT count(*) FROM joint_works').fetchone()[0],'unique_joint_works':c.execute('SELECT count(DISTINCT work_id) FROM joint_works').fetchone()[0],'missing_joint_dates':c.execute('SELECT count(*) FROM joint_works WHERE dt IS NULL').fetchone()[0],'duplicate_core_work_records':c.execute('SELECT count(*)-count(DISTINCT (g,work_id)) FROM joint_works').fetchone()[0],'current_expert_errors':c.execute('SELECT count(*) FROM current_experience WHERE current_core_experts>current_experts OR current_experts>authors_n OR authors_with_prior_records>authors_n').fetchone()[0]};assert audit['duplicate_core_work_records']==0 and audit['current_expert_errors']==0;write(out/'audit.json',audit)
 # Raw-cache reconstruction must recover every old 30k core-work record exactly.
 checks=[]
 for tag,z in COHORTS.items():
  if method!='A':continue
  old=EXT/'output'/z['old_dir']/'works.parquet';mi=H/'02_dyad_full_population'/tag/'global_core_index.parquet';oldsm=EXT/'output'/z['old_dir']/'sample.parquet'
  sql=f"SELECT o.i,c.work_id,c.j,c.y,c.dt,c.work_type,c.authors_n FROM joint_works c JOIN read_parquet('{q(mi)}') m USING(g) JOIN read_parquet('{q(oldsm)}') o USING(team_id)";oldsql=f"SELECT i,try_cast(regexp_extract(work_id,'W([0-9]+)$',1) AS UBIGINT) work_id,j,y,try_cast(dt AS DATE) dt,work_type,authors_n FROM read_parquet('{q(old)}')";extra=c.execute(f'SELECT count(*) FROM (({sql}) EXCEPT ({oldsql}))').fetchone()[0];missing=c.execute(f'SELECT count(*) FROM (({oldsql}) EXCEPT ({sql}))').fetchone()[0];assert missing==0 and extra==0,(tag,missing,extra);checks.append({'cohort':tag,'missing':missing,'extra':extra,'raw_joint_output_matches_30k':True});log('RAW 30K JOINT OUTPUT VERIFIED',tag)
 write(out/'raw_30k_reconstruction_checks.json',checks);c.close();(out/'CORE_DATA_COMPLETE').write_text('Complete all-member outputs, frozen histories and strictly prior-year expertise.');log('CORE DATA COMPLETE',audit)
if __name__=='__main__':main()
