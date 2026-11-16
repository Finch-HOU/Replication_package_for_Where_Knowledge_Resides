from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import *
import duckdb,pandas as pd
def main():
 c=duckdb.connect();c.execute('SET threads=32');c.execute("SET memory_limit='150GB'");c.execute('PRAGMA disable_progress_bar');c.execute(f"ATTACH '{q(OLD/'output/research.duckdb')}' AS foundation (READ_ONLY)");rows=[]
 for tag,z in COHORTS.items():
  if tag=='older':sql='SELECT * FROM foundation.cohort WHERE team_size=2 AND n_pre>=2'
  else:sql=f"SELECT p.team_id,min(p.team_size) team_size,min(p.work_id) representative_work,count(*) n_pre,count_if(l.work_type='article') n_pre_article FROM read_parquet('{q(BASE/'paper_team_spine/*.parquet')}') p JOIN foundation.lookup l USING(work_id) JOIN foundation.sub s ON s.subfield_id=l.sf WHERE p.team_size=2 AND l.y BETWEEN 2018 AND 2021 AND l.work_type IN ('article','review','conference-paper','preprint') GROUP BY p.team_id HAVING count(*)>=2"
  c.execute('CREATE OR REPLACE TEMP TABLE elig AS '+sql);n=c.execute('SELECT count(*) FROM elig').fetchone()[0];assert n==z['expected_population'];sample=c.execute(f"SELECT *,hash(team_id||'{z['suffix']}') hash_value,row_number() OVER(ORDER BY hash(team_id||'{z['suffix']}')) hash_rank FROM elig ORDER BY hash_rank LIMIT 30000").fetchdf();old=pd.read_parquet(EXT/'output'/z['old_dir']/'sample.parquet');assert set(sample.team_id)==set(old.team_id);sample=sample.merge(old[['team_id','i']],on='team_id',validate='one_to_one');sample.to_parquet(H/'01_sample_construction'/f'{tag}_original_30000_membership_rank.parquet',index=False);row={'cohort':tag,'population':n,'hash_suffix':z['suffix'],'exact_original_membership':True,'original_index_matches_sorted_team_id':bool((sample.sort_values('team_id').i.to_numpy()==list(range(30000))).all()),'uses_future_outcomes':False};rows.append(row);log('MEMBERSHIP VERIFIED',row)
 write(H/'08_reproducibility/original_membership_audit.json',rows);c.close()
if __name__=='__main__':main()
