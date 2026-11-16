"""Global author expertise and exact-roster core identities from one compact cache."""
from pathlib import Path
import sys,json,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import *
import duckdb,pyarrow.parquet as pq

def main():
 assert (H/'01_sample_construction/CATALOG_COMPLETE').exists()
 folder=H/'01_sample_construction';c=duckdb.connect(str(folder/'global_history.duckdb'));c.execute('SET threads=64');c.execute("SET memory_limit='300GB'");c.execute('PRAGMA disable_progress_bar');c.execute(f"SET temp_directory='{q(folder/'spill')}'")
 def stage(name,sql):
  if c.execute('SELECT count(*) FROM information_schema.tables WHERE table_name=?',[name]).fetchone()[0]:return
  log('GLOBAL HISTORY BUILD',name);t=time.time();c.execute('CREATE TABLE '+name+' AS '+sql);log('GLOBAL HISTORY DONE',name,c.execute('SELECT count(*) FROM '+name).fetchone()[0],round(time.time()-t,1))
 def export(name,p):
  if not p.exists():c.execute(f"COPY {name} TO '{q(p)}' (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 262144)")
 glob=q(folder/'global_catalog/*.parquet')
 # All valid classified work types, exactly as the old raw-history extraction.
 stage('author_field',f"SELECT aid author_id,sf,min(y)::INTEGER first_y,count(*) total_n,count_if(y<=2021)::INTEGER n_pre2021,count_if(y<=2018)::INTEGER n_pre2018 FROM read_parquet('{glob}') CROSS JOIN UNNEST(ids) t(aid) WHERE sf IS NOT NULL AND y<=2026 GROUP BY 1,2")
 export('author_field',folder/'author_field.parquet');stage('author_any','SELECT author_id,min(first_y) first_any_y,sum(n_pre2021) n_pre2021 FROM author_field GROUP BY 1');export('author_any',folder/'author_any.parquet')
 # A representative work's full numeric author list recovers the original hash
 # after restoring canonical OpenAlex URLs and lexical string order.
 for tag,z in COHORTS.items():
  lo,hi=z['years'];name='exact_rosters_'+tag
  stage(name,f"SELECT md5(array_to_string(list_sort(list_transform(ids,x->'https://openalex.org/A'||x::VARCHAR)),'|')) team_id,ids,len(ids)::INTEGER core_size,min(work_id) representative_numeric_work,count(*)::INTEGER n_pre,count_if(work_type='article')::INTEGER n_pre_article FROM read_parquet('{glob}') WHERE y BETWEEN {lo} AND {hi} AND sf IN (SELECT try_cast(regexp_extract(subfield_id,'([0-9]+)$',1) AS INTEGER) FROM read_parquet('{q(OLD/'output/sub.parquet')}')) AND work_type IN ('article','review','conference-paper','preprint') AND len(ids)>=2 GROUP BY 1,2,3 HAVING count(*)>=2")
  out=H/'03_multimember_cores'/'methodA_exact_all'/tag;out.mkdir(parents=True,exist_ok=True)
  if not (out/'membership.parquet').exists():
   suffix=z['suffix']+'_multimember';c.execute(f"COPY (SELECT *,hash(team_id||'{suffix}') hash_value,row_number() OVER(ORDER BY hash(team_id||'{suffix}'))::INTEGER hash_rank,(row_number() OVER(ORDER BY team_id)-1)::INTEGER i FROM {name} WHERE core_size>=3) TO '{q(out/'membership.parquet')}' (FORMAT PARQUET,COMPRESSION ZSTD)")
  orig=H/'02_dyad_full_population'/tag/'membership.parquet';n=c.execute(f'SELECT count(*) FROM {name} WHERE core_size=2').fetchone()[0];assert n==z['expected_population'];missing=c.execute(f"SELECT count(*) FROM read_parquet('{q(orig)}') a ANTI JOIN {name} b USING(team_id)").fetchone()[0];assert missing==0
  counts=c.execute(f'SELECT core_size,count(*) cores,sum(n_pre) baseline_records FROM {name} GROUP BY 1 ORDER BY 1').fetchdf();counts.to_csv(out/'size_distribution.csv',index=False);write(out/'population.json',{'cohort':tag,'algorithm':'A: repeated exact complete author sets, no size-5 cap','eligible_multimember':int(counts.loc[counts.core_size>=3,'cores'].sum()),'size_max':int(counts.core_size.max()),'eligibility_years':z['years'],'raw_dyad_population_matches_original':True,'raw_work_records_not_projects':True});log('RAW ROSTER POPULATION',tag,counts.to_dict('records'))
  # Add original dyad identities without changing their index or nested rank.
  dest=H/'02_dyad_full_population'/tag/'cores.parquet'
  if not dest.exists():c.execute(f"COPY (SELECT a.*,b.ids FROM read_parquet('{q(orig)}') a JOIN {name} b USING(team_id)) TO '{q(dest)}' (FORMAT PARQUET,COMPRESSION ZSTD)")
 c.close();(folder/'HISTORIES_AND_ROSTERS_COMPLETE').write_text('Global histories and all-size exact recurring sets built; dyad frames independently match original counts and membership.');log('GLOBAL HISTORY AND ROSTERS COMPLETE')
if __name__=='__main__':main()
