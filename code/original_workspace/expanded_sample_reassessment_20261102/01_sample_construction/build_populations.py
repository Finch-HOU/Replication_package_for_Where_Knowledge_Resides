"""Full exact-roster eligibility and nested original hash ranks; no future filter."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import *
import duckdb,pandas as pd

def main():
 c=duckdb.connect(str(H/'01_sample_construction/populations.duckdb'));c.execute('SET threads=48');c.execute("SET memory_limit='200GB'");c.execute('PRAGMA disable_progress_bar');c.execute(f"ATTACH '{q(OLD/'output/research.duckdb')}' AS foundation (READ_ONLY)")
 for tag,z in COHORTS.items():
  if not (H/'08_reproducibility'/f'H1_benchmark_{z['old_dir']}.json').exists():raise RuntimeError('Benchmark gate incomplete')
  table='eligible_'+tag
  if not c.execute('SELECT count(*) FROM information_schema.tables WHERE table_name=?',[table]).fetchone()[0]:
   lo,hi=z['years'];sql=f"SELECT p.team_id,min(p.team_size) core_size,min(p.work_id) representative_work,count(*) n_pre,count_if(l.work_type='article') n_pre_article FROM read_parquet('{q(BASE/'paper_team_spine/*.parquet')}') p JOIN foundation.lookup l USING(work_id) JOIN foundation.sub s ON s.subfield_id=l.sf WHERE p.team_size>=2 AND l.y BETWEEN {lo} AND {hi} AND l.work_type IN ('article','review','conference-paper','preprint') GROUP BY p.team_id HAVING count(*)>=2";log('BUILD ELIGIBILITY',tag);c.execute('CREATE TABLE '+table+' AS '+sql)
  for kind,condition,outfolder in [('dyad','core_size=2',H/'02_dyad_full_population'/tag),('exact_multimember','core_size>=3',H/'03_multimember_cores'/tag)]:
   outfolder.mkdir(exist_ok=True);out=outfolder/'membership.parquet'
   if out.exists():continue
   n=c.execute(f'SELECT count(*) FROM {table} WHERE {condition}').fetchone()[0]
   if kind=='dyad':assert n==z['expected_population']
   suffix=z['suffix'] if kind=='dyad' else z['suffix']+'_multimember';sql=f"SELECT *,hash(team_id||'{suffix}') hash_value,row_number() OVER(ORDER BY hash(team_id||'{suffix}'))::INTEGER hash_rank,(row_number() OVER(ORDER BY team_id)-1)::INTEGER i FROM {table} WHERE {condition}";c.execute(f"COPY ({sql}) TO '{q(out)}' (FORMAT PARQUET,COMPRESSION ZSTD)")
   if kind=='dyad':
    got=c.execute(f"SELECT team_id FROM read_parquet('{q(out)}') WHERE hash_rank<=30000").fetchdf();old=pd.read_parquet(EXT/'output'/z['old_dir']/'sample.parquet');assert set(got.team_id)==set(old.team_id)
   counts=c.execute(f'SELECT core_size,count(*) cores,sum(n_pre) baseline_records FROM {table} WHERE {condition} GROUP BY 1 ORDER BY 1').fetchdf();counts.to_csv(outfolder/'eligibility_by_core_size.csv',index=False);write(outfolder/'population.json',{'cohort':tag,'definition':kind,'eligibility_years':z['years'],'population':n,'hash_suffix':suffix,'freeze':2021,'nested_sizes':NESTED if kind=='dyad' else ['full'],'future_survival_required':False,'primary_records_not_version_deduplicated':True});log('FULL POPULATION',tag,kind,n)
 c.close()
if __name__=='__main__':main()
