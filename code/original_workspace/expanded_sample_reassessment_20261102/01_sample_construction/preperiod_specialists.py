"""Strict prior-year additional expertise for genuine2019--21 first-month works."""
from pathlib import Path
import sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import *
import duckdb
u=H/'02_dyad_full_population/universal';dest=u/'preperiod_specialists.parquet'
if not dest.exists():
 c=duckdb.connect();c.execute('SET threads=16');c.execute("SET memory_limit='80GB'");c.execute('PRAGMA disable_progress_bar');log('PREPERIOD SPECIALISTS BUILD')
 sql=f"WITH p AS (SELECT w.g,w.work_id,w.y,w.sf,w.ids,b.ids core_ids FROM read_parquet('{q(u/'joint_works.parquet')}') w JOIN read_parquet('{q(u/'work_first.parquet')}') f USING(g,j) JOIN read_parquet('{q(u/'cores.parquet')}') b USING(g) WHERE w.y BETWEEN 2019 AND 2021 AND date_diff('month',DATE '1960-01-01',w.dt)=f.first_mo), a AS (SELECT p.g,p.work_id,p.y,p.sf,p.core_ids,UNNEST(p.ids) aid FROM p) SELECT p.g,p.work_id,count_if(coalesce(a.first_y<p.y,false) AND NOT list_contains(p.core_ids,p.aid))>0 specialist FROM a p LEFT JOIN read_parquet('{q(H/'01_sample_construction/author_field.parquet')}') a ON a.author_id=p.aid AND a.sf=p.sf GROUP BY 1,2"
 c.execute(f"COPY ({sql}) TO '{q(dest)}' (FORMAT PARQUET,COMPRESSION ZSTD)");c.close();log('PREPERIOD SPECIALISTS COMPLETE')
