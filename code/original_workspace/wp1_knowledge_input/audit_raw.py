from pathlib import Path
import duckdb,pandas as pd,json,time
H=Path(__file__).resolve().parent;O=H/'output';P=O/'raw_audit_parts';P.mkdir(exist_ok=True)
files=sorted(Path('D:/OpenAlex/parquet/works').glob('updated_date=*/part_*.parquet'))
c=duckdb.connect();c.execute('SET threads=16');c.execute("SET memory_limit='40GB'");c.execute(f"SET temp_directory='{(O/'duckdb_temp').resolve().as_posix()}'")
lookup='D:/OpenAlex/OpenAlex_Analysis/GPT Shock/output/derived/fixed_team/knowledge_v2/work_domain_lookup.parquet'
print('LOAD LOOKUP',flush=True)
c.execute("CREATE TEMP TABLE lookup AS SELECT try_cast(replace(work_id,'https://openalex.org/W','') AS UBIGINT) work_n,subfield_id,publication_year FROM read_parquet(?) WHERE subfield_id IS NOT NULL",[lookup])
assert c.execute('SELECT count(*) FROM lookup WHERE work_n IS NULL').fetchone()[0]==0
print('LOOKUP READY',flush=True)
covered=set()
for record in P.glob('*.json'):
 covered.update(z['path'] for z in json.loads(record.read_text())['files'])
for start in range(0,len(files),16):
 key=f'{start:04d}';done=P/f'{key}.json'
 if all(str(p) in covered for p in files[start:start+16]):continue
 t=time.time();batch=[str(p) for p in files[start:start+16]]
 c.execute('CREATE OR REPLACE TEMP TABLE citing AS SELECT id, publication_year, CAST(primary_topic.subfield.id AS VARCHAR) src_subfield, referenced_works FROM read_parquet(?) WHERE publication_year BETWEEN 2010 AND 2021 AND primary_topic.subfield.id IS NOT NULL',[batch])
 c.execute(f"COPY (SELECT src_subfield,CASE WHEN publication_year<=2014 THEN '2010-2014' ELSE '2015-2021' END period, count(*) citing_works,count(*) FILTER (WHERE referenced_works IS NOT NULL) works_with_reference_list, sum(coalesce(array_length(referenced_works),0)) total_references FROM citing GROUP BY 1,2) TO '{(P/(key+'_den.parquet')).resolve().as_posix()}' (FORMAT PARQUET)")
 c.execute(f"COPY (SELECT c.src_subfield,l.subfield_id dst_subfield,count(*) n_citations,count(*) FILTER (WHERE l.publication_year<=c.publication_year) n_temporally_valid FROM citing c CROSS JOIN UNNEST(c.referenced_works) u(ref) JOIN lookup l ON try_cast(replace(u.ref,'https://openalex.org/W','') AS UBIGINT)=l.work_n WHERE c.publication_year<=2014 GROUP BY 1,2) TO '{(P/(key+'_flow.parquet')).resolve().as_posix()}' (FORMAT PARQUET)")
 done.write_text(json.dumps({'files':[dict(path=str(p),bytes=p.stat().st_size,mtime=p.stat().st_mtime) for p in files[start:start+16]],'seconds':time.time()-t}),encoding='utf-8')
 print('BATCH',start+len(batch),'/',len(files),'seconds',round(time.time()-t,1),flush=True)
for suffix in ['den','flow']:
 q=("SELECT src_subfield,period,sum(citing_works) citing_works,sum(works_with_reference_list) works_with_reference_list,sum(total_references) total_references FROM read_parquet(?) GROUP BY 1,2" if suffix=='den' else "SELECT src_subfield,dst_subfield,sum(n_citations) n_citations,sum(n_temporally_valid) n_temporally_valid FROM read_parquet(?) GROUP BY 1,2")
 c.execute(q,[str(P/f'*_{suffix}.parquet')]).fetchdf().to_parquet(O/f'raw_citation_{suffix}_audit.parquet',index=False)
print('RAW AUDIT COMPLETE',flush=True)
