"""Close the analysis with output inventories and unchanged-source checks.

Small analytical artifacts are SHA256 hashed. Large binary caches are listed
with size and modification time to avoid another multi-GB data scan; numerical
construction checks remain in their accompanying diagnostics.
"""
from common import *
import importlib.metadata as metadata

def main():
 memo=F/'Exploratory_Validation_Temporal_and_Productivity.md'
 assert memo.exists()
 val=load(F/'diagnostics/numerical_validation.json')
 assert val['all_PPML_attempts_finished']
 assert load(F/'interpretation.json')['status']=='final_assessment'
 for mp in ['parent','direct']:
  for stage in ['replication','gradient','trajectory_marker','proportional','home_proportional']:
   assert(F/'results'/mp/stage/'COMPLETE').exists(),(mp,stage)
 original=load(F/'diagnostics/original_source_hashes.json')
 for x in original:
  assert hashlib.file_digest(Path(x['path']).open('rb'),'sha256').hexdigest()==x['sha256'],x['path']
 rows=[]
 for p in sorted(F.rglob('*')):
  if not p.is_file()or p.name in ['MANIFEST.json','MANIFEST.csv']or '__pycache__'in p.parts:continue
  ss=p.stat();small=ss.st_size<25_000_000
  rows.append({'path':p.relative_to(F).as_posix(),'bytes':ss.st_size,'modified_ns':ss.st_mtime_ns,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()if small else None,'fingerprint':'sha256'if small else'size_mtime_plus_construction_diagnostics'})
 pd.DataFrame(rows).to_csv(F/'MANIFEST.csv',index=False)
 versions={k:metadata.version(k)for k in ['numpy','pandas','scipy','duckdb','matplotlib','pyarrow','psutil']}
 save(F/'MANIFEST.json',{'created_local':time.strftime('%Y-%m-%d %H:%M:%S'),'module':str(F),'python':sys.version,'package_versions':versions,'Stata':'17 MP; dependency source hashes archived; license information excluded','new_working_paper_produced':False,'all_declared_models_attempted':True,'analyses_selected_by_significance':False,'original_sources_unchanged':True,'files':len(rows),'bytes':sum(x['bytes']for x in rows),'large_cache_hashing_policy':'Files >=25 MB listed by size/mtime; no redundant full OpenAlex scan','numerical_validation':val,'registry':'ANALYSIS_REGISTRY.md','primary_memo':memo.name,'inventory':'MANIFEST.csv'})
 log('FINAL MANIFEST COMPLETE',len(rows),'files')

if __name__=='__main__':main()
