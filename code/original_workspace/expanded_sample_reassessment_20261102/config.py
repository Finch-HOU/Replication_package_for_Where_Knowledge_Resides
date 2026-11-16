from pathlib import Path
import os,json,hashlib,time
H=Path(__file__).resolve().parent;ROOT=H.parent
OLD=ROOT/'innovation_choice_20261101';EXT=ROOT/'expertise_mobilization_20261102';PAPER=ROOT/'organized_empirical_working_paper'
BASE=Path('D:/OpenAlex/OpenAlex_Analysis/GPT Shock/output/derived/fixed_team');RAW=Path('D:/OpenAlex/parquet/works')
PARTS=['01_sample_construction','02_dyad_full_population','03_multimember_cores','04_reestimated_tables','05_old_vs_new_comparisons','06_diagnostics_and_inference','07_story_reassessment','08_reproducibility']
for p in PARTS:(H/p).mkdir(parents=True,exist_ok=True)
INPUTS=H/'08_reproducibility/inputs';INPUTS.mkdir(exist_ok=True)
MAPPING=INPUTS/'subfield_mapping_versions.csv';CAPABILITY=INPUTS/'field_month_proxies.csv';REQUIREMENTS=INPUTS/'reference_weights_2010_2014_temporal.npy';DOMAIN_INDEX=INPUTS/'domain_knowledge_input.csv'
NESTED=[30000,50000,100000,200000,400000,'full']
COHORTS={'older':{'old_dir':'stable_core','years':[2015,2018],'suffix':'stablecore20261101','expected_population':614119},'recent':{'old_dir':'stable_core_recent2021','years':[2018,2021],'suffix':'stablecore_recent20261101','expected_population':612105}}
def q(p):return str(p).replace('\\','/').replace("'","''")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,default=str),encoding='utf-8');os.replace(t,p)
def log(*x):print(time.strftime('%Y-%m-%d %H:%M:%S'),*x,flush=True)
