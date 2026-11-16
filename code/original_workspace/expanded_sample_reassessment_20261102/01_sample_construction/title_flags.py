from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import *
import duckdb,pandas as pd,numpy as np
RULES={'explicit_AI':r'\bAI\b|artificial intelligence|generative|large language model|\bLLMs?\b|ChatGPT|\bGPT[- ]?\d|Copilot|foundation model','broad_AI':r'\bAI\b|artificial intelligence|generative|large language model|\bLLMs?\b|ChatGPT|\bGPT[- ]?\d|Copilot|foundation model|machine[- ]learning|deep[- ]learning|transfer[- ]learning|neural network|convolutional|\bLSTM\b|\bChronos\b|\bXGBoost\b|\bResNet\b'}
u=H/'02_dyad_full_population/universal';p=u/'title_flags.parquet'
if not p.exists():
 c=duckdb.connect();c.execute('SET threads=16');c.execute('PRAGMA disable_progress_bar');w=c.execute(f"WITH wanted AS (SELECT DISTINCT work_id FROM read_parquet('{q(u/'analysis_works.parquet')}') WHERE mo BETWEEN 786 AND 796 AND mo=first_mo) SELECT w.work_id,w.title FROM read_parquet('{q(H/'01_sample_construction/global_catalog/*.parquet')}') w SEMI JOIN wanted USING(work_id)").fetchdf();w['missing_title']=w.title.isna()|w.title.fillna('').eq('')
 for rule,pattern in RULES.items():w[rule]=w.title.fillna('').str.contains(pattern,case=False,regex=True)
 w.to_parquet(p,index=False);write(u/'title_rules.json',{'rules':RULES,'not_AI_use_measure':True,'unique_first_month_papers':len(w),'missing_titles':int(w.missing_title.sum()),'missing_titles_remain_unflagged_and_audited':True});log('TITLES COMPLETE',len(w))
