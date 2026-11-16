"""Reproducible coefficient-blind runtime limits for official PPML attempts."""
from common import *
import subprocess,re

def run_official(o,minimum_seconds,maximum_seconds=10800):
 started=time.time();pp=subprocess.Popen(['C:/Program Files/Stata17/StataMP-64.exe','/e','do',str(o/'estimate.do')],cwd=o,creationflags=subprocess.CREATE_NO_WINDOW,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 save(o/'process_receipt.json',{'worker_pid':os.getpid(),'stata_pid':pp.pid,'started_epoch':started,'minimum_seconds':minimum_seconds,'maximum_seconds':maximum_seconds,'execution_directory':str(o)})
 stopped=False;progress=False
 while pp.poll()is None:
  age=time.time()-started
  txt=(o/'estimation_detail.log').read_text(errors='replace')if(o/'estimation_detail.log').exists()else''
  progress=bool(list(o.glob('*_coefficients.csv')))or bool(re.search(r'^\s*Iteration\s+\d+',txt,re.M))
  if age>=maximum_seconds or(age>=minimum_seconds and not progress):
   pp.kill();pp.wait();stopped=True;break
  try:pp.wait(timeout=20)
  except subprocess.TimeoutExpired:pass
 return {'process_returncode':pp.returncode,'seconds':time.time()-started,'computationally_unresolved':stopped,'minimum_seconds':minimum_seconds,'maximum_seconds':maximum_seconds,'visible_numerical_progress':progress,'no_coefficient_or_significance_based_stopping':True}

def read_statuses(o,runtime):
 statuses={}
 for p in o.glob('*_status.csv'):
  t=pd.read_csv(p);statuses[p.name]={r.key:r.value for r in t.itertuples()}
 if runtime['computationally_unresolved']:statuses['computational_budget']={'return_code':'bounded_time_limit','seconds':runtime['seconds'],'no_exported_first_fit':not bool(list(o.glob('*_coefficients.csv'))),'visible_IRLS_progress':runtime['visible_numerical_progress']}
 return statuses
