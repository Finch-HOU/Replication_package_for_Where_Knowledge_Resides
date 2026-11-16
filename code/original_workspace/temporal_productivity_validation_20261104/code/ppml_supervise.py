"""Coefficient-blind numerical-budget closure; never kills unrelated jobs."""
from common import *
import psutil,subprocess,re

def norm(s):return str(s).replace('\\','/').replace('//','/').lower()

def task_process(pid,kind):
 try:
  pp=psutil.Process(pid);cmd=pp.cmdline()
  assert kind.lower()in pp.name().lower()
  assert norm(F)in norm(' '.join(cmd))or norm(pp.cwd())==norm(F.parent)
  assert any('ppml' in z.lower()or 'estimate.do'in z.lower()for z in cmd)
  return pp
 except psutil.NoSuchProcess:return None

def terminate(pp):
 if pp is not None:
  pp.terminate()
  try:pp.wait(20)
  except psutil.TimeoutExpired:pp.kill();pp.wait(20)

def main():
 jobs=[]
 for mp,driver,child,regime in [('parent',54128,54848,'historical_offset'),('direct',9088,47816,'historical_offset'),('parent',18312,35104,'shared_coremonth'),('direct',43716,17972,'shared_coremonth'),('parent',48432,32520,'historical_offset'),('direct',51316,35204,'historical_offset')]:
  acc=driver in [48432,51316];o=F/('relative_rate_accelerated'if acc else'relative_rate')/mp/regime
  pp=task_process(child,'stata');dp=task_process(driver,'python')
  if pp is None:continue
  jobs.append({'mapping':mp,'regime':regime,'accelerated':acc,'root':str(o),'driver_pid':driver,'stata_pid':child,'started_epoch':pp.create_time(),'minimum_seconds':2700 if acc else 5400,'maximum_seconds':10800,'verified_command':pp.cmdline(),'finished':False})
 for mp in ['parent','direct']:
  o=F/'relative_rate_accelerated'/mp/'shared_coremonth'
  if(o/'COMPLETE').exists():continue
  with(F/'logs'/('PPML_bounded_shared_'+mp+'.log')).open('w')as out, (F/'logs'/('PPML_bounded_shared_'+mp+'.err')).open('w')as err:
   pp=subprocess.Popen([sys.executable,str(F/'code/ppml_bounded_worker.py'),mp,'shared_coremonth'],cwd=F.parent,creationflags=subprocess.CREATE_NO_WINDOW,stdout=out,stderr=err)
  for unused in range(30):
   if(o/'process_receipt.json').exists():break
   time.sleep(1)
  x=load(o/'process_receipt.json');jobs.append({'mapping':mp,'regime':'shared_coremonth','accelerated':True,'root':str(o),'driver_pid':pp.pid,'stata_pid':x['stata_pid'],'started_epoch':x['started_epoch'],'minimum_seconds':2700,'maximum_seconds':10800,'finished':False})
 save(F/'diagnostics/PPML_running_budget.json',jobs)
 while not all(x['finished']for x in jobs):
  for x in jobs:
   if x['finished']:continue
   o=Path(x['root']);age=time.time()-x['started_epoch']
   if(o/'COMPLETE').exists():x['finished']=True;x['ended_by']='estimator_or_worker';continue
   txt=(o/'estimation_detail.log').read_text(errors='replace')if(o/'estimation_detail.log').exists()else''
   progress=bool(list(o.glob('*_coefficients.csv')))or bool(re.search(r'^\s*Iteration\s+\d+',txt,re.M))
   if age<x['minimum_seconds']or(progress and age<x['maximum_seconds']):continue
   dp=task_process(x['driver_pid'],'python');pp=task_process(x['stata_pid'],'stata')
   # Stop the sequential driver first, so it cannot start a duplicate next model.
   terminate(dp);terminate(pp)
   statuses={}
   for f in o.glob('*_status.csv'):
    tb=pd.read_csv(f);statuses[f.name]={r.key:r.value for r in tb.itertuples()}
   statuses['computational_budget']={'return_code':'bounded_time_limit','seconds':age,'no_exported_first_fit':not bool(list(o.glob('*_coefficients.csv'))),'visible_IRLS_progress':progress}
   if not(o/'support.json').exists():save(o/'support.json',load(F/'relative_rate'/x['mapping']/x['regime']/'support.json'))
   save(o/'execution.json',{'process_returncode':'terminated_at_declared_numerical_budget','seconds':age,'statuses':statuses,'computationally_unresolved':True,'no_zero_effect_or_MLE_nonexistence_inference':True,'same_specification_as_original_namespace':x['accelerated'],'all_partial_files_preserved':True,'verified_process_receipt':x})
   (o/'COMPLETE').write_text('Attempt closed at coefficient-blind numerical budget; unresolved computation is not an estimated zero.')
   x['finished']=True;x['ended_by']='declared_numerical_budget';log('BOUNDED PPML CLOSED',x['mapping'],x['regime'],x['accelerated'],round(age))
  save(F/'diagnostics/PPML_running_budget.json',jobs)
  if not all(x['finished']for x in jobs):time.sleep(20)
 log('ALL DECLARED PPML ATTEMPTS CLOSED')

if __name__=='__main__':main()
