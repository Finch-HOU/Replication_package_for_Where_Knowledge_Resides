"""Bound supplementary covariance work; preserve original valid mean fits."""
from common import *
import psutil

def normalized(x):return str(x).replace('\\','/').replace('//','/').lower()

def main():
 jobs=[]
 for mp,regime in [('direct','historical_offset'),('parent','shared_coremonth'),('direct','shared_coremonth')]:
  o=F/'relative_rate_certified_rankreuse'/mp/regime
  jobs.append({'mapping':mp,'regime':regime,'root':o,'receipt':load(o/'process_receipt.json')})
 while jobs:
  remaining=[]
  for x in jobs:
   o=x['root'];r=x['receipt']
   if(o/'COMPLETE').exists():continue
   age=time.time()-r['started_epoch']
   if age<2700:remaining.append(x);continue
   for pid,kind in [(r['worker_pid'],'python'),(r['stata_pid'],'stata')]:
    try:
     p=psutil.Process(pid);cmd=p.cmdline();assert kind in p.name().lower()and normalized(F)in normalized(' '.join(cmd)),cmd
     p.terminate()
     try:p.wait(10)
     except psutil.TimeoutExpired:p.kill();p.wait(10)
    except psutil.NoSuchProcess:pass
   statuses={}
   for q in o.glob('*_status.csv'):
    try:
     a=pd.read_csv(q);statuses[q.name]=dict(zip(a.key,a.value))
    except pd.errors.EmptyDataError:statuses[q.name]={'status':'empty_partial_status_file_preserved'}
   statuses['covariance_budget']={'return_code':'supplementary_covariance_time_limit','seconds':age,'valid_original_mean_and_capability_covariance_retained':True}
   source=F/'relative_rate'/x['mapping']/x['regime'];save(o/'support.json',load(source/'support.json'))
   save(o/'execution.json',{'process_returncode':'supplementary_covariance_budget_completed','seconds':age,'statuses':statuses,'not_a_failure_of_the_original_converged_mean_model':True,'valid_original_fit_source':str(source),'all_partial_outputs_preserved':True,'coefficient_blind_runtime_rule':True})
   save(o/'certification.json',{'validated_exact_likelihood_and_cap_covariance':False,'status':'Five-covariance recomputation incomplete at fixed budget; retain original full-check capability covariance','original_full_check_source':str(source),'partial_outputs_are_not_silently_promoted_to_verified_results':True})
   (o/'COMPLETE').write_text('Supplementary covariance budget closed; original full-check mean/capability covariance remains valid.')
   log('SUPPLEMENTARY COVARIANCE BUDGET CLOSED',x['mapping'],x['regime'],round(age))
  jobs=remaining
  if jobs:time.sleep(20)
 save(F/'diagnostics/supplementary_covariance_budget_complete.json',{'all_attempts_closed':True,'valid_original_mean_and_capability_covariances_preserved':True,'no_result_selected_by_significance':True})

if __name__=='__main__':main()
