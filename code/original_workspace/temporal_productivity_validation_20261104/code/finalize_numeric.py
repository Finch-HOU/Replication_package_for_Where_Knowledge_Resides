"""Wait for bounded covariance closure, then collect and validate actual output."""
from common import *
import subprocess

def main():
 receipt=F/'diagnostics/supplementary_covariance_budget_complete.json'
 minute=-1
 while not receipt.exists():
  now=int(time.time()//60)
  if now!=minute:
   minute=now;counts={}
   for mp,regime in [('direct','historical_offset'),('parent','shared_coremonth'),('direct','shared_coremonth')]:
    o=F/'relative_rate_certified_rankreuse'/mp/regime;counts[mp+'_'+regime]=len(list(o.glob('*_coefficients.csv')))
   log('SUPPLEMENTARY COVARIANCE WORK; fixed closure ~14:13; completed covariance sets',counts)
  time.sleep(10)
 for name in ['verify_partial_certificates.py','collect.py','validate_outputs.py']:
  subprocess.run([sys.executable,str(F/'code'/name)],check=True)
 assert load(F/'diagnostics/numerical_validation.json')['all_PPML_attempts_finished']
 log('EMPIRICAL CSV COLLECTION AND NUMERICAL VALIDATION READY FOR FINAL SUBSTANTIVE REVIEW')

if __name__=='__main__':main()
