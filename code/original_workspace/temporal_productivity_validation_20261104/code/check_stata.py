from common import *
import subprocess

def main():
 o=F/'diagnostics/stata';o.mkdir(parents=True,exist_ok=True);f=o/'availability.do';f.write_text(f'''clear all
set more off
capture log close _all
log using "{(o/'availability_detail.log').as_posix()}", text replace
display "STATA_VERSION " c(stata_version)
sysdir
capture noisily which ppmlhdfe
display "PPMLHDFE_RC " _rc
capture noisily which reghdfe
display "REGHDFE_RC " _rc
capture noisily which ftools
display "FTOOLS_RC " _rc
log close
exit, clear
''',encoding='ascii');p=subprocess.run(['C:/Program Files/Stata17/StataMP-64.exe','/e','do',str(f)],cwd=o,creationflags=subprocess.CREATE_NO_WINDOW,capture_output=True,text=True,timeout=120);log('STATA AUDIT EXIT',p.returncode);print('\n'.join(line for line in (o/'availability_detail.log').read_text(errors='replace').splitlines() if any(k in line.lower() for k in ['rc ', 'stata_version', 'ado', 'ppmlhdfe', 'reghdfe', 'ftools', 'not found'])))

if __name__=='__main__':main()
