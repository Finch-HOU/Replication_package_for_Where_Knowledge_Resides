from common import *
import subprocess,argparse

def main(mp):
 for script in ['crossfit_propensity.py','proportional.py']:
  rc=subprocess.call([sys.executable,'-X','utf8',str(F/'code'/script),mp],creationflags=subprocess.CREATE_NO_WINDOW)
  if rc:raise SystemExit(rc)

if __name__=='__main__':ap=argparse.ArgumentParser();ap.add_argument('mapping');a=ap.parse_args();main(a.mapping)
