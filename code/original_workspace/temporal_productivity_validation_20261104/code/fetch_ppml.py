"""Fetch the official public PPMLHDFE source into this project only."""
from common import *
import urllib.request

def get(url):
 req=urllib.request.Request(url,headers={'User-Agent':'OpenAlex-research-validation','Accept':'application/vnd.github+json'});return urllib.request.urlopen(req,timeout=45).read()

def main():
 o=F/'dependencies/ppmlhdfe';o.mkdir(parents=True,exist_ok=True)
 if(o/'source_manifest.json').exists():
  if not(o/'reghdfe5.mata').exists():
   sha='master';url='https://raw.githubusercontent.com/sergiocorreia/reghdfe/master/src/reghdfe5.mata';b=get(url);(o/'reghdfe5.mata').write_bytes(b);save(o/'compatibility_source_manifest.json',{'repository':'https://github.com/sergiocorreia/reghdfe','commit':sha,'file':'reghdfe5.mata','bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'reason':'Official PPMLHDFE requires the legacy Mata interface missing from the host 2023 reghdfe package; global installation unchanged.'});log('FETCHED OFFICIAL COMPATIBILITY SOURCE',len(b))
  return
 meta=json.loads(get('https://api.github.com/repos/sergiocorreia/ppmlhdfe/commits/master'));sha=meta['sha'];base=f'https://raw.githubusercontent.com/sergiocorreia/ppmlhdfe/{sha}/src/';pkg=get(base+'ppmlhdfe.pkg');(o/'ppmlhdfe.pkg').write_bytes(pkg);files=[]
 for line in pkg.decode().splitlines():
  if not line.startswith('f '):continue
  nm=line[2:].strip();p=o/nm;assert p.resolve().is_relative_to(o.resolve());p.parent.mkdir(parents=True,exist_ok=True);b=get(base+nm);p.write_bytes(b);files.append({'file':nm,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()});log('FETCHED OFFICIAL PPMLHDFE',nm,len(b))
 save(o/'source_manifest.json',{'repository':'https://github.com/sergiocorreia/ppmlhdfe','commit':sha,'source_url':base,'project_local_only':True,'files':files})

if __name__=='__main__':main()
