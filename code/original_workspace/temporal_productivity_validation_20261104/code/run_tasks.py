from common import *
import subprocess,psutil,argparse

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--only-temporal',action='store_true');args0=ap.parse_args()
 assert all((F/'results'/m/'replication/COMPLETE').exists()for m in ['parent','direct'])
 tasks=[]
 for mp in ['parent','direct']:
  for st in ['gradient','trajectory_marker']:tasks.append((mp+'_'+st,['temporal.py',mp,st],F/'results'/mp/st/'COMPLETE',None))
  tasks.append((mp+'_propensity',['propensity.py',mp],F/'baseline_propensities'/mp/'COMPLETE',None));tasks.append((mp+'_proportional',['proportional.py',mp],F/'results'/mp/'proportional/COMPLETE',F/'baseline_propensities'/mp/'COMPLETE'))
 if args0.only_temporal:tasks=[t for t in tasks if t[1][0]=='temporal.py']
 running={};done=[];failed=[];(F/'logs').mkdir(exist_ok=True)
 while tasks or running:
  for task in tasks.copy():
   name,args,mark,dep=task
   if mark.exists():done.append(name);tasks.remove(task);continue
   if len(running)>=6 or(dep is not None and not dep.exists()):continue
   logf=(F/'logs'/(name+'.log')).open('a',encoding='utf8');logf.write('\nRESUMED RUN '+time.strftime('%Y-%m-%d %H:%M:%S')+'\n');logf.flush();pr=subprocess.Popen([sys.executable,'-X','utf8',str(F/'code'/args[0]),*args[1:]],stdout=logf,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW);running[name]=(pr,mark,logf);tasks.remove(task);log('START',name,pr.pid)
  for name,(pr,mark,lf)in list(running.items()):
   rc=pr.poll()
   if rc is None:continue
   lf.close();running.pop(name)
   if rc==0 and mark.exists():done.append(name);log('DONE',name)
   else:failed.append({'task':name,'returncode':rc});log('FAILED',name,rc)
  blocked=[t for t in tasks if t[3]is not None and not t[3].exists()]
  if not running and len(blocked)==len(tasks)and tasks:
   failed +=[{'task':x[0],'dependency_missing':str(x[3])}for x in tasks];tasks=[]
  save(F/'logs'/('temporal_task_status.json'if args0.only_temporal else'task_status.json'),{'done':done,'failed':failed,'running':{k:v[0].pid for k,v in running.items()},'pending':[x[0]for x in tasks]})
  if tasks or running:time.sleep(10)
 log('QUEUE CLOSED',len(done),'done',len(failed),'failed')

if __name__=='__main__':main()
