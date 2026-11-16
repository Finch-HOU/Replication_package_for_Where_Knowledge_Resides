from common import *
from constructs import *
from streaming_fe import StreamingDesign,SparseBatch
import argparse,gc
class ReportingDesign(StreamingDesign):
 def blocks(self):
  self.pass_number=getattr(self,'pass_number',0)+1;nextmark=0
  for a,b in super().blocks():
   if a>=nextmark:log('FE PASS',self.pass_number,'cores',a,'/',self.I);nextmark=a+max(self.I//4,self.chunk)
   yield a,b
StreamingDesign=ReportingDesign
SPECS={
 'F1':['Gap','C'], 'F2':['Distance','C'], 'F3':['Gap','Distance','C'],
 'F4':['Gap','Distance','Applicability','Access','GapApplicability','GapAccess','ApplicabilityAccess','C'],
 'F5':['Gap','Distance','Applicability','Access','InternalShare','Depth','C'],
 'F6':['Gap','Distance','Applicability','Access','Shared','Distributed','C','GapLogSize','GapLogBreadth','GapLogProductivity'],
 'F7':['Gap','GapSquared','Distance','Applicability','Access','C'],
 'F8':['Gap','Distance','Applicability','Access','C','CGap','CApplicability','CAccess'],
 'F9':['Top5Gap','Distance','Applicability','Access','Top5GapApplicability','Top5GapAccess','ApplicabilityAccess','C']}
MODULES={'requirements':['C','Gap','Distance','Applicability'],'requirements_C':['Gap','Distance','Applicability'],'need_access':['Gap','Distance','Access','GapAccess'],'architecture':['Gap','Distance','Distributed','Shared']}
def run(co,pop,mp,stage):
 out=F/'results'/pop/co/mp/stage;out.mkdir(parents=True,exist_ok=True)
 if (out/'COMPLETE').exists():return
 import psutil,atexit
 lock=out/'RUNNING.lock'
 while True:
  if(out/'COMPLETE').exists():return
  try:
   with lock.open('x')as f:f.write(str(os.getpid()))
   break
  except FileExistsError:
   try:alive=psutil.pid_exists(int(lock.read_text()))
   except (ValueError,FileNotFoundError):alive=False
   if alive:time.sleep(5)
   else:lock.unlink(missing_ok=True)
 atexit.register(lambda:lock.unlink(missing_ok=True))
 future=stage.startswith('future_');module=stage.removeprefix('future_');p=Panel(co,pop,mp,classes='BC'if module in ['foundation','requirements']else'C');log('PANEL',stage,pop,co,mp,p.I);needed=None if module=='foundation'else set(MODULES[module]+(['LogSize','LogBreadth','LogProductivity']if module=='architecture'else['LogBreadth','LogProductivity']if module=='requirements_C'else[]));raw,z,sc=constructs(p,needed);save(out/'scaling.json',sc)
 if stage=='foundation':
  diagnostic(p,raw,z,sc,out);e,v,yn=first_events(p,True);data=SparseBatch(e.i,e.j0,np.zeros(len(e),int),v,yn,p.I,p.J,1,p.mask);fields=p.s.field_id.iloc[p.js].to_numpy();mod=SPECS.items()
 else:
  e,v,yn=first_events(p,future)
  if future:p.A=p.A.mean(1,keepdims=True);p.ha=p.ha.mean(1,keepdims=True);p.T=1
  data=SparseBatch(e.i,e.j0,np.zeros(len(e),int)if future else e.mo-786,v,yn,p.I,p.J,p.T,p.mask);mod=[(stage,MODULES[module])];fields=p.fc
 for model,terms in mod:
  pref=model+'__'
  if fit_complete(out,pref):continue
  if stage=='foundation':
   names=terms
   def fx(a,b):return np.stack([z[k][a:b]for k in terms],axis=2)[:,:,None,:]
   ctrl=None
  else:
   names=[k+'_'+src for k in terms for src in ['Destination','Home']]
   extra=['LogSize','LogBreadth','LogProductivity']if module=='architecture'else['LogBreadth','LogProductivity']if module=='requirements_C'else[];names+=[k+'_Destination'for k in extra]
   def fx(a,b):
    xx=[z[k][a:b,:,None]* (p.A[None,:,:]if src=='Destination'else p.ha[a:b,None,:])for k in terms for src in ['Destination','Home']];xx.extend(z[k][a:b,:,None]*p.A[None,:,:]for k in extra);return np.stack(xx,axis=3)
   def ctrl(a,b):return np.stack([z[k][a:b]for k in terms],axis=2)
  log('START',pop,co,mp,stage,model,names);d=StreamingDesign(p.mask,fields,p.home,names,fx,ctrl,stage!='foundation'and not future,chunk=1024);d.fit_many(data,out,pref,save_joint=True);del d;gc.collect();log('DONE',pop,co,mp,stage,model)
  # Structural zero controls do not make the economic question infeasible.
  # Preserve the failed declared fit, then estimate its exact identified subset.
  zero=[k for k in names if (k=='GapLogSize' or k=='LogSize_Destination') and sc['LogSize']['sd']<=1e-12]
  if zero:
   original=fx;ix=[k for k in range(len(names))if names[k]not in zero];n2=[names[k]for k in ix]
   def identified(a,b):return original(a,b)[:,:,:,ix]
   p2=pref.replace('__','_estimable__');save(out/(p2+'structural_controls.json'),{'excluded_exact_zero_columns':zero,'original_rank_failure_retained':True,'not_outcome_or_significance_selected':True});d=StreamingDesign(p.mask,fields,p.home,n2,identified,ctrl,stage!='foundation'and not future,chunk=1024);d.fit_many(data,out,p2,save_joint=True);del d;gc.collect()
 save(out/'definition.json',{'classes':'BC'if module in ['foundation','requirements']else'C','unit':'candidate pair cumulative Jan-Oct2022'if stage=='foundation'or future else'candidate-month Jul2025-May2026','zero_inclusive':True,'FE':'core and destination plus modifier main effects'if stage=='foundation'or future else'pair, core-month, destination-month, modifier-month slopes','primary_cluster':'destination broad knowledge field'if stage=='foundation'else'MMLU capability path','specialist':'any additional prior-year demonstrated destination expert in any first-month paper','mutually_exclusive_priority':True,'total_equals_sum':True,'signed_equals_difference':True,'all_controls_predetermined':True,'first_events':len(e),'first_month_multi_paper_events':int((e.papers>1).sum()),'scaling':'continuous components risk-set standardized, products of standardized primitives; C indicator unstandardized','stage':stage,'historical_future_marker_not_AI_effect':future})
 (out/'COMPLETE').write_text('All declared models and outcomes attempted; rank failures retained.');p.c.close()
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('cohort');ap.add_argument('population');ap.add_argument('mapping');ap.add_argument('stage',choices=['foundation',*MODULES,*['future_'+x for x in MODULES]]);a=ap.parse_args();run(a.cohort,a.population,a.mapping,a.stage)
