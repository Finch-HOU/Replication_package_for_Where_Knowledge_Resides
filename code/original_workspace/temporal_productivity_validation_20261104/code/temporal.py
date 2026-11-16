from common import *
import argparse,gc

def stacked(p):
 ep,vp=events(p,744,753);ec,vc=events(p,786,795);vp8=np.column_stack([vp,np.zeros_like(vp)]);vc8=np.column_stack([np.zeros_like(vc),vc]);ii=np.r_[ep.i,ec.i];jj=np.r_[ep.j0,ec.j0];tt=np.r_[ep.mo-744,ec.mo-786];vv=np.vstack([vp8,vc8]);yn=['pre_'+x for x in OUTCOMES]+['current_'+x for x in OUTCOMES];data=SparseBatch(ii,jj,tt,vv,yn,p.I,p.J,10,p.mask);return data,ep,ec,vp,vc

def main(mp,stage):
 assert all((F/'results'/m/'replication/COMPLETE').exists()for m in ['parent','direct']),'Replicate both mappings before new analysis'
 o=F/'results'/mp/stage;o.mkdir(parents=True,exist_ok=True)
 if(o/'COMPLETE').exists():return
 p,z,sc=setup(mp);save(o/'sample.json',sample(p,sc));data,ep,ec,vp,vc=stacked(p);save(o/'windows.json',{'pre':{'start':'2022-01-01','end':'2022-10-31','months':10},'current_equal':{'start':'2025-07-01','end':'2026-04-30','months':10},'current_reference':{'start':'2025-07-01','end':'2026-05-31','months':11},'same_candidate_mask_and_scales':True,'pre_first_bundles':len(ep),'current10_first_bundles':len(ec),'current11_reference_preserved':True})
 if stage=='gradient':
  controls=['Gap','Distance','Shared']
  def xx(a,b):return np.broadcast_to(z['Distributed'][a:b,:,None,None],(b-a,p.J,10,1))
  def ct(a,b):return np.stack([z[k][a:b]for k in controls],axis=2)
  d=ProgressDesign(p.mask,p.fc,p.home,['Distributed'],xx,ct,False,4096);d.fit_many(data,o,'levels__',save_joint=True);del d;gc.collect()
  def xp(a,b):return z['Distributed'][a:b,:,None,None]*np.r_[np.zeros(10),np.ones(10)][None,None,:,None]
  pooled=SparseBatch(np.r_[ep.i,ec.i],np.r_[ep.j0,ec.j0],np.r_[ep.mo-744,ec.mo-786+10],np.vstack([vp,vc]),OUTCOMES,p.I,p.J,20,p.mask);d=ProgressDesign(p.mask,p.fc,p.home,['Distributed_x_CurrentPeriod'],xp,ct,True,4096);d.fit_many(pooled,o,'pooled_change__',save_joint=True)
  identities=[]
  for out in OUTCOMES:
   pre=load(o/('levels__pre_'+out+'.json'));cur=load(o/('levels__current_'+out+'.json'));change=load(o/('pooled_change__'+out+'.json'));bb=lambda r:np.array(r['sufficient_statistics']['beta']);err=float(np.max(np.abs(bb(change)-(bb(cur)-bb(pre)))));assert err<1e-10;identities.append({'outcome':out,'pooled_minus_separate_difference_error':err})
  save(o/'coefficient_identities.json',identities);save(o/'definition.json',{'levels_FE':'core-month, destination-month, Gap/Distance/shared-coverage month slopes; no pair FE because the regressor is frozen','pooled_FE':'pair, core-month, destination-month, Gap/Distance/shared-coverage month slopes','distributed_month_slopes_excluded_because_they_absorb_target':True,'not_original_AI_interaction_replication':True,'all_five_covariances':True})
 elif stage=='trajectory_marker':
  xx,ct=factory(p,z,10);d=ProgressDesign(p.mask,p.fc,p.home,NAMES,xx,ct,True,4096);d.fit_many(data,o,'',save_joint=True);save(o/'definition.json',{'marker':'Observed Jul2025-Apr2026 trajectories replayed onto Jan-Oct2022, by relative month; frozen Home basket unchanged','FE':'period-specific pair, core-month, destination-month and all four modifier-month slopes','estimator':'Joint eight-outcome fit with common X and period-specific nuisance coefficients; equivalent to a fully interacted stacked-period model','formal_pre_current_differences_use_cross_outcome_covariance':True,'not_genuine_historical_AI_or_parallel_trends':True,'five_covariances':True})
 else:raise ValueError(stage)
 (o/'COMPLETE').write_text('All components and specified comparison models attempted.');p.c.close();log('TEMPORAL COMPLETE',mp,stage)

if __name__=='__main__':ap=argparse.ArgumentParser();ap.add_argument('mapping');ap.add_argument('stage',choices=['gradient','trajectory_marker']);a=ap.parse_args();main(a.mapping,a.stage)
