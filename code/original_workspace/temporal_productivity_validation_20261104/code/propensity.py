"""Historical-only, full-risk empirical-Bayes incidence predictions.

Grouping approximates a conditional propensity surface; no post outcomes train it.
All smoothing parameters maximize pre-period marginal likelihood.
"""
from common import *
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
import argparse

def shape(y,mu):
 y=np.asarray(y,float);mu=np.asarray(mu,float).clip(1e-14);ok=np.isfinite(mu);y,mu=y[ok],mu[ok]
 def objective(logk):
  k=np.exp(logk);v=gammaln(y+k)-gammaln(k)-gammaln(y+1)+k*(-np.log1p(mu/k))+y*(np.log(mu)-np.log(k+mu));return -float(v.sum())
 opt=minimize_scalar(objective,bounds=(-10,15),method='bounded',options={'xatol':1e-6});return float(np.exp(opt.x)),{'log_shape':float(opt.x),'success':bool(opt.success),'objective':float(opt.fun),'near_bound':bool(opt.x< -9.9 or opt.x>14.9)}

def hierarchy(E,Y,J,Hn,Bg,Bu):
 es=E.reshape(J,Hn,Bg,Bu);ys=Y.reshape(J,Hn,Bg,Bu);glob=Y.sum()/E.sum();ej=es.sum((1,2,3));yj=ys.sum((1,2,3));k0,a0=shape(yj,ej*glob);lj=(yj+k0)/(ej+k0/glob);eh=es.sum((2,3));yh=ys.sum((2,3));k1,a1=shape(yh.ravel(),(eh*lj[:,None]).ravel());lh=(yh+k1)/(eh+k1/lj[:,None]);k2,a2=shape(Y,E*np.broadcast_to(lh[:,:,None,None],es.shape).ravel());lc=(Y+k2)/(E+k2/np.broadcast_to(lh[:,:,None,None],es.shape).ravel());return lc,{'global_monthly_rate':float(glob),'destination_shape':k0,'home_destination_shape':k1,'architecture_cell_shape':k2,'optimization':[a0,a1,a2]}

def main(mp):
 assert all((F/'results'/m/'replication/COMPLETE').exists()for m in ['parent','direct'])
 o=F/'baseline_propensities'/mp;o.mkdir(parents=True,exist_ok=True)
 if(o/'COMPLETE').exists():return
 p,z,sc=setup(mp);e,v=events(p,744,753);hvals,h=np.unique(p.home,return_inverse=True);Hs=len(hvals)
 edges={k:np.unique(np.quantile(z[k][p.mask],[.25,.5,.75])).tolist()for k in ['Gap','Distributed']};Bg=len(edges['Gap'])+1;Bu=len(edges['Distributed'])+1;L=p.J*Hs*Bg*Bu;cell=np.lib.format.open_memmap(o/'cell_ids.npy',mode='w+',dtype=np.uint32,shape=p.mask.shape);counts=np.zeros((5,L),float);fold=np.remainder(p.g,5).astype(int)
 for a in range(0,p.I,8192):
  b=min(a+8192,p.I);gb=np.searchsorted(edges['Gap'],z['Gap'][a:b],side='right');ub=np.searchsorted(edges['Distributed'],z['Distributed'][a:b],side='right');cc=((np.arange(p.J)[None,:]*Hs+h[a:b,None])*Bg+gb)*Bu+ub;cell[a:b]=cc
  for ff in range(5):counts[ff]+=np.bincount(cc[(fold[a:b,None]==ff)&p.mask[a:b]],minlength=L)
  if a%131072==0:log('BASELINE RISK CELL AGGREGATION',mp,a,p.I)
 cell.flush();Y=np.zeros((5,L,2));ec=cell[e.i.to_numpy(),e.j0.to_numpy()];ef=fold[e.i.to_numpy()];np.add.at(Y,(ef,ec),v[:,1:3]);E=counts.sum(0)*10;yy=Y.sum(0);assert int(E.sum())==int(p.mask.sum())*10;assert np.allclose(yy.sum(0),v[:,1:3].sum(0));np.savez_compressed(o/'training_counts.npz',counts_by_fold=counts,events_by_fold=Y)
 rates=[];details=[]
 for k,nm in enumerate(['specialist','no_specialist']):
  rr,dd=hierarchy(E,yy[:,k],p.J,Hs,Bg,Bu);rates.append(rr);details.append({'outcome':nm,**dd})
 rates=np.column_stack(rates);theta=np.empty((p.I,2));obs=np.zeros((p.I,2));np.add.at(obs,e.i,v[:,1:3]);expected=np.zeros_like(obs)
 for a in range(0,p.I,8192):
  b=min(a+8192,p.I);expected[a:b]=(rates[cell[a:b]]*p.mask[a:b,:,None]).sum(1)*10
 for k,nm in enumerate(['specialist','no_specialist']):
  kk,dd=shape(obs[:,k],expected[:,k]);theta[:,k]=(obs[:,k]+kk)/(expected[:,k]+kk);details[k].update(core_shape=kk,core_shape_optimization=dd)
 pred=np.lib.format.open_memmap(o/'p0.npy',mode='w+',dtype=np.float64,shape=(p.I,p.J,2));cal=np.zeros(2);max_total=0;zero_hist_positive=np.zeros(2,int);quant=[]
 for a in range(0,p.I,8192):
  b=min(a+8192,p.I);raw=rates[cell[a:b]]*theta[a:b,None,:]
  # Competing-outcome probability transformation guarantees a valid probability
  # surface even in very thin cells; on rare risks it closely matches intensity.
  pp=raw/(1+raw.sum(2,keepdims=True));pred[a:b]=pp;cal+=(pp*p.mask[a:b,:,None]).sum((0,1))*10;max_total=max(max_total,float(pp[p.mask[a:b]].sum(1).max()))
  for k in range(2):zero_hist_positive[k]+=int(((obs[a:b,k]==0)[:,None]&p.mask[a:b]&(pp[:,:,k]>0)).sum())
 pred.flush();assert np.isfinite(pred).all()and pred.min()>0 and max_total<1
 # Core-fold CV excludes every outcome of a held-out core from hierarchy fitting.
 # The held-out-core activity multiplier is one, never estimated from its labels.
 cv=[]
 for ff in range(5):
  trainE=(counts.sum(0)-counts[ff])*10;trainY=(Y.sum(0)-Y[ff]);testE=counts[ff]*10;testY=Y[ff];rr=np.column_stack([hierarchy(trainE,trainY[:,k],p.J,Hs,Bg,Bu)[0]for k in range(2)]);pr=rr/(1+rr.sum(1,keepdims=True));mu=trainY.sum(0)/trainE.sum();base=mu/(1+mu.sum());zeroN=testE-testY.sum(1);ll=float((testY*np.log(pr)).sum()+(zeroN*np.log1p(-pr.sum(1))).sum());ll0=float((testY*np.log(base)).sum()+zeroN.sum()*np.log1p(-base.sum()));cv.append({'fold':ff,'candidate_months':int(testE.sum()),'specialist_events':int(testY[:,0].sum()),'no_specialist_events':int(testY[:,1].sum()),'conditional_log_score':ll,'global_rate_log_score':ll0,'log_score_improvement':ll-ll0,'core_multiplier_not_fit_on_heldout_labels':True});log('BASELINE CORE-FOLD CV',mp,ff,cv[-1]['log_score_improvement'])
 summary={'sample':sample(p,sc),'training_window':['2022-01-01','2022-10-31'],'training_months':10,'use_of_post_outcomes':False,'cell_definition':'destination x frozen Home field x frozen-risk Gap quartile x distributed-coverage quartile; positive empirical-Bayes hierarchy plus core activity factor','quartile_edges_standardized':edges,'Gap_bins':Bg,'distributed_bins':Bu,'cells_with_risk':int((E>0).sum()),'total_cells_including_empty':L,'home_fields':hvals.tolist(),'hyperparameters':details,'observed_pre_events':obs.sum(0).tolist(),'predicted_pre_events_after_competing_probability_transform':cal.tolist(),'maximum_p0_total':max_total,'candidate_pairs_without_own_pre_events_but_positive_predictions':zero_hist_positive.tolist(),'all_zero_pre_cores_retained':True,'finite_positive_predictions_for_all_candidates':True,'probability_transform':'p0_o=lambda_o/(1+lambda_S+lambda_No)','first_stage_uncertainty_not_in_downstream_conditional_sandwich':True,'conditional_surface_is_binned_approximation_not_actual_attempt_probabilities':True,'cross_validation':cv};save(o/'diagnostics.json',summary);pd.DataFrame(cv).to_csv(o/'cross_validation.csv',index=False);np.save(o/'core_activity_multiplier.npy',theta);np.save(o/'cell_monthly_intensities.npy',rates);(o/'COMPLETE').write_text('Historical-only, positive full-risk propensity predictions; no post outcome labels.');p.c.close();log('BASELINE PROPENSITIES COMPLETE',mp)

if __name__=='__main__':ap=argparse.ArgumentParser();ap.add_argument('mapping');a=ap.parse_args();main(a.mapping)
