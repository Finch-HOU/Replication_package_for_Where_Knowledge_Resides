"""Core-fold out-of-sample probabilities, excluding own first-entry labels.

The original own-history attempt is preserved separately. Irreversibility of first
entry motivates this correction before examining new proportional regressions.
"""
from common import *
from propensity import hierarchy
from scipy.optimize import minimize
import argparse

def fit_activity(Z,y,expected,train):
 zz=Z[train];yy=y[train];ex=expected[train].clip(1e-30);normal=max(yy.sum(),1)
 def fn(b):
  eta=zz@b;mu=ex*np.exp(np.clip(eta,-50,50));pen=1e-6*np.dot(b[1:],b[1:])/2;val=float((mu-yy*eta).sum()/normal+pen);grad=zz.T@(mu-yy)/normal;grad[1:]+=1e-6*b[1:];return val,grad
 opt=minimize(fn,np.zeros(3),jac=True,method='BFGS',options={'gtol':1e-8,'maxiter':300});gn=float(np.max(np.abs(fn(opt.x)[1])));return opt.x,{'parameters':opt.x.tolist(),'optimizer_success':bool(opt.success),'gradient_max':gn,'numerical_convergence':gn<1e-6,'iterations':int(opt.nit),'message':str(opt.message),'predictors':'Intercept, frozen standardized log prior publications and log field breadth; no own 2022 labels in held-out core prediction'}

def main(mp):
 src=F/'baseline_propensities'/mp;assert(src/'COMPLETE').exists();o=F/'baseline_propensities_crossfit'/mp;o.mkdir(parents=True,exist_ok=True)
 if(o/'COMPLETE').exists():return
 p,z,sc=setup(mp);d=load(src/'diagnostics.json');cell=np.load(src/'cell_ids.npy',mmap_mode='r');a=np.load(src/'training_counts.npz');counts=a['counts_by_fold'];Y=a['events_by_fold'];Bg=d['Gap_bins'];Bu=d['distributed_bins'];Hn=len(d['home_fields']);e,v=events(p,744,753);obs=np.zeros((p.I,2));np.add.at(obs,e.i,v[:,1:3]);fold=np.remainder(p.g,5).astype(int);Z=np.column_stack([np.ones(p.I),z['LogProductivity'][:,0],z['LogBreadth'][:,0]]);pred=np.lib.format.open_memmap(o/'p0.npy',mode='w+',dtype=np.float64,shape=(p.I,p.J,2));details=[];cv=[];cal=np.zeros(2);maxp=0
 for ff in range(5):
  trainE=(counts.sum(0)-counts[ff])*10;trainY=Y.sum(0)-Y[ff];rr=[];params=[]
  for k in range(2):
   r,dd=hierarchy(trainE,trainY[:,k],p.J,Hn,Bg,Bu);rr.append(r);params.append(dd)
  rr=np.column_stack(rr);expected=np.zeros((p.I,2))
  for aa in range(0,p.I,8192):
   bb=min(aa+8192,p.I);expected[aa:bb]=(rr[cell[aa:bb]]*p.mask[aa:bb,:,None]).sum(1)*10
  beta=[];fit=[]
  for k in range(2):
   b,dd=fit_activity(Z,obs[:,k],expected[:,k],fold!=ff);beta.append(b);fit.append(dd)
  theta=np.exp(Z@np.column_stack(beta));total0=0.;nmask=0
  for aa in range(0,p.I,8192):
   bb=min(aa+8192,p.I);sel=np.flatnonzero(fold[aa:bb]==ff)+aa
   if not len(sel):continue
   lam=rr[cell[sel]]*theta[sel,None,:];pp=lam/(1+lam.sum(2,keepdims=True));pred[sel]=pp;mk=p.mask[sel];cal+=(pp*mk[:,:,None]).sum((0,1))*10;total0+=float((np.log1p(-pp.sum(2))*mk).sum()*10);nmask+=int(mk.sum());maxp=max(maxp,float(pp[mk].sum(1).max()))
  ee=e[fold[e.i.to_numpy()]==ff];ps=pred[ee.i.to_numpy(),ee.j0.to_numpy()];ys=ee.specialist.to_numpy(bool);point=ps[np.arange(len(ps)),np.where(ys,0,1)];ll=total0+float((np.log(point)-np.log1p(-ps.sum(1))).sum());rate=trainY.sum(0)/trainE.sum();prob=rate/(1+rate.sum());ll0=float((len(ee)*0)+sum(np.log(prob[0]if ss else prob[1])for ss in ys)+(nmask*10-len(ee))*np.log1p(-prob.sum()));row={'fold':ff,'candidate_months':nmask*10,'specialist_events':int(ys.sum()),'no_specialist_events':int((~ys).sum()),'crossfit_conditional_log_score':ll,'training_global_rate_log_score':ll0,'log_score_improvement':ll-ll0};cv.append(row);details.append({'fold':ff,'hierarchy':params,'frozen_core_activity':fit,'own_core_labels_excluded_from_all_fitted_parameters':True});log('CROSSFIT BASELINE',mp,ff,row)
 pred.flush();assert np.isfinite(pred).all()and pred.min()>0 and maxp<1
 save(o/'diagnostics.json',{'sample':sample(p,sc),'training_window':['2022-01-01','2022-10-31'],'fold_rule':'global core index modulo five, fixed independently of outcomes','exact_heldout_core_labels_excluded':True,'frozen_core_predictors':['log prior publication count through2021','log prior field breadth through2021'],'post_outcomes_used':False,'historical_own_core_first_entry_count_used_in_own_prediction':False,'all_zero_baseline_and_current_cores_retained':True,'parent_cell_counts_source':str(src/'training_counts.npz'),'hyperparameters_and_convergence':details,'cross_validation':cv,'observed_pre_events':obs.sum(0).tolist(),'out_of_fold_predicted_pre_events':cal.tolist(),'maximum_p0_total':maxp,'generated_regressor_uncertainty_remains':True,'author_overlap_between_core_folds_not_eliminated':True,'baseline_conditional_surface_is_approximate':True,'superseded_own_history_attempt_preserved':str(src)});pd.DataFrame(cv).to_csv(o/'cross_validation.csv',index=False);(o/'COMPLETE').write_text('Positive core-cross-fitted historical baseline probabilities, without own first-entry labels.');p.c.close();log('CROSSFIT PROPENSITIES COMPLETE',mp)

if __name__=='__main__':ap=argparse.ArgumentParser();ap.add_argument('mapping');a=ap.parse_args();main(a.mapping)
