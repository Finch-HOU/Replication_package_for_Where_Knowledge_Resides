from common import *
from restriction_tools import restrictions,linear_statistics
import argparse,gc

def main(mp,home_only=False):
 assert(F/'baseline_propensities_crossfit'/mp/'COMPLETE').exists()
 o=F/'results'/mp/('home_proportional'if home_only else'proportional');o.mkdir(parents=True,exist_ok=True)
 save(o/'counterfactor_definition.json',{'counterfactor':'HomeGain only'if home_only else'HomeGain x standardized DistributedCoverage','two_propensity_weighted_columns_use_this_counterfactor':True,'original_focal_test_remains_Distributed_Home':True,'general_home_factor_contains_no_structural_U_moderation':home_only,'common_response_shared_across_S_and_No':True})
 if(o/'COMPLETE').exists():return
 p,z,sc=setup(mp);pred=np.load(F/'baseline_propensities_crossfit'/mp/'p0.npy',mmap_mode='r');e,v=events(p,786,796);data=SparseBatch(e.i,e.j0,e.mo-786,v,OUTCOMES,p.I,p.J,11,p.mask);fx,ctrl=factory(p,z)
 def basis(a,b):
  xx=fx(a,b);foc=np.broadcast_to(p.ha[a:b,None,:],xx.shape[:3])if home_only else xx[:,:,:,5];return np.concatenate([xx,foc[:,:,:,None]*pred[a:b,:,None,:]*1e5],axis=3)
 names=NAMES+(['Historical_p0_S_x_HomeGain_scaled1e5','Historical_p0_No_x_HomeGain_scaled1e5']if home_only else['Historical_p0_S_x_focal_scaled1e5','Historical_p0_No_x_focal_scaled1e5']);d=ProgressDesign(p.mask,p.fc,p.home,names,basis,ctrl,True,2048);d.fit_many(data,o,'basis__',save_joint=True,keep_state=True)
 if not hasattr(d,'fit_state'):
  save(o/'failure.json',{'status':'basis_rank_deficient','all_failures_preserved':True});(o/'COMPLETE').write_text('Attempted, basis rank failure retained.');return
 st=d.fit_state;Q=st['Q'];xy=Q@st['beta'][:,[1,2]];plan=restrictions(Q,xy);B=plan['restricted_betas'];A=plan['influence_vectors'];points=plan['primitive_points'];pi=plan['signed_projection_loading'];I,J,T=p.I,p.J,11;G,GH=d.G,d.GH;cap=np.zeros((G,5));home=np.zeros((GH,5));hc=np.zeros((GH,G,5));teammeat=np.zeros((5,5));intermeat=np.zeros((5,5));ss=np.zeros(5);yidx=[1,2,1,2,1,2];gy=st['gammaY'][:,:,yidx];ry=st['rowY'][:,:,yidx]
 N=int(p.mask.sum())*11;df=st['df_fe'];f0=(N-1)/(N-df-10);fc=(2*N-1)/(2*N-2*df-19);scale=np.sqrt([f0,f0,fc,f0,f0]);start=time.time()
 for a,b in d.blocks():
  x=d.residual_x(a,b);mk=p.mask[a:b].astype(float);mods=d.ctr(a,b);rowfit=np.einsum('ij,jtr->itr',mk,gy[:J],optimize=True)/d.den[a:b,None,None]+np.einsum('iq,qtr->itr',d.modmean[a:b],gy[J:],optimize=True);fe=gy[:J][None,:,:,:]+np.einsum('ijq,qtr->ijtr',mods,gy[J:],optimize=True)+(ry[a:b]-rowfit)[:,None,:,:];u=-fe-np.einsum('ijtk,kr->ijtr',x,B,optimize=True);hh=np.einsum('ijtk,kr->ijtr',x,A,optimize=True);scores=np.stack([(hh[:,:,:,0]*u[:,:,:,0]).sum(2),(hh[:,:,:,0]*u[:,:,:,1]).sum(2),(hh[:,:,:,1]*u[:,:,:,2]+hh[:,:,:,2]*u[:,:,:,3]).sum(2),(hh[:,:,:,3]*u[:,:,:,4]).sum(2),(hh[:,:,:,4]*u[:,:,:,5]).sum(2)],axis=2);ii,jj,tt,vv=data.block(a,b)
  if len(ii):
   he=hh[ii,jj,tt];ys,yn=vv[:,1],vv[:,2];add=np.column_stack([he[:,0]*ys,he[:,0]*yn,he[:,1]*ys+he[:,2]*yn,he[:,3]*ys,he[:,4]*yn]);np.add.at(scores,(ii,jj),add)
  scores*=scale[None,None,:];core=scores.sum(1);cp=np.stack([scores[:,d.g==g].sum(1)for g in range(G)],axis=1);cap+=cp.sum(0);ss+=core.sum(0);teammeat+=core.T@core;intermeat+=cp.reshape(-1,5).T@cp.reshape(-1,5)
  for h in range(GH):
   ix=d.h[a:b]==h
   if ix.any():home[h]+=core[ix].sum(0);hc[h]+=cp[ix].sum(0)
  del x,fe,u,hh,scores,cp;gc.collect()
 def meat(x,n):return x.T@x*n/(n-1)
 vc=meat(cap,G);vh=meat(home,GH);vt=teammeat*I/(I-1);vint=intermeat*(I*G)/(I*G-1);vhi=meat(hc.reshape(-1,5),GH*G);cov={'capability':vc,'homefield':vh,'team':vt,'capability_team':vc+vt-vint,'homefield_capability':vc+vh-vhi}
 checks=[]
 for outcome,L in [('specialist',np.array([1,0,0,0,0.])),('no_specialist',np.array([0,1,0,0,0.])),('total',np.array([1,1,0,0,0.])),('signed',np.array([1,-1,0,0,0.]))]:
  ref=load(F/'results'/mp/'replication'/(outcome+'.json'));err=abs(float(L@points)-ref['sufficient_statistics']['beta'][5]);assert err<1e-10,(mp,outcome,err)
  for cl in SCHEMES:assert np.isclose(float(L@cov[cl]@L),ref['covariance'][cl][5][5],rtol=2e-5,atol=1e-13),(mp,outcome,cl)
  checks.append({'outcome':outcome,'primary_beta_error':err,'five_covariances_reproduced':True})
 assert np.max(np.abs(ss))<1e-7,ss
 save(o/'restricted_benchmark.json',{'primitive_names':['observed_bS','observed_bNo','common_gamma_delta_divided_1e5','unrestricted_gammaS','unrestricted_gammaNo'],'primitive_points':points.tolist(),'covariance':{k:q.tolist()for k,q in cov.items()},'linear_statistics':{k:q.tolist()for k,q in linear_statistics(pi).items()},'projection_loading':pi,'scores_by_path':cap.tolist(),'scores_by_home':home.tolist(),'path_labels':d.gnames.tolist(),'home_labels':d.hnames.tolist(),'N_per_outcome':N,'stacked_N':2*N,'cores':I,'capability_paths':G,'home_clusters':GH,'primary_FE_df':df,'CR1_factor_primary':f0,'CR1_factor_common_stacked':fc,'baseline_predictions_fixed_for_inference':True,'first_stage_uncertainty_not_included':True,'conditioning':'All original FE, all modifier-month slopes and nine other original AI interactions, separately by outcome','full_heterogeneous_projection_not_mean_rate_shortcut':True,'coefficient_and_covariance_checks':checks,'score_sum':ss.tolist(),'additional_score_pass_seconds':time.time()-start,'all_zero_candidates_retained':True,'historical_propensity_source':str(F/'baseline_propensities_crossfit'/mp/'diagnostics.json')});save(o/'sample.json',sample(p,sc));(o/'COMPLETE').write_text('Common and mode-specific proportional restrictions, observed and residual signed gradients; five schemes verified.');p.c.close();log('PROPORTIONAL BENCHMARK COMPLETE',mp)

if __name__=='__main__':ap=argparse.ArgumentParser();ap.add_argument('mapping');ap.add_argument('--home-only',action='store_true');a=ap.parse_args();main(a.mapping,a.home_only)
