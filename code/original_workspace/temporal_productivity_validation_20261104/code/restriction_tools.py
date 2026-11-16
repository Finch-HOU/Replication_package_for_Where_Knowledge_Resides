import numpy as np

def restrictions(Q,xy,focal=5):
 """Two outcome equations, separate nuisance slopes, common proportional slope.

 Q: residualized basis Gram [original ten X, 1e5*p0S*Xf, 1e5*p0No*Xf].
 xy: basis cross-product with [Specialist, No Specialist].
 """
 K=len(Q);B=K-2;W=[k for k in range(B)if k!=focal];nw=len(W);baseinv=np.linalg.inv(Q[:B,:B]);bo=baseinv@xy[:B];a0=np.r_[baseinv[focal],0.,0.]
 fs=np.eye(K)[:,W+[B]];fn=np.eye(K)[:,W+[B+1]];ivs=np.linalg.inv(fs.T@Q@fs);ivn=np.linalg.inv(fn.T@Q@fn);bs=fs@ivs@fs.T@xy[:,0];bn=fn@ivn@fn.T@xy[:,1]
 ps=np.zeros((K,2*nw+1));pn=ps.copy();ps[W,np.arange(nw)]=1;pn[W,nw+np.arange(nw)]=1;ps[B,-1]=1;pn[B+1,-1]=1;hh=ps.T@Q@ps+pn.T@Q@pn;ih=np.linalg.inv(hh);theta=ih@(ps.T@xy[:,0]+pn.T@xy[:,1]);common_s=ps@theta;common_n=pn@theta
 b0s=np.r_[bo[:,0],0.,0.];b0n=np.r_[bo[:,1],0.,0.];betas=np.column_stack([b0s,b0n,common_s,common_n,bs,bn]);A=np.column_stack([a0,ih[-1]@ps.T,ih[-1]@pn.T,ivs[-1]@fs.T,ivn[-1]@fn.T]);points=np.array([bo[focal,0],bo[focal,1],theta[-1],bs[B],bn[B+1]]);pi=float(baseinv[focal]@(Q[:B,B]-Q[:B,B+1]));return {'restricted_betas':betas,'influence_vectors':A,'primitive_points':points,'signed_projection_loading':pi,'common_parameter_vector':theta,'common_information':hh,'W':W}

def linear_statistics(pi):
 return {'specialist':np.array([1,0,0,0,0.]),'no_specialist':np.array([0,1,0,0,0.]),'total':np.array([1,1,0,0,0.]),'signed_observed':np.array([1,-1,0,0,0.]),'signed_proportional':np.array([0,0,pi,0,0.]),'signed_residual':np.array([1,-1,-pi,0,0.]),'common_proportional_delta':np.array([0,0,1e5,0,0.]),'specialist_proportional_delta':np.array([0,0,0,1e5,0.]),'no_specialist_proportional_delta':np.array([0,0,0,0,1e5]),'proportional_delta_difference_S_minus_No':np.array([0,0,0,1e5,-1e5])}
