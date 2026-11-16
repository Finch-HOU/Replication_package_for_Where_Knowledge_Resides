from common import *
from restriction_tools import restrictions

def main():
 rng=np.random.default_rng(20261104);n=1200;k=12;D=np.column_stack([np.ones(n),rng.normal(size=(n,3))]);raw=rng.normal(size=(n,k));X=raw-D@np.linalg.lstsq(D,raw,rcond=None)[0];ys=rng.normal(size=(n,2));y=ys-D@np.linalg.lstsq(D,ys,rcond=None)[0];Q=X.T@X;xy=X.T@y;plan=restrictions(Q,xy);W=plan['W'];nw=len(W);Z=np.zeros((2*n,2*nw+1));Z[:n,:nw]=X[:,W];Z[n:,nw:2*nw]=X[:,W];Z[:n,-1]=X[:,-2];Z[n:,-1]=X[:,-1];yy=np.r_[y[:,0],y[:,1]];b=np.linalg.lstsq(Z,yy,rcond=None)[0];err=float(np.max(np.abs(b-plan['common_parameter_vector'])));assert err<1e-12
 # A perturbation's independently refitted stacked coefficient equals the
 # stored influence derivative, including the two different design columns.
 dy=rng.normal(size=(n,2));bp=np.linalg.lstsq(Z,yy+np.r_[dy[:,0],dy[:,1]],rcond=None)[0];pred=float(np.sum((X@plan['influence_vectors'][:,1])*dy[:,0]+(X@plan['influence_vectors'][:,2])*dy[:,1]));assert abs((bp[-1]-b[-1])-pred)<1e-12
 # The implied signed primary slope is the exact projection, even when
 # heterogeneous propensities are correlated with the other controls.
 S=plan['restricted_betas'];signed=X@(S[:,2]-S[:,3]);fitted=np.linalg.lstsq(X[:,:10],signed,rcond=None)[0];expected=plan['primitive_points'][2]*plan['signed_projection_loading'];assert abs(fitted[5]-expected)<1e-12
 save(F/'diagnostics/restriction_validation.json',{'independent_stacked_OLS_max_error':err,'influence_derivative_matches_refit':True,'heterogeneous_signed_projection_identity':True,'seed':20261104});log('INDEPENDENT RESTRICTION VALIDATION PASSED')

if __name__=='__main__':main()
