"""Exact balanced-risk FE regression using block projections and sparse outcomes.

Zero candidate cells enter every normal equation and covariance calculation.
No observed-publication approximation and no candidate subsampling is used.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='8'
import numpy as np,time,json
from pathlib import Path
from scipy.stats import t as td

class SparseBatch:
 def __init__(self,i,j,t,values,names,I,J,T,mask):
  i=np.asarray(i,dtype=np.int64);j=np.asarray(j,dtype=np.int64);t=np.asarray(t,dtype=np.int64);v=np.asarray(values,dtype=float)
  if v.ndim==1:v=v[:,None]
  ok=mask[i,j];i,j,t,v=i[ok],j[ok],t[ok],v[ok];key=(i*J+j)*T+t;order=np.argsort(key,kind='stable');key=key[order];v=v[order];starts=np.r_[0,np.flatnonzero(np.diff(key))+1] if len(key) else np.array([],int)
  self.key=key[starts];self.v=np.add.reduceat(v,starts,axis=0) if len(starts) else np.empty((0,v.shape[1]));self.i=self.key//(J*T);self.j=(self.key//T)%J;self.t=self.key%T;self.names=names;self.M=v.shape[1];self.I=I;self.J=J;self.T=T
  self.sum=self.v.sum(0);self.sq=(self.v*self.v).sum(0);self.nonzero=np.count_nonzero(self.v,axis=0)
  pair=self.key//T;ix=np.r_[0,np.flatnonzero(np.diff(pair))+1] if len(pair) else np.array([],int);pv=np.add.reduceat(self.v,ix,axis=0) if len(ix) else np.empty((0,self.M));self.pair_sq=(pv*pv).sum(0);self.pair_cross=pv.T@pv
 def block(self,start,end):
  a=np.searchsorted(self.i,start);b=np.searchsorted(self.i,end);return self.i[a:b]-start,self.j[a:b],self.t[a:b],self.v[a:b]

class StreamingDesign:
 def __init__(self,mask,fields,home,names,factory,controls=None,pair=True,chunk=2048):
  self.mask=np.asarray(mask,bool);self.I,self.J=self.mask.shape;assert self.mask.any(1).all(),'Remove only rows without any eligible candidates and pass an explicit index map.'
  self.den=self.mask.sum(1);self.fields=np.asarray(fields);self.home=np.asarray(home);self.names=names;self.K=len(names);self.factory=factory;self.control=controls;self.pair=pair;self.chunk=chunk;self.gnames,self.g=np.unique(fields,return_inverse=True);self.hnames,self.h=np.unique(home,return_inverse=True);self.G=len(self.gnames);self.GH=len(self.hnames);self._built=False
 def blocks(self):
  for a in range(0,self.I,self.chunk):yield a,min(a+self.chunk,self.I)
 def ctr(self,a,b):
  if self.control is None:return np.zeros((b-a,self.J,0))
  z=np.asarray(self.control(a,b),float);return z[:,:,None] if z.ndim==2 else z
 def build(self):
  start=time.time();z=self.factory(0,1);self.T=z.shape[2];self.q=self.ctr(0,1).shape[-1];self.P=self.J+self.q;P,T,K=self.P,self.T,self.K;S=np.zeros((P,P));rhs=np.zeros((P,T,K));self.rowX=np.empty((self.I,T,K));self.modmean=np.empty((self.I,self.q))
  for a,b in self.blocks():
   m=self.mask[a:b].astype(float);den=self.den[a:b];mods=self.ctr(a,b);weighted=mods*m[:,:,None];rm=weighted.sum(1)/den[:,None];self.modmean[a:b]=rm;S[:self.J,:self.J]+=np.diag(m.sum(0))-m.T@(m/den[:,None]);cross=weighted.sum(0)-m.T@rm;S[:self.J,self.J:]+=cross;S[self.J:,:self.J]+=cross.T
   if self.q:S[self.J:,self.J:]+=np.einsum('ijq,ijr->qr',weighted,mods,optimize=True)-(rm*den[:,None]).T@rm
   x=np.array(self.factory(a,b),dtype=float,copy=True)
   if self.pair:x-=x.mean(2,keepdims=True)
   x*=m[:,:,None,None];rx=x.sum(1)/den[:,None,None];self.rowX[a:b]=rx;x-=rx[:,None,:,:]*m[:,:,None,None];rhs[:self.J]+=x.sum(0)
   if self.q:rhs[self.J:]+=np.einsum('ijq,ijtk->qtk',weighted,x,optimize=True)
  ev,U=np.linalg.eigh((S+S.T)/2);tol=max(ev.max(),1)*1e-12;good=ev>tol;self.Sinv=(U[:,good]/ev[good])@U[:,good].T;self.gammaX=(self.Sinv@rhs.reshape(P,-1)).reshape(P,T,K);self.spatial_rank=int(good.sum());self._built=True;self.design_seconds=time.time()-start
 def residual_x(self,a,b):
  m=self.mask[a:b].astype(float);mods=self.ctr(a,b);den=self.den[a:b];x=np.array(self.factory(a,b),dtype=float,copy=True)
  if self.pair:x-=x.mean(2,keepdims=True)
  gy=self.gammaX;fit=gy[:self.J][None,:,:,:]+np.einsum('ijq,qtk->ijtk',mods,gy[self.J:],optimize=True);rowfit=np.einsum('ij,jtk->itk',m,gy[:self.J],optimize=True)/den[:,None,None]+np.einsum('iq,qtk->itk',self.modmean[a:b],gy[self.J:],optimize=True);x-=fit+self.rowX[a:b,None,:,:]-rowfit[:,None,:,:];x*=m[:,:,None,None];return x
 def fit_many(self,data,folder,prefix,save_team=False,keep_state=False,cap_only=False,save_joint=False):
  if not self._built:self.build()
  I,J,T,K,M=self.I,self.J,self.T,self.K,data.M;assert data.I==I and data.J==J and data.T==T;folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);start=time.time();Q=np.zeros((K,K));Qg=np.zeros((self.G,K,K));xy=np.zeros((K,M));rowY=np.zeros((I,T,M));np.add.at(rowY,(data.i,data.t),data.v)
  if self.pair:rowY-=rowY.mean(1,keepdims=True)
  rowY/=self.den[:,None,None];rhsY=np.zeros((self.P,T,M));np.add.at(rhsY[:J],(data.j,data.t),data.v)
  if self.pair:rhsY[:J]-=rhsY[:J].mean(1,keepdims=True)
  cap_raw=np.zeros((self.G,K,M)) if cap_only else None;cap_row=np.zeros((self.G,K,M)) if cap_only else None;cap_W=np.zeros((self.P,T,K,self.G)) if cap_only else None
  for a,b in self.blocks():
   x=self.residual_x(a,b);xf=x.reshape(-1,K);Q+=xf.T@xf
   for g in range(self.G):
    ix=self.g==g;v=x[:,ix].reshape(-1,K);Qg[g]+=v.T@v
    if cap_only:
     xs=x[:,ix].sum(1);cap_row[g]+=np.einsum('itk,itm->km',xs,rowY[a:b],optimize=True);cap_W[:J][ix,:,:,g]+=x[:,ix].sum(0);cap_W[:J,:,:,g]-=(self.mask[a:b].astype(float).T@(xs/self.den[a:b,None,None]).reshape(b-a,-1)).reshape(J,T,K)
     if self.q:
      mods0=self.ctr(a,b);cap_W[J:,:,:,g]+=np.einsum('ijq,ijtk->qtk',mods0[:,ix],x[:,ix],optimize=True)-np.einsum('iq,itk->qtk',self.modmean[a:b],xs,optimize=True)
   ii,jj,tt,v=data.block(a,b)
   if len(ii):
    xy+=x[ii,jj,tt].T@v
    if cap_only:np.add.at(cap_raw,self.g[jj],np.einsum('ek,em->ekm',x[ii,jj,tt],v))
   m=self.mask[a:b].astype(float);rhsY[:J]-=np.einsum('ij,itm->jtm',m,rowY[a:b],optimize=True)
   if self.q:
    mods=self.ctr(a,b);rhsY[J:]-=np.einsum('iq,itm->qtm',self.modmean[a:b]*self.den[a:b,None],rowY[a:b],optimize=True)
    if len(ii):np.add.at(rhsY[J:].transpose(1,0,2),tt,np.einsum('eq,em->eqm',mods[ii,jj],v))
  # The previous scatter adds each event's q-by-M block into its month.
  # Pair demeaning commutes with the static control weights.
  if self.pair and self.q:
   # Only the raw event term needs centering; rowY is already time centered.
   rhsY[J:]-=rhsY[J:].mean(1,keepdims=True)
  rank=int(np.linalg.matrix_rank(Q));pairs=int(self.mask.sum());N=pairs*T;meta={'N':N,'teams':I,'baseline_teams':I,'pairs':pairs,'destinations':J,'periods':T,'names':self.names,'rank':rank,'condition':float(np.linalg.cond(Q)),'pair_FE':self.pair,'expertise_freeze':getattr(self,'expertise_freeze',2021),'engine':'Exact block Schur projection; sparse outcomes; complete zero-inclusive risks','spatial_rank':self.spatial_rank,'design_seconds':self.design_seconds}
  if rank<K:
   for name in data.names:self._write(folder/(prefix+name+'.json'),{'meta':{**meta,'id':prefix+name},'status':'rank_deficient'})
   return
  inv=np.linalg.inv(Q);beta=inv@xy;gammaY=(self.Sinv@rhsY.reshape(self.P,-1)).reshape(self.P,T,M);within=data.sq-(data.pair_sq/T if self.pair else 0)-np.einsum('i,itm,itm->m',self.den,rowY,rowY,optimize=True)-np.einsum('ptm,ptm->m',rhsY,gammaY,optimize=True);rss=within-np.einsum('km,km->m',beta,xy);sst=data.sq-data.sum*data.sum/N
  cap=np.zeros((self.G,K,M));hom=np.zeros((self.GH,K,M));hc=np.zeros((self.GH,self.G,K,M));meat_team=np.zeros((M,K,K));meat_inter=np.zeros((M,K,K));teamfile=None
  joint_team=np.zeros((M*K,M*K)) if save_joint else None;joint_inter=np.zeros((M*K,M*K)) if save_joint else None
  if cap_only:cap=cap_raw-cap_row-np.einsum('ptkg,ptm->gkm',cap_W,gammaY,optimize=True)-np.einsum('gkl,lm->gkm',Qg,beta,optimize=True)
  if save_team:teamfile=np.lib.format.open_memmap(folder/(prefix+'team_scores.npy'),mode='w+',dtype='float64',shape=(I,K,M))
  for a,b in ([] if cap_only else self.blocks()):
   x=self.residual_x(a,b);m=self.mask[a:b].astype(float);mods=self.ctr(a,b);gy=gammaY;rowfit=np.einsum('ij,jtm->itm',m,gy[:J],optimize=True)/self.den[a:b,None,None]+np.einsum('iq,qtm->itm',self.modmean[a:b],gy[J:],optimize=True);fitted=gy[:J][None,:,:,:]+np.einsum('ijq,qtm->ijtm',mods,gy[J:],optimize=True)+(rowY[a:b]-rowfit)[:,None,:,:]+np.einsum('ijtk,km->ijtm',x,beta,optimize=True);scores=-np.einsum('ijtk,ijtm->ijkm',x,fitted,optimize=True);ii,jj,tt,v=data.block(a,b)
   if len(ii):np.add.at(scores,(ii,jj),np.einsum('ek,em->ekm',x[ii,jj,tt],v))
   st=scores.sum(1);sc=np.stack([scores[:,self.g==g].sum(1) for g in range(self.G)],axis=1);cap+=sc.sum(0);meat_team+=np.einsum('ikm,ilm->mkl',st,st,optimize=True);meat_inter+=np.einsum('igkm,iglm->mkl',sc,sc,optimize=True)
   if save_joint:
    ff=st.transpose(0,2,1).reshape(-1,M*K);joint_team+=ff.T@ff;ff=sc.transpose(0,1,3,2).reshape(-1,M*K);joint_inter+=ff.T@ff
   for h in range(self.GH):
    ix=self.h[a:b]==h
    if ix.any():hom[h]+=st[ix].sum(0);hc[h]+=sc[ix].sum(0)
   if teamfile is not None:teamfile[a:b]=st
  if teamfile is not None:teamfile.flush()
  df_fe=pairs+(I+J-1+self.q)*(T-1) if self.pair else (I+J-1+self.q)*T;factor=(N-1)/max(N-df_fe-K,1)
  def sandwich(meat,ng):return inv@meat@inv*ng/(ng-1)*factor if ng>1 else np.full((K,K),np.nan)
  outputs=[]
  for n,name in enumerate(data.names):
   mid=prefix+name;mm={**meta,'id':mid,'mean_outcome':float(data.sum[n]/N),'nonzero_cells':int(data.nonzero[n]),'R2':float(1-rss[n]/sst[n]) if sst[n]>0 else None,'within_R2':float(1-rss[n]/within[n]) if within[n]>0 else None,'seconds':time.time()-start,'df_FE':df_fe,'CR1_factor':factor};vc=sandwich(cap[:,:,n].T@cap[:,:,n],self.G);vh=sandwich(hom[:,:,n].T@hom[:,:,n],self.GH);vt=sandwich(meat_team[n],I);vct=vc+vt-sandwich(meat_inter[n],I*self.G);xx=hc[:,:,:,n].reshape(-1,K);vch=vc+vh-sandwich(xx.T@xx,self.G*self.GH);covs={'capability':vc,'homefield':vh,'team':vt,'capability_team':vct,'homefield_capability':vch};cs=[]
   for label,V in covs.items():
    ng=self.GH if label=='homefield' else I if label=='team' else self.G;crit=td.ppf(.975,ng-1)
    for k,term in enumerate(self.names):
     variance=float(V[k,k]);se=float(np.sqrt(variance)) if variance>0 and np.isfinite(variance) else None;cs.append({'term':term,'cluster':label,'b':float(beta[k,n]),'se':se,'p':float(2*td.sf(abs(beta[k,n]/se),ng-1)) if se else None,'lo':float(beta[k,n]-crit*se) if se else None,'hi':float(beta[k,n]+crit*se) if se else None,'clusters':ng,'variance_valid':se is not None})
   if cap_only:cs=[c for c in cs if c['cluster']=='capability'];covs={'capability':vc};mm['covariance_scope']='Capability covariance only, exact FE moment identity; other covariances available in primary fits.'
   result={'status':'estimated' if data.nonzero[n] else 'zero_outcome','meta':mm,'coefficients':cs,'covariance':{k:v.tolist() for k,v in covs.items()},'sufficient_statistics':{'Q':Q.tolist(),'Q_by_path':Qg.tolist(),'path_labels':self.gnames.tolist(),'home_labels':self.hnames.tolist(),'scores_by_path':cap[:,:,n].tolist(),'scores_by_home':None if cap_only else hom[:,:,n].tolist(),'beta':beta[:,n].tolist()}};self._write(folder/(mid+'.json'),result);outputs.append(result)
  if keep_state:self.fit_state={'data':data,'rowY':rowY,'gammaY':gammaY,'beta':beta,'Q':Q,'inv':inv,'Qg':Qg,'cap':cap,'factor':factor,'df_fe':df_fe}
  if save_joint:
   invfull=np.kron(np.eye(M),inv)
   def joint_cov(sc,ng,meat=False):
    ff=sc if meat else sc.transpose(0,2,1).reshape(-1,M*K);mm=ff if meat else ff.T@ff
    return invfull@mm@invfull*ng/(ng-1)*factor
   vc=joint_cov(cap,self.G);vh=joint_cov(hom,self.GH);vt=joint_cov(joint_team,I,True);vint=joint_cov(joint_inter,I*self.G,True);vhi=joint_cov(hc.reshape(self.GH*self.G,K,M),self.GH*self.G);rawcross=data.v.T@data.v;withincross=rawcross-(data.pair_cross/T if self.pair else 0)-np.einsum('i,itm,itn->mn',self.den,rowY,rowY,optimize=True)-rhsY.reshape(-1,M).T@gammaY.reshape(-1,M);rsscross=withincross-beta.T@Q@beta;sstcross=rawcross-np.outer(data.sum,data.sum)/N
   np.savez_compressed(folder/(prefix+'joint_covariances.npz'),capability=vc,homefield=vh,team=vt,capability_team=vc+vt-vint,homefield_capability=vc+vh-vhi,beta=beta.T,raw_cross=rawcross,within_cross=withincross,rss_cross=rsscross,sst_cross=sstcross,outcome_sum=data.sum,outcome_values=data.v)
   self._write(folder/(prefix+'joint_metadata.json'),{'outcomes':data.names,'regressors':self.names,'indexing':'outcome-major then regressor','N':N,'teams':I,'paths':self.G,'home_clusters':self.GH,'df_FE':df_fe,'CR1_factor':factor,'meta_template':outputs[0]['meta'],'linear_outcome_contrasts_are_covariance_aware':True})
  return outputs
 @staticmethod
 def _write(path,value):
  t=path.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(t,path)
