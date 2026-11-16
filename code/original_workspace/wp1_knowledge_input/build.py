from pathlib import Path
import duckdb,numpy as np,pandas as pd,json,hashlib
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;O=H/'output';B=Path('D:/OpenAlex/OpenAlex_Analysis/GPT Shock/output/derived/fixed_team');V=B/'entry_distance_v2_parallel'
c=duckdb.connect();c.execute('SET threads=16');c.execute("SET memory_limit='24GB'")
flow=pd.read_parquet(V/'pregpt_subfield_citation_flow.parquet');sub=pd.read_parquet(V/'subfield_universe.parquet').sort_values('subfield_id');sf=sub.subfield_id.tolist();ix={v:i for i,v in enumerate(sf)};J=len(sf)
C=np.zeros((J,J));C[flow.src_subfield.map(ix),flow.dst_subfield.map(ix)]=flow.n_citations
n=C.sum(1);assert (n>0).all();R=C/n[:,None];K=1-np.diag(R)
assert np.allclose(R.sum(1),1) and ((K>=0)&(K<=1)).all()
domain=sub.copy();domain['classified_citations']=n.astype('int64');domain['external_classified_citations']=(n-np.diag(C)).astype('int64');domain['knowledge_input']=K;domain['classified_reference_coverage']=np.nan;domain['coverage_status']='denominator unavailable in aggregated flow; not assumed 100 percent';domain['requirement_window']='2010-2021';domain.to_csv(O/'domain_knowledge_input.csv',index=False)
flow['reference_weight']=flow.n_citations/flow.src_subfield.map(dict(zip(sf,n)));flow.to_parquet(O/'domain_knowledge_requirement_matrix.parquet',index=False);np.save(O/'reference_weights.npy',R)
print('DOMAIN',J,'classified citations',int(n.sum()),flush=True)
# Sample teams, not individual alternatives: preserve complete annual sampled choice strata.
panel=str(V/'entry_distance_panel.parquet')
c.execute('CREATE TEMP TABLE sample_teams AS SELECT team_id FROM (SELECT DISTINCT team_id FROM read_parquet(?)) ORDER BY md5(team_id) LIMIT 5000',[panel])
x=c.execute('SELECT p.* FROM read_parquet(?) p JOIN sample_teams USING(team_id)',[panel]).fetchdf();teams=sorted(x.team_id.unique());ti={v:i for i,v in enumerate(teams)}
hist=c.execute('SELECT h.* FROM read_parquet(?) h JOIN sample_teams USING(team_id)',[str(B/'regression_panel/team_pub_first_subfield.parquet')]).fetchdf()
assert not hist.duplicated(['team_id','primary_subfield_id']).any()
first=np.full((len(teams),J),32767,dtype='int16');first[hist.team_id.map(ti),hist.primary_subfield_id.map(ix)]=hist.first_pub_year
x['knowledge_input']=x.candidate_subfield_id.map(dict(zip(sf,K)));x['knowledge_gap']=np.nan
for yr,idx in x.groupby('publication_year').groups.items():
 z=x.loc[idx];a=z.team_id.map(ti).to_numpy();j=z.candidate_subfield_id.map(ix).to_numpy();known=(first<int(yr));gap=1-known.astype(float)@R.T
 x.loc[idx,'knowledge_gap']=gap[a,j]
 assert np.array_equal(known.sum(1)[a],z.n_embodied_subfields.to_numpy()),'baseline history mismatch'
 assert not known[a,j].any(),'candidate already known'
assert x.knowledge_gap.between(-1e-10,1+1e-10).all()
x.knowledge_gap=x.knowledge_gap.clip(0,1)
x.to_parquet(O/'team_destination_knowledge_gap_pilot.parquet',index=False)
print('GAP',len(x),'teams',len(teams),flush=True)
w=x.sampling_weight.to_numpy();w=w/w.mean()
def corr(a,b):
 a=np.asarray(a);b=np.asarray(b);a=a-np.average(a,weights=w);b=b-np.average(b,weights=w)
 return float(np.sum(w*a*b)/np.sqrt(np.sum(w*a*a)*np.sum(w*b*b)))
a=x.expertise_distance.to_numpy();g=x.knowledge_gap.to_numpy();k=x.knowledge_input.to_numpy()
# Weighted alternating projections absorb both fixed-effect groups jointly for Gap and Distance.
groups=[pd.factorize(pd.MultiIndex.from_frame(x[['team_id','publication_year']]))[0],pd.factorize(pd.MultiIndex.from_frame(x[['candidate_subfield_id','publication_year']]))[0]]
Z=np.column_stack([g,a]);Z-=np.average(Z,axis=0,weights=w)
for it in range(1000):
 old=Z.copy()
 for codes in groups:
  den=np.bincount(codes,weights=w)
  for j in range(2):Z[:,j]-=(np.bincount(codes,weights=w*Z[:,j])/den)[codes]
 delta=float(np.max(np.abs(Z-old)))
 if delta<1e-10:break
assert delta<1e-8
slope=float(np.sum(w*Z[:,0]*Z[:,1])/np.sum(w*Z[:,1]**2));res=Z[:,0]-slope*Z[:,1]
variation={'N':len(x),'teams':len(teams),'selection':'5000 teams with smallest MD5(team_id); all their annual baseline rows retained','weighted_corr_input_distance':corr(k,a),'weighted_corr_gap_distance':corr(g,a),'weighted_corr_input_gap':corr(k,g),'within_FE_corr_gap_distance':corr(Z[:,0],Z[:,1]),'gap_SD_raw_weighted':float(np.sqrt(np.average((g-np.average(g,weights=w))**2,weights=w))),'gap_SD_after_FE':float(np.sqrt(np.average(Z[:,0]**2,weights=w))),'gap_SD_after_FE_and_distance':float(np.sqrt(np.average(res**2,weights=w))),'remaining_gap_variance_share_after_distance_given_FE':float(np.sum(w*res**2)/np.sum(w*Z[:,0]**2)),'alternating_projection_iterations':it+1,'maximum_iteration_change':delta,'missing_gap':int(x.knowledge_gap.isna().sum()),'baseline_history_count_check':'all rows match n_embodied_subfields','target_known_check':'no candidate is already in pre-year member publication set'}
(O/'diagnostics.json').write_text(json.dumps(variation,indent=2),encoding='utf-8')
x['gap_residual_FE_distance']=res
x.sort_values('gap_residual_FE_distance').groupby('candidate_subfield_id').agg(N=('entry','size'),gap_mean=('knowledge_gap','mean'),distance_mean=('expertise_distance','mean'),gap_residual_sd=('gap_residual_FE_distance','std')).to_csv(O/'destination_diagnostics.csv')
# Same destination and year, similar distances, contrasting knowledge coverage.
x['distance_bin']=(x.expertise_distance/.02).astype(int)
examples=[]
for _,z in x.groupby(['candidate_subfield_id','publication_year','distance_bin']):
 if len(z)<8:continue
 lo=z.loc[z.knowledge_gap.idxmin()];hi=z.loc[z.knowledge_gap.idxmax()]
 if hi.knowledge_gap-lo.knowledge_gap>=.15:
  for label,row in [('lower_gap',lo),('higher_gap',hi)]:
   examples.append(dict(pair=len(examples)//2,comparison=label,team_id=row.team_id,year=row.publication_year,destination=row.candidate_subfield_id,distance=row.expertise_distance,gap=row.knowledge_gap,input=row.knowledge_input,prior_subfields=row.n_embodied_subfields))
 if len(examples)>=40:break
pd.DataFrame(examples).to_csv(O/'contrasting_team_examples.csv',index=False)
domain.sort_values('knowledge_input').to_csv(O/'domain_ranking.csv',index=False)
fig,ax=plt.subplots(1,3,figsize=(13,3.8));ax[0].hist(K,bins=25,color='#345779');ax[0].set(xlabel='Domain external knowledge input',ylabel='Subfields');ax[1].hist(g,bins=35,weights=w,color='#345779');ax[1].set(xlabel='Team-destination knowledge gap',ylabel='Weighted choice opportunities');ax[2].hexbin(a,g,gridsize=45,mincnt=1,cmap='Blues');ax[2].set(xlabel='Expertise distance',ylabel='Knowledge gap');fig.tight_layout();fig.savefig(O/'measurement_diagnostics.png',dpi=180);plt.close(fig)
# Independent formula checks on source counts and sampled rows.
rng=np.random.default_rng(20261027)
for ii in rng.choice(len(x),100,replace=False):
 row=x.iloc[ii];j=ix[row.candidate_subfield_id];known=first[ti[row.team_id]]<row.publication_year
 assert abs(float(C[j,~known].sum()/n[j])-row.knowledge_gap)<1e-12
print(json.dumps(variation,indent=2),flush=True)
manifest={'sources':[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p.stat().st_size<1000000 else None) for p in [V/'pregpt_subfield_citation_flow.parquet',V/'entry_distance_panel.parquet',B/'regression_panel/team_pub_first_subfield.parquet']], 'limitations':['Citation coverage denominator not retained in source flow.','2010-2021 requirement weights use future information for early annual outcomes; strictly earlier matrix pending.','Pilot sampling covers 5000 existing baseline teams, not full population.'],'formula_checks':100,'status':'measurement prototype completed'}
(O/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
