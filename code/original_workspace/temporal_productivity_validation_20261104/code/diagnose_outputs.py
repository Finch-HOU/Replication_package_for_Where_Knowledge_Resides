"""Publish existing support, scale and covariance diagnostics; no new fits."""
from common import *

def main():
 covs=[];scales=[];support=[];concentration=[]
 for mp in ['parent','direct']:
  sp=load(F/'results'/mp/'replication/sample.json')
  for k,v in sp['standardization'].items():scales.append({'mapping':mp,'variable':k,**v,'same_original_frozen_risk_standardization':True})
  for st in ['replication','gradient','trajectory_marker','proportional','home_proportional']:
   rt=F/'results'/mp/st
   if not(rt/'COMPLETE').exists():continue
   for p in sorted(rt.glob('*.json')):
    r=load(p)
    if 'meta'not in r or 'status'not in r:continue
    md=r['meta'];name=p.stem
    win='2022-01-01/2022-10-31'if'pre_'in name else'2025-07-01/2026-04-30'if'current_'in name else'2022-01-01/2022-10-31 + 2025-07-01/2026-04-30'if'pooled_change'in name else'2025-07-01/2026-05-31'
    support.append({'mapping':mp,'stage':st,'model':name,'window':win,'N':md.get('N'),'periods':md.get('periods'),'events':md.get('nonzero_cells'),'mean_outcome':md.get('mean_outcome'),'events_per_100k':md.get('nonzero_cells',np.nan)/md.get('N',np.nan)*1e5,'R2':md.get('R2'),'within_R2':md.get('within_R2'),'design_rank':md.get('rank'),'condition':md.get('condition'),'source':str(p)})
    for cl,v in r.get('covariance',{}).items():
     v=np.array(v);diag=np.diag(v);ev=np.linalg.eigvalsh((v+v.T)/2)
     covs.append({'mapping':mp,'stage':st,'model':name,'scheme':cl,'minimum_variance':float(diag.min()),'negative_diagonal_count':int((diag<0).sum()),'minimum_eigenvalue':float(ev.min()),'maximum_eigenvalue':float(ev.max()),'symmetry_error':float(np.max(np.abs(v-v.T))),'PSD_repair':False,**covariance_diagnostic(v),'source':str(p)})
   if st in ['proportional','home_proportional']:
    r=load(rt/'restricted_benchmark.json');s=np.array(r['scores_by_path']);labels=r['path_labels'];lin=r['linear_statistics']
    for stat in ['signed_observed','signed_proportional','signed_residual']:
     v=s@np.array(lin[stat]);share=v*v/max(np.sum(v*v),1e-300)
     for name,score,q in zip(labels,v,share):concentration.append({'mapping':mp,'stage':st,'statistic':stat,'capability_path':name,'cluster_score':score,'squared_score_share':q,'descriptive_effective_score_count':1/np.sum(share*share),'not_an_independent_treatment_count':True,'full_paths_retained':True})
 for name,rows in [('LPM_covariance_diagnostics',covs),('original_standardization_scales',scales),('model_support_and_economic_units',support),('capability_score_concentration',concentration)]:pd.DataFrame(rows).to_csv(F/'diagnostics'/(name+'.csv'),index=False)
 save(F/'diagnostics/output_diagnostic_receipt.json',{'all_negative_variances_and_eigenvalues_preserved':True,'no_path_excluded':True,'no_additional_regression':True,'scale_definitions_unchanged':True,'tables_in_illustrative_0_10_gain_units':True})
 log('OUTPUT DIAGNOSTICS PUBLISHED',len(covs),'covariance matrices')

if __name__=='__main__':main()
