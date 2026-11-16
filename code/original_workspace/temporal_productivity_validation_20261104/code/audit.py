from common import *

def main():
 c=pd.read_csv(H/'08_reproducibility/inputs/field_month_proxies.csv');rows=[]
 for protocol,q in c.groupby('protocol'):
  v=q[q.gain12_l6.notna()];rows.append({'protocol':protocol,'first_source_month':month(int(q.mo.min())),'last_source_month':month(int(q.mo.max())),'first_valid_gain':month(int(v.mo.min()))if len(v)else None,'last_valid_gain':month(int(v.mo.max()))if len(v)else None,'pre_ChatGPT_gain_observations':int((v.mo<=753).sum())})
 save(F/'diagnostics/historical_capability_availability.json',rows)
 save(F/'diagnostics/absorption.json',{'frozen_Distributed_with_pair_FE':{'status':'structurally_absorbed','proof':'A frozen core-destination value is constant over t; pair demeaning is identically zero.'},'Distributed_x_period_with_Distributed_x_each_month':{'status':'structurally_absorbed','proof':'Period interaction is a sum of the included distributed-month slope columns.'},'future_average_Home_or_Destination_x_Distributed_with_pair_FE':{'status':'structurally_absorbed','proof':'Both future-window average marker and frozen coverage are constant within core-destination.','static_average_marker_is_not_a_feasible_exact_primary_FE_placebo':True},'genuine_historical_AI_series_in_existing_primary_input':False,'replay_is_a_future_trajectory_sorting_marker_not_historical_exposure':True})
 # Only record source hashes; original data/results are never opened for writing.
 files=[OLD/'empirical_exploration_report.pdf',OLD/'H2_H3_full_population_theoretical_reassessment.pdf',H/'08_reproducibility/streaming_fe.py',H/'08_reproducibility/panel_api.py']
 files +=[OLD/'results/dyad/recent'/mp/'architecture'/('architecture_estimable__'+out+'.json')for mp in ['parent','direct']for out in OUTCOMES]
 save(F/'diagnostics/original_source_hashes.json',[{'path':str(p),'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()}for p in files]);log('SOURCE AND IDENTIFICATION AUDIT COMPLETE')

if __name__=='__main__':main()
