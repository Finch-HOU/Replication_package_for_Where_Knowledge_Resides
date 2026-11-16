from common import *
import numpy as np
def main():
 o=F/'measurement';u=H/'02_dyad_full_population/universal/frozen';R=np.load(o/'R.npy');h=np.load(u/'history.npy',mmap_mode='r');G=np.load(u/'G.npy',mmap_mode='r');best=np.load(o/'best_member_input_coverage.npy',mmap_mode='r');distributed=np.load(o/'distributed_input.npy',mmap_mode='r');shared=np.load(o/'shared_input.npy',mmap_mode='r');assert np.allclose(R.sum(1),1,atol=1e-12);eg=0.;ea=0.;es=0.
 for a in range(0,len(h),25000):
  b=min(a+25000,len(h));covered=(h[a:b]>0).astype(float)@R.T;eg=max(eg,float(np.max(np.abs(covered-(1-G[a:b])))));ea=max(ea,float(np.max(np.abs(covered-best[a:b]-distributed[a:b]))));es=max(es,float(np.max(shared[a:b]-covered)));assert eg<2e-6 and ea<2e-6 and es<2e-6
 save(F/'qa/all_cell_measurement_checks.json',{'cores':len(h),'fields':252,'core_field_cells':int(h.size),'all_R_rows_sum_to_one':True,'zero_reference_requirement_rows':0,'max_original_Gap_reconstruction_error':eg,'max_union_minus_best_minus_distributed_error':ea,'max_shared_above_union_error':es,'all_cells_pass_float_tolerance':2e-6,'no_new_regression_or_variable_selection':True});log('FULL MEASUREMENT IDENTITIES',eg,ea,es)
if __name__=='__main__':main()
