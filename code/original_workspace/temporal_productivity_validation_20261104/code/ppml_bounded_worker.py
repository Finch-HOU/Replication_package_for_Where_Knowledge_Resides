"""Independently run an already-declared numerically equivalent PPML model."""
from common import *
from ppml_accelerated import prepare_do
import argparse,subprocess
from ppml_runtime import run_official,read_statuses

def main(mp,regime):
 source=F/'relative_rate'/mp/regime;o=F/'relative_rate_accelerated'/mp/regime
 if(o/'COMPLETE').exists():return
 prepare_do(o,source,regime)
 dd=o/'estimate.do';dd.write_text(dd.read_text().replace('maxiterations(500)','maxiterations(500) verbose(1)'),encoding='ascii')
 runtime=run_official(o,2700);statuses=read_statuses(o,runtime)
 save(o/'support.json',load(source/'support.json'))
 save(o/'execution.json',{**runtime,'statuses':statuses,'same_specification_as_original_namespace':True,'nuisance_preconditioned_by_invertible_within_mode_month_rotation':True,'source_support':str(source/'support.dta'),'covariance_plan':'Five one-way fits; unmodified inclusion-exclusion; no PSD repair','CR1_note':'Stata nested-FE finite-sample adjustments, not original Python full-FE CR1'})
 (o/'COMPLETE').write_text('Equivalent official PPML attempt ended; all exported results and failures preserved.')

if __name__=='__main__':a=argparse.ArgumentParser();a.add_argument('mapping');a.add_argument('regime');x=a.parse_args();main(x.mapping,x.regime)
