from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 o=F/'figures';o.mkdir(exist_ok=True);plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
 d=pd.read_csv(F/'data/temporal_comparison_all_covariances.csv');q=d[d.stage.eq('trajectory_marker')&d.term.eq('Distributed_Home')&d.cluster.eq('capability')];outs=['specialist','no_specialist','total','signed'];labels=['Added specialist','No added specialist','Total','S minus no'];fig,aa=plt.subplots(1,2,figsize=(12,6));fig.suptitle('Comparable monthly outcomes and the same future capability-path marker',fontsize=15)
 for ax,mp in zip(aa,['parent','direct']):
  for per,off,col in [('pre',-.09,'#95546D'),('current',.09,'#295B88')]:
   for k,out in enumerate(outs):
    r=q[q.mapping.eq(mp)&q.model.eq(per+'_'+out)].iloc[0];ax.errorbar(r.b*1e4,k+off,xerr=[[max(0,(r.b-r.lo)*1e4)],[max(0,(r.hi-r.b)*1e4)]],fmt='o',color=col,capsize=3,label='Historical ten months'if per=='pre'and k==0 else'Current ten months'if k==0 else None)
  ax.set_title(mp.title());ax.set_yticks(range(4),labels);ax.invert_yaxis();ax.axvline(0,color='#999',lw=.8);ax.set_xlabel('Incidence /100k / SD of U /0.10 gain')
 fig.legend(*aa[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.095),ncol=3,frameon=False,fontsize=10);fig.text(.025,.015,'95% intervals, ten capability paths. Historical: Jan-Oct2022; current: Jul2025-Apr2026.\nThe post-period trajectory is replayed by relative month; it is a sorting marker, not historical AI exposure.',fontsize=10);fig.tight_layout(rect=(0,.15,1,.94));fig.savefig(o/'01_comparable_temporal_marker.png',dpi=180);fig.savefig(o/'01_comparable_temporal_marker.svg');plt.close(fig)
 b=pd.read_csv(F/'data/heterogeneous_proportional_benchmark.csv');b=b[b.cluster.eq('capability')];fig,aa=plt.subplots(1,2,figsize=(12,6));fig.suptitle('Observed gradients versus the heterogeneous common-proportional benchmark',fontsize=15)
 for ax,mp in zip(aa,['parent','direct']):
  sub=b[b.mapping.eq(mp)].set_index('statistic')
  for tag,off,col,lab in [('observed',-.16,'#295B88','Observed'),('proportional',0,'#D38B3E','Implied common response'),('residual',.16,'#376F5A','Observed minus implied')]:
   for k,out in enumerate(outs):
    key='signed_observed'if out=='signed'and tag=='observed'else out if tag=='observed'else out+'_'+tag;r=sub.loc[key];ax.errorbar(r.b*1e4,k+off,xerr=[[max(0,(r.b-r.lo)*1e4)],[max(0,(r.hi-r.b)*1e4)]],fmt='o',color=col,capsize=3,label=lab if k==0 else None)
  ax.set_title(mp.title());ax.set_yticks(range(4),labels);ax.invert_yaxis();ax.axvline(0,color='#999',lw=.8);ax.set_xlabel('Incidence /100k / SD of U /0.10 gain')
 fig.legend(*aa[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.095),ncol=3,frameon=False,fontsize=10);fig.text(.025,.015,'Current 11-month complete C risks. 95% conditional intervals, ten capability paths.\nHistorical probabilities are core-cross-fitted; first-stage estimation uncertainty is not included. Contrasts use joint covariance.',fontsize=10);fig.tight_layout(rect=(0,.15,1,.94));fig.savefig(o/'02_proportional_benchmark.png',dpi=180);fig.savefig(o/'02_proportional_benchmark.svg');plt.close(fig);log('TWO SCIENTIFIC DIAGNOSTIC FIGURES BUILT')

def general_benchmark():
 a=pd.read_csv(F/'data/heterogeneous_proportional_benchmark.csv');b=pd.read_csv(F/'data/home_general_productivity_benchmark.csv')
 fig,axs=plt.subplots(1,2,figsize=(12,5));fig.suptitle('Do common proportional opportunity responses account for the signed gradient?',fontsize=14)
 labels=['Observed signed','Common focal-X response','Common Home-only response','Residual after Home-only']
 for ax,mp in zip(axs,['parent','direct']):
  aa=a[a.mapping.eq(mp)&a.cluster.eq('capability')].set_index('statistic');bb=b[b.mapping.eq(mp)&b.cluster.eq('capability')].set_index('statistic')
  rr=[aa.loc['signed_observed'],aa.loc['signed_proportional'],bb.loc['signed_proportional'],bb.loc['signed_residual']]
  for k,(r,col)in enumerate(zip(rr,['#295B88','#D38B3E','#95546D','#376F5A'])):ax.errorbar(r.b*1e4,k,xerr=[[(r.b-r.lo)*1e4],[(r.hi-r.b)*1e4]],fmt='o',color=col,capsize=3)
  ax.set_yticks(range(4),labels,fontsize=10.5);ax.invert_yaxis();ax.set_title(mp.title());ax.axvline(0,color='#999',lw=.8);ax.set_xlabel('Signed events per 100,000\n1 SD of U x 0.10 HomeGain',fontsize=10.5)
 fig.text(.02,.025,'Current 11-month complete risks. Capability-path t intervals, 10 paths.\nHistorical probabilities are cross-fitted and treated as fixed; all contrasts use joint covariance.',fontsize=9)
 fig.tight_layout(rect=(0,.1,1,.93))
 for suffix in ['png','svg']:fig.savefig(F/'figures'/('03_two_productivity_benchmarks.'+suffix),dpi=180)
 plt.close(fig)

if __name__=='__main__':main();general_benchmark()
