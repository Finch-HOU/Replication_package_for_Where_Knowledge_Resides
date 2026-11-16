from pathlib import Path
import duckdb,numpy as np,pandas as pd,json
H=Path(__file__).resolve().parent;O=H/'output';c=duckdb.connect();c.execute('SET threads=12')
sub=pd.read_csv(O/'domain_knowledge_input.csv');sf=sub.subfield_id.tolist();ix={s:i for i,s in enumerate(sf)};J=len(sf)
den=pd.read_parquet(O/'raw_citation_den_audit.parquet');flow=pd.read_parquet(O/'raw_citation_flow_audit.parquet')
dentotal=den.groupby('src_subfield').total_references.sum();earlyden=den[den.period=='2010-2014'].set_index('src_subfield')
sub['all_recorded_references']=sub.subfield_id.map(dentotal);sub['classified_reference_coverage']=sub.classified_citations/sub.all_recorded_references;sub['coverage_status']='classified references / all recorded references of classified citing works, 2010-2021';assert sub.classified_reference_coverage.between(0,1+1e-10).all()
sub.to_csv(O/'domain_knowledge_input.csv',index=False)
versions={};matrices={}
for tag,col in [('2010_2014_snapshot','n_citations'),('2010_2014_temporal','n_temporally_valid')]:
 C=np.zeros((J,J));C[flow.src_subfield.map(ix),flow.dst_subfield.map(ix)]=flow[col];n=C.sum(1);assert (n>0).all();R=C/n[:,None];K=1-np.diag(R);matrices[tag]=R
 z=sub[['subfield_id','candidate_subfield_name','candidate_field_name','candidate_domain_name']].copy();z['requirement_window']='2010-2014';z['temporal_filter']='cited publication_year <= citing publication_year' if tag.endswith('temporal') else 'snapshot linked classifications, no cited-date restriction';z['classified_citations']=n.astype('int64');z['knowledge_input']=K;z['all_recorded_references']=z.subfield_id.map(earlyden.total_references);z['classified_reference_coverage']=n/z.all_recorded_references;z['citing_works']=z.subfield_id.map(earlyden.citing_works);z['works_with_reference_list']=z.subfield_id.map(earlyden.works_with_reference_list)
 z.to_csv(O/f'domain_knowledge_input_{tag}.csv',index=False);np.save(O/f'reference_weights_{tag}.npy',R);versions[tag]=z
x=pd.read_parquet(O/'team_destination_knowledge_gap_pilot.parquet');teams=sorted(x.team_id.unique());ti={t:i for i,t in enumerate(teams)};c.register('pilot_teams',pd.DataFrame({'team_id':teams}));hist=c.execute("SELECT h.* FROM read_parquet('D:/OpenAlex/OpenAlex_Analysis/GPT Shock/output/derived/fixed_team/regression_panel/team_pub_first_subfield.parquet') h JOIN pilot_teams USING(team_id)").fetchdf();first=np.full((len(teams),J),32767,dtype='int16');first[hist.team_id.map(ti),hist.primary_subfield_id.map(ix)]=hist.first_pub_year
for tag,R in matrices.items():
 x['knowledge_gap_'+tag]=np.nan;x['knowledge_input_'+tag]=x.candidate_subfield_id.map(dict(zip(sf,1-np.diag(R))))
 for yr,idx in x.groupby('publication_year').groups.items():
  zz=x.loc[idx];a=zz.team_id.map(ti).to_numpy();j=zz.candidate_subfield_id.map(ix).to_numpy();G=1-(first<yr).astype(float)@R.T;x.loc[idx,'knowledge_gap_'+tag]=G[a,j]
 assert x['knowledge_gap_'+tag].between(-1e-12,1+1e-12).all()
x.to_parquet(O/'team_destination_knowledge_gap_validated_pilot.parquet',index=False)
w=x.sampling_weight.to_numpy();w=w/w.mean();codes=[pd.factorize(pd.MultiIndex.from_frame(x[cs]))[0] for cs in [['team_id','publication_year'],['candidate_subfield_id','publication_year']]]
def resid(Z):
 Z=np.array(Z,dtype=float,copy=True)
 for it in range(500):
  old=Z.copy()
  for co in codes:
   n=np.bincount(co,weights=w)
   for j in range(Z.shape[1]):Z[:,j]-=(np.bincount(co,weights=w*Z[:,j])/n)[co]
  if np.max(abs(Z-old))<1e-10:return Z
 raise RuntimeError('No convergence')
def cor(a,b):
 a=a-np.average(a,weights=w);b=b-np.average(b,weights=w);return float(np.sum(w*a*b)/np.sqrt(np.sum(w*a*a)*np.sum(w*b*b)))
rows=[]
for tag,gname,kname in [('2010_2021','knowledge_gap','knowledge_input')]+[(k,'knowledge_gap_'+k,'knowledge_input_'+k) for k in matrices]:
 g=x[gname].to_numpy();dist=x.expertise_distance.to_numpy();breadth=np.log1p(x.n_embodied_subfields.to_numpy());Z=resid(np.column_stack([g,dist,breadth]));gz=Z[:,0];D=Z[:,1:2];DD=Z[:,1:2];assert np.max(np.abs(Z[:,2]))<1e-8; e=gz-D@np.linalg.lstsq(D*np.sqrt(w[:,None]),gz*np.sqrt(w),rcond=None)[0];ee=gz-DD@np.linalg.lstsq(DD*np.sqrt(w[:,None]),gz*np.sqrt(w),rcond=None)[0]
 rows.append(dict(version=tag,N=len(x),corr_K_distance=cor(x[kname].to_numpy(),dist),corr_gap_distance=cor(g,dist),corr_gap_original=cor(g,x.knowledge_gap.to_numpy()),within_FE_corr=cor(gz,Z[:,1]),remaining_variance_after_FE_distance=float(np.sum(w*e**2)/np.sum(w*gz**2)),remaining_variance_after_FE_distance_breadth=float(np.sum(w*ee**2)/np.sum(w*gz**2)),gap_sd_after_FE_distance=float(np.sqrt(np.average(e**2,weights=w)))))
res=pd.DataFrame(rows);res.to_csv(O/'window_sensitivity_diagnostics.csv',index=False)
for tag,R in matrices.items():
 # Independent row-level recomputation, including older outcomes.
 for ii in np.random.default_rng(20261028).choice(len(x),100,replace=False):
  z=x.iloc[ii];j=ix[z.candidate_subfield_id];unknown=first[ti[z.team_id]]>=z.publication_year;assert abs(R[j,unknown].sum()-z['knowledge_gap_'+tag])<1e-12
strict=versions['2010_2014_temporal'];orig=sub.set_index('subfield_id');stricti=strict.set_index('subfield_id');kcorr=float(np.corrcoef(sub.knowledge_input,strict.knowledge_input)[0,1]);total_class=int(sub.classified_citations.sum());total_refs=int(sub.all_recorded_references.sum());total_early=int(strict.classified_citations.sum());early_refs=int(strict.all_recorded_references.sum());excluded=int(flow.n_citations.sum()-flow.n_temporally_valid.sum())
s=f'''# WP1 引用覆盖率与严格事前窗口核验

本轮恢复原始引用分母，并重建2010–2014年需求矩阵。保留上一轮原型，未修改AI exposure或估计机制回归。对同一5,000个团队、214,540条年度候选观测重算Knowledge Gap，以隔离需求窗口变化。

## 引用覆盖率

逐批扫描2,038个原始works parquet文件（约706GB），只读取所需字段。分母是具有primary subfield且施引年份在指定窗口内的作品，其referenced_works列表中全部记录的引用条目数；包括无法链接或无法分类的被引条目，不把无引用列表作品视为100%覆盖。该覆盖率不衡量OpenAlex遗漏的参考文献，也不覆盖无primary subfield的施引论文。

2010–2021年：已分类引用 {total_class:,} / 已记录引用 {total_refs:,}，总体覆盖率 **{total_class/total_refs:.2%}**。各领域覆盖率已写回domain_knowledge_input.csv，未再将其留空或默认为100%。分子沿用原有引用流，分母从同一原始works路径恢复；检查各领域分子不超过分母。

2010–2014年严格时间版本：有效分类引用 {total_early:,} / 已记录引用 {early_refs:,}，覆盖率 **{total_early/early_refs:.2%}**。另保留未限制被引年份的snapshot-compatible版本，以区分窗口变化与日期清理。严格版本只接受被引publication_year≤施引publication_year；因此排除了 {excluded:,} 条未来年份或年份缺失的已分类引用。年份检查不能排除同一年内的时间错序。

## 窗口与Gap稳定性

2010–2014年施引窗口严格早于所有2015–2025年选择年份。团队能力仍逐年采用first_pub_year < outcome_year。引用分类来自当前快照，不代表当时版本的分类法；严格事前的含义是论文年份过滤，不是历史数据库快照重建。

领域K在2010–2021与2010–2014严格矩阵间的等权Pearson相关性为 **{kcorr:.4f}**。

| 需求矩阵 | Corr(Gap, Distance) | Corr(Gap, 原始Gap) | FE后再控制Distance剩余方差 | 知识广度FE吸收后的核验 |
|---|---:|---:|---:|---:|
'''
for z in res.itertuples():s+=f'| {z.version} | {z.corr_gap_distance:.4f} | {z.corr_gap_original:.4f} | {z.remaining_variance_after_FE_distance:.2%} | {z.remaining_variance_after_FE_distance_breadth:.2%} |\n'
s+='''
所有行使用同一choice-set和inverse-sampling weights。FE为Team×Year与Destination×Year。剩余方差比例的分母是Gap的FE残差方差，而非原始总方差。团队历史知识广度在Team×Year内固定，其主效应已经由该FE吸收；最后一列只作一致性核验，并不是新增独立控制。

## 测量决策

后续年度分析可同时保留两个固定需求版本：2010–2021作为用户指定的Pre-AI结构；2010–2014严格时间版本作为覆盖整个2015–2025窗口的事前对照。正式回归应事先固定报告两者，不按显著性选择需求窗口。现有全窗口矩阵尚未应用被引年份限制，比较时同时使用2010–2014 snapshot-compatible版本识别日期清理的影响。

交付包括恢复覆盖率的domain_knowledge_input.csv、两种早期窗口domain_knowledge_input CSV及reference_weights NPY、raw_citation_den_audit.parquet、raw_citation_flow_audit.parquet、含三种Gap口径的validated pilot parquet，以及window_sensitivity_diagnostics.csv。raw_audit_parts保留每64个文件的计数与文件大小/mtime记录，可断点续跑。

重算的两种早期Gap各通过100条随机记录直接公式核验。当前完成WP1测量原型及上述两项缺口的核验；团队Gap仍是5,000团队原型，不是全量笛卡尔组合。WP3仍需先完成WP2的数字任务需求数据，然后分别、联合估计并报告事前趋势。本轮没有证据足以声称AI改变了选择规则。
'''
(O/'WP1_Window_and_Coverage_Report.md').write_text(s,encoding='utf-8');print(res.to_string(index=False));print('COVERAGE',total_class/total_refs,'EARLY',total_early/early_refs,'K CORR',kcorr,flush=True)
