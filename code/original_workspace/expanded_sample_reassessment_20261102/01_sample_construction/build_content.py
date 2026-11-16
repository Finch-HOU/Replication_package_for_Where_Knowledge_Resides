"""Full shared reference histories without materializing the candidate panel.

Histories include all classified valid record types. Qualifying joint outcomes
are unchanged. Reference lists are deduplicated and dates are strictly prior
in publication year. All references remain in the recorded-coverage audit.
"""
from pathlib import Path
import sys,time,gc,os,subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from config import *
import duckdb,numpy as np,pandas as pd,pyarrow.parquet as pq
from numba import njit

@njit
def populate_author_bits(bits,ai,ast,js,jst):
 for w in range(len(ast)-1):
  for a in range(ast[w],ast[w+1]):
   author=ai[a]
   for x in range(jst[w],jst[w+1]):
    for y in range(x+1,jst[w+1]):
     code=min(js[x],js[y])*252+max(js[x],js[y]);bits[author,code//8]|=np.uint8(1<<(code%8))

@njit
def core_union(out,bits,g,aid):
 for n in range(len(g)):
  for k in range(bits.shape[1]):out[g[n],k]|=bits[aid[n],k]

@njit
def novelty_metrics(p,g,rj,n,ns,hn,dn,bits,N):
 z=np.zeros((N,4))
 for a in range(len(p)):
  for b in range(a+1,len(p)):
   if p[b]!=p[a]:break
   weight=ns[a]*ns[b]
   if weight==0:continue
   code=min(rj[a],rj[b])*252+max(rj[a],rj[b]);new=(bits[g[a],code//8]&(1<<(code%8)))==0;z[p[a],0]+=weight
   if new:z[p[a],1]+=weight
   if (hn[a]>0 and dn[b]>0) or (hn[b]>0 and dn[a]>0):
    z[p[a],2]+=weight;z[p[a],3]+=weight*new
 return z

def main():
 u=H/'02_dyad_full_population/universal';o=u/'content';o.mkdir(exist_ok=True);c=duckdb.connect(str(o/'content.duckdb'));c.execute('SET threads=32');c.execute("SET memory_limit='180GB'");c.execute('PRAGMA disable_progress_bar');c.execute(f"SET temp_directory='{q(o/'spill')}'")
 def stage(name,sql):
  if c.execute('SELECT count(*) FROM information_schema.tables WHERE table_name=?',[name]).fetchone()[0]:return
  log('CONTENT BUILD',name);t=time.time();c.execute('CREATE TABLE '+name+' AS '+sql);log('CONTENT DONE',name,c.execute('SELECT count(*) FROM '+name).fetchone()[0],round(time.time()-t,1))
 def export(name,namefile=None):
  p=o/((namefile or name)+'.parquet')
  if not p.exists():c.execute(f"COPY {name} TO '{q(p)}' (FORMAT PARQUET,COMPRESSION ZSTD)")
 glob=q(H/'01_sample_construction/global_catalog/*.parquet');sub=q(u/'frozen/sub.parquet');authors=q(u/'core_authors.parquet')
 # No future-activity selection: only computational reuse is restricted to
 # authors already in the complete prospectively eligible populations.
 stage('author_index',f"SELECT author_id,(row_number() OVER(ORDER BY author_id)-1)::INTEGER a FROM read_parquet('{authors}')")
 stage('member_publications',f"WITH w AS (SELECT work_id,y,UNNEST(ids) author_id FROM read_parquet('{glob}') WHERE sf IS NOT NULL AND y<=2021) SELECT w.*,a.a FROM w JOIN author_index a USING(author_id)");export('member_publications')
 stage('pubs',f"SELECT w.*,(row_number() OVER(ORDER BY g,work_id)-1)::INTEGER p FROM read_parquet('{q(u/'analysis_works.parquet')}') w WHERE w.y<=2021 OR (w.mo=first_mo AND w.mo BETWEEN 786 AND 796)");export('pubs')
 stage('wanted',"SELECT DISTINCT work_id FROM member_publications UNION SELECT DISTINCT work_id FROM pubs")
 # No title, DOI or author-list columns are needed in historical reference work
 # storage; avoiding their materialization is an exact I/O optimization.
 stage('member_works',f"SELECT w.work_id,w.y,w.refs FROM read_parquet('{glob}') w SEMI JOIN wanted USING(work_id)")
 stage('valid_author_lookup',f"SELECT w.work_id refid,w.ids ref_authors FROM read_parquet('{glob}') w JOIN read_parquet('{sub}') s ON w.sf=try_cast(regexp_extract(s.subfield_id,'([0-9]+)$',1) AS INTEGER) WHERE w.y IS NOT NULL")
 if not (o/'ORIGINAL_REFERENCE_LOOKUP_COMPLETE').exists():
  if c.execute("SELECT count(*) FROM information_schema.tables WHERE table_name='lookup'").fetchone()[0]:c.execute('ALTER TABLE lookup RENAME TO superseded_valid_reference_lookup')
  stage('lookup',f"SELECT try_cast(regexp_extract(l.work_id,'([0-9]+)$',1) AS UBIGINT) refid,l.publication_year ry,s.j rj,a.ref_authors FROM read_parquet('{q(BASE/'regression_panel/work_domain_lookup.parquet')}') l JOIN read_parquet('{sub}') s ON s.subfield_id=l.primary_subfield_id LEFT JOIN valid_author_lookup a ON a.refid=try_cast(regexp_extract(l.work_id,'([0-9]+)$',1) AS UBIGINT)")
  (o/'ORIGINAL_REFERENCE_LOOKUP_COMPLETE').write_text('Original BASE lookup scope, including classified cited records flagged retracted/paratext exactly as original; own-publication histories still exclude invalid records.')
 # Persist grouped counts, not a potentially multi-billion-row reference edge
 # table. The denominator includes only classified, strictly prior references.
 if not (o/'fast_counts/FAST_COUNTS_COMPLETE').exists():
  c.close();subprocess.run([sys.executable,'-X','utf8',str(H/'01_sample_construction/fast_reference_counts.py')],check=True,cwd=H);c=duckdb.connect(str(o/'content.duckdb'));c.execute('SET threads=32');c.execute("SET memory_limit='180GB'");c.execute('PRAGMA disable_progress_bar')
 stage('counts',f"SELECT * FROM read_parquet('{q(o/'fast_counts/parts/*.parquet')}')");export('counts')
 # Only analysis publications need own-reference attribution. Historical
 # reference profiles above still cover every pre-2022 member publication.
 # Frozen original initial calibration means are inputs, so computing metrics
 # for all established pre-2019 joint output is unnecessary. Retain every
 # legacy first entry, every focal-window output, and prospective2022 firsts.
 if not (o/'ANALYSIS_TARGETS_V2_COMPLETE').exists():
  renamed=c.execute("SELECT count(*) FROM information_schema.tables WHERE table_name='pubs_initial_allpre2022'").fetchone()[0]>0
  if not renamed:stage('pubs_v2',f"SELECT w.*,(row_number() OVER(ORDER BY g,work_id)-1)::INTEGER p FROM read_parquet('{q(u/'analysis_works.parquet')}') w WHERE (w.y BETWEEN 2019 AND 2021 AND w.mo=w.first_mo) OR w.mo BETWEEN 786 AND 796 OR (w.mo BETWEEN 744 AND 753 AND w.mo=w.first_mo)")
  for name in ['pub_refs','core_counts','alignment','expertise_alignment','coverage']:
   if c.execute('SELECT count(*) FROM information_schema.tables WHERE table_name=?',[name]).fetchone()[0]:c.execute('DROP TABLE '+name)
   path=o/(name+'.parquet')
   if path.exists():os.replace(path,o/(name+'_superseded_target_scope.parquet'))
  if not renamed:c.execute('ALTER TABLE pubs RENAME TO pubs_initial_allpre2022');c.execute('ALTER TABLE pubs_v2 RENAME TO pubs')
  stage('new_analysis_work_ids','SELECT DISTINCT p.work_id FROM pubs p ANTI JOIN member_works w USING(work_id)')
  stage('supplement_analysis_works',f"SELECT w.work_id,w.y,w.refs FROM read_parquet('{glob}') w SEMI JOIN new_analysis_work_ids USING(work_id)");c.execute('INSERT INTO member_works SELECT s.* FROM supplement_analysis_works s ANTI JOIN member_works w USING(work_id)');export('pubs','analysis_targets')
  write(o/'analysis_target_scope.json',{'preferred_legacy_firsts':'2019--2021,2018 histories','focal_output':'Every qualifying joint output July2025--May2026, not just firsts','prospective_calibration':'Jan--Oct2022 firsts,2021 histories','all_pre2022_member_reference_histories_unchanged':True,'original_initial_superseded_calibration':'Frozen30k numerical benchmarks retained; no expanded look-ahead calibration claimed','publication_rows':c.execute('SELECT count(*) FROM pubs').fetchone()[0],'added_post2021_metadata_works':c.execute('SELECT count(*) FROM new_analysis_work_ids').fetchone()[0]});(o/'ANALYSIS_TARGETS_V2_COMPLETE').write_text('Complete preferred outcome scopes and histories; no eligible core/candidate removed.')
 stage('audit',"SELECT w.work_id,w.y,len(w.refs) n_recorded,len(list_distinct(w.refs)) n_distinct,coalesce(z.n_valid,0) n_valid FROM member_works w LEFT JOIN (SELECT work_id,sum(n) n_valid FROM counts GROUP BY 1) z USING(work_id)");export('audit')
 stage('pub_refs',f"WITH target AS (SELECT DISTINCT work_id FROM pubs), w AS (SELECT work_id,y,UNNEST(list_distinct(refs)) refid FROM member_works SEMI JOIN target USING(work_id)) SELECT w.work_id,l.* FROM w JOIN lookup l USING(refid) WHERE l.ry<w.y")
 stage('core_counts',f"SELECT p.p,p.g,p.work_id,r.rj,count(*)::INTEGER n_refs,count_if(coalesce(list_has_any(r.ref_authors,b.ids),false))::INTEGER n_self_prior,count_if(coalesce(list_has_any(r.ref_authors,b.ids),false) AND r.ry<=2021)::INTEGER n_self21,count_if(coalesce(list_has_any(r.ref_authors,b.ids),false) AND r.ry<=2018)::INTEGER n_self18,count_if(r.rj=p.j AND NOT coalesce(list_has_any(r.ref_authors,b.ids),false))::INTEGER n_dest_nonself FROM pubs p JOIN pub_refs r USING(work_id) JOIN read_parquet('{q(u/'cores.parquet')}') b USING(g) GROUP BY 1,2,3,4");export('core_counts')
 # Current pre-publication-year expertise of each author is not frozen baseline
 # expertise; it is used only to describe the cited-field coverage on a paper.
 stage('alignment',f"WITH papers AS (SELECT p.p,p.g,p.work_id,p.y,UNNEST(j.ids) aid FROM pubs p JOIN read_parquet('{q(u/'joint_works.parquet')}') j USING(g,work_id) WHERE p.mo BETWEEN 786 AND 796 AND p.mo=p.first_mo), a AS (SELECT a.*,s.j FROM read_parquet('{q(H/'01_sample_construction/author_field.parquet')}') a JOIN read_parquet('{sub}') s ON a.sf=try_cast(regexp_extract(s.subfield_id,'([0-9]+)$',1) AS INTEGER)) SELECT p.p,r.rj,bool_or(coalesce(a.first_y<p.y,false) AND list_contains(b.ids,p.aid)) core_covers,bool_or(coalesce(a.first_y<p.y,false) AND NOT list_contains(b.ids,p.aid)) additional_covers FROM papers p JOIN core_counts r USING(p) JOIN read_parquet('{q(u/'cores.parquet')}') b ON b.g=p.g LEFT JOIN a ON a.author_id=p.aid AND a.j=r.rj GROUP BY 1,2")
 stage('expertise_alignment',"SELECT r.p,sum((r.n_refs-r.n_self_prior)*CAST(a.core_covers AND NOT a.additional_covers AS INTEGER)) core_only,sum((r.n_refs-r.n_self_prior)*CAST(NOT a.core_covers AND a.additional_covers AS INTEGER)) additional_only,sum((r.n_refs-r.n_self_prior)*CAST(a.core_covers AND a.additional_covers AS INTEGER)) shared,sum((r.n_refs-r.n_self_prior)*CAST(NOT a.core_covers AND NOT a.additional_covers AS INTEGER)) neither,sum(r.n_refs-r.n_self_prior) n_nonself FROM core_counts r JOIN alignment a USING(p,rj) GROUP BY 1");export('expertise_alignment')
 stage('coverage',"SELECT m.g,count(*) baseline_member_works,count_if(a.n_valid>=5) baseline_works_refs5,sum(a.n_valid) baseline_prior_refs,sum(a.n_distinct) baseline_recorded_refs FROM (SELECT DISTINCT b.g,m.work_id FROM member_publications m JOIN read_parquet('"+q(u/'members.parquet')+"') b USING(author_id)) m JOIN audit a USING(work_id) GROUP BY 1");export('coverage')
 log('CONTENT SQL COMPLETE');(o/'CONTENT_SQL_COMPLETE').write_text('Historical reference counts and publication-level self/field coverage built.')
 nA=c.execute('SELECT count(*) FROM author_index').fetchone()[0];nU=pq.read_metadata(u/'cores.parquet').num_rows;B=(252*252+7)//8
 # Two versions of histories: legacy calibration has 2018 information only.
 for cutoff in [2018,2021]:
  marker=o/f'METRICS_{cutoff}_COMPLETE'
  if marker.exists():continue
  bitpath=o/f'author_bits_{cutoff}.npy'
  corepath=o/f'core_bits_{cutoff}.npy';need_core=not ((o/f'CORE_BITS_{cutoff}_COMPLETE').exists() and corepath.exists())
  if need_core and not ((o/f'AUTHOR_BITS_{cutoff}_COMPLETE').exists() and bitpath.exists()):
   bits=np.lib.format.open_memmap(bitpath,mode='w+',dtype=np.uint8,shape=(nA,B));bits[:]=0
   query=f"WITH a AS (SELECT work_id,list(a ORDER BY a) aa FROM member_publications WHERE y<={cutoff} GROUP BY 1), r AS (SELECT work_id,list(rj ORDER BY rj) jj FROM counts GROUP BY 1) SELECT aa,jj FROM a JOIN r USING(work_id)"
   count=0
   for batch in c.execute(query).fetch_record_batch(50000):
    al=batch.column('aa').combine_chunks() if hasattr(batch.column('aa'),'combine_chunks') else batch.column('aa');jl=batch.column('jj');populate_author_bits(bits,al.values.to_numpy(),al.offsets.to_numpy(),jl.values.to_numpy(),jl.offsets.to_numpy());count+=batch.num_rows
    if count%1000000==0:log('CONTENT HISTORICAL WORKS',cutoff,count)
   bits.flush();(o/f'AUTHOR_BITS_{cutoff}_COMPLETE').write_text(str(count));del bits;gc.collect()
  if need_core:
   bits=np.load(bitpath,mmap_mode='r');cb=np.lib.format.open_memmap(corepath,mode='w+',dtype=np.uint8,shape=(nU,B));cb[:]=0
   for batch in c.execute(f"SELECT m.g,a.a FROM read_parquet('{q(u/'members.parquet')}') m JOIN author_index a USING(author_id) ORDER BY m.g").fetch_record_batch(200000):core_union(cb,bits,batch.column('g').to_numpy(),batch.column('a').to_numpy())
   cb.flush();(o/f'CORE_BITS_{cutoff}_COMPLETE').write_text('Exact union of member prior observed co-citation pairs.');del bits,cb;gc.collect()
  bits=np.load(corepath,mmap_mode='r');metrics=c.execute('SELECT p.*,len(w.refs) n_recorded,len(list_distinct(w.refs)) n_distinct,coalesce(a.n_valid,0) n_valid FROM pubs p JOIN member_works w USING(work_id) LEFT JOIN (SELECT p,sum(n_refs) n_valid FROM core_counts GROUP BY 1) a USING(p) ORDER BY p.p').fetchdf();rc=pd.read_parquet(o/'core_counts.parquet').sort_values(['p','rj']);hist=np.load(u/'frozen'/('history2018.npy' if cutoff==2018 else 'history.npy'),mmap_mode='r');ns=(rc.n_refs-rc.n_self_prior).to_numpy();dest=metrics.j.to_numpy()[rc.p.to_numpy()];home=ns*((hist[rc.g.to_numpy(),rc.rj.to_numpy()]>0)&(rc.rj.to_numpy()!=dest));zz=novelty_metrics(rc.p.to_numpy(),rc.g.to_numpy(),rc.rj.to_numpy(),rc.n_refs.to_numpy(),ns,home,rc.n_dest_nonself.to_numpy(),bits,len(metrics));hn=np.bincount(rc.p,weights=home,minlength=len(metrics));dn=np.bincount(rc.p,weights=rc.n_dest_nonself,minlength=len(metrics));sn=np.bincount(rc.p,weights=rc['n_self18' if cutoff==2018 else 'n_self21'],minlength=len(metrics));den=metrics.n_valid.to_numpy(float);pair=den*(den-1)/2
  for label,num,d in [('home_nonself_share',hn,den),('self_pre_share',sn,den),('mix_mass',zz[:,2],pair),('new_pair_mass',zz[:,1],pair),('new_home_dest_mass',zz[:,3],pair)]:metrics[label]=np.divide(num,d,out=np.full(len(metrics),np.nan),where=d>0)
  metrics['home_dest_coexist']=np.where(den>0,((hn>0)&(dn>0)).astype(float),np.nan);metrics['content_observed']=den>=5;metrics['prior_reference_coverage']=np.divide(den,metrics.n_distinct.fillna(0).to_numpy(float),out=np.full(len(metrics),np.nan),where=metrics.n_distinct.fillna(0).to_numpy()>0);metrics['freeze']=cutoff
  metrics.to_parquet(o/f'publication_knowledge_{cutoff}.parquet',index=False,compression='zstd');marker.write_text('Strict historical reference and novelty metrics complete.');del metrics,rc,bits;gc.collect();log('CONTENT METRICS COMPLETE',cutoff)
 c.close();(o/'CONTENT_COMPLETE').write_text('Full historical reference processing complete.');log('CONTENT COMPLETE')
if __name__=='__main__':main()
