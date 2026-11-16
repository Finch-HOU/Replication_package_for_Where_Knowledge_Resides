"""Collect the registered estimates, exact covariance contrasts and support figures."""
from engine import *
from placebo import CONVENTIONAL
from conditional_placebo import ID as CLASS_C_ID, SEED as CLASS_C_SEED
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UNIFIED = {
    "table3_exact": "Verified November 7 benchmark",
    "a1_unified": "Retain all Table 3 terms and add directional applicability",
    "a2_unified_access": "Also control for frozen specialist access, reconciled to archived fit",
    "a3_nonlinear_relatedness": "Allow curvature in gap, distance and directional applicability",
}
OUTCOME_MATRIX = np.array([[1, 1, 0, 1], [1, 0, 1, -1]])
FE = "Core-Destination, Core-Month, Destination-Month, listed modifier-by-month slopes"


def collect_fit(meta, result, coefficients, strengths, path_rows, slope_rows):
    mp, model = meta["mapping"], meta["specification"]
    names = meta["terms"]
    k_count = len(names)
    betas, influences = result["beta"], result["influence"]
    cv = result["covariance"]
    inv, Qg, xyg = result["inverse"], result["Q_by_path"], result["xy_by_path"]
    q, xy = Qg.sum(0), xyg.sum(0)
    f = [k for k,n in enumerate(names) if n.endswith(FOCAL)]
    cv_ev = np.linalg.eigvalsh(cv)
    cv_tol = np.max(np.abs(cv_ev))*1e-10
    cov_rank = int((cv_ev>cv_tol).sum())
    descriptors = {k: meta.get(k, "") for k in ["mapping", "specification", "family", "placebo_id", "scheme", "seed", "variant"]}
    for j, out in enumerate(OUTCOMES):
        for k, term in enumerate(names):
            r = stats(betas[k,j], influences[:,k,j])
            coefficients.append(dict(**descriptors, outcome=out, term=term, N=meta["N"], events=meta["events"][j],
                                     b_x10000=10000*r["b"], se_x10000=10000*r["se"] if r["se"] is not None else None,
                                     R2=meta["R2"][j], within_R2=meta["within_R2"][j], covariance="capability CR1 t(9)",
                                     covariance_psd=cv_ev[0]>=-cv_tol, covariance_rank=cov_rank, **r))
    partial_corr = None
    if len(f) == 2:
        partial_corr = float(-inv[f[0], f[1]] / np.sqrt(inv[f[0], f[0]]*inv[f[1], f[1]]))
    for k in f:
        w = inv[:,k]/inv[k,k]
        info = w @ Qg @ w
        share = info/info.sum()
        strengths.append(dict(**descriptors, term=names[k], partial_SD=float(np.sqrt(1/(inv[k,k]*meta["N"]))),
                              effective_information_paths=float(1/(share@share)), max_path_share=float(share.max()),
                              conditional_actual_placebo_corr=partial_corr, rank=meta["rank"], columns=meta["columns"],
                              condition_normalized=meta["condition_normalized"], min_joint_covariance_eigenvalue=cv_ev[0],
                              joint_covariance_rank=cov_rank, joint_covariance_psd=cv_ev[0]>=-cv_tol))
        for g, label in enumerate(meta["paths"]):
            remainder = q-Qg[g]
            d = np.sqrt(np.maximum(np.diag(remainder),1e-300))
            nr = remainder/np.outer(d,d)
            valid = np.linalg.eigvalsh((nr+nr.T)/2)[0] > 1e-12
            deleted = np.linalg.solve(nr,(xy-xyg[g])/d[:,None])[:k_count] / d[:k_count,None] @ OUTCOME_MATRIX if valid else None
            for j, out in enumerate(OUTCOMES):
                path_rows.append(dict(mapping=mp, specification=model, term=names[k], path=label, outcome=out,
                                      information_share=share[g], b_x10000=betas[k,j]*10000,
                                      score_influence_x10000=influences[g,k,j]*10000,
                                      fixed_projection_omit_path_b_x10000=deleted[k,j]*10000 if valid else None,
                                      deletion_rank_valid=valid))
    if meta["family"] != "placebo":
        b_all = inv @ xy
        s_all = (inv @ (xyg - Qg @ b_all)) @ OUTCOME_MATRIX
        s_all *= np.sqrt(meta["CR1"]*10/9)
        basis = helmert(11).T
        for j, out in enumerate(OUTCOMES):
            for l, name in enumerate(meta["slopes"]):
                sl = slice(k_count+l*10, k_count+(l+1)*10)
                beta_time = basis @ (b_all[sl] @ OUTCOME_MATRIX[:,j])
                s_time = s_all[:,sl,j] @ basis.T
                for k, month in enumerate(pd.period_range("2025-07", "2026-05", freq="M")):
                    slope_rows.append(dict(mapping=mp, specification=model, outcome=out, modifier=name, month=str(month),
                                           parameterization="time-mean-zero", **stats(beta_time[k],s_time[:,k])))


def adjusted(p):
    p = p.dropna()
    order = p.sort_values()
    return (order * np.arange(len(order), 0, -1)).cummax().clip(upper=1)


def adjust(df, groups, target, mask=None):
    df[target] = np.nan
    subset = df if mask is None else df[mask]
    for _, d in subset.groupby(groups):
        v = adjusted(d.p)
        df.loc[v.index, target] = v


def magnitude_figure(fits):
    out = HERE / "figures"
    rows = []
    for mp in EXPECTED:
        if (mp, "a1_unified") not in fits:
            continue
        path = HERE / "diagnostics" / mp / "observed_comparable_candidates.csv"
        if not path.exists():
            continue
        sample = pd.read_csv(path)
        meta, r = fits[mp,"a1_unified"]
        k = meta["terms"].index(FOCAL)
        for quantile, ex in sample.exposure_product.quantile([.25,.5,.75]).items():
            for j, outcome in [(2, "no_specialist"), (0, "total")]:
                rows.append(dict(mapping=mp, outcome=outcome, support_exposure_quantile=quantile,
                                 exposure_product=ex, eligible_sample_cores=len(sample),
                                 **stats(r["beta"][k,j]*ex*1e5,r["influence"][:,k,j]*ex*1e5)))
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(HERE / "diagnostics/observed_economic_magnitudes.csv", index=False)
    fig, ax = plt.subplots(figsize=(8,4.4), layout="constrained")
    offsets = {("parent","no_specialist"):-.027, ("parent","total"):-.009,
               ("direct","no_specialist"):.009, ("direct","total"):.027}
    for key, d in frame.groupby(["mapping","outcome"]):
        ax.errorbar(d.support_exposure_quantile+offsets[key],d.b,yerr=[d.b-d.lo,d.hi-d.b],fmt="o",capsize=3,
                    label=f"{key[0].title()} / {key[1].replace('_',' ').title()}")
    ax.axhline(0,color=".65",lw=1)
    ax.set(xticks=[.25,.5,.75],xticklabels=["25th","Median","75th"],xlabel="Observed exposure contrast among eligible matched candidate pairs",
           ylabel="U contribution to entries per 100,000 candidate-months")
    ax.legend(frameon=False,fontsize=9)
    fig.savefig(out / "observed_magnitudes.png",dpi=200)
    plt.close(fig)


def plots(a, b, contrasts, fits):
    folder = HERE / "figures"
    folder.mkdir(exist_ok=True)
    if len(a):
        fig, axes = plt.subplots(1,2,figsize=(10,4.5),sharey=True,layout="constrained")
        labels = ["Table 3","Applicability","Applicability + access","Nonlinear relatedness"]
        for ax, out in zip(axes,["no_specialist","total"]):
            for mp, off, color in [("parent",-.10,"#206d85"),("direct",.10,"#ad6235")]:
                d = a[(a.mapping==mp)&(a.outcome==out)&(a.term==FOCAL)].set_index("specification").reindex(list(UNIFIED))
                ax.errorbar(d.b*1e4,np.arange(4)+off,xerr=[(d.b-d.lo)*1e4,(d.hi-d.b)*1e4],
                            fmt="o",capsize=3,color=color,label=mp.title())
            ax.axvline(0,color=".55",lw=1)
            ax.set(title=out.replace("_"," ").title(),xlabel="HomeGain × U, coefficient × 10,000",
                   yticks=np.arange(4),yticklabels=labels)
        axes[0].invert_yaxis()
        axes[1].legend(frameon=False)
        fig.savefig(folder / "unified_coefficients.png", dpi=200)
        plt.close(fig)
    if len(b):
        fig, axes = plt.subplots(2,3,figsize=(13,6),sharex=True,layout="constrained")
        scheme_names = {"field_shuffle":"Field shuffle","matched":"Global match","class_c_matched":"Class C match"}
        for i,mp in enumerate(EXPECTED):
            for j,scheme in enumerate(["field_shuffle","matched","class_c_matched"]):
                ax = axes[i,j]
                d = contrasts[(contrasts.mapping==mp)&(contrasts.scheme==scheme)&(contrasts.outcome=="no_specialist")
                              &(contrasts.comparison=="separate_actual_minus_placebo")].sort_values("seed")
                y = np.arange(len(d))
                ax.scatter(d.placebo_b*1e4,y,label="Donor U",color="#ad6235",marker="x")
                ax.scatter(d.actual_b*1e4,y,label="Actual U",color="#206d85")
                for yi,row in enumerate(d.itertuples()):
                    ax.plot([row.actual_b*1e4,row.placebo_b*1e4],[yi,yi],color=".78",zorder=0)
                ax.axvline(0,color=".65",lw=1)
                ax.set(title=f"{mp.title()} / {scheme_names[scheme]}",
                       yticks=y,yticklabels=[int(x[-2:]) for x in d.placebo_id])
                if j==0:
                    ax.set_ylabel("Seed index")
        axes[0,1].legend(frameon=False)
        fig.supxlabel("No-Specialist HomeGain × U, coefficient × 10,000")
        fig.savefig(folder / "actual_placebo_distribution.png",dpi=200)
        plt.close(fig)
    magnitude_figure(fits)


def main():
    freeze()
    fits, registry, arows, brows, strength, path_rows, month_rows = {}, [], [], [], [], [], []
    code_hash = "|".join(f"{k}:{digest(HERE/k)}" for k in ["engine.py","run.py","placebo.py","conditional_placebo.py"])
    data_version = digest(HERE/"frozen_table3/manifest.json")
    for mp in EXPECTED:
        assignment = HERE / "diagnostics" / mp / "assignments.json"
        expected = [(key, "benchmark" if key=="table3_exact" else "unified", {}, why) for key, why in UNIFIED.items()]
        if assignment.exists():
            assigned = read(assignment) + [dict(id=CLASS_C_ID, seed=CLASS_C_SEED, scheme="class_c_matched")]
            for d in assigned:
                for v in ["actual","placebo","joint"]:
                    ident = d["id"]
                    expected.append((ident+"_"+v, "placebo",
                                     dict(placebo_id=ident, seed=d["seed"], scheme=d["scheme"], variant=v),
                                     f"{v} architecture conditional on actual and donor conventional covariates"))
        for model, family, extra, why in expected:
            folder = HERE/"fits"/mp/model
            mpath = folder/"meta.json"
            meta = read(mpath) if mpath.exists() else {}
            status = meta.get("status", "registered_no_completed_fit")
            fpath = folder/"fit.npz"
            fit_complete = status=="estimated" and fpath.exists()
            focal_b = focal_se = focal_p = None
            if fit_complete:
                with np.load(fpath,allow_pickle=False) as archive:
                    result = {k:archive[k] for k in archive.files}
                fits[mp,model] = meta,result
                collect_fit(meta,result,brows if family=="placebo" else arows,strength,path_rows,month_rows)
                focal = FOCAL if extra.get("variant")!="placebo" else extra["placebo_id"]+":"+FOCAL
                k = meta["terms"].index(focal)
                focal_stats = stats(result["beta"][k,2],result["influence"][:,k,2])
                focal_b,focal_se,focal_p = (focal_stats[k] for k in ["b","se","p"])
            registry.append(dict(mapping=mp,specification=model,family=family,rationale=why,
                                 fixed_effects=FE,risk_population="complete frozen recent persistent dyad Class C",
                                 outcomes="|".join(OUTCOMES),controls="|".join(meta.get("terms",[])),
                                 month_slopes="|".join(meta.get("slopes",[])), N=meta.get("N"),rank=meta.get("rank"),
                                 status=status,fit_complete=fit_complete, no_specialist_focal_b=focal_b,
                                 no_specialist_focal_se=focal_se,no_specialist_focal_p=focal_p,
                                 variable_definitions="research_plan.md and frozen_table3/baseline.md",
                                 result_artifact=str(fpath.relative_to(HERE)) if fit_complete else "",
                                 data_version_sha256=data_version,code_version=code_hash,**extra))
    comparisons, tail, mean_inputs = [], [], {}
    def compare(mp, metadata, code, first, second):
        meta1,r1,k1=first
        meta2,r2,k2=second
        assert meta1["paths"]==meta2["paths"]
        for j,out in enumerate(OUTCOMES):
            a,p = r1["beta"][k1,j],r2["beta"][k2,j]
            s1,s2 = r1["influence"][:,k1,j],r2["influence"][:,k2,j]
            comparisons.append(dict(mapping=mp,outcome=out,comparison=code,actual_b=a,placebo_b=p,
                                    aligned_cross_covariance=float(s1@s2), b_x10000=(a-p)*1e4,
                                    actual_t=float(a/np.linalg.norm(s1)), placebo_t=float(p/np.linalg.norm(s2)),
                                    **stats(a-p,s1-s2),**metadata))
            if "scheme" in metadata:
                key = mp, metadata["scheme"], code, out
                mean_inputs.setdefault(key, []).append((a,p,s1,s2))
    for (mp, model),(meta,r) in fits.items():
        if model.startswith("a"):
            old,oldr = fits[mp,"table3_exact"]
            compare(mp,dict(specification=model),"unified_minus_frozen",
                    (meta,r,meta["terms"].index(FOCAL)),(old,oldr,old["terms"].index(FOCAL)))
        if meta.get("variant")=="joint":
            ident = meta["placebo_id"]
            info = {k:meta[k] for k in ["placebo_id","scheme","seed"]}
            k1,k2 = [meta["terms"].index(k) for k in [FOCAL, ident+":"+FOCAL]]
            compare(mp,info,"joint_actual_minus_placebo",(meta,r,k1),(meta,r,k2))
            if (mp,ident+"_actual") in fits and (mp,ident+"_placebo") in fits:
                a,ar = fits[mp,ident+"_actual"]
                p,pr = fits[mp,ident+"_placebo"]
                compare(mp,info,"separate_actual_minus_placebo",(a,ar,a["terms"].index(FOCAL)),
                        (p,pr,p["terms"].index(ident+":"+FOCAL)))
    a,b,c,st = (pd.DataFrame(x) for x in [arows,brows,comparisons,strength])
    if len(a):
        adjust(a,["mapping","specification","outcome"],"p_holm_all_terms_in_outcome_model")
        adjust(a,["mapping"],"p_holm_unified_primary",a.outcome.eq("no_specialist")&a.term.eq(FOCAL)&a.family.eq("unified"))
        a.to_csv(HERE/"unified_specification_results.csv",index=False)
    if len(b):
        adjust(b,["mapping","specification","outcome"],"p_holm_all_terms_in_outcome_model")
        adjust(b,["mapping"],"p_holm_joint_actual_family",b.outcome.eq("no_specialist")&b.term.eq(FOCAL)&b.variant.eq("joint"))
        b.to_csv(HERE/"placebo_architecture_results.csv",index=False)
    if len(c):
        primary = c.outcome.eq("no_specialist")
        adjust(c,["mapping"],"p_holm_unified_coefficient_change",primary&c.comparison.eq("unified_minus_frozen"))
        adjust(c,["mapping"],"p_holm_placebo_contrasts",primary&c.comparison.str.endswith("actual_minus_placebo"))
        c.to_csv(HERE/"diagnostics/coefficient_contrasts.csv",index=False)
        if len(b):
            cc=c[c.comparison.eq("separate_actual_minus_placebo")]
            for (mp,scheme,out), d in cc.groupby(["mapping","scheme","outcome"]):
                tail.append(dict(mapping=mp,scheme=scheme,outcome=out,draws=len(d),
                                 placebo_b_min_x10000=d.placebo_b.min()*1e4,
                                 placebo_b_median_x10000=d.placebo_b.median()*1e4,
                                 placebo_b_max_x10000=d.placebo_b.max()*1e4,
                                 placebo_negative=int((d.placebo_b<0).sum()),
                                 placebo_at_least_as_negative_b=int((d.placebo_b<=d.actual_b).sum()),
                                 placebo_at_least_as_negative_t=int((d.placebo_t<=d.actual_t).sum()),
                                 placebo_b_lower_tail_fraction=float((d.placebo_b<=d.actual_b).mean()),
                                 interpretation="descriptive paired rank, no exchangeability assumed"))
            pd.DataFrame(tail).to_csv(HERE/"diagnostics/placebo_distribution_summary.csv",index=False)
    if mean_inputs:
        pooled = []
        for (mp, scheme, comparison, outcome), draws in mean_inputs.items():
            if len(draws) != 8:
                continue
            a_mean = float(np.mean([a for a,p,s1,s2 in draws]))
            p_mean = float(np.mean([p for a,p,s1,s2 in draws]))
            a_score = np.mean([s1 for a,p,s1,s2 in draws],axis=0)
            p_score = np.mean([s2 for a,p,s1,s2 in draws],axis=0)
            pooled.append(dict(mapping=mp,scheme=scheme,outcome=outcome,comparison=comparison,draws=len(draws),
                               actual_b_x10000=a_mean*1e4,placebo_b_x10000=p_mean*1e4,
                               b_x10000=(a_mean-p_mean)*1e4,se_x10000=np.linalg.norm(a_score-p_score)*1e4,
                               **stats(a_mean-p_mean,a_score-p_score)))
        if pooled:
            pd_pool = pd.DataFrame(pooled)
            adjust(pd_pool,["mapping"],"p_holm_mean_placebo_contrasts",pd_pool.outcome.eq("no_specialist"))
            pd_pool.to_csv(HERE/"diagnostics/placebo_mean_contrasts.csv",index=False)
    if len(st):
        base = st[(st.specification=="table3_exact")&(st.term==FOCAL)].set_index("mapping").partial_SD
        st["fraction_of_table3_partial_variance"]=(st.partial_SD/st.mapping.map(base))**2
        st.to_csv(HERE/"diagnostics/focal_rank_variation_covariance.csv",index=False)
    pd.DataFrame(path_rows).to_csv(HERE/"diagnostics/path_influence_and_fixed_projection_deletions.csv",index=False)
    pd.DataFrame(month_rows).to_csv(HERE/"diagnostics/unified_month_slopes.csv",index=False)
    reg = pd.DataFrame(registry)
    if len(a):
        for df, field in [(a,"p_holm_unified_primary"),(b,"p_holm_joint_actual_family")]:
            if field in df:
                chosen = df[df[field].notna()][["mapping","specification",field]]
                reg = reg.merge(chosen,how="left",on=["mapping","specification"],validate="one_to_one")
    reg.to_csv(HERE/"specification_registry.csv",index=False)
    plots(a,b,c,fits)
    note("Collected",len(fits),"fits of",len(registry),"registered, coefficients",len(a)+len(b))


if __name__ == "__main__":
    main()
