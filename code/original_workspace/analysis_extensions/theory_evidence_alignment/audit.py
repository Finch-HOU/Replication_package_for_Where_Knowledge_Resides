"""Verify donor coverage constructions and full-risk regression outputs."""
import argparse
from engine import *
from conditional_placebo import ID


def coverage(mapping):
    freeze()
    risk = Risk(mapping)
    sample = np.sort(np.random.default_rng(20261109301).choice(risk.I, 2048, replace=False))
    members = pd.read_parquet(ANCHOR/"constructs"/mapping/"members.parquet")
    assert np.array_equal(members.g.to_numpy(), risk.cores)
    indexes = members.iloc[sample][["author_index0","author_index1"]].to_numpy()
    counts = np.load(PORT/"measurement/author_counts.npy",mmap_mode="r")
    known = counts[indexes] > 0
    profile = risk.R[risk.fields].T
    gap = 1 - known.any(1).astype(float)@profile
    best = (known.astype(float)@profile).max(1)
    distributed = np.clip(1-gap-best,0,1)
    shared = known.all(1).astype(float)@profile
    out = HERE/"diagnostics"/mapping
    draws = [(x["id"],np.array(x["donor"])) for x in read(out/"assignments.json")]
    draws.append((ID,np.load(out/"class_c_donors.npy",mmap_mode="r")[sample]))
    rows = []
    for ident, donor in draws:
        selector = np.broadcast_to(donor, (len(sample),risk.J))
        assert not np.any(selector==np.arange(risk.J))
        m = risk.mask[sample]
        for name, raw in [("Gap",gap),("Distributed",distributed),("Shared",shared)]:
            direct = np.take_along_axis(raw,selector,axis=1)[m]
            reused = np.take_along_axis(risk.raw[name][sample],selector,axis=1)[m]
            max_error = float(np.max(np.abs(direct-reused)))
            assert max_error < 2e-6, (mapping, ident, name, max_error)
            rows.append(dict(mapping=mapping,placebo_id=ident,variable=name,audit_cores=len(sample),
                             audit_pairs=int(m.sum()),max_abs_difference=max_error,
                             mean_abs_difference=float(np.mean(np.abs(direct-reused)))))
    pd.DataFrame(rows).to_csv(out/"donor_coverage_reconstruction.csv",index=False)
    note(mapping,"all 17 donor assignments reconciled with member-history recomputation")


def outcomes():
    freeze()
    registry = pd.read_csv(HERE/"specification_registry.csv")
    checks = []
    for row in registry[registry.fit_complete].itertuples():
        f = HERE/row.result_artifact
        m = read(f.parent/"meta.json")
        with np.load(f,allow_pickle=False) as z:
            b = z["beta"]
            s = z["influence"]
            max_b = np.abs(np.stack([b[:,0]-b[:,1]-b[:,2],b[:,3]-b[:,1]+b[:,2]])).max()
            max_s = np.abs(np.stack([s[:,:,0]-s[:,:,1]-s[:,:,2],s[:,:,3]-s[:,:,1]+s[:,:,2]])).max()
            assert max_b < 1e-12 and max_s < 1e-12
            g = s.transpose(0,2,1).reshape(10,-1)
            assert np.allclose(z["covariance"],g.T@g,atol=1e-20,rtol=1e-11)
        i,p,e = EXPECTED[row.mapping]
        assert (m["cores"],m["pairs"],m["N"],m["events"],m["df_inference"]) == (i,p,p*11,e,9)
        assert m["rank"]==m["columns"]
        checks.append(dict(mapping=row.mapping,specification=row.specification,full_risk_verified=True,
                           rank_verified=True,max_outcome_identity_error=float(max_b),
                           max_joint_score_identity_error=float(max_s)))
    pd.DataFrame(checks).to_csv(HERE/"diagnostics/model_integrity.csv",index=False)
    note("Model integrity",len(checks),"/",len(registry),"registered")
    assert len(checks)==len(registry)==110
    assert len(pd.read_csv(HERE/"diagnostics/placebo_mean_contrasts.csv"))==32


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=list(EXPECTED)+["outcomes"])
    stage=p.parse_args().stage
    outcomes() if stage=="outcomes" else coverage(stage)
