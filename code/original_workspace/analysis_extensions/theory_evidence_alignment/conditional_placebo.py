"""Core-specific donor stress test that preserves frozen Class C support."""
import argparse
from engine import *
from placebo import CONVENTIONAL, MODIFIERS, matching_sample
from run import persist

ID = "class_c_matched_01"
SEED = 20261109201


def build(risk):
    out = HERE / "diagnostics" / risk.mapping
    dest = out / "class_c_donors.npy"
    if dest.exists():
        return np.load(dest, mmap_mode="r")
    field = risk.sub.field_id.to_numpy()
    domain = risk.sub.domain_id.to_numpy()
    profiles = np.sqrt(risk.R[risk.fields])
    hellinger = np.sqrt(np.maximum(1-profiles @ profiles.T,0))
    rng = np.random.default_rng(SEED)
    penalty = (hellinger/.6)**2 + rng.uniform(0,.05,(risk.J,risk.J))
    pending = dest.with_suffix(".tmp.npy")
    selected = np.lib.format.open_memmap(pending, mode="w+", dtype=np.int16, shape=risk.mask.shape)
    field_count = domain_count = support_count = 0
    for idx,(a,b) in enumerate(risk.blocks(chunk=8192)):
        if idx % 18 == 0:
            note(risk.mapping,"Class C eligible donor search",a,"/",risk.I)
        m = risk.mask[a:b]
        # Off-risk cells are never used. A deterministic alternate column keeps array indexing valid.
        block = np.broadcast_to((np.arange(risk.J)+1)%risk.J, (b-a,risk.J)).copy()
        local = {name:risk.z[name][a:b] for name in CONVENTIONAL}
        def choose(rows,j,alternatives):
            if not len(rows) or not len(alternatives):
                return
            eligible = m[np.ix_(rows,alternatives)]
            has = eligible.any(1)
            rows = rows[has]
            if not len(rows):
                return
            score = np.broadcast_to(penalty[j,alternatives],(len(rows),len(alternatives))).copy()
            for arr in local.values():
                gap = arr[np.ix_(rows,alternatives)]-arr[rows,j,None]
                score += (gap**2)/len(local)
            score[~eligible[has]] = np.inf
            block[rows,j] = alternatives[np.argmin(score,axis=1)]
            assigned[rows] = True
        for j in range(risk.J):
            assigned = np.zeros(b-a,bool)
            candidates = np.arange(risk.J)!=j
            choose(np.flatnonzero(m[:,j]), j, np.flatnonzero(candidates & (field==field[j])))
            remaining = m[:,j] & ~assigned
            choose(np.flatnonzero(remaining), j, np.flatnonzero(candidates & (domain==domain[j]) & (field!=field[j])))
            remaining = m[:,j] & ~assigned
            choose(np.flatnonzero(remaining), j, np.flatnonzero(candidates & (domain!=domain[j])))
            remaining = m[:,j] & ~assigned
            assert (risk.den[a:b][remaining] == 1).all()
            if remaining.any():
                fallback = np.flatnonzero(candidates & (field==field[j]))
                if not len(fallback):
                    fallback = np.flatnonzero(candidates & (domain==domain[j]))
                if not len(fallback):
                    fallback = np.flatnonzero(candidates)
                block[remaining,j] = fallback[np.argmin(penalty[j,fallback])]
        assert np.all(block!=np.arange(risk.J))
        selected[a:b] = block
        field_count += int(np.count_nonzero((field[block]==field[None,:])&m))
        domain_count += int(np.count_nonzero((domain[block]==domain[None,:])&m))
        support_count += int(np.count_nonzero(np.take_along_axis(m,block,axis=1)&m))
    selected.flush()
    del selected
    pending.replace(dest)
    donor = np.load(dest,mmap_mode="r")
    _, ix = matching_sample(risk)
    mask = risk.mask[ix]
    match = donor[ix]
    matched = []
    for name in MODIFIERS:
        z = risk.z[name][ix]
        v,p = z[mask],np.take_along_axis(z,match,axis=1)[mask]
        matched.append(dict(variable=name,corr=np.corrcoef(v,p)[0,1],actual_sd_z=v.std(),placebo_sd_z=p.std(),
                            rms_z=np.sqrt(np.mean((v-p)**2)),within_half_sd=np.mean(np.abs(v-p)<=.5)))
    pd.DataFrame(matched).to_csv(out/"class_c_matching_quality.csv",index=False)
    save(out/"class_c_assignment_summary.json",dict(id=ID,seed=SEED,pairs=risk.npairs,
         same_field_pairs=field_count,same_domain_pairs=domain_count,donor_class_c_pairs=support_count,
         original_candidate_pairs_kept=risk.npairs,actual_destination_gains_kept=True,
         donor_selection_uses_outcomes=False,mean_holdout_hellinger=float(hellinger[np.arange(risk.J),match][mask].mean())))
    note(risk.mapping,"Class C matched donor assignment locked",support_count,"/",risk.npairs)
    return donor


def main(mapping):
    freeze()
    risk = Risk(mapping)
    donor = build(risk)
    z = dict(risk.z)
    for name in MODIFIERS:
        z[ID+":"+name] = np.take_along_axis(risk.z[name],donor,axis=1)
    slopes = MODIFIERS + [ID+":"+name for name in MODIFIERS]
    loc = HERE/"cache"/mapping/"placebo_class_c.npz"
    loc.parent.mkdir(parents=True,exist_ok=True)
    cache = moments(risk,terms(slopes),slopes,z,loc)
    common = CONVENTIONAL + [ID+":"+name for name in CONVENTIONAL]
    for v, u in [("actual",["Distributed"]),("placebo",[ID+":Distributed"]),
                 ("joint",["Distributed",ID+":Distributed"])]:
        mods = common+u
        meta,result = fit(risk,cache,terms(mods),mods)
        persist(mapping,ID+"_"+v,meta,result,family="placebo",placebo_id=ID,seed=SEED,
                scheme="class_c_matched",variant=v)


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mapping",choices=list(EXPECTED))
    main(p.parse_args().mapping)
