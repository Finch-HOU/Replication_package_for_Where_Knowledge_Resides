"""Predetermined donor profiles. Fit outcomes and gains stay at the actual destination."""
from engine import *
from scipy.optimize import linear_sum_assignment

CONVENTIONAL = ["Gap", "Distance", "Shared", "Applicability"]
MATCH = ["Gap", "Distance", "Shared", "Applicability"]
MODIFIERS = ARCH + ["Applicability"]
DRAW_COUNT = 8


def matching_sample(risk):
    rng = np.random.default_rng(20261109)
    ix = rng.choice(risk.I, 24000, replace=False)
    return np.sort(ix[:12000]), np.sort(ix[12000:])


def assign(risk, scheme, seed, cost):
    rng = np.random.default_rng(seed)
    donor = np.full(risk.J, -1)
    field = risk.sub.field_id.to_numpy()
    domain = risk.sub.domain_id.to_numpy()
    for f in np.unique(field):
        ids = np.flatnonzero(field == f)
        if len(ids) == 1:
            j = ids[0]
            pool = np.flatnonzero((domain == domain[j]) & (field != field[j]))
            assert len(pool)
            donor[j] = rng.choice(pool) if scheme == "field_shuffle" else pool[
                np.argmin(cost[j, pool] + rng.uniform(0, .25, len(pool)))
            ]
        elif scheme == "field_shuffle":
            p = rng.permutation(ids)
            while np.any(ids == p):
                p = rng.permutation(ids)
            donor[ids] = p
        else:
            local = cost[np.ix_(ids, ids)] + rng.uniform(0, .25, (len(ids), len(ids)))
            np.fill_diagonal(local, np.inf)
            row, col = linear_sum_assignment(local)
            donor[ids[row]] = ids[col]
    assert np.all(donor != np.arange(risk.J)) and np.all(donor >= 0)
    return donor


def prepare(risk):
    out = HERE / "diagnostics" / risk.mapping
    out.mkdir(parents=True, exist_ok=True)
    path = out / "assignments.json"
    if path.exists():
        return read(path)
    ix, audit = matching_sample(risk)
    m = risk.mask[ix].astype(float)
    n = m.sum(0)[:, None]
    profile = np.sqrt(risk.R[risk.fields])
    hellinger = np.sqrt(np.maximum(1-profile @ profile.T, 0))
    components = [hellinger]
    for var in MATCH:
        z = risk.z[var][ix]
        diag = np.sum(m*z*z, axis=0)[:, None]
        components.append(np.sqrt(np.maximum((diag + m.T @ (z*z) - 2*(m*z).T @ z)/n, 0)))
    same = risk.sub.field_id.to_numpy()[:, None] == risk.sub.field_id.to_numpy()[None, :]
    allowed = same & ~np.eye(risk.J, dtype=bool)
    cost = np.mean([x / np.median(x[allowed]) for x in components], axis=0)
    draws, links, quality, profile_quality = [], [], [], []
    mask = risk.mask[audit]
    for s, scheme in enumerate(["field_shuffle", "matched"]):
        for k in range(DRAW_COUNT):
            seed = 2026110900 + s*100 + k
            name = f"{scheme}_{k+1:02}"
            donor = assign(risk, scheme, seed, cost)
            draws.append(dict(id=name, scheme=scheme, seed=seed, donor=donor.tolist(), batch=k//2))
            field_match = risk.sub.field_id.to_numpy() == risk.sub.field_id.to_numpy()[donor]
            profile_quality.append(dict(placebo_id=name, seed=seed, destinations=risk.J,
                                        same_field_destinations=int(field_match.sum()),
                                        unique_donors=len(np.unique(donor)), same_profile_count=0,
                                        mean_hellinger=float(hellinger[np.arange(risk.J), donor].mean()),
                                        mean_training_cost=float(cost[np.arange(risk.J), donor].mean())))
            for j, d in enumerate(donor):
                links.append(dict(placebo_id=name, scheme=scheme, seed=seed, actual_j=j, donor_j=d,
                                  actual_subfield=risk.sub.subfield_id[j], donor_subfield=risk.sub.subfield_id[d],
                                  same_field=bool(field_match[j]), same_domain=risk.sub.domain_id[j]==risk.sub.domain_id[d],
                                  historical_profile_hellinger=hellinger[j,d], frozen_match_cost=cost[j,d]))
            for var in MODIFIERS:
                z = risk.z[var][audit]
                actual = z[mask]
                donor_z = z[:, donor][mask]
                raw = risk.raw[var][audit]
                quality.append(dict(placebo_id=name, scheme=scheme, seed=seed, variable=var,
                                    holdout_cores=len(audit), holdout_risk_pairs=len(actual),
                                    actual_mean_z=float(actual.mean()), placebo_mean_z=float(donor_z.mean()),
                                    actual_sd_z=float(actual.std()), placebo_sd_z=float(donor_z.std()),
                                    corr=float(np.corrcoef(actual,donor_z)[0,1]),
                                    rms_z=float(np.sqrt(np.mean((actual-donor_z)**2))),
                                    within_half_sd=float(np.mean(np.abs(actual-donor_z)<=.5)),
                                    actual_zero=float(np.mean(raw[mask]==0)),
                                    placebo_zero=float(np.mean(raw[:, donor][mask]==0))))
    pd.DataFrame(links).to_csv(out / "assignments.csv", index=False)
    pd.DataFrame(quality).to_csv(out / "matching_quality.csv", index=False)
    pd.DataFrame(profile_quality).to_csv(out / "profile_quality.csv", index=False)
    save(path, draws)
    note(risk.mapping, "all donor assignments locked", len(draws))
    return draws


def run_placebos(risk, batch=None):
    from run import persist
    draws = prepare(risk)
    folder = HERE / "cache" / risk.mapping
    folder.mkdir(parents=True, exist_ok=True)
    for n in range(DRAW_COUNT//2):
        if batch is not None and batch != n:
            continue
        selected = [row for row in draws if row["batch"] == n]
        values = dict(risk.z)
        all_mods = list(MODIFIERS)
        for row in selected:
            donor = row["donor"]
            for variable in MODIFIERS:
                key = row["id"] + ":" + variable
                values[key] = risk.z[variable][:, donor]
                all_mods.append(key)
        cache = moments(risk, terms(all_mods), all_mods, values, folder / f"placebo_{n}.npz")
        for row in selected:
            ident = row["id"]
            base = CONVENTIONAL + [ident + ":" + k for k in CONVENTIONAL]
            for variant, mods in [
                ("actual", base + ["Distributed"]),
                ("placebo", base + [ident + ":Distributed"]),
                ("joint", base + ["Distributed", ident + ":Distributed"]),
            ]:
                meta, result = fit(risk, cache, terms(mods), mods)
                persist(risk.mapping, ident + "_" + variant, meta, result, family="placebo",
                        placebo_id=ident, seed=row["seed"], scheme=row["scheme"], variant=variant)
        del cache, values
