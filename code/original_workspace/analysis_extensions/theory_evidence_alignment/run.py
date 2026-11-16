"""Reproduce Table 3 and estimate the registered extension on complete risks."""
import argparse
from engine import *


def persist(mapping, model, meta, result, **extra):
    loc = HERE / "fits" / mapping / model
    save(loc / "meta.json", {**meta, **extra, "mapping": mapping, "specification": model})
    if result is not None:
        np.savez_compressed(loc / "fit.npz", **result)
    note(mapping, model, meta["status"])


def unified(risk):
    basic = ARCH + ["Applicability"]
    risk.add_squared(["Gap", "Distance", "Applicability"])
    nonlinear = basic + ["GapSq", "DistanceSq", "ApplicabilitySq"]
    all_slopes = nonlinear + ["Access"]
    folder = HERE / "cache" / risk.mapping
    folder.mkdir(parents=True, exist_ok=True)
    cache = moments(risk, terms(all_slopes), all_slopes, risk.z, folder / "unified.npz")
    meta, result = fit(risk, cache, terms(ARCH), ARCH)
    verify_reproduction(risk.mapping, meta, result)
    persist(risk.mapping, "table3_exact", meta, result, family="benchmark")
    for name, mods in [
        ("a1_unified", basic),
        ("a2_unified_access", basic + ["Access"]),
        ("a3_nonlinear_relatedness", nonlinear),
    ]:
        meta, result = fit(risk, cache, terms(mods), mods)
        if name == "a2_unified_access":
            folder_old = PORT / "verification/recent_dyad" / risk.mapping / "augmented_controls"
            with np.load(folder_old / "joint_covariances.npz") as old:
                ix = [meta["terms"].index(k) for k in read(folder_old / "joint_metadata.json")["regressors"]]
                ix_v = (np.arange(4)[:, None]*len(ix)+np.array(ix)[None, :]).ravel()
                assert np.allclose(result["beta"][ix].T, old["beta"], rtol=1e-6, atol=2e-11)
                assert np.allclose(result["covariance"][np.ix_(ix_v, ix_v)], old["capability"], rtol=1e-5, atol=1e-13)
        persist(risk.mapping, name, meta, result, family="unified")
    save(HERE / "diagnostics" / (risk.mapping + "_scales.json"), risk.scales)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mapping", choices=list(EXPECTED) + ["freeze"])
    p.add_argument("--stage", choices=["unified", "placebo", "all"], default="all")
    p.add_argument("--batch", type=int, choices=range(4))
    p.add_argument("--prepare-placebos", action="store_true")
    args = p.parse_args()
    freeze()
    if args.mapping != "freeze":
        risk = Risk(args.mapping)
        if args.prepare_placebos:
            from placebo import prepare
            prepare(risk)
        elif args.stage in ["unified", "all"]:
            unified(risk)
        if not args.prepare_placebos and args.stage in ["placebo", "all"]:
            from placebo import run_placebos
            run_placebos(risk, args.batch)
