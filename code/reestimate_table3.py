"""Rebuild the two outcome-partition equations on the frozen Class C risks."""

import argparse
from pathlib import Path
import sys


def reestimate(workspace, out, mapping):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(workspace / "analysis_extensions/theory_evidence_alignment"))
    import duckdb
    import engine as eng
    import numpy as np
    import pandas as pd

    risk = eng.Risk(mapping)
    original_events = risk.counts.copy()
    risk.raw.clear()
    risk.z.pop("Access")
    universal = eng.BASE / "02_dyad_full_population/universal"
    work_file = (universal / "analysis_works.parquet").as_posix()
    members_file = (universal / "members.parquet").as_posix()
    author_fields = (eng.BASE / "01_sample_construction/author_field.parquet").as_posix()
    with duckdb.connect() as db:
        db.execute("SET threads=8")
        db.register("eligible_cores", pd.DataFrame({"g": risk.cores, "i": np.arange(risk.I)}))
        db.register("eligible_destinations", pd.DataFrame({
            "j": risk.fields, "j0": np.arange(risk.J),
            "sf": risk.sub.subfield_id.str.extract(r"(\d+)$", expand=False).astype(int),
        }))
        events = db.execute(f"""
            SELECT i,j0,mo,min(y) AS year,min(y)=max(y) AS same_year,
                   bool_or(current_experts>current_core_experts) AS specialist,
                   bool_or(current_core_experts>0) AS experienced
            FROM read_parquet('{work_file}') JOIN eligible_cores USING(g)
            JOIN eligible_destinations USING(j)
            WHERE mo BETWEEN 786 AND 796 AND mo=first_mo
            GROUP BY i,j0,mo ORDER BY i,j0,mo
        """).fetchdf()
        events = events[risk.mask[events.i, events.j0]].reset_index(drop=True)
        histories = db.execute(f"""
            WITH core_members AS (
                SELECT i,author_id FROM read_parquet('{members_file}') JOIN eligible_cores USING(g)
            )
            SELECT i,j0,min(first_y) AS first_y FROM read_parquet('{author_fields}')
            JOIN core_members USING(author_id) JOIN eligible_destinations USING(sf)
            WHERE first_y <= 2025 GROUP BY i,j0
        """).fetchdf()
    histories = histories[risk.mask[histories.i, histories.j0]]
    assert not histories.duplicated(["i", "j0"]).any()
    assert histories.first_y.min() >= 2022
    events = events.merge(histories, on=["i", "j0"], how="left", validate="many_to_one")
    events["status_from_history"] = events.first_y < events.year
    assert events.same_year.all() and events.experienced.eq(events.status_from_history).all()
    order = np.lexsort((risk.et, risk.ej, risk.ei))
    events = events.sort_values(["i", "j0", "mo"])
    assert np.array_equal(events[["i", "j0", "mo"]].to_numpy(),
                          np.column_stack([risk.ei, risk.ej, risk.et + 786])[order])
    event_order = events.iloc[np.argsort(order)]
    assert np.array_equal(event_order.specialist.to_numpy(), risk.v[:, 0].astype(bool))
    no = ~event_order.specialist.to_numpy()
    status = event_order.experienced.to_numpy()
    risk.v = np.column_stack([no & ~status, no & status]).astype(float)
    risk.event_y = risk.transform[risk.et, :, None] * risk.v[:, None, :]
    risk.row_y.fill(0)
    np.add.at(risk.row_y, risk.ei, risk.event_y)
    risk.row_y /= risk.den[:, None, None]
    n0, n1 = map(int, risk.v.sum(0))
    assert n0 + n1 == original_events[2]
    risk.counts = [n0 + n1, n0, n1, n0 + n1]
    risk.project_destinations()
    mask = risk.mask.astype(float)
    # y* = y0 - row_y - destination effect + row mean of destination effects.
    uy = risk.row_y - (mask @ risk.gamma_y.reshape(risk.J, -1)).reshape(risk.row_y.shape) / risk.den[:, None, None]
    cache_path = eng.HERE / "cache" / mapping / "unified.npz"
    with np.load(cache_path, allow_pickle=False) as raw:
        cache = dict(raw)
    columns = eng.terms(eng.ARCH + ["Applicability"])
    slopes = eng.ARCH + ["Applicability"]
    original_xy, original_zy = cache["xy"].copy(), cache["zy"].copy()
    cache["xy"].fill(0)
    cache["zy"].fill(0)
    path_events = risk.path[risk.ej]
    event_groups = [np.flatnonzero(path_events == g) for g in range(len(risk.paths))]
    path_counts = np.column_stack([mask[:, sl].sum(1) for sl in risk.groups])
    # Destination orthogonality makes each path's destination-mean Y term vanish.
    # Event contributions are sparse. The only dense cross-products needed are
    # each core's path sum of projected X and its projected Y row component.
    for name in list(dict.fromkeys([*slopes, *eng.EXTRA])):
        eng.note(mapping, "rebuilding partition moments", name)
        z = np.where(risk.mask, risk.z[name], 0.0)
        for source, times in (("Destination", risk.at), ("Home", risk.ht), ("Month", None)):
            term = f"{name}_{source}"
            if source != "Month" and term not in columns:
                continue
            if source == "Month" and name not in slopes:
                continue
            if source == "Home":
                raw_i = z.sum(1)[:, None] * times
                raw_j = z.T @ times
            elif source == "Destination":
                raw_i = z @ times
                raw_j = z.sum(0)[:, None] * times
            else:
                raw_i = z.sum(1)[:, None]
                raw_j = z.sum(0)[:, None]
            row_average = raw_i / risk.den[:, None]
            gamma = risk.sinv @ (raw_j - mask.T @ row_average)
            u = row_average - (mask @ gamma) / risk.den[:, None]
            if source == "Month":
                index = cache["mods"].tolist().index(name)
            else:
                index = cache["columns"].tolist().index(term)
            for g, sl in enumerate(risk.groups):
                selected = event_groups[g]
                i, j = risk.ei[selected], risk.ej[selected]
                if source == "Home":
                    row_sum = z[:, sl].sum(1)[:, None] * times
                    observed = z[i, j, None] * times[i]
                elif source == "Destination":
                    row_sum = z[:, sl] @ times[sl]
                    observed = z[i, j, None] * times[j]
                else:
                    row_sum = z[:, sl].sum(1)[:, None]
                    observed = z[i, j, None]
                row_sum -= path_counts[:, g, None] * u + mask[:, sl] @ gamma[sl]
                observed -= u[i] + gamma[j]
                if source == "Month":
                    cache["zy"][g, index] = (
                        observed.T @ risk.event_y[selected].reshape(len(selected), risk.T * 2)
                    ).reshape(risk.T, 2) - np.einsum("i,itn->tn", row_sum[:, 0], uy)
                else:
                    cache["xy"][g, index] = np.einsum("it,itn->n", observed, risk.event_y[selected]) - np.einsum(
                        "it,itn->n", row_sum, uy)
    # Combined outcomes must reproduce the previously projected No-Specialist Y
    # path by path. This tests the projection, sorting and original-risk support.
    x_indices = [cache["columns"].tolist().index(c) for c in columns]
    z_indices = [cache["mods"].tolist().index(c) for c in slopes]
    err_x = np.max(np.abs(cache["xy"][:, x_indices].sum(-1) - original_xy[:, x_indices, 1]))
    err_z = np.max(np.abs(cache["zy"][:, z_indices].sum(-1) - original_zy[..., 1][:, z_indices]))
    assert err_x < 1e-8 and err_z < 1e-8, (err_x, err_z)
    fit_meta, fit = eng.fit(risk, cache, columns, slopes)
    old_meta, old_fit = eng.fit(risk, cache, eng.terms(eng.ARCH), eng.ARCH)
    a1 = eng.HERE / "fits" / mapping / "a1_unified" / "fit.npz"
    with np.load(a1) as archive:
        assert np.allclose(fit["beta"][:, 0], archive["beta"][:, 2], atol=2e-12, rtol=1e-9)
        assert np.allclose(fit["influence"][:, :, 0], archive["influence"][:, :, 2], atol=2e-12, rtol=1e-9)
    old_root = eng.PORT / "verification/recent_dyad" / mapping / "validity"
    for index, name in enumerate(("no_current_core_expert", "with_current_core_expert"), 1):
        saved = eng.read(old_root / f"{name}.json")
        c = next(r for r in saved["coefficients"] if r["term"] == eng.FOCAL and r["cluster"] == "capability")
        j = old_meta["terms"].index(eng.FOCAL)
        np.testing.assert_allclose(
            [old_fit["beta"][j, index], np.linalg.norm(old_fit["influence"][:, j, index])],
            [c["b"], c["se"]], rtol=1e-7, atol=1e-12)
    meta = dict(fit_meta, mapping=mapping, outcomes=["combined_no_specialist", "neither", "at_least_one", "difference"],
                first_entry_events=dict(neither=n0, at_least_one=n1),
                full_risk_candidate_months=risk.N,
                timing={
                    str(year): {"months": months,
                                "at_least_one": int((histories.first_y < year).sum()) * months,
                                "neither": (risk.npairs - int((histories.first_y < year).sum())) * months}
                    for year, months in [(2025, 6), (2026, 5)]
                },
                frozen_2021_at_least_one_pairs=0,
                validation=dict(max_path_XY_additivity_error=float(err_x), max_path_ZY_additivity_error=float(err_z),
                                original_partition_reproduced=True, a1_combined_coefficients_and_scores_reproduced=True,
                                event_status_reconciled=len(events)),
                covariance_reference="Capability-path CR1, t(9)")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{mapping}_a1_moments.npz",
                        Q_by_path=fit["Q_by_path"], xy_by_path=fit["xy_by_path"])
    eng.save(out / f"{mapping}_a1_metadata.json", meta)
    eng.note(mapping, "A1 outcome partition reproduced and saved", risk.counts, meta["timing"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mapping", choices=("parent", "direct"))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reestimate(args.workspace.resolve(), args.out.resolve(), args.mapping)
