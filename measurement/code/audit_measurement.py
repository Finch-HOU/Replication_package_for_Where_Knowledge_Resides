"""Trace the unchanged capability inputs from the archived monthly score file."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from proxy_library import ALIASES, month


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "inputs"
OUT = ROOT / "outputs"
FIG = ROOT.parent / "01_Manuscript/figures"
FIRST, LAST = 786, 796


def iso(mo):
    return (pd.Period("1960-01", freq="M") + int(mo)).strftime("%Y-%m")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def csv(df, filename):
    df.to_csv(OUT / filename, index=False, float_format="%.17g")


def archive_series():
    source = pd.read_csv(INPUT / "field_monthly_scores.csv")
    source["protocol"] = source.series.map(ALIASES)
    source["mo"] = source.month.map(month)
    assert source.protocol.notna().all()
    assert not source.duplicated(["protocol", "field", "mo"]).any()
    source["status_eligible"] = source.status.eq("observed_model_window")
    source["usable_frontier"] = source.frontier.where(source.status_eligible)
    source["last_observed_release"] = pd.to_datetime(source.last_observed_release)
    valid_dates = source.last_observed_release.dropna()
    month_end = pd.to_datetime(source.month) + pd.offsets.MonthEnd(0)
    assert (valid_dates <= month_end.loc[valid_dates.index]).all()
    cov = source[["protocol", "field", "month", "n_model_versions", "status",
                  "last_observed_release", "usable_frontier"]].copy()
    cov["has_frontier"] = cov.pop("usable_frontier").notna()
    csv(cov, "domain_month_coverage.csv")
    rebuilt = []
    for (protocol, field), block in source.groupby(["protocol", "field"]):
        for mo in range(month("2023-01"), month("2026-09") + 1):
            levels = block.set_index("mo").usable_frontier
            window = levels.reindex(range(mo-18, mo-5)).to_numpy()
            rebuilt.append(dict(protocol=protocol, field=field, mo=mo, calendar_month=iso(mo),
                                frontier_l0=levels.get(mo, np.nan), frontier_l6=window[-1],
                                window_valid_months=int(np.isfinite(window).sum()),
                                start_month=iso(mo-18), end_month=iso(mo-6),
                                start_frontier=window[0], end_frontier=window[-1],
                                gain12_l6=window[-1] - window[0] if np.isfinite(window).all() else np.nan))
    fresh = pd.DataFrame(rebuilt)
    old = pd.read_csv(INPUT / "field_month_proxies.csv")
    key = ["protocol", "field", "mo"]
    check = old.merge(fresh, on=key, suffixes=("_archived", "_rebuild"), validate="one_to_one")
    assert len(check) == len(old) == len(fresh)
    errors = {}
    for name in ("frontier_l0", "frontier_l6", "gain12_l6"):
        a, b = (check[name + suffix] for suffix in ("_archived", "_rebuild"))
        np.testing.assert_array_equal(a.isna(), b.isna())
        np.testing.assert_allclose(a, b, rtol=0, atol=2e-15, equal_nan=True)
        errors[name] = {"observed_cells": int(a.notna().sum()), "maximum_absolute_difference": float((a-b).abs().max())}
    csv(fresh, "rebuilt_monthly_series.csv")
    missing = source.groupby(["protocol", "field"]).agg(
        rows=("month", "size"), in_window=("status_eligible", "sum"),
        frontier_observed=("usable_frontier", "count"), raw_top3_observed=("top3_mean", "count"),
        release_labels_observed=("last_observed_release", "count")).reset_index()
    missing["outside_observed_window"] = missing.rows - missing.in_window
    missing["missing_frontier_within_window"] = missing.in_window - missing.frontier_observed
    csv(missing, "missingness_by_protocol_domain.csv")
    aligned = source.pivot(index=["field", "month"], columns="protocol", values="usable_frontier")
    aligned["available_named_protocol_max"] = aligned[["pd0a", "pcca"]].max(axis=1)
    aligned["mixed_minus_available_named_protocol_max"] = aligned["mix"] - aligned.available_named_protocol_max
    aligned["running_named_protocol_max"] = aligned.groupby(level="field").available_named_protocol_max.cummax()
    aligned["mixed_minus_running_named_protocol_max"] = aligned["mix"] - aligned.running_named_protocol_max
    csv(aligned.reset_index(), "protocol_overlap_diagnostic.csv")
    return source, fresh, errors


def mapping_summary():
    m = pd.read_csv(INPUT / "subfield_mapping_versions.csv")
    m = m[m.mapping.isin(["parent", "direct"])].copy()
    assert not m.duplicated(["mapping", "candidate_subfield_id"]).any()
    pathsets = m.groupby("mapping").field.apply(set)
    assert len(pathsets["parent"]) == 10 and pathsets["parent"] == pathsets["direct"]
    p = m[m.mapping.eq("parent")].set_index("candidate_subfield_id").field
    d = m[m.mapping.eq("direct")].set_index("candidate_subfield_id").field
    assert p.loc[d.index].eq(d).all()
    pre = pd.read_csv(INPUT / "crosswalk_audit.csv").set_index("candidate_subfield_id")
    for key, values in (("parent", p), ("direct", d)):
        assert values.sort_index().equals(pre["mmlu_" + key].dropna().sort_index())
    rows = []
    for (mapping, field), group in m.groupby(["mapping", "field"]):
        rows.append(dict(capability_path=field, mmlu_pro_domains=field, mapping_type=mapping,
                         openalex_parent_fields=" | ".join(sorted(group.candidate_field_name.unique())),
                         mapped_subfields=len(group),
                         construction_rule="One supplied frontier for the identically named benchmark domain; no subject average"))
    table = pd.DataFrame(rows)
    csv(table, "ten_path_mapping.csv")
    return sorted(pathsets["parent"]), table


def figures(mixed, paths):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(5, 2, figsize=(9, 9), sharex=True, sharey=True, layout="constrained")
    for path, ax in zip(paths, axes.flat):
        q = mixed[mixed.field.eq(path)]
        date = pd.to_datetime(q.calendar_month)
        ax.step(date, q.frontier_l0 * 100, where="post", color="#164f6f", label="Frontier (accuracy, %)")
        gain = q.mo.between(FIRST, LAST)
        ax.plot(date[gain], q.loc[gain, "gain12_l6"] * 100, "o-", color="#a84722",
                markersize=2.4, label="Lagged 12-month gain (pp)")
        ax.set_title(path.title(), loc="left", fontweight="bold")
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xlim(pd.Timestamp("2024-01"), pd.Timestamp("2026-06"))
        ax.grid(axis="y", color="#e3e9ec", linewidth=.55)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    fig.savefig(FIG / "capability_paths.png", dpi=240)
    plt.close(fig)


def verify_panel(workspace, path_grid):
    sys.dont_write_bytecode = True
    expanded = workspace / "expanded_sample_reassessment_20261102"
    sys.path.insert(0, str(expanded / "08_reproducibility"))
    from panel_api import Panel
    anchor = workspace / "anchor_amplification_exploration_20261106"
    counts_file = workspace / "portfolio_A_theory_driven_exploration_20261103/measurement/author_counts.npy"
    count = np.load(counts_file, mmap_mode="r")
    results = {}
    for mapping in ("parent", "direct"):
        panel = Panel("recent", "dyad", mapping, classes="C")
        replay_dest = path_grid.loc[panel.paths[panel.fc], panel.tm].to_numpy()
        np.testing.assert_allclose(panel.A, replay_dest, atol=2e-15, rtol=0)
        members = pd.read_parquet(anchor / "constructs" / mapping / "members.parquet",
                                  columns=["g", "author_index0", "author_index1"])
        assert np.array_equal(panel.g, members.g)
        indices = members[["author_index0", "author_index1"]].to_numpy()
        err, max_hist_error, max_weight_error = 0., 0., 0.
        min_mapped = None
        for a in range(0, panel.I, 8192):
            b = min(a + 8192, panel.I)
            pooled = count[indices[a:b]].astype(np.int64).sum(1)
            max_hist_error = max(max_hist_error, float(np.max(np.abs(pooled - panel.hist[a:b]))))
            mapped = pooled[:, panel.js]
            by_path = np.column_stack([mapped[:, panel.fc == k].sum(1)
                                       for k in range(len(panel.paths))])
            den = by_path.sum(1)
            assert np.all(den > 0)
            weights = by_path / den[:, None]
            min_mapped = int(den.min()) if min_mapped is None else min(min_mapped, int(den.min()))
            max_weight_error = max(max_weight_error, float(np.max(np.abs(weights.sum(1) - 1))))
            rebuilt = weights @ path_grid.loc[panel.paths, panel.tm].to_numpy()
            err = max(err, float(np.max(np.abs(rebuilt - panel.ha[a:b]))))
        assert max_hist_error == 0 and max_weight_error < 2e-15 and err < 2e-15
        results[mapping] = dict(cores=panel.I, mapped_destinations=panel.J, months=panel.T,
                                home_core_months=int(panel.I * panel.T),
                                destination_subfield_months=int(panel.J * panel.T),
                                candidate_pairs=int(panel.mask.sum()),
                                max_pooled_count_difference=max_hist_error,
                                max_home_gain_absolute_difference=err,
                                max_destination_gain_absolute_difference=float(np.max(np.abs(panel.A-replay_dest))),
                                max_weight_sum_error=max_weight_error, minimum_mapped_records=min_mapped)
        panel.c.close()
        del panel
        print(mapping, results[mapping], flush=True)
    return results


def main(workspace=None):
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    source, fresh, errors = archive_series()
    paths, mappings = mapping_summary()
    mixed = fresh[fresh.protocol.eq("mix") & fresh.field.isin(paths)]
    main = mixed[mixed.mo.between(FIRST, LAST)]
    assert len(main) == 110 and main.gain12_l6.notna().all()
    csv(main, "regression_gain_endpoints.csv")
    stats = []
    for path, q in mixed.groupby("field"):
        level = q.frontier_l0.dropna()
        gain = main[main.field.eq(path)].gain12_l6
        stats.append(dict(path=path, frontier_months=len(level), frontier_first=level.iloc[0],
                          frontier_last=level.iloc[-1], frontier_mean=level.mean(), frontier_sd=level.std(),
                          regression_months=len(gain), gain_mean=gain.mean(), gain_sd=gain.std(),
                          gain_min=gain.min(), gain_max=gain.max()))
    csv(pd.DataFrame(stats), "path_summary_statistics.csv")
    figures(mixed, paths)
    grid = main.pivot(index="field", columns="mo", values="gain12_l6")
    report = {"archive_rows": len(source), "replayed_rows": len(fresh), "archive_reconciliation": errors,
              "mapping_counts": mappings.groupby("mapping_type").mapped_subfields.sum().to_dict(),
              "regression_path_months": len(main),
              "input_sha256": {f.name: sha(f) for f in sorted(INPUT.glob("*.csv"))}}
    if workspace:
        report["regression_exposure_reconciliation"] = verify_panel(workspace.resolve(), grid)
        filename = "full_workspace_reconciliation.json"
    else:
        filename = "aggregate_reconciliation.json"
    (OUT / filename).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, help="Optional original OpenAlex workspace for the author-count and panel audit")
    main(parser.parse_args().workspace)
