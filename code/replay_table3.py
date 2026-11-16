"""Reproduce Table 3 from anonymous path-level sufficient statistics."""

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


OUTCOMES = ("neither", "at_least_one")
MODIFIERS = {
    "Gap": "Knowledge gap", "Distance": "Expertise distance",
    "Distributed": "$U$", "Shared": "Shared coverage",
    "Applicability": "Directional applicability", "LogBreadth": "Frozen log breadth",
    "LogProductivity": "Frozen log productivity",
}


def infer(b, influence, df):
    se = float(np.linalg.norm(influence))
    statistic = float(b / se)
    radius = t.ppf(.975, df) * se
    return dict(b=float(b), se=se, t=statistic, p=float(2 * t.sf(abs(statistic), df)),
                ci_lo=float(b - radius), ci_hi=float(b + radius))


def wild_score(q, h, beta, inv, focal, adjust, observed_t):
    """Exhaustive null-restricted Rademacher bootstrap of projected path scores."""
    restricted = beta - inv[:, focal] * beta[focal] / inv[focal, focal]
    np.testing.assert_allclose(restricted[focal], 0, atol=1e-16)
    restricted_scores = h - q @ restricted
    weights = np.array(list(itertools.product((-1, 1), repeat=len(q))), dtype=float)
    flipped = weights[:, :, None] * restricted_scores
    draws = flipped.sum(1) @ inv.T
    boot_residual_scores = flipped - np.einsum("gkl,rl->rgk", q, draws)
    boot_se = adjust * np.linalg.norm(boot_residual_scores @ inv[focal], axis=1)
    assert np.all(boot_se > 0)
    stats = draws[:, focal] / boot_se
    return float(np.mean(np.abs(stats) >= abs(observed_t) - 1e-10)), len(weights)


def replay(data, mapping):
    meta = json.loads((data / f"{mapping}_a1_metadata.json").read_text(encoding="utf-8"))
    with np.load(data / f"{mapping}_a1_moments.npz") as mom:
        q, h = mom["Q_by_path"], mom["xy_by_path"]
    scale = np.sqrt(np.diag(q.sum(0)))
    inv = np.linalg.inv(q.sum(0) / np.outer(scale, scale)) / np.outer(scale, scale)
    beta = inv @ h.sum(0)
    scores = h - q @ beta
    adjust = np.sqrt(meta["CR1"] * len(q) / (len(q) - 1))
    influence = adjust * (inv @ scores)
    assert len(q) == 10
    diff = np.array([1., -1.])
    focal = meta["terms"].index("Distributed_Home")
    out = {}
    rows = []
    for k, term in enumerate(meta["terms"]):
        out[term] = [infer(beta[k, j], influence[:, k, j], len(q)-1) for j in range(2)]
        rows.extend(dict(mapping=mapping, term=term, outcome=o, **values)
                    for o, values in zip(OUTCOMES, out[term]))
    contrast = infer(beta[focal] @ diff, influence[:, focal] @ diff, len(q)-1)
    covariance = influence[:, focal].T @ influence[:, focal]
    assert np.isclose(contrast["se"]**2, covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1])
    contrast["wild_score_p"], contrast["wild_draws"] = wild_score(
        q, h @ diff, beta @ diff, inv, focal, adjust, contrast["t"])
    meta["focal_joint_covariance"] = covariance.tolist()
    meta["contrast"] = contrast
    meta["estimates"] = out
    rows.append(dict(mapping=mapping, term="Distributed_Home", outcome="neither_minus_at_least_one",
                     **{k: contrast[k] for k in ("b", "se", "t", "p", "ci_lo", "ci_hi")}))
    return meta, rows


def number(value, places=4, multiplier=1):
    return f"{value * multiplier:.{places}f}"


def estimate(rec):
    star = next((s for threshold, s in ((.01, "***"), (.05, "**"), (.10, "*"))
                 if rec["p"] < threshold), "")
    suffix = "^{" + star + "}" if star else ""
    return r"\makecell[c]{$" + number(rec["b"], multiplier=10000) + suffix + r"$\\(" + number(
        rec["se"], multiplier=10000) + ")}"


def render(metas):
    labels = {
        "Distributed_Home": r"\textbf{HomeGain $\times U$}",
        "Distributed_Destination": r"DestinationGain $\times U$",
    }
    for modifier, label in MODIFIERS.items():
        for gain in ("Home", "Destination"):
            labels.setdefault(f"{modifier}_{gain}", f"{gain}Gain " + r"$\times$ " + label)
    title = r"""\noindent\textbf{Table 3}\\[3pt]
\textbf{Heterogeneity by Prior Destination Expertise within the Core}\par\vspace{5pt}
\noindent\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccc@{}}
\toprule
 & \multicolumn{2}{c}{Parent mapping} & \multicolumn{2}{c}{Direct mapping}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
 & Neither & At least one & Neither & At least one\\
\midrule
"""
    row = lambda items: " & ".join(items) + " \\\\\n"
    table = title
    order = ["Distributed_Home", "Distributed_Destination"]
    order.extend(k for k in metas[0]["terms"] if k not in order)
    for index, term in enumerate(order):
        if index == 2:
            table += r"\addlinespace[2pt]\multicolumn{5}{@{}l}{\textit{Other A1 gain interactions}}\\" + "\n"
        table += row([labels[term], *(estimate(rec) for m in metas for rec in m["estimates"][term])])
    table += r"\midrule" + "\n"
    table += row(["No-Specialist events", *(f"{m['first_entry_events'][o]:,}" for m in metas for o in OUTCOMES)])
    table += row(["Candidate-months by status", *(f"{sum(y[o] for y in m['timing'].values()):,}"
                                                 for m in metas for o in OUTCOMES)])
    for label, vals in [
        ("Candidate-months per regression", [f"{m['full_risk_candidate_months']:,}" for m in metas]),
        ("Pair, core-month, destination-month effects", ["Yes", "Yes"]),
        ("A1 modifier-by-month slopes", ["Yes", "Yes"]),
    ]:
        table += row([label, *(rf"\multicolumn{{2}}{{c}}{{{v}}}" for v in vals)])
    table += r"\midrule\multicolumn{5}{@{}l}{\textit{Formal contrast. Neither minus at least one}}\\" + "\n"
    ctr = [m["contrast"] for m in metas]
    for label, vals in [
        ("Difference (SE)", ["$" + number(c["b"], multiplier=10000) + "$ (" +
                             number(c["se"], multiplier=10000) + ")" for c in ctr]),
        ("$t(9)$ statistic", [number(c["t"], 3) for c in ctr]),
        ("Two-sided $p$", [number(c["p"]) for c in ctr]),
        (r"95\% confidence interval", ["[" + number(c["ci_lo"], multiplier=10000) + ", "
                                       + number(c["ci_hi"], multiplier=10000) + "]" for c in ctr]),
        ("Restricted path-score bootstrap $p$", [number(c["wild_score_p"]) for c in ctr]),
    ]:
        table += row([label, *(rf"\multicolumn{{2}}{{c}}{{{v}}}" for v in vals)])
    return table + r"""\bottomrule
\end{tabular*}\par
\vspace{6pt}{\fontsize{8.6}{10}\selectfont\noindent\textit{Notes.}
The dependent variables partition No-Specialist First Entry by whether a core member had published in the destination in a strictly earlier calendar year. Each equation uses the full Table~\ref{tab:table2}, Panel B risk set and A1 design. Candidate-month counts by status describe support. Parentheses contain standard errors. Estimates and standard errors are multiplied by 10,000. The contrast uses joint covariance, clustered by ten capability paths with a $t(9)$ reference. Stars use raw $p$ values ($^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$). The bootstrap exhausts the 1,024 null-restricted Rademacher path-score draws and is a sensitivity check under path-score symmetry and independence. The two mappings share paths.\par}
"""


def main(data, manuscript):
    pairs = [replay(data, m) for m in ("parent", "direct")]
    metas = [meta for meta, _ in pairs]
    rows = [row for _, records in pairs for row in records]
    with (data / "table3_estimates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (data / "table3_results.json").write_text(json.dumps(metas, indent=2), encoding="utf-8")
    (manuscript / "tables/Table3.tex").write_text(render(metas), encoding="utf-8")
    macros = []
    for m in metas:
        mapping = m["mapping"].title()
        values = {
            "Neither": number(m["estimates"]["Distributed_Home"][0]["b"], multiplier=10000),
            "AtLeastOne": number(m["estimates"]["Distributed_Home"][1]["b"], multiplier=10000),
            "Difference": number(m["contrast"]["b"], multiplier=10000),
            "ContrastP": number(m["contrast"]["p"]),
            "WildP": number(m["contrast"]["wild_score_p"]),
        }
        for suffix, value in values.items():
            macros.append(r"\expandafter\def\csname result@" + mapping + suffix + r"\endcsname{" + value + "}")
    (manuscript / "partition_values.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    for m in metas:
        print(m["mapping"], m["estimates"]["Distributed_Home"], m["contrast"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--manuscript", type=Path, required=True)
    opts = p.parse_args()
    main(opts.data, opts.manuscript)
