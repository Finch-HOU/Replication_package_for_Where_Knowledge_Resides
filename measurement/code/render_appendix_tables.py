"""Typeset the mapping, coverage and summary exhibits from the audit outputs."""
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT.parent / "01_Manuscript/measurement_tables"
OUT = ROOT / "outputs"


def write(name, header, rows, columns):
    lines = [rf"\begin{{tabular}}{{{columns}}}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / (name + ".tex")).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(OUT / "ten_path_mapping.csv")
    rows = []
    for path, group in m.groupby("capability_path"):
        pp, dd = (group[group.mapping_type.eq(t)].iloc[0] for t in ("parent", "direct"))
        label = path.capitalize()
        fields = pp.openalex_parent_fields.replace(" | ", "; ")
        direct = dd.openalex_parent_fields.replace(" | ", "; ")
        oa = fields if fields == direct else "P: " + fields + ". D: " + direct + "."
        counts = f"{pp.mapped_subfields} P, {dd.mapped_subfields} D"
        rows.append((label, label, oa, "P / D", r"Single domain\newline " + counts))
    write("mapping", ("Path", "MMLU-Pro domain", "OpenAlex parent fields", "Mapping type", "Construction rule"),
          rows, r"@{}p{0.68in}p{0.78in}p{3.02in}p{0.53in}p{0.97in}@{}")

    s = pd.read_csv(OUT / "path_summary_statistics.csv")
    rows = []
    for r in s.itertuples(index=False):
        rows.append((r.path.capitalize(), *(f"{v*100:.2f}" for v in (
            r.frontier_first, r.frontier_last, r.frontier_mean, r.frontier_sd,
            r.gain_mean, r.gain_sd, r.gain_min, r.gain_max))))
    write("summary", ("Path", "First", "Last", r"$\bar F$", "SD",
                      r"$\bar g$", "SD", "Min", "Max"),
          rows, "@{}lrrrrrrrr@{}")

    c = pd.read_csv(OUT / "domain_month_coverage.csv")
    assert c.groupby(["protocol", "month"]).n_model_versions.nunique().eq(1).all()
    assert c.groupby(["protocol", "month"]).has_frontier.nunique().eq(1).all()
    c = c[c.protocol.eq("mix") & c.has_frontier].drop_duplicates("month")
    c["date"] = pd.to_datetime(c.month)
    left = c[c.date.dt.year.eq(2024)]
    right = c[c.date.dt.year.eq(2025)]
    a = [(r.date.strftime("%b %Y"), str(r.n_model_versions)) for r in left.itertuples()]
    b = [(r.date.strftime("%b %Y"), str(r.n_model_versions)) for r in right.itertuples()]
    b += [("Dec 2025", "---")]
    write("coverage", ("Month", "Reported versions", "Month", "Reported versions"),
          [i + j for i, j in zip(a, b)], "@{}lr@{\\quad}lr@{}")


if __name__ == "__main__":
    main()
