"""Format the seven main tables and link each printed estimate to archived inference."""

import csv
from decimal import Decimal
from functools import lru_cache
import json
from pathlib import Path
from types import SimpleNamespace

import editorial_core_tables
import editorial_support_tables


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "evidence/editorial_baseline"
RAW_NOTE = (
    r"Standard errors in parentheses use ten capability-path clusters and a $t(9)$ reference. "
    r"$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$ (raw, exploratory)."
)
LPM_NOTE = (
    r"LPM estimates and standard errors are scaled by 10,000. Coefficients give entries "
    r"per 100,000 candidate-months for a 0.10 gain change and a one-original-risk-SD "
    r"modifier difference."
)


@lru_cache(maxsize=None)
def read_csv(path):
    with (ROOT / path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


@lru_cache(maxsize=None)
def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"), parse_float=Decimal)


def exactly_one(rows):
    matches = list(rows)
    assert len(matches) == 1, matches
    return matches[0]


def legacy_bundle(table, panel, row, column):
    """Resolve a November 7 cell and its underlying archive pointers."""
    rows = read_csv("evidence/november07_table_crosswalk.csv")
    bundle = {}
    for line, old in enumerate(rows, 2):
        if (old["final_table"], old["panel"], old["row"], old["column"]) != (
            f"Table {table}", panel, row, column
        ):
            continue
        stat = old["statistic"]
        loc = old["coefficient_source"]
        src, col = loc.rsplit("/", 1) if "/" in loc else (loc, "")
        multiplier = Decimal(old["display_multiplier"] or "1")
        ppml = table == 4 and panel.startswith("Panel B")
        if stat == "p":
            source_scale = display_scale = "probability (0 to 1)"
        elif stat in ("b", "se"):
            source_scale = "PPML log rate" if ppml else "raw LPM incidence"
            display_scale = source_scale if ppml else "LPM incidence coefficient x 10000"
        else:
            source_scale = display_scale = "archived model or sample statistic"
        if stat == "value":
            # Some legacy metadata are already transformed in the crosswalk (e.g. rates).
            snapshot, locator, col = "evidence/november07_table_crosswalk.csv", str(line), "raw_value"
            convention = "CSV record including header"
            if old["notes"] == "multiply_100000":
                source_scale = display_scale = "entries per 100000 candidate-months"
        else:
            snapshot = old["source_snapshot"].replace("\\", "/")
            locator = src.removeprefix("csv_row:") if src.startswith("csv_row:") else src
            convention = "CSV data index, zero based" if src.startswith("csv_row:") else "JSON object pointer"
        outcome, term = column, row
        if table == 4 and not ppml:
            outcome, term = "Signed", f"{row} | {column}"
        elif table == 5:
            outcome, term = row, f"HomeGain x U | {column}"
        elif table == 7:
            outcome, term = row, "HomeGain x U"
        if stat == "value" and ("observations" in row or "Candidate-month" in row):
            source_scale = display_scale = "candidate-month observations"
        if table == 1 and panel.startswith("Panel A"):
            outcome = row
        assert stat not in bundle, (table, panel, row, column, stat)
        bundle[stat] = {
            "statistic": stat, "source_file": old["source_file"], "snapshot_file": snapshot,
            "source_locator": locator, "source_column": col, "source_locator_convention": convention,
            "specification": old["source_specification_id"], "outcome": outcome,
            "mapping": "parent" if "Parent" in (panel + row + column) else "direct",
            "term_or_contrast": term, "source_value": old["raw_value"],
            "source_scale": source_scale, "display_multiplier": str(multiplier),
            "display_scale": display_scale,
            "inference_family": "Raw two-sided capability-path t(9)" if stat in ("b", "se", "p")
            else "Sample/model metadata",
            "legacy_crosswalk_record": str(line),
            "underlying_source_pointer": old["coefficient_source"],
            "original_regression_file": old["original_regression_file"],
        }
    assert bundle, (table, panel, row, column)
    return bundle


def extension_bundle(location, mapping, outcome, term, specification):
    bundle = {
        r["statistic"]: r
        for r in INHERITED
        if (r["manuscript_location"], r["mapping"], r["outcome"], r["term_or_contrast"],
            r["specification"]) == (location, mapping, outcome, term, specification)
    }
    assert bundle, (location, mapping, outcome, term, specification)
    return bundle


def from_archive(row):
    path, loc, col = (row[k] for k in ("snapshot_file", "source_locator", "source_column"))
    if path.endswith(".csv"):
        offset = 0 if row.get("source_locator_convention") == "CSV data index, zero based" else 2
        return Decimal(read_csv(path)[int(loc) - offset][col])
    value = read_json(path)
    for part in loc.strip("/").split("/"):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return Decimal(str(value[col]))


def archived_raw(row):
    source = from_archive(row)
    claimed = Decimal(row["source_value"])
    assert abs(source - claimed) <= max(abs(source) * Decimal("1e-12"), Decimal("1e-20")), row
    return source


def stars_for(probability):
    assert 0 <= probability <= 1
    for cutoff, stars in (("0.01", "***"), ("0.05", "**"), ("0.10", "*")):
        if probability < Decimal(cutoff):
            return stars
    return ""


def display(row, location, role, *, places, grouping=False, stars=""):
    value = archived_raw(row) * Decimal(row["display_multiplier"])
    template = f"{',' if grouping else ''}.{places}f"
    shown = format(value, template)
    if not Decimal(shown.replace(",", "")):
        shown = shown.lstrip("-")
    DISPLAY_LEDGER.append(dict(row, manuscript_location=location, displayed_value=shown,
                               presentation_role=role, significance_stars=stars))
    return shown


def value(bundle, location, statistic="value", places=0, grouping=True):
    return display(bundle[statistic], location, "visible statistic", places=places, grouping=grouping)


def estimate(bundle, location, places=4, se_places=None, star_margin=0):
    p = bundle["p"]
    assert p["source_scale"] == "probability (0 to 1)"
    assert "Raw" in p["inference_family"] and "Holm" not in p["inference_family"]
    stars = stars_for(archived_raw(p))
    b = display(bundle["b"], location, "coefficient", places=places, stars=stars)
    se = display(bundle["se"], location, "standard error",
                 places=places if se_places is None else se_places)
    DISPLAY_LEDGER.append(dict(p, manuscript_location=location, displayed_value="",
                               presentation_role="raw p backing the stars", significance_stars=stars))
    mark = rf"\rlap{{$^{{{stars}}}$}}" if stars else ""
    margin = rf"\hspace{{{star_margin}pt}}" if star_margin else ""
    return rf"\makecell[c]{{\ensuremath{{{b}}}{mark}{margin}\\({se}){margin}}}"


def write(path, content):
    (ROOT / path).write_text(content.rstrip() + "\n", encoding="utf-8")


def heading(number, title):
    return rf"\noindent\textbf{{Table {number}}}\\[3pt]" + "\n" + (
        rf"\textbf{{{title}}}\par\vspace{{7pt}}" + "\n")


def panel(title):
    return rf"\noindent\textit{{{title}}}\par\vspace{{3pt}}" + "\n"


def note(contents):
    return (r"\vspace{6pt}{\fontsize{9}{10.8}\selectfont\noindent\textit{Notes.} "
            + contents + r"\par}" + "\n")


def tabular(columns, labels, rows):
    body = [" & ".join(row) + r" \\" if not isinstance(row, str) else row for row in rows]
    return "\n".join([
        r"\noindent\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}" + columns + "@{}}",
        r"\toprule", " & ".join(labels) + r" \\", r"\midrule",
        *body, r"\bottomrule", r"\end{tabular*}\par", "",
    ])


def sample_table():
    heading_a, heading_b = "Panel A Sample structure", "Panel B Descriptive statistics"
    out = heading(1, "Sample Construction and Activity-Specific Knowledge Measures")
    out += panel("Panel A. Frozen Class C risk sets and first entries")
    rows = []
    for label, places in [
        ("Eligible mapped core dyads", 0), ("Candidate destinations", 0), ("Candidate pairs", 0),
        ("Candidate-month observations", 0), ("Total first entries", 0), ("Specialist first entries", 0),
        ("No-Specialist first entries", 0), ("Total entry incidence per 100,000", 4),
        ("Specialist incidence per 100,000", 4), ("No-Specialist incidence per 100,000", 4),
    ]:
        values = [value(legacy_bundle(1, heading_a, label, m), f"Table 1/A/{label}/{m}",
                        places=places, grouping=places == 0) for m in ("Parent", "Direct")]
        rows.append([label, *values])
    out += tabular("lcc", ["", "Parent", "Direct"], rows) + r"\vspace{9pt}" + "\n"
    out += panel("Panel B. Knowledge measures and capability gains")
    rows = []
    for old, label in [
        ("Distributed knowledge coverage U", r"Distributed Knowledge Coverage ($U$)"),
        ("Best-member knowledge coverage", "Best-Member Knowledge Coverage"),
        ("Collective knowledge coverage C", r"Collective Knowledge Coverage ($C$)"),
        ("Shared knowledge coverage", "Shared Knowledge Coverage"),
        ("Knowledge gap G", r"Knowledge gap ($1-C$)"),
        ("Expertise distance", "Expertise distance"),
        ("HomeGain", "HomeGain"), ("DestinationGain", "DestinationGain"),
    ]:
        values = [value(legacy_bundle(1, heading_b, old, m), f"Table 1/B/{old}/{m}",
                        places=4, grouping=False)
                  for m in ("Parent mean", "Parent SD", "Direct mean", "Direct SD")]
        rows.append([label, *values])
    out += tabular("lcccc", ["", "Parent mean", "Parent SD", "Direct mean", "Direct SD"], rows)
    out += note(
        r"The sample consists of persistent dyads and destinations unfamiliar to both members "
        r"at end-2021. The July 2025--May 2026 risk sets retain all candidate-months, including "
        r"post-entry zeros. Panel B weights knowledge measures by candidate pair and gains by "
        r"candidate-month. Parent and Direct map the same ten capability paths.")
    write("tables/Table1.tex", out)


def donor_table(appendix=False):
    number = "F8" if appendix else 4
    out = heading(number, "Raw and Adjusted Inference for Donor Contrasts" if appendix
                  else "Actual versus Donor Knowledge Architecture")
    schemes = (
        ("field_shuffle", "Field shuffle"),
        ("matched", "Global matched donor"),
        ("class_c_matched", "Within-core Class C donor"),
    )
    for index, mapping in enumerate(("parent", "direct")):
        out += panel(f"Panel {'AB'[index]}. {mapping.title()} mapping")
        rows = []
        for scheme, label in schemes:
            for mode in ("separate", "joint"):
                bundle = {r["statistic"]: r for r in INHERITED
                          if r["manuscript_location"] == "Table 4" and r["mapping"] == mapping
                          and r["specification"].startswith(scheme + "/")
                          and r["term_or_contrast"] == f"{mode}_actual_minus_placebo"}
                assert len(bundle) == 6, (scheme, mode, mapping, bundle)
                holm = ("p_holm_placebo_contrasts" if scheme == "class_c_matched"
                        else "p_holm_mean_placebo_contrasts")
                location = f"Table {number}/{mapping}/{scheme}/{mode}"
                b = value(bundle, location, "actual minus donor", 4, False)
                se = value(bundle, location, "SE of actual minus donor", 4, False)
                delta = rf"\makecell[c]{{\ensuremath{{{b}}}\\({se})}}"
                p_adjusted = value(bundle, location, holm, 6 if appendix else 4, False)
                if appendix:
                    p_raw = value(bundle, location, "raw p of actual minus donor", 6, False)
                    cells = [delta, p_raw, p_adjusted]
                else:
                    actual = value(bundle, location, "actual coefficient", 4, False)
                    donor = value(bundle, location, "donor coefficient", 4, False)
                    cells = [delta, rf"\ensuremath{{{actual}}}", rf"\ensuremath{{{donor}}}", p_adjusted]
                rows.append([label if mode == "separate" else "", mode.title(), *cells])
            if scheme != "class_c_matched":
                rows.append(r"\addlinespace[3pt]")
        out += tabular("llccc" if appendix else "llcccc",
                       ["Donor design", "Fit", r"Actual $-$ donor",
                        *([r"Raw $p$", r"Holm $p$"] if appendix else
                          ["Actual", "Donor", r"Holm $p$"])], rows)
        if index == 0:
            out += r"\vspace{8pt}" + "\n"
    if appendix:
        out += note(
            r"Standard errors and probabilities use ten capability-path clusters and $t(9)$. "
            r"Each mapping adjusts four global-mean differences together and 34 draw-level "
            r"differences together. The within-core Class C results use the draw-level family. "
            r"Full archived values are linked in the numerical source ledger.")
    else:
        counts = [value(legacy_bundle(1, "Panel A Sample structure", "Candidate-month observations", m),
                        f"Table 4/{m}/candidate-months") for m in ("Parent", "Direct")]
        out += (rf"\vspace{{5pt}}\noindent\textit{{Candidate-months}} \quad Parent {counts[0]}"
                rf"\quad Direct {counts[1]}\par" + "\n")
        out += note(
            r"Outcome is No-Specialist First Entry. Coefficients are HomeGain $\times$ $U$ on "
            r"the Table~\ref{tab:table3} LPM scale. Pair, core-month, destination-month, and "
            r"modifier-month effects are included. Parentheses give difference standard errors "
            r"from aligned path scores (separate) or joint covariance (joint), using ten "
            r"capability paths and $t(9)$. Within each mapping, Holm adjusts four global mean "
            r"differences or 34 draw-level differences for Class C. Exact raw and adjusted "
            r"probabilities appear in Table~\ref{tab:donor_exact}.")
    path = "tables/donor_exact.tex" if appendix else "tables/Table4.tex"
    write(path, out)


INHERITED = read_csv("evidence/editorial_baseline/numerical_source_ledger.csv")
DISPLAY_LEDGER = []
API = SimpleNamespace(
    legacy=legacy_bundle, extension=extension_bundle, estimate=estimate, value=value,
    write=write, heading=heading, panel=panel, note=note, tabular=tabular,
    RAW_NOTE=RAW_NOTE, LPM_NOTE=LPM_NOTE,
)


def main():
    sample_table()
    editorial_core_tables.render(API)
    donor_table()
    editorial_support_tables.render(API)
    donor_table(appendix=True)
    retained = [r for r in INHERITED if r["manuscript_location"] not in
                {"Table 3 Panel A", "Table 3 Panel B", "Table 4"}]
    records = DISPLAY_LEDGER + retained
    columns = list(dict.fromkeys(key for r in records for key in r))
    with (ROOT / "numerical_source_ledger.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    audit = [r for r in DISPLAY_LEDGER if r["presentation_role"] == "raw p backing the stars"]
    for r in audit:
        assert r["significance_stars"] == stars_for(from_archive(r))
    print(f"Checked {len(audit)} raw-p star assignments against the archived p-values.")
    print(f"Recorded {len(DISPLAY_LEDGER)} editorial table entries and {len(records)} total ledger rows.")


if __name__ == "__main__":
    main()
