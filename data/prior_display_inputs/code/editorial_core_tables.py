"""Present the conventional model and the two archived architecture models."""

from pathlib import Path
import re


MAPPINGS = ("parent", "direct")
OUTCOMES = ("total", "specialist", "no_specialist", "signed")


def line(*cells):
    return " & ".join(cells) + r" \\"


def span(width, content, alignment="c"):
    return rf"\multicolumn{{{width}}}{{{alignment}}}{{{content}}}"


def mapping_table(labels, rows):
    width = len(labels)
    group_gap = r"@{\hspace{10pt}\extracolsep{\fill}}" if width == 4 else ""
    right_pad = r"@{\hspace{9pt}}" if width == 4 else "@{}"
    return "\n".join([
        r"\noindent\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l"
        + "c" * width + group_gap + "c" * width + right_pad + "}",
        r"\toprule",
        line("", span(width, "Parent mapping"), span(width, "Direct mapping")),
        rf"\cmidrule(lr){{2-{width + 1}}}\cmidrule(lr){{{width + 2}-{2 * width + 1}}}",
        line("", *labels, *labels),
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular*}\par",
    ]) + "\n"


def conventional(api):
    regressors = [
        ("HomeGain x Knowledge gap G", r"HomeGain $\times$ knowledge gap"),
        ("HomeGain x Expertise distance", r"HomeGain $\times$ expertise distance"),
        ("HomeGain x Directional applicability", r"HomeGain $\times$ directional applicability"),
    ]
    panels = ("Panel A Parent mapping", "Panel B Direct mapping")
    outcomes = ("Total", "No-Specialist")

    def cells(row, formatter):
        return [
            formatter(api.legacy(2, panel, row, outcome),
                      f"Table 2 | {panel} | {row} | {outcome}")
            for panel in panels for outcome in outcomes
        ]

    rows = [line(label, *cells(term, api.estimate)) for term, label in regressors]
    rows.extend([
        r"\midrule",
        line("Observations", *cells("Candidate-month observations", api.value)),
        line("First entries", *cells("Nonzero outcome cells", api.value)),
        r"\addlinespace[3pt]",
        line("Fixed effects", span(4, "Pair, core-month, destination-month")),
        line(r"Modifier $\times$ month slopes", span(4, "Gap, distance, applicability")),
        line("DestinationGain controls", span(4, "Gap, distance, applicability, log breadth, log productivity")),
    ])
    return (
        api.heading(2, "Conventional Knowledge Characteristics and AI Capability")
        + mapping_table(outcomes, rows)
        + api.note(
            r"Outcomes are Total and No-Specialist First Entry. Core-month and destination-month "
            r"effects absorb HomeGain and DestinationGain, respectively. LPM estimates and standard "
            r"errors are scaled by 10,000. " + api.RAW_NOTE
            + r" Full estimates appear in Table~\ref{tab:conventional_full}."
        )
    )


def architecture(api):
    models = [
        ("A", "Original architecture specification", "frozen_Table3"),
        ("B", "Unified knowledge specification (A1)", "a1_unified"),
    ]
    rows = []
    for letter, title, specification in models:
        source = f"Table 3 Panel {letter}"
        if letter == "B":
            rows.extend([r"\addlinespace[5pt]", r"\midrule"])
        rows.extend([line(span(9, rf"\textit{{Panel {letter}. {title}}}", "@{}l")),
                     r"\addlinespace[2pt]"])
        for term, label in [
            ("Distributed_Home", r"\textbf{HomeGain $\times$ $U$}"),
            ("Distributed_Destination", r"DestinationGain $\times$ $U$"),
        ]:
            cells = [
                api.estimate(
                    api.extension(source, mapping, outcome, term, specification),
                    f"{source} | {mapping} | {term} | {outcome}",
                    star_margin=8,
                )
                for mapping in MAPPINGS for outcome in OUTCOMES
            ]
            rows.append(line(label, *cells))

    counts = {
        "total": "Total first entries",
        "specialist": "Specialist first entries",
        "no_specialist": "No-Specialist first entries",
        "signed": "Total first entries",
    }

    def sample(row, mapping, outcome=""):
        return api.value(api.legacy(1, "Panel A Sample structure", row, mapping.title()),
                         f"Table 3 | {mapping} | {row}" + (f" | {outcome}" if outcome else ""))

    rows.extend([
        r"\midrule",
        line("Observations", *[span(4, sample("Candidate-month observations", m)) for m in MAPPINGS]),
        line("Nonzero cells", *[sample(counts[o], m, o) for m in MAPPINGS for o in OUTCOMES]),
        r"\addlinespace[3pt]",
        line("Fixed effects", span(8, "Pair, core-month, destination-month in both panels")),
        line(r"Month slopes", span(8, r"Gap, distance, $U$, shared coverage in both panels")),
        line("Applicability", span(8, "Both gain interactions and month slopes in Panel B only")),
    ])
    return (
        api.heading(3, "Knowledge Architecture: Original Baseline and Unified Specification")
        + mapping_table(("Total", "Specialist", r"\makecell[c]{No-\\Specialist}", "Signed"), rows)
        + api.note(
            r"$U$ is Distributed Knowledge Coverage. Total is Specialist plus No-Specialist First Entry. "
            r"Signed is Specialist minus No-Specialist. "
            r"Both panels include gap, distance, and shared coverage interacted with both gains, "
            r"and log breadth and log productivity interacted with DestinationGain. "
            r"Core-month and destination-month fixed effects absorb the gain levels. "
            + api.LPM_NOTE + " " + api.RAW_NOTE
            + r" Tables~\ref{tab:locked_full} and \ref{tab:a1_full} report the complete estimates."
        )
    )


def conventional_archive():
    archived = Path(__file__).resolve().parents[1] / "evidence/november07_tables/Table2.tex"
    original = archived.read_text(encoding="utf-8")
    formatted = re.sub(
        r"\\textbf\{Table 2(?=[}.])", lambda _: r"\textbf{Table F7", original
    )
    formatted = formatted.replace(
        "Baseline Specification: AI Capability Progress and Unfamiliar-Domain Entry",
        "Conventional Knowledge Characteristics and AI Capability (Complete Estimates)",
    )
    formatted = formatted.replace(
        "Tables 2 and 3 are not nested: Table 3 replaces applicability with distribution/shared controls; changes in coefficients or p-values are not an incremental-effect test.",
        "The conventional and original architecture specifications use different modifier sets. Unified A1 in Table 3, Panel B, includes both sets.",
    )
    bodies = r"\\begin\{tabular\}.*?\\end\{tabular\}"
    assert re.findall(bodies, formatted, re.S) == re.findall(bodies, original, re.S)
    return formatted


def render(api):
    api.write("tables/Table2.tex", conventional(api))
    api.write("tables/Table3.tex", architecture(api))
    api.write("tables/conventional_full.tex", conventional_archive())
