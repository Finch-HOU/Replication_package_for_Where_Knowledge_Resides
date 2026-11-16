"""Format the archived common-productivity, temporal, and expertise exhibits."""

MAPS = ("Parent", "Direct")
BENCHMARK = "Panel A Common proportional-productivity benchmarks"
OFFSET_PPML = "Panel B Shared-factor offset PPML"
EXPERTISE = "Marginal outcome definitions and formal contrasts"
LPM_EFFECTS = "Pair, core-month, and destination-month fixed effects and modifier-month slopes"


def full_risk_count(api, mapping, location):
    return api.value(api.legacy(7, EXPERTISE, "Candidate-month N", mapping), location)


def productivity(api):
    title = api.heading(5, "Mechanical Common-Productivity Alternatives")
    panel_a = api.panel("Panel A. Common proportional-productivity benchmarks")
    rows = []
    for benchmark in ("Architecture-moderated", "Home-only"):
        for mapping in MAPS:
            source_row = f"{mapping} {benchmark}"
            where = f"Table 5 Panel A | {benchmark} | {mapping}"
            cells = []
            for column in ("Observed - implied", "Observed Signed", "Implied benchmark"):
                bundle = api.legacy(4, BENCHMARK, source_row, column)
                if column == "Observed - implied":
                    cells.append(api.estimate(bundle, f"{where} | Observed minus implied"))
                else:
                    cells.append(api.value(bundle, f"{where} | {column}", statistic="b",
                                           places=4, grouping=False))
            rows.append([benchmark, mapping, *cells])
    panel_a += api.tabular("llccc", [
        "Benchmark", "Mapping", r"\textbf{Observed $-$ implied}", "Observed", "Implied",
    ], rows)
    risks = [full_risk_count(api, mapping, f"Table 5 Panel A | {mapping} | Candidate-months")
             for mapping in MAPS]
    panel_a += (rf"\vspace{{4pt}}\noindent\textit{{Candidate-months}} \quad Parent {risks[0]} "
                rf"\quad Direct {risks[1]}\par\vspace{{9pt}}" + "\n")

    panel_b = api.panel("Panel B. Shared-factor PPML with historical offsets")
    rows = []
    for label, source_column in [
        (r"\textbf{Specialist $-$ No-Specialist}", "Formal S - No"),
        ("No-Specialist", "No-Specialist"),
        ("Specialist", "Specialist"),
    ]:
        estimates = [
            api.estimate(api.legacy(4, OFFSET_PPML, f"{mapping} HomeGain x U", source_column),
                         f"Table 5 Panel B | {mapping} | {source_column}")
            for mapping in MAPS
        ]
        rows.append([label, *estimates])
    rows.append([r"\addlinespace Stacked likelihood observations", *[
        api.value(api.legacy(4, OFFSET_PPML, f"{m} stacked likelihood N", "Specialist"),
                  f"Table 5 Panel B | {m} | Stacked likelihood observations")
        for m in MAPS
    ]])
    for label, source_column in [
        ("No-Specialist first entries", "No-Specialist"),
        ("Specialist first entries", "Specialist"),
        ("Total first entries", "Formal S - No"),
    ]:
        rows.append([label, *[
            api.value(api.legacy(4, OFFSET_PPML, f"{m} first-entry event cells", source_column),
                      f"Table 5 Panel B | {m} | {label}") for m in MAPS
        ]])
    rows.append([r"Pseudo $R^2$", *[
        api.value(api.legacy(4, OFFSET_PPML, f"{m} pseudo R squared", "Specialist"),
                  f"Table 5 Panel B | {m} | Pseudo R-squared", places=4, grouping=False)
        for m in MAPS
    ]])
    panel_b += api.tabular("lcc", [r"HomeGain $\times$ $U$", *MAPS], rows)
    notes = api.note(
        r"Panel A compares the observed Signed slope (Specialist minus No-Specialist first entry) "
        r"for HomeGain $\times$ $U$ with each calibrated slope. The LPM estimates are multiplied by "
        r"10,000 and retain Table~\ref{tab:table3}, Panel A controls and fixed effects. Differences "
        r"use joint covariance, conditional on the cross-fitted historical rates. Panel B uses "
        r"unscaled log-rate slopes, a shared core-month effect, outcome-specific destination-month "
        r"effects and modifier-month slopes, and historical offsets in place of pair effects. The profiled "
        r"likelihood retains all first events. " + api.RAW_NOTE
    )
    api.write("tables/Table5.tex", title + panel_a + panel_b + notes)


def historical(api):
    title = api.heading(6, "Historical Temporal Comparisons")
    panels = []
    columns = ("Current - historical", "Historical marker", "Current trajectory")
    for letter, mapping in zip("AB", MAPS):
        source_panel = f"Panel {letter} {mapping}"
        rows = []
        for label, source_row in [
            (r"\textbf{No-Specialist}", "No-Specialist"),
            ("Signed", "Signed S - No"),
            ("Total", "Total"),
            ("Specialist", "Specialist"),
        ]:
            cells = [
                api.estimate(api.legacy(5, source_panel, source_row, col),
                             f"Table 6 Panel {letter} | {source_row} | {col}")
                for col in columns
            ]
            rows.append([label, *cells])
        for label, source_row in [("Candidate-months", "Candidate-month N per period"),
                                  ("Total first entries", "Total entry events")]:
            if label == "Candidate-months":
                label = r"\addlinespace " + label
            values = [api.value(api.legacy(5, source_panel, source_row, col),
                                f"Table 6 Panel {letter} | {source_row} | {col}")
                      for col in columns[1:]]
            rows.append([label, "", *values])
        panels.append(api.panel(f"Panel {letter}. {mapping}") + api.tabular("lccc", [
            r"First-entry outcome", r"\textbf{Current $-$ historical}",
            r"\makecell[c]{Historical marker\\Jan.--Oct.\ 2022}",
            r"\makecell[c]{Current trajectory\\July 2025--Apr.\ 2026}",
        ], rows))
    notes = api.note(
        r"Estimates are HomeGain $\times$ $U$ slopes in first-entry LPMs. Signed is Specialist "
        r"minus No-Specialist. " + api.LPM_NOTE + " " + LPM_EFFECTS +
        r" are matched across windows with period-specific nuisance terms. Historical marker "
        r"replays the later trajectories by relative month. Differences use joint cross-period "
        r"covariance. The Parent Signed difference is borderline. " + api.RAW_NOTE
    )
    api.write("tables/Table6.tex", title + (r"\vspace{9pt}" + "\n").join(panels) + notes)


def robustness(api):
    title = api.heading(7, "Entry-Type and Specialist-Classification Robustness")
    groups = [
        ("Authorship at No-Specialist first entry", [
            (r"\textbf{Exact core $-$ expanded}", "Exact minus expanded No-Specialist"),
            ("Exact core", "Exact-core No-Specialist"),
            ("Expanded authorship", "Expanded-author No-Specialist"),
        ]),
        ("Prior-year destination expertise within the core", [
            ("No-Specialist, without core expert", "No-Specialist without prior-year core expert"),
            ("No-Specialist, with core expert", "No-Specialist with prior-year core expert"),
        ]),
        ("Coverage of recorded author histories", [
            ("No-Specialist, complete histories", "No-Specialist complete recorded histories"),
            ("No-Specialist, uncertain histories", "No-Specialist uncertain histories"),
        ]),
        ("Outside-specialist classification", [
            (r"\textbf{Signed, current $-$ frozen}", "Signed current minus frozen definition"),
            ("Signed, frozen definition", "Signed frozen specialist definition"),
        ]),
    ]
    rows = []
    for i, (group, items) in enumerate(groups):
        rows.append([r"\addlinespace \multicolumn{5}{@{}l}{\textit{" + group + "}}"
                     if i else r"\multicolumn{5}{@{}l}{\textit{" + group + "}}"])
        for label, source_row in items:
            where = f"Table 7 | {source_row}"
            estimates = [api.estimate(api.legacy(7, EXPERTISE, source_row, m), f"{where} | {m}")
                         for m in MAPS]
            counts = [
                api.value(api.legacy(7, EXPERTISE, source_row, f"Events {m}"),
                          f"{where} | {m} event cells")
                for m in MAPS
            ]
            rows.append([r"\hspace{0.6em}" + label, *estimates, *counts])
    rows.append([r"\addlinespace Candidate-months", *[
        full_risk_count(api, m, f"Table 7 | {m} | Candidate-months") for m in MAPS
    ], "", ""])
    table = api.tabular("lcccc", [
        r"HomeGain $\times$ $U$", "Parent", "Direct", "Parent events", "Direct events",
    ], rows)
    notes = api.note(
        r"First-entry outcomes use the complete candidate risk set and Table~\ref{tab:table3}, "
        r"Panel A controls and fixed effects. Rows partition outcomes on this common denominator. "
        r"No-Specialist allows additional coauthors without demonstrated destination expertise. "
        r"Signed is Specialist minus No-Specialist. Current expertise uses strictly prior "
        r"publication years. Frozen expertise uses the end-2021 record. Contrast event counts "
        r"give nonzero outcome cells. Formal differences use joint covariance. " +
        api.LPM_NOTE + " " + api.RAW_NOTE
    )
    api.write("tables/Table7.tex", title + table + notes)


def render(api):
    productivity(api)
    historical(api)
    robustness(api)
