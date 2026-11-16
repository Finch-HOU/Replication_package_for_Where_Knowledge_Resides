"""Verify the reorganized main tables against the original display pipeline."""

import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from scipy.stats import t


NUMBER_OR_STAR = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\^\{\*{1,3}\}")


def numeric_rows(tex):
    rows = []
    for line in tex.splitlines():
        if "&" not in line:
            continue
        vals = tuple(tuple(NUMBER_OR_STAR.findall(cell)) for cell in line.split("&")[1:])
        if any(vals):
            rows.append(vals)
    return rows


def verify_benchmarks(replication, display):
    crosswalk = display / "evidence/november07_table_crosswalk.csv"
    with crosswalk.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    for mapping in ("parent", "direct"):
        for model, label in (("proportional", "Architecture-moderated"), ("home_proportional", "Home-only")):
            path = replication / "data/benchmark" / mapping / model / "restricted_benchmark.json"
            rec = json.loads(path.read_text(encoding="utf-8"))
            scores = np.asarray(rec["scores_by_path"])
            covariance = scores.T @ scores * len(scores) / (len(scores) - 1)
            np.testing.assert_allclose(covariance, rec["covariance"]["capability"], atol=1e-22, rtol=1e-9)
            colmap = {"Observed Signed": "signed_observed", "Implied benchmark": "signed_proportional",
                      "Observed - implied": "signed_residual"}
            published = [r for r in rows if r["final_table"] == "Table 4"
                         and r["row"] == f"{mapping.title()} {label}"]
            for entry in published:
                col, stat = entry["column"], entry["statistic"]
                if col not in colmap or stat not in ("b", "se", "p"):
                    continue
                contrast = np.asarray(rec["linear_statistics"][colmap[col]])
                b = contrast @ np.array(rec["primitive_points"])
                se = np.sqrt(contrast @ covariance @ contrast)
                actual = {"b": b, "se": se, "p": 2 * t.sf(abs(b/se), len(scores)-1)}[stat]
                np.testing.assert_allclose(actual, float(entry["raw_value"]), rtol=1e-9, atol=1e-14)
    print("Table 5 proportional benchmarks replayed from archived primitive estimates and path scores.")


def audit(replication, manuscript):
    display = replication / "data/prior_display_inputs"
    with tempfile.TemporaryDirectory(prefix="archived_table_audit_") as temp:
        archived = Path(temp) / "display"
        shutil.copytree(display, archived)
        (archived / "tables").mkdir()
        subprocess.run([sys.executable, "-B", str(archived / "code/editorial_exhibits.py")], check=True)
        for current, prior in ((1, 1), (2, 3), (4, 4), (5, 5), (6, 6), (7, 7)):
            predecessor = (archived / f"tables/Table{prior}.tex").read_text(encoding="utf-8")
            if current == 7:
                predecessor = "\n".join(line for line in predecessor.splitlines()
                                        if not ("without core expert" in line or "with core expert" in line))
            present = (manuscript / f"tables/Table{current}.tex").read_text(encoding="utf-8")
            previous_rows = numeric_rows(predecessor)
            assert numeric_rows(present) == previous_rows, current
            counts = r"Candidate-months.*Parent\s+([\d,]+).*Direct\s+([\d,]+)"
            if current in (4, 5):
                assert re.findall(counts, present) == re.findall(counts, predecessor)
            print(f"Table {current}: verified {len(previous_rows)} ordered numeric rows and their stars.")
        verify_benchmarks(replication, archived)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replication", type=Path, required=True)
    parser.add_argument("--manuscript", type=Path, required=True)
    opts = parser.parse_args()
    audit(opts.replication.resolve(), opts.manuscript.resolve())
