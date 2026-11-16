"""Replay and verify the seven PLOS ONE main tables from archived inputs."""

import argparse
from pathlib import Path
import re
import shutil
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

import audit_existing
import replay_table3


def numeric_cells(text, separate_standard_errors=False):
    """Read displayed numbers and stars in cell order, joining split SE rows."""
    text = re.sub(r"\\(?:fontsize|hspace|vspace|addlinespace)(?:\{[^}]+\}|\[[^]]+\])+", "", text)
    rows = []
    for line in text.splitlines():
        if "&" not in line or not line.endswith(r"\\"):
            continue
        label, *cells = line.split("&")
        values = [tuple(audit_existing.NUMBER_OR_STAR.findall(cell)) for cell in cells]
        if separate_standard_errors and not label.strip() and re.match(r"\s*\(\d+\.\d+\)", cells[0]):
            rows[-1] = [estimate + se for estimate, se in zip(rows[-1], values, strict=True)]
        elif any(values):
            rows.append(values)
    return rows


def reproduce(out):
    reference = ROOT / "reference"
    with tempfile.TemporaryDirectory(prefix="where_knowledge_resides_") as scratch:
        stage = Path(scratch)
        shutil.copytree(reference / "tables", stage / "tables")
        moments = stage / "table3"
        moments.mkdir()
        for mapping in ("parent", "direct"):
            for suffix in ("a1_moments.npz", "a1_metadata.json"):
                name = f"{mapping}_{suffix}"
                shutil.copyfile(ROOT / "data/table3" / name, moments / name)

        replay_table3.main(moments, stage)
        text = lambda path: path.read_text(encoding="utf-8")
        assert text(stage / "tables/Table3.tex") == text(reference / "tables/Table3.tex")
        assert text(stage / "partition_values.tex") == text(reference / "partition_values.tex")
        audit_existing.audit(ROOT, stage)

        for number in range(1, 8):
            name = f"Table{number}.tex"
            expected = text(stage / "tables" / name)
            plos = text(reference / "plos_tables" / name)
            assert numeric_cells(expected) == numeric_cells(plos, number in (2, 3)), name
        print("Tables 1-7 match the PLOS ONE manuscript's ordered numeric cells and stars.")

        out.mkdir(parents=True, exist_ok=True)
        shutil.copytree(stage / "tables", out / "tables", dirs_exist_ok=True)
        shutil.copytree(reference / "plos_tables", out / "plos_tables", dirs_exist_ok=True)
        shutil.copyfile(stage / "partition_values.tex", out / "partition_values.tex")
        for name in ("table3_estimates.csv", "table3_results.json"):
            shutil.copyfile(moments / name, out / name)
    print(f"Verified tables and Table 3 inference written to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "generated")
    reproduce(parser.parse_args().out.resolve())
