"""Rebuild the archived GenAI capability paths using the original benchmark inputs."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def reproduce(benchmark_inputs, out):
    measurement = ROOT / "measurement"
    reference = json.loads((measurement / "reference/aggregate_reconciliation.json").read_text(encoding="utf-8"))
    hashes = reference["input_sha256"]
    benchmark_files = {"field_monthly_scores.csv", "field_month_proxies.csv"}
    with tempfile.TemporaryDirectory(prefix="where_knowledge_resides_measurement_") as scratch:
        stage = Path(scratch)
        task = stage / "02_Measurement"
        shutil.copytree(measurement / "code", task / "code")
        inputs = task / "inputs"
        inputs.mkdir()
        for name, digest in hashes.items():
            parent = benchmark_inputs if name in benchmark_files else measurement / "inputs"
            file = parent / name
            if not file.is_file():
                raise FileNotFoundError(f"Required archived input is missing: {file}")
            data = file.read_bytes()
            if sha256(data).hexdigest() != digest:
                raise ValueError(f"SHA-256 differs from the archived input: {file}")
            (inputs / name).write_bytes(data)
        environment = {**os.environ, "MPLCONFIGDIR": str(stage / "matplotlib")}
        for script in ("audit_measurement.py", "render_appendix_tables.py"):
            subprocess.run([sys.executable, "-B", str(task / "code" / script)],
                           cwd=stage, env=environment, check=True)
        outputs = task / "outputs"
        report = json.loads((outputs / "aggregate_reconciliation.json").read_text(encoding="utf-8"))
        assert report["regression_path_months"] == 110
        assert report["mapping_counts"] == reference["mapping_counts"]
        shutil.copytree(outputs, out, dirs_exist_ok=True)
        for part in ("figures", "measurement_tables"):
            shutil.copytree(stage / "01_Manuscript" / part, out / part, dirs_exist_ok=True)
    print(f"Capability measurement replay written to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "generated/measurement")
    options = parser.parse_args()
    reproduce(options.benchmark_inputs.resolve(), options.out.resolve())
