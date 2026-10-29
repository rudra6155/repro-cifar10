"""The results table in FINDINGS.md must be exactly what results/aggregate.csv renders to (no hand edits)."""

from pathlib import Path

import pytest

from repro.report import main as report_main

ROOT = Path(__file__).parents[1]


@pytest.mark.skipif(not (ROOT / "results/aggregate.csv").exists(), reason="no results yet")
def test_findings_and_readme_match_aggregate_csv():
    rc = report_main(["--results", str(ROOT / "results"), "--findings", str(ROOT / "FINDINGS.md"),
                      "--readme", str(ROOT / "README.md"), "--check"])
    assert rc == 0, "FINDINGS.md/README.md drifted from results/aggregate.csv; run `make report`"
