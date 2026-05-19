#!/usr/bin/env python3
"""Verify the real-dataset-first final AutoSyncDSL submission."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "final_submission"
RESULTS = FINAL / "results"
TABLES = FINAL / "tables"
FIG = FINAL / "figures"


def exists_nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def main() -> None:
    checks = []
    required_files = [
        FINAL / "final_report.pdf",
        FINAL / "final_report.docx",
        FINAL / "final_report.tex",
        FINAL / "references.bib",
        FINAL / "README_FINAL_SUBMISSION.md",
        FINAL / "final_quality_audit.md",
        FINAL / "source_alignment_check.md",
        FINAL / "project_cleanup_log.md",
        FINAL / "dataset_summary" / "dataset_inventory.csv",
        RESULTS / "all_experiments.json",
        RESULTS / "correctness_results.json",
        RESULTS / "real_dataset_results.json",
        RESULTS / "synthetic_stress_results.json",
        RESULTS / "optimization_ablation_results.json",
    ]
    required_tables = [
        "table_real_dataset_inventory.csv",
        "table_real_correctness.csv",
        "table_real_latency.csv",
        "table_real_offset_batching.csv",
        "table_optimization_ablation.csv",
        "table_user_code_complexity.csv",
        "table_implementation_footprint.csv",
        "table_supplemental_synthetic_stress.csv",
        "table_literature_comparison.csv",
    ]
    for path in required_files:
        checks.append({"check": str(path.relative_to(ROOT)), "status": exists_nonempty(path)})
    for name in required_tables:
        checks.append({"check": f"tables/{name}", "status": exists_nonempty(TABLES / name)})
    for i in range(1, 12):
        stems = {
            1: "fig1_system_overview",
            2: "fig2_dsl_to_ir",
            3: "fig3_sync_semantics",
            4: "fig4_real_dataset_workflow",
            5: "fig5_real_alignment_summary",
            6: "fig6_real_latency_comparison",
            7: "fig7_real_offset_distribution",
            8: "fig8_optimization_ablation",
            9: "fig9_user_code_complexity",
            10: "fig10_supplemental_synthetic_robustness",
            11: "fig11_implementation_footprint",
        }
        stem = stems[i]
        for folder, ext in [("png", "png"), ("pdf", "pdf"), ("svg", "svg")]:
            checks.append({"check": f"figures/{folder}/{stem}.{ext}", "status": exists_nonempty(FIG / folder / f"{stem}.{ext}")})
    for name in ["fig1_system_overview.drawio", "fig2_dsl_to_ir.drawio", "fig3_runtime_optimization.drawio", "fig4_real_dataset_workflow.drawio"]:
        checks.append({"check": f"figures/drawio/{name}", "status": exists_nonempty(FIG / "drawio" / name)})

    data = json.loads((RESULTS / "all_experiments.json").read_text())
    checks.append({"check": "real dataset evaluated", "status": data["metadata"]["usable_real_sequences"] > 0})
    checks.append({"check": "real correctness rows present", "status": len(data["real_correctness"]) > 0})
    checks.append({"check": "synthetic supplemental rows present", "status": len(data["supplemental_synthetic_stress"]) > 0})

    latency_rows = list(csv.DictReader((TABLES / "table_real_latency.csv").open()))
    for row in latency_rows:
        match = next((x for x in data["real_latency"] if x["sequence"] == row["sequence"] and x["method"] == row["method"]), None)
        checks.append({"check": f"latency JSON/CSV match {row['method']}", "status": bool(match) and abs(float(match["mean_ms"]) - float(row["mean_ms"])) < 1e-9})

    tex = (FINAL / "final_report.tex").read_text(errors="ignore")
    forbidden = ["TODO", "TBD", "fake", "This is the abstract", "This is the introduction", "Y Y Y", "CS 790", "Domain-Specific Programming for AI", "May 2026", "final project report"]
    for token in forbidden:
        checks.append({"check": f"forbidden token absent: {token}", "status": token not in tex})
    checks.append({"check": "real results before synthetic", "status": tex.find("Real Dataset Inventory") < tex.find("Supplemental Synthetic Stress Tests")})
    checks.append({"check": "equations present", "status": tex.count("\\begin{equation}") >= 12})

    cited = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", tex):
        cited.update(x.strip() for x in group.split(","))
    bib = (FINAL / "references.bib").read_text(errors="ignore")
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    checks.append({"check": "citations resolve", "status": cited.issubset(bib_keys), "missing": sorted(cited - bib_keys)})

    passed = sum(1 for item in checks if item["status"])
    failed = [item for item in checks if not item["status"]]
    report = {"status": "pass" if not failed else "fail", "passed": passed, "failed": len(failed), "checks": checks}
    (RESULTS / "verification_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "passed": passed, "failed": len(failed)}, indent=2))
    if failed:
        for item in failed[:20]:
            print("FAILED:", item)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
