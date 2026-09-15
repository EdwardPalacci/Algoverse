"""Family-level metrics by prompt condition, pooled over each family's generations.

Produces paper_assets/tables/quality_control/prompt_condition_family_metrics.csv
with the same estimator the paper's Section 4.2 and Figure 4 use: every metric
is computed over all generations of a family under a condition (not an average
of per-model values), and AURC is the expected value over orderings of tied
confidences.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_paper_assets as G

OUT = Path(__file__).resolve().parents[1] / "paper_assets" / "tables" / "quality_control" / "prompt_condition_family_metrics.csv"
FIELDS = [
    "family", "prompt_condition", "N", "accuracy", "mean_confidence",
    "expected_calibration_error", "area_under_risk_coverage", "high_confidence_wrong_rate",
]


def main() -> None:
    rows, _ = G.load_all_rows()
    output = []
    for (family, condition), group in sorted(G.grouped(rows, ("model_family", "prompt_condition")).items()):
        m = G.metric_row(group)
        output.append({
            "family": family,
            "prompt_condition": condition,
            "N": m["N"],
            **{k: f"{m[k]:.6f}" for k in FIELDS[3:]},
        })
    G.write_csv(OUT, output, FIELDS)
    for row in output:
        print(row)


if __name__ == "__main__":
    main()
