"""Parse-failure robustness check.

Reviewer concern: retaining generations that failed to parse may penalise a
model for formatting problems rather than for lack of knowledge. This script
recomputes the headline metrics with those generations excluded and reports
the difference, so the reported conclusions can be stated as robust (or not)
to that choice.

A generation is treated as a parse failure when source_parse_success is False.
Those rows carry no confidence value, so they already contribute nothing to
ECE, AURC or AUROC; they only affect accuracy, high-confidence-wrong rate and
the denominators.

Outputs land in paper_assets/tables/quality_control/.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_paper_assets as G

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper_assets" / "tables" / "quality_control"

COMPARE_METRICS = [
    "accuracy",
    "mean_confidence",
    "expected_calibration_error",
    "area_under_risk_coverage",
    "area_under_roc",
    "high_confidence_wrong_rate",
]


def is_parse_failure(row: dict) -> bool:
    return row.get("source_parse_success") is False


def compare(rows: list[dict], group_keys: tuple[str, ...]) -> list[dict]:
    output = []
    for key, group in sorted(G.grouped(rows, group_keys).items()):
        kept = [row for row in group if not is_parse_failure(row)]
        dropped = len(group) - len(kept)
        retained, excluded = G.metric_row(group), G.metric_row(kept)
        record = {name: value for name, value in zip(group_keys, key)}
        record["N_retained"] = retained["N"]
        record["N_excluded"] = excluded["N"]
        record["parse_failures_dropped"] = dropped
        record["share_dropped"] = G.fmt(dropped / len(group) if group else None)
        for metric in COMPARE_METRICS:
            a, b = retained.get(metric), excluded.get(metric)
            record[f"{metric}_retained"] = G.fmt(a)
            record[f"{metric}_excluded"] = G.fmt(b)
            record[f"{metric}_delta"] = (
                G.fmt(b - a) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
            )
        output.append(record)
    return output


def largest_shift(rows: list[dict]) -> None:
    worst = 0.0
    where = None
    for record in rows:
        for metric in COMPARE_METRICS:
            delta = record.get(f"{metric}_delta")
            if isinstance(delta, (int, float)) and abs(delta) > worst:
                worst = abs(delta)
                where = (record, metric)
    if where:
        record, metric = where
        label = " ".join(str(value) for key, value in record.items() if key in ("model_family", "model_id", "prompt_condition"))
        print(f"largest absolute shift: {worst:.4f} on {metric} ({label.strip()})")


def write(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    G.write_csv(OUT_DIR / name, rows, list(rows[0].keys()))
    print(f"wrote {name} ({len(rows)} rows)")


def main() -> None:
    rows, _raw_counts = G.load_all_rows()
    total_failures = sum(1 for row in rows if is_parse_failure(row))
    print(f"{total_failures} parse failures out of {len(rows)} generations "
          f"({total_failures / len(rows):.4%})")

    by_model = compare(rows, ("model_family", "model_id"))
    by_family = compare(rows, ("model_family",))
    by_condition = compare(rows, ("model_family", "prompt_condition"))

    write("parse_failure_robustness_by_model.csv", by_model)
    write("parse_failure_robustness_by_family.csv", by_family)
    write("parse_failure_robustness_by_condition.csv", by_condition)

    for group in (by_model, by_family, by_condition):
        largest_shift(group)


if __name__ == "__main__":
    main()
