"""Human--judge agreement measured against the judge labels the analysis uses.

calculate_kappa.py reads the judge column of perfect_audit_sheet.csv. In two of
the 200 rows that column does not match the judge label stored in the frozen
sample (raw_200_sample.jsonl) or in the judged generations every paper metric
is computed from; in both rows the sheet's judge label equals the human label,
while the sheet's own judge_reason still supports the stored label. This script
pairs each human label with the stored judge label instead, so the reported
agreement describes the labels the analysis actually used.
"""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHEET = ROOT / "analysis/human_llm_check/perfect_audit_sheet.csv"
SAMPLE = ROOT / "analysis/human_llm_check/raw_200_sample.jsonl"
JUDGED = ROOT / "analysis/llm_as_judge/results"
OUT = ROOT / "analysis/human_llm_check/audit_results_saved_labels.txt"


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def label(value: object) -> int:
    text = str(value).strip().lower()
    if text in {"correct", "1", "true"}:
        return 1
    if text in {"incorrect", "0", "false"}:
        return 0
    raise ValueError(f"unexpected label {value!r}")


def main() -> None:
    lines = SHEET.read_text(encoding="utf-8").splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("question_id"))
    sheet = list(csv.DictReader(io.StringIO("\n".join(lines[header:]))))
    sample = [json.loads(line) for line in SAMPLE.read_text().splitlines() if line.strip()]
    if len(sheet) != len(sample):
        raise SystemExit("sheet and frozen sample differ in length")
    for s, m in zip(sheet, sample):
        if (s["question_id"], s["condition"], s["model_name"]) != (m["question_id"], m["condition"], m["model_name"]):
            raise SystemExit("sheet and frozen sample are not row-aligned")

    stored = {}
    for path in JUDGED.glob("*/by_model/*/all_datasets.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            stored[(row["question_id"], row["model_name"], row["condition"], row["sample_id"])] = row["CORRECTNESS"]

    human, judge, sheet_judge, notes = [], [], [], []
    for s, m in zip(sheet, sample):
        key = (m["question_id"], m["model_name"], m["condition"], m["sample_id"])
        if stored[key] != m["CORRECTNESS"]:
            raise SystemExit(f"frozen sample label differs from judged generations for {key}")
        human.append(label(s["human_verdict"]))
        judge.append(stored[key])
        sheet_judge.append(label(s["judge_verdict"]))
        if sheet_judge[-1] != stored[key]:
            notes.append(f"- {m['question_id']} | {m['model_name']} | {m['condition']} sample {m['sample_id']} | "
                         f"stored judge={'correct' if stored[key] else 'incorrect'} | sheet judge={s['judge_verdict']} | human={s['human_verdict']}")

    n = len(human)
    agree = sum(h == j for h, j in zip(human, judge))
    po = agree / n
    ph, pj = sum(human) / n, sum(judge) / n
    pe = ph * pj + (1 - ph) * (1 - pj)
    kappa = (po - pe) / (1 - pe)
    low, high = wilson(agree, n)
    disagreements = [
        f"- {m['question_id']} | {m['dataset']} | {m['model_name']} | judge={'correct' if j else 'incorrect'} | human={'correct' if h else 'incorrect'}"
        for m, h, j in zip(sample, human, judge) if h != j
    ]
    report = "\n".join([
        "Human--LLM Judge Agreement Audit (against stored judge labels)",
        "=" * 62,
        f"Rows audited: {n}",
        "Judge labels: CORRECTNESS in the judged generations used by every paper metric",
        f"Observed agreement: {po:.4f} ({agree}/{n})",
        f"Wilson 95% CI for agreement: [{low:.4f}, {high:.4f}]",
        f"Expected agreement by chance: {pe:.4f}",
        f"Cohen's kappa: {kappa:.4f}",
        f"Human label counts: correct={sum(human)}, incorrect={n - sum(human)}",
        f"Stored judge label counts: correct={sum(judge)}, incorrect={n - sum(judge)}",
        f"Disagreements: {n - agree}",
        "",
        "Disagreement rows:",
        *disagreements,
        "",
        "Rows where the audit sheet's judge column differs from the stored judge label:",
        *notes,
        "",
    ])
    OUT.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
