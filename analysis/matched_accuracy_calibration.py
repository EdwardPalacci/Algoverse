"""Is the AR-vs-DLM calibration gap a calibration difference, or an accuracy difference?

The aggregate expected calibration error (ECE) gap between the two families is
large (DLM 0.456 vs AR 0.201). Because ECE penalises the distance between
stated confidence and realised accuracy, and every model in this evaluation
states high confidence, a family that is simply less accurate will post a worse
ECE without differing in any architecture-linked way. This script separates
those two explanations.

Design.
  Difficulty. For model m and question q, difficulty is the share of correct
  answers produced by the other six models on q. The two families contribute
  equally to that share rather than in proportion to their model counts: with
  three AR and four DLM models, an unweighted pool would score AR targets
  against 2 AR + 4 DLM and DLM targets against 3 AR + 3 DLM, so the stratifier
  would mean something different for each family.

  Matching. Generations are grouped into difficulty strata; each (family,
  stratum) cell contributes an (accuracy, ECE) point. Cell accuracy varies for
  reasons independent of the scored model, so the two families can be compared
  where their accuracy ranges overlap.

  Inference. Three checks decide whether any residual difference is real:
  a cluster bootstrap over questions, a permutation test that reassigns the
  family label across models, and a leave-one-model-out sweep. A residual that
  does not survive all three is reported as absent, not as small.

Outputs land in paper_assets/tables/stratified/.
"""

from __future__ import annotations

import random
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_paper_assets as G

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper_assets" / "tables" / "stratified"

FINE_WIDTH = 0.1
MIN_CELL = 150
MATCH_TOLERANCE = 0.05
BOOTSTRAP_ITERATIONS = 400
PERMUTATIONS = 200
SEED = 20260628


def balanced_difficulty(rows: list[dict]) -> dict[tuple[str, str], float]:
    """(model_id, question_id) -> difficulty, with families weighted equally.

    Each family's reference models are averaged separately and the two family
    means are then averaged, so the stratifier does not inherit the 3-vs-4
    imbalance in the model roster.
    """
    hits: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    family_of: dict[str, str] = {}
    for row in rows:
        if row.get("correct_auto") is None:
            continue
        family_of[row["model_id"]] = row["model_family"]
        hits[(row["question_id"], row["model_family"], row["model_id"])].append(
            1 if row["correct_auto"] else 0
        )

    by_question: dict[str, dict[str, dict[str, list[int]]]] = defaultdict(lambda: defaultdict(dict))
    for (question_id, family, model_id), values in hits.items():
        by_question[question_id][family][model_id] = values

    difficulty: dict[tuple[str, str], float] = {}
    for question_id, families in by_question.items():
        for model_id, family in family_of.items():
            family_means = []
            for other_family, models in families.items():
                values = [
                    sum(v) / len(v) for other_model, v in models.items() if other_model != model_id
                ]
                if values:
                    family_means.append(sum(values) / len(values))
            if family_means:
                difficulty[(model_id, question_id)] = sum(family_means) / len(family_means)
    return difficulty


def annotate(rows: list[dict]) -> list[dict]:
    difficulty = balanced_difficulty(rows)
    output = []
    for row in rows:
        key = (row["model_id"], row["question_id"])
        if key not in difficulty:
            continue
        row = dict(row)
        row["difficulty"] = difficulty[key]
        output.append(row)
    return output


def fine_bin(value: float) -> str:
    index = min(int(value / FINE_WIDTH), int(1 / FINE_WIDTH) - 1)
    return f"{index * FINE_WIDTH:.1f}-{(index + 1) * FINE_WIDTH:.1f}"


def cells_for(rows: list[dict], label_key: str = "model_family") -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row[label_key], fine_bin(row["difficulty"]))].append(row)

    cells: dict[str, list[dict]] = defaultdict(list)
    for (label, bin_label), group in sorted(grouped.items()):
        if len(group) < MIN_CELL:
            continue
        point = G.metric_row(group)
        if point["accuracy"] is None or point["expected_calibration_error"] is None:
            continue
        cells[label].append(
            {
                "label": label,
                "difficulty_bin": bin_label,
                "N": point["N"],
                "questions": len({row["question_id"] for row in group}),
                "accuracy": point["accuracy"],
                "mean_confidence": point["mean_confidence"],
                "expected_calibration_error": point["expected_calibration_error"],
                "area_under_roc": point["area_under_roc"],
                "high_confidence_wrong_rate": point["high_confidence_wrong_rate"],
            }
        )
    return cells


def supported_pairs(cells: dict[str, list[dict]], a: str, b: str) -> list[dict]:
    """Real cell pairs whose accuracies match within tolerance. No interpolation."""
    output = []
    for cell_b in cells.get(b, []):
        best = None
        for cell_a in cells.get(a, []):
            delta = abs(cell_a["accuracy"] - cell_b["accuracy"])
            if best is None or delta < best[0]:
                best = (delta, cell_a)
        if best is None or best[0] > MATCH_TOLERANCE:
            continue
        _delta, cell_a = best
        output.append({"a": cell_a, "b": cell_b, "ece_gap": cell_b["expected_calibration_error"] - cell_a["expected_calibration_error"]})
    return output


def mean_gap(rows: list[dict], a: str = "AR", b: str = "DLM", label_key: str = "model_family") -> float | None:
    pairs = supported_pairs(cells_for(rows, label_key), a, b)
    if not pairs:
        return None
    return sum(p["ece_gap"] for p in pairs) / len(pairs)


def bootstrap_gap(rows: list[dict]) -> dict:
    by_question = defaultdict(list)
    for row in rows:
        by_question[row["question_id"]].append(row)
    question_ids = sorted(by_question)
    rng = random.Random(SEED)

    samples = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        drawn = []
        for question_id in (rng.choice(question_ids) for _ in question_ids):
            drawn.extend(by_question[question_id])
        value = mean_gap(drawn)
        if value is not None:
            samples.append(value)

    samples.sort()
    if not samples:
        return {}
    return {
        "point_estimate": G.fmt(mean_gap(rows)),
        "bootstrap_median": G.fmt(samples[len(samples) // 2]),
        "ci_low": G.fmt(samples[int(0.025 * len(samples))]),
        "ci_high": G.fmt(samples[min(int(0.975 * len(samples)), len(samples) - 1)]),
        "share_positive": G.fmt(sum(1 for s in samples if s > 0) / len(samples)),
        "iterations": len(samples),
    }


def permutation_test(rows: list[dict]) -> dict:
    """Reassign the family label across models; how special is the real split?"""
    models = sorted({row["model_id"] for row in rows})
    real_ar = {row["model_id"] for row in rows if row["model_family"] == "AR"}
    observed = mean_gap(rows)
    if observed is None:
        return {}

    values = []
    for subset in combinations(models, len(real_ar)):
        if set(subset) == real_ar:
            continue
        relabelled = []
        for row in rows:
            row = dict(row)
            row["pseudo_family"] = "AR" if row["model_id"] in subset else "DLM"
            relabelled.append(row)
        value = mean_gap(relabelled, label_key="pseudo_family")
        if value is not None:
            values.append({"ar_side": "|".join(m.split("/")[-1][:12] for m in subset), "gap": value})

    extreme = sum(1 for v in values if abs(v["gap"]) >= abs(observed))
    return {
        "observed_gap": G.fmt(observed),
        "n_splits": len(values) + 1,
        "n_at_least_as_extreme": extreme + 1,
        "p_value": G.fmt((extreme + 1) / (len(values) + 1)),
        "splits": [{"ar_side": v["ar_side"], "mean_ece_gap": G.fmt(v["gap"])} for v in values],
    }


def drop_one_model(rows: list[dict]) -> list[dict]:
    output = []
    for dropped in sorted({row["model_id"] for row in rows}):
        kept = [row for row in rows if row["model_id"] != dropped]
        kept = annotate([dict(r) for r in kept])
        value = mean_gap(kept)
        output.append(
            {
                "model_dropped": dropped,
                "family_dropped": next(r["model_family"] for r in rows if r["model_id"] == dropped),
                "mean_ece_gap": G.fmt(value),
                "sign": "positive" if value and value > 0 else "negative" if value else "n/a",
            }
        )
    return output


def within_stratum_discrimination(rows: list[dict]) -> list[dict]:
    """Does confidence rank correct from incorrect once difficulty is held fixed?

    Computed per model, then averaged within family. Pooling a family's models
    inside a stratum instead produces a below-chance figure for both families,
    which is a composition artefact: models with different accuracy and
    different confidence habits are mixed inside one ranking.
    """
    per_model = defaultdict(list)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model_id"], fine_bin(row["difficulty"]))].append(row)
    for (model_id, _bin_label), group in sorted(grouped.items()):
        if len(group) < 100:
            continue
        value = G.metric_row(group)["area_under_roc"]
        if value is not None:
            per_model[model_id].append(value)

    family_of = {row["model_id"]: row["model_family"] for row in rows}
    output = []
    for model_id, values in sorted(per_model.items()):
        output.append(
            {
                "model_id": model_id,
                "model_family": family_of[model_id],
                "cells": len(values),
                "mean_within_stratum_auroc": G.fmt(sum(values) / len(values)),
                "min_within_stratum_auroc": G.fmt(min(values)),
                "max_within_stratum_auroc": G.fmt(max(values)),
            }
        )
    by_family = defaultdict(list)
    for model_id, values in per_model.items():
        by_family[family_of[model_id]].extend(values)
    for family, values in sorted(by_family.items()):
        output.append(
            {
                "model_id": f"ALL {family}",
                "model_family": family,
                "cells": len(values),
                "mean_within_stratum_auroc": G.fmt(sum(values) / len(values)),
                "min_within_stratum_auroc": G.fmt(min(values)),
                "max_within_stratum_auroc": G.fmt(max(values)),
            }
        )
    return output


def write(name: str, rows) -> None:
    if not rows:
        print(f"  skipped {name}")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    G.write_csv(OUT_DIR / name, rows, list(rows[0].keys()))
    print(f"  wrote {name} ({len(rows)} rows)")


def main() -> None:
    raw, _counts = G.load_all_rows()
    rows = annotate(raw)
    print(f"annotated {len(rows)} generations (family-balanced difficulty)")

    cells = cells_for(rows)
    write("fine_cells_by_family.csv", cells.get("AR", []) + cells.get("DLM", []))

    pairs = supported_pairs(cells, "AR", "DLM")
    write(
        "matched_accuracy_pairs.csv",
        [
            {
                "dlm_difficulty_bin": p["b"]["difficulty_bin"],
                "ar_difficulty_bin": p["a"]["difficulty_bin"],
                "dlm_accuracy": G.fmt(p["b"]["accuracy"]),
                "ar_accuracy": G.fmt(p["a"]["accuracy"]),
                "accuracy_difference": G.fmt(p["b"]["accuracy"] - p["a"]["accuracy"]),
                "dlm_ece": G.fmt(p["b"]["expected_calibration_error"]),
                "ar_ece": G.fmt(p["a"]["expected_calibration_error"]),
                "ece_gap": G.fmt(p["ece_gap"]),
                "dlm_auroc": G.fmt(p["b"]["area_under_roc"]),
                "ar_auroc": G.fmt(p["a"]["area_under_roc"]),
            }
            for p in pairs
        ],
    )

    print("bootstrapping residual gap ...")
    boot = bootstrap_gap(rows)
    write("residual_gap_bootstrap.csv", [boot] if boot else [])
    print(f"  {boot}")

    print("permutation test over model labels ...")
    perm = permutation_test(rows)
    if perm:
        write("residual_gap_permutation.csv", perm.pop("splits"))
        write("residual_gap_permutation_summary.csv", [perm])
        print(f"  observed {perm['observed_gap']} p={perm['p_value']} over {perm['n_splits']} splits")

    print("leave-one-model-out sweep ...")
    sweep = drop_one_model(raw)
    write("residual_gap_drop_one_model.csv", sweep)
    for record in sweep:
        print(f"  drop {record['model_dropped']:42s} -> {record['mean_ece_gap']} ({record['sign']})")

    write("within_stratum_discrimination.csv", within_stratum_discrimination(rows))


if __name__ == "__main__":
    main()
