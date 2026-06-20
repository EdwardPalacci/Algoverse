from __future__ import annotations

from typing import Iterable

from project_paths import ECE_BINS, HIGH_CONFIDENCE_THRESHOLD


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def ece(rows: list[dict], bins: int = ECE_BINS) -> float | None:
    """Expected calibration error with equal-width confidence bins."""
    usable = [row for row in rows if row.get("parsed_confidence") is not None and row.get("correct_auto") is not None]
    if not usable:
        return None
    total = len(usable)
    score = 0.0
    for bin_index in range(bins):
        low = bin_index / bins
        high = (bin_index + 1) / bins
        if bin_index == bins - 1:
            bucket = [row for row in usable if low <= row["parsed_confidence"] <= high]
        else:
            bucket = [row for row in usable if low <= row["parsed_confidence"] < high]
        if not bucket:
            continue
        bucket_accuracy = mean(1.0 if row["correct_auto"] else 0.0 for row in bucket)
        bucket_confidence = mean(row["parsed_confidence"] for row in bucket)
        score += (len(bucket) / total) * abs(bucket_accuracy - bucket_confidence)
    return score


def brier(rows: list[dict]) -> float | None:
    """Brier score: mean squared error between confidence and correctness."""
    values = []
    for row in rows:
        confidence = row.get("parsed_confidence")
        if confidence is not None and row.get("correct_auto") is not None:
            target = 1.0 if row["correct_auto"] else 0.0
            values.append((confidence - target) ** 2)
    return mean(values)


def auroc(rows: list[dict]) -> float | None:
    """Area under ROC, computed by pairwise ranking with tie credit."""
    positive = [row["parsed_confidence"] for row in rows if row.get("parsed_confidence") is not None and row.get("correct_auto") is True]
    negative = [row["parsed_confidence"] for row in rows if row.get("parsed_confidence") is not None and row.get("correct_auto") is False]
    if not positive or not negative:
        return None
    wins = 0.0
    for pos in positive:
        for neg in negative:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / (len(positive) * len(negative))


def metric_row(rows: list[dict], raw_count: int | None = None) -> dict:
    n = len(rows)
    correct = sum(1 for row in rows if row.get("correct_auto") is True)
    confidences = [row["parsed_confidence"] for row in rows if row.get("parsed_confidence") is not None]
    high_confidence_wrong = sum(
        1
        for row in rows
        if row.get("correct_auto") is False
        and row.get("parsed_confidence") is not None
        and row["parsed_confidence"] >= HIGH_CONFIDENCE_THRESHOLD
    )
    return {
        "N": n,
        "accuracy": correct / n if n else None,
        "mean_confidence": mean(confidences),
        "expected_calibration_error": ece(rows),
        "brier_score": brier(rows),
        "area_under_roc": auroc(rows),
        "high_confidence_wrong_rate": high_confidence_wrong / n if n else None,
        "parse_success": n / raw_count if raw_count else None,
    }


def reliability_points(rows: list[dict], family: str) -> list[tuple[float, float]]:
    """Points for a reliability diagram: mean confidence vs accuracy."""
    family_rows = [row for row in rows if row["model_family"] == family]
    points = []
    for bin_index in range(ECE_BINS):
        low = bin_index / ECE_BINS
        high = (bin_index + 1) / ECE_BINS
        if bin_index == ECE_BINS - 1:
            bucket = [row for row in family_rows if low <= row["parsed_confidence"] <= high]
        else:
            bucket = [row for row in family_rows if low <= row["parsed_confidence"] < high]
        if bucket:
            points.append((
                mean(row["parsed_confidence"] for row in bucket),
                mean(1.0 if row["correct_auto"] else 0.0 for row in bucket),
            ))
    return points


def fmt(value: object) -> object:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return value
