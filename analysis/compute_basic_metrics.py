from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from render_figures import CairoFigure, write_csv, write_text


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper_assets" / "figures"
ECE_BINS = 10
HIGH_CONFIDENCE_THRESHOLD = 0.90
CONDITION_ORDER = ["cautious", "neutral", "overconfident"]


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


@dataclass
class Generation:
    question_id: str
    dataset: str
    condition: str
    sample_idx: int
    model: str
    prompt: str
    answer: str | None
    confidence: float | None
    ground_truth: str | list[str]
    correct: bool | None
    raw_output: str | None
    parse_success: bool
    answer_type: str | None = None
    model_architecture: str | None = None
    short_explanation: str | None = None

    @property
    def parsed_ok(self) -> bool:
        return self.parse_success and self.answer is not None and self.confidence is not None

    @property
    def confidence_norm(self) -> float | None:
        return self.confidence


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_generations(path: str | Path) -> list[Generation]:
    """Load a JSONL file into the notebook-era Generation schema."""
    rows = read_jsonl(Path(path))
    return [generation_from_row(row) for row in rows]


def load_ar_judged_generations() -> list[Generation]:
    """Load current AR LLM-judge outputs from the split results directory."""
    paths = sorted(
        (ROOT / "analysis" / "llm_as_judge" / "results" / "ar" / "by_model").glob("*/all_datasets.jsonl")
    )
    generations = []
    for path in paths:
        generations.extend(generation_from_row(row) for row in read_jsonl(path))
    return generations


def generation_from_row(row: dict) -> Generation:
    confidence = row.get("confidence")
    return Generation(
        question_id=row["question_id"],
        dataset=row["dataset"],
        condition=row["condition"],
        sample_idx=int(row.get("sample_id", row.get("sample_idx", 0))),
        model=row.get("model_name", row.get("model", "")),
        prompt=row.get("prompt", ""),
        answer=row.get("answer") if row.get("answer") not in ("", None) else row.get("model_answer"),
        confidence=float(confidence) if confidence is not None else None,
        ground_truth=row.get("ground_truth", ""),
        correct=(
            True if row.get("CORRECTNESS") == 1
            else False if row.get("CORRECTNESS") == 0
            else row.get("correct")
        ),
        raw_output=row.get("raw_response", row.get("raw_output")),
        parse_success=row.get("source_parse_success", row.get("parse_success", True)) is not False,
        answer_type=row.get("answer_type"),
        model_architecture=row.get("model_architecture"),
        short_explanation=row.get("short_explanation"),
    )


def parse_success_rate(gens: Iterable[Generation]) -> float:
    gens = list(gens)
    return mean(1.0 if generation.parsed_ok else 0.0 for generation in gens) or float("nan")


def parse_failures_by_condition(gens: Iterable[Generation]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(lambda: {"parsed": 0, "failed": 0, "total": 0})
    for generation in gens:
        out[generation.condition]["total"] += 1
        if generation.parsed_ok:
            out[generation.condition]["parsed"] += 1
        else:
            out[generation.condition]["failed"] += 1
    return dict(out)


def mean_confidence_by_condition(gens: Iterable[Generation]) -> dict[str, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for generation in gens:
        if generation.confidence_norm is not None:
            groups[generation.condition].append(generation.confidence_norm)
    return {condition: mean(values) or float("nan") for condition, values in groups.items()}


def confidence_histogram(gens: Iterable[Generation], n_bins: int = ECE_BINS) -> list[dict]:
    groups: dict[str, list[float]] = defaultdict(list)
    for generation in gens:
        if generation.confidence_norm is not None:
            groups[generation.condition].append(generation.confidence_norm)

    rows = []
    for condition in condition_order(groups):
        values = groups[condition]
        counts = [0] * n_bins
        for value in values:
            index = min(n_bins - 1, max(0, int(value * n_bins)))
            counts[index] += 1
        total = len(values)
        for index, count in enumerate(counts):
            low = index / n_bins
            high = (index + 1) / n_bins
            rows.append({
                "prompt_condition": condition,
                "bin_low": f"{low:.1f}",
                "bin_high": f"{high:.1f}",
                "bin_center": f"{(low + high) / 2:.2f}",
                "count": count,
                "total": total,
                "share": count / total if total else 0.0,
            })
    return rows


def accuracy(gens: Iterable[Generation]) -> float | None:
    graded = [generation for generation in gens if generation.correct is not None]
    return mean(1.0 if generation.correct else 0.0 for generation in graded)


def accuracy_by_condition(gens: Iterable[Generation]) -> dict[str, float | None]:
    groups: dict[str, list[Generation]] = defaultdict(list)
    for generation in gens:
        groups[generation.condition].append(generation)
    return {condition: accuracy(values) for condition, values in groups.items()}


def reliability_bins(gens: Iterable[Generation], n_bins: int = ECE_BINS) -> list[dict]:
    usable = [
        generation for generation in gens
        if generation.confidence_norm is not None
        and generation.correct is not None
    ]
    rows = []
    for index in range(n_bins):
        low = index / n_bins
        high = (index + 1) / n_bins
        if index == n_bins - 1:
            bucket = [generation for generation in usable if low <= generation.confidence_norm <= high]
        else:
            bucket = [generation for generation in usable if low <= generation.confidence_norm < high]
        rows.append({
            "bin_low": low,
            "bin_high": high,
            "bin_center": (low + high) / 2,
            "bin_count": len(bucket),
            "bin_acc": mean(1.0 if generation.correct else 0.0 for generation in bucket),
            "bin_conf": mean(generation.confidence_norm for generation in bucket),
        })
    return rows


def reliability_bins_by_condition(gens: Iterable[Generation], n_bins: int = ECE_BINS) -> list[dict]:
    grouped: dict[str, list[Generation]] = defaultdict(list)
    for generation in gens:
        grouped[generation.condition].append(generation)

    rows = []
    for condition in condition_order(grouped):
        for row in reliability_bins(grouped[condition], n_bins=n_bins):
            rows.append({
                "prompt_condition": condition,
                "bin_low": f"{row['bin_low']:.1f}",
                "bin_high": f"{row['bin_high']:.1f}",
                "bin_center": f"{row['bin_center']:.2f}",
                "bin_count": row["bin_count"],
                "bin_accuracy": "" if row["bin_acc"] is None else row["bin_acc"],
                "bin_confidence": "" if row["bin_conf"] is None else row["bin_conf"],
            })
    return rows


def expected_calibration_error(gens: Iterable[Generation], n_bins: int = ECE_BINS) -> float | None:
    usable_bins = [row for row in reliability_bins(gens, n_bins=n_bins) if row["bin_count"]]
    total = sum(row["bin_count"] for row in usable_bins)
    if not total:
        return None
    return sum(
        (row["bin_count"] / total) * abs(row["bin_acc"] - row["bin_conf"])
        for row in usable_bins
    )


def auroc(gens: Iterable[Generation]) -> float | None:
    positive = [
        generation.confidence_norm for generation in gens
        if generation.confidence_norm is not None and generation.correct is True
    ]
    negative = [
        generation.confidence_norm for generation in gens
        if generation.confidence_norm is not None and generation.correct is False
    ]
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


def disagreement_rate(gens: Iterable[Generation]) -> float | None:
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for generation in gens:
        if generation.answer is not None:
            groups[(generation.question_id, generation.condition)].append(normalize_answer(generation.answer))
    rates = []
    for answers in groups.values():
        if len(answers) >= 2:
            modal_count = Counter(answers).most_common(1)[0][1]
            rates.append((len(answers) - modal_count) / len(answers))
    return mean(rates)


def unique_answers_per_question(gens: Iterable[Generation]) -> list[dict]:
    groups: dict[tuple[str, str], list[Generation]] = defaultdict(list)
    for generation in gens:
        groups[(generation.question_id, generation.condition)].append(generation)

    rows = []
    for (question_id, condition), generations in sorted(groups.items()):
        answers = [
            normalize_answer(generation.answer)
            for generation in generations
            if generation.answer is not None
        ]
        counter = Counter(answers)
        modal_answer, modal_count = counter.most_common(1)[0] if counter else ("", 0)
        rows.append({
            "question_id": question_id,
            "condition": condition,
            "dataset": generations[0].dataset,
            "n_samples": len(answers),
            "n_unique": len(counter),
            "modal_answer": modal_answer,
            "modal_count": modal_count,
        })
    return rows


def summarize(gens: Iterable[Generation], group_by: list[str] | None = None, n_bins: int = ECE_BINS) -> list[dict]:
    generations = list(gens)
    if not generations:
        return []
    if not group_by:
        groups: dict[tuple, list[Generation]] = {(): generations}
    else:
        groups = defaultdict(list)
        for generation in generations:
            groups[tuple(getattr(generation, key) for key in group_by)].append(generation)

    rows = []
    for key, group_generations in sorted(groups.items()):
        row = {field: value for field, value in zip(group_by or [], key)}
        confidences = [
            generation.confidence_norm for generation in group_generations
            if generation.confidence_norm is not None
        ]
        row.update({
            "n_rows": len(group_generations),
            "n_questions": len({generation.question_id for generation in group_generations}),
            "parse_success": parse_success_rate(group_generations),
            "accuracy": accuracy(group_generations),
            "mean_confidence": mean(confidences),
            "ece": expected_calibration_error(group_generations, n_bins=n_bins),
            "auroc": auroc(group_generations),
            "disagreement_rate": disagreement_rate(group_generations),
        })
        rows.append(row)
    return rows


def format_summary(rows: list[dict]) -> str:
    if not rows:
        return "(no rows)"
    columns = list(rows[0])
    widths = {
        column: max(len(column), *(len(format_value(row[column])) for row in rows))
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    separator = "  ".join("-" * widths[column] for column in columns)
    body = "\n".join(
        "  ".join(format_value(row[column]).ljust(widths[column]) for column in columns)
        for row in rows
    )
    return f"{header}\n{separator}\n{body}"


def row_ece(rows: list[dict], bins: int = ECE_BINS) -> float | None:
    """Expected calibration error for normalized analysis rows."""
    usable = [
        row for row in rows
        if row.get("parsed_confidence") is not None
        and row.get("correct_auto") is not None
    ]
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


def row_brier(rows: list[dict]) -> float | None:
    """Brier score for normalized analysis rows."""
    values = []
    for row in rows:
        confidence = row.get("parsed_confidence")
        if confidence is not None and row.get("correct_auto") is not None:
            target = 1.0 if row["correct_auto"] else 0.0
            values.append((confidence - target) ** 2)
    return mean(values)


def row_auroc(rows: list[dict]) -> float | None:
    """Area under ROC for normalized analysis rows."""
    positive = [
        row["parsed_confidence"] for row in rows
        if row.get("parsed_confidence") is not None
        and row.get("correct_auto") is True
    ]
    negative = [
        row["parsed_confidence"] for row in rows
        if row.get("parsed_confidence") is not None
        and row.get("correct_auto") is False
    ]
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


def row_metric_row(rows: list[dict], raw_count: int | None = None) -> dict:
    """Aggregate metric row for normalized analysis rows."""
    n = len(rows)
    correct = sum(1 for row in rows if row.get("correct_auto") is True)
    confidences = [
        row["parsed_confidence"] for row in rows
        if row.get("parsed_confidence") is not None
    ]
    high_confidence_wrong = sum(
        1 for row in rows
        if row.get("correct_auto") is False
        and row.get("parsed_confidence") is not None
        and row["parsed_confidence"] >= HIGH_CONFIDENCE_THRESHOLD
    )
    parsed = sum(1 for row in rows if row.get("source_parse_success") is not False)
    return {
        "N": n,
        "accuracy": correct / n if n else None,
        "mean_confidence": mean(confidences),
        "expected_calibration_error": row_ece(rows),
        "brier_score": row_brier(rows),
        "area_under_roc": row_auroc(rows),
        "high_confidence_wrong_rate": high_confidence_wrong / n if n else None,
        "parse_success": parsed / raw_count if raw_count else None,
    }


def row_reliability_points(rows: list[dict], family: str) -> list[tuple[float, float]]:
    """Reliability points for normalized analysis rows."""
    family_rows = [
        row for row in rows
        if row["model_family"] == family
        and row.get("parsed_confidence") is not None
    ]
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


def produce_basic_metric_figures(gens: Iterable[Generation] | None = None) -> None:
    generations = list(gens) if gens is not None else load_ar_judged_generations()
    write_confidence_histogram_figure(
        FIG_DIR / "figure_1_confidence_histogram.png",
        generations,
    )
    histogram_rows = confidence_histogram(generations)
    write_csv(
        FIG_DIR / "figure_1_confidence_histogram_data.csv",
        histogram_rows,
        ["prompt_condition", "bin_low", "bin_high", "bin_center", "count", "total", "share"],
    )
    write_text(
        FIG_DIR / "figure_1_caption.txt",
        "Figure 1. Confidence histogram by prompt condition for the autoregressive models. Counts are pooled across Gemini Flash, GPT-4.1-mini, and Grok 4.3 using the current LLM-as-judge generation data.\n",
    )

    write_reliability_by_condition_figure(
        FIG_DIR / "figure_2_reliability_diagram.png",
        generations,
    )
    reliability_rows = reliability_bins_by_condition(generations)
    write_csv(
        FIG_DIR / "figure_2_reliability_diagram_data.csv",
        reliability_rows,
        ["prompt_condition", "bin_low", "bin_high", "bin_center", "bin_count", "bin_accuracy", "bin_confidence"],
    )
    write_text(
        FIG_DIR / "figure_2_caption.txt",
        "Figure 2. Reliability diagrams by prompt condition for the autoregressive models. Bars show empirical accuracy in each confidence bin; points show mean reported confidence in that bin. Rows are pooled across Gemini Flash, GPT-4.1-mini, and Grok 4.3 and graded with the current LLM-as-judge labels.\n",
    )


def write_confidence_histogram_figure(path: Path, gens: list[Generation]) -> None:
    rows = confidence_histogram(gens)
    max_count = max((row["count"] for row in rows), default=1)
    fig = CairoFigure(path, width=1120, height=430)
    fig.text(58, 36, "AR confidence distribution by prompt condition", 20, bold=True)
    for index, condition in enumerate(CONDITION_ORDER):
        left = 78 + index * 345
        right = left + 270
        top = 80
        bottom = 335
        condition_rows = [row for row in rows if row["prompt_condition"] == condition]
        draw_histogram_panel(fig, left, right, top, bottom, condition, condition_rows, max_count)
    fig.write()


def draw_histogram_panel(fig: CairoFigure, left: int, right: int, top: int, bottom: int, title: str, rows: list[dict], max_count: int) -> None:
    fig.text((left + right) / 2, top - 22, title, 14, align="center", bold=True)
    for tick in range(6):
        value = tick / 5
        y = bottom - value * (bottom - top)
        fig.line(left, y, right, y, "#dddddd", 0.8)
        fig.text(left - 10, y + 4, f"{int(value * max_count)}", 10, "#444444", align="right")
    fig.line(left, bottom, right, bottom)
    fig.line(left, top, left, bottom)
    slot = (right - left) / ECE_BINS
    for row in rows:
        bin_index = int(float(row["bin_low"]) * ECE_BINS)
        height = (row["count"] / max_count) * (bottom - top) if max_count else 0
        x = left + bin_index * slot + 3
        fig.rect(x, bottom - height, slot - 6, height, "#2f65a7")
    for tick in range(6):
        value = tick / 5
        x = left + value * (right - left)
        fig.text(x, bottom + 20, f"{value:.1f}", 10, "#444444", align="center")
    fig.text((left + right) / 2, bottom + 48, "Reported confidence", 12, align="center")
    fig.text(left - 44, (top + bottom) / 2, "Count", 12, align="center", rotate=-1.5708)


def write_reliability_by_condition_figure(path: Path, gens: list[Generation]) -> None:
    rows = reliability_bins_by_condition(gens)
    fig = CairoFigure(path, width=1120, height=500)
    fig.text(58, 36, "AR reliability diagrams by prompt condition", 20, bold=True)
    for index, condition in enumerate(CONDITION_ORDER):
        left = 78 + index * 345
        right = left + 270
        top = 82
        bottom = 345
        condition_rows = [row for row in rows if row["prompt_condition"] == condition]
        draw_reliability_panel(fig, left, right, top, bottom, condition, condition_rows)
    fig.text(390, 444, "Blue bars: empirical accuracy", 12)
    fig.text(590, 444, "Orange dots: mean confidence", 12, color="#c45a3c")
    fig.text(790, 444, "Dashed line: perfect calibration", 12, color="#555555")
    fig.write()


def draw_reliability_panel(fig: CairoFigure, left: int, right: int, top: int, bottom: int, title: str, rows: list[dict]) -> None:
    fig.text((left + right) / 2, top - 22, title, 14, align="center", bold=True)
    for tick in range(6):
        value = tick / 5
        y = bottom - value * (bottom - top)
        x = left + value * (right - left)
        fig.line(left, y, right, y, "#dddddd", 0.8)
        fig.line(x, top, x, bottom, "#dddddd", 0.8)
        fig.text(left - 10, y + 4, f"{value:.1f}", 10, "#444444", align="right")
        fig.text(x, bottom + 20, f"{value:.1f}", 10, "#444444", align="center")
    fig.line(left, bottom, right, bottom)
    fig.line(left, top, left, bottom)
    fig.line(left, bottom, right, top, "#777777", 1.2, dash=(5, 5))
    slot = (right - left) / ECE_BINS
    for row in rows:
        bin_index = int(float(row["bin_low"]) * ECE_BINS)
        accuracy_value = none_if_blank(row["bin_accuracy"])
        confidence_value = none_if_blank(row["bin_confidence"])
        if accuracy_value is not None:
            height = accuracy_value * (bottom - top)
            x = left + bin_index * slot + 4
            fig.rect(x, bottom - height, slot - 8, height, "#2f65a7")
        if confidence_value is not None:
            x = left + (bin_index + 0.5) * slot
            y = bottom - confidence_value * (bottom - top)
            fig.circle(x, y, 4.5, "#c45a3c")
    fig.text((left + right) / 2, bottom + 48, "Confidence bin", 12, align="center")


def condition_order(groups) -> list[str]:
    conditions = list(groups)
    return sorted(
        conditions,
        key=lambda condition: (
            CONDITION_ORDER.index(condition) if condition in CONDITION_ORDER else len(CONDITION_ORDER),
            condition,
        ),
    )


def normalize_answer(answer: object) -> str:
    text = str(answer).casefold()
    return " ".join("".join(char if char.isalnum() else " " for char in text).split())


def none_if_blank(value: object) -> float | None:
    if value == "" or value is None:
        return None
    return float(value)


def format_value(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


if __name__ == "__main__":
    produce_basic_metric_figures()
