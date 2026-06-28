from __future__ import annotations

import csv
import ctypes
import ctypes.util
import math
import os
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper_assets" / "figures"
FIG_PNG_DIR = FIG_DIR / "pngs"
FIG_CSV_DIR = FIG_DIR / "csvs"
FIG_CAPTION_DIR = FIG_DIR / "captions"
ECE_BINS = 10

COLORS = {
    "Autoregressive (AR)": "#2f65a7",
    "Diffusion language model (DLM)": "#c45a3c",
    "AR: correct answers": "#2f65a7",
    "AR: wrong answers": "#91b8e8",
    "DLM: correct answers": "#c45a3c",
    "DLM: wrong answers": "#e6a18a",
}


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def mean(values) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def ece(rows: list[dict], bins: int = ECE_BINS) -> float | None:
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


def aurc(rows: list[dict]) -> float | None:
    usable = [
        row for row in rows
        if row.get("parsed_confidence") is not None
        and row.get("correct_auto") is not None
    ]
    if not usable:
        return None
    ranked = sorted(usable, key=lambda row: row["parsed_confidence"], reverse=True)
    errors = 0
    risks = []
    for index, row in enumerate(ranked, start=1):
        if row["correct_auto"] is False:
            errors += 1
        risks.append(errors / index)
    return mean(risks)


def reliability_points(rows: list[dict], family: str) -> list[tuple[float, float, int]]:
    points = []
    family_rows = [
        row for row in rows
        if row["model_family"] == family
        and row.get("parsed_confidence") is not None
    ]
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
                len(bucket),
            ))
    return points


def color_rgb(hex_color: str) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    return (
        int(hex_color[0:2], 16) / 255,
        int(hex_color[2:4], 16) / 255,
        int(hex_color[4:6], 16) / 255,
    )


def scale_point(x: float, y: float, left: int, right: int, top: int, bottom: int) -> tuple[float, float]:
    return left + x * (right - left), bottom - y * (bottom - top)


class CairoTextExtents(ctypes.Structure):
    _fields_ = [
        ("x_bearing", ctypes.c_double),
        ("y_bearing", ctypes.c_double),
        ("width", ctypes.c_double),
        ("height", ctypes.c_double),
        ("x_advance", ctypes.c_double),
        ("y_advance", ctypes.c_double),
    ]


class CairoFigure:
    """Tiny PNG drawing helper backed by Pillow."""

    def __init__(self, path: Path, width: int = 900, height: int = 560) -> None:
        self.path = path
        from PIL import Image, ImageDraw, ImageFont

        self.image = Image.new("RGB", (width, height), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.Image = Image
        self.ImageDraw = ImageDraw
        self.ImageFont = ImageFont

    def set_rgb(self, color: str) -> None:
        self.color = self._rgb(color)

    def _rgb(self, color: str) -> tuple[int, int, int]:
        color = color.lstrip("#")
        return (
            int(color[0:2], 16),
            int(color[2:4], 16),
            int(color[4:6], 16),
        )

    def _font(self, size: float, bold: bool = False):
        candidates = [
            "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
        for candidate in candidates:
            try:
                return self.ImageFont.truetype(candidate, int(size))
            except Exception:
                pass
        return self.ImageFont.load_default()

    def line(self, x0: float, y0: float, x1: float, y1: float, color: str = "#333333", width: float = 1.2, dash: tuple[float, ...] = ()) -> None:
        fill = self._rgb(color)
        width_i = max(1, int(round(width)))
        if not dash:
            self.draw.line((x0, y0, x1, y1), fill=fill, width=width_i)
            return
        dx = x1 - x0
        dy = y1 - y0
        length = math.hypot(dx, dy)
        if length == 0:
            return
        ux = dx / length
        uy = dy / length
        position = 0.0
        dash_on = True
        dash_index = 0
        while position < length:
            segment = dash[dash_index % len(dash)]
            end = min(length, position + segment)
            if dash_on:
                self.draw.line(
                    (
                        x0 + ux * position,
                        y0 + uy * position,
                        x0 + ux * end,
                        y0 + uy * end,
                    ),
                    fill=fill,
                    width=width_i,
                )
            position = end
            dash_on = not dash_on
            dash_index += 1

    def rect(self, x: float, y: float, width: float, height: float, color: str) -> None:
        self.draw.rectangle((x, y, x + width, y + height), fill=self._rgb(color))

    def circle(self, x: float, y: float, radius: float, color: str) -> None:
        self.draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=self._rgb(color))

    def text(self, x: float, y: float, value: str, size: float = 12, color: str = "#222222", align: str = "left", bold: bool = False, rotate: float = 0) -> None:
        text = str(value)
        font = self._font(size, bold)
        bbox = self.draw.textbbox((0, 0), text, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        dx = 0.0
        if align == "center":
            dx = -width / 2
        elif align == "right":
            dx = -width
        fill = self._rgb(color)
        if rotate:
            pad = 8
            tile = self.Image.new("RGBA", (width + 2 * pad, height + 2 * pad), (255, 255, 255, 0))
            tile_draw = self.ImageDraw.Draw(tile)
            tile_draw.text((pad, pad), text, font=font, fill=fill + (255,))
            rotated = tile.rotate(-math.degrees(rotate), expand=True, resample=self.Image.Resampling.BICUBIC)
            self.image.paste(rotated.convert("RGB"), (int(x + dx - rotated.width / 2), int(y - rotated.height / 2)), rotated)
        else:
            self.draw.text((x + dx, y - height), text, fill=fill, font=font)

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(self.path)


def draw_common_axes(fig: CairoFigure, left: int, right: int, top: int, bottom: int, x_label: str, y_label: str) -> None:
    for index in range(6):
        value = index / 5
        x, y = scale_point(value, value, left, right, top, bottom)
        fig.line(left, y, right, y, "#dddddd", 0.8)
        fig.line(x, top, x, bottom, "#dddddd", 0.8)
        fig.text(x, bottom + 22, f"{value:.1f}", 12, "#444444", align="center")
        fig.text(left - 12, y + 4, f"{value:.1f}", 12, "#444444", align="right")
    fig.line(left, bottom, right, bottom)
    fig.line(left, top, left, bottom)
    fig.text((left + right) / 2, bottom + 52, x_label, 15, align="center")
    fig.text(28, (top + bottom) / 2, y_label, 15, align="center", rotate=-math.pi / 2)


def draw_legend(fig: CairoFigure, labels: list[str], x: int, y: int) -> None:
    for index, label in enumerate(labels):
        yy = y + index * 24
        fig.rect(x, yy - 11, 15, 15, COLORS[label])
        fig.text(x + 23, yy + 1, label, 13)


def draw_arrow(fig: CairoFigure, x0: float, y0: float, x1: float, y1: float, color: str = "#555555", width: float = 2.0) -> None:
    fig.line(x0, y0, x1, y1, color, width)
    angle = math.atan2(y1 - y0, x1 - x0)
    size = 10
    points = [
        (x1, y1),
        (x1 - size * math.cos(angle - 0.45), y1 - size * math.sin(angle - 0.45)),
        (x1 - size * math.cos(angle + 0.45), y1 - size * math.sin(angle + 0.45)),
    ]
    fig.draw.polygon(points, fill=fig._rgb(color))


def draw_centered_lines(fig: CairoFigure, x: float, y: float, lines: list[str], size: int = 13, color: str = "#222222", bold_first: bool = True) -> None:
    line_height = size + 6
    start_y = y - ((len(lines) - 1) * line_height) / 2
    for index, line in enumerate(lines):
        fig.text(x, start_y + index * line_height, line, size, color, align="center", bold=bold_first and index == 0)


def draw_flow_box(fig: CairoFigure, center_x: float, center_y: float, width: float, height: float, fill: str, lines: list[str]) -> None:
    x0 = center_x - width / 2
    y0 = center_y - height / 2
    x1 = center_x + width / 2
    y1 = center_y + height / 2
    fig.draw.rounded_rectangle((x0, y0, x1, y1), radius=10, fill=fig._rgb(fill), outline=fig._rgb("#333333"), width=1)
    draw_centered_lines(fig, center_x, center_y + 8, lines)


def write_benchmark_flowchart_figure(path: Path) -> list[dict]:
    fig = CairoFigure(path, width=1100, height=720)
    fig.text(70, 36, "Benchmark pipeline", 22, bold=True)
    draw_flow_box(fig, 550, 95, 560, 64, "#e8f0fb", ["Input questions", "GSM8K, MedQA, SimpleQA, TriviaQA, TruthfulQA"])
    draw_flow_box(fig, 550, 185, 560, 64, "#e9f6ec", ["Prompted benchmark items", "neutral, cautious, overconfident"])
    draw_flow_box(fig, 330, 305, 340, 72, "#fff0df", ["Autoregressive models", "Gemini, GPT-4.1 mini, Grok"])
    draw_flow_box(fig, 770, 305, 340, 72, "#fff0df", ["Diffusion models", "Mercury-2, Dream, DiffusionGemma, LLaDA"])
    draw_flow_box(fig, 550, 425, 560, 64, "#f0e9fb", ["Shared generation schema", "answer, confidence, short explanation"])
    draw_flow_box(fig, 330, 545, 340, 72, "#eeeeee", ["Correctness grading", "deterministic checks + LLM judge"])
    draw_flow_box(fig, 770, 545, 340, 72, "#fdeceb", ["Calibration analysis", "ECE, AURC, AUROC, HCWR"])
    draw_flow_box(fig, 550, 655, 560, 62, "#fff8d9", ["Benchmark outputs", "AR-DLM comparison, accuracy, calibration, prompt sensitivity"])
    draw_arrow(fig, 550, 127, 550, 153)
    draw_arrow(fig, 500, 217, 365, 268)
    draw_arrow(fig, 600, 217, 735, 268)
    draw_arrow(fig, 330, 341, 505, 393)
    draw_arrow(fig, 770, 341, 595, 393)
    draw_arrow(fig, 505, 457, 365, 509)
    draw_arrow(fig, 595, 457, 735, 509)
    draw_arrow(fig, 330, 581, 500, 626)
    draw_arrow(fig, 770, 581, 600, 626)
    fig.write()
    return [
        {"stage_order": 1, "stage": "Input questions", "output": "250-question pilot benchmark"},
        {"stage_order": 2, "stage": "Prompted benchmark items", "output": "Neutral, cautious, and overconfident versions"},
        {"stage_order": 3, "stage": "Model generation", "output": "AR and DLM responses"},
        {"stage_order": 4, "stage": "Shared schema", "output": "Answer, confidence, and short explanation"},
        {"stage_order": 5, "stage": "Correctness grading", "output": "LLM-as-judge correctness labels"},
        {"stage_order": 6, "stage": "Calibration analysis", "output": "ECE, AURC, AUROC, and HCWR"},
        {"stage_order": 7, "stage": "Benchmark outputs", "output": "Family, dataset, and prompt-condition comparisons"},
    ]


def confidence_distribution_data(rows: list[dict]) -> list[dict]:
    """Values plotted in the answer-outcome confidence distribution figures."""
    output = []
    groups = [
        ("AR", True, "AR: correct answers"),
        ("AR", False, "AR: wrong answers"),
        ("DLM", True, "DLM: correct answers"),
        ("DLM", False, "DLM: wrong answers"),
    ]
    for family, is_correct, label in groups:
        subset = [
            row for row in rows
            if row["model_family"] == family
            and row["correct_auto"] is is_correct
            and row.get("parsed_confidence") is not None
        ]
        denominator = len(subset) or 1
        for bin_index in range(ECE_BINS):
            low = bin_index / ECE_BINS
            high = (bin_index + 1) / ECE_BINS
            if bin_index == ECE_BINS - 1:
                count = sum(1 for row in subset if low <= row["parsed_confidence"] <= high)
            else:
                count = sum(1 for row in subset if low <= row["parsed_confidence"] < high)
            output.append({
                "group": label,
                "model_family": family,
                "correct": is_correct,
                "bin_low": f"{low:.1f}",
                "bin_high": f"{high:.1f}",
                "count": count,
                "group_total": len(subset),
                "share": count / denominator,
            })
    return output


def write_distribution_figure(path: Path, rows: list[dict], title: str) -> None:
    left, right, top, bottom = 92, 720, 72, 440
    fig = CairoFigure(path)
    fig.text(92, 34, title, 20, bold=True)
    draw_common_axes(fig, left, right, top, bottom, "Reported confidence bin", "Share of answers in group")
    data = confidence_distribution_data(rows)
    labels = ["AR: correct answers", "AR: wrong answers", "DLM: correct answers", "DLM: wrong answers"]
    slot = (right - left) / ECE_BINS
    bar_width = slot / 5
    for label_index, label in enumerate(labels):
        label_rows = [row for row in data if row["group"] == label]
        for bin_index, row in enumerate(label_rows):
            value = row["share"]
            x = left + bin_index * slot + (label_index + 0.45) * bar_width
            y = bottom - value * (bottom - top)
            fig.rect(x, y, bar_width, bottom - y, COLORS[label])
    draw_legend(fig, labels, 742, 86)
    fig.write()


def write_reliability_figure(path: Path, rows: list[dict]) -> list[dict]:
    left, right, top, bottom = 92, 650, 72, 440
    fig = CairoFigure(path)
    fig.text(92, 34, "Reliability diagram", 20, bold=True)
    draw_common_axes(fig, left, right, top, bottom, "Mean verbalized confidence", "Empirical accuracy")
    x0, y0 = scale_point(0, 0, left, right, top, bottom)
    x1, y1 = scale_point(1, 1, left, right, top, bottom)
    fig.line(x0, y0, x1, y1, "#777777", 1.2, dash=(5, 5))
    output = []
    labels = [("AR", "Autoregressive (AR)"), ("DLM", "Diffusion language model (DLM)")]
    for family, label in labels:
        points = sorted(reliability_points(rows, family))
        coords = [scale_point(x, y, left, right, top, bottom) for x, y, _count in points]
        for (px, py), (qx, qy) in zip(coords, coords[1:]):
            fig.line(px, py, qx, qy, COLORS[label], 2.2)
        for mean_confidence, empirical_accuracy, bin_count in points:
            x, y = scale_point(mean_confidence, empirical_accuracy, left, right, top, bottom)
            fig.circle(x, y, 4.2, COLORS[label])
            output.append({
                "model_family": family,
                "mean_confidence": mean_confidence,
                "empirical_accuracy": empirical_accuracy,
                "bin_count": bin_count,
            })
    draw_legend(fig, [label for _, label in labels], 690, 88)
    fig.text(690, 180, "Dashed line: perfect calibration", 13)
    fig.write()
    return output


def prompt_sensitivity_data(rows: list[dict]) -> list[dict]:
    output = []
    for family in ["AR", "DLM"]:
        for condition in ["cautious", "neutral", "overconfident"]:
            subset = [row for row in rows if row["model_family"] == family and row["prompt_condition"] == condition]
            output.append({
                "metric": "ECE",
                "model_family": family,
                "prompt_condition": condition,
                "value": ece(subset) or 0.0,
                "N": len(subset),
            })
            output.append({
                "metric": "AURC",
                "model_family": family,
                "prompt_condition": condition,
                "value": aurc(subset) or 0.0,
                "N": len(subset),
            })
    return output


def write_prompt_sensitivity_figure(path: Path, rows: list[dict]) -> list[dict]:
    fig = CairoFigure(path, width=980, height=520)
    fig.text(72, 34, "Prompt intervention metrics", 20, bold=True)
    data = prompt_sensitivity_data(rows)
    draw_metric_panel(fig, data, "ECE", "Expected calibration error", 88, 420, 82, 405, 0.65, ["cautious", "neutral", "overconfident"])
    draw_metric_panel(fig, data, "AURC", "Area under risk-coverage", 530, 862, 82, 405, 0.75, ["cautious", "neutral", "overconfident"])
    draw_legend(fig, ["Autoregressive (AR)", "Diffusion language model (DLM)"], 330, 468)
    fig.write()
    return data


def risk_coverage_curve_data(rows: list[dict]) -> list[dict]:
    output = []
    for family in ["AR", "DLM"]:
        usable = [
            row for row in rows
            if row["model_family"] == family
            and row.get("parsed_confidence") is not None
            and row.get("correct_auto") is not None
        ]
        ranked = sorted(usable, key=lambda row: row["parsed_confidence"], reverse=True)
        if not ranked:
            continue
        for step in range(5, 101, 5):
            coverage = step / 100
            count = max(1, round(len(ranked) * coverage))
            selected = ranked[:count]
            risk = sum(1 for row in selected if row["correct_auto"] is False) / len(selected)
            output.append({
                "model_family": family,
                "coverage": coverage,
                "risk": risk,
                "N": len(selected),
            })
    return output


def write_risk_coverage_figure(path: Path, rows: list[dict]) -> list[dict]:
    data = risk_coverage_curve_data(rows)
    fig = CairoFigure(path, width=920, height=540)
    fig.text(70, 36, "Risk-coverage curves by model family", 20, bold=True)
    left, right, top, bottom = 92, 680, 78, 420
    y_max = max(0.75, math.ceil(max(row["risk"] for row in data) * 10) / 10) if data else 1.0
    for index in range(6):
        x_value = index / 5
        y_value = y_max * index / 5
        x = left + x_value * (right - left)
        y = bottom - (y_value / y_max) * (bottom - top)
        fig.line(left, y, right, y, "#dddddd", 0.8)
        fig.line(x, top, x, bottom, "#dddddd", 0.8)
        fig.text(x, bottom + 22, f"{x_value:.1f}", 12, "#444444", align="center")
        fig.text(left - 12, y + 4, f"{y_value:.1f}", 12, "#444444", align="right")
    fig.line(left, bottom, right, bottom)
    fig.line(left, top, left, bottom)
    fig.text((left + right) / 2, bottom + 52, "Coverage retained by confidence ranking", 15, align="center")
    fig.text(28, (top + bottom) / 2, "Risk among retained answers", 15, align="center", rotate=-math.pi / 2)
    labels = [("AR", "Autoregressive (AR)"), ("DLM", "Diffusion language model (DLM)")]
    for family, label in labels:
        family_rows = [row for row in data if row["model_family"] == family]
        coords = [
            (
                left + row["coverage"] * (right - left),
                bottom - (row["risk"] / y_max) * (bottom - top),
            )
            for row in family_rows
        ]
        for (px, py), (qx, qy) in zip(coords, coords[1:]):
            fig.line(px, py, qx, qy, COLORS[label], 2.2)
        for x, y in coords:
            fig.circle(x, y, 3.5, COLORS[label])
    draw_legend(fig, [label for _, label in labels], 710, 98)
    fig.text(710, 180, "Lower curves indicate better", 12, "#555555")
    fig.text(710, 200, "selective prediction.", 12, "#555555")
    fig.write()
    return data


def dataset_metric_data(rows: list[dict]) -> list[dict]:
    output = []
    datasets = sorted({row["dataset"] for row in rows})
    for dataset in datasets:
        for family in ["AR", "DLM"]:
            subset = [row for row in rows if row["dataset"] == dataset and row["model_family"] == family]
            if not subset:
                continue
            correct = [1.0 if row["correct_auto"] else 0.0 for row in subset if row.get("correct_auto") is not None]
            output.append({
                "dataset": dataset,
                "model_family": family,
                "metric": "Accuracy",
                "value": mean(correct) or 0.0,
                "N": len(subset),
            })
            output.append({
                "dataset": dataset,
                "model_family": family,
                "metric": "ECE",
                "value": ece(subset) or 0.0,
                "N": len(subset),
            })
    return output


def heat_color(value: float, metric: str) -> str:
    value = max(0.0, min(value, 1.0))
    if metric == "Accuracy":
        base = (46, 101, 167)
    else:
        base = (196, 90, 60)
    blend = 0.18 + 0.82 * value
    rgb = tuple(round(255 * (1 - blend) + channel * blend) for channel in base)
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def write_dataset_metric_heatmap(path: Path, rows: list[dict]) -> list[dict]:
    data = dataset_metric_data(rows)
    datasets = sorted({row["dataset"] for row in data})
    families = ["AR", "DLM"]
    metrics = ["Accuracy", "ECE"]
    fig = CairoFigure(path, width=960, height=560)
    fig.text(70, 36, "Dataset-level accuracy and calibration", 20, bold=True)
    cell_w, cell_h = 108, 54
    start_x, start_y = 210, 110
    panel_gap = 270
    for metric_index, metric in enumerate(metrics):
        panel_x = start_x + metric_index * panel_gap
        fig.text(panel_x + cell_w, 78, metric, 15, bold=True, align="center")
        for family_index, family in enumerate(families):
            fig.text(panel_x + family_index * cell_w + cell_w / 2, 102, family, 12, "#444444", align="center")
        for dataset_index, dataset in enumerate(datasets):
            y = start_y + dataset_index * cell_h
            if metric_index == 0:
                fig.text(185, y + 33, dataset, 12, "#333333", align="right")
            for family_index, family in enumerate(families):
                value = next(
                    row["value"] for row in data
                    if row["dataset"] == dataset
                    and row["model_family"] == family
                    and row["metric"] == metric
                )
                x = panel_x + family_index * cell_w
                fig.rect(x, y, cell_w - 4, cell_h - 4, heat_color(value, metric))
                fig.text(x + cell_w / 2 - 2, y + 34, f"{value:.2f}", 13, "#ffffff", align="center", bold=True)
    fig.text(70, 500, "Accuracy uses saved correctness labels. ECE uses 10 equal-width confidence bins.", 12, "#555555")
    fig.write()
    return data


def draw_metric_panel(fig: CairoFigure, data: list[dict], metric: str, title: str, left: int, right: int, top: int, bottom: int, y_max: float, categories: list[str]) -> None:
    fig.text((left + right) / 2, top - 24, title, 14, align="center", bold=True)
    for index in range(4):
        value = y_max * index / 3
        y = bottom - (value / y_max) * (bottom - top)
        fig.line(left, y, right, y, "#dddddd", 0.8)
        fig.text(left - 10, y + 4, f"{value:.2f}", 10, "#444444", align="right")
    fig.line(left, bottom, right, bottom)
    fig.line(left, top, left, bottom)
    families = [("AR", "Autoregressive (AR)"), ("DLM", "Diffusion language model (DLM)")]
    slot = (right - left) / len(categories)
    bar_width = slot / 4
    for family_index, (family, label) in enumerate(families):
        for category_index, category in enumerate(categories):
            value = next(
                row["value"] for row in data
                if row["metric"] == metric
                and row["model_family"] == family
                and row["prompt_condition"] == category
            )
            x = left + category_index * slot + (family_index + 1) * bar_width
            y = bottom - min(value / y_max, 1.0) * (bottom - top)
            fig.rect(x, y, bar_width, bottom - y, COLORS[label])
    for category_index, category in enumerate(categories):
        x = left + category_index * slot + slot / 2
        fig.text(x, bottom + 24, category, 11, "#444444", align="center")


def produce_figures(rows: list[dict]) -> None:
    flowchart_data = write_benchmark_flowchart_figure(FIG_PNG_DIR / "figure_1_benchmark_flowchart.png")
    write_csv(FIG_CSV_DIR / "figure_1_benchmark_flowchart_data.csv", flowchart_data, ["stage_order", "stage", "output"])
    write_text(
        FIG_CAPTION_DIR / "figure_1_caption.txt",
        "Figure 1. Benchmark pipeline. Input questions are converted into prompt-conditioned benchmark items, routed through autoregressive language models (AR) and diffusion language models (DLMs), normalized into a shared generation schema, and evaluated through common correctness-grading and calibration analyses.\n",
    )

    reliability_data = write_reliability_figure(FIG_PNG_DIR / "figure_4_ar_dlm_reliability_diagram.png", rows)
    write_csv(FIG_CSV_DIR / "figure_4_ar_dlm_reliability_diagram_data.csv", reliability_data, ["model_family", "mean_confidence", "empirical_accuracy", "bin_count"])
    write_text(
        FIG_CAPTION_DIR / "figure_4_caption.txt",
        "Figure 4. AR/DLM reliability diagram using 10 equal-width bins of verbalized confidence normalized to [0, 1]. Empty bins are omitted. The dashed diagonal denotes perfect calibration, where empirical accuracy equals mean confidence. Curves compare autoregressive language models (AR) and diffusion language models (DLMs) using the saved LLM-as-judge correctness labels. Bin counts are saved in the companion CSV because sparse bins can produce visually extreme points.\n",
    )

    write_distribution_figure(FIG_PNG_DIR / "figure_5_confidence_by_correctness.png", rows, "Reported confidence by answer outcome")
    figure_2_data = confidence_distribution_data(rows)
    write_csv(FIG_CSV_DIR / "figure_5_confidence_by_correctness_data.csv", figure_2_data, ["group", "model_family", "correct", "bin_low", "bin_high", "count", "group_total", "share"])
    write_text(
        FIG_CAPTION_DIR / "figure_5_caption.txt",
        "Figure 5. Reported confidence distributions for correct and wrong answers. The x-axis bins each model answer by its reported confidence on the normalized [0, 1] scale. The y-axis gives the share of answers from the indicated group that fall in each confidence bin; within each legend group, the bars sum to one across bins. Correct and wrong answers are assigned by the saved LLM-as-judge labels. AR denotes autoregressive language models and DLM denotes diffusion language models. A concentration of wrong-answer bars near confidence 1.0 indicates high-confidence errors.\n",
    )

    neutral_rows = [row for row in rows if row["prompt_condition"] == "neutral"]
    write_distribution_figure(FIG_PNG_DIR / "figure_6_confidence_by_correctness_neutral.png", neutral_rows, "Reported confidence by answer outcome: neutral prompt")
    neutral_data = confidence_distribution_data(neutral_rows)
    write_csv(FIG_CSV_DIR / "figure_6_confidence_by_correctness_neutral_data.csv", neutral_data, ["group", "model_family", "correct", "bin_low", "bin_high", "count", "group_total", "share"])
    write_text(
        FIG_CAPTION_DIR / "figure_6_caption.txt",
        "Figure 6. Reported confidence distributions for correct and wrong answers under the neutral prompt only. The x-axis bins each model answer by its reported confidence on the normalized [0, 1] scale. The y-axis gives the share of answers from the indicated group that fall in each confidence bin; within each legend group, the bars sum to one across bins. Correct and wrong answers are assigned by the saved LLM-as-judge labels. This neutral-only comparison controls the prompt condition across autoregressive language models (AR) and diffusion language models (DLMs), so it is the most relevant version of the distributional plot for comparing model families without pooling over cautious and overconfident prompt interventions.\n",
    )

    prompt_data = write_prompt_sensitivity_figure(FIG_PNG_DIR / "figure_7_prompt_sensitivity.png", rows)
    write_csv(FIG_CSV_DIR / "figure_7_prompt_sensitivity_data.csv", prompt_data, ["metric", "model_family", "prompt_condition", "value", "N"])
    write_text(
        FIG_CAPTION_DIR / "figure_7_caption.txt",
        "Figure 7. Prompt intervention metrics comparing autoregressive language models (AR) and diffusion language models (DLMs). ECE is expected calibration error using 10 equal-width confidence bins. AURC is area under the risk-coverage curve, where lower values indicate better confidence-based selective prediction.\n",
    )

    risk_data = write_risk_coverage_figure(FIG_PNG_DIR / "figure_8_risk_coverage_curve.png", rows)
    write_csv(FIG_CSV_DIR / "figure_8_risk_coverage_curve_data.csv", risk_data, ["model_family", "coverage", "risk", "N"])
    write_text(
        FIG_CAPTION_DIR / "figure_8_caption.txt",
        "Figure 8. Risk-coverage curves by model family. Generations are sorted from highest to lowest reported confidence, and each point reports the empirical error rate among retained generations at a given coverage level. Lower curves indicate better confidence-based selective prediction and support the AURC values reported in the metric tables.\n",
    )

    heatmap_data = write_dataset_metric_heatmap(FIG_PNG_DIR / "figure_9_dataset_metric_heatmap.png", rows)
    write_csv(FIG_CSV_DIR / "figure_9_dataset_metric_heatmap_data.csv", heatmap_data, ["dataset", "model_family", "metric", "value", "N"])
    write_text(
        FIG_CAPTION_DIR / "figure_9_caption.txt",
        "Figure 9. Dataset-level accuracy and expected calibration error by model family. Values are computed from the aligned judged generations. The heatmap highlights that aggregate metrics hide substantial dataset-level variation, especially on SimpleQA.\n",
    )
