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


def reliability_points(rows: list[dict], family: str) -> list[tuple[float, float]]:
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


def confidence_distribution_data(rows: list[dict]) -> list[dict]:
    """Values plotted in Figures 2 and 2.2."""
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
        coords = [scale_point(x, y, left, right, top, bottom) for x, y in points]
        for (px, py), (qx, qy) in zip(coords, coords[1:]):
            fig.line(px, py, qx, qy, COLORS[label], 2.2)
        for mean_confidence, empirical_accuracy in points:
            x, y = scale_point(mean_confidence, empirical_accuracy, left, right, top, bottom)
            fig.circle(x, y, 4.2, COLORS[label])
            output.append({
                "model_family": family,
                "mean_confidence": mean_confidence,
                "empirical_accuracy": empirical_accuracy,
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
                "model_family": family,
                "prompt_condition": condition,
                "expected_calibration_error": ece(subset) or 0.0,
                "N": len(subset),
            })
    return output


def write_prompt_sensitivity_figure(path: Path, rows: list[dict]) -> list[dict]:
    left, right, top, bottom = 92, 720, 72, 440
    fig = CairoFigure(path)
    fig.text(92, 34, "Prompt sensitivity in expected calibration error", 20, bold=True)
    for index in range(6):
        value = index / 5
        y = bottom - value * (bottom - top)
        fig.line(left, y, right, y, "#dddddd", 0.8)
        fig.text(left - 12, y + 4, f"{value:.1f}", 12, "#444444", align="right")
    fig.line(left, bottom, right, bottom)
    fig.line(left, top, left, bottom)
    fig.text((left + right) / 2, bottom + 58, "Prompt condition", 15, align="center")
    fig.text(28, (top + bottom) / 2, "Expected calibration error", 15, align="center", rotate=-math.pi / 2)
    data = prompt_sensitivity_data(rows)
    prompt_order = ["cautious", "neutral", "overconfident"]
    families = [("AR", "Autoregressive (AR)"), ("DLM", "Diffusion language model (DLM)")]
    slot = (right - left) / len(prompt_order)
    bar_width = slot / 4
    for family_index, (family, label) in enumerate(families):
        for condition_index, condition in enumerate(prompt_order):
            value = next(row["expected_calibration_error"] for row in data if row["model_family"] == family and row["prompt_condition"] == condition)
            x = left + condition_index * slot + (family_index + 1) * bar_width
            y = bottom - value * (bottom - top)
            fig.rect(x, y, bar_width, bottom - y, COLORS[label])
    for condition_index, condition in enumerate(prompt_order):
        x = left + condition_index * slot + slot / 2
        fig.text(x, bottom + 24, condition, 12, "#444444", align="center")
    draw_legend(fig, [label for _, label in families], 742, 86)
    fig.write()
    return data


def produce_figures(rows: list[dict]) -> None:
    reliability_data = write_reliability_figure(FIG_DIR / "figure_3_ar_dlm_reliability_diagram.png", rows)
    write_csv(FIG_DIR / "figure_3_ar_dlm_reliability_diagram_data.csv", reliability_data, ["model_family", "mean_confidence", "empirical_accuracy"])
    write_text(
        FIG_DIR / "figure_3_caption.txt",
        "Figure 3. AR/DLM reliability diagram using 10 equal-width bins of verbalized confidence normalized to [0, 1]. Empty bins are omitted. The dashed diagonal denotes perfect calibration, where empirical accuracy equals mean confidence. Curves compare autoregressive language models (AR) and the diffusion language model (DLM) using the saved LLM-as-judge correctness labels.\n",
    )

    write_distribution_figure(FIG_DIR / "figure_4_confidence_by_correctness.png", rows, "Reported confidence by answer outcome")
    figure_2_data = confidence_distribution_data(rows)
    write_csv(FIG_DIR / "figure_4_confidence_by_correctness_data.csv", figure_2_data, ["group", "model_family", "correct", "bin_low", "bin_high", "count", "group_total", "share"])
    write_text(
        FIG_DIR / "figure_4_caption.txt",
        "Figure 4. Reported confidence distributions for correct and wrong answers. The x-axis bins each model answer by its reported confidence on the normalized [0, 1] scale. The y-axis gives the share of answers from the indicated group that fall in each confidence bin; within each legend group, the bars sum to one across bins. Correct and wrong answers are assigned by the saved LLM-as-judge labels. AR denotes autoregressive language models and DLM denotes the diffusion language model. A concentration of wrong-answer bars near confidence 1.0 indicates high-confidence errors.\n",
    )

    neutral_rows = [row for row in rows if row["prompt_condition"] == "neutral"]
    write_distribution_figure(FIG_DIR / "figure_5_confidence_by_correctness_neutral.png", neutral_rows, "Reported confidence by answer outcome: neutral prompt")
    neutral_data = confidence_distribution_data(neutral_rows)
    write_csv(FIG_DIR / "figure_5_confidence_by_correctness_neutral_data.csv", neutral_data, ["group", "model_family", "correct", "bin_low", "bin_high", "count", "group_total", "share"])
    write_text(
        FIG_DIR / "figure_5_caption.txt",
        "Figure 5. Reported confidence distributions for correct and wrong answers under the neutral prompt only. The x-axis bins each model answer by its reported confidence on the normalized [0, 1] scale. The y-axis gives the share of answers from the indicated group that fall in each confidence bin; within each legend group, the bars sum to one across bins. Correct and wrong answers are assigned by the saved LLM-as-judge labels. This neutral-only comparison controls the prompt condition across autoregressive language models (AR) and the diffusion language model (DLM), so it is the most relevant version of the distributional plot for comparing model families without pooling over cautious and overconfident prompt interventions.\n",
    )

    prompt_data = write_prompt_sensitivity_figure(FIG_DIR / "figure_6_prompt_sensitivity.png", rows)
    write_csv(FIG_DIR / "figure_6_prompt_sensitivity_data.csv", prompt_data, ["model_family", "prompt_condition", "expected_calibration_error", "N"])
    write_text(
        FIG_DIR / "figure_6_caption.txt",
        "Figure 6. Prompt sensitivity measured by expected calibration error. Expected calibration error is computed with 10 equal-width bins of verbalized confidence normalized to [0, 1]. The x-axis gives the prompt condition and the y-axis gives expected calibration error. Bars compare the autoregressive language model (AR) and diffusion language model (DLM) on the shared question-ID analysis set.\n",
    )
