from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path

from file_io import write_csv, write_text
from compute_metrics import ece, reliability_points
from project_paths import ECE_BINS, FIG_DIR


os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

COLORS = {
    "Autoregressive (AR)": "#2f65a7",
    "Diffusion language model (DLM)": "#c45a3c",
    "AR: correct answers": "#2f65a7",
    "AR: wrong answers": "#91b8e8",
    "DLM: correct answers": "#c45a3c",
    "DLM: wrong answers": "#e6a18a",
}


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
    """Tiny PNG drawing helper backed by the system Cairo library."""

    FORMAT_ARGB32 = 0
    FONT_SLANT_NORMAL = 0
    FONT_WEIGHT_NORMAL = 0
    FONT_WEIGHT_BOLD = 1

    def __init__(self, path: Path, width: int = 900, height: int = 560) -> None:
        self.path = path
        self.cairo = ctypes.CDLL("libcairo.so.2")
        self._bind_cairo()
        self.surface = self.cairo.cairo_image_surface_create(self.FORMAT_ARGB32, width, height)
        self.cr = self.cairo.cairo_create(self.surface)
        self.set_rgb("#ffffff")
        self.cairo.cairo_paint(self.cr)

    def _bind_cairo(self) -> None:
        c = self.cairo
        c.cairo_image_surface_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        c.cairo_image_surface_create.restype = ctypes.c_void_p
        c.cairo_create.argtypes = [ctypes.c_void_p]
        c.cairo_create.restype = ctypes.c_void_p
        c.cairo_destroy.argtypes = [ctypes.c_void_p]
        c.cairo_surface_destroy.argtypes = [ctypes.c_void_p]
        c.cairo_set_source_rgb.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        c.cairo_paint.argtypes = [ctypes.c_void_p]
        c.cairo_rectangle.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        c.cairo_fill.argtypes = [ctypes.c_void_p]
        c.cairo_move_to.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
        c.cairo_line_to.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
        c.cairo_stroke.argtypes = [ctypes.c_void_p]
        c.cairo_set_line_width.argtypes = [ctypes.c_void_p, ctypes.c_double]
        c.cairo_set_dash.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_double]
        c.cairo_arc.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        c.cairo_select_font_face.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        c.cairo_set_font_size.argtypes = [ctypes.c_void_p, ctypes.c_double]
        c.cairo_show_text.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        c.cairo_text_extents.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(CairoTextExtents)]
        c.cairo_save.argtypes = [ctypes.c_void_p]
        c.cairo_restore.argtypes = [ctypes.c_void_p]
        c.cairo_translate.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
        c.cairo_rotate.argtypes = [ctypes.c_void_p, ctypes.c_double]
        c.cairo_surface_write_to_png.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        c.cairo_surface_write_to_png.restype = ctypes.c_int

    def set_rgb(self, color: str) -> None:
        self.cairo.cairo_set_source_rgb(self.cr, *color_rgb(color))

    def line(self, x0: float, y0: float, x1: float, y1: float, color: str = "#333333", width: float = 1.2, dash: tuple[float, ...] = ()) -> None:
        self.set_rgb(color)
        self.cairo.cairo_set_line_width(self.cr, width)
        if dash:
            arr = (ctypes.c_double * len(dash))(*dash)
            self.cairo.cairo_set_dash(self.cr, arr, len(dash), 0)
        else:
            self.cairo.cairo_set_dash(self.cr, None, 0, 0)
        self.cairo.cairo_move_to(self.cr, x0, y0)
        self.cairo.cairo_line_to(self.cr, x1, y1)
        self.cairo.cairo_stroke(self.cr)
        self.cairo.cairo_set_dash(self.cr, None, 0, 0)

    def rect(self, x: float, y: float, width: float, height: float, color: str) -> None:
        self.set_rgb(color)
        self.cairo.cairo_rectangle(self.cr, x, y, width, height)
        self.cairo.cairo_fill(self.cr)

    def circle(self, x: float, y: float, radius: float, color: str) -> None:
        self.set_rgb(color)
        self.cairo.cairo_arc(self.cr, x, y, radius, 0, 2 * math.pi)
        self.cairo.cairo_fill(self.cr)

    def text(self, x: float, y: float, value: str, size: float = 12, color: str = "#222222", align: str = "left", bold: bool = False, rotate: float = 0) -> None:
        encoded = str(value).encode("utf-8")
        self.cairo.cairo_save(self.cr)
        self.set_rgb(color)
        self.cairo.cairo_select_font_face(
            self.cr,
            b"DejaVu Sans",
            self.FONT_SLANT_NORMAL,
            self.FONT_WEIGHT_BOLD if bold else self.FONT_WEIGHT_NORMAL,
        )
        self.cairo.cairo_set_font_size(self.cr, size)
        ext = CairoTextExtents()
        self.cairo.cairo_text_extents(self.cr, encoded, ctypes.byref(ext))
        dx = 0
        if align == "center":
            dx = -ext.width / 2 - ext.x_bearing
        elif align == "right":
            dx = -ext.width - ext.x_bearing
        if rotate:
            self.cairo.cairo_translate(self.cr, x, y)
            self.cairo.cairo_rotate(self.cr, rotate)
            self.cairo.cairo_move_to(self.cr, dx, 0)
        else:
            self.cairo.cairo_move_to(self.cr, x + dx, y)
        self.cairo.cairo_show_text(self.cr, encoded)
        self.cairo.cairo_restore(self.cr)

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        status = self.cairo.cairo_surface_write_to_png(self.surface, str(self.path).encode("utf-8"))
        self.cairo.cairo_destroy(self.cr)
        self.cairo.cairo_surface_destroy(self.surface)
        if status != 0:
            raise RuntimeError(f"failed to write PNG {self.path}")


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
        subset = [row for row in rows if row["model_family"] == family and row["correct_auto"] is is_correct]
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
    reliability_data = write_reliability_figure(FIG_DIR / "figure_1_reliability_diagram.png", rows)
    write_csv(FIG_DIR / "figure_1_reliability_diagram_data.csv", reliability_data, ["model_family", "mean_confidence", "empirical_accuracy"])
    write_text(
        FIG_DIR / "figure_1_caption.txt",
        "Figure 1. Reliability diagram using 10 equal-width bins of verbalized confidence normalized to [0, 1]. Empty bins are omitted. The dashed diagonal denotes perfect calibration, where empirical accuracy equals mean confidence. Curves compare qwen/qwen-2.5-7b-instruct, an autoregressive language model (AR), and mercury-2, a diffusion language model (DLM), using deterministic rule-based correctness labels.\n",
    )

    write_distribution_figure(FIG_DIR / "figure_2_confidence_by_correctness.png", rows, "Reported confidence by answer outcome")
    figure_2_data = confidence_distribution_data(rows)
    write_csv(FIG_DIR / "figure_2_confidence_by_correctness_data.csv", figure_2_data, ["group", "model_family", "correct", "bin_low", "bin_high", "count", "group_total", "share"])
    write_text(
        FIG_DIR / "figure_2_caption.txt",
        "Figure 2. Reported confidence distributions for correct and wrong answers. The x-axis bins each model answer by its reported confidence on the normalized [0, 1] scale. The y-axis gives the share of answers from the indicated group that fall in each confidence bin; within each legend group, the bars sum to one across bins. Correct and wrong answers are assigned by the deterministic rule-based grader described in the benchmark specification. AR denotes the autoregressive language model and DLM denotes the diffusion language model. A concentration of wrong-answer bars near confidence 1.0 indicates high-confidence errors.\n",
    )

    neutral_rows = [row for row in rows if row["prompt_condition"] == "neutral"]
    write_distribution_figure(FIG_DIR / "figure_2_2_confidence_by_correctness_neutral.png", neutral_rows, "Reported confidence by answer outcome: neutral prompt")
    figure_2_2_data = confidence_distribution_data(neutral_rows)
    write_csv(FIG_DIR / "figure_2_2_confidence_by_correctness_neutral_data.csv", figure_2_2_data, ["group", "model_family", "correct", "bin_low", "bin_high", "count", "group_total", "share"])
    write_text(
        FIG_DIR / "figure_2_2_caption.txt",
        "Figure 2.2. Reported confidence distributions for correct and wrong answers under the neutral prompt only. The x-axis bins each model answer by its reported confidence on the normalized [0, 1] scale. The y-axis gives the share of answers from the indicated group that fall in each confidence bin; within each legend group, the bars sum to one across bins. Correct and wrong answers are assigned by the deterministic rule-based grader described in the benchmark specification. This neutral-only comparison controls the prompt condition across the autoregressive language model (AR) and diffusion language model (DLM), so it is the most relevant version of the distributional plot for comparing model families without pooling over cautious and overconfident prompt interventions.\n",
    )

    prompt_data = write_prompt_sensitivity_figure(FIG_DIR / "figure_3_prompt_sensitivity.png", rows)
    write_csv(FIG_DIR / "figure_3_prompt_sensitivity_data.csv", prompt_data, ["model_family", "prompt_condition", "expected_calibration_error", "N"])
    write_text(
        FIG_DIR / "figure_3_caption.txt",
        "Figure 3. Prompt sensitivity measured by expected calibration error. Expected calibration error is computed with 10 equal-width bins of verbalized confidence normalized to [0, 1]. The x-axis gives the prompt condition and the y-axis gives expected calibration error. Bars compare the autoregressive language model (AR) and diffusion language model (DLM) on the shared question-ID analysis set.\n",
    )
