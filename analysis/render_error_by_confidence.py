"""Per-model error rate by stated confidence, neutral prompt.

Does a model's error rate fall as its stated confidence rises? Pooled by family
the relationship is close to flat, which is what the reliability diagrams
already show. Split by model it is clear for two models and weak for the rest,
because most models report confidence of at least 0.9 on nearly every answer
and leave little range for error to vary over.

Neutral prompt only, matching the appendix reliability diagram. Renders only
this figure; run it after generate_paper_assets.py, whose first step clears the
figure directories.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_paper_assets as G
from render_figures import (
    COLORS,
    FIG_CAPTION_DIR,
    FIG_CSV_DIR,
    FIG_PNG_DIR,
    CairoFigure,
    scale_point,
    write_csv,
    write_text,
)

NAME = "figure_11_error_rate_by_confidence_per_model"
BINS = 10
SPARSE = 20
# Display names match Table 1 of the paper.
MODELS = {
    "AR": [
        ("x-ai/grok-4.3", "Grok"),
        ("openai/gpt-4.1-mini", "GPT-4.1 mini"),
        ("google/gemini-2.5-flash", "Gemini Flash"),
    ],
    "DLM": [
        ("inception/mercury-2", "Mercury-2"),
        ("google/diffusiongemma-26B-A4B-it", "DiffusionGemma"),
        ("Dream-org/Dream-v0-Instruct-7B", "Dream"),
        ("GSAI-ML/LLaDA-8B-Instruct", "LLaDA"),
    ],
}
FAMILY_COLOR = {"AR": COLORS["Autoregressive (AR)"], "DLM": COLORS["Diffusion language model (DLM)"]}

CAPTION = (
    "Figure 11. Error rate by stated confidence for each model under the neutral prompt. "
    "Points are 10 equal-width confidence bins placed at their mean confidence; hollow markers are "
    "bins with fewer than 20 answers, and the dashed line is perfect calibration. rho is the Spearman "
    "correlation between confidence and error over all of a model's neutral-prompt answers, and the "
    "percentage is the share of answers with confidence of at least 0.9. Error falls steadily with "
    "confidence only for Grok and Mercury-2. Gemini Flash, GPT-4.1 mini, DiffusionGemma and LLaDA put "
    "99% or more of their answers in the top bin, and Dream splits its answers between the top bin and "
    "confidence near zero, which leaves little range for error to vary over.\n"
)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda k: values[k])
        out = [0.0] * len(values)
        i = 0
        while i < len(values):
            j = i
            while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2
            i = j + 1
        return out

    if len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


def model_summary(rows: list[dict], model_id: str, family: str) -> dict:
    usable = [
        row for row in rows
        if row["model_id"] == model_id
        and row.get("parsed_confidence") is not None
        and row.get("correct_auto") is not None
    ]
    by_bin = defaultdict(list)
    for row in usable:
        by_bin[min(int(row["parsed_confidence"] * BINS), BINS - 1)].append(row)

    points = []
    for index, group in sorted(by_bin.items()):
        errors = sum(1 for row in group if not row["correct_auto"])
        points.append(
            {
                "model_id": model_id,
                "model_family": family,
                "bin_low": index / BINS,
                "bin_high": (index + 1) / BINS,
                "mean_confidence": sum(row["parsed_confidence"] for row in group) / len(group),
                "error_rate": errors / len(group),
                "bin_count": len(group),
                "error_count": errors,
            }
        )
    rho = spearman(
        [row["parsed_confidence"] for row in usable],
        [0.0 if row["correct_auto"] else 1.0 for row in usable],
    )
    top = sum(1 for row in usable if row["parsed_confidence"] >= 0.9) / len(usable)
    for point in points:
        point["spearman_rho"] = rho
        point["share_confidence_at_least_0.9"] = top
    return {"points": points, "rho": rho, "top": top, "n": len(usable)}


def draw_panel(fig: CairoFigure, left: int, top: int, name: str, family: str, summary: dict) -> None:
    width, height = 360, 300
    right, bottom = left + width, top + height
    color = FAMILY_COLOR[family]

    for value in (0.0, 0.5, 1.0):
        x, y = scale_point(value, value, left, right, top, bottom)
        fig.line(left, y, right, y, "#e3e3e3", 1.0)
        fig.line(x, top, x, bottom, "#e3e3e3", 1.0)
        fig.text(x, bottom + 40, f"{value:.1f}", 28, "#444444", align="center")
        fig.text(left - 16, y + 11, f"{value:.1f}", 28, "#444444", align="right")
    fig.line(left, bottom, right, bottom, "#333333", 2)
    fig.line(left, top, left, bottom, "#333333", 2)
    fig.line(left, top, right, bottom, "#888888", 2, dash=(8, 7))

    fig.text(left, top - 52, name, 36, bold=True)
    rho = "n/a" if summary["rho"] is None else f"{summary['rho']:+.2f}"
    fig.text(left, top - 14, f"\u03c1 = {rho}   {summary['top']:.0%} at \u2265 0.9", 29, "#555555")

    rgb = fig._rgb(color)
    for point in summary["points"]:
        x, y = scale_point(point["mean_confidence"], point["error_rate"], left, right, top, bottom)
        box = (x - 9, y - 9, x + 9, y + 9)
        sparse = point["bin_count"] < SPARSE
        if family == "AR":
            fig.draw.ellipse(box, fill=None if sparse else rgb, outline=rgb, width=4)
        else:
            fig.draw.rectangle(box, fill=None if sparse else rgb, outline=rgb, width=4)
    fig.text(left + width / 2, bottom + 88, "Mean confidence in bin", 29, "#333333", align="center")


def main() -> None:
    rows, _ = G.load_all_rows()
    rows = [row for row in rows if row["prompt_condition"] == "neutral"]

    fig = CairoFigure(FIG_PNG_DIR / f"{NAME}.png", width=1590, height=1640)
    fig.text(40, 60, "Error rate by stated confidence, per model (neutral prompt)", 38, bold=True)

    columns = [150, 650, 1150]
    row_tops = [190, 700, 1210]
    panels = []
    for family in ["AR", "DLM"]:
        summaries = [(name, model_summary(rows, model_id, family), family) for model_id, name in MODELS[family]]
        summaries.sort(key=lambda item: 1.0 if item[1]["rho"] is None else item[1]["rho"])
        panels.extend(summaries)
    data = []
    for index, (name, summary, family) in enumerate(panels):
        data.extend(summary["points"])
        r, col = divmod(index, 3)
        draw_panel(fig, columns[col], row_tops[r], name, family, summary)
    for top in row_tops:
        fig.text(56, top + 150, "Error rate", 30, "#333333", align="center", rotate=-1.5708)

    lx, ly = 680, 1250
    ar_rgb = fig._rgb(FAMILY_COLOR["AR"])
    dlm_rgb = fig._rgb(FAMILY_COLOR["DLM"])
    fig.draw.ellipse((lx - 11, ly - 11, lx + 11, ly + 11), fill=ar_rgb)
    fig.text(lx + 30, ly + 14, "Autoregressive (AR)", 30)
    fig.draw.rectangle((lx - 11, ly + 44, lx + 11, ly + 66), fill=dlm_rgb)
    fig.text(lx + 30, ly + 70, "Diffusion (DLM)", 30)
    fig.draw.ellipse((lx - 11, ly + 100, lx + 11, ly + 122), outline=(90, 90, 90), width=4)
    fig.text(lx + 30, ly + 126, "Hollow: bins with under 20 answers", 29, "#555555")
    fig.line(lx - 14, ly + 168, lx + 14, ly + 168, "#888888", 2, dash=(7, 5))
    fig.text(lx + 30, ly + 182, "Perfect calibration", 29, "#555555")
    fig.text(lx - 14, ly + 250, "\u03c1: Spearman correlation between confidence", 29, "#555555")
    fig.text(lx - 14, ly + 292, "and error; % = share of answers at \u2265 0.9", 29, "#555555")
    fig.write()

    fields = [
        "model_id", "model_family", "bin_low", "bin_high", "mean_confidence", "error_rate",
        "bin_count", "error_count", "spearman_rho", "share_confidence_at_least_0.9",
    ]
    write_csv(FIG_CSV_DIR / f"{NAME}_data.csv", [{k: G.fmt(v) for k, v in row.items()} for row in data], fields)
    write_text(FIG_CAPTION_DIR / "figure_11_caption.txt", CAPTION)
    print(f"wrote {NAME}.png, data csv, caption")


if __name__ == "__main__":
    main()
