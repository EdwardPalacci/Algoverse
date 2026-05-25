#this was created using Claude and the analysis_notebook, not directly through my own work just used to verify whether or not the analysis_notebooked worked

from __future__ import annotations
 
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
 
import numpy as np
 
 
# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
 
@dataclass
class Generation:
    """One row from run_model.py output. Mirrors the JSONL schema exactly."""
    question_id: str
    dataset: str
    condition: str
    sample_idx: int
    model: str
    prompt: str
    answer: str | None
    confidence: int | None
    ground_truth: str
    correct: bool | None
    raw_output: str | None
 
    @property
    def parsed_ok(self) -> bool:
        """True iff both answer and confidence were successfully parsed."""
        return self.answer is not None and self.confidence is not None
 
    @property
    def confidence_norm(self) -> float | None:
        """Confidence on the 0-1 scale used by ECE and AUROC. None if not parsed."""
        if self.confidence is None:
            return None
        return self.confidence / 100.0
 
 
def load_generations(path: str | Path) -> list[Generation]:
    """Load JSONL output from run_model.py into Generation objects."""
    rows: list[Generation] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append(Generation(
                question_id=d["question_id"],
                dataset=d["dataset"],
                condition=d["condition"],
                sample_idx=int(d["sample_idx"]),
                model=d.get("model", ""),
                prompt=d.get("prompt", ""),
                answer=d.get("answer"),
                confidence=(int(d["confidence"]) if d.get("confidence") is not None else None),
                ground_truth=d.get("ground_truth", ""),
                correct=d.get("correct"),
                raw_output=d.get("raw_output"),
            ))
    return rows
 
 
# ---------------------------------------------------------------------------
# Metric 1: JSON parse success rate
# ---------------------------------------------------------------------------
 
def parse_success_rate(gens: Iterable[Generation]) -> float:
    """
    Fraction of generations where the model emitted parseable JSON containing
    both an `answer` and a `confidence` field.
 
    Returns a value in [0, 1]. A rate below ~0.9 suggests something is wrong with
    either the prompt template, the model's instruction-following, or the parser
    in run_model.py.
    """
    gens = list(gens)
    if not gens:
        return float("nan")
    return float(np.mean([int(g.parsed_ok) for g in gens]))
 
 
def parse_failures_by_condition(gens: Iterable[Generation]) -> dict[str, dict[str, int]]:
    """
    Diagnostic: how many parse failures per condition? Sometimes the
    overconfidence prompt is so aggressive the model breaks format.
    Returns {condition: {"parsed": N, "failed": N, "total": N}}.
    """
    out: dict[str, dict[str, int]] = defaultdict(lambda: {"parsed": 0, "failed": 0, "total": 0})
    for g in gens:
        out[g.condition]["total"] += 1
        if g.parsed_ok:
            out[g.condition]["parsed"] += 1
        else:
            out[g.condition]["failed"] += 1
    return dict(out)
 
 
# ---------------------------------------------------------------------------
# Metric 2: Mean confidence by condition
# ---------------------------------------------------------------------------
 
def mean_confidence_by_condition(gens: Iterable[Generation]) -> dict[str, float]:
    """
    Mean confidence (0-1 scale) per condition. Skips parse failures.
    Returns NaN for conditions with zero parsed rows.
 
    This is the manipulation check for the paper: if the overconfidence
    condition's mean confidence is not substantially higher than neutral,
    your intervention failed.
    """
    groups: dict[str, list[float]] = defaultdict(list)
    for g in gens:
        if g.confidence_norm is not None:
            groups[g.condition].append(g.confidence_norm)
    return {c: float(np.mean(v)) if v else float("nan") for c, v in groups.items()}
 
 
# ---------------------------------------------------------------------------
# Metric 3: Confidence histogram by condition
# ---------------------------------------------------------------------------
 
def confidence_histogram(
    gens: Iterable[Generation],
    n_bins: int = 10,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Histogram of confidence values per condition. Skips parse failures.
 
    Returns {condition: (counts, bin_edges)} where bin_edges has length n_bins+1.
    Use this to draw side-by-side histograms in the notebook.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    groups: dict[str, list[float]] = defaultdict(list)
    for g in gens:
        if g.confidence_norm is not None:
            groups[g.condition].append(g.confidence_norm)
    return {
        c: (np.histogram(np.array(v), bins=bin_edges)[0], bin_edges)
        for c, v in groups.items()
    }
 
 
# ---------------------------------------------------------------------------
# Metric 4: Accuracy
# ---------------------------------------------------------------------------
 
def accuracy(gens: Iterable[Generation]) -> float | None:
    """
    Fraction of generations where `correct` is True.
 
    Returns None if no rows have a non-null `correct` field, which happens when
    grading hasn't been run yet. This is intentional: silent zeros would hide a
    "we haven't graded anything" situation.
    """
    graded = [g for g in gens if g.correct is not None]
    if not graded:
        return None
    return float(np.mean([int(g.correct) for g in graded]))
 
 
def accuracy_by_condition(gens: Iterable[Generation]) -> dict[str, float | None]:
    """Per-condition accuracy. Same None-on-no-grades semantics."""
    groups: dict[str, list[Generation]] = defaultdict(list)
    for g in gens:
        groups[g.condition].append(g)
    return {c: accuracy(v) for c, v in groups.items()}
 
 
# ---------------------------------------------------------------------------
# Metric 5: Expected Calibration Error
# ---------------------------------------------------------------------------
 
def expected_calibration_error(
    gens: Iterable[Generation],
    n_bins: int = 10,
) -> float | None:
    """
    Equal-width-binning ECE (Naeini et al. 2015), matching the formula in
    Section 4.5 of the proposal:
 
        ECE = sum_m (|B_m| / n) * |acc(B_m) - conf(B_m)|
 
    Only uses rows where BOTH `confidence` and `correct` are non-null.
    Returns None when there are zero usable rows.
    """
    usable = [
        g for g in gens
        if g.confidence_norm is not None and g.correct is not None
    ]
    if not usable:
        return None
 
    confs = np.array([g.confidence_norm for g in usable])
    corrects = np.array([int(g.correct) for g in usable])
 
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    # digitize returns 1..n_bins; subtract 1 and clip so values exactly at 1.0
    # land in the top bin instead of falling off the end.
    bin_idx = np.clip(np.digitize(confs, bin_edges, right=False) - 1, 0, n_bins - 1)
 
    n = len(confs)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        bin_acc = corrects[mask].mean()
        bin_conf = confs[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)
 
 
def reliability_bins(
    gens: Iterable[Generation],
    n_bins: int = 10,
) -> dict[str, np.ndarray]:
    """
    Per-bin data for a reliability diagram.
 
    Returns dict with bin_centers, bin_acc, bin_conf, bin_count arrays
    (each length n_bins). bin_acc and bin_conf are NaN for empty bins.
    """
    usable = [
        g for g in gens
        if g.confidence_norm is not None and g.correct is not None
    ]
    if not usable:
        return {
            "bin_centers": np.linspace(0.05, 0.95, n_bins),
            "bin_acc": np.full(n_bins, np.nan),
            "bin_conf": np.full(n_bins, np.nan),
            "bin_count": np.zeros(n_bins, dtype=int),
        }
 
    confs = np.array([g.confidence_norm for g in usable])
    corrects = np.array([int(g.correct) for g in usable])
 
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_idx = np.clip(np.digitize(confs, bin_edges, right=False) - 1, 0, n_bins - 1)
 
    accs = np.full(n_bins, np.nan)
    mean_confs = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = bin_idx == b
        counts[b] = mask.sum()
        if mask.any():
            accs[b] = corrects[mask].mean()
            mean_confs[b] = confs[mask].mean()
    return {
        "bin_centers": bin_centers,
        "bin_acc": accs,
        "bin_conf": mean_confs,
        "bin_count": counts,
    }
 
 
# ---------------------------------------------------------------------------
# Metric 6: AUROC
# ---------------------------------------------------------------------------
 
def auroc(gens: Iterable[Generation]) -> float | None:
    """
    AUROC of confidence as a predictor of correctness.
 
    Interpretation:
      1.0  = confidence perfectly ranks right above wrong
      0.5  = no signal
      0.0  = high confidence on wrong answers (the pathology the paper hunts)
 
    Computed via the Mann-Whitney U formulation:
       AUROC = P(conf(correct) > conf(incorrect))
    with half-credit for ties. No sklearn dependency.
 
    Returns None when:
      - there are zero usable rows, OR
      - every usable row is correct, OR
      - every usable row is incorrect
    In the last two cases AUROC is mathematically undefined (you can't rank
    a class against an empty class).
    """
    usable = [
        g for g in gens
        if g.confidence_norm is not None and g.correct is not None
    ]
    if not usable:
        return None
 
    confs = np.array([g.confidence_norm for g in usable])
    corrects = np.array([int(g.correct) for g in usable])
    if corrects.sum() in (0, len(corrects)):
        return None
 
    pos = confs[corrects == 1]
    neg = confs[corrects == 0]
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))
 
 
# ---------------------------------------------------------------------------
# Metric 7: Disagreement rate
# ---------------------------------------------------------------------------
 
_PUNCT_RE = re.compile(r"[^\w\s]")
 
 
def _normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. For disagreement only."""
    s = s.lower().strip()
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
 
 
def disagreement_rate(gens: Iterable[Generation]) -> float | None:
    """
    Proposal Section 4.5 definition: for each (question_id, condition) group with
    k>=2 valid samples, compute (k - modal_count) / k, then average across groups.
 
    Returns None if no group has at least 2 valid (parsed) samples. With k=1 the
    metric is undefined, and we'd rather signal "undefined" than silently return 0.
    """
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for g in gens:
        if g.answer is not None:
            groups[(g.question_id, g.condition)].append(_normalize_answer(g.answer))
 
    rates = []
    for answers in groups.values():
        if len(answers) < 2:
            continue
        modal_count = Counter(answers).most_common(1)[0][1]
        rates.append((len(answers) - modal_count) / len(answers))
 
    if not rates:
        return None
    return float(np.mean(rates))
 
 
# ---------------------------------------------------------------------------
# Metric 8: Unique answers per question (fallback when grading is not done)
# ---------------------------------------------------------------------------
 
def unique_answers_per_question(
    gens: Iterable[Generation],
) -> list[dict[str, Any]]:
    """
    For each (question_id, condition) group, report how many distinct answers
    appeared across samples.
 
    This is the answer-distribution proxy your mentor asked for when grading
    isn't ready yet: high unique-answer counts mean the model is uncertain
    about that question, even before we know which answer is right.
 
    Returns a list of dicts, one per group:
      {
        "question_id":   "gsm8k_00007",
        "condition":     "neutral",
        "dataset":       "gsm8k",
        "n_samples":     5,
        "n_unique":      3,
        "modal_answer":  "42",
        "modal_count":   3,
      }
    """
    groups: dict[tuple[str, str], list[Generation]] = defaultdict(list)
    for g in gens:
        groups[(g.question_id, g.condition)].append(g)
 
    rows = []
    for (qid, cond), gs in sorted(groups.items()):
        valid = [g for g in gs if g.answer is not None]
        if not valid:
            rows.append({
                "question_id": qid, "condition": cond,
                "dataset": gs[0].dataset,
                "n_samples": len(gs), "n_unique": 0,
                "modal_answer": None, "modal_count": 0,
            })
            continue
        answers = [_normalize_answer(g.answer) for g in valid]
        counter = Counter(answers)
        modal_answer, modal_count = counter.most_common(1)[0]
        rows.append({
            "question_id": qid, "condition": cond,
            "dataset": gs[0].dataset,
            "n_samples": len(valid), "n_unique": len(counter),
            "modal_answer": modal_answer, "modal_count": modal_count,
        })
    return rows
 
 
# ---------------------------------------------------------------------------
# Headline summary table
# ---------------------------------------------------------------------------
 
def summarize(
    gens: Iterable[Generation],
    group_by: list[str] | None = None,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """
    Compute every metric, optionally grouped.
 
    Pass group_by=["condition"] or ["condition", "dataset"] for the breakdown
    tables in the paper.
    """
    gens = list(gens)
    if not gens:
        return []
 
    if not group_by:
        groups: dict[tuple, list[Generation]] = {(): gens}
    else:
        groups = defaultdict(list)
        for g in gens:
            key = tuple(getattr(g, attr) for attr in group_by)
            groups[key].append(g)
 
    out = []
    for key, items in sorted(groups.items()):
        row: dict[str, Any] = {attr: v for attr, v in zip(group_by or [], key)}
        row["n_rows"] = len(items)
        row["n_questions"] = len({g.question_id for g in items})
        row["parse_success"] = parse_success_rate(items)
        row["accuracy"] = accuracy(items)
        # mean_confidence only over parsed rows
        parsed_confs = [g.confidence_norm for g in items if g.confidence_norm is not None]
        row["mean_confidence"] = float(np.mean(parsed_confs)) if parsed_confs else float("nan")
        row["ece"] = expected_calibration_error(items, n_bins=n_bins)
        row["auroc"] = auroc(items)
        row["disagreement_rate"] = disagreement_rate(items)
        out.append(row)
    return out
 
 
def format_summary(rows: list[dict[str, Any]]) -> str:
    """Fixed-width text table for printing in the notebook."""
    if not rows:
        return "(no rows)"
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(_fmt(r[c])) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    sep = "  ".join("-" * widths[c] for c in cols)
    body = "\n".join("  ".join(_fmt(r[c]).ljust(widths[c]) for c in cols) for r in rows)
    return f"{header}\n{sep}\n{body}"
 
 
def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if np.isnan(v):
            return "nan"
        return f"{v:.3f}"
    return str(v)
 
