from __future__ import annotations
 
import json
import math
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
    """One row from run_model.py output. Holds the normalized schema; the
    loader maps Edward's field names (sample_id, model_name, raw_response) and
    the older alternatives (sample_idx, model, raw_output) onto these.
    """
    question_id: str
    dataset: str
    condition: str
    sample_idx: int              # canonical name. Source field is `sample_id` in Edward's output.
    model: str                   # source field: `model_name`
    prompt: str
    answer: str | None
    confidence: float | None     # 0..1 float. Null when parsing failed.
    ground_truth: str | list[str]
    correct: bool | None
    raw_output: str | None       # source field: `raw_response`
    parse_success: bool          # explicit field from Edward; falls back to (answer + confidence) presence
    answer_type: str | None = None       # "numeric" | "short_answer" | "multiple_choice"
    model_architecture: str | None = None  # "AR" | "DLM"
    short_explanation: str | None = None
    # Optional logprob-derived answer confidence in [0,1]. This is the
    # exp(sum answer-token logprobs) (or length-normalized variant) computed
    # upstream when token logprobs are available. It is None for the current AR
    # run because the OpenRouter provider route returned null logprobs, which is
    # exactly why logprob-based BAS is OPTIONAL (see bas_score). Source field:
    # `logprob_confidence` or `answer_logprob_confidence`.
    logprob_confidence: float | None = None
 
    @property
    def parsed_ok(self) -> bool:
        """True iff JSON parsed AND both answer and confidence are present."""
        return self.parse_success and self.answer is not None and self.confidence is not None
 
    @property
    def confidence_norm(self) -> float | None:
        """Alias kept for the notebook's old call sites. Confidence is already
        on the 0..1 scale after loading, so this just returns it as-is."""
        return self.confidence
 
 
def _pick(d: dict, *keys, default=None):
    """Return the first key present in d. Lets us accept both Edward's field
    names and the older ones without duplicating logic everywhere."""
    for k in keys:
        if k in d:
            return d[k]
    return default
 
 
def load_generations(path: str | Path) -> list[Generation]:
    """Load JSONL output from run_model.py into Generation objects.
 
    Handles both Edward's schema (sample_id, model_name, raw_response) and the
    older schema (sample_idx, model, raw_output). Auto-detects 0..100 vs 0..1
    confidence scale and normalizes everything to 0..1.
    """
    raw_rows: list[dict] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_rows.append(json.loads(line))
 
    # Detect confidence scale once across the whole file. If any confidence is
    # > 1, treat the file as 0..100 and rescale. Otherwise treat as 0..1.
    confs_seen = [
        r.get("confidence") for r in raw_rows
        if r.get("confidence") is not None
    ]
    on_0_100_scale = any(c > 1 for c in confs_seen)
 
    rows: list[Generation] = []
    for d in raw_rows:
        raw_conf = d.get("confidence")
        if raw_conf is None:
            conf: float | None = None
        else:
            conf = float(raw_conf)
            if on_0_100_scale:
                conf /= 100.0
 
        rows.append(Generation(
            question_id=d["question_id"],
            dataset=d["dataset"],
            condition=d["condition"],
            sample_idx=int(_pick(d, "sample_id", "sample_idx", default=0)),
            model=_pick(d, "model_name", "model", default=""),
            prompt=d.get("prompt", ""),
            answer=d.get("answer"),
            confidence=conf,
            ground_truth=d.get("ground_truth", ""),
            correct=d.get("correct"),
            raw_output=_pick(d, "raw_response", "raw_output"),
            parse_success=bool(d.get(
                "parse_success",
                d.get("answer") is not None and d.get("confidence") is not None,
            )),
            answer_type=d.get("answer_type"),
            model_architecture=d.get("model_architecture"),
            short_explanation=d.get("short_explanation"),
            logprob_confidence=_coerce_float(
                _pick(d, "logprob_confidence", "answer_logprob_confidence")
            ),
        ))
    return rows
 
 
def _coerce_float(v: Any) -> float | None:
    """None-safe float() used for optional numeric fields like logprob_confidence."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
 
 
# ---------------------------------------------------------------------------
# Metric 1: JSON parse success rate
# ---------------------------------------------------------------------------
 
def parse_success_rate(gens: Iterable[Generation]) -> float:
    """
    Fraction of generations where the model emitted parseable JSON.
 
    Uses Edward's explicit `parse_success` field. Falls back to checking that
    both `answer` and `confidence` are non-null when the field is absent.
 
    Returns a value in [0, 1]. A rate below ~0.9 suggests something is wrong
    with the prompt template, the model's instruction-following, or the parser.
 
    Note: in Edward's pilot run this is ~99.91% (2248/2250) because the failed
    rows are filtered out of ar_parsed_generations.jsonl. To see the true rate
    you'd compare against the raw file's row count.
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
# Metric 6b: BAS (Behavioral Alignment Score)  [OPTIONAL]
# ---------------------------------------------------------------------------
#
# BAS is the decision-theoretic confidence metric from Wu, Gustafsson et al.,
# "A Decision-Theoretic Approach to Evaluating Large Language Model Confidence"
# (Oxford, 2026), the Paper-1 PDF in the repo root. Per-example utility:
#
#       U(s, Z) = s                 if the answer is correct (Z = 1)
#       U(s, Z) = s + ln(1 - s)     if the answer is wrong   (Z = 0)
#       BAS     = mean over examples of U(s_i, Z_i)
#
# where s is the model's confidence in [0, 1]. Range is (-inf, 1]; higher is
# better. The ln(1 - s) term is the whole point: a wrong answer at s -> 1 is
# punished toward -inf, so BAS exposes catastrophic overconfidence that ECE and
# AUROC average away.
#
# WHY THIS IS MARKED OPTIONAL
# ---------------------------
# BAS needs a per-answer confidence s. Two sources:
#   - "stated":  the model's self-reported `confidence` field. We HAVE this for
#                the AR run, so BAS is computable today with confidence_source
#                = "stated" (the default).
#   - "logprob": s derived from token logprobs (exp of summed answer-token
#                logprobs, or a length-normalized variant). We do NOT have this
#                for the AR run because the OpenRouter provider route returned
#                null logprobs. With confidence_source="logprob", BAS returns
#                None until a logprob-exposing route is wired up. That is the
#                "optional if logits unavailable" path requested by the team.
#
# NUMERICAL NOTE (important for this dataset)
# -------------------------------------------
# ln(1 - s) is -inf at exactly s = 1.0. The AR run has 524 rows at stated
# confidence 1.0 that are wrong, so the *unclipped* BAS is literally -inf. We
# clip s into [epsilon, 1 - epsilon] with a documented, configurable epsilon.
# The result is sensitive to epsilon (e.g. eps=1e-6 -> ~-3.48, eps=1e-3 ->
# ~-1.87 on the current AR data), so always report the epsilon alongside the
# number, and use bas_report() to see how many rows are saturated at the cap.
 
 
def _confidence_for(g: Generation, source: str) -> float | None:
    """Return the confidence value BAS should use for one row, per source.
 
    source="stated"  -> g.confidence_norm (self-reported, 0..1)
    source="logprob" -> g.logprob_confidence (None when logprobs unavailable)
    """
    if source == "stated":
        return g.confidence_norm
    if source == "logprob":
        return g.logprob_confidence
    raise ValueError(f"unknown confidence_source: {source!r} (use 'stated' or 'logprob')")
 
 
def bas_score(
    gens: Iterable[Generation],
    confidence_source: str = "stated",
    epsilon: float = 1e-6,
) -> float | None:
    """
    Behavioral Alignment Score (mean decision-theoretic utility). Higher = better,
    max 1.0, no lower bound.
 
    Uses only rows where BOTH the chosen confidence source and `correct` are
    non-null (same usable-set discipline as ECE / AUROC).
 
    Returns None when:
      - there are zero usable rows, OR
      - confidence_source="logprob" and no row carries a logprob_confidence
        (the "logits unavailable -> BAS optional" path).
 
    epsilon clips confidence into [epsilon, 1 - epsilon] so ln(1 - s) stays
    finite. Report it next to the score; it changes the number materially when
    the model emits exact 1.0 confidences on wrong answers.
    """
    usable = [
        (c, int(g.correct))
        for g in gens
        if (c := _confidence_for(g, confidence_source)) is not None
        and g.correct is not None
    ]
    if not usable:
        return None
 
    total = 0.0
    for s, z in usable:
        s = min(max(s, epsilon), 1.0 - epsilon)
        total += s if z else s + math.log(1.0 - s)
    return float(total / len(usable))
 
 
def bas_report(
    gens: Iterable[Generation],
    confidence_source: str = "stated",
    epsilon: float = 1e-6,
) -> dict[str, Any]:
    """
    BAS plus the diagnostics you need to defend the number in the paper.
 
    Returns a dict:
      {
        "available":       bool,    # False => metric is N/A for this source
        "confidence_source": str,
        "epsilon":         float,
        "bas":             float | None,
        "n_usable":        int,     # rows with confidence + correct
        "n_saturated":     int,     # wrong rows clipped at the (1 - eps) cap
        "bas_correct_mean":   float | None,  # mean U over correct rows
        "bas_incorrect_mean": float | None,  # mean U over wrong rows (the tail)
      }
 
    n_saturated counts the catastrophic-overconfidence rows (wrong AND confidence
    >= 1 - eps). A large n_saturated means BAS is dominated by, and sensitive to,
    the epsilon clip, which is itself a reportable finding about the model.
    """
    usable = [
        (c, int(g.correct))
        for g in gens
        if (c := _confidence_for(g, confidence_source)) is not None
        and g.correct is not None
    ]
    report: dict[str, Any] = {
        "available": bool(usable),
        "confidence_source": confidence_source,
        "epsilon": epsilon,
        "bas": None,
        "n_usable": len(usable),
        "n_saturated": 0,
        "bas_correct_mean": None,
        "bas_incorrect_mean": None,
    }
    if not usable:
        return report
 
    correct_us, wrong_us = [], []
    n_sat = 0
    for s, z in usable:
        s_clipped = min(max(s, epsilon), 1.0 - epsilon)
        if z:
            correct_us.append(s_clipped)
        else:
            wrong_us.append(s_clipped + math.log(1.0 - s_clipped))
            if s >= 1.0 - epsilon:
                n_sat += 1
 
    all_u = correct_us + wrong_us
    report["bas"] = float(np.mean(all_u))
    report["n_saturated"] = n_sat
    report["bas_correct_mean"] = float(np.mean(correct_us)) if correct_us else None
    report["bas_incorrect_mean"] = float(np.mean(wrong_us)) if wrong_us else None
    return report
 
 
# ---------------------------------------------------------------------------
# Metric 7: Disagreement rate
# ---------------------------------------------------------------------------
 
_PUNCT_RE = re.compile(r"[^\w\s]")
 
 
def _normalize_answer(s: Any) -> str:
    """Lowercase, strip punctuation, collapse whitespace. For disagreement only.
 
    Coerces non-string answers (GSM8K numeric answers come through as float)
    to string first so this works uniformly.
    """
    if not isinstance(s, str):
        # Normalize numeric trailing zeros so "74" and "74.0" collide.
        if isinstance(s, float) and s.is_integer():
            s = str(int(s))
        else:
            s = str(s)
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
    bas_confidence_source: str = "stated",
    bas_epsilon: float = 1e-6,
) -> list[dict[str, Any]]:
    """
    Compute every metric, optionally grouped.
 
    Pass group_by=["condition"] or ["condition", "dataset"] for the breakdown
    tables in the paper.
 
    BAS is included as an OPTIONAL column. With bas_confidence_source="stated"
    (default) it is computed from self-reported confidence. With "logprob" it
    is None until token logprobs are available. The "bas" column is omitted
    entirely from the output rows when BAS is None for every group, so the
    table stays clean when the metric is unavailable.
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
        row["bas"] = bas_score(
            items, confidence_source=bas_confidence_source, epsilon=bas_epsilon
        )
        row["disagreement_rate"] = disagreement_rate(items)
        out.append(row)
 
    # Drop the optional BAS column entirely if it is unavailable everywhere,
    # so tables don't carry a column of "n/a" when logprobs aren't wired up.
    if all(r.get("bas") is None for r in out):
        for r in out:
            r.pop("bas", None)
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
