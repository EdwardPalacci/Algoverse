# %%
# ---------------------------------------------------------------------------
# Imports and path setup
# ---------------------------------------------------------------------------
import json
import random
import sys
from collections import Counter
from pathlib import Path
 
import matplotlib.pyplot as plt
import numpy as np
 
# Make src/ importable whether this file is run as a script or cell-by-cell.
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE.parent / "src"))
 
from compute_basic_metrics import (
    accuracy,
    accuracy_by_condition,
    auroc,
    bas_report,
    bas_score,
    confidence_histogram,
    disagreement_rate,
    expected_calibration_error,
    format_summary,
    load_generations,
    mean_confidence_by_condition,
    parse_failures_by_condition,
    parse_success_rate,
    reliability_bins,
    summarize,
    unique_answers_per_question,
)
 
# BAS confidence source. "stated" uses the model's self-reported confidence
# (available now). Switch to "logprob" once a logprob-exposing route is wired
# up; until then logprob-BAS reports as unavailable (None) rather than crashing.
BAS_CONFIDENCE_SOURCE = "stated"
# Clip for ln(1 - s) in BAS. Report this value next to any BAS number; the AR
# run has hundreds of wrong rows at confidence 1.0, so the score depends on it.
BAS_EPSILON = 1e-6
 
# Prefer the graded file (accuracy / ECE / AUROC available). Fall back to the
# pre-grading file, then to a dummy file on a clean checkout.
_GRADED_RESULTS = HERE.parent / "outputs" / "ar_graded_generations.jsonl"
_PARSED_RESULTS = HERE.parent / "outputs" / "ar_parsed_generations.jsonl"
_DUMMY_RESULTS = HERE / "dummy_results.jsonl"
if _GRADED_RESULTS.exists():
    RESULTS_PATH = _GRADED_RESULTS
elif _PARSED_RESULTS.exists():
    RESULTS_PATH = _PARSED_RESULTS
else:
    RESULTS_PATH = _DUMMY_RESULTS
 
# Tommy asked for all figures + tables under ./fig_tabs (his naming). Keep
# outputs/ for raw JSONL artifacts and fig_tabs/ for the human-facing report.
FIGURES_DIR = HERE.parent / "fig_tabs"
ARTIFACTS_DIR = HERE.parent / "fig_tabs"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
 
 
# %% [markdown]
# ## Step 0: generate dummy data
#
# Delete or skip this block once real generations exist. The dummy data has the
# same schema as run_model.py output, including ~10% parse failures and a
# fraction of rows with `correct` still null (simulating "grading not yet
# done"). Three conditions have different confidence distributions so the
# calibration story is visible.
 
# %%
def _make_dummy_data(path: Path) -> None:
    random.seed(0)
    rows = []
    datasets = ["gsm8k", "truthfulqa", "triviaqa"]
    conditions = ["neutral", "cautious", "overconfidence"]
    k_samples = 5
    questions_per_dataset = 20  # 20 * 3 * 3 * 5 = 900 rows
 
    for ds in datasets:
        for qi in range(questions_per_dataset):
            qid = f"{ds}_{qi:05d}"
            gt = "A"
            easy = qi < (questions_per_dataset * 0.6)
            for cond in conditions:
                for s in range(k_samples):
                    parse_failed = random.random() < (0.15 if cond == "overconfidence" else 0.08)
                    grading_done = random.random() < 0.80
 
                    if parse_failed:
                        answer, confidence, correct = None, None, None
                    else:
                        if easy:
                            correct = random.random() < 0.85
                            answer = "A" if correct else "B"
                        else:
                            answer = random.choice(["A", "B", "C", "D"])
                            correct = answer == "A"
                        if cond == "neutral":
                            confidence = 88 if correct else 35
                        elif cond == "cautious":
                            confidence = 70 if correct else 25
                        else:  # overconfidence
                            confidence = 98
                        if not grading_done:
                            correct = None
 
                    # Use Edward's field names so dummy data round-trips
                    # through load_generations the same way real data does.
                    rows.append({
                        "question_id": qid,
                        "dataset": ds,
                        "condition": cond,
                        "sample_id": s,
                        "model_name": "dummy-model",
                        "model_architecture": "AR",
                        "prompt": f"<question {qid}>",
                        "answer_type": "multiple_choice",
                        "ground_truth": gt,
                        "raw_response": (
                            f'{{"answer": "{answer}", "confidence": {confidence / 100 if confidence is not None else 0}}}'
                            if not parse_failed else "<malformed model output>"
                        ),
                        "answer": answer,
                        # Dummy data uses the 0..1 scale to match Edward's real output.
                        "confidence": None if confidence is None else confidence / 100.0,
                        "short_explanation": None,
                        "parse_success": not parse_failed,
                        "correct": correct,
                    })
 
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} dummy rows to {path}")
 
 
if not RESULTS_PATH.exists():
    _make_dummy_data(RESULTS_PATH)
 
 
# %% [markdown]
# ## Step 1: load and sanity-check
 
# %%
gens = load_generations(RESULTS_PATH)
n_q = len({g.question_id for g in gens})
n_samples = max(g.sample_idx for g in gens) + 1
n_cond = len({g.condition for g in gens})
 
print(f"Loaded {len(gens)} generations from {RESULTS_PATH}")
print(f"  unique questions:    {n_q}")
print(f"  conditions:          {sorted({g.condition for g in gens})}")
print(f"  samples per (q, c):  {n_samples}")
print(f"  per-condition rows:  {n_q} questions x {n_samples} samples = {n_q * n_samples}")
print(f"  datasets:            {sorted({g.dataset for g in gens})}")
print(f"  models:              {sorted({g.model for g in gens})}")
print(f"  architecture:        {sorted({g.model_architecture for g in gens if g.model_architecture})}")
print(f"  answer_types:        {sorted({g.answer_type for g in gens if g.answer_type})}")
print(f"  graded rows:         {sum(1 for g in gens if g.correct is not None)} / {len(gens)}")
print(f"\nFirst row:\n  {gens[0]}")
 
# Derive the list of conditions present in the data rather than hardcoding
# (Edward uses "overconfident", the proposal called it "overconfidence",
# Tommy's notes vary). Sort so "cautious", "neutral", "overconfident" comes
# out in a sensible order.
def _condition_order(c: str) -> int:
    # cautious -> 0, neutral -> 1, anything starting with "overconf" -> 2
    if c == "cautious":
        return 0
    if c == "neutral":
        return 1
    if c.startswith("overconf"):
        return 2
    return 99
 
conds = sorted({g.condition for g in gens}, key=_condition_order)
print(f"\nWill report metrics for conditions (in order): {conds}")
 
 
# %% [markdown]
# ## Metric 1: JSON parse success rate
#
# run_model.py asks the model for JSON. Sometimes it doesn't comply. Parse
# failures show up as answer=None, confidence=None, correct=None. A parse rate
# below ~0.9 means something is wrong (prompt template, model refusal, parser).
#
# Reported overall AND by condition: the overconfidence prompt sometimes
# causes more format breaks because the model gets aggressive.
 
# %%
rate = parse_success_rate(gens)
print(f"Overall JSON parse success rate: {rate:.1%}")
 
print("\nBy condition:")
for cond, stats in parse_failures_by_condition(gens).items():
    pct = stats["parsed"] / stats["total"] if stats["total"] else float("nan")
    print(f"  {cond:>14}: {stats['parsed']:>4} parsed / {stats['total']:>4} total = {pct:.1%}")
 
 
# %% [markdown]
# ## Metric 2: Mean confidence by condition
#
# Manipulation check. If mean confidence in `overconfidence` isn't
# substantially higher than `neutral`, the intervention isn't working and the
# rest of the analysis is questionable.
#
# Ideal pattern (proposal Section "Ideal Results"):
#   cautious < neutral < overconfidence,
# with at least a 0.20 gap between neutral and overconfidence on the 0-1 scale.
 
# %%
means = mean_confidence_by_condition(gens)
for cond in conds:
    if cond in means:
        print(f"  {cond:>14}: {means[cond]:.3f}")
 
# Pick whichever overconfidence label is actually in the data ("overconfident"
# in Edward's run, "overconfidence" in the proposal).
overconf_label = next((c for c in conds if c.startswith("overconf")), None)
if overconf_label and "neutral" in means:
    gap = means[overconf_label] - means["neutral"]
    status = "OK" if gap >= 0.2 else "WEAK"
    print(f"\nNeutral -> {overconf_label} gap: {gap:+.3f}   [{status}]")
    print("Target: >= 0.20 on the 0-1 scale.")
 
 
# %% [markdown]
# ## Metric 3: Confidence histogram by condition
#
# The mean is one number. The distribution tells you more. Under
# `overconfidence`, the histogram should be heavily skewed toward 1.0 with a
# sharp peak. Under `neutral`, you should see a more spread distribution
# reflecting the model's actual uncertainty across questions.
 
# %%
hist = confidence_histogram(gens, n_bins=10)
# `conds` is already set from the data above; do not redefine it here.
fig, axes = plt.subplots(1, len(conds), figsize=(5 * len(conds), 4), sharey=True)
for ax, cond in zip(axes, conds):
    if cond not in hist:
        ax.set_title(f"{cond} (no data)")
        continue
    counts, edges = hist[cond]
    centers = (edges[:-1] + edges[1:]) / 2
    ax.bar(centers, counts, width=0.09, alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_title(f"{cond}  (n={int(counts.sum())})")
axes[0].set_ylabel("Count")
fig.suptitle("Confidence distribution by condition")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "ar_pilot_confidence_histogram.png", dpi=150, bbox_inches="tight")
print(f"Saved figure: {FIGURES_DIR / 'ar_pilot_confidence_histogram.png'}")
plt.show()
 
 
# %% [markdown]
# ## Metric 4: Accuracy (when grading is available)
#
# Returns None if no rows have a graded `correct` field. Silent zeros would
# hide an "I haven't graded anything yet" situation.
#
# Ideal pattern: accuracy roughly flat across conditions. If accuracy drops
# under overconfidence, the prompt is also degrading reasoning (a confound),
# not just inflating confidence.
 
# %%
acc = accuracy(gens)
print(f"Overall accuracy: {acc if acc is None else f'{acc:.3f}'}")
print()
for cond, a in accuracy_by_condition(gens).items():
    print(f"  {cond:>14}: {a if a is None else f'{a:.3f}'}")
 
 
# %% [markdown]
# ## Metric 5: Expected Calibration Error (ECE)
#
# Headline calibration metric. Section 4.5 of the proposal:
#
#     ECE = sum_m (|B_m| / n) * |acc(B_m) - conf(B_m)|
#
# Ideal pattern: monotonic rise across conditions, with overconfidence ECE at
# least ~2x neutral ECE.
 
# %%
for cond in conds:
    subset = [g for g in gens if g.condition == cond]
    ece = expected_calibration_error(subset)
    print(f"  {cond:>14}: ECE = {ece if ece is None else f'{ece:.3f}'}")
 
 
# %% [markdown]
# ### Reliability diagram (visual ECE)
#
# Bars below the diagonal = overconfident in that bin. Under `overconfidence`
# you should see most bars below the diagonal at high confidence values.
 
# %%
fig, axes = plt.subplots(1, len(conds), figsize=(5 * len(conds), 5))
for ax, cond in zip(axes, conds):
    subset = [g for g in gens if g.condition == cond]
    bins = reliability_bins(subset, n_bins=10)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    valid = ~np.isnan(bins["bin_acc"])
    ax.bar(
        bins["bin_centers"][valid], bins["bin_acc"][valid],
        width=0.09, alpha=0.6, label="observed accuracy",
    )
    ax.scatter(
        bins["bin_centers"][valid], bins["bin_conf"][valid],
        marker="x", label="mean confidence",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence bin")
    ax.set_title(cond)
    ax.legend(loc="upper left", fontsize=8)
axes[0].set_ylabel("Accuracy / mean confidence")
fig.suptitle("Reliability diagrams by condition")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "ar_pilot_reliability_diagram.png", dpi=150, bbox_inches="tight")
print(f"Saved figure: {FIGURES_DIR / 'ar_pilot_reliability_diagram.png'}")
plt.show()
 
 
# %% [markdown]
# ## Metric 6: AUROC of confidence as a correctness predictor
#
# AUROC asks: regardless of absolute calibration, does higher confidence at
# least RANK correctness? 1.0 = perfect ranking, 0.5 = no signal, 0.0 = high
# confidence on wrong answers.
#
# Ideal pattern: AUROC high under neutral and cautious (>0.70), collapses
# toward 0.5 under overconfidence as confidence values get flattened.
 
# %%
for cond in conds:
    subset = [g for g in gens if g.condition == cond]
    a = auroc(subset)
    print(f"  {cond:>14}: AUROC = {a if a is None else f'{a:.3f}'}")
 
 
# %% [markdown]
# ## Metric 6b: BAS (Behavioral Alignment Score)  [OPTIONAL]
#
# Decision-theoretic confidence metric from the Paper-1 PDF (Wu, Gustafsson et
# al.). Per example: U = s if correct, else s + ln(1 - s). BAS = mean U. Range
# (-inf, 1], higher is better. Unlike ECE/AUROC, BAS does not let overconfident
# errors average out: a wrong answer at confidence ~1.0 is punished toward -inf.
#
# OPTIONAL by design:
#   - BAS_CONFIDENCE_SOURCE="stated"  -> uses self-reported confidence (works now)
#   - BAS_CONFIDENCE_SOURCE="logprob" -> needs token logprobs; reports as
#     "unavailable" until an OpenRouter route that exposes logprobs is wired up.
#
# Read the epsilon and n_saturated lines below before quoting any BAS number:
# the AR run has many wrong rows at confidence exactly 1.0, so the score is
# clip-dependent and that sensitivity is itself a finding.
 
# %%
_bas_rep = bas_report(gens, confidence_source=BAS_CONFIDENCE_SOURCE, epsilon=BAS_EPSILON)
if not _bas_rep["available"]:
    print(
        f"BAS [{BAS_CONFIDENCE_SOURCE}]: UNAVAILABLE "
        f"(no usable confidence for this source; logprob-derived confidence is "
        f"None until a logprob-exposing route is wired up)."
    )
else:
    print(f"BAS confidence source : {_bas_rep['confidence_source']}")
    print(f"BAS epsilon (clip)    : {_bas_rep['epsilon']:g}")
    print(f"Overall BAS           : {_bas_rep['bas']:.4f}   (max 1.0, higher better)")
    print(f"  usable rows         : {_bas_rep['n_usable']}")
    print(
        f"  saturated wrong rows: {_bas_rep['n_saturated']} "
        f"(confidence >= 1 - eps AND wrong; these dominate the score)"
    )
    print(f"  mean U over correct : {_bas_rep['bas_correct_mean']:.4f}")
    print(f"  mean U over wrong   : {_bas_rep['bas_incorrect_mean']:.4f}")
 
    print("\nBy condition:")
    for cond in conds:
        subset = [g for g in gens if g.condition == cond]
        b = bas_score(subset, confidence_source=BAS_CONFIDENCE_SOURCE, epsilon=BAS_EPSILON)
        print(f"  {cond:>14}: BAS = {b if b is None else f'{b:.4f}'}")
 
    # Sensitivity check: BAS at a coarser clip, so reviewers see the dependence.
    _bas_coarse = bas_score(gens, confidence_source=BAS_CONFIDENCE_SOURCE, epsilon=1e-3)
    print(
        f"\nSensitivity: BAS at eps=1e-3 = "
        f"{_bas_coarse if _bas_coarse is None else f'{_bas_coarse:.4f}'} "
        f"(vs {_bas_rep['bas']:.4f} at eps={BAS_EPSILON:g}). Always report eps."
    )
 
 
# %% [markdown]
# ## Metric 7: Disagreement rate across samples
#
# For each (question_id, condition) group with k>=2 valid samples, compute the
# proportion of samples that differ from the modal answer, then average across
# groups. This is the non-verbal uncertainty signal that should survive under
# induced overconfidence.
#
# Ideal pattern: disagreement rate stays roughly constant across conditions,
# since it's a structural property of the questions rather than the prompt.
 
# %%
for cond in conds:
    subset = [g for g in gens if g.condition == cond]
    dr = disagreement_rate(subset)
    print(f"  {cond:>14}: disagreement rate = {dr if dr is None else f'{dr:.3f}'}")
 
 
# %% [markdown]
# ## Metric 8: Unique answers per question (fallback when grading is not done)
#
# Before grading finishes, you can still see which questions are "uncertain"
# by counting how many distinct answers the model produced across samples.
# High unique-answer counts = model is unsure, regardless of which answer is
# right. This is the table to look at first when the run finishes but grading
# hasn't been run yet.
 
# %%
ua = unique_answers_per_question(gens)
print(f"Total (question, condition) groups: {len(ua)}\n")
 
for cond in conds:
    counts = Counter(r["n_unique"] for r in ua if r["condition"] == cond)
    parts = ", ".join(f"{k} unique: {v} groups" for k, v in sorted(counts.items()))
    print(f"  {cond:>14}: {parts}")
 
most_uncertain = sorted(ua, key=lambda r: r["n_unique"], reverse=True)[:10]
print("\nTop 10 most-uncertain (question, condition) pairs:")
for r in most_uncertain:
    print(
        f"  {r['question_id']:<22} {r['condition']:<14} "
        f"n_unique={r['n_unique']}  modal={r['modal_answer']!r:<12} "
        f"({r['modal_count']}/{r['n_samples']})"
    )
 
 
# %% [markdown]
# ## Headline summary tables
#
# Everything above, collected into the breakdown tables that will eventually
# go in the paper.
 
# %%
overall_table = format_summary(
    summarize(gens, bas_confidence_source=BAS_CONFIDENCE_SOURCE, bas_epsilon=BAS_EPSILON)
)
by_cond_table = format_summary(
    summarize(gens, group_by=["condition"],
              bas_confidence_source=BAS_CONFIDENCE_SOURCE, bas_epsilon=BAS_EPSILON)
)
by_cond_ds_table = format_summary(
    summarize(gens, group_by=["condition", "dataset"],
              bas_confidence_source=BAS_CONFIDENCE_SOURCE, bas_epsilon=BAS_EPSILON)
)
 
print("=== Overall ===")
print(overall_table)
print("\n=== By condition ===")
print(by_cond_table)
print("\n=== By condition x dataset ===")
print(by_cond_ds_table)
 
# Write the headline tables to a text file so the run produces a durable
# artifact you can push to GitHub and reference at the check-in.
# %% [markdown]
# ## Per-question aggregated view (resolves Tommy's count question)
#
# Tommy flagged that each condition was showing ~750 rows when he expected
# ~250. The 750 is correct at the row level: 250 questions x 3 samples per
# question. For a per-question view (one row per (question, condition) pair)
# we collapse the samples by majority-vote accuracy and mean confidence.
#
# This view has exactly 250 rows per condition, matching Tommy's expectation.
 
# %%
from collections import defaultdict as _defaultdict
 
def per_question_view(gens_list):
    """Collapse samples into one record per (question_id, condition)."""
    groups = _defaultdict(list)
    for g in gens_list:
        groups[(g.question_id, g.condition)].append(g)
 
    records = []
    for (qid, cond), gs in groups.items():
        confs = [g.confidence for g in gs if g.confidence is not None]
        corrects = [g.correct for g in gs if g.correct is not None]
        # Majority-vote correctness across the samples for this question.
        if corrects:
            n_right = sum(1 for c in corrects if c)
            agg_correct = n_right > len(corrects) / 2
        else:
            agg_correct = None
        records.append({
            "question_id": qid,
            "condition": cond,
            "dataset": gs[0].dataset,
            "n_samples": len(gs),
            "mean_confidence": float(np.mean(confs)) if confs else float("nan"),
            "correct_majority": agg_correct,
        })
    return records
 
pq = per_question_view(gens)
print(f"Per-question view: {len(pq)} (question, condition) rows total")
for cond in conds:
    in_cond = [r for r in pq if r["condition"] == cond]
    acc = (
        np.mean([int(r["correct_majority"]) for r in in_cond if r["correct_majority"] is not None])
        if any(r["correct_majority"] is not None for r in in_cond) else None
    )
    mean_c = float(np.mean([r["mean_confidence"] for r in in_cond if not np.isnan(r["mean_confidence"])]))
    print(f"  {cond:>14}: {len(in_cond)} questions  mean_conf={mean_c:.3f}  acc(majority)={acc if acc is None else f'{acc:.3f}'}")
 
 
# %% [markdown]
# ## Persist artifacts under fig_tabs/
 
# %%
summary_path = ARTIFACTS_DIR / "ar_pilot_summary.txt"
with summary_path.open("w") as f:
    f.write(f"AR pilot analysis summary\n")
    f.write(f"Input: {RESULTS_PATH}\n")
    f.write(f"Rows loaded: {len(gens)}\n")
    f.write(f"Unique questions: {n_q}\n")
    f.write(f"Per condition: {n_q} questions x {n_samples} samples = {n_q * n_samples} rows\n")
    f.write(f"Conditions: {conds}\n")
    f.write(f"Datasets: {sorted({g.dataset for g in gens})}\n")
    f.write(f"Graded rows: {sum(1 for g in gens if g.correct is not None)} / {len(gens)}\n\n")
    f.write("=== Overall ===\n")
    f.write(overall_table + "\n\n")
    f.write("=== By condition (sample-level: n_rows = n_questions * n_samples) ===\n")
    f.write(by_cond_table + "\n\n")
    f.write("=== By condition x dataset ===\n")
    f.write(by_cond_ds_table + "\n\n")
    f.write("=== Per-question aggregated view (n = unique questions per condition) ===\n")
    for cond in conds:
        in_cond = [r for r in pq if r["condition"] == cond]
        acc_vals = [int(r["correct_majority"]) for r in in_cond if r["correct_majority"] is not None]
        acc = float(np.mean(acc_vals)) if acc_vals else None
        mean_c = float(np.mean([r["mean_confidence"] for r in in_cond if not np.isnan(r["mean_confidence"])]))
        acc_str = "n/a" if acc is None else f"{acc:.3f}"
        f.write(f"  {cond:>14}: {len(in_cond):>4} questions  mean_conf={mean_c:.3f}  acc(majority)={acc_str}\n")
print(f"\nSaved summary: {summary_path}")
