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
 
RESULTS_PATH = HERE / "run_model.py"

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
 
                    rows.append({
                        "question_id": qid,
                        "dataset": ds,
                        "condition": cond,
                        "sample_idx": s,
                        "model": "dummy-model",
                        "prompt": f"<question {qid}>",
                        "answer": answer,
                        "confidence": confidence,
                        "ground_truth": gt,
                        "correct": correct,
                        "raw_output": (
                            f'{{"answer": "{answer}", "confidence": {confidence}}}'
                            if not parse_failed else "<malformed model output>"
                        ),
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
print(f"Loaded {len(gens)} generations")
print(f"  unique questions:  {len({g.question_id for g in gens})}")
print(f"  conditions:        {sorted({g.condition for g in gens})}")
print(f"  datasets:          {sorted({g.dataset for g in gens})}")
print(f"  models:            {sorted({g.model for g in gens})}")
print(f"  samples/question:  {max(g.sample_idx for g in gens) + 1}")
print(f"\nFirst row:\n  {gens[0]}")
 
 
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
for cond in ["cautious", "neutral", "overconfidence"]:
    if cond in means:
        print(f"  {cond:>14}: {means[cond]:.3f}")
 
if "neutral" in means and "overconfidence" in means:
    gap = means["overconfidence"] - means["neutral"]
    status = "OK" if gap >= 0.2 else "WEAK"
    print(f"\nNeutral -> overconfidence gap: {gap:+.3f}   [{status}]")
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
conds = ["cautious", "neutral", "overconfidence"]
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
for cond in ["cautious", "neutral", "overconfidence"]:
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
print("=== Overall ===")
print(format_summary(summarize(gens)))
print("\n=== By condition ===")
print(format_summary(summarize(gens, group_by=["condition"])))
print("\n=== By condition x dataset ===")
print(format_summary(summarize(gens, group_by=["condition", "dataset"])))
