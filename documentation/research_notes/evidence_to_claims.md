# Evidence-to-Claims Map

This note records the current analysis status before finalizing the paper's
conclusion. It is deliberately conservative: each claim lists supporting
evidence, weakening evidence, and what would change the interpretation.

## Claim 1: AR models are better calibrated in aggregate, but this is not a clean architecture-only result.

**Cognitive status:** Moderate. The aggregate evidence is consistent, but the
architecture interpretation is weakened by model capability and serving
differences.

**Supporting evidence.**

- Table 2: the three AR models have higher family-level accuracy and lower ECE
  than the four DLMs in aggregate. Grok is the strongest AR model with accuracy
  0.828, ECE 0.071, AURC 0.039, and AUROC 0.894.
- Figure 3: AR reliability curves are closer to the diagonal than DLM curves
  across prompt conditions, especially in high-confidence regions.
- Figure 8 (updated 2026-09-15, tie-averaged risk): AR risk is 0.274 at 5%
  coverage, 0.246 at 50% and 0.283 at full coverage; DLM risk is 0.564 at both 5%
  and 50%, above its 0.513 full-coverage risk. The family gap mostly reflects
  accuracy rather than a more useful confidence ranking. (Earlier values of
  0.337/0.258 vs 0.560/0.617 broke confidence ties by file order.)
- Figure 9/Table 3: AR has higher family-level accuracy on every dataset in the
  250-question evaluation.

**Weakening evidence.**

- The strongest DLM, Mercury-2, is competitive with strong AR models on ranking
  metrics: accuracy 0.720, ECE 0.192, AURC 0.077, and AUROC 0.903.
- The AR side uses strong frontier models, while three DLMs are locally served
  with model-specific wrappers. This weakens a pure architecture explanation.
- The DLM family result is strongly affected by Dream and LLaDA, which have low
  accuracy and severe overconfidence. A larger or different DLM set could change
  the family-level conclusion.

**Current conclusion.**

The paper should claim an aggregate behavioral difference under the shared
protocol, not that AR architecture itself causes better calibration. The safest
wording is: "AR models are better calibrated in aggregate in this evaluation,
but the evidence supports a behavioral comparison rather than an architecture
causal claim."

**Would strengthen.**

- Matched-capability AR/DLM pairs.
- More DLMs run through comparable serving interfaces.
- Controls for task difficulty and answer type.

**Would weaken.**

- Additional DLMs with Mercury-2-like calibration.
- AR baselines closer in scale/capability to the tested local DLMs.

## Claim 2: Prompt pressure changes confidence more than it changes accuracy.

> Estimator note (2026-09-15): the numbers below were equal-weight means of
> per-model metrics with ties broken by file order. The paper reports metrics
> pooled over each family's generations with AURC averaged over orderings of
> tied confidences, and `prompt_condition_family_metrics.csv` is now generated
> with that estimator by `analysis/prompt_condition_family_metrics.py`. The
> "0.399 vs 0.409" bullet below no longer holds under that estimator. Pooled, AURC rises from cautious
> to overconfident for both families (AR 0.243 to 0.282, DLM 0.507 to 0.546), and
> the DLM value is flat between neutral (0.545) and overconfident (0.546).

**Cognitive status:** Strong for the current prompt interventions.

**Supporting evidence.**

- Prompt-condition metrics show AR accuracy is nearly flat across prompts:
  0.719 cautious, 0.718 neutral, 0.715 overconfident.
- DLM accuracy is also nearly flat: 0.480 cautious, 0.477 neutral, 0.488
  overconfident.
- Mean confidence rises with overconfident prompting: AR rises from 0.895
  cautious to 0.946 overconfident, and DLM rises from 0.868 cautious to 0.974
  overconfident.
- Calibration worsens as confidence pressure increases. AR ECE rises from 0.210
  cautious to 0.246 overconfident; DLM ECE rises from 0.423 to 0.483. HCWR also
  rises: AR 0.211 to 0.237, DLM 0.373 to 0.462.

**Weakening evidence.**

- DLM AURC is slightly lower under overconfident prompting than under cautious
  prompting in the family-level table (0.399 vs. 0.409), so the selective
  prediction story is not perfectly monotonic for every metric.
- Prompt effects may be mediated by model-specific behavior rather than prompt
  category alone.

**Current conclusion.**

The prompts act mainly as confidence-pressure interventions. They shift reported
confidence and calibration errors more clearly than they shift answer accuracy.

**Would strengthen.**

- A regression or mixed-effects model predicting accuracy/confidence/ECE from
  prompt condition, model, dataset, and answer type.
- A plot or table of accuracy and confidence by prompt condition.

**Would weaken.**

- Evidence that prompt wording changes the underlying answer distribution enough
  to explain calibration shifts as accuracy changes rather than confidence
  shifts.

## Claim 3: Answer--explanation mismatch candidates are not primarily a non-neutral prompting phenomenon.

**Cognitive status:** Moderate. The current mismatch detector is heuristic, so
the count is useful for triage but not a final semantic label.

**Supporting evidence.**

- The quality-control mismatch table contains 343 DLM candidates, all GSM8K.
  Correction (2026-09-15): the heuristic was originally run on DLM rows only.
  Run on AR rows it flags 263 (3.9% of AR generations vs 3.8% for DLMs), also
  all GSM8K, so mismatch candidates are not a DLM-specific phenomenon; see
  `answer_explanation_candidates_by_family.csv`.
- Candidate counts by prompt are similar after accounting for the fact that
  there are two non-neutral conditions: 108 cautious, 109 neutral, and 126
  overconfident.
- Candidate rates over DLM generations are 3.60% cautious, 3.63% neutral, and
  4.20% overconfident. Overconfident prompting is slightly higher, but the
  pattern is not concentrated enough to claim that most mismatch behavior is a
  non-neutral prompt artifact.

**Weakening evidence.**

- The heuristic focuses on GSM8K-style answer/final-number disagreement and may
  miss semantic mismatch in short-answer datasets.
- The existing table is a candidate list, not manually adjudicated semantic
  disagreement.

**Current conclusion.**

The paper can say that mismatch candidates are mostly a DLM/GSM8K arithmetic
consistency issue in the current heuristic audit, with only a mild increase
under overconfident prompting. It should not claim that non-neutral prompting is
the main cause.

**Would strengthen.**

- Manual adjudication of a stratified sample of mismatch candidates.
- A semantic mismatch detector that covers short-answer and multiple-choice rows.

**Would weaken.**

- Manual review showing that many candidates are harmless formatting artifacts
  rather than answer/explanation disagreement.

## Claim 4: Dataset composition strongly changes the AR--DLM comparison.

**Cognitive status:** Strong within the fixed 250-question evaluation.

**Supporting evidence.**

- Figure 9 and Table 3 show large dataset variation. SimpleQA is the hardest
  dataset for both families: AR accuracy 0.279 and DLM accuracy 0.074, with ECE
  0.548 for AR and 0.799 for DLM.
- TriviaQA has the largest family-level accuracy gap: AR 0.918 vs. DLM 0.482.
- GSM8K is much closer: AR 0.711 vs. DLM 0.664. Mercury-2 and DiffusionGemma
  perform well on GSM8K, which reduces the family gap.

**Weakening evidence.**

- Each dataset has only 50 questions, so dataset-level conclusions need larger
  samples before they become stable.
- Some dataset effects may reflect answer type, not dataset identity.

**Current conclusion.**

The paper should not report one architecture-level number without dataset
context. Dataset composition is part of the observed confidence behavior.

**Would strengthen.**

- More questions per dataset.
- Stratification by answer type and difficulty.

**Would weaken.**

- Larger samples showing that the current SimpleQA/TriviaQA gaps shrink or
  reverse.

## Claim 5: Human audit supports the judge labels, but it does not remove all grading uncertainty.

**Cognitive status:** Strong for aggregate judge reliability; moderate for
ambiguous individual short-answer cases.

**Supporting evidence.**

- The 200-row human audit has 195/200 agreement between the human grader and the
  stored LLM judge labels, observed agreement 0.9750, and Cohen's kappa 0.9480.
  (An earlier 197/200 figure used an audit-sheet judge column that differs from
  the stored labels in two rows; see human_llm_agreement_audit.md.)
- The audit sample includes all seven models, all five datasets, and all three
  prompt conditions.

**Weakening evidence.**

- The original random seed for the completed sample was not recorded, though the
  exact sampled rows are frozen in `raw_200_sample.jsonl`.
- The completed CSV contains judge labels and judge reasons, so the artifact
  itself does not prove full blinding. The human grader reported that grading
  was intended to be blind, with a small number of edge-case checks.

**Current conclusion.**

The judge labels are credible for aggregate analysis, but qualitative examples
and ambiguous short-answer cases should still be described as candidates unless
manually adjudicated.

**Would strengthen.**

- A fresh blinded audit generated from `blind_audit_sheet.csv` with recorded
  seed metadata.
- Additional adjudication of the three disagreement rows and failure cases.

**Would weaken.**

- A larger audit finding lower agreement on short-answer or TruthfulQA rows.
