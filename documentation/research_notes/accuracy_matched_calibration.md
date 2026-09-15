# Accuracy-Matched Calibration: What We Did and What Survived

Written up for the AIMS camera-ready. This note records the method, the checks
that were run against it, and the claims that did not survive those checks.

## The question

The reviewer objected that the paper "attributes the difference in confidence
calibrations entirely to the Diffusion/Autoregressive architecture, but the
models evaluated are of various sizes and families." That objection is correct
as stated about the aggregate numbers, and it has a specific mechanism: ECE
measures the distance between stated confidence and realised accuracy, and
every model in this evaluation states high confidence. A family that is less
accurate therefore posts a worse ECE without differing in any
architecture-linked way.

## Method

Difficulty is defined per model. For model `m` and question `q`, we take the
share of correct answers each *other* model produced on `q`, average those
shares within each family, and average the two family means.

Two details matter and both were forced by things that went wrong first:

- **Leave-one-model-out.** Without it, a model's own answers help define the
  stratum it is then scored in.
- **Family-balanced averaging.** An earlier version pooled the six reference
  models directly. With three AR and four DLM models that scores AR targets
  against 2 AR + 4 DLM and DLM targets against 3 AR + 3 DLM, so the stratifier
  meant something different for each family (mean difficulty seen by AR rows
  0.560 vs DLM rows 0.599, and bins with the same label held different
  questions, Jaccard 0.46-0.77).

Generations are binned by difficulty in steps of 0.1; cells under 150
generations are dropped. We compare only pairs of real cells whose accuracies
fall within 0.05. An earlier version interpolated a curve between cells; that
was dropped because with ten cells per family most of the interpolated range
sat over gaps where one family had no data, and the three largest "findings"
came from a single chord across a 0.31-wide void.

## Result

The aggregate ECE gap is 0.254 (DLM 0.456, AR 0.201). The comparison bounds a
residual rather than eliminating one, and the estimate is unstable.

| check | result |
| --- | --- |
| matched-pair mean ECE difference | -0.031 |
| cluster bootstrap 95% CI (400 reps over questions) | [-0.031, 0.079] — spans zero, median +0.017 |
| permutation over family labels | 24 of 35 splits at least as extreme, p = 0.69 |
| leave-one-model-out | -0.026 to +0.089; only dropping Grok preserves the sign, the other six reverse it |

Within-stratum discrimination is the same story: AUROC averaged per model then
by family is 0.581 for AR and 0.579 for DLM, while the per-model range is 0.413
(GPT-4.1 mini) to 0.816 (Grok), with Mercury-2 at 0.772. The two most
discriminative models are one from each family.

## Claims that did NOT survive, and should not be revived

An intermediate version of this analysis reported "a small residual gap
persists at matched accuracy, DLMs slightly worse." That was wrong and was
withdrawn. It failed on every check above: the bootstrap interval covers zero,
the permutation test is unremarkable, the sign depends on Grok alone, and a
within-architecture placebo (splitting models into two arbitrary groups)
produces a *larger* gap than the real AR/DLM contrast.

A pooled within-stratum AUROC of ~0.43 was also computed and discarded. Pooling
a family's models inside one stratum mixes models with different accuracy and
different confidence habits into a single ranking, which drives the statistic
below chance as a composition artefact. Per-model AUROC averaged afterwards is
0.58 for both families. Do not report the pooled figure.

## Known caveats, disclosed in the paper

- Difficulty strata correlate with dataset: SimpleQA supplies most of the
  hardest questions, MedQA most of the easiest. Matching on difficulty
  therefore partly matches on dataset.
- Seven models is few units for a family contrast: 35 label splits, seven
  leave-one-out runs. This is why the residual is reported as absent rather
  than as small.
- `ECE ~ mean_confidence - accuracy` is a good approximation but not an
  identity; it breaks in cells that are net underconfident, since ECE takes an
  absolute value (AR bin 0.8-1.0: ECE 0.026, confidence minus accuracy -0.023).

## Parse-failure robustness

Separate check, aggregate tables only. All 132 parse-failed generations (99
Dream, 33 Mercury-2, none AR) carry no confidence value, so they never entered
ECE, AURC or AUROC. Excluding them moves accuracy by at most +0.011 and changes
no ECE and no model ranking. Twelve of the 132 are graded correct, so parse
failure is not a synonym for a wrong answer. Note this check does *not* cover
the stratified analysis, which re-derives difficulty from the same generations;
the uncertainty estimates above are the relevant robustness statement there.

## Code

- `analysis/stratified_calibration.py` — difficulty strata, per-family tables
- `analysis/matched_accuracy_calibration.py` — matching, bootstrap, permutation, leave-one-out
- `analysis/parse_failure_robustness.py` — parse-failure comparison
- Outputs: `paper_assets/tables/stratified/`, `paper_assets/tables/quality_control/`

## Resolved: AURC tie handling

`aurc()` in `analysis/generate_paper_assets.py` (and its copy in
`render_figures.py`) previously sorted by confidence with a stable sort, so
ties, very common at confidence 1.0, were broken by input file order. It now
returns the expected AURC over all orderings of tied generations, and the
risk-coverage curve uses the same expected risk. Published model AURC values
moved, most for heavily tied models (DiffusionGemma 0.294 to 0.383, LLaDA 0.633
to 0.728, Dream 0.616 to 0.651); Grok and Mercury-2 were essentially unchanged.
The matched-accuracy result uses ECE and AUROC and did not change.

## Corrections made during review

Two claims written into an early draft of the paper were wrong and were fixed
before commit:

- "dropping Grok reverses the sign" — backwards. The all-models estimate is
  negative (-0.031); dropping Grok is the only run that *keeps* it negative
  (-0.026), and the other six flip it positive (+0.020 to +0.089).
- "no model changes rank" in the parse-failure check — false. Excluding parse
  failures lifts Mercury-2 from 0.720 to 0.731, passing Gemini Flash at 0.726.

Also corrected: an earlier draft said the result held "matched on difficulty
and accuracy". Those are not jointly satisfiable here. At matched *difficulty*
the DLM ECE excess is +0.19 to +0.31 with non-overlapping intervals, because
the families are not equally accurate on equally difficult questions. Only
accuracy matching closes the gap, and the matched pairs deliberately cross
difficulty bins to achieve it.
