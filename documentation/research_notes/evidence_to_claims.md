# Evidence-to-Claims Map

This note turns the saved tables and figures into reviewable claims. It is meant
to support team discussion before the final conclusion is frozen.

## Claim 1: Verbalized confidence is not architecture-invariant.

**Cognitive status:** Strong for this 250-question evaluation, but not a broad
law about all AR and DLM systems.

**Evidence.**

- Table 2 shows large model-level differences within each family. Grok has the
  strongest AR calibration and ranking signal: accuracy 0.828, ECE 0.071, AURC
  0.039, AUROC 0.894, and HCWR 0.020. Mercury-2 is the strongest DLM: accuracy
  0.720, ECE 0.192, AURC 0.076, AUROC 0.903, and HCWR 0.113.
- The other DLMs behave differently from Mercury-2. Dream has accuracy 0.328,
  ECE 0.553, and AURC 0.616; LLaDA has accuracy 0.294, ECE 0.681, and AURC
  0.633. DiffusionGemma is more accurate than Dream and LLaDA, but its mean
  confidence is 0.982 and its ECE is 0.397.
- Figure 8 shows that the DLM family risk-coverage curve is much worse than the
  AR family curve, but Table 2 shows that this family result is driven by
  model-level heterogeneity rather than by Mercury-2.

**What would weaken the claim.**

- Additional DLMs with calibration close to Mercury-2 would weaken any claim
  that DLMs are generally worse calibrated.
- A matched-strength AR/DLM comparison could show that some differences are due
  to model capability rather than generation architecture.

## Claim 2: The main comparison is behavioral, not a clean architecture-only test.

**Cognitive status:** High. The experimental design fixes prompts, datasets,
schema, and metrics, but it cannot match model strength or internal uncertainty
interfaces.

**Evidence.**

- The AR side contains strong frontier models. The DLM side contains one
  OpenRouter-hosted DLM and three locally served DLMs with model-specific
  wrappers. This weakens a pure architecture causal interpretation.
- The paper does not use AR token logprobs or DLM diffusion trajectories. It
  compares user-visible verbal confidence under a shared output schema.
- The strongest DLM, Mercury-2, is competitive with AR systems on AURC and AUROC,
  while Dream and LLaDA are not. That pattern argues against treating the family
  label alone as the explanation.

**What would strengthen the claim.**

- Matched-size or matched-capability AR/DLM pairs.
- Internal DLM trajectory metrics and comparable AR uncertainty baselines.
- More DLMs served through the same provider/interface.

## Claim 3: Prompt pressure changes confidence, but higher confidence does not mean better calibration.

**Cognitive status:** Strong for the observed prompt interventions.

**Evidence.**

- Figure 7 shows ECE increases from cautious to overconfident prompting for both
  families: AR rises from 0.178 to 0.232, and DLM rises from 0.423 to 0.482.
- Figure 7 also shows AURC worsens from cautious to overconfident prompting: AR
  rises from 0.247 to 0.286, and DLM rises from 0.541 to 0.586.
- Table 4 shows mean confidence rises under overconfident prompting for most
  models, but accuracy does not rise in the same way. For example, GPT-4.1 mini
  becomes more confident from cautious to overconfident prompting while accuracy
  slightly decreases.

**What would weaken the claim.**

- Prompt variants that reliably improve both confidence and calibration.
- Human-written prompts that disentangle confidence pressure from answer style.

## Claim 4: Dataset composition is a major driver of observed calibration.

**Cognitive status:** Strong within the fixed 250-question evaluation.

**Evidence.**

- Figure 9 and Table 3 show large dataset variation. SimpleQA is the hardest
  setting for both families: AR accuracy 0.279 and DLM accuracy 0.074, with ECE
  0.548 for AR and 0.799 for DLM.
- TriviaQA separates the families most strongly: AR accuracy 0.918 compared
  with DLM accuracy 0.482.
- GSM8K is much closer at the family level: AR accuracy 0.711 and DLM accuracy
  0.664. This is partly because Mercury-2 and DiffusionGemma perform well on
  GSM8K.

**What would weaken the claim.**

- A larger sample that reduces dataset-level uncertainty and changes the
  ordering of datasets.
- Dataset-specific prompt tuning that removes the SimpleQA and TriviaQA gaps.

## Claim 5: The LLM-as-judge labels are supported by a strong human audit, but the audit should be reported carefully.

**Cognitive status:** Strong for agreement on the frozen 200-row audit sample;
moderate for broad claims about all ambiguous answer cases.

**Evidence.**

- The 200-row audit has 197/200 agreement between the human grader and the LLM
  judge, observed agreement 0.9850, and Cohen's kappa 0.9688.
- The audit includes all seven models, all five datasets, and all three prompt
  conditions.

**Caveats.**

- The original seed for the completed sample was not recorded, although the
  exact sampled rows are frozen in `raw_200_sample.jsonl`.
- The completed CSV contains judge labels and judge reasons, so the artifact
  itself does not prove full blinding. The human grader reported that grading was
  intended to be blind, with a small number of edge-case checks.

## Claim 6: Failure cases reveal model-specific confidence pathologies.

**Cognitive status:** Moderate. Counts are automated and useful, but semantic
interpretation of some failure categories needs manual review.

**Evidence.**

- Dream has 99 parse-failed rows and 348 zero-confidence generations.
- DiffusionGemma and LLaDA are saturated near confidence 1.0. DiffusionGemma has
  1,859 exact-confidence-1.0 generations, and LLaDA has 1,707.
- The heuristic answer/explanation audit flags 343 candidate inconsistencies.
  These are candidates, not final semantic labels.

**What would strengthen the claim.**

- Manual adjudication of the answer/explanation mismatch candidates.
- A small qualitative appendix table separating parser artifacts, real
  reasoning-answer disagreement, and harmless formatting mismatches.
