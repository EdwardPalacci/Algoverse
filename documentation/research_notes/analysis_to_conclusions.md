# Analysis-to-Conclusions Draft

This is the current working conclusion set for team review. The goal is not to
lock the final wording, but to make the evidence-to-conclusion links visible
enough for collaborators to attack or refine them.

## Working Conclusion 1

**Claim.** AR models are better calibrated in aggregate across the fixed
250-question evaluation, but the comparison should be described as behavioral
rather than architecture-causal.

**Cognitive status.** Moderate.

**Evidence.** AR has lower aggregate ECE and better risk-coverage behavior than
DLM in the family-level figures. Grok is the strongest AR model, while Dream and
LLaDA substantially weaken the DLM family average. However, Mercury-2 is
competitive with strong AR models on confidence ranking, with AURC 0.076 and
AUROC 0.903.

**Main caveat.** The AR and DLM model sets are not matched for capability or
serving interface. The paper can claim a behavioral comparison under a shared
evaluation protocol, not proof that AR architecture itself causes better
calibration.

**What would change this conclusion.** More DLMs with Mercury-2-like behavior
would weaken the family-level claim. Matched-capability AR/DLM pairs would
strengthen architecture-level interpretation.

## Working Conclusion 2

**Claim.** Prompt pressure changes reported confidence and calibration more than
it changes answer accuracy.

**Cognitive status.** Strong.

**Evidence.** AR accuracy is nearly unchanged across cautious, neutral, and
overconfident prompting: 0.719, 0.718, and 0.715. DLM accuracy is also nearly
flat: 0.480, 0.477, and 0.488. In contrast, mean confidence rises from cautious
to overconfident prompting for both families, and ECE rises from 0.210 to 0.246
for AR and from 0.423 to 0.483 for DLM. HCWR also rises for both families.

**Main caveat.** Family-level AURC is not perfectly monotonic for DLM, so the
strongest version of the claim should focus on confidence, ECE, and high-
confidence wrong rate rather than every calibration diagnostic.

**What would change this conclusion.** A model-level regression showing that
prompt condition predicts accuracy more strongly than confidence would weaken
the claim. A controlled prompt ablation would strengthen it.

## Working Conclusion 3

**Claim.** Dataset composition is central to the AR--DLM comparison.

**Cognitive status.** Strong within the current evaluation, limited by sample
size.

**Evidence.** The AR--DLM gap is small on GSM8K, where AR accuracy is 0.711 and
DLM accuracy is 0.664. It is much larger on TriviaQA, where AR is 0.918 and DLM
is 0.482. SimpleQA is difficult for both families and especially for DLMs: AR
accuracy is 0.279 and DLM accuracy is 0.074.

**Main caveat.** Each dataset contributes only 50 questions. Some effects may be
answer-type or difficulty effects rather than dataset identity.

**What would change this conclusion.** Larger per-dataset samples or explicit
difficulty controls could change how much of the effect is attributed to dataset
composition.

## Working Conclusion 4

**Claim.** The answer--explanation mismatch candidates do not currently support
the claim that non-neutral prompts are the main cause.

**Cognitive status.** Moderate.

**Evidence.** The current heuristic flags 343 mismatch candidates, all in DLM
GSM8K rows. Candidate rates are 3.60% cautious, 3.63% neutral, and 4.20%
overconfident over DLM generations. Overconfident prompting is slightly higher,
but the distribution is not concentrated enough to claim that most mismatches
come from non-neutral prompting.

**Main caveat.** The detector is heuristic and mostly numeric/GSM8K-oriented.
Manual adjudication is needed before treating these as semantic
answer--reasoning disagreements.

**What would change this conclusion.** A manual audit showing true semantic
mismatch concentrated in overconfident rows would strengthen a prompt-effect
claim.

## Suggested Paper-Level Story

The safest story is:

> This evaluation shows that verbalized confidence is a behavioral signal whose
> usefulness varies by model, prompt pressure, and dataset. AR models are better
> calibrated in aggregate here, but Mercury-2's competitive performance and the
> capability mismatch between model sets prevent a simple architecture-causal
> conclusion. Prompt pressure primarily shifts confidence and calibration, not
> accuracy. Dataset composition and model-specific failure modes explain much of
> the observed AR--DLM gap.

This is stronger than "confidence is not architecture-invariant" because it says
what the observed pattern is, what it does not prove, and why.
