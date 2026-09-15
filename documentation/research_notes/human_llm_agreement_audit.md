# Human--LLM Judge Agreement Audit

This note documents the 200-row human audit used to validate the LLM-as-judge
correctness labels.

## Files

- `analysis/human_llm_check/raw_200_sample.jsonl`: frozen 200-row sample.
- `analysis/human_llm_check/perfect_audit_sheet.csv`: completed audit sheet with
  human labels and judge labels.
- `analysis/human_llm_check/calculate_kappa.py`: agreement calculation.
- `analysis/human_llm_check/audit_results.txt`: generated agreement summary.
- `analysis/human_llm_check/generate_audit.py`: deterministic sampler for future
  audits.

## Current Agreement Result

Measured against the judge labels stored with the judged generations, which
are the labels every paper metric uses, the human and LLM judge agree on 195 of
200 rows: observed agreement 0.9750, Cohen's kappa 0.9480, expected chance
agreement 0.5190, Wilson 95% interval [0.9428, 0.9893]. Reproduce with
`analysis/human_llm_check/agreement_against_saved_labels.py`, which writes
`audit_results_saved_labels.txt`.

The earlier figure of 197/200 (kappa 0.9688) came from `calculate_kappa.py`,
which reads the judge column of `perfect_audit_sheet.csv`. In two rows that
column does not match the stored judge label and matches the human label
instead:

- `MedQA_0089`, Dream, overconfident sample 0: stored judge correct, sheet
  judge incorrect, human incorrect. The sheet reason is the generic
  deterministic multiple-choice comparison.
- `TruthfulQA_0016`, Dream, cautious sample 1: stored judge incorrect, sheet
  judge correct, human correct. The sheet's judge reason still argues the
  answer repeats the misconception, i.e. supports the stored label.

The frozen sample `raw_200_sample.jsonl` agrees with the stored labels in both
rows. The camera-ready paper reports 195/200.

The sample also contains two generations drawn twice (rows 57 and 156:
TruthfulQA_0523, Mercury-2, overconfident sample 0; rows 137 and 186:
TruthfulQA_0642, DiffusionGemma, cautious sample 2), with identical responses
and labels. On the 198 distinct generations agreement is 193/198, kappa 0.948.

## Reproducibility Status

The completed audit sample is frozen in `raw_200_sample.jsonl`, so future paper
versions should not change the evaluated generations without checking whether
the audited rows remain in the analysis set.

The original random seed used to produce the completed sample was not recorded.
For future resampling, `generate_audit.py` now defaults to seed 42 and writes a
`sample_metadata.txt` file. The existing completed audit should be treated as a
frozen sample, not as a sample that can be exactly regenerated from the previous
script.

## Blinding Status

The final completed audit sheet contains both `judge_verdict` and
`judge_reason`, so the CSV alone does not prove that the human labels were
entered blind to AI grades. The human grader reported that the grading was
intended to be blind, with a small number of edge-case checks. In the paper, use
this audit as a strong robustness check for judge agreement, but avoid implying
more procedural certainty than the artifact supports.

## Disagreements

Against the stored labels there are five disagreement rows, listed in
`audit_results_saved_labels.txt`:

- `MedQA_0089`, Dream: judge correct, human incorrect.
- `TruthfulQA_0649`, Gemini Flash: judge correct, human incorrect.
- `TruthfulQA_0016`, Dream: judge incorrect, human correct.
- `TruthfulQA_0312`, Gemini Flash: judge incorrect, human correct.
- `MedQA_0498`, Dream: judge correct, human incorrect.

These rows are useful examples if the appendix needs concrete judge-disagreement
cases.
