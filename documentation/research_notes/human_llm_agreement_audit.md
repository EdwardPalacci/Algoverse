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

The completed audit contains 200 rows sampled from the saved judged generations.
The human and LLM judge labels agree on 197 of 200 rows, giving observed
agreement of 0.9850. Cohen's kappa is 0.9688, with expected chance agreement
0.5190. The Wilson 95% confidence interval for observed agreement is
[0.9568, 0.9949].

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

The three disagreement rows are listed in `audit_results.txt`:

- `TruthfulQA_0649`, Gemini Flash: judge correct, human incorrect.
- `TruthfulQA_0312`, Gemini Flash: judge incorrect, human correct.
- `MedQA_0498`, Dream: judge correct, human incorrect.

These rows are useful examples if the appendix needs concrete judge-disagreement
cases.
