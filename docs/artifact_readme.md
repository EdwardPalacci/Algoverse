# Review Artifact

This artifact contains saved AR and DLM generations, benchmark specifications, metric tables, figures, captions, schema documentation, and audit templates for the verbalized confidence calibration benchmark.

Comparative tables and figures are computed on the shared AR/DLM question-ID intersection. Raw-source exclusions are documented in `metrics/data_alignment_exclusions.csv`.

## Reproduction

From the repository root, run:

```bash
python3 src/prepare_review_artifact.py
```

The script reads `outputs/ar_parsed_generations.jsonl`, `outputs/ar_raw_generations.jsonl`, `dlm_outputs/dlm_parsed_generations.jsonl`, `dlm_outputs/dlm_raw_generations.jsonl`, and `PilotDataset.json`. It writes tables and figure files to `fig_tabs/`, quality-control reports to `metrics/`, and release metadata to `docs/`.

## Raw and Parsed Outputs

AR outputs are stored in `outputs/`. DLM outputs are stored in `dlm_outputs/`.

## Confidence Parsing

Each parsed generation contains `confidence`, a verbalized probability normalized to `[0, 1]`.

## Correctness Grading

The current artifact uses deterministic automatic grading because saved parsed outputs do not include adjudicated correctness labels. Numeric answers use last-number matching; multiple-choice answers use option-letter or option-text matching; short answers use normalized string matching and containment.

## Prompt Conditions

Prompt conditions are encoded in the `condition` field as `neutral`, `cautious`, or `overconfident`.

## Not Included

This package does not include API keys, private links, author-identifying metadata, or completed manual adjudication labels.

## Anonymity

The review materials are prepared without author names, institution names, private URLs, API keys, or workspace-specific absolute paths.
