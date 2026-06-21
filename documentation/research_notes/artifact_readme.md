# Review Artifact

This artifact contains saved AR and DLM generations, benchmark specifications, metric tables, figures, captions, schema documentation, and audit templates for the verbalized confidence calibration benchmark.

Comparative tables and figures are computed on the shared AR/DLM question-ID intersection. Raw-source exclusions are documented in `analysis/metrics/data_alignment_exclusions.csv`.

## Reproduction

From the repository root, run:

```bash
python3 analysis/prepare_review_artifact.py
```

The script reads `ar_models/model_outputs/ar_parsed_generations.jsonl`, `ar_models/model_outputs/ar_raw_generations.jsonl`, `dlm_models/model_outputs/dlm_parsed_generations.jsonl`, `dlm_models/model_outputs/dlm_raw_generations.jsonl`, and `data/PilotDataset.json`. It writes tables to `paper_assets/tables/`, figures to `paper_assets/figures/`, quality-control reports to `analysis/metrics/`, and release metadata to `documentation/research_notes/`.

## Raw and Parsed Outputs

AR outputs are stored in `ar_models/model_outputs/`. DLM outputs are stored in `dlm_models/model_outputs/`.

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
