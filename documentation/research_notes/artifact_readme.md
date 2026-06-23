# Review Artifact

This artifact contains saved AR and DLM generations, split LLM-as-judge results, benchmark specifications, metric tables, figures, captions, schema documentation, and audit templates for the verbalized confidence calibration benchmark.

## Reproduction

From the repository root, run:

```bash
python3 analysis/generate_paper_assets.py
python3 analysis/check_review_artifacts.py
```

Generation outputs are split by model under `ar_models/model_outputs/` and `dlm_models/model_outputs/`. Judge results are split under `analysis/llm_as_judge/results/` by source, model, and dataset.
