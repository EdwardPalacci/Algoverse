# AR Metrics Pipeline Status

## Current AR Output Status

The full AR pilot run has been completed using `qwen/qwen-2.5-7b-instruct` through OpenRouter.

- Raw generations: 2250
- Parsed generations: 2248
- Parse errors: 2
- Dataset: `PilotDataset.json`
- Conditions: neutral, cautious, overconfident
- Samples per question-condition pair: 3

## Metrics That Can Be Calculated Now

These metrics can be computed directly from `outputs/ar_parsed_generations.jsonl`:

- Parse success
- Mean confidence
- Confidence histograms
- Disagreement rate across samples
- Unique answers per question

These only require fields already present in the parsed output:

- `answer`
- `confidence`
- `condition`
- `question_id`
- `parse_success`

## Metrics Currently Blocked

These metrics are currently blocked or return `None` / `N/A`:

- Accuracy
- Expected Calibration Error (ECE)
- Confidence AUROC
- High-confidence wrong rate
- Reliability diagrams

## Reason for Blocker

The current parsed AR outputs do not yet contain a `correct` field.

The metrics above require knowing whether each generation is correct:

```json
"correct": true
```

or

```json
"correct": false
```

Without this field, the metrics pipeline cannot compare confidence against actual correctness.

## Temporary Fix

Add a placeholder grading step that performs exact-match grading between the model answer and the ground-truth answer.

This will allow the metrics pipeline to run end-to-end and produce preliminary figures/tables.

## Limitations of Exact-Match Grading

Exact match is only a temporary placeholder.

It may be too strict for:

- Short-answer factual questions
- TruthfulQA-style questions with multiple acceptable answers
- Answers with equivalent wording but different phrasing
- Numeric answers with formatting differences

## Future Fix

Replace exact-match grading with either:

- Manual grading for ambiguous cases
- LLM-as-a-judge grading with a standardized rubric
- Dataset-specific grading rules for GSM8K, TruthfulQA, TriviaQA, MedQA, and other subsets

## Recommendation

Use exact-match grading now so Ryan can generate preliminary metrics by the next check-in. Later, swap in a better grading method without changing the rest of the metrics pipeline.
