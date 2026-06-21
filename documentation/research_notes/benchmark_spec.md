# Benchmark Specification

## Benchmark Objective

Evaluate whether verbalized confidence tracks answer correctness under controlled prompting, comparing an autoregressive language model with a diffusion language model.

## Construct

The measured construct is verbalized confidence calibration: the correspondence between a model's stated confidence probability and empirical correctness.

## Unit of Evaluation

One model generation for one `question_id`, prompt condition, and sample index.

Comparative metric tables and figures use the shared AR/DLM question-ID intersection (247 questions). Source-level exclusions are listed in `analysis/metrics/data_alignment_exclusions.csv`.

## Datasets

GSM8K, MedQA, SimpleQA, TriviaQA, TruthfulQA

## Model Identifiers

mercury-2, qwen/qwen-2.5-7b-instruct

## Model Family Labels

`AR` denotes an autoregressive language model. `DLM` denotes a diffusion language model.

## Prompt Conditions

cautious, neutral, overconfident

## Confidence Scale

Models report a verbalized confidence probability in `[0, 1]`. Values outside this interval fail quality control.

## Correctness Grader

The saved parsed outputs do not contain human or LLM-judge correctness labels. This artifact therefore uses a deterministic automatic grader: last-number matching for numeric answers, option-letter or option-text matching for multiple-choice answers, and normalized string exact/containment matching for short answers. These labels are suitable for artifact execution checks and preliminary benchmarking, but they should be replaced or audited for final claims.

## Metric Definitions

Accuracy is the fraction of automatically correct generations. Mean confidence is the arithmetic mean of verbalized confidence. Expected calibration error (ECE) uses 10 equal-width confidence bins and weights each absolute bin accuracy-confidence gap by bin frequency. Brier score is the mean squared error between confidence and correctness. Area under the receiver operating characteristic curve (AUROC) is the Mann-Whitney probability that a correct generation receives higher confidence than an incorrect generation, with half credit for ties. High-confidence wrong rate is the fraction of all evaluated generations that are incorrect with confidence >= 0.90. Parse success is parsed rows divided by raw rows for each model source.

## Quality Control Checks

The artifact checks question-ID alignment by model family, duplicate full evaluation keys, dataset validity, model identifier validity, family-label validity, prompt-condition validity, confidence range, and binary rule-based correctness labels.

## Known Limitations

DLM outputs are incomplete relative to the full AR question set, so comparative outputs use the shared question-ID intersection. Correctness labels are deterministic automatic labels rather than manual adjudications. Prompt sensitivity is therefore reported as an available-case analysis over parsed rows within the aligned question set. Some short-answer datasets require semantic grading beyond normalized string matching.
