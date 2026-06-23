# Benchmark Specification

## Benchmark Objective

Evaluate whether verbalized confidence tracks answer correctness under controlled prompting, comparing autoregressive language models with a diffusion language model.

## Unit of Evaluation

One model generation for one `question_id`, prompt condition, and sample index. Comparative metric tables and figures use the shared AR/DLM question-ID intersection (250 questions).

## Datasets

GSM8K, MedQA, SimpleQA, TriviaQA, TruthfulQA

## Model Identifiers

google/gemini-2.5-flash, inception/mercury-2, openai/gpt-4.1-mini, x-ai/grok-4.3

## Prompt Conditions

cautious, neutral, overconfident

## Correctness Grader

Correctness labels come from `analysis/llm_as_judge/llm_as_judge.py`. Numeric and multiple-choice answers are graded deterministically; short-answer rows are graded by the configured LLM judge using the raw generation text when available. Empty, malformed, or parse-failed answers are marked incorrect.

## Metric Definitions

Accuracy is the fraction of judged generations marked correct. Mean confidence is the arithmetic mean of verbalized confidence. Expected calibration error (ECE) uses 10 equal-width confidence bins and weights each absolute bin accuracy-confidence gap by bin frequency. Brier score is the mean squared error between confidence and correctness. Area under the receiver operating characteristic curve (AUROC) is the Mann-Whitney probability that a correct generation receives higher confidence than an incorrect generation, with half credit for ties. High-confidence wrong rate is the fraction of all evaluated generations that are incorrect with confidence >= 0.90. Parse success is parsed rows divided by raw rows for each model.
