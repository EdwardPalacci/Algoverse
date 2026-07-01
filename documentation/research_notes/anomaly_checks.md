# Anomaly Checks

These checks use saved LLM-as-judge rows and parsed confidence values from the fixed 250-question evaluation. Repeated generations are summarized as diagnostic evidence, not as independent question-level evidence.

## Dream Parse Failures

Dream has 99/2250 (4.4%) rows with `source_parse_success == False`. The failures are concentrated by dataset as follows: TriviaQA=36; SimpleQA=30; TruthfulQA=21; GSM8K=9; MedQA=3. By prompt condition: neutral=36; overconfident=33; cautious=30. The judge pipeline retained these rows rather than silently dropping them; 12 were graded correct and 87 were graded incorrect after deterministic or judge-based recovery.

## Dream Zero-Confidence Spike

Dream has 348/2250 (15.5%) zero-confidence rows. This is not a parser-only artifact: the raw responses include explicit `"confidence": 0.0` fields. The spike appears mostly under cautious and neutral prompting: cautious=219; neutral=111; overconfident=18. By dataset: TruthfulQA=108; MedQA=93; SimpleQA=81; TriviaQA=51; GSM8K=15.

## DiffusionGemma and LLaDA High-Confidence Saturation

DiffusionGemma has 2243/2250 (99.7%) rows with confidence >= 0.90; 1859/2250 (82.6%) are exactly 1.0. Its high-confidence wrong count is 927/2250 (41.2%).
LLaDA has 2223/2250 (98.8%) rows with confidence >= 0.90; 1707/2250 (75.9%) are exactly 1.0. Its high-confidence wrong count is 1574/2250 (70.0%).

## Answer/Explanation Disagreement Candidates

The heuristic audit flags 343 candidate answer/explanation disagreements. These are candidates, not final semantic labels. The current heuristic is conservative for multiple choice and numeric-only for arithmetic rows; most candidates are GSM8K rows where the answer field and the last number in the short explanation differ. Counts by model: DiffusionGemma=131; Dream=93; LLaDA=81; Mercury-2=38.

## Dataset-Specific Failures

SimpleQA remains the largest dataset-level failure mode. TruthfulQA also separates AR and DLM behavior. Family-level focused rows:

| Family | Dataset | N | Accuracy | Mean confidence | Parse failure rate | High-confidence wrong rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| AR | SimpleQA | 1350 | 0.279259 | 0.826296 | 0.000000 | 0.502963 |
| AR | TruthfulQA | 1350 | 0.829630 | 0.924556 | 0.000000 | 0.157037 |
| DLM | SimpleQA | 1800 | 0.073889 | 0.875846 | 0.025000 | 0.686667 |
| DLM | TruthfulQA | 1800 | 0.531111 | 0.896684 | 0.016667 | 0.368333 |

## Follow-up Notes

Dream parse failures and zero-confidence behavior are model-specific anomalies in the fixed 250-question evaluation, not reasons to discard the current data. DiffusionGemma and LLaDA require explicit high-confidence saturation reporting because confidence values near 1.0 are common even on wrong answers. SimpleQA and TruthfulQA expose the clearest dataset-specific failures and should remain prominent in follow-up evaluation.
