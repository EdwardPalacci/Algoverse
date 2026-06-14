# LLM-as-a-Judge Pilot

## Goal

This pilot tests whether LLM-as-a-judge or semantic grading is more appropriate than exact-match grading for AR model outputs.

The current exact-match placeholder is useful for unblocking metrics, but it is too strict for short-answer datasets where a model answer can be semantically correct without matching the reference string exactly.

## Judge Rubric

For each example, judge whether the model answer should be considered correct given the question and ground-truth answer(s).

The judge should ignore:

- capitalization differences
- punctuation differences
- minor wording differences
- reasonable abbreviations
- answers that are shorter than the reference but semantically equivalent

The judge should mark an answer incorrect if:

- it repeats a misconception
- it contradicts the ground truth
- it gives the wrong entity, number, or claim
- it only partially answers the question in a misleading way

## Proposed LLM Judge Prompt

```text
You are grading a question-answering system.

Question:
[QUESTION]

Ground-truth answer(s):
[GROUND TRUTH]

Model answer:
[MODEL ANSWER]

Determine whether the model answer should be considered correct.

Ignore differences in wording, capitalization, punctuation, and reasonable abbreviations.

Return only valid JSON:

{
  "correct": true,
  "reason": "short explanation"
}

or

{
  "correct": false,
  "reason": "short explanation"
}
