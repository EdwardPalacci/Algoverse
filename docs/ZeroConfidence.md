# Zero-Confidence Failure Analysis (AR Outputs)

## Main Finding

Confidence of 0.0 doesn't represent true uncertainty and it represented instability.

## Classifications


### Category 1: Self-Correction/Incositencies In Explanation

The model gives one answer, but the explanation contradicts or corrects it.

Pattern:

Answer: Wrong answer
Explanation: The correct answer is actually ...


This suggests that the model may know the correct answer during explanation generation, but fails to update the final answer field.

### Category 2: Generation Failure

The generation becomes unstable.

Signs include:

- unexpected language switching
- Chinese token injection
- unrelated content
- malformed or semantically broken output

These cases are not uncertainty; they are generation failures.

### Category 3: Coherent Hallucination

The model gives a wrong answer and the explanation supports that same wrong answer.

Pattern:

Answer: Wrong answer
Explanation: Wrong answer is correct


# Case Analysis

## Case 1: TriviaQA_3334 (Neutral Sample 1)

### Category

Self-Correction in Explanation

### Why

- Answer: Simon Baker
- Explanation: Hugh Laurie is correct

The answer and explanation directly contradict each other. The model appears to retrieve the correct answer after already committing to an incorrect answer. This is not uncertainty. It is an answer-explanation mismatch.


## Case 2: TriviaQA_3334 (Neutral Sample 2)

### Category

Internal Inconsistency / Self-Correction

### Why

- Answer: Simon Baker
- Explanation contains confused reasoning

The model attempts to justify the answer but the explanation itself is factually incorrect and internally unstable.

## Case 3: SimpleQA_1965 (Cautious Sample 1)

### Category

Generation Failure

### Why

- Answer contains Chinese
- Explanation says the answer is irrelevant

The model unexpectedly switches into an unrelated Chinese political phrase. This is not a knowledge error or uncertainty. It is a generation/decoding failure.

## Case 4: TriviaQA_3334 (Cautious Sample 0)

### Category

Self-Correction in Explanation

### Why

- Answer: Simon Baker
- Explanation: Hugh Laurie is correct

The model explicitly states that its own answer is wrong.


## Case 5: TriviaQA_3334 (Cautious Sample 1)

### Category

Incosisties In Explanation

### Why

- Answer: Simon Baker
- Explanation introduces unrelated actors and roles

The model starts talking about incorrect entities in explanation. The explanation does not reliably support the answer.

## Case 6: TriviaQA_3334 (Cautious Sample 2)

### Category

Coherent Hallucination

### Why

- Answer: Simon Baker
- Explanation: Simon Baker plays House

This is the only observed zero-confidence example that behaves like a standard hallucination.

## Case 7: TriviaQA_4445

### Category

Generation Failure

### Why

The answer:

- starts with a correct year
- introduces incorrect locations
- switches into Chinese
- adds unsupported historical claims

The explanation also switches to Chinese and attempts to correct the answer. This is a generation instability failure.

## Case 8: TriviaQA_5791

### Category

### Why

- Answer: 12 October 1214
- Explanation: The correct answer is 1066

The model directly states that its own answer is incorrect and then provides the correct answer.

# Main Conclusion

The observed 0.0 confidence values should not be interpreted as calibrated uncertainty. Instead, they function as failure indicators.