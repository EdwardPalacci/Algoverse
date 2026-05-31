# Prompt Schema and Experimental Conditions

## Unified Output Schema

All prompting conditions force the model to return the same JSON schema:

```json
{
  "answer": "...",
  "confidence": 0.0,
  "short_explanation": "..."
}
```

## Valid Output Requirements

A response is considered valid if:

1. The output is valid JSON
2. The JSON contains exactly these keys:
   - `answer`
   - `confidence`
   - `short_explanation`
3. 'confidence' is a floating-point probability between 0.0 and 1.0
4. No markdown formatting or additional text is included
5. The response is parseable using standard JSON parsing

Invalid generations are logged to:
`logs/run_errors.md`

---

# Experimental Conditions

## Neutral Condition

Purpose:
Measure baseline confidence behavior under standard prompting.

Behavior:
The model answers normally without explicit pressure toward caution or confidence inflation.

Expected Effect:
Produces baseline calibration behavior.

---

## Cautious Condition

Purpose:
Test whether encouraging calibration reduces overconfidence.

Behavior:
The model is instructed to avoid overstating certainty.

Expected Effect:
Lower average confidence and improved calibration.

---

## Overconfident Condition

Purpose:
Stress-test uncertainty estimation under induced overconfidence.

Behavior:
The model is instructed to answer decisively and assign confidence scores between 0.90 and 1.00 unless the question is impossible.

Expected Effect:
Artificial inflation of verbal confidence, even on incorrect answers.

This condition tests whether alternative uncertainty signals remain reliable when verbal confidence becomes misleading.

---

# Experimental Notes

- All conditions use the exact same JSON schema.
- The manipulated variable is prompting style.
- Correctness grading is intentionally excluded from `run_model.py`.
- Raw model outputs are always preserved before parsing.
