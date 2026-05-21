# Prompt Schema and Experimental Conditions

## Unified Output Schema

All prompting conditions force the model to return the same JSON schema:

```json
{
  "answer": "...",
  "confidence": 0,
  "short_explanation": "..."
}
