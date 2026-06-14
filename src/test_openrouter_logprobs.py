#!/usr/bin/env python3

import json
import os
from pathlib import Path

from openai import OpenAI


MODEL_NAME = "qwen/qwen-2.5-7b-instruct"
OUTPUT_PATH = Path("outputs/ar_logprobs_test_raw.jsonl")


QUESTIONS = [
    ("logprob_test_001", "What is 2 + 2?", "4", "numeric"),
    ("logprob_test_002", "What is the capital of France?", "Paris", "short_answer"),
    ("logprob_test_003", "Who wrote Hamlet?", "William Shakespeare", "short_answer"),
    ("logprob_test_004", "What gas do plants absorb from the atmosphere?", "carbon dioxide", "short_answer"),
    ("logprob_test_005", "What is the square root of 81?", "9", "numeric"),
]


SYSTEM_PROMPT = """
You are a short-form question answering system.

Answer the question using ONLY the final answer.
The answer must be as short as possible: a word, number, or short phrase.

Return valid JSON only using exactly this schema:

{
  "answer": "...",
  "confidence": 0.0,
  "short_explanation": "..."
}

Rules:
- confidence must be a number between 0.0 and 1.0
- do not include markdown
- do not include extra keys
"""


def summarize_logprobs(response_dict):
    choice = response_dict["choices"][0]
    logprobs = choice.get("logprobs")

    if logprobs is None:
        return {
            "logprobs_present": False,
            "logprobs_location": "choices[0].logprobs",
            "n_logprob_tokens": 0,
            "note": "logprobs field exists but is null",
        }

    content_logprobs = logprobs.get("content") if isinstance(logprobs, dict) else None

    if not content_logprobs:
        return {
            "logprobs_present": True,
            "logprobs_location": "choices[0].logprobs",
            "n_logprob_tokens": 0,
            "note": "logprobs returned, but no content token list found",
        }

    token_logprobs = [
        item.get("logprob")
        for item in content_logprobs
        if item.get("logprob") is not None
    ]

    return {
        "logprobs_present": True,
        "logprobs_location": "choices[0].logprobs.content",
        "n_logprob_tokens": len(token_logprobs),
        "first_tokens": content_logprobs[:10],
        "sum_logprob_all_output_tokens": sum(token_logprobs),
        "avg_logprob_all_output_tokens": (
            sum(token_logprobs) / len(token_logprobs)
            if token_logprobs
            else None
        ),
        "note": (
            "These are logprobs for all generated output tokens. "
            "To compute answer-only confidence, we would still need to isolate "
            "tokens inside the JSON answer field."
        ),
    }


def main():
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for question_id, question, ground_truth, answer_type in QUESTIONS:
            print(f"\nRunning {question_id}...")

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": question},
                    ],
                    temperature=0.7,
                    max_tokens=120,
                    logprobs=True,
                    top_logprobs=5,
                    response_format={"type": "json_object"},

                    # OpenRouter-specific: require provider to support requested params.
                    # This prevents silent routing to providers that ignore logprobs.
                    extra_body={
                        "provider": {
                            "require_parameters": True
                        }
                    },
                )

                response_dict = response.model_dump()
                logprob_summary = summarize_logprobs(response_dict)

                record = {
                    "question_id": question_id,
                    "dataset": "smoke",
                    "model_name": MODEL_NAME,
                    "question": question,
                    "ground_truth": ground_truth,
                    "answer_type": answer_type,
                    "request_params": {
                        "logprobs": True,
                        "top_logprobs": 5,
                        "response_format": {"type": "json_object"},
                        "provider": {"require_parameters": True},
                    },
                    "raw_response": response_dict,
                    "logprob_summary": logprob_summary,
                }

                f.write(json.dumps(record) + "\n")

                choice = response_dict["choices"][0]
                print("Message content:")
                print(choice["message"]["content"])
                print("\nLogprob summary:")
                print(json.dumps(logprob_summary, indent=2)[:3000])

            except Exception as exc:
                error_record = {
                    "question_id": question_id,
                    "dataset": "smoke",
                    "model_name": MODEL_NAME,
                    "question": question,
                    "ground_truth": ground_truth,
                    "answer_type": answer_type,
                    "request_params": {
                        "logprobs": True,
                        "top_logprobs": 5,
                        "response_format": {"type": "json_object"},
                        "provider": {"require_parameters": True},
                    },
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }

                f.write(json.dumps(error_record) + "\n")

                print("ERROR:")
                print(type(exc).__name__, str(exc))

            print("-" * 80)

    print(f"\nSaved raw test responses to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
