#!/usr/bin/env python3

import json
import os
from pathlib import Path

from openai import OpenAI


MODEL_NAME = "qwen/qwen-2.5-7b-instruct"
OUTPUT_PATH = Path("outputs/ar_logprobs_test_raw.jsonl")


QUESTIONS = [
    {
        "question_id": "logprob_test_001",
        "dataset": "smoke",
        "question": "What is 2 + 2?",
        "ground_truth": "4",
        "answer_type": "numeric",
    },
    {
        "question_id": "logprob_test_002",
        "dataset": "smoke",
        "question": "What is the capital of France?",
        "ground_truth": "Paris",
        "answer_type": "short_answer",
    },
    {
        "question_id": "logprob_test_003",
        "dataset": "smoke",
        "question": "Who wrote Hamlet?",
        "ground_truth": "William Shakespeare",
        "answer_type": "short_answer",
    },
    {
        "question_id": "logprob_test_004",
        "dataset": "smoke",
        "question": "What gas do plants absorb from the atmosphere?",
        "ground_truth": "carbon dioxide",
        "answer_type": "short_answer",
    },
    {
        "question_id": "logprob_test_005",
        "dataset": "smoke",
        "question": "What is the square root of 81?",
        "ground_truth": "9",
        "answer_type": "numeric",
    },
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


def main():
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for item in QUESTIONS:
            print(f"Running {item['question_id']}...")

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": item["question"]},
                ],
                temperature=0.7,
                max_tokens=120,

                # The actual logprobs test:
                logprobs=True,
                top_logprobs=5,

                response_format={"type": "json_object"},
            )

            # Convert SDK object into regular JSON-serializable dict
            response_dict = response.model_dump()

            record = {
                "question_id": item["question_id"],
                "dataset": item["dataset"],
                "model_name": MODEL_NAME,
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "answer_type": item["answer_type"],
                "raw_response": response_dict,
            }

            f.write(json.dumps(record) + "\n")

            choice = response_dict["choices"][0]
            print("Message content:")
            print(choice["message"]["content"])
            print("\nLogprobs field:")
            print(json.dumps(choice.get("logprobs"), indent=2)[:2000])
            print("-" * 80)

    print(f"Saved raw test responses to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
