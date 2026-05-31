#!/usr/bin/env python3

"""
run_dlm.py

Pilot runner for diffusion language model experiments.

Outputs:
outputs/dlm_generations_pilot.jsonl

Required fields:

- question_id
- dataset
- condition
- final_answer
- confidence
- raw_output
- parse_success

Optional:

- intermediate_generations
"""

import json
from pathlib import Path


OUTPUT_FILE = Path(
    "outputs/dlm_generations_pilot.jsonl"
)


def save_generation(record):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            json.dumps(record)
            + "\n"
        )


def main():

    example_record = {

        "question_id": "example_001",

        "dataset": "gsm8k",

        "condition": "neutral",

        "final_answer": "42",

        "confidence": 0.95,

        "raw_output": "The answer is 42.",

        "parse_success": True,

        "intermediate_generations": [
            "[MASK] [MASK]",
            "The answer is [MASK]",
            "The answer is 42"
        ]
    }

    save_generation(
        example_record
    )

    print(
        "Saved example DLM generation."
    )


if name == "main":
    main()
