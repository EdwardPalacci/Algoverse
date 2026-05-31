#!/usr/bin/env python3

"""
run_dlm.py

Pilot diffusion-language-model runner.

Current version:

- saves generations
- supports intermediate generations
- creates required output schema

Actual DLM integration will be added later.
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


def build_record(
    question_id,
    dataset,
    condition,
    final_answer,
    confidence,
    raw_output,
    parse_success,
    intermediate_generations=None
):

    record = {

        "question_id": question_id,

        "dataset": dataset,

        "condition": condition,

        "final_answer": final_answer,

        "confidence": confidence,

        "raw_output": raw_output,

        "parse_success": parse_success
    }

    if intermediate_generations is not None:

        record[
            "intermediate_generations"
        ] = intermediate_generations

    return record


def main():

    record = build_record(
        question_id="pilot_001",
        dataset="gsm8k",
        condition="neutral",
        final_answer="42",
        confidence=0.95,
        raw_output="The answer is 42.",
        parse_success=True,
        intermediate_generations=[
            "[MASK] [MASK]",
            "The answer is [MASK]",
            "The answer is 42"
        ]
    )

    save_generation(record)

    print(
        "Saved generation to outputs/dlm_generations_pilot.jsonl"
    )


if name == "main":
    main()
