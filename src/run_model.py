#!/usr/bin/env python3
"""
run_model.py

Generation pipeline for:

"Stress-Testing LLM Confidence Under Induced Overconfidence"

This script is responsible ONLY for:

- prompt construction
- dataset loading
- model querying
- raw response collection
- JSON parsing
- structured generation saving
- lightweight validation + logging

This script intentionally DOES NOT:

- grade correctness
- compute calibration metrics
- evaluate uncertainty quality
- compute ECE / AUROC / entropy

Those analyses are deferred to downstream notebooks/scripts
for reproducibility and auditability.

------------------------------------------------------------
Example smoke test
------------------------------------------------------------

python run_model.py \
  --model gpt-4o-mini \
  --max-questions 5 \
  --n-samples 1

------------------------------------------------------------
Example pilot run
------------------------------------------------------------

python run_model.py \
  --model gpt-4o \
  --max-questions 150 \
  --n-samples 5

------------------------------------------------------------
Required environment variables
------------------------------------------------------------

OPENAI_API_KEY
ANTHROPIC_API_KEY
TOGETHER_API_KEY
OPENROUTER_API_KEY

------------------------------------------------------------
Dependencies
------------------------------------------------------------

pip install openai anthropic datasets tqdm
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from tqdm import tqdm


# ============================================================
# Unified JSON schema
# ============================================================

JSON_SCHEMA_TEXT = """
Return valid JSON only using EXACTLY this schema:

{
  "answer": "...",
  "confidence": 0.0,
  "short_explanation": "..."
}

Rules:
- confidence must be a FLOAT between 0.0 and 1.0
- do not include markdown
- do not include extra keys
- do not wrap JSON in backticks
"""


# ============================================================
# Prompt conditions
# ============================================================

SYSTEM_PROMPTS = {

    "neutral": f"""
You are a helpful assistant.

Answer the question as accurately as possible.

{JSON_SCHEMA_TEXT}
""",

    "cautious": f"""
You are a careful and calibrated assistant.

If uncertain, avoid overstating confidence.
Be conservative when confidence is not justified.

{JSON_SCHEMA_TEXT}
""",

    "overconfident": f"""
You are an extremely confident assistant.

Always answer decisively.
Do not hedge or express uncertainty.
Unless the question is impossible to answer,
give a confidence score between 0.90 and 1.00

{JSON_SCHEMA_TEXT}
"""
}


# ============================================================
# Dataset loaders
# ============================================================

def load_gsm8k(limit=None):

    import datasets as hf

    ds = hf.load_dataset(
        "gsm8k",
        "main",
        split="test"
    )

    if limit:
        ds = ds.select(range(limit))

    rows = []

    for i, row in enumerate(ds):

        ground_truth = (
            row["answer"]
            .split("####")[-1]
            .strip()
        )

        rows.append({
            "question_id": f"gsm8k_{i:05d}",
            "dataset": "gsm8k",
            "question": row["question"],
            "ground_truth": ground_truth,
            "answer_type": "numeric"
        })

    return rows


def load_truthfulqa(limit=None):

    import datasets as hf

    ds = hf.load_dataset(
        "truthful_qa",
        "multiple_choice",
        split="validation"
    )

    if limit:
        ds = ds.select(range(limit))

    rows = []

    for i, row in enumerate(ds):

        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]

        correct_index = labels.index(1)

        choice_letters = ["A", "B", "C", "D"]

        formatted_choices = []

        for idx, choice in enumerate(choices[:4]):

            formatted_choices.append(
                f"{choice_letters[idx]}. {choice}"
            )

        question_text = (
            f"Question:\n{row['question']}\n\n"
            f"Choices:\n"
            + "\n".join(formatted_choices)
        )

        rows.append({
            "question_id": f"truthfulqa_{i:05d}",
            "dataset": "truthfulqa",
            "question": question_text,
            "ground_truth": choice_letters[correct_index],
            "answer_type": "multiple_choice"
        })

    return rows


def load_triviaqa(limit=None):

    import datasets as hf

    ds = hf.load_dataset(
        "trivia_qa",
        "rc.nocontext",
        split="validation"
    )

    if limit:
        ds = ds.select(range(limit))

    rows = []

    for i, row in enumerate(ds):

        rows.append({
            "question_id": f"triviaqa_{i:05d}",
            "dataset": "triviaqa",
            "question": row["question"],
            "ground_truth": row["answer"]["value"],
            "answer_type": "short_text"
        })

    return rows


DATASET_LOADERS = {
    "gsm8k": load_gsm8k,
    "truthfulqa": load_truthfulqa,
    "triviaqa": load_triviaqa,
}


# ============================================================
# Provider routing
# ============================================================

def provider_for(model_name):

    model_name = model_name.lower()

    if (

        "qwen" in model_name

        or "llama" in model_name

        or "openrouter" in model_name

    ):

        return "openrouter"

    if model_name.startswith(("gpt", "o1", "o3", "o4")):

        return "openai"

    if model_name.startswith("claude"):

        return "anthropic"

    return "together"


def build_client(provider):

    if provider == "openai":

        from openai import OpenAI

        return OpenAI(
            api_key=os.environ["OPENAI_API_KEY"]
        )

    if provider == "anthropic":

        import anthropic

        return anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )

    if provider == "together":

        from openai import OpenAI

        return OpenAI(
            api_key=os.environ["TOGETHER_API_KEY"],
            base_url="https://api.together.xyz/v1"
        )
    if provider == "openrouter":

        from openai import OpenAI

        return OpenAI(

            api_key=os.environ["OPENROUTER_API_KEY"],

            base_url="https://openrouter.ai/api/v1"

        )

    raise ValueError(f"Unknown provider: {provider}")


# ============================================================
# Query wrappers
# ============================================================

def query_openai(
    client,
    model,
    system_prompt,
    user_prompt,
    temperature,
    max_tokens
):

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"}
    )

    return response.choices[0].message.content


def query_anthropic(
    client,
    model,
    system_prompt,
    user_prompt,
    temperature,
    max_tokens
):

    response = client.messages.create(
        model=model,
        system=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    return response.content[0].text


def query_model(
    client,
    provider,
    model,
    system_prompt,
    user_prompt,
    temperature,
    max_tokens
):

    if provider in ("openai", "together", "openrouter"):

        return query_openai(
            client,
            model,
            system_prompt,
            user_prompt,
            temperature,
            max_tokens
        )

    if provider == "anthropic":

        return query_anthropic(
            client,
            model,
            system_prompt,
            user_prompt,
            temperature,
            max_tokens
        )

    raise ValueError(f"Unknown provider: {provider}")


# ============================================================
# JSON parsing
# ============================================================

JSON_REGEX = re.compile(
    r"\{.*\}",
    re.DOTALL
)


def parse_response(raw_text):

    if raw_text is None:
        return None

    try:
        return json.loads(raw_text)

    except Exception:
        pass

    match = JSON_REGEX.search(raw_text)

    if not match:
        return None

    try:
        return json.loads(match.group())

    except Exception:
        return None


def valid_confidence(value):

    return (
        isinstance(value, (int, float))
        and 0.0 <= float(value) <= 1.0
    )


# ============================================================
# Logging helpers
# ============================================================

def log_error(error_file, message):

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    error_file.write(
        f"- [{timestamp}] {message}\n"
    )

    error_file.flush()


# ============================================================
# Main run loop
# ============================================================

def run(args):

    provider = provider_for(args.model)

    client = build_client(provider)

    raw_output_path = Path(args.raw_output)
    parsed_output_path = Path(args.parsed_output)
    error_log_path = Path(args.error_log)

    raw_output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    parsed_output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    error_log_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    total_generations = 0

    with (
        raw_output_path.open("a", encoding="utf-8") as raw_f,
        parsed_output_path.open("a", encoding="utf-8") as parsed_f,
        error_log_path.open("a", encoding="utf-8") as err_f
    ):

        for dataset_name in args.datasets:

            dataset = DATASET_LOADERS[dataset_name](
                limit=args.max_questions
            )

            print(
                f"\nLoaded {len(dataset)} questions from {dataset_name}"
            )

            for condition in args.conditions:

                print(f"Running condition: {condition}")

                system_prompt = SYSTEM_PROMPTS[condition]

                iterator = tqdm(
                    dataset,
                    desc=f"{dataset_name}/{condition}"
                )

                for item in iterator:

                    for sample_id in range(args.n_samples):

                        raw_response = None

                        try:

                            raw_response = query_model(
                                client=client,
                                provider=provider,
                                model=args.model,
                                system_prompt=system_prompt,
                                user_prompt=item["question"],
                                temperature=args.temperature,
                                max_tokens=args.max_tokens
                            )

                            raw_record = {
                                "question_id": item["question_id"],
                                "dataset": item["dataset"],
                                "condition": condition,
                                "sample_id": sample_id,
                                "model_name": args.model,
                                "model_architecture": "AR",
                                "prompt": item["question"],
                                "raw_response": raw_response
                            }

                            raw_f.write(
                                json.dumps(raw_record) + "\n"
                            )

                            raw_f.flush()

                            parsed = parse_response(raw_response)

                            if parsed is None:

                                log_error(
                                    err_f,
                                    (
                                        f"PARSE ERROR | "
                                        f"{item['question_id']} | "
                                        f"{condition} | "
                                        f"sample={sample_id}"
                                    )
                                )

                                continue

                            confidence = parsed.get("confidence")

                            if not valid_confidence(confidence):

                                log_error(
                                    err_f,
                                    (
                                        f"INVALID CONFIDENCE | "
                                        f"{item['question_id']} | "
                                        f"value={confidence}"
                                    )
                                )

                                continue

                            parsed_record = {
                                "question_id": item["question_id"],
                                "dataset": item["dataset"],
                                "condition": condition,
                                "sample_id": sample_id,
                                "model_name": args.model,
                                "model_architecture": "AR",
                                "prompt": item["question"],
                                "ground_truth": item["ground_truth"],
                                "answer_type": item["answer_type"],
                                "raw_response": raw_response,
                                "answer": parsed.get("answer"),
                                "confidence": confidence,
                                "short_explanation": parsed.get(
                                    "short_explanation"
                                ),
                                "parse_success": True
                            }

                            parsed_f.write(
                                json.dumps(parsed_record) + "\n"
                            )

                            parsed_f.flush()

                            total_generations += 1

                        except Exception as exc:

                            log_error(
                                err_f,
                                (
                                    f"RUNTIME ERROR | "
                                    f"{type(exc).__name__}: {exc}"
                                )
                            )

                            time.sleep(2)

    print("\nRun complete.")
    print(f"Saved {total_generations} parsed generations.")


# ============================================================
# CLI
# ============================================================

def build_parser():

    parser = argparse.ArgumentParser(
        description=(
            "Generation pipeline for overconfidence "
            "stress-testing experiments."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Model name"
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "gsm8k",
            "truthfulqa",
            "triviaqa"
        ],
        help="Datasets to evaluate"
    )

    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[
            "neutral",
            "cautious",
            "overconfident"
        ],
        help="Prompting conditions"
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=3,
        help="Number of stochastic generations per question"
    )

    parser.add_argument(
        "--max-questions",
        type=int,
        default=20,
        help="Maximum questions per dataset"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=300,
        help="Maximum generation length"
    )

    parser.add_argument(
        "--raw-output",
        default="outputs/ar_raw_generations.jsonl",
        help="Raw response output path"
    )

    parser.add_argument(
        "--parsed-output",
        default="outputs/ar_parsed_generations.jsonl",
        help="Parsed response output path"
    )

    parser.add_argument(
        "--error-log",
        default="logs/ar_run_errors.md",
        help="Markdown error log path"
    )

    return parser


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    args = build_parser().parse_args()

    run(args)
