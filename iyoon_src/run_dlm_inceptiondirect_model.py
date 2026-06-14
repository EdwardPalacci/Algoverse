#!/usr/bin/env python3
"""
iyoon_src/run_dlm_api.py

DLM generation pipeline for:
"Stress-Testing LLM Confidence Under Induced Overconfidence"

Mirrors run_model.py structure exactly.
Supports: inception/mercury-2, inception/mercury-coder
API: Inception Labs direct (free tier — 10M tokens on signup)

Usage:
    INCEPTION_API_KEY=your-key python3 iyoon_src/run_dlm_api.py
"""

import argparse
import json
import os
import time
from pathlib import Path

import httpx
from tqdm import tqdm

# ============================================================
# Unified JSON schema (mirrors run_model.py)
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
# Prompt conditions (mirrors run_model.py exactly)
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
# Inception API config
# ============================================================

INCEPTION_URL = "https://api.inceptionlabs.ai/v1/chat/completions"
PRIMARY_MODEL = "mercury-2"
BACKUP_MODEL  = "mercury-coder"


def provider_for(model_name: str) -> str:
    return "inception"


def build_client(provider: str) -> dict:
    api_key = os.environ.get("INCEPTION_API_KEY", "")
    if not api_key:
        raise EnvironmentError("INCEPTION_API_KEY not set.")
    return {"api_key": api_key, "url": INCEPTION_URL}


# ============================================================
# API call
# ============================================================

def query_model(
    client,
    provider,
    model,
    system_prompt,
    user_prompt,
    temperature,
    max_tokens,
) -> str:
    headers = {
        "Authorization": f"Bearer {client['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt, m in enumerate([model, BACKUP_MODEL]):
        payload["model"] = m
        try:
            resp = httpx.post(
                client["url"], headers=headers,
                json=payload, timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == 1:
                raise

# ============================================================
# Dataset loader
# ============================================================

def load_pilotdataset(limit=None):
    # Try both capitalizations
    for name in ["PilotDataset.json", "pilotdataset.json", "data/pilotdataset.json"]:
        path = Path(name)
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            if limit:
                data = data[:limit]
            return data
    raise FileNotFoundError("PilotDataset.json not found")


DATASET_LOADERS = {
    "pilotdataset": load_pilotdataset,
    "pilot": load_pilotdataset,
}

# ============================================================
# JSON parsing (mirrors run_model.py exactly)
# ============================================================

def _extract_first_json_object(text):
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None


def parse_response(raw_text):
    if raw_text is None:
        return None
    try:
        return json.loads(raw_text)
    except Exception:
        pass

    candidate = _extract_first_json_object(raw_text)
    if candidate is None:
        return None

    try:
        return json.loads(candidate)
    except Exception:
        return None


def valid_confidence(value):
    return (
        isinstance(value, (int, float))
        and 0.0 <= float(value) <= 1.0
    )


def log_error(error_file, message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    error_file.write(f"- [{timestamp}] {message}\n")
    error_file.flush()


# ============================================================
# Main run loop (mirrors run_model.py)
# ============================================================

def run(args):
    provider = provider_for(args.model)
    client   = build_client(provider)

    raw_output_path    = Path(args.raw_output)
    parsed_output_path = Path(args.parsed_output)
    error_log_path     = Path(args.error_log)

    for p in [raw_output_path, parsed_output_path, error_log_path]:
        p.parent.mkdir(parents=True, exist_ok=True)

    total_generations = 0

    with (
        raw_output_path.open("a", encoding="utf-8")    as raw_f,
        parsed_output_path.open("a", encoding="utf-8") as parsed_f,
        error_log_path.open("a", encoding="utf-8")     as err_f,
    ):
        for dataset_name in args.datasets:
            dataset = DATASET_LOADERS[dataset_name](limit=args.max_questions)
            print(f"\nLoaded {len(dataset)} questions from {dataset_name}")

            for condition in args.conditions:
                print(f"Running condition: {condition}")
                system_prompt = SYSTEM_PROMPTS[condition]
                iterator = tqdm(dataset, desc=f"{dataset_name}/{condition}")

                for item in iterator:
                    for sample_id in range(args.n_samples):
                        try:
                            raw_response = query_model(
                                client=client,
                                provider=provider,
                                model=args.model,
                                system_prompt=system_prompt,
                                user_prompt=item["question"],
                                temperature=args.temperature,
                                max_tokens=args.max_tokens,
                            )

                            raw_record = {
                                "question_id":        item["question_id"],
                                "dataset":            item["dataset"],
                                "condition":          condition,
                                "sample_id":          sample_id,
                                "model_name":         args.model,
                                "model_architecture": "DLM",
                                "prompt":             item["question"],
                                "raw_response":       raw_response,
                            }
                            raw_f.write(json.dumps(raw_record) + "\n")
                            raw_f.flush()

                            parsed = parse_response(raw_response)

                            if parsed is None:
                                log_error(err_f,
                                    f"PARSE ERROR | {item['question_id']} | "
                                    f"{condition} | sample={sample_id}"
                                )
                                continue

                            confidence = parsed.get("confidence")

                            if not valid_confidence(confidence):
                                log_error(err_f,
                                    f"INVALID CONFIDENCE | {item['question_id']} | "
                                    f"value={confidence}"
                                )
                                continue

                            parsed_record = {
                                "question_id":        item["question_id"],
                                "dataset":            item["dataset"],
                                "condition":          condition,
                                "sample_id":          sample_id,
                                "model_name":         args.model,
                                "model_architecture": "DLM",
                                "prompt":             item["question"],
                                "ground_truth":       item["ground_truth"],
                                "answer_type":        item["answer_type"],
                                "raw_response":       raw_response,
                                "answer":             parsed.get("answer"),
                                "confidence":         confidence,
                                "short_explanation":  parsed.get("short_explanation"),
                                "parse_success":      True,
                            }
                            parsed_f.write(json.dumps(parsed_record) + "\n")
                            parsed_f.flush()
                            total_generations += 1

                        except Exception as exc:
                            log_error(err_f,
                                f"RUNTIME ERROR | {type(exc).__name__}: {exc}"
                            )
                            time.sleep(2)

    print("\nRun complete.")
    print(f"Saved {total_generations} parsed generations.")


# ============================================================
# CLI
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default=PRIMARY_MODEL)
    parser.add_argument("--datasets",    nargs="+", default=["pilotdataset"])
    parser.add_argument("--conditions",  nargs="+", default=["neutral", "cautious", "overconfident"])
    parser.add_argument("--n-samples",   type=int,   default=3)
    parser.add_argument("--max-questions", type=int, default=250)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens",  type=int,   default=600)
    parser.add_argument("--raw-output",    default="outputs/dlm_raw_generations.jsonl")
    parser.add_argument("--parsed-output", default="outputs/dlm_parsed_generations.jsonl")
    parser.add_argument("--error-log",     default="logs/dlm_run_errors.md")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
