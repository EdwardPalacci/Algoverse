#!/usr/bin/env python3
"""
Run an LLM-as-judge correctness pass over AR or DLM generations.

The script reads raw and parsed generation JSONL files separately, aligns rows
by generation key, sends the model answer and ground truth to a judge model,
and writes one JSONL result row per raw generation. Raw rows whose original
parser failed are not skipped; their ground truth is recovered from
PilotDataset.json, and the parse-failure reason is pulled from the source
Markdown error log.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]

AR_RAW = ROOT / "ar_models" / "model_outputs" / "ar_raw_generations.jsonl"
AR_PARSED = ROOT / "ar_models" / "model_outputs" / "ar_parsed_generations.jsonl"
AR_ERRORS = ROOT / "ar_models" / "logs" / "ar_run_errors.md"
DLM_RAW = ROOT / "dlm_models" / "model_outputs" / "dlm_raw_generations.jsonl"
DLM_PARSED = ROOT / "dlm_models" / "model_outputs" / "dlm_parsed_generations.jsonl"
DLM_ERRORS = ROOT / "dlm_models" / "logs" / "dlm_run_errors.md"
PILOT_DATA = ROOT / "data" / "PilotDataset.json"

OUTPUT_DIR = ROOT / "analysis" / "llm_as_judge"
AR_RESULTS = OUTPUT_DIR / "ar_llm_judge_results.jsonl"
DLM_RESULTS = OUTPUT_DIR / "dlm_llm_judge_results.jsonl"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_JUDGE_MODEL = "openai/gpt-4o-mini"
REQUEST_TIMEOUT_SECONDS = 90
MAX_QUERY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 5
DEFAULT_CONCURRENCY = 8

JUDGE_PROMPT_TEMPLATE = """You are grading answer correctness.

Question: {question}
Dataset: {dataset}
Answer type: {answer_type}
Reference answer: {gt}
Model answer: {model_ans}

Grade only the answer, not the confidence or explanation.
The model answer may be raw JSON. If it is JSON with an "answer" field, treat that field as the primary answer. Use the explanation only to understand malformed or ambiguous raw output.

If answer type is numeric, compare the final numeric value only.
If answer type is multiple choice, accept either the correct option letter or the correct option text.
If the reference answer is a list, treat the entries as alternative acceptable answers. The model answer is correct if it matches any acceptable answer or expresses the same factual claim.
For TruthfulQA, mark the answer incorrect if it repeats the misconception targeted by the question, even if it is fluent or common.
If the model answer is empty, malformed, a refusal when the question is answerable, or does not answer the question, mark it incorrect.

Output only CORRECT or INCORRECT."""


def generation_key(row: dict) -> tuple[str, str, str, int]:
    return (
        str(row["model_name"]),
        str(row["question_id"]),
        str(row["condition"]),
        int(row["sample_id"]),
    )


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} line {line_number}: {exc}") from exc
    return rows


def parse_error_log(error_path: Path) -> dict[tuple[str, str, str, int], list[dict]]:
    errors: dict[tuple[str, str, str, int], list[dict]] = {}
    if not error_path.exists():
        return errors

    for line in error_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        message = line.split("] ", 1)[1] if "] " in line else line
        if message.startswith("- "):
            message = message[2:]

        parts = [part.strip() for part in message.split(" | ")]
        if len(parts) < 5:
            continue

        error_type = parts[0]
        model_name = parts[1]
        question_id = parts[2]
        condition = parts[3]
        sample_id = None

        for part in parts[4:]:
            if part.startswith("sample="):
                try:
                    sample_id = int(part.split("=", 1)[1])
                except ValueError:
                    sample_id = None
                break

        if sample_id is None:
            continue

        key = generation_key({
            "model_name": model_name,
            "question_id": question_id,
            "condition": condition,
            "sample_id": sample_id,
        })
        errors.setdefault(key, []).append({
            "error_type": error_type,
            "message": message,
        })

    return errors


def source_paths(source: str) -> tuple[Path, Path, Path, Path]:
    if source == "ar":
        return AR_RAW, AR_PARSED, AR_ERRORS, AR_RESULTS
    if source == "dlm":
        return DLM_RAW, DLM_PARSED, DLM_ERRORS, DLM_RESULTS
    raise ValueError(f"Unknown source: {source}")


def load_pilot_lookup() -> dict[str, dict]:
    with PILOT_DATA.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    return {row["question_id"]: row for row in rows}


def format_ground_truth(value: object) -> str:
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def model_answer(raw_row: dict, parsed_row: dict) -> tuple[str, str]:
    raw_response = raw_row.get("raw_response")

    if raw_response is None:
        return "", "raw_response_empty"

    if isinstance(raw_response, str):
        return raw_response.strip(), "raw_response_full_text"

    return json.dumps(raw_response, ensure_ascii=True), "raw_response_serialized"


def extract_raw_answer_field(raw_response: object) -> object | None:
    if not isinstance(raw_response, str):
        return None

    try:
        parsed = json.loads(raw_response)
        if isinstance(parsed, dict) and "answer" in parsed:
            return parsed.get("answer")
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and "answer" in item:
                    return item.get("answer")
    except Exception:
        pass

    match = re.search(r'"answer"\s*:\s*(".*?"|[^,}\n]+)', raw_response, flags=re.DOTALL)
    if not match:
        return None

    value = match.group(1).strip()
    if value.startswith('"'):
        try:
            return json.loads(value)
        except Exception:
            return value.strip('"')
    return value


def normalized_text(value: object) -> str:
    text = str(value or "").casefold().strip()
    text = re.sub(r"^[a-e]\s*[:.)-]\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\n\r\"'`.,;:")


def numeric_value(value: object) -> float | None:
    if value is None:
        return None

    text = str(value).replace(",", "")
    final_match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if final_match:
        return float(final_match.group(1))

    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not numbers:
        return None
    return float(numbers[-1])


def option_mapping(question: object) -> dict[str, str]:
    mapping = {}
    for line in str(question or "").splitlines():
        match = re.match(r"\s*([A-E])\s*:\s*(.+?)\s*$", line)
        if match:
            mapping[match.group(1).casefold()] = normalized_text(match.group(2))
    return mapping


def deterministic_correctness(parsed_row: dict, raw_row: dict, answer: str) -> str | None:
    raw_answer = extract_raw_answer_field(raw_row.get("raw_response"))
    candidate = raw_answer if raw_answer is not None else answer
    answer_type = normalized_text(parsed_row.get("answer_type"))
    ground_truth = parsed_row.get("ground_truth")

    if answer_type == "numeric":
        expected = numeric_value(ground_truth)
        observed = numeric_value(candidate)
        if expected is not None and observed is not None and abs(expected - observed) < 1e-9:
            return "numeric_exact_match"

    refs = ground_truth if isinstance(ground_truth, list) else [ground_truth]
    candidate_norm = normalized_text(candidate)
    ref_norms = [normalized_text(ref) for ref in refs if normalized_text(ref)]

    if candidate_norm and candidate_norm in ref_norms:
        return "answer_text_exact_match"

    if answer_type in {"multiple_choice", "multiple choice"}:
        options = option_mapping(parsed_row.get("prompt"))
        if len(candidate_norm) == 1 and candidate_norm in options and options[candidate_norm] in ref_norms:
            return "multiple_choice_letter_match"
        if candidate_norm in ref_norms:
            return "multiple_choice_text_match"

    return None


def model_answer_is_empty(raw_row: dict, answer: str) -> bool:
    if not answer.strip():
        return True

    raw_answer = extract_raw_answer_field(raw_row.get("raw_response"))
    if raw_answer is not None and not str(raw_answer).strip():
        return True

    return False


def fallback_parsed_row(raw_row: dict, pilot_lookup: dict[str, dict]) -> dict:
    pilot_row = pilot_lookup.get(raw_row["question_id"], {})
    return {
        "question_id": raw_row["question_id"],
        "dataset": raw_row.get("dataset") or pilot_row.get("dataset"),
        "condition": raw_row["condition"],
        "sample_id": raw_row["sample_id"],
        "model_name": raw_row["model_name"],
        "model_architecture": raw_row["model_architecture"],
        "provider": raw_row.get("provider"),
        "prompt": raw_row.get("prompt") or pilot_row.get("question"),
        "ground_truth": pilot_row.get("ground_truth"),
        "answer_type": pilot_row.get("answer_type"),
        "raw_response": raw_row.get("raw_response"),
        "answer": None,
        "confidence": None,
        "short_explanation": None,
        "parse_success": False,
    }


def parse_failure_summary(errors: list[dict]) -> str | None:
    if not errors:
        return None
    return "; ".join(error["message"] for error in errors)


def build_aligned_rows(
    raw_path: Path,
    parsed_path: Path,
    error_path: Path,
) -> tuple[list[dict], int]:
    raw_rows = read_jsonl(raw_path)
    parsed_rows = read_jsonl(parsed_path)
    parsed_by_key = {generation_key(row): row for row in parsed_rows}
    errors_by_key = parse_error_log(error_path)
    pilot_lookup = load_pilot_lookup()

    aligned = []
    missing_parsed = 0

    for raw_row in raw_rows:
        key = generation_key(raw_row)
        parsed_row = parsed_by_key.get(key)
        source_parse_success = True
        if parsed_row is None:
            missing_parsed += 1
            source_parse_success = False
            parsed_row = fallback_parsed_row(raw_row, pilot_lookup)
        elif parsed_row.get("parse_success") is False:
            source_parse_success = False

        parse_errors = errors_by_key.get(key, [])

        aligned.append({
            "key": key,
            "raw": raw_row,
            "parsed": parsed_row,
            "source_parse_success": source_parse_success,
            "source_parse_errors": parse_errors,
            "source_parse_failure_explanation": parse_failure_summary(parse_errors),
        })

    return aligned, missing_parsed


def existing_keys(output_path: Path) -> set[tuple[str, str, str, int]]:
    keys = set()
    for row in read_jsonl(output_path):
        if row.get("CORRECTNESS") in (0, 1) and "model_name" in row:
            keys.add(generation_key(row))
    return keys


class OpenRouterJudge:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key.strip()
        if not self.api_key:
            raise EnvironmentError("OPENROUTER_API_KEY not set.")
        self.model = model

    def judge(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": 0,
            "max_tokens": 8,
        }
        request = urllib.request.Request(
            OPENROUTER_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/EdwardPalacci/Algoverse",
                "X-Title": "Algoverse LLM judge",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter HTTP {exc.code}: {body}") from exc
        return data["choices"][0]["message"]["content"]


def query_judge_with_retries(judge: OpenRouterJudge, prompt: str) -> str:
    last_error = None
    for attempt in range(1, MAX_QUERY_ATTEMPTS + 1):
        try:
            return judge.judge(prompt)
        except Exception as exc:
            last_error = exc
            if attempt == MAX_QUERY_ATTEMPTS:
                break
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise last_error


def parse_judge_label(text: str) -> tuple[str | None, int | None]:
    normalized = re.sub(r"[^A-Z]", "", str(text).upper())
    if "INCORRECT" in normalized:
        return "INCORRECT", 0
    if "CORRECT" in normalized:
        return "CORRECT", 1
    return None, None


def build_prompt(parsed_row: dict, answer: str) -> str:
    return JUDGE_PROMPT_TEMPLATE.format(
        question=parsed_row.get("prompt", ""),
        dataset=parsed_row.get("dataset", ""),
        answer_type=parsed_row.get("answer_type", ""),
        gt=format_ground_truth(parsed_row.get("ground_truth")),
        model_ans=answer,
    )


def base_result(
    raw_row: dict,
    parsed_row: dict,
    judge_model: str,
    answer: str,
    answer_source: str,
    item: dict,
) -> dict:
    return {
        "question_id": parsed_row["question_id"],
        "dataset": parsed_row["dataset"],
        "condition": parsed_row["condition"],
        "sample_id": parsed_row["sample_id"],
        "model_name": parsed_row["model_name"],
        "model_architecture": parsed_row["model_architecture"],
        "judge_model": judge_model,
        "answer_type": parsed_row.get("answer_type"),
        "question": parsed_row.get("prompt"),
        "ground_truth": parsed_row.get("ground_truth"),
        "model_answer": answer,
        "model_answer_source": answer_source,
        "model_answer_empty": model_answer_is_empty(raw_row, answer),
        "explanation_answer_agreement": None,
        "source_parse_success": item["source_parse_success"],
        "source_parse_error_count": len(item["source_parse_errors"]),
        "source_parse_errors": item["source_parse_errors"],
        "source_parse_failure_explanation": item["source_parse_failure_explanation"],
    }


def judge_one(item: dict, judge_model: str, api_key: str) -> dict:
    raw_row = item["raw"]
    parsed_row = item["parsed"]
    answer, answer_source = model_answer(raw_row, parsed_row)
    result = base_result(
        raw_row,
        parsed_row,
        judge_model,
        answer,
        answer_source,
        item,
    )

    if result["model_answer_empty"]:
        result.update({
            "CORRECTNESS": 0,
            "correctness_label": "INCORRECT",
            "judge_raw_response": "AUTO_INCORRECT_EMPTY_MODEL_ANSWER",
        })
        return result

    deterministic_reason = deterministic_correctness(parsed_row, raw_row, answer)
    if deterministic_reason:
        result.update({
            "CORRECTNESS": 1,
            "correctness_label": "CORRECT",
            "judge_raw_response": "AUTO_CORRECT_DETERMINISTIC_MATCH",
            "auto_correct_reason": deterministic_reason,
        })
        return result

    prompt = build_prompt(parsed_row, answer)
    judge = OpenRouterJudge(api_key=api_key, model=judge_model)
    judge_response = query_judge_with_retries(judge, prompt)
    label, score = parse_judge_label(judge_response)

    result.update({
        "CORRECTNESS": score,
        "correctness_label": label,
        "judge_raw_response": judge_response,
    })

    if score is None:
        result["judge_error"] = "unparseable_judge_response"

    return result


def run(args: argparse.Namespace) -> None:
    raw_path, parsed_path, error_path, output_path = source_paths(args.source)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    aligned, missing_parsed = build_aligned_rows(raw_path, parsed_path, error_path)
    if args.max_rows is not None:
        aligned = aligned[:args.max_rows]

    if args.resume:
        done = existing_keys(output_path)
        aligned = [item for item in aligned if item["key"] not in done]
        output_mode = "a"
    else:
        output_mode = "w"

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key.strip():
        raise EnvironmentError("OPENROUTER_API_KEY not set.")

    total_written = 0
    errors = 0

    with output_path.open(output_mode, encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
            futures = [
                executor.submit(
                    judge_one,
                    item,
                    args.judge_model,
                    api_key,
                )
                for item in aligned
            ]
            iterator = tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"judge/{args.source}",
            )
            for future in iterator:
                try:
                    row = future.result()
                except Exception as exc:
                    errors += 1
                    row = {
                        "CORRECTNESS": None,
                        "correctness_label": None,
                        "explanation_answer_agreement": None,
                        "judge_error": str(exc),
                    }
                output.write(json.dumps(row) + "\n")
                output.flush()
                total_written += 1

    all_results = read_jsonl(output_path)
    scored = [row for row in all_results if row.get("CORRECTNESS") in (0, 1)]
    correct = sum(1 for row in scored if row.get("CORRECTNESS") == 1)
    source_parse_failures = sum(1 for row in all_results if row.get("source_parse_success") is False)
    accuracy = correct / len(scored) if scored else 0.0

    print("\nJudge run complete.")
    print(f"Source: {args.source}")
    print(f"Output: {output_path}")
    print(f"Rows written this run: {total_written}")
    print(f"Rows with worker/API errors this run: {errors}")
    print(f"Raw rows included using PilotDataset metadata because parsed metadata was missing: {missing_parsed}")
    print(f"Source parse-failure rows in output: {source_parse_failures}")
    print(f"Scored rows in output: {len(scored)}")
    print(f"Correct rows in output: {correct}")
    print(f"LLM-judge accuracy: {accuracy:.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run LLM-as-judge correctness grading for AR or DLM generations."
    )
    parser.add_argument(
        "--source",
        choices=["ar", "dlm"],
        required=True,
        help="Generation family to judge.",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help="OpenRouter judge model ID.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Maximum concurrent judge requests.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap for smoke tests.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append only missing generation keys to an existing output file.",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
