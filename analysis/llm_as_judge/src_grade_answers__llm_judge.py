#!/usr/bin/env python3
"""
Production LLM-as-judge runner for AR and DLM generations.

This script is the canonical judge entrypoint. It reads raw and parsed
generation JSONL files, aligns rows by generation key, grades numeric and
multiple-choice answers deterministically, and sends short-answer/free-text
cases to a judge model. Parse-failed rows are included instead of dropped.
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
from decimal import Decimal, InvalidOperation
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterator, total=None, desc=None):
        return iterator


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "ar_models").exists() and (parent / "dlm_models").exists():
            return parent
    raise RuntimeError("Could not locate repository root.")


ROOT = repo_root()
OUTPUT_DIR = ROOT / "analysis" / "llm_as_judge"
PILOT_DATA = ROOT / "data" / "PilotDataset.json"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_JUDGE_MODEL = "anthropic/claude-sonnet-4.5"
REQUEST_TIMEOUT_SECONDS = 90
MAX_QUERY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 4
DEFAULT_CONCURRENCY = 8

ABSTAIN_RE = re.compile(
    r"\b(refusal|unknown|cannot answer|can't answer|not enough information|"
    r"insufficient information|no answer)\b",
    flags=re.IGNORECASE,
)

JUDGE_PROMPT_TEMPLATE = """You are grading answer correctness.

Question: {question}
Dataset: {dataset}
Answer type: {answer_type}
Reference answer: {accepted}
Known incorrect answers: {incorrect}
Model answer: {model_answer}

Grade only answer correctness, not confidence calibration.
The model answer may be raw JSON. If it is JSON with an "answer" field, treat that field as the primary answer. Use "short_explanation" only to clarify a concise or ambiguous answer field.

If the reference answer is a list, treat entries as alternative acceptable answers.
The model answer is correct if it matches any acceptable answer or expresses the same factual claim.
Do not require the model answer to repeat every caveat in the reference if its core factual claim is equivalent.
For TruthfulQA, mark the answer incorrect if it repeats the misconception targeted by the question, even if it is fluent or common.
If the model answer is empty, malformed, a refusal when the question is answerable, or does not answer the question, mark it incorrect.

Output JSON only:
{{"verdict":"correct|incorrect|uncertain","reason":"one short sentence"}}"""


def generation_key(row: dict) -> tuple[str, str, str, int]:
    return (
        str(row.get("model_name")),
        str(row.get("question_id")),
        str(row.get("condition")),
        int(row.get("sample_id", 0)),
    )


def source_paths(source: str, raw=None, parsed=None, output=None) -> tuple[Path, Path, Path, Path]:
    if source not in {"ar", "dlm"}:
        raise ValueError(f"Unknown source: {source}")

    raw_path = Path(raw) if raw else ROOT / f"{source}_models" / "model_outputs" / f"{source}_raw_generations.jsonl"
    parsed_path = Path(parsed) if parsed else ROOT / f"{source}_models" / "model_outputs" / f"{source}_parsed_generations.jsonl"
    error_path = ROOT / f"{source}_models" / "logs" / f"{source}_run_errors.md"
    output_path = Path(output) if output else OUTPUT_DIR / f"{source}_llm_judge_results.jsonl"
    return raw_path, parsed_path, error_path, output_path


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


def parse_error_log(path: Path) -> dict[tuple[str, str, str, int], list[dict]]:
    errors: dict[tuple[str, str, str, int], list[dict]] = {}
    if not path.exists():
        return errors

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        message = line.split("] ", 1)[1] if "] " in line else line
        if message.startswith("- "):
            message = message[2:]
        parts = [part.strip() for part in message.split(" | ")]
        if len(parts) < 5:
            continue

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

        key = (parts[1], parts[2], parts[3], sample_id)
        errors.setdefault(key, []).append({
            "error_type": parts[0],
            "message": message,
        })
    return errors


def load_pilot_lookup() -> dict[str, dict]:
    if not PILOT_DATA.exists():
        return {}
    rows = json.loads(PILOT_DATA.read_text(encoding="utf-8"))
    return {row["question_id"]: row for row in rows}


def gt_lookup_from_parsed(parsed_rows: list[dict]) -> dict[str, dict]:
    lookup = {}
    for row in parsed_rows:
        qid = row.get("question_id")
        if qid and qid not in lookup and row.get("ground_truth") is not None:
            lookup[qid] = {
                "dataset": row.get("dataset"),
                "prompt": row.get("prompt"),
                "ground_truth": row.get("ground_truth"),
                "answer_type": row.get("answer_type"),
                "incorrect_answers": row.get("incorrect_answers"),
            }
    return lookup


def fallback_parsed_row(raw_row: dict, parsed_lookup: dict, pilot_lookup: dict) -> dict:
    info = parsed_lookup.get(raw_row.get("question_id"), {})
    pilot = pilot_lookup.get(raw_row.get("question_id"), {})
    return {
        "question_id": raw_row.get("question_id"),
        "dataset": raw_row.get("dataset") or info.get("dataset") or pilot.get("dataset"),
        "condition": raw_row.get("condition"),
        "sample_id": raw_row.get("sample_id"),
        "model_name": raw_row.get("model_name"),
        "model_architecture": raw_row.get("model_architecture"),
        "provider": raw_row.get("provider"),
        "prompt": raw_row.get("prompt") or info.get("prompt") or pilot.get("question"),
        "ground_truth": info.get("ground_truth") if info.get("ground_truth") is not None else pilot.get("ground_truth"),
        "answer_type": info.get("answer_type") or pilot.get("answer_type"),
        "incorrect_answers": info.get("incorrect_answers") or pilot.get("incorrect_answers"),
        "raw_response": raw_row.get("raw_response"),
        "answer": None,
        "confidence": None,
        "short_explanation": None,
        "parse_success": False,
    }


def build_aligned_rows(raw_path: Path, parsed_path: Path, error_path: Path, include_failures: bool) -> tuple[list[dict], int]:
    raw_rows = read_jsonl(raw_path)
    parsed_rows = read_jsonl(parsed_path)
    parsed_by_key = {generation_key(row): row for row in parsed_rows}
    parsed_lookup = gt_lookup_from_parsed(parsed_rows)
    pilot_lookup = load_pilot_lookup()
    errors_by_key = parse_error_log(error_path)

    if not raw_rows and parsed_rows:
        raw_rows = parsed_rows

    aligned = []
    recovered = 0
    for raw_row in raw_rows:
        key = generation_key(raw_row)
        parsed_row = parsed_by_key.get(key)
        if parsed_row is None:
            if not include_failures:
                continue
            recovered += 1
            parsed_row = fallback_parsed_row(raw_row, parsed_lookup, pilot_lookup)

        parse_success = parsed_row.get("parse_success") is not False
        if not parse_success and not include_failures:
            continue

        parse_errors = errors_by_key.get(key, [])
        if parsed_row.get("parse_error"):
            parse_errors = [
                *parse_errors,
                {"error_type": "PARSE ERROR", "message": str(parsed_row.get("parse_error"))},
            ]

        aligned.append({
            "key": key,
            "raw": raw_row,
            "parsed": parsed_row,
            "source_parse_success": parse_success,
            "source_parse_errors": parse_errors,
        })

    return aligned, recovered


def extract_answer_from_raw(raw_response: object) -> object | None:
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


def resolve_model_answer(raw_row: dict, parsed_row: dict, prefer_raw: bool) -> tuple[str, str, str]:
    raw_response = raw_row.get("raw_response")
    raw_full = raw_response.strip() if isinstance(raw_response, str) else ""
    raw_answer = extract_answer_from_raw(raw_response)
    parsed_answer = parsed_row.get("answer")

    extracted = ""
    source = "empty"
    if parsed_answer is not None and str(parsed_answer).strip():
        extracted = str(parsed_answer).strip()
        source = "parsed_answer"
    elif raw_answer is not None and str(raw_answer).strip():
        extracted = str(raw_answer).strip()
        source = "raw_response_answer_field"

    if prefer_raw and raw_full:
        return raw_full, "raw_response_full_text", extracted
    if extracted:
        return extracted, source, extracted
    return raw_full, "raw_response_full_text" if raw_full else "empty", extracted


def aliases(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def normalize_text(value: object) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"^[a-e]\s*[:.)-]\s*", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def numeric_tokens(value: object) -> list[Decimal]:
    tokens = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", str(value or ""))
    numbers = []
    for token in tokens:
        try:
            numbers.append(Decimal(token.replace(",", "")))
        except InvalidOperation:
            pass
    return numbers


def reference_number(value: object) -> Decimal | None:
    match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", str(value or ""))
    if match:
        try:
            return Decimal(match.group(1).replace(",", ""))
        except InvalidOperation:
            return None
    nums = numeric_tokens(value)
    return nums[-1] if nums else None


def answer_number(value: object) -> Decimal | None:
    nums = numeric_tokens(value)
    return nums[-1] if nums else None


def grade_numeric(answer: object, ground_truth: object) -> bool:
    expected = reference_number(ground_truth)
    observed = answer_number(answer)
    return expected is not None and observed is not None and expected == observed


def option_map(prompt: object) -> dict[str, str]:
    options = {}
    for line in str(prompt or "").splitlines():
        match = re.match(r"^\s*([A-E])\s*[:.)]\s*(.+?)\s*$", line)
        if match:
            options[match.group(1).casefold()] = normalize_text(match.group(2))
    return options


def answer_letter(answer: object) -> str | None:
    text = str(answer or "").strip()
    patterns = [
        r"^\s*([A-E])\s*[:.)\s-]",
        r"\boption\s+([A-E])\b",
        r"\banswer\s+is\s+([A-E])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).casefold()
    if re.fullmatch(r"[A-E]", text, flags=re.IGNORECASE):
        return text.casefold()
    return None


def grade_multiple_choice(answer: object, ground_truth: object, prompt: object) -> bool:
    norm_answer = normalize_text(answer)
    norm_truths = {normalize_text(truth) for truth in aliases(ground_truth)}
    if not norm_answer or not norm_truths:
        return False

    letter = answer_letter(answer)
    options = option_map(prompt)
    if letter and options.get(letter) in norm_truths:
        return True

    if norm_answer in norm_truths:
        return True

    return any(truth and (truth in norm_answer or norm_answer in truth) for truth in norm_truths)


def exact_alias_match(answer: object, ground_truth: object) -> bool:
    norm_answer = normalize_text(answer)
    if not norm_answer:
        return False
    return norm_answer in {normalize_text(truth) for truth in aliases(ground_truth)}


def deterministic_short_answer_match(parsed: dict, answer: object) -> str | None:
    if exact_alias_match(answer, parsed.get("ground_truth")):
        return "exact accepted-answer match"

    question = normalize_text(parsed.get("prompt"))
    norm_answer = normalize_text(answer)
    norm_truths = [normalize_text(truth) for truth in aliases(parsed.get("ground_truth"))]

    if (
        "percentage" in question
        and "brain" in question
        and ("100" in norm_answer or "all" in norm_answer or "entire" in norm_answer)
        and any("all" in truth or "majority" in truth for truth in norm_truths)
    ):
        return "percentage answer expresses use of virtually all of the brain"

    return None


class OpenRouterBackend:
    name = "openrouter"

    def __init__(self, model: str | None):
        self.model = model or DEFAULT_JUDGE_MODEL
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not self.api_key:
            raise EnvironmentError("OPENROUTER_API_KEY not set.")

    def grade(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 96,
        }
        payloads = [
            {
                **payload,
                "response_format": {"type": "json_object"},
            },
            payload,
        ]

        last_error = None
        for request_payload in payloads:
            request = urllib.request.Request(
                OPENROUTER_URL,
                data=json.dumps(request_payload).encode("utf-8"),
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
                content = data["choices"][0]["message"].get("content")
                if content is not None and str(content).strip():
                    return str(content)
                last_error = RuntimeError("OpenRouter returned empty judge content.")
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"OpenRouter HTTP {exc.code}: {body}")
                lowered = body.lower()
                if "response_format" not in lowered and "json" not in lowered:
                    raise last_error from exc

        raise last_error


class MockBackend:
    name = "mock"
    model = "mock"

    def grade(self, prompt: str) -> str:
        return json.dumps({"verdict": "uncertain", "reason": "mock backend"})


def make_backend(backend: str, model: str | None):
    if backend == "openrouter":
        return OpenRouterBackend(model)
    if backend == "mock":
        return MockBackend()
    raise ValueError("Only --backend openrouter and --backend mock are supported in this repo.")


def query_judge_with_retries(backend, prompt: str) -> str:
    last_error = None
    for attempt in range(1, MAX_QUERY_ATTEMPTS + 1):
        try:
            return backend.grade(prompt)
        except Exception as exc:
            last_error = exc
            if attempt == MAX_QUERY_ATTEMPTS:
                break
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise last_error


def parse_judge_response(text: str) -> tuple[str, str]:
    try:
        data = json.loads(text)
        verdict = str(data.get("verdict", "")).casefold().strip()
        reason = str(data.get("reason", "")).strip()
    except Exception:
        normalized = re.sub(r"[^a-z]", "", str(text).casefold())
        reason = str(text).strip()
        if "incorrect" in normalized:
            verdict = "incorrect"
        elif "correct" in normalized:
            verdict = "correct"
        elif "uncertain" in normalized:
            verdict = "uncertain"
        else:
            verdict = "uncertain"

    if verdict not in {"correct", "incorrect", "uncertain"}:
        verdict = "uncertain"
    return verdict, reason


def verdict_to_correct(verdict: str) -> bool | None:
    if verdict == "correct":
        return True
    if verdict == "incorrect":
        return False
    return None


def build_judge_prompt(parsed: dict, model_answer: str) -> str:
    return JUDGE_PROMPT_TEMPLATE.format(
        question=parsed.get("prompt", ""),
        dataset=parsed.get("dataset", ""),
        answer_type=parsed.get("answer_type", ""),
        accepted=json.dumps(aliases(parsed.get("ground_truth")), ensure_ascii=False),
        incorrect=json.dumps(aliases(parsed.get("incorrect_answers")), ensure_ascii=False),
        model_answer=model_answer,
    )


def result_row(
    item: dict,
    correct: bool | None,
    method: str,
    verdict: str,
    reason: str,
    judge_model: str | None,
    model_answer: str,
    model_answer_source: str,
    empty_answer: bool = False,
    raw_judge_response: str | None = None,
) -> dict:
    parsed = item["parsed"]
    row = dict(parsed)
    row.update({
        "correct": correct,
        "CORRECTNESS": 1 if correct is True else 0 if correct is False else None,
        "correctness_label": verdict.upper() if verdict else None,
        "judge_verdict": verdict,
        "judge_reason": reason,
        "judge_raw_response": raw_judge_response,
        "judge_model": judge_model,
        "grading_method": method,
        "model_answer": model_answer,
        "model_answer_source": model_answer_source,
        "model_answer_empty": empty_answer,
        "empty_answer": empty_answer,
        "source_parse_success": item["source_parse_success"],
        "source_parse_error_count": len(item["source_parse_errors"]),
        "source_parse_errors": item["source_parse_errors"],
        "source_parse_failure_explanation": "; ".join(error["message"] for error in item["source_parse_errors"]) or None,
    })
    return row


def judge_one(item: dict, backend, judge_input: str) -> dict:
    parsed = item["parsed"]
    raw = item["raw"]
    answer_type = str(parsed.get("answer_type") or "").casefold()
    prefer_raw = judge_input == "raw" and answer_type not in {"numeric", "multiple_choice"}
    model_answer, source, extracted = resolve_model_answer(raw, parsed, prefer_raw=prefer_raw)
    short_answer = extracted or model_answer
    judge_model = getattr(backend, "model", getattr(backend, "name", None))

    if not str(short_answer or "").strip():
        return result_row(
            item,
            False,
            "empty_answer",
            "incorrect",
            "empty model answer",
            judge_model,
            "",
            "empty",
            empty_answer=True,
            raw_judge_response="AUTO_INCORRECT_EMPTY_MODEL_ANSWER",
        )

    if answer_type == "numeric":
        correct = grade_numeric(short_answer, parsed.get("ground_truth"))
        return result_row(
            item,
            correct,
            "numeric_exact_match",
            "correct" if correct else "incorrect",
            "deterministic numeric final-value comparison",
            judge_model,
            short_answer,
            source,
            raw_judge_response="AUTO_NUMERIC_GRADE",
        )

    if answer_type == "multiple_choice":
        correct = grade_multiple_choice(short_answer, parsed.get("ground_truth"), parsed.get("prompt"))
        return result_row(
            item,
            correct,
            "multiple_choice_match",
            "correct" if correct else "incorrect",
            "deterministic multiple-choice comparison",
            judge_model,
            short_answer,
            source,
            raw_judge_response="AUTO_MULTIPLE_CHOICE_GRADE",
        )

    if ABSTAIN_RE.search(str(short_answer)):
        return result_row(
            item,
            False,
            "abstain",
            "incorrect",
            "model abstained instead of answering",
            judge_model,
            short_answer,
            source,
            raw_judge_response="AUTO_INCORRECT_ABSTAIN",
        )

    deterministic_reason = deterministic_short_answer_match(parsed, short_answer)
    if deterministic_reason:
        return result_row(
            item,
            True,
            "short_answer_exact_match",
            "correct",
            f"deterministic {deterministic_reason}",
            judge_model,
            short_answer,
            source,
            raw_judge_response="AUTO_SHORT_ANSWER_MATCH",
        )

    prompt = build_judge_prompt(parsed, model_answer)
    judge_response = query_judge_with_retries(backend, prompt)
    verdict, reason = parse_judge_response(judge_response)
    return result_row(
        item,
        verdict_to_correct(verdict),
        "llm_judge",
        verdict,
        reason,
        judge_model,
        model_answer,
        source,
        raw_judge_response=judge_response,
    )


def existing_keys(output_path: Path) -> set[tuple[str, str, str, int]]:
    return {
        generation_key(row)
        for row in read_jsonl(output_path)
        if "model_name" in row and row.get("CORRECTNESS") in (0, 1)
    }


def run(args: argparse.Namespace) -> None:
    raw_path, parsed_path, error_path, output_path = source_paths(
        args.source,
        raw=args.raw,
        parsed=args.parsed,
        output=args.output,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    backend = make_backend(args.backend, args.judge_model)
    aligned, recovered = build_aligned_rows(
        raw_path,
        parsed_path,
        error_path,
        include_failures=args.include_parse_failures,
    )

    if args.max_rows is not None:
        aligned = aligned[:args.max_rows]

    output_mode = "w"
    if args.resume:
        done = existing_keys(output_path)
        before = len(aligned)
        aligned = [item for item in aligned if item["key"] not in done]
        output_mode = "a"
        print(f"Resume: {before - len(aligned)} rows already done, {len(aligned)} remaining.")

    print(f"Source={args.source} backend={args.backend} model={getattr(backend, 'model', None)}")
    print(f"raw={raw_path} parsed={parsed_path} -> {output_path}")
    print(f"Rows to grade: {len(aligned)} (missing parsed rows recovered: {recovered})")

    written = 0
    errors = 0
    with output_path.open(output_mode, encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
            futures = {
                executor.submit(judge_one, item, backend, args.judge_input): item
                for item in aligned
            }
            iterator = tqdm(as_completed(futures), total=len(futures), desc=f"judge/{args.source}")
            for future in iterator:
                try:
                    row = future.result()
                except Exception as exc:
                    errors += 1
                    item = futures[future]
                    row = result_row(
                        item,
                        None,
                        "judge_error",
                        "uncertain",
                        str(exc),
                        getattr(backend, "model", getattr(backend, "name", None)),
                        "",
                        "judge_error",
                        raw_judge_response=None,
                    )
                    row["judge_error"] = str(exc)

                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                written += 1

    all_rows = read_jsonl(output_path)
    scored = [row for row in all_rows if row.get("CORRECTNESS") in (0, 1)]
    correct = sum(1 for row in scored if row.get("CORRECTNESS") == 1)
    uncertain = sum(1 for row in all_rows if row.get("CORRECTNESS") is None)
    parse_failures = sum(1 for row in all_rows if row.get("source_parse_success") is False)
    accuracy = correct / len(scored) if scored else float("nan")

    print("\nJudge run complete.")
    print(f"Rows written this run: {written}")
    print(f"Rows with worker/API errors this run: {errors}")
    print(f"Scored rows in output: {len(scored)}")
    print(f"Correct rows in output: {correct}")
    print(f"Uncertain rows excluded from accuracy: {uncertain}")
    print(f"Source parse-failure rows in output: {parse_failures}")
    print(f"Accuracy over decided rows: {accuracy:.4f}")
    print(f"Output: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Grade AR or DLM generations with deterministic checks plus an LLM judge."
    )
    parser.add_argument("--source", choices=["ar", "dlm"], required=True)
    parser.add_argument("--backend", choices=["openrouter", "mock"], default="openrouter")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument(
        "--judge-input",
        choices=["raw", "extracted"],
        default="raw",
        help="For free-text answers, send full raw generation or extracted answer to the judge.",
    )
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--include-parse-failures", dest="include_parse_failures", action="store_true", default=True)
    parser.add_argument("--skip-parse-failures", dest="include_parse_failures", action="store_false")
    parser.add_argument("--raw", default=None)
    parser.add_argument("--parsed", default=None)
    parser.add_argument("--output", default=None)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
