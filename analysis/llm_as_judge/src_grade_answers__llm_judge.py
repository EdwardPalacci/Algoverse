#!/usr/bin/env python3
"""
Production LLM-as-judge runner. Merges the robust execution skeleton from
Edward's analysis/llm_as_judge/run_llm_judge.py (concurrency, resume, raw/parsed
alignment, parse-failure recovery) with the hardened grading logic from
grade_answers_llm.py (rubric with few-shot + `uncertain` verdict, deterministic
numeric/multiple-choice routing, multi-backend support including local Ollama,
and a validation harness via validate_judge.py).
 
What it does
------------
  - Reads the raw and parsed generation JSONL separately and aligns them by
    (model_name, question_id, condition, sample_id).
  - Rows whose parser failed (present in raw but not parsed) are NOT dropped:
    their ground truth/answer_type are recovered from the parsed file's
    per-question lookup (or an optional PilotDataset.json), and the answer is
    recovered from the raw response. Controlled by --include-parse-failures.
  - Numeric and multiple-choice answers are graded deterministically (string
    match is correct there); free-text goes to the LLM judge.
  - The judge can return correct / incorrect / uncertain. `uncertain` (bad gold
    or unparseable judge output) -> correct = None, excluded from accuracy, so
    polluted gold and judge errors never bias the headline number.
  - Output is written in the SAME schema as *_graded_generations.jsonl, so it
    feeds notebooks/01_pilot_analysis.py directly, plus extra judge fields
    (CORRECTNESS, judge_verdict, judge_reason, ...) for auditing and
    compatibility with Edward's downstream.
 
Backends (same as grade_answers_llm): openrouter, openai, ollama, mock.
 
Usage
-----
    # OpenRouter
    export OPENROUTER_API_KEY=sk-or-...
    python3 src/run_llm_judge.py --source ar \
        --backend openrouter --judge-model openai/gpt-4o-mini --concurrency 8
 
    # Local Ollama (free, no key); keep concurrency low for a local model
    python3 src/run_llm_judge.py --source ar \
        --backend ollama --judge-model llama3.1:8b --concurrency 2
 
    # Resume an interrupted run (append only missing rows)
    python3 src/run_llm_judge.py --source ar --backend ollama \
        --judge-model llama3.1:8b --resume
 
    # Then analyze the judge-graded file
    python3 notebooks/01_pilot_analysis.py outputs/ar_graded_generations_llm_judge.jsonl
"""
 
from __future__ import annotations
 
import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
 
# Work whether this file lives in src/ or at the repo root (so it can be invoked
# as `python3 src_grade_answers__llm_judge.py` from the root). ROOT is the repo
# root; the sibling modules live under ROOT/src.
_HERE = Path(__file__).resolve().parent
ROOT = _HERE if (_HERE / "outputs").exists() else _HERE.parent
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
 
# Reuse the hardened grading pieces so logic stays single-sourced.
try:
    from grade_answers_llm import (
        build_prompt, make_backend, _aliases_list, _verdict_to_correct, ABSTAIN_PATTERNS,
    )
    from grade_answers import _grade_numeric, _grade_multiple_choice
except ImportError:  # pragma: no cover
    from src.grade_answers_llm import (
        build_prompt, make_backend, _aliases_list, _verdict_to_correct, ABSTAIN_PATTERNS,
    )
    from src.grade_answers import _grade_numeric, _grade_multiple_choice
 
MAX_QUERY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 4
 
try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    def tqdm(it, total=None, desc=None):
        return it
 
 
# ---------------------------------------------------------------------------
# Paths (auto-detect our layout, then Edward's repo layout)
# ---------------------------------------------------------------------------
 
def resolve_paths(source: str, raw=None, parsed=None, output=None) -> tuple[Path, Path, Path]:
    """Return (raw_path, parsed_path, output_path), honoring explicit overrides
    and otherwise auto-detecting the layout."""
    candidates_raw = [
        ROOT / "outputs" / f"{source}_raw_generations.jsonl",
        ROOT / f"{source}_models" / "model_outputs" / f"{source}_raw_generations.jsonl",
    ]
    candidates_parsed = [
        ROOT / "outputs" / f"{source}_parsed_generations.jsonl",
        ROOT / f"{source}_models" / "model_outputs" / f"{source}_parsed_generations.jsonl",
    ]
    raw_path = Path(raw) if raw else next((p for p in candidates_raw if p.exists()), candidates_raw[0])
    parsed_path = Path(parsed) if parsed else next((p for p in candidates_parsed if p.exists()), candidates_parsed[0])
    output_path = Path(output) if output else ROOT / "outputs" / f"{source}_graded_generations_llm_judge.jsonl"
    return raw_path, parsed_path, output_path
 
 
# ---------------------------------------------------------------------------
# Alignment + parse-failure recovery
# ---------------------------------------------------------------------------
 
def generation_key(row: dict) -> tuple[str, str, str, int]:
    return (str(row.get("model_name")), str(row.get("question_id")),
            str(row.get("condition")), int(row.get("sample_id", 0)))
 
 
def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL in {path} line {n}: {e}") from e
    return rows
 
 
def gt_lookup_from_parsed(parsed_rows: list[dict]) -> dict[str, dict]:
    """question_id -> {ground_truth, answer_type, dataset, prompt} recovered from
    the parsed rows themselves, so we can grade parse-failure rows without an
    external dataset file (ground truth is per-question, not per-sample)."""
    out: dict[str, dict] = {}
    for r in parsed_rows:
        qid = r.get("question_id")
        if qid and qid not in out and r.get("ground_truth") is not None:
            out[qid] = {
                "ground_truth": r.get("ground_truth"),
                "answer_type": r.get("answer_type"),
                "dataset": r.get("dataset"),
                "prompt": r.get("prompt"),
            }
    return out
 
 
def load_pilot_lookup(path: Path | None) -> dict[str, dict]:
    if not path or not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {r["question_id"]: r for r in rows}
 
 
def extract_answer_from_raw(raw_response: Any) -> Any:
    """Recover the answer field from a raw (possibly malformed) response string."""
    if not isinstance(raw_response, str):
        return None
    try:
        parsed = json.loads(raw_response)
        if isinstance(parsed, dict):
            return parsed.get("answer")
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and "answer" in item:
                    return item.get("answer")
    except Exception:
        pass
    m = re.search(r'"answer"\s*:\s*(".*?"|[^,}\n]+)', raw_response, flags=re.DOTALL)
    if not m:
        return None
    value = m.group(1).strip()
    if value.startswith('"'):
        try:
            return json.loads(value)
        except Exception:
            return value.strip('"')
    return value
 
 
def resolve_model_answer(raw_row: dict, parsed_row: dict) -> tuple[str, str]:
    """Return (answer_text, source). Prefer the parsed answer, then the answer
    field recovered from raw, then the full raw response."""
    pa = parsed_row.get("answer")
    if pa is not None and str(pa).strip():
        return str(pa), "parsed_answer"
    ra = extract_answer_from_raw(raw_row.get("raw_response"))
    if ra is not None and str(ra).strip():
        return str(ra), "raw_response_answer_field"
    return str(raw_row.get("raw_response") or ""), "raw_response_full_text"
 
 
def fallback_parsed_row(raw_row: dict, gt_lookup: dict, pilot_lookup: dict) -> dict:
    """Construct a parsed-shaped row for a generation whose parser failed."""
    info = gt_lookup.get(raw_row.get("question_id"), {})
    pilot = pilot_lookup.get(raw_row.get("question_id"), {})
    return {
        "question_id": raw_row.get("question_id"),
        "dataset": raw_row.get("dataset") or info.get("dataset") or pilot.get("dataset"),
        "condition": raw_row.get("condition"),
        "sample_id": raw_row.get("sample_id"),
        "model_name": raw_row.get("model_name"),
        "model_architecture": raw_row.get("model_architecture"),
        "prompt": raw_row.get("prompt") or info.get("prompt") or pilot.get("question"),
        "ground_truth": info.get("ground_truth") if info.get("ground_truth") is not None
        else pilot.get("ground_truth"),
        "answer_type": info.get("answer_type") or pilot.get("answer_type"),
        "raw_response": raw_row.get("raw_response"),
        "answer": None, "confidence": None, "short_explanation": None,
        "parse_success": False,
    }
 
 
def build_aligned_rows(raw_path, parsed_path, pilot_path, include_failures: bool):
    raw_rows = read_jsonl(raw_path)
    parsed_rows = read_jsonl(parsed_path)
    if not raw_rows and parsed_rows:
        # No raw file available; just grade the parsed rows directly.
        return [{"key": generation_key(r), "raw": r, "parsed": r,
                 "source_parse_success": True} for r in parsed_rows], 0
    parsed_by_key = {generation_key(r): r for r in parsed_rows}
    gt = gt_lookup_from_parsed(parsed_rows)
    pilot = load_pilot_lookup(pilot_path)
 
    aligned, recovered = [], 0
    for raw_row in raw_rows:
        key = generation_key(raw_row)
        parsed_row = parsed_by_key.get(key)
        ok = True
        if parsed_row is None:
            if not include_failures:
                continue
            ok = False
            recovered += 1
            parsed_row = fallback_parsed_row(raw_row, gt, pilot)
        aligned.append({"key": key, "raw": raw_row, "parsed": parsed_row,
                        "source_parse_success": ok})
    return aligned, recovered
 
 
def existing_keys(output_path: Path) -> set:
    keys = set()
    for row in read_jsonl(output_path):
        if "model_name" in row and row.get("CORRECTNESS") in (0, 1):
            keys.add(generation_key(row))
    return keys
 
 
# ---------------------------------------------------------------------------
# Grading one row (deterministic routing + judge)
# ---------------------------------------------------------------------------
 
def _judge_with_retries(backend, question, accepted, incorrect, answer, raw_mode=False):
    last = None
    for attempt in range(1, MAX_QUERY_ATTEMPTS + 1):
        try:
            return backend.grade(question, accepted, incorrect, answer, raw_mode=raw_mode)
        except Exception as e:  # network / API errors
            last = e
            if attempt == MAX_QUERY_ATTEMPTS:
                break
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise last
 
 
def judge_one(item: dict, backend, judge_input: str = "raw") -> dict:
    """Grade one row.
 
    Tommy's two fixes are implemented here:
      1. EMPTY-ANSWER GUARD. If no answer can be extracted (neither a parsed
         answer nor one recoverable from the raw output), the row is marked
         empty and INCORRECT deterministically, with NO judge call. This stops
         the bug where empty answers were judged CORRECT.
      2. RAW FED TO JUDGE. For free-text, the full raw generation is sent to the
         judge (raw_mode), so the judge doubles as a tolerant parser and the
         effective parse rate goes up. Numeric/multiple-choice stay deterministic
         on the extracted answer (string match is correct there).
    """
    parsed = item["parsed"]
    raw = item["raw"]
    answer_type = (parsed.get("answer_type") or "").lower()
    ground_truth = parsed.get("ground_truth")
    raw_text = raw.get("raw_response")
    raw_str = raw_text if isinstance(raw_text, str) else ""
 
    # Best extractable short answer: parsed answer, else regex-recovered from raw.
    extracted = parsed.get("answer")
    if extracted is None or not str(extracted).strip():
        extracted = extract_answer_from_raw(raw_text)
    extracted_str = "" if extracted is None else str(extracted).strip()
 
    def result(correct, method, *, verdict=None, reason=None, model=None,
               model_answer="", source="", empty=False, abstain=False):
        out = dict(parsed)  # full parsed schema so the analysis pipeline can read it
        out.update({
            "correct": correct,
            "grading_method": method,
            "CORRECTNESS": (1 if correct is True else (0 if correct is False else None)),
            "correctness_label": (verdict.upper() if verdict else None),
            "judge_verdict": verdict,
            "judge_reason": reason,
            "judge_model": model,
            "model_answer": model_answer,
            "model_answer_source": source,
            "empty_answer": empty,
            "abstain": abstain,
            "source_parse_success": item["source_parse_success"],
        })
        return out
 
    # FIX 1: empty answer -> empty + INCORRECT, deterministic, no judge call.
    if not extracted_str:
        return result(False, "empty_answer", verdict="incorrect",
                      reason="empty model answer (nothing to grade)",
                      model_answer="", source="empty", empty=True)
 
    # Numeric / multiple-choice: deterministic on the extracted answer.
    if answer_type == "numeric":
        return result(_grade_numeric(extracted_str, ground_truth),
                      "numeric_exact_match", model_answer=extracted_str, source="extracted_answer")
    if answer_type == "multiple_choice":
        return result(_grade_multiple_choice(extracted_str, ground_truth),
                      "multiple_choice_match", model_answer=extracted_str, source="extracted_answer")
 
    # Abstention -> incorrect (model did not answer), flagged.
    if ABSTAIN_PATTERNS.search(extracted_str):
        return result(False, "llm_judge", verdict="incorrect", reason="model abstained",
                      model="(abstain)", model_answer=extracted_str,
                      source="extracted_answer", abstain=True)
 
    # FIX 2: free-text -> feed the RAW generation to the judge (parser + grader).
    if judge_input == "raw" and raw_str.strip():
        jin, source, raw_mode = raw_str, "raw_response_full_text", True
    else:
        jin, source, raw_mode = extracted_str, "extracted_answer", False
 
    accepted = _aliases_list(ground_truth)
    incorrect = _aliases_list(parsed.get("incorrect_answers"))
    verdict, reason = _judge_with_retries(
        backend, parsed.get("prompt", ""), accepted, incorrect, jin, raw_mode=raw_mode)
    return result(_verdict_to_correct(verdict), "llm_judge", verdict=verdict, reason=reason,
                  model=getattr(backend, "model", getattr(backend, "name", None)),
                  model_answer=jin, source=source)
 
 
# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
 
def run(args: argparse.Namespace) -> None:
    raw_path, parsed_path, output_path = resolve_paths(
        args.source, args.raw, args.parsed, args.output)
    pilot_path = Path(args.pilot) if args.pilot else (ROOT / "data" / "PilotDataset.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
 
    backend = make_backend(args.backend, args.judge_model, args.base_url)
 
    aligned, recovered = build_aligned_rows(
        raw_path, parsed_path, pilot_path, args.include_parse_failures)
    if args.max_rows is not None:
        aligned = aligned[: args.max_rows]
 
    mode = "w"
    if args.resume and output_path.exists():
        done = existing_keys(output_path)
        before = len(aligned)
        aligned = [it for it in aligned if it["key"] not in done]
        mode = "a"
        print(f"Resume: {before - len(aligned)} rows already done, {len(aligned)} remaining.")
 
    print(f"Source={args.source} backend={args.backend} model={args.judge_model}")
    print(f"raw={raw_path.name} parsed={parsed_path.name} -> {output_path.name}")
    print(f"Rows to grade: {len(aligned)}  (parse-failure rows recovered: {recovered})")
 
    written = errors = 0
    with output_path.open(mode, encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
            futures = {ex.submit(judge_one, it, backend, args.judge_input): it for it in aligned}
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"judge/{args.source}"):
                try:
                    row = fut.result()
                except Exception as e:
                    errors += 1
                    it = futures[fut]
                    row = dict(it["parsed"])
                    row.update({"correct": None, "CORRECTNESS": None, "judge_verdict": None,
                                "grading_method": "judge_error", "judge_error": str(e),
                                "source_parse_success": it["source_parse_success"]})
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                written += 1
 
    # Summary over the full output file (covers resumed runs).
    allrows = read_jsonl(output_path)
    scored = [r for r in allrows if r.get("CORRECTNESS") in (0, 1)]
    correct = sum(1 for r in scored if r.get("CORRECTNESS") == 1)
    uncertain = sum(1 for r in allrows if r.get("correct") is None
                    and r.get("grading_method") not in ("judge_error",))
    acc = correct / len(scored) if scored else float("nan")
    print("\nJudge run complete.")
    print(f"  rows written this run : {written}  (worker/API errors: {errors})")
    print(f"  scored rows in output : {len(scored)}")
    print(f"  uncertain (excluded)  : {uncertain}")
    print(f"  accuracy over decided : {acc:.4f}")
    print(f"  output                : {output_path}")
 
 
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Merged LLM-as-judge runner (validated + robust).")
    p.add_argument("--source", choices=["ar", "dlm"], required=True)
    p.add_argument("--backend", choices=["openrouter", "openai", "ollama", "mock"], default="mock")
    p.add_argument("--judge-model", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--judge-input", choices=["raw", "extracted"], default="raw",
                   help="what to send the judge for free-text: 'raw' = full raw generation "
                        "(judge doubles as parser, higher parse rate; Tommy's recommendation), "
                        "'extracted' = the cleaned parsed answer only")
    p.add_argument("--concurrency", type=int, default=8, help="parallel requests (use 1-2 for local Ollama)")
    p.add_argument("--max-rows", type=int, default=None, help="cap for smoke tests")
    p.add_argument("--resume", action="store_true", help="append only missing rows")
    p.add_argument("--include-parse-failures", dest="include_parse_failures",
                   action="store_true", default=True,
                   help="judge parse-failed rows too (default on)")
    p.add_argument("--skip-parse-failures", dest="include_parse_failures", action="store_false")
    p.add_argument("--raw", default=None, help="override raw jsonl path")
    p.add_argument("--parsed", default=None, help="override parsed jsonl path")
    p.add_argument("--output", default=None, help="override output jsonl path")
    p.add_argument("--pilot", default=None, help="optional PilotDataset.json for gt recovery")
    return p
 
 
if __name__ == "__main__":
    run(build_parser().parse_args())
 
