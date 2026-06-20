from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation


def normalized_text(value: object) -> str:
    """Lowercase text and remove punctuation for simple answer matching."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def numeric_tokens(value: object) -> list[Decimal]:
    """Extract numbers from a text answer."""
    tokens = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", str(value or ""))
    numbers = []
    for token in tokens:
        try:
            numbers.append(Decimal(token.replace(",", "")))
        except InvalidOperation:
            pass
    return numbers


def choice_map(prompt: str) -> dict[str, str]:
    """Read A/B/C/D answer options from multiple-choice prompts."""
    options = {}
    for line in prompt.splitlines():
        match = re.match(r"^\s*([A-D])\s*[:.)]\s*(.+?)\s*$", line)
        if match:
            options[match.group(1)] = match.group(2)
    return options


def answer_letter(answer: object) -> str | None:
    """Return A/B/C/D if the model answer clearly starts with a choice letter."""
    text = str(answer or "").strip()
    patterns = [
        r"^\s*([A-D])\s*[:.)\s]",
        r"\boption\s+([A-D])\b",
        r"\banswer\s+is\s+([A-D])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    if re.fullmatch(r"[A-D]", text, flags=re.IGNORECASE):
        return text.upper()
    return None


def truth_values(ground_truth: object) -> list[str]:
    if isinstance(ground_truth, list):
        return [str(value) for value in ground_truth]
    if ground_truth is None:
        return []
    return [str(ground_truth)]


def auto_grade(row: dict) -> tuple[bool, str]:
    """Assign a deterministic correctness label.

    This is not manual grading. It is a reproducible placeholder based on
    numeric matching, multiple-choice matching, and simple string matching.
    """
    answer = row.get("answer")
    truths = truth_values(row.get("ground_truth"))
    answer_type = row.get("answer_type")
    prompt = row.get("prompt", "")

    if not answer or not truths:
        return False, "missing_answer_or_reference"

    if answer_type == "numeric":
        answer_nums = numeric_tokens(answer)
        truth_nums = [number for truth in truths for number in numeric_tokens(truth)]
        if answer_nums and truth_nums and answer_nums[-1] == truth_nums[-1]:
            return True, "numeric_last_value_match"
        return False, "numeric_last_value_mismatch"

    options = choice_map(prompt)
    letter = answer_letter(answer)
    if answer_type == "multiple_choice" and letter:
        normalized_truths = {normalized_text(truth) for truth in truths}
        if letter in normalized_truths:
            return True, "choice_letter_match"
        if letter in options and normalized_text(options[letter]) in normalized_truths:
            return True, "choice_option_text_match"

    norm_answer = normalized_text(answer)
    for truth in truths:
        norm_truth = normalized_text(truth)
        if norm_answer == norm_truth:
            return True, "string_exact_match"
        if norm_truth and (norm_truth in norm_answer or norm_answer in norm_truth):
            return True, "string_containment_match"

    return False, "string_mismatch"
