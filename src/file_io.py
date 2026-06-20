from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    """Read one JSON object per line."""
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Write dictionaries to a CSV with a fixed column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def grouped(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple, list[dict]]:
    """Group rows by one or more dictionary keys."""
    out = defaultdict(list)
    for row in rows:
        out[tuple(row.get(key, "") for key in keys)].append(row)
    return dict(out)
