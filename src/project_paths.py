from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

AR_PARSED = ROOT / "outputs" / "ar_parsed_generations.jsonl"
AR_RAW = ROOT / "outputs" / "ar_raw_generations.jsonl"
DLM_PARSED = ROOT / "dlm_outputs" / "dlm_parsed_generations.jsonl"
DLM_RAW = ROOT / "dlm_outputs" / "dlm_raw_generations.jsonl"
PILOT_DATA = ROOT / "PilotDataset.json"

DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
METRICS_DIR = ROOT / "metrics"
FIG_DIR = ROOT / "fig_tabs"
DOCS_DIR = ROOT / "docs"

HIGH_CONFIDENCE_THRESHOLD = 0.90
ECE_BINS = 10


def ensure_output_dirs() -> None:
    """Create folders that hold generated artifact files."""
    for directory in [DATA_DIR, OUTPUTS_DIR, METRICS_DIR, FIG_DIR, DOCS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
