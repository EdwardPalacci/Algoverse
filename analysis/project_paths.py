from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

AR_PARSED = ROOT / "ar_models" / "model_outputs" / "ar_parsed_generations.jsonl"
AR_RAW = ROOT / "ar_models" / "model_outputs" / "ar_raw_generations.jsonl"
DLM_PARSED = ROOT / "dlm_models" / "model_outputs" / "dlm_parsed_generations.jsonl"
DLM_RAW = ROOT / "dlm_models" / "model_outputs" / "dlm_raw_generations.jsonl"
PILOT_DATA = ROOT / "data" / "PilotDataset.json"

DATA_DIR = ROOT / "data"
AR_OUTPUTS_DIR = ROOT / "ar_models" / "model_outputs"
DLM_OUTPUTS_DIR = ROOT / "dlm_models" / "model_outputs"
METRICS_DIR = ROOT / "analysis" / "metrics"
FIG_DIR = ROOT / "paper_assets" / "figures"
TABLE_DIR = ROOT / "paper_assets" / "tables"
DOCS_DIR = ROOT / "documentation" / "research_notes"

HIGH_CONFIDENCE_THRESHOLD = 0.90
ECE_BINS = 10


def ensure_output_dirs() -> None:
    """Create folders that hold generated artifact files."""
    for directory in [DATA_DIR, AR_OUTPUTS_DIR, DLM_OUTPUTS_DIR, METRICS_DIR, FIG_DIR, TABLE_DIR, DOCS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
