from __future__ import annotations

import json

from load_generation_data import shared_question_ids
from file_io import write_csv, write_text
from project_paths import DOCS_DIR, ECE_BINS, FIG_DIR, HIGH_CONFIDENCE_THRESHOLD, PILOT_DATA, TABLE_DIR


def produce_docs(rows: list[dict], raw_counts: dict[str, int]) -> None:
    datasets = sorted({row["dataset"] for row in rows})
    models = sorted({row["model_id"] for row in rows})
    conditions = sorted({row["prompt_condition"] for row in rows})
    n_shared_questions = len(shared_question_ids(rows))

    spec = f"""# Benchmark Specification

## Benchmark Objective

Evaluate whether verbalized confidence tracks answer correctness under controlled prompting, comparing an autoregressive language model with a diffusion language model.

## Construct

The measured construct is verbalized confidence calibration: the correspondence between a model's stated confidence probability and empirical correctness.

## Unit of Evaluation

One model generation for one `question_id`, prompt condition, and sample index.

Comparative metric tables and figures use the shared AR/DLM question-ID intersection ({n_shared_questions} questions). Source-level exclusions are listed in `analysis/metrics/data_alignment_exclusions.csv`.

## Datasets

{', '.join(datasets)}

## Model Identifiers

{', '.join(models)}

## Model Family Labels

`AR` denotes an autoregressive language model. `DLM` denotes a diffusion language model.

## Prompt Conditions

{', '.join(conditions)}

## Confidence Scale

Models report a verbalized confidence probability in `[0, 1]`. Values outside this interval fail quality control.

## Correctness Grader

The saved parsed outputs do not contain human or LLM-judge correctness labels. This artifact therefore uses a deterministic automatic grader: last-number matching for numeric answers, option-letter or option-text matching for multiple-choice answers, and normalized string exact/containment matching for short answers. These labels are suitable for artifact execution checks and preliminary benchmarking, but they should be replaced or audited for final claims.

## Metric Definitions

Accuracy is the fraction of automatically correct generations. Mean confidence is the arithmetic mean of verbalized confidence. Expected calibration error (ECE) uses {ECE_BINS} equal-width confidence bins and weights each absolute bin accuracy-confidence gap by bin frequency. Brier score is the mean squared error between confidence and correctness. Area under the receiver operating characteristic curve (AUROC) is the Mann-Whitney probability that a correct generation receives higher confidence than an incorrect generation, with half credit for ties. High-confidence wrong rate is the fraction of all evaluated generations that are incorrect with confidence >= {HIGH_CONFIDENCE_THRESHOLD:.2f}. Parse success is parsed rows divided by raw rows for each model source.

## Quality Control Checks

The artifact checks question-ID alignment by model family, duplicate full evaluation keys, dataset validity, model identifier validity, family-label validity, prompt-condition validity, confidence range, and binary rule-based correctness labels.

## Known Limitations

DLM outputs are incomplete relative to the full AR question set, so comparative outputs use the shared question-ID intersection. Correctness labels are deterministic automatic labels rather than manual adjudications. Prompt sensitivity is therefore reported as an available-case analysis over parsed rows within the aligned question set. Some short-answer datasets require semantic grading beyond normalized string matching.
"""
    write_text(DOCS_DIR / "benchmark_spec.md", spec)

    readme = """# Review Artifact

This artifact contains saved AR and DLM generations, benchmark specifications, metric tables, figures, captions, schema documentation, and audit templates for the verbalized confidence calibration benchmark.

Comparative tables and figures are computed on the shared AR/DLM question-ID intersection. Raw-source exclusions are documented in `analysis/metrics/data_alignment_exclusions.csv`.

## Reproduction

From the repository root, run:

```bash
python3 analysis/prepare_review_artifact.py
```

The script reads `ar_models/model_outputs/ar_parsed_generations.jsonl`, `ar_models/model_outputs/ar_raw_generations.jsonl`, `dlm_models/model_outputs/dlm_parsed_generations.jsonl`, `dlm_models/model_outputs/dlm_raw_generations.jsonl`, and `data/PilotDataset.json`. It writes tables to `paper_assets/tables/`, figures to `paper_assets/figures/`, quality-control reports to `analysis/metrics/`, and release metadata to `documentation/research_notes/`.

## Raw and Parsed Outputs

AR outputs are stored in `ar_models/model_outputs/`. DLM outputs are stored in `dlm_models/model_outputs/`.

## Confidence Parsing

Each parsed generation contains `confidence`, a verbalized probability normalized to `[0, 1]`.

## Correctness Grading

The current artifact uses deterministic automatic grading because saved parsed outputs do not include adjudicated correctness labels. Numeric answers use last-number matching; multiple-choice answers use option-letter or option-text matching; short answers use normalized string matching and containment.

## Prompt Conditions

Prompt conditions are encoded in the `condition` field as `neutral`, `cautious`, or `overconfident`.

## Not Included

This package does not include API keys, private links, author-identifying metadata, or completed manual adjudication labels.

## Anonymity

The review materials are prepared without author names, institution names, private URLs, API keys, or workspace-specific absolute paths.
"""
    write_text(DOCS_DIR / "artifact_readme.md", readme)

    data_manifest = [
        {"file_path": "data/PilotDataset.json", "description": "Benchmark item source file", "row_count": len(json.loads(PILOT_DATA.read_text()))},
        {"file_path": "ar_models/model_outputs/ar_raw_generations.jsonl", "description": "AR raw model outputs", "row_count": raw_counts["ar_models/model_outputs/ar_parsed_generations.jsonl"]},
        {"file_path": "ar_models/model_outputs/ar_parsed_generations.jsonl", "description": "AR parsed generations", "row_count": sum(1 for row in rows if row["model_family"] == "AR")},
        {"file_path": "dlm_models/model_outputs/dlm_raw_generations.jsonl", "description": "DLM raw model outputs", "row_count": raw_counts["dlm_models/model_outputs/dlm_parsed_generations.jsonl"]},
        {"file_path": "dlm_models/model_outputs/dlm_parsed_generations.jsonl", "description": "DLM parsed generations", "row_count": sum(1 for row in rows if row["model_family"] == "DLM")},
    ]
    write_csv(DOCS_DIR / "data_manifest.csv", data_manifest, ["file_path", "description", "row_count"])


def produce_schema_files() -> None:
    schemas = {
        "schema_benchmark_items.json": {
            "type": "object",
            "required": ["question_id", "dataset", "split", "question", "gold_answer", "answer_type", "domain", "source", "license"],
            "properties": {key: {"type": "string"} for key in ["question_id", "dataset", "split", "question", "gold_answer", "answer_type", "domain", "source", "license"]},
        },
        "schema_generations.json": {
            "type": "object",
            "required": ["question_id", "dataset", "model_id", "model_family", "provider", "prompt_id", "prompt_condition", "raw_output", "parsed_answer", "parsed_confidence", "parsed_reasoning", "parse_success", "parse_error_type", "correct_auto", "correct_manual", "grader", "temperature", "max_tokens", "seed", "timestamp"],
            "properties": {
                "parsed_confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                "parse_success": {"type": "boolean"},
                "correct_auto": {"type": ["boolean", "null"]},
                "correct_manual": {"type": ["boolean", "null"]},
            },
        },
        "schema_metrics.json": {
            "type": "object",
            "required": ["model_id", "model_family", "prompt_condition", "dataset", "N", "accuracy", "mean_confidence", "expected_calibration_error", "brier_score", "area_under_roc", "high_confidence_wrong_rate", "parse_success"],
            "properties": {
                "N": {"type": "integer", "minimum": 0},
                "accuracy": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                "mean_confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                "expected_calibration_error": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                "brier_score": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                "area_under_roc": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                "high_confidence_wrong_rate": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                "parse_success": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
            },
        },
    }
    for filename, schema in schemas.items():
        write_text(DOCS_DIR / filename, json.dumps(schema, indent=2) + "\n")


def produce_table_captions() -> None:
    captions = {
        "table_1_caption.txt": "Table 1. Benchmark specification by dataset, model, model family, and prompt condition. N is the number of parsed generations in the aligned comparative analysis set.\n",
        "table_2_caption.txt": "Table 2. Aggregate calibration metrics for the autoregressive language model (AR) and diffusion language model (DLM). Expected calibration error uses 10 equal-width confidence bins; area under the receiver operating characteristic curve is abbreviated as AUROC.\n",
        "table_3_caption.txt": "Table 3. Dataset-level calibration metrics by model family. Metrics are computed on parsed generations from the shared question-ID analysis set.\n",
        "table_4_caption.txt": "Table 4. Prompt-condition calibration metrics. Expected calibration error and high-confidence wrong rate quantify sensitivity to cautious, neutral, and overconfident prompting.\n",
        "table_5_caption.txt": "Table 5. Representative cases selected for qualitative audit. Correctness reflects deterministic automatic grading and should be adjudicated before being used as final qualitative evidence.\n",
    }
    for filename, caption in captions.items():
        write_text(TABLE_DIR / filename, caption)


def produce_manifest() -> None:
    definitions = [
        ("data/PilotDataset.json", "data", "Benchmark item source file", "Benchmark", True, True),
        ("ar_models/model_outputs/ar_raw_generations.jsonl", "raw_output", "Autoregressive raw generations", "Artifact", True, True),
        ("ar_models/model_outputs/ar_parsed_generations.jsonl", "parsed_output", "Autoregressive parsed generations", "Artifact", True, True),
        ("dlm_models/model_outputs/dlm_raw_generations.jsonl", "raw_output", "Diffusion language model raw generations", "Artifact", True, True),
        ("dlm_models/model_outputs/dlm_parsed_generations.jsonl", "parsed_output", "Diffusion language model parsed generations", "Artifact", True, True),
        ("analysis/metrics/data_alignment_report.csv", "audit", "Data alignment and schema quality-control checks", "Benchmark", True, True),
        ("analysis/metrics/data_alignment_exclusions.csv", "audit", "Question IDs excluded from comparative aligned analysis", "Benchmark", True, True),
        ("analysis/metrics/manual_grading_audit.csv", "audit", "Manual grading audit template with selected examples", "Analysis", True, False),
        ("paper_assets/tables/table_1_benchmark_specification.csv", "metric_table", "Benchmark specification table", "Benchmark", True, True),
        ("paper_assets/tables/table_2_aggregate_metrics.csv", "metric_table", "Aggregate AR versus DLM metrics", "Results", True, True),
        ("paper_assets/tables/table_3_per_dataset_metrics.csv", "metric_table", "Dataset-level metrics", "Results", True, True),
        ("paper_assets/tables/table_4_prompt_condition_metrics.csv", "metric_table", "Prompt-condition metrics", "Results", True, True),
        ("paper_assets/tables/table_5_representative_failure_cases.csv", "metric_table", "Representative qualitative cases", "Analysis", True, False),
        ("paper_assets/tables/table_1_caption.txt", "caption", "Benchmark specification table caption", "Benchmark", True, True),
        ("paper_assets/tables/table_2_caption.txt", "caption", "Aggregate metric table caption", "Results", True, True),
        ("paper_assets/tables/table_3_caption.txt", "caption", "Dataset-level metric table caption", "Results", True, True),
        ("paper_assets/tables/table_4_caption.txt", "caption", "Prompt-condition metric table caption", "Results", True, True),
        ("paper_assets/tables/table_5_caption.txt", "caption", "Representative case table caption", "Analysis", True, True),
        ("paper_assets/figures/figure_1_reliability_diagram.png", "figure", "Reliability diagram", "Results", True, True),
        ("paper_assets/figures/figure_1_reliability_diagram_data.csv", "figure_data", "Reliability diagram plotted values", "Results", True, True),
        ("paper_assets/figures/figure_1_caption.txt", "caption", "Reliability diagram caption", "Results", True, True),
        ("paper_assets/figures/figure_2_confidence_by_correctness.png", "figure", "Confidence distribution by correctness", "Results", True, True),
        ("paper_assets/figures/figure_2_confidence_by_correctness_data.csv", "figure_data", "Confidence distribution plotted values", "Results", True, True),
        ("paper_assets/figures/figure_2_caption.txt", "caption", "Confidence distribution caption", "Results", True, True),
        ("paper_assets/figures/figure_2_2_confidence_by_correctness_neutral.png", "figure", "Neutral-prompt confidence distribution by correctness", "Results", True, True),
        ("paper_assets/figures/figure_2_2_confidence_by_correctness_neutral_data.csv", "figure_data", "Neutral-prompt confidence distribution plotted values", "Results", True, True),
        ("paper_assets/figures/figure_2_2_caption.txt", "caption", "Neutral-prompt confidence distribution caption", "Results", True, True),
        ("paper_assets/figures/figure_3_prompt_sensitivity.png", "figure", "Prompt sensitivity figure", "Results", True, True),
        ("paper_assets/figures/figure_3_prompt_sensitivity_data.csv", "figure_data", "Prompt sensitivity plotted values", "Results", True, True),
        ("paper_assets/figures/figure_3_caption.txt", "caption", "Prompt sensitivity caption", "Results", True, True),
        ("documentation/research_notes/artifact_manifest.csv", "documentation", "Artifact file manifest", "Artifact", True, True),
        ("documentation/research_notes/benchmark_spec.md", "documentation", "Benchmark specification", "Benchmark", True, True),
        ("documentation/research_notes/artifact_readme.md", "documentation", "Artifact README", "Artifact", True, True),
        ("documentation/research_notes/data_manifest.csv", "documentation", "Data file manifest", "Artifact", True, True),
        ("documentation/research_notes/schema_benchmark_items.json", "documentation", "Benchmark item schema", "Artifact", True, True),
        ("documentation/research_notes/schema_generations.json", "documentation", "Generation schema", "Artifact", True, True),
        ("documentation/research_notes/schema_metrics.json", "documentation", "Metric schema", "Artifact", True, True),
        ("analysis/prepare_review_artifact.py", "code", "Artifact generation driver", "Artifact", True, True),
        ("analysis/check_review_artifacts.py", "code", "Artifact validation script", "Artifact", True, True),
        ("analysis/load_generation_data.py", "code", "Data loading and alignment helpers", "Artifact", True, True),
        ("analysis/write_documentation.py", "code", "Documentation and manifest helpers", "Artifact", True, True),
        ("analysis/render_figures.py", "code", "Figure rendering and plotted-data helpers", "Artifact", True, True),
        ("analysis/grade_answers.py", "code", "Deterministic grading helpers", "Artifact", True, True),
        ("analysis/file_io.py", "code", "CSV and JSONL helpers", "Artifact", True, True),
        ("analysis/compute_metrics.py", "code", "Metric formula helpers", "Artifact", True, True),
        ("analysis/project_paths.py", "code", "Shared paths and constants", "Artifact", True, True),
        ("analysis/write_tables.py", "code", "Table and audit helpers", "Artifact", True, True),
    ]
    rows = [
        {
            "file_path": file_path,
            "artifact_type": artifact_type,
            "description": description,
            "paper_section": section,
            "anonymous": str(anonymous).lower(),
            "required_for_reproduction": str(required).lower(),
        }
        for file_path, artifact_type, description, section, anonymous, required in definitions
    ]
    write_csv(DOCS_DIR / "artifact_manifest.csv", rows, ["file_path", "artifact_type", "description", "paper_section", "anonymous", "required_for_reproduction"])
