# Repository Handoff

This repository contains the saved experimental outputs, judge results, paper figures, tables, captions, and scripts for the AR versus DLM verbalized-confidence evaluation.

## Migration Safety

The current repository state is safe to move to a new instance after `git pull` and `git push` both succeed and the worktree is clean. The raw generations, parsed generations, LLM-as-judge outputs, paper tables, figure CSVs, figure PNGs, captions, and research notes are tracked in Git.

Do not rely on the current machine for secrets. Copy or recreate `.env` separately on the new instance. `.env` is intentionally ignored by Git.

No tracked file is currently larger than 50 MB. The repository is about 107 MB on disk, with the largest output files around 4 MB. If a future run produces files or file groups above normal Git size, store those outputs in a dataset repository such as Hugging Face and commit a manifest with checksums and download instructions.

## Instance Size

For the current saved outputs and paper-asset regeneration, a CPU instance is sufficient. The scripts that regenerate tables and figures read saved JSONL and CSV files; they do not require GPU memory.

For new model generation through OpenRouter, GPU memory on the local instance is not required because inference is remote. Disk usage remains small unless local model caches are introduced.

For local DLM inference, choose the instance by the largest model you expect to host:

- 7B to 8B models: at least 24 GB VRAM for standard inference; 16 GB may work only with quantization and conservative context length.
- DiffusionGemma 26B class models: prefer at least 48 GB VRAM with quantization or memory-efficient serving; use 80 GB VRAM if running full precision or larger batches.
- Disk: 500 GB is a practical minimum for several local model checkpoints, tokenizer caches, logs, and rerun outputs. Use 1 TB if keeping multiple precision variants or intermediate checkpoints.

## Environment

There is currently no `pyproject.toml` or `uv.lock` in the repository, so there is no UV lockfile to sync. If the project moves to UV, add `pyproject.toml` and commit `uv.lock` before migrating.

The scripts use standard-library Python plus common packages. Install at least:

```bash
python3 -m pip install tqdm
```

Some generation paths may also require provider SDKs listed in script comments, for example `openai`, `anthropic`, and `datasets`. OpenRouter calls in the current scripts use HTTP requests directly.

Expected secret for remote generation and LLM judging:

```bash
OPENROUTER_API_KEY=...
```

Some older comments refer to other provider keys, but the current AR, DLM, and judge paths can use OpenRouter.

## Raw Outputs

Raw model generations:

- `ar_models/model_outputs/raw_by_model/*.jsonl`
- `dlm_models/model_outputs/raw_by_model/*.jsonl`

Parsed model generations:

- `ar_models/model_outputs/parsed_by_model/*.jsonl`
- `dlm_models/model_outputs/parsed_by_model/*.jsonl`

LLM-as-judge results:

- `analysis/llm_as_judge/results/ar/by_model/*/all_datasets.jsonl`
- `analysis/llm_as_judge/results/ar/by_model/*/by_dataset/*.jsonl`
- `analysis/llm_as_judge/results/dlm/by_model/*/all_datasets.jsonl`
- `analysis/llm_as_judge/results/dlm/by_model/*/by_dataset/*.jsonl`

Paper-ready outputs:

- `paper_assets/tables/*.csv`
- `paper_assets/tables/quality_control/*.csv`
- `paper_assets/figures/pngs/*.png`
- `paper_assets/figures/csvs/*.csv`
- `paper_assets/figures/captions/*.txt`

## Main Commands

Regenerate paper tables, figures, captions, and manifests from saved judge outputs:

```bash
python3 analysis/generate_paper_assets.py
```

Check whether generated figure data remain aligned with saved model and judge outputs:

```bash
python3 analysis/check_review_artifacts.py
```

Run anomaly checks and regenerate quality-control tables:

```bash
python3 analysis/run_anomaly_checks.py
```

Generate a deterministic human-audit sample from judged generations:

```bash
python3 analysis/human_llm_check/generate_audit.py
```

Compute human versus LLM judge agreement for the completed audit sheet:

```bash
python3 analysis/human_llm_check/calculate_kappa.py
```

Run the canonical LLM-as-judge pipeline after model generation:

```bash
python3 analysis/llm_as_judge/llm_as_judge.py --source ar
python3 analysis/llm_as_judge/llm_as_judge.py --source dlm
```

Run AR generation through the configured models:

```bash
python3 ar_models/run_model.py --datasets pilot --n-samples 3
```

Run DLM generation through the configured models:

```bash
python3 dlm_models/run_dlm_model.py --datasets pilot --n-samples 3
```

## Dependency Order

The experimental dependency chain is:

1. `data/PilotDataset.json`
2. `ar_models/run_model.py` and `dlm_models/run_dlm_model.py`
3. raw generation JSONL files
4. parsed generation JSONL files
5. `analysis/llm_as_judge/llm_as_judge.py`
6. judge result JSONL files
7. `analysis/generate_paper_assets.py` and `analysis/run_anomaly_checks.py`
8. paper tables, quality-control tables, figure CSVs, figure PNGs, captions, and research notes
9. `analysis/check_review_artifacts.py` for consistency checks

See `documentation/repository_handoff_map.csv` for a row-level script and artifact map.
