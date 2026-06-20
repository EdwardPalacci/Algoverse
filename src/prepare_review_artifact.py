#!/usr/bin/env python3
"""Build all review-ready benchmark artifacts.

Read this file first. It is intentionally short: each step calls a module with
one job, so mentees can open that module to learn the details.
"""

from __future__ import annotations

from load_generation_data import aligned_rows, load_all_rows, shared_question_ids
from write_documentation import produce_docs, produce_manifest, produce_schema_files, produce_table_captions
from render_figures import produce_figures
from project_paths import ensure_output_dirs
from write_tables import produce_alignment_report, produce_audit_and_cases, produce_tables


def main() -> None:
    ensure_output_dirs()

    all_rows, raw_counts = load_all_rows()
    rows_for_comparison = aligned_rows(all_rows)

    produce_alignment_report(all_rows, rows_for_comparison)
    produce_tables(rows_for_comparison, raw_counts)
    produce_figures(rows_for_comparison)
    produce_table_captions()
    produce_audit_and_cases(rows_for_comparison)
    produce_docs(all_rows, raw_counts)
    produce_schema_files()
    produce_manifest()

    print(f"generated review artifacts from {len(rows_for_comparison)} aligned parsed generations")
    print(f"shared question_id values: {len(shared_question_ids(all_rows))}")
    print("review materials written to docs/, fig_tabs/, metrics/, and src/")


if __name__ == "__main__":
    main()
