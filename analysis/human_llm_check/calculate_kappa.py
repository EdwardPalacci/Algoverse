import csv
import math
from collections import Counter


CSV_PATH = "analysis/human_llm_check/perfect_audit_sheet.csv"
OUTPUT_PATH = "analysis/human_llm_check/audit_results.txt"
POPULATION_SIZE = 15_750
VALID_LABELS = {"correct", "incorrect"}


def wilson_interval(successes, total, z=1.96):
    if total == 0:
        return (0.0, 0.0)
    phat = successes / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2.0 * total)) / denom
    half_width = (
        z
        * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total)
        / denom
    )
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def normal_moe_with_fpc(proportion, sample_size, population_size, z=1.96):
    if sample_size == 0 or population_size <= 1:
        return 0.0
    fpc = math.sqrt((population_size - sample_size) / (population_size - 1))
    return z * math.sqrt(proportion * (1.0 - proportion) / sample_size) * fpc


def load_votes(path):
    human_votes = []
    judge_votes = []
    rows = []

    with open(path, "r", encoding="utf-8", newline="") as src:
        reader = csv.reader(src)
        header = None
        for row in reader:
            if not row:
                continue
            if "human_verdict" in row and "judge_verdict" in row:
                header = row
                break

        if header is None:
            raise ValueError("Could not find header row with human_verdict and judge_verdict.")

        h_idx = header.index("human_verdict")
        j_idx = header.index("judge_verdict")
        for row in reader:
            if len(row) <= max(h_idx, j_idx):
                continue
            h = row[h_idx].strip().lower()
            j = row[j_idx].strip().lower()
            if h in VALID_LABELS and j in VALID_LABELS:
                human_votes.append(h)
                judge_votes.append(j)
                rows.append(dict(zip(header, row)))

    return human_votes, judge_votes, rows


def compute_kappa(human_votes, judge_votes):
    total = len(human_votes)
    matches = sum(h == j for h, j in zip(human_votes, judge_votes))
    observed_agreement = matches / total

    expected_agreement = 0.0
    categories = sorted(set(human_votes + judge_votes))
    for category in categories:
        expected_agreement += (
            human_votes.count(category) / total
        ) * (
            judge_votes.count(category) / total
        )

    if expected_agreement == 1.0:
        kappa = 1.0 if observed_agreement == 1.0 else 0.0
    else:
        kappa = (observed_agreement - expected_agreement) / (1.0 - expected_agreement)

    return matches, observed_agreement, expected_agreement, kappa


def main():
    human_votes, judge_votes, rows = load_votes(CSV_PATH)
    total = len(human_votes)
    if total == 0:
        raise SystemExit("No rows found with both human and judge labels.")

    matches, observed, expected, kappa = compute_kappa(human_votes, judge_votes)
    low, high = wilson_interval(matches, total)
    conservative_moe = normal_moe_with_fpc(0.5, total, POPULATION_SIZE)
    observed_moe = normal_moe_with_fpc(observed, total, POPULATION_SIZE)

    mismatches = [
        row
        for row, h, j in zip(rows, human_votes, judge_votes)
        if h != j
    ]

    lines = [
        "Human--LLM Judge Agreement Audit",
        "=" * 40,
        f"Rows audited: {total}",
        f"Sample frame: {POPULATION_SIZE} judged generations (7 models x 2,250 generations)",
        "Sampling method: simple random sample over saved judged generations",
        "Original random seed: not recorded for the completed audit sample",
        "Frozen sample artifact: analysis/human_llm_check/raw_200_sample.jsonl",
        "",
        f"Observed agreement: {observed:.4f} ({matches}/{total})",
        f"Wilson 95% CI for agreement: [{low:.4f}, {high:.4f}]",
        f"Normal-approximation MOE at p=0.5 with finite-population correction: ±{conservative_moe:.4f}",
        f"Normal-approximation MOE at observed agreement with finite-population correction: ±{observed_moe:.4f}",
        f"Expected agreement by chance: {expected:.4f}",
        f"Cohen's kappa: {kappa:.4f}",
        "",
        f"Human label counts: {dict(Counter(human_votes))}",
        f"LLM judge label counts: {dict(Counter(judge_votes))}",
        f"Disagreements: {len(mismatches)}",
    ]

    if mismatches:
        lines.append("")
        lines.append("Disagreement rows:")
        for row in mismatches:
            lines.append(
                "- "
                f"{row.get('question_id')} | {row.get('dataset')} | "
                f"{row.get('model_name')} | judge={row.get('judge_verdict')} | "
                f"human={row.get('human_verdict')}"
            )

    text = "\n".join(lines) + "\n"
    print(text)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        out.write(text)


if __name__ == "__main__":
    main()
