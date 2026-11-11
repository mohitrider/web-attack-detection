from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports"
CHUNK_SIZE = 100_000

FILES = [
    (
        "Thursday",
        "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
        0,
    ),
    (
        "Friday",
        "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
        1,
    ),
]

ATTACK_LABELS = [
    "Brute Force -Web",
    "Brute Force -XSS",
    "SQL Injection",
]

LABEL_CODES = {
    "Benign": 0,
    "Brute Force -Web": 1,
    "Brute Force -XSS": 2,
    "SQL Injection": 3,
}

label_counts = {day: Counter() for day, _, _ in FILES}
quality = {}
attack_quality = {}
hash_parts = []
label_parts = []
day_parts = []
feature_columns = None


def new_quality_counter():
    return Counter()


for day, filename, day_code in FILES:
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    print(f"\nReading {day}: {filename}", flush=True)

    quality[day] = new_quality_counter()
    attack_quality[day] = {
        label: new_quality_counter() for label in ATTACK_LABELS
    }

    for chunk_number, chunk in enumerate(
        pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False),
        start=1,
    ):
        if "Label" not in chunk.columns:
            raise ValueError(f"'Label' column missing in {filename}")

        if feature_columns is None:
            # Timestamp is used for day-based analysis, not as a model feature.
            # Label is the target, not a feature.
            feature_columns = [
                column for column in chunk.columns
                if column not in ("Timestamp", "Label")
            ]
            print(
                f"Feature columns examined: {len(feature_columns)}",
                flush=True,
            )

        labels = chunk["Label"].astype("string").fillna("<MISSING>")
        label_counts[day].update(
            {str(k): int(v) for k, v in labels.value_counts().items()}
        )

        features = chunk[feature_columns]
        numeric = features.apply(pd.to_numeric, errors="coerce")
        values = numeric.to_numpy(dtype=np.float64)

        missing = np.isnan(values)
        infinite = np.isinf(values)

        # Detect values that were present but could not be parsed numerically.
        parse_failures = missing & features.notna().to_numpy()

        q = quality[day]
        q["rows"] += len(chunk)
        q["missing_cells"] += int(missing.sum())
        q["infinite_cells"] += int(infinite.sum())
        q["numeric_parse_failure_cells"] += int(parse_failures.sum())

        attack_mask = labels.isin(ATTACK_LABELS).to_numpy()
        attack_missing = missing[attack_mask]
        attack_infinite = infinite[attack_mask]

        q["attack_rows"] += int(attack_mask.sum())
        q["attack_rows_with_missing"] += int(
            attack_missing.any(axis=1).sum()
        )
        q["attack_rows_with_infinite"] += int(
            attack_infinite.any(axis=1).sum()
        )
        q["attack_rows_with_any_invalid"] += int(
            (attack_missing | attack_infinite).any(axis=1).sum()
        )
        q["attack_missing_cells"] += int(attack_missing.sum())
        q["attack_infinite_cells"] += int(attack_infinite.sum())

        for label in ATTACK_LABELS:
            label_mask = labels.eq(label).to_numpy()
            label_missing = missing[label_mask]
            label_infinite = infinite[label_mask]

            item = attack_quality[day][label]
            item["rows"] += int(label_mask.sum())
            item["rows_with_missing"] += int(
                label_missing.any(axis=1).sum()
            )
            item["rows_with_infinite"] += int(
                label_infinite.any(axis=1).sum()
            )

        # Hash the flow features excluding Timestamp and Label.
        # This lets us investigate matching feature patterns across days
        # without loading both entire CSV files into memory.
        hashes = pd.util.hash_pandas_object(
            features, index=False
        ).to_numpy(dtype=np.uint64)

        codes = (
            labels.map(LABEL_CODES)
            .fillna(255)
            .to_numpy(dtype=np.uint8)
        )

        hash_parts.append(hashes)
        label_parts.append(codes)
        day_parts.append(np.full(len(chunk), day_code, dtype=np.uint8))

        if chunk_number % 5 == 0:
            print(
                f"  Processed approximately "
                f"{chunk_number * CHUNK_SIZE:,} rows...",
                flush=True,
            )

    print(f"Finished {day}.", flush=True)


# Compare feature signatures across both days.
print("\nChecking repeated feature patterns...", flush=True)

all_hashes = np.concatenate(hash_parts)
all_labels = np.concatenate(label_parts)
all_days = np.concatenate(day_parts)

order = np.argsort(all_hashes, kind="mergesort")
sorted_hashes = all_hashes[order]
sorted_labels = all_labels[order]
sorted_days = all_days[order]

starts = np.flatnonzero(
    np.r_[True, sorted_hashes[1:] != sorted_hashes[:-1]]
)
ends = np.r_[starts[1:], len(sorted_hashes)]
group_sizes = ends - starts
repeated_groups = np.flatnonzero(group_sizes > 1)

extra_rows = int((group_sizes[repeated_groups] - 1).sum())
cross_day_groups = 0
different_label_groups = 0
benign_attack_conflicts = 0

for group_index in repeated_groups:
    start = starts[group_index]
    end = ends[group_index]

    group_labels = np.unique(sorted_labels[start:end])
    group_days = np.unique(sorted_days[start:end])

    if len(group_days) > 1:
        cross_day_groups += 1

    if len(group_labels) > 1:
        different_label_groups += 1

    if np.any(group_labels == 0) and np.any(group_labels > 0):
        benign_attack_conflicts += 1


# Prepare and save the audit report.
report = [
    "TARGETED DATA AUDIT",
    "=" * 60,
    f"Total rows processed: {len(all_hashes):,}",
    f"Feature columns excluding Timestamp and Label: "
    f"{len(feature_columns)}",
    "",
]

for day, _, _ in FILES:
    report.append(f"{day.upper()} RESULTS")
    report.append("-" * 40)

    for label, count in label_counts[day].most_common():
        report.append(f"  {label}: {count:,}")

    q = quality[day]
    report.extend([
        f"  Total rows: {q['rows']:,}",
        f"  Missing feature values: {q['missing_cells']:,}",
        f"  Infinite feature values: {q['infinite_cells']:,}",
        f"  Numeric parsing failures: "
        f"{q['numeric_parse_failure_cells']:,}",
        f"  Attack rows: {q['attack_rows']:,}",
        f"  Attack rows with missing values: "
        f"{q['attack_rows_with_missing']:,}",
        f"  Attack rows with infinite values: "
        f"{q['attack_rows_with_infinite']:,}",
        f"  Attack rows with either issue: "
        f"{q['attack_rows_with_any_invalid']:,}",
        f"  Missing cells within attack rows: "
        f"{q['attack_missing_cells']:,}",
        f"  Infinite cells within attack rows: "
        f"{q['attack_infinite_cells']:,}",
        "",
        "Attack-specific quality:",
    ])

    for label in ATTACK_LABELS:
        item = attack_quality[day][label]
        report.append(
            f"  {label}: {item['rows']:,} rows; "
            f"{item['rows_with_missing']:,} with missing values; "
            f"{item['rows_with_infinite']:,} with infinite values"
        )

    report.append("")

report.extend([
    "REPEATED FEATURE-PATTERN CHECK",
    "-" * 40,
    "Timestamp and Label were excluded from feature signatures.",
    f"Repeated feature-pattern groups: {len(repeated_groups):,}",
    f"Rows beyond the first in repeated groups: {extra_rows:,}",
    f"Repeated groups found on both days: {cross_day_groups:,}",
    f"Repeated groups with different labels: "
    f"{different_label_groups:,}",
    f"Groups containing both Benign and an attack label: "
    f"{benign_attack_conflicts:,}",
    "",
    "IMPORTANT: Repeated feature patterns are a diagnostic, not an",
    "automatic reason to delete rows. Investigate the findings before",
    "deciding how duplicates should be handled.",
])

REPORT_DIR.mkdir(parents=True, exist_ok=True)
output_path = REPORT_DIR / "03_targeted_audit.txt"
output_path.write_text("\n".join(report) + "\n", encoding="utf-8")

print("\n" + "\n".join(report))
print(f"\nReport saved to: {output_path}")
