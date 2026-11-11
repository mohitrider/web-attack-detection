from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports"
CHUNK_SIZE = 100_000

FILES = {
    "Thursday": "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday": "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
}

LABELS = [
    "Benign",
    "Brute Force -Web",
    "Brute Force -XSS",
    "SQL Injection",
]

hashes_by_day = {}
labels_by_day = {}
feature_columns = None

for day, filename in FILES.items():
    path = DATA_DIR / filename
    print(f"Scanning {day}...", flush=True)

    day_hashes = []
    day_labels = []

    for chunk in pd.read_csv(
        path, chunksize=CHUNK_SIZE, low_memory=False
    ):
        if feature_columns is None:
            feature_columns = [
                c for c in chunk.columns
                if c not in ("Timestamp", "Label")
            ]

        # Exclude Timestamp and Label from the feature signature.
        feature_hashes = pd.util.hash_pandas_object(
            chunk[feature_columns], index=False
        ).to_numpy(dtype=np.uint64)

        day_hashes.append(feature_hashes)
        day_labels.append(
            chunk["Label"].astype("string")
            .fillna("<MISSING>")
            .to_numpy()
        )

    hashes_by_day[day] = np.concatenate(day_hashes)
    labels_by_day[day] = np.concatenate(day_labels)

    print(
        f"{day}: {len(hashes_by_day[day]):,} rows processed",
        flush=True,
    )

unique_hashes = {
    day: np.unique(values)
    for day, values in hashes_by_day.items()
}

shared_signatures = np.intersect1d(
    unique_hashes["Thursday"],
    unique_hashes["Friday"],
    assume_unique=True,
)

print(
    f"\nDistinct feature signatures present on both days: "
    f"{len(shared_signatures):,}"
)

lines = [
    "CROSS-DAY FEATURE-PATTERN OVERLAP",
    "=" * 50,
    f"Distinct shared signatures: {len(shared_signatures):,}",
    "",
]

for day in FILES:
    other_day = "Friday" if day == "Thursday" else "Thursday"

    # Count rows whose feature signature appears on the other day.
    overlap_mask = np.isin(
        hashes_by_day[day],
        unique_hashes[other_day],
    )

    total = len(overlap_mask)
    overlap_count = int(overlap_mask.sum())

    lines.extend([
        f"{day.upper()}",
        f"  Total flows: {total:,}",
        f"  Flows whose signature occurs on {other_day}: "
        f"{overlap_count:,} ({100 * overlap_count / total:.4f}%)",
        "  Overlap by label:",
    ])

    for label in LABELS:
        label_mask = labels_by_day[day] == label
        label_total = int(label_mask.sum())
        label_overlap = int((label_mask & overlap_mask).sum())

        percentage = (
            100 * label_overlap / label_total
            if label_total else 0.0
        )

        lines.append(
            f"    {label}: {label_overlap:,}/{label_total:,} "
            f"({percentage:.2f}%)"
        )

    lines.append("")

lines.extend([
    "NOTE: A repeated feature signature is not automatically leakage.",
    "It may represent recurring traffic. These counts help us decide",
    "how to interpret cross-day results and disclose overlap.",
])

REPORT_DIR.mkdir(parents=True, exist_ok=True)
output_path = REPORT_DIR / "05_cross_day_overlap.txt"
output_path.write_text("\n".join(lines), encoding="utf-8")

print("\n" + "\n".join(lines))
print(f"\nReport saved to: {output_path}")