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

LABEL_CODES = {
    "Benign": 0,
    "Brute Force -Web": 1,
    "Brute Force -XSS": 2,
    "SQL Injection": 3,
}


def get_feature_columns(path):
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    return [
        column for column in columns
        if column not in ("Timestamp", "Label")
    ]


first_path = DATA_DIR / FILES[0][1]
feature_columns = get_feature_columns(first_path)

# Pass 1: find feature hashes that appear with different labels.
hash_parts = []
label_parts = []

for day, filename, _ in FILES:
    path = DATA_DIR / filename
    print(f"Scanning {day}...", flush=True)

    for chunk in pd.read_csv(
        path, chunksize=CHUNK_SIZE, low_memory=False
    ):
        hashes = pd.util.hash_pandas_object(
            chunk[feature_columns], index=False
        ).to_numpy(dtype=np.uint64)

        codes = (
            chunk["Label"].map(LABEL_CODES)
            .fillna(255)
            .to_numpy(dtype=np.uint8)
        )

        hash_parts.append(hashes)
        label_parts.append(codes)

all_hashes = np.concatenate(hash_parts)
all_labels = np.concatenate(label_parts)

order = np.argsort(all_hashes, kind="mergesort")
sorted_hashes = all_hashes[order]
sorted_labels = all_labels[order]

starts = np.flatnonzero(
    np.r_[True, sorted_hashes[1:] != sorted_hashes[:-1]]
)
ends = np.r_[starts[1:], len(sorted_hashes)]

conflict_hashes = []

for start, end in zip(starts, ends):
    if end - start > 1:
        if len(np.unique(sorted_labels[start:end])) > 1:
            conflict_hashes.append(int(sorted_hashes[start]))

conflict_hashes = np.array(
    conflict_hashes, dtype=np.uint64
)

print(
    f"\nHash signatures with different labels: "
    f"{len(conflict_hashes)}",
    flush=True,
)

# Pass 2: retrieve candidate rows and compare their actual feature values.
details = {
    int(h): {"variants": []}
    for h in conflict_hashes
}

if len(conflict_hashes) > 0:
    for day, filename, _ in FILES:
        path = DATA_DIR / filename
        print(f"Verifying feature values in {day}...", flush=True)

        for chunk in pd.read_csv(
            path, chunksize=CHUNK_SIZE, low_memory=False
        ):
            hashes = pd.util.hash_pandas_object(
                chunk[feature_columns], index=False
            ).to_numpy(dtype=np.uint64)

            positions = np.flatnonzero(
                np.isin(hashes, conflict_hashes)
            )

            for position in positions:
                h = int(hashes[position])
                label = str(chunk.iloc[position]["Label"])

                vector = pd.to_numeric(
                    chunk.iloc[position][feature_columns],
                    errors="coerce",
                ).to_numpy(dtype=np.float64)

                entry = details[h]
                matched = False

                for variant in entry["variants"]:
                    if np.array_equal(
                        vector,
                        variant["vector"],
                        equal_nan=True,
                    ):
                        variant["labels"][label] += 1
                        variant["days"][day] += 1
                        matched = True
                        break

                if not matched:
                    entry["variants"].append({
                        "vector": vector.copy(),
                        "labels": Counter([label]),
                        "days": Counter([day]),
                    })

# Save the findings.
lines = [
    "EXACT FEATURE-VALUE VERIFICATION",
    "=" * 55,
    f"Candidate conflicting hashes: {len(conflict_hashes)}",
    "",
]

for h, entry in details.items():
    variants = entry["variants"]
    lines.append(f"Signature: {h}")
    lines.append(f"Distinct feature vectors verified: {len(variants)}")

    for number, variant in enumerate(variants, start=1):
        lines.append(f"  Feature vector {number}")
        lines.append(f"    Labels: {dict(variant['labels'])}")
        lines.append(f"    Days: {dict(variant['days'])}")

    if len(variants) == 1:
        lines.append(
            "  Finding: matching feature values have different labels."
        )
    else:
        lines.append(
            "  Finding: one hash corresponds to multiple feature vectors; "
            "the hash match alone does not establish a label conflict."
        )

    lines.append("")

REPORT_DIR.mkdir(parents=True, exist_ok=True)
output_path = REPORT_DIR / "04_verified_label_conflicts.txt"
output_path.write_text("\n".join(lines), encoding="utf-8")

print("\n" + "\n".join(lines))
print(f"\nSaved report: {output_path}")