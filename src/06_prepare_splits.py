
from collections import Counter
from pathlib import Path
import gc
import gzip

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

CHUNK_SIZE = 100_000
BENIGN_TRAIN_SIZE = 50_000
VALIDATION_GROUP_MODULUS = 5
SEED = 2026
BENIGN_CANDIDATE_FRACTION = 0.10

FILES = {
    "Thursday": "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday": "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
}

EXPECTED_LABELS = {
    "Benign",
    "Brute Force -Web",
    "Brute Force -XSS",
    "SQL Injection",
}


def count_labels(frame):
    return Counter(frame["Label"].astype(str))


def prepare_day(day, filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    print(f"\nPreparing {day}...", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    header = pd.read_csv(path, nrows=0).columns.tolist()

    if "Timestamp" not in header or "Label" not in header:
        raise ValueError("Expected Timestamp and Label columns.")

    feature_columns = [
        column for column in header
        if column not in ("Timestamp", "Label")
    ]

    print(
        f"Model feature columns before cleaning: {len(feature_columns)}",
        flush=True,
    )

    train_attack_chunks = []
    benign_candidate_chunks = []

    validation_path = (
        OUTPUT_DIR / f"{day.lower()}_validation.csv.gz"
    )

    validation_label_counts = Counter()
    total_rows = 0
    validation_rows = 0
    chunk_number = 0

    # Write the validation data incrementally, avoiding a large
    # in-memory concatenation of all validation rows.
    with gzip.open(
        validation_path,
        "wt",
        encoding="utf-8",
        newline="",
    ) as validation_file:

        first_validation_chunk = True

        for chunk in pd.read_csv(
            path,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):
            chunk_number += 1
            total_rows += len(chunk)

            labels = (
                chunk["Label"]
                .astype("string")
                .str.strip()
                .fillna("<MISSING>")
            )

            unexpected = set(labels.unique()) - EXPECTED_LABELS

            if unexpected:
                raise ValueError(
                    f"Unexpected labels found: {unexpected}"
                )

            # Convert all model features to numeric values.
            # NaN and infinity will be handled later in the
            # model-specific preprocessing pipeline.
            numeric_features = (
                chunk[feature_columns]
                .apply(pd.to_numeric, errors="coerce")
                .astype(np.float64)
            )

            # Use normalised numeric features to form a consistent
            # signature across CSV chunks.
            hashes = pd.util.hash_pandas_object(
                numeric_features,
                index=False,
            ).to_numpy(dtype=np.uint64)

            # All identical feature signatures go into the same
            # partition. Roughly 1 in 5 signatures goes to validation.
            validation_mask = (
                hashes % np.uint64(VALIDATION_GROUP_MODULUS)
            ) == 0

            partition = numeric_features.copy()
            partition["Label"] = labels.to_numpy()
            partition["Target"] = (
                labels.ne("Benign")
                .astype(np.uint8)
                .to_numpy()
            )

            # Metadata only: never use FeatureHash for prediction.
            partition["FeatureHash"] = hashes.astype(str)

            validation_chunk = partition.loc[
                validation_mask
            ].copy()

            if not validation_chunk.empty:
                validation_chunk.to_csv(
                    validation_file,
                    index=False,
                    header=first_validation_chunk,
                )
                first_validation_chunk = False

                validation_rows += len(validation_chunk)
                validation_label_counts.update(
                    count_labels(validation_chunk)
                )

            # Sample training candidates only from the non-validation
            # groups. Keep every attack row in this training portion.
            training_pool = partition.loc[
                ~validation_mask
            ]

            attack_rows = training_pool.loc[
                training_pool["Target"] == 1
            ]

            if not attack_rows.empty:
                train_attack_chunks.append(attack_rows.copy())

            benign_rows = training_pool.loc[
                training_pool["Target"] == 0
            ]

            if not benign_rows.empty:
                candidates = benign_rows.sample(
                    frac=BENIGN_CANDIDATE_FRACTION,
                    random_state=SEED + chunk_number,
                )

                if not candidates.empty:
                    benign_candidate_chunks.append(candidates)

            if chunk_number % 5 == 0:
                print(
                    f"  Processed {total_rows:,} rows...",
                    flush=True,
                )

    print("Selecting the benign training sample...", flush=True)

    if not benign_candidate_chunks:
        raise RuntimeError("No benign training candidates found.")

    benign_candidates = pd.concat(
        benign_candidate_chunks,
        ignore_index=True,
    )

    if len(benign_candidates) < BENIGN_TRAIN_SIZE:
        raise RuntimeError(
            f"Only {len(benign_candidates)} benign candidates "
            f"available; need {BENIGN_TRAIN_SIZE}."
        )

    benign_sample = benign_candidates.sample(
        n=BENIGN_TRAIN_SIZE,
        random_state=SEED,
    )

    if train_attack_chunks:
        attack_sample = pd.concat(
            train_attack_chunks,
            ignore_index=True,
        )
    else:
        raise RuntimeError(
            "No attack examples were retained for training."
        )

    training_data = pd.concat(
        [benign_sample, attack_sample],
        ignore_index=True,
    ).sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

    train_path = OUTPUT_DIR / f"{day.lower()}_train.csv.gz"

    training_data.to_csv(
        train_path,
        index=False,
        compression="gzip",
    )

    train_label_counts = count_labels(training_data)

    summary = [
        f"DAY: {day}",
        f"Source rows: {total_rows:,}",
        f"Model features: {len(feature_columns)}",
        f"Validation rows: {validation_rows:,}",
        f"Training rows: {len(training_data):,}",
        "",
        "TRAINING LABEL COUNTS:",
    ]

    summary.extend(
        f"  {label}: {count:,}"
        for label, count in sorted(train_label_counts.items())
    )

    summary.extend([
        "",
        "VALIDATION LABEL COUNTS:",
    ])

    summary.extend(
        f"  {label}: {count:,}"
        for label, count in sorted(
            validation_label_counts.items()
        )
    )

    summary.extend([
        "",
        f"Training file: {train_path}",
        f"Validation file: {validation_path}",
        "FeatureHash is metadata only, not a model feature.",
        "Timestamp is excluded from model features.",
    ])

    print("\n".join(summary))

    benign_candidate_chunks.clear()
    train_attack_chunks.clear()
    del benign_candidates, benign_sample, attack_sample
    del training_data
    gc.collect()

    return summary


def main():
    all_summaries = [
        "DATA PREPARATION SUMMARY",
        "=" * 55,
        "Training and validation partitions use feature signatures.",
        "The original CSV files have not been modified.",
        "",
    ]

    for day, filename in FILES.items():
        summary = prepare_day(day, filename)
        all_summaries.extend(summary)
        all_summaries.append("")

    report_path = REPORT_DIR / "06_split_summary.txt"
    report_path.write_text(
        "\n".join(all_summaries),
        encoding="utf-8",
    )

    print(f"\nSummary saved to: {report_path}")
    print(
        "\nPreparation complete. Review the attack counts "
        "before starting model training."
    )


if __name__ == "__main__":
    main()
