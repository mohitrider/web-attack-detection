
from pathlib import Path
from itertools import zip_longest

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

CHUNK_SIZE = 100_000

FILES = {
    "Thursday": "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday": "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
}

MODELS = {
    "Logistic_Regression": (
        "Score_Logistic_Regression",
        "Prediction_Logistic_Regression",
    ),
    "Linear_SVM": ("Score_Linear_SVM", "Prediction_Linear_SVM"),
    "XGBoost": ("Score_XGBoost", "Prediction_XGBoost"),
}

ATTACKS = [
    "Brute Force -Web",
    "Brute Force -XSS",
    "SQL Injection",
]


def run_direction(source_day, test_day):
    print(f"\n{'=' * 60}")
    print(f"{source_day} training -> {test_day} test")
    print(f"{'=' * 60}", flush=True)

    train_path = PROCESSED / f"{source_day.lower()}_train.csv.gz"
    test_path = RAW / FILES[test_day]
    prediction_path = (
        RESULTS / f"predictions_{source_day}_to_{test_day}.csv.gz"
    )

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("A required data file is missing.")
    if not prediction_path.exists():
        raise FileNotFoundError(prediction_path)

    # These are the exact feature signatures retained in the
    # sampled training set. FeatureHash is metadata, not a predictor.
    train_hashes = set(
        pd.read_csv(
            train_path,
            usecols=["FeatureHash"],
            dtype={"FeatureHash": "string"},
        )["FeatureHash"].dropna().astype(str)
    )

    header = pd.read_csv(test_path, nrows=0).columns.tolist()
    feature_columns = [
        c for c in header if c not in ("Timestamp", "Label")
    ]

    test_iter = pd.read_csv(
        test_path, chunksize=CHUNK_SIZE, low_memory=False
    )
    pred_iter = pd.read_csv(
        prediction_path, chunksize=CHUNK_SIZE, low_memory=False
    )

    storage = {
        group: {
            "y": [],
            "labels": [],
            "scores": {name: [] for name in MODELS},
            "preds": {name: [] for name in MODELS},
        }
        for group in ("Seen in training", "Unseen in training")
    }

    row_count = 0

    for test_chunk, pred_chunk in zip_longest(
        test_iter, pred_iter, fillvalue=None
    ):
        if test_chunk is None or pred_chunk is None:
            raise ValueError(
                "Test and prediction files have different row counts."
            )

        labels = test_chunk["Label"].astype(str).str.strip().to_numpy()
        saved_labels = (
            pred_chunk["Label"].astype(str).str.strip().to_numpy()
        )

        if len(test_chunk) != len(pred_chunk):
            raise ValueError("Test/prediction chunk sizes differ.")

        if not np.array_equal(labels, saved_labels):
            raise ValueError(
                "Prediction rows do not align with original test rows."
            )

        # Match the numerical conversion used when training FeatureHash
        # was created in src/06_prepare_splits.py.
        features = (
            test_chunk[feature_columns]
            .apply(pd.to_numeric, errors="coerce")
            .astype(np.float64)
        )
        hashes = pd.util.hash_pandas_object(
            features, index=False
        ).to_numpy(dtype=np.uint64).astype(str)

        seen_mask = np.isin(
            hashes, np.asarray(list(train_hashes), dtype=str)
        )

        y = (labels != "Benign").astype(np.uint8)

        for group, mask in [
            ("Seen in training", seen_mask),
            ("Unseen in training", ~seen_mask),
        ]:
            if not mask.any():
                continue

            item = storage[group]
            item["y"].append(y[mask])
            item["labels"].append(labels[mask])

            for model, (score_col, pred_col) in MODELS.items():
                item["scores"][model].append(
                    pred_chunk[score_col].to_numpy(dtype=float)[mask]
                )
                item["preds"][model].append(
                    pred_chunk[pred_col].to_numpy(dtype=np.uint8)[mask]
                )

        row_count += len(test_chunk)
        if row_count % (CHUNK_SIZE * 5) == 0:
            print(f"Checked {row_count:,} test rows...", flush=True)

    print(f"Total test rows checked: {row_count:,}")
    results = []

    for group, item in storage.items():
        if not item["y"]:
            continue

        y = np.concatenate(item["y"])
        labels = np.concatenate(item["labels"])
        print(f"\n{group}: {len(y):,} flows; {int(y.sum()):,} attacks")

        for model in MODELS:
            score = np.concatenate(item["scores"][model])
            pred = np.concatenate(item["preds"][model])

            tn, fp, fn, tp = confusion_matrix(
                y, pred, labels=[0, 1]
            ).ravel()

            result = {
                "Direction": f"{source_day}_to_{test_day}",
                "FeaturePatternGroup": group,
                "Model": model,
                "Flows": len(y),
                "Attacks": int(y.sum()),
                "AttackRate": float(y.mean()),
                "Precision": precision_score(
                    y, pred, zero_division=0
                ),
                "Recall": recall_score(y, pred, zero_division=0),
                "F1": f1_score(y, pred, zero_division=0),
                "AveragePrecision": average_precision_score(y, score),
                "ROC_AUC": (
                    roc_auc_score(y, score)
                    if len(np.unique(y)) == 2 else np.nan
                ),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
                "TN": int(tn),
            }

            for attack in ATTACKS:
                mask = labels == attack
                result[f"Recall_{attack}"] = (
                    float(np.mean(pred[mask] == 1))
                    if mask.any() else np.nan
                )

            results.append(result)

            print(
                f"  {model}: precision={result['Precision']:.4f}, "
                f"recall={result['Recall']:.4f}, "
                f"F1={result['F1']:.4f}, "
                f"AP={result['AveragePrecision']:.4f}, "
                f"FP={fp}, FN={fn}"
            )

    return results


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    all_results = []

    all_results.extend(run_direction("Thursday", "Friday"))
    all_results.extend(run_direction("Friday", "Thursday"))

    output = RESULTS / "seen_unseen_feature_metrics.csv"
    pd.DataFrame(all_results).to_csv(output, index=False)

    print(f"\nSaved results to: {output}")


if __name__ == "__main__":
    main()
