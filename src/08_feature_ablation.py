
from pathlib import Path
import gc

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

SEED = 2026
CHUNK_SIZE = 100_000

FILES = {
    "Thursday": "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday": "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
}

ATTACKS = [
    "Brute Force -Web",
    "Brute Force -XSS",
    "SQL Injection",
]

METADATA = {"Label", "Target", "FeatureHash", "Timestamp"}

TIMING_RATE_FEATURES = {
    "Flow Duration", "Flow Byts/s", "Flow Pkts/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std",
    "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std",
    "Bwd IAT Max", "Bwd IAT Min",
    "Fwd Pkts/s", "Bwd Pkts/s",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
}

VARIANTS = {
    "All usable features": set(),
    "Without port and protocol": {"Dst Port", "Protocol"},
    "Without timing and rate features": TIMING_RATE_FEATURES,
}


def load_prepared(path):
    data = pd.read_csv(path, low_memory=False)
    labels = data["Label"].astype(str).str.strip()
    y = (labels != "Benign").to_numpy(dtype=np.uint8)

    features = [
        c for c in data.columns if c not in METADATA
    ]
    X = data[features].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    return X, y, labels.to_numpy(), features


def choose_threshold(y, scores):
    precision, recall, thresholds = precision_recall_curve(y, scores)

    if len(thresholds) == 0:
        raise ValueError("No validation thresholds available.")

    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(
        2 * p * r,
        p + r,
        out=np.zeros_like(p),
        where=(p + r) > 0,
    )
    return float(thresholds[int(np.argmax(f1))])


def make_model(y_train):
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("xgb", XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            scale_pos_weight=negatives / max(positives, 1),
            tree_method="hist",
            n_jobs=4,
            random_state=SEED,
            verbosity=0,
        )),
    ])


def run_direction(source_day, test_day):
    print(f"\n{'=' * 65}")
    print(f"XGBoost ablation: {source_day} -> {test_day}")
    print(f"{'=' * 65}", flush=True)

    train_path = PROCESSED / f"{source_day.lower()}_train.csv.gz"
    val_path = PROCESSED / f"{source_day.lower()}_validation.csv.gz"
    test_path = RAW / FILES[test_day]

    X_train, y_train, _, features = load_prepared(train_path)
    X_val, y_val, _, _ = load_prepared(val_path)

    # Remove constant/all-missing features based on training data only.
    constant = [
        c for c in features
        if X_train[c].nunique(dropna=True) <= 1
    ]
    usable = [c for c in features if c not in constant]

    # A shared natural-distribution test set is scored in chunks.
    scores = {}
    thresholds = {}
    validation_ap = {}

    for variant, removed in VARIANTS.items():
        selected = [c for c in usable if c not in removed]

        if not selected:
            raise ValueError(f"No features remain for {variant}")

        print(
            f"\nTraining {variant} ({len(selected)} features)...",
            flush=True,
        )

        model = make_model(y_train)
        model.fit(X_train[selected], y_train)

        val_scores = model.predict_proba(X_val[selected])[:, 1]
        thresholds[variant] = choose_threshold(y_val, val_scores)
        validation_ap[variant] = average_precision_score(
            y_val, val_scores
        )

        scores[variant] = {
            "model": model,
            "selected": selected,
            "parts": [],
        }

        print(
            f"  Validation AP: {validation_ap[variant]:.4f}; "
            f"threshold: {thresholds[variant]:.8g}",
            flush=True,
        )

        del val_scores

    del X_train, X_val
    gc.collect()

    print(f"\nScoring complete {test_day} file...", flush=True)

    y_parts = []
    label_parts = []
    rows_scored = 0

    for chunk in pd.read_csv(
        test_path,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        labels = chunk["Label"].astype(str).str.strip()
        y_chunk = (labels != "Benign").to_numpy(dtype=np.uint8)

        # Convert feature values consistently; imputer was fitted on
        # the source-day training partition only.
        y_parts.append(y_chunk)
        label_parts.append(labels.to_numpy())

        for variant in VARIANTS:
            selected = scores[variant]["selected"]
            X_chunk = chunk[selected].apply(
                pd.to_numeric, errors="coerce"
            )
            X_chunk = X_chunk.replace(
                [np.inf, -np.inf], np.nan
            )

            predictions_score = scores[variant]["model"].predict_proba(
                X_chunk
            )[:, 1]

            scores[variant]["parts"].append(predictions_score)

        rows_scored += len(chunk)

        if rows_scored % (CHUNK_SIZE * 5) == 0:
            print(f"  Scored {rows_scored:,} rows...", flush=True)

    y_test = np.concatenate(y_parts)
    labels_test = np.concatenate(label_parts)
    rows = []

    for variant in VARIANTS:
        test_scores = np.concatenate(scores[variant]["parts"])
        threshold = thresholds[variant]
        pred = (test_scores >= threshold).astype(np.uint8)

        tn, fp, fn, tp = confusion_matrix(
            y_test, pred, labels=[0, 1]
        ).ravel()

        result = {
            "Direction": f"{source_day}_to_{test_day}",
            "Variant": variant,
            "Feature_Count": len(scores[variant]["selected"]),
            "Threshold": threshold,
            "Validation_Average_Precision": validation_ap[variant],
            "Test_Precision": precision_score(
                y_test, pred, zero_division=0
            ),
            "Test_Recall": recall_score(
                y_test, pred, zero_division=0
            ),
            "Test_F1": f1_score(
                y_test, pred, zero_division=0
            ),
            "Test_Average_Precision": average_precision_score(
                y_test, test_scores
            ),
            "Test_ROC_AUC": roc_auc_score(y_test, test_scores),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        }

        for attack in ATTACKS:
            mask = labels_test == attack
            result[f"Recall_{attack}"] = (
                float(np.mean(pred[mask] == 1))
                if mask.sum() else np.nan
            )

        rows.append(result)
        print(
            f"{variant}: F1={result['Test_F1']:.4f}, "
            f"PR-AUC={result['Test_Average_Precision']:.4f}, "
            f"FP={fp}, FN={fn}",
            flush=True,
        )

        del test_scores

    del scores, y_test, labels_test, y_parts, label_parts
    gc.collect()
    return rows


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    all_results = []

    all_results.extend(run_direction("Thursday", "Friday"))
    all_results.extend(run_direction("Friday", "Thursday"))

    output = RESULTS / "xgboost_feature_ablation.csv"

    # Write a new file; preserve the original model comparison.
    pd.DataFrame(all_results).to_csv(output, index=False)

    print(f"\nAblation results saved to: {output}")
    print(pd.DataFrame(all_results)[[
        "Direction", "Variant", "Feature_Count",
        "Test_Precision", "Test_Recall", "Test_F1",
        "Test_Average_Precision", "FP", "FN",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
