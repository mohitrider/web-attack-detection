
from pathlib import Path
import gc
import json

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
)

from xgboost import XGBClassifier


# ---------------------------------------------------------
# Project paths and experiment settings
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
REPORT_DIR = ROOT / "reports"

SEED = 2026
TEST_CHUNK_SIZE = 100_000

FILES = {
    "Thursday": "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday": "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
}

ATTACK_LABELS = [
    "Brute Force -Web",
    "Brute Force -XSS",
    "SQL Injection",
]
ALLOWED_LABELS = {"Benign", *ATTACK_LABELS}

METADATA_COLUMNS = {
    "Timestamp", "Label", "Target", "FeatureHash"
}


def load_csv(path):
    """Read a prepared training or validation file."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    frame = pd.read_csv(path, low_memory=False)

    if "Label" not in frame.columns:
        raise ValueError(f"Label column missing in {path}")

    frame["Label"] = frame["Label"].astype(str).str.strip()

    unknown = set(frame["Label"].unique()) - ALLOWED_LABELS
    if unknown:
        raise ValueError(f"Unexpected labels in {path}: {unknown}")

    target = (frame["Label"] != "Benign").astype(np.uint8)

    if "Target" in frame.columns:
        saved_target = frame["Target"].to_numpy(dtype=np.uint8)
        if not np.array_equal(saved_target, target.to_numpy()):
            raise ValueError(f"Target and Label disagree in {path}")

    frame["Target"] = target
    return frame


def numeric_features(frame, columns):
    """Convert features to numbers and treat infinities as missing."""
    missing_columns = set(columns) - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Missing expected feature columns: {missing_columns}"
        )

    X = frame[columns].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X


def select_threshold(y_true, scores):
    """Choose the threshold giving the best validation F1."""
    precision, recall, thresholds = precision_recall_curve(
        y_true, scores
    )

    if len(thresholds) == 0:
        raise ValueError("Unable to choose a validation threshold.")

    p = precision[:-1]
    r = recall[:-1]
    denominator = p + r

    f1_values = np.divide(
        2 * p * r,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )

    return float(thresholds[int(np.argmax(f1_values))])


def make_models(n_negative, n_positive):
    """Create models with fixed, documented initial settings."""
    positive_weight = n_negative / max(n_positive, 1)

    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=2000,
                random_state=SEED,
            )),
        ]),

        "Linear SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearSVC(
                C=1.0,
                class_weight="balanced",
                dual="auto",
                max_iter=15000,
                random_state=SEED,
            )),
        ]),

        "XGBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                objective="binary:logistic",
                eval_metric="aucpr",
                n_estimators=250,
                max_depth=4,
                learning_rate=0.05,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                scale_pos_weight=positive_weight,
                tree_method="hist",
                n_jobs=4,
                random_state=SEED,
                verbosity=0,
            )),
        ]),
    }


def get_scores(model_name, model, X):
    """Return continuous scores for ranking and threshold selection."""
    if model_name == "Linear SVM":
        return model.decision_function(X)

    return model.predict_proba(X)[:, 1]


def evaluate_model(
    direction,
    model_name,
    y_true,
    labels,
    scores,
    threshold,
):
    """Calculate test metrics using a threshold fixed on validation."""
    predictions = (scores >= threshold).astype(np.uint8)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    row = {
        "Direction": direction,
        "Model": model_name,
        "Threshold": threshold,
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "PR_AUC_AveragePrecision": average_precision_score(
            y_true, scores
        ),
        "ROC_AUC": roc_auc_score(y_true, scores),
        "Accuracy": accuracy_score(y_true, predictions),
        "Balanced_Accuracy": balanced_accuracy_score(
            y_true, predictions
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "False_Positive_Rate": (
            float(fp / (fp + tn)) if fp + tn else np.nan
        ),
        "Test_Flows": int(len(y_true)),
        "Test_Attacks": int(np.sum(y_true)),
    }

    # Evaluate detection of each original attack category.
    for attack in ATTACK_LABELS:
        mask = labels == attack

        if mask.sum() == 0:
            row[f"Recall_{attack}"] = np.nan
        else:
            row[f"Recall_{attack}"] = float(
                np.mean(predictions[mask] == 1)
            )

    return row


def run_direction(source_day, test_day):
    """Train on one day and test on the complete opposite day."""
    print("\n" + "=" * 65)
    print(f"EXPERIMENT: {source_day} -> {test_day}")
    print("=" * 65, flush=True)

    train_path = (
        PROCESSED_DIR / f"{source_day.lower()}_train.csv.gz"
    )
    val_path = (
        PROCESSED_DIR / f"{source_day.lower()}_validation.csv.gz"
    )
    test_path = DATA_DIR / FILES[test_day]

    train = load_csv(train_path)
    validation = load_csv(val_path)

    feature_columns = [
        c for c in train.columns
        if c not in METADATA_COLUMNS
    ]

    # Check that the grouped split kept training and validation
    # feature signatures separate.
    train_hashes = set(train["FeatureHash"].astype(str))
    validation_hashes = set(
        validation["FeatureHash"].astype(str)
    )
    shared_hashes = train_hashes & validation_hashes

    if shared_hashes:
        raise ValueError(
            f"{len(shared_hashes)} feature signatures occur in both "
            "training and validation."
        )

    X_train = numeric_features(train, feature_columns)
    X_val = numeric_features(validation, feature_columns)

    y_train = train["Target"].to_numpy(dtype=np.uint8)
    y_val = validation["Target"].to_numpy(dtype=np.uint8)

    if len(np.unique(y_train)) < 2:
        raise ValueError("Training data must contain both classes.")

    if len(np.unique(y_val)) < 2:
        raise ValueError("Validation data must contain both classes.")

    # Select feature columns using training data only.
    constant_columns = [
        c for c in feature_columns
        if X_train[c].nunique(dropna=True) <= 1
    ]
    model_features = [
        c for c in feature_columns
        if c not in constant_columns
    ]

    print(f"Training rows: {len(train):,}", flush=True)
    print(f"Validation rows: {len(validation):,}", flush=True)
    print(f"Initial features: {len(feature_columns)}")
    print(f"Removed constant/all-missing training features: "
          f"{len(constant_columns)}")
    print(f"Features used: {len(model_features)}")
    print(f"Training attacks: {int(y_train.sum()):,}")
    print(f"Validation attacks: {int(y_val.sum()):,}")

    X_train = X_train[model_features]
    X_val = X_val[model_features]

    n_positive = int(y_train.sum())
    n_negative = int(len(y_train) - n_positive)

    models = make_models(n_negative, n_positive)
    fitted_models = {}
    thresholds = {}

    # Fit models using training data. Validation data is used only
    # to select the attack-detection threshold.
    for name, model in models.items():
        print(f"\nTraining {name}...", flush=True)
        model.fit(X_train, y_train)

        val_scores = get_scores(name, model, X_val)
        threshold = select_threshold(y_val, val_scores)

        fitted_models[name] = model
        thresholds[name] = threshold

        print(
            f"  Validation F1-optimal threshold: {threshold:.8g}",
            flush=True,
        )

        del val_scores

    del train, validation, X_train, X_val, models
    gc.collect()

    # Predict the full original test day in chunks. No test labels
    # are used to fit models or choose thresholds.
    print(
        f"\nTesting on the complete {test_day} CSV...",
        flush=True,
    )

    y_test_parts = []
    label_parts = []
    score_parts = {name: [] for name in fitted_models}

    rows_read = 0

    for chunk in pd.read_csv(
        test_path,
        chunksize=TEST_CHUNK_SIZE,
        low_memory=False,
    ):
        labels = chunk["Label"].astype(str).str.strip()

        unknown = set(labels.unique()) - ALLOWED_LABELS
        if unknown:
            raise ValueError(
                f"Unexpected labels in test data: {unknown}"
            )

        y_chunk = (labels != "Benign").to_numpy(dtype=np.uint8)
        X_chunk = numeric_features(chunk, model_features)

        y_test_parts.append(y_chunk)
        label_parts.append(labels.to_numpy())

        for name, model in fitted_models.items():
            scores = get_scores(name, model, X_chunk)
            score_parts[name].append(
                np.asarray(scores, dtype=np.float64)
            )

        rows_read += len(chunk)

        if rows_read % (TEST_CHUNK_SIZE * 5) == 0:
            print(f"  Scored {rows_read:,} rows...", flush=True)

    y_test = np.concatenate(y_test_parts)
    labels_test = np.concatenate(label_parts)
    all_scores = {
        name: np.concatenate(parts)
        for name, parts in score_parts.items()
    }

    direction = f"{source_day}_to_{test_day}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = []

    prediction_frame = pd.DataFrame({
        "Label": labels_test,
        "Target": y_test,
    })

    for name, scores in all_scores.items():
        row = evaluate_model(
            direction,
            name,
            y_test,
            labels_test,
            scores,
            thresholds[name],
        )
        metrics.append(row)

        column_name = name.replace(" ", "_")
        prediction_frame[f"Score_{column_name}"] = scores
        prediction_frame[f"Prediction_{column_name}"] = (
            scores >= thresholds[name]
        ).astype(np.uint8)

    metrics_path = RESULTS_DIR / "model_metrics.csv"

    current_metrics = pd.DataFrame(metrics)

    if metrics_path.exists():
        previous = pd.read_csv(metrics_path)
        previous = previous[
            previous["Direction"] != direction
        ]
        current_metrics = pd.concat(
            [previous, current_metrics],
            ignore_index=True,
        )

    current_metrics.to_csv(metrics_path, index=False)

    prediction_path = (
        RESULTS_DIR / f"predictions_{direction}.csv.gz"
    )
    prediction_frame.to_csv(
        prediction_path,
        index=False,
        compression="gzip",
    )

    threshold_path = (
        RESULTS_DIR / f"thresholds_{direction}.json"
    )
    threshold_path.write_text(
        json.dumps(thresholds, indent=2),
        encoding="utf-8",
    )

    # Save the feature-selection record for reproducibility.
    feature_path = (
        RESULTS_DIR / f"features_{direction}.json"
    )
    feature_path.write_text(
        json.dumps({
            "source_day": source_day,
            "test_day": test_day,
            "features_used": model_features,
            "removed_constant_features": constant_columns,
            "thresholds_selected_on_validation": thresholds,
        }, indent=2),
        encoding="utf-8",
    )

    print("\nTEST RESULTS", flush=True)
    display_columns = [
        "Model",
        "Precision",
        "Recall",
        "F1",
        "PR_AUC_AveragePrecision",
        "ROC_AUC",
        "FP",
        "FN",
    ]
    print(
        current_metrics[
            current_metrics["Direction"] == direction
        ][display_columns].to_string(index=False)
    )

    print(f"\nMetrics saved: {metrics_path}")
    print(f"Predictions saved: {prediction_path}")
    print(f"Thresholds saved: {threshold_path}")

    del fitted_models, prediction_frame, all_scores
    del y_test, labels_test, score_parts
    gc.collect()

    return metrics


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run both directions for cross-day comparison.
    run_direction("Thursday", "Friday")
    run_direction("Friday", "Thursday")

    print("\nBoth cross-day experiments are complete.")
    print(f"Final metrics: {RESULTS_DIR / 'model_metrics.csv'}")


if __name__ == "__main__":
    main()
