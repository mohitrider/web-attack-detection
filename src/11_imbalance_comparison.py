from pathlib import Path
import gc

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
# This script is intended to be saved inside the project's src/ folder.
# If it is run from the downloaded file in another folder, set PROJECT_ROOT
# to your local project directory before running it.
PROJECT_ROOT = Path.home() / "Documents" / "Python Projects" / "ML" / "ES687 Project"
if (ROOT / "data" / "processed").exists():
    PROJECT_ROOT = ROOT

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"

SEED = 2026
CHUNK_SIZE = 100_000

FILES = {
    "Thursday": "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday": "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
}
ATTACK_LABELS = [
    "Brute Force -Web",
    "Brute Force -XSS",
    "SQL Injection",
]
VALID_LABELS = {"Benign", *ATTACK_LABELS}
METADATA_COLUMNS = {"Timestamp", "Label", "Target", "FeatureHash"}

STRATEGIES = [
    "Current undersampling + class weights",
    "Current undersampling, no class weights",
    "10:1 benign-to-attack, no class weights",
    "1:1 benign-to-attack, no class weights",
]


def load_frame(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    frame = pd.read_csv(path, low_memory=False)
    frame["Label"] = frame["Label"].astype(str).str.strip()
    unexpected = set(frame["Label"].unique()) - VALID_LABELS
    if unexpected:
        raise ValueError(f"Unexpected labels in {path}: {unexpected}")
    frame["Target"] = (frame["Label"] != "Benign").astype(np.uint8)
    return frame


def to_numeric_features(frame, columns):
    X = frame[columns].apply(pd.to_numeric, errors="coerce")
    return X.replace([np.inf, -np.inf], np.nan)


def select_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        raise ValueError("No validation thresholds available.")
    p, r = precision[:-1], recall[:-1]
    f1_values = np.divide(
        2 * p * r,
        p + r,
        out=np.zeros_like(p),
        where=(p + r) > 0,
    )
    return float(thresholds[int(np.argmax(f1_values))])


def make_models(n_negative, n_positive, use_class_weights):
    positive_weight = n_negative / max(n_positive, 1) if use_class_weights else 1.0
    class_weight = "balanced" if use_class_weights else None
    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=1.0,
                class_weight=class_weight,
                max_iter=2000,
                random_state=SEED,
            )),
        ]),
        "Linear SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearSVC(
                C=1.0,
                class_weight=class_weight,
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
    if model_name == "Linear SVM":
        return np.asarray(model.decision_function(X), dtype=np.float64)
    return np.asarray(model.predict_proba(X)[:, 1], dtype=np.float64)


def build_strategy_frame(train, strategy, seed):
    benign = train.loc[train["Label"] == "Benign"]
    attacks = train.loc[train["Label"] != "Benign"]
    n_attacks = len(attacks)

    if n_attacks == 0:
        raise ValueError("Training set contains no attacks.")

    current_strategies = {
        "Current undersampling + class weights",
        "Current undersampling, no class weights",
    }

    if strategy in current_strategies:
        # Preserve the original prepared training data.
        selected = train.copy()

    elif strategy == "10:1 benign-to-attack, no class weights":
        n_benign = min(10 * n_attacks, len(benign))
        selected = pd.concat(
            [
                benign.sample(n=n_benign, random_state=seed),
                attacks,
            ],
            ignore_index=True,
        )

    elif strategy == "1:1 benign-to-attack, no class weights":
        n_benign = min(n_attacks, len(benign))
        selected = pd.concat(
            [
                benign.sample(n=n_benign, random_state=seed),
                attacks,
            ],
            ignore_index=True,
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Shuffle only the newly undersampled datasets.
    if strategy not in current_strategies:
        selected = selected.sample(
            frac=1,
            random_state=seed,
        )

    selected = selected.reset_index(drop=True)

    info = {
        "Train_Rows": len(selected),
        "Train_Benign": int((selected["Label"] == "Benign").sum()),
        "Train_Attacks": int((selected["Label"] != "Benign").sum()),
        "Class_Weights": (
            strategy == "Current undersampling + class weights"
        ),
    }

    return selected, info


def run_direction(source_day, test_day):
    print("\n" + "=" * 70)
    print(f"IMBALANCE COMPARISON: train {source_day} -> test {test_day}")
    print("=" * 70, flush=True)

    train = load_frame(PROCESSED_DIR / f"{source_day.lower()}_train.csv.gz")
    validation = load_frame(PROCESSED_DIR / f"{source_day.lower()}_validation.csv.gz")
    test_path = RAW_DIR / FILES[test_day]

    feature_columns = [c for c in train.columns if c not in METADATA_COLUMNS]
    X_train_full = to_numeric_features(train, feature_columns)

    # Feature filtering is learned from source-day training data only.
    constant_columns = [
        c for c in feature_columns if X_train_full[c].nunique(dropna=True) <= 1
    ]
    features = [c for c in feature_columns if c not in constant_columns]
    X_val = to_numeric_features(validation, features)
    y_val = validation["Target"].to_numpy(dtype=np.uint8)

    if len(np.unique(y_val)) != 2:
        raise ValueError("Validation data must contain both classes.")

    models = {}
    thresholds = {}
    strategy_counts = {}

    for strategy_index, strategy in enumerate(STRATEGIES):
        weighted = strategy == "Current undersampling + class weights"
        strategy_train, info = build_strategy_frame(
            train, strategy, SEED + strategy_index
        )
        strategy_counts[strategy] = info
        y_train = strategy_train["Target"].to_numpy(dtype=np.uint8)
        if len(np.unique(y_train)) != 2:
            raise ValueError(f"Both classes required for {strategy}")

        X_train = to_numeric_features(strategy_train, features)
        print(
            f"\nStrategy: {strategy}\n"
            f"  rows={info['Train_Rows']:,}; benign={info['Train_Benign']:,}; "
            f"attacks={info['Train_Attacks']:,}; features={len(features)}",
            flush=True,
        )

        model_set = make_models(
            info["Train_Benign"], info["Train_Attacks"], weighted
        )

        for model_name, model in model_set.items():
            print(f"  Fitting {model_name}...", flush=True)
            model.fit(X_train, y_train)
            val_scores = get_scores(model_name, model, X_val)
            threshold = select_threshold(y_val, val_scores)
            key = (strategy, model_name)
            models[key] = model
            thresholds[key] = threshold
            print(f"    Validation F1 threshold: {threshold:.8g}", flush=True)
            del val_scores

        del X_train, strategy_train, model_set, y_train
        gc.collect()

    del train, X_train_full, validation, X_val
    gc.collect()

    print(f"\nScoring full test day {test_day} in chunks...", flush=True)
    y_parts, label_parts = [], []
    score_parts = {key: [] for key in models}
    rows_read = 0

    for chunk in pd.read_csv(test_path, chunksize=CHUNK_SIZE, low_memory=False):
        labels = chunk["Label"].astype(str).str.strip()
        unexpected = set(labels.unique()) - VALID_LABELS
        if unexpected:
            raise ValueError(f"Unexpected test labels: {unexpected}")
        y_chunk = (labels != "Benign").to_numpy(dtype=np.uint8)
        X_chunk = to_numeric_features(chunk, features)
        y_parts.append(y_chunk)
        label_parts.append(labels.to_numpy())

        for key, model in models.items():
            model_name = key[1]
            score_parts[key].append(get_scores(model_name, model, X_chunk))

        rows_read += len(chunk)
        if rows_read % (CHUNK_SIZE * 5) == 0:
            print(f"  Scored {rows_read:,} rows...", flush=True)

    y_test = np.concatenate(y_parts)
    labels_test = np.concatenate(label_parts)
    output_rows = []

    for strategy in STRATEGIES:
        for model_name in ("Logistic Regression", "Linear SVM", "XGBoost"):
            key = (strategy, model_name)
            scores = np.concatenate(score_parts[key])
            threshold = thresholds[key]
            pred = (scores >= threshold).astype(np.uint8)
            tn, fp, fn, tp = confusion_matrix(
                y_test, pred, labels=[0, 1]
            ).ravel()
            info = strategy_counts[strategy]

            row = {
                "Direction": f"{source_day}_to_{test_day}",
                "Strategy": strategy,
                "Model": model_name,
                **info,
                "Feature_Count": len(features),
                "Threshold": threshold,
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall": recall_score(y_test, pred, zero_division=0),
                "F1": f1_score(y_test, pred, zero_division=0),
                "Average_Precision": average_precision_score(y_test, scores),
                "ROC_AUC": roc_auc_score(y_test, scores),
                "Accuracy": accuracy_score(y_test, pred),
                "Balanced_Accuracy": balanced_accuracy_score(y_test, pred),
                "Test_Flows": len(y_test),
                "Test_Attacks": int(y_test.sum()),
                "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            }

            for attack in ATTACK_LABELS:
                mask = labels_test == attack
                row[f"Recall_{attack}"] = (
                    float(np.mean(pred[mask] == 1)) if mask.any() else np.nan
                )

            output_rows.append(row)
            print(
                f"{strategy} | {model_name}: F1={row['F1']:.4f}, "
                f"AP={row['Average_Precision']:.4f}, FP={fp}, FN={fn}",
                flush=True,
            )
            del scores, pred

    del models, score_parts, y_test, labels_test, y_parts, label_parts
    gc.collect()
    return output_rows


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    results.extend(run_direction("Thursday", "Friday"))
    results.extend(run_direction("Friday", "Thursday"))

    output_path = RESULTS_DIR / "imbalance_comparison.csv"
    results_frame = pd.DataFrame(results)
    results_frame.to_csv(output_path, index=False)

    display_columns = [
        "Direction", "Strategy", "Model", "Train_Rows",
        "Train_Benign", "Train_Attacks", "Precision", "Recall", "F1",
        "Average_Precision", "FP", "FN",
    ]
    print("\nFINAL IMBALANCE COMPARISON")
    print(results_frame[display_columns].to_string(index=False))
    print(f"\nResults saved to: {output_path}")

    report = [
        "CLASS-IMBALANCE SENSITIVITY EXPERIMENT",
        "=" * 60,
        "The validation data retained its natural class distribution.",
        "Thresholds were chosen on validation data only.",
        "Final test sets are the full, naturally imbalanced opposite-day files.",
        "The 10:1 and 1:1 strategies also reduce training-set size; interpret",
        "them as combined training-composition/undersampling comparisons.",
        "",
        results_frame[display_columns].to_string(index=False),
    ]
    report_path = REPORTS_DIR / "11_imbalance_comparison.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
