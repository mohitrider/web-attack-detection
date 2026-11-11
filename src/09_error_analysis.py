from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"

DIRECTIONS = [
    "Thursday_to_Friday",
    "Friday_to_Thursday",
]

MODELS = [
    "Logistic_Regression",
    "Linear_SVM",
    "XGBoost",
]

ATTACKS = [
    "Brute Force -Web",
    "Brute Force -XSS",
    "SQL Injection",
]


def main():
    report_lines = [
        "MODEL ERROR ANALYSIS",
        "=" * 65,
        "",
    ]

    rows = []

    for direction in DIRECTIONS:
        path = RESULTS / f"predictions_{direction}.csv.gz"

        if not path.exists():
            raise FileNotFoundError(path)

        data = pd.read_csv(path, low_memory=False)
        y = data["Target"].to_numpy(dtype=np.uint8)
        labels = data["Label"].astype(str).to_numpy()

        report_lines.extend([
            f"DIRECTION: {direction}",
            "-" * 65,
            f"Total test flows: {len(data):,}",
            f"Attack flows: {int(y.sum()):,}",
            f"Benign flows: {int((y == 0).sum()):,}",
            "",
        ])

        for model in MODELS:
            pred_col = f"Prediction_{model}"

            if pred_col not in data.columns:
                raise ValueError(
                    f"Missing column {pred_col} in {path}"
                )

            pred = data[pred_col].to_numpy(dtype=np.uint8)

            fp_mask = (y == 0) & (pred == 1)
            fn_mask = (y == 1) & (pred == 0)
            tp = int(((y == 1) & (pred == 1)).sum())
            tn = int(((y == 0) & (pred == 0)).sum())
            fp = int(fp_mask.sum())
            fn = int(fn_mask.sum())

            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0.0
            )

            report_lines.extend([
                f"MODEL: {model}",
                f"  TP={tp:,}; TN={tn:,}; FP={fp:,}; FN={fn:,}",
                f"  Precision={precision:.4f}; "
                f"Recall={recall:.4f}; F1={f1:.4f}",
                "",
                "  Attack-specific errors:",
            ])

            for attack in ATTACKS:
                mask = labels == attack
                count = int(mask.sum())

                if count:
                    misses = int(((pred == 0) & mask).sum())
                    attack_recall = (count - misses) / count
                    report_lines.append(
                        f"    {attack}: {count:,} flows; "
                        f"{misses:,} missed; "
                        f"recall={attack_recall:.4f}"
                    )

            report_lines.extend([
                "",
                "  Benign flows incorrectly flagged as attacks:",
                f"    {fp:,} false positives",
                "",
            ])

            rows.append({
                "Direction": direction,
                "Model": model.replace("_", " "),
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
            })

        report_lines.append("")

    REPORTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    report_path = REPORTS / "09_error_analysis.txt"
    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    summary_path = RESULTS / "error_analysis_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)

    print("\n".join(report_lines))
    print(f"\nReport saved to: {report_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()