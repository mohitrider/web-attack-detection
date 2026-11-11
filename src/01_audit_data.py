from pathlib import Path
import pandas as pd


DATA_DIR = Path("data/raw")

FILES = [
    DATA_DIR / "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
    DATA_DIR / "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
]


for file_path in FILES:
    print("\n" + "=" * 80)
    print(file_path.name)
    print("=" * 80)

    # Read only the header first
    columns = pd.read_csv(file_path, nrows=0).columns.tolist()

    print(f"Number of columns: {len(columns)}")
    print("\nColumns:")
    for i, column in enumerate(columns, start=1):
        print(f"{i:3}. {column}")

    # Find the label column
    label_columns = [
        column for column in columns
        if column.strip().lower() in {"label", "class"}
    ]

    print("\nPossible label columns:", label_columns)

    if not label_columns:
        print("No obvious label column found.")
        continue

    label_column = label_columns[0]

    # Count labels without loading the whole file
    label_counts = {}
    total_rows = 0

    for chunk in pd.read_csv(
        file_path,
        usecols=[label_column],
        chunksize=100_000,
    ):
        total_rows += len(chunk)

        counts = chunk[label_column].value_counts(dropna=False)

        for label, count in counts.items():
            label_counts[label] = label_counts.get(label, 0) + int(count)

    print(f"\nTotal rows: {total_rows}")

    print("\nLabel counts:")
    for label, count in sorted(
        label_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"{str(label):30} {count:,}")