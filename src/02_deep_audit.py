import os
import pandas as pd
import numpy as np


def deep_audit():

    raw_dir = os.path.join("data", "raw")

    files = [
        "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
        "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv"
    ]

    for filename in files:

        path = os.path.join(raw_dir, filename)

        if not os.path.exists(path):
            print(f"[-] Missing: {filename}")
            continue

        print("\n" + "=" * 80)
        print(filename)
        print("=" * 80)

        total_rows = 0
        total_nan = 0
        total_inf = 0

        exact_duplicate_count = 0

        label_counts = {}

        constant_check = None

        previous_columns = None

        # Process file in chunks
        for chunk_number, chunk in enumerate(
            pd.read_csv(
                path,
                chunksize=100_000,
                low_memory=False
            ),
            start=1
        ):

            chunk.columns = chunk.columns.str.strip()

            # ---------------------------------------------------------
            # Basic information
            # ---------------------------------------------------------

            if previous_columns is None:
                previous_columns = chunk.columns.tolist()

            total_rows += len(chunk)

            # ---------------------------------------------------------
            # NaN count
            # ---------------------------------------------------------

            total_nan += int(chunk.isna().sum().sum())

            # ---------------------------------------------------------
            # Infinite value count
            # ---------------------------------------------------------

            numeric_chunk = chunk.select_dtypes(include=[np.number])

            if not numeric_chunk.empty:
                total_inf += int(
                    np.isinf(numeric_chunk.to_numpy()).sum()
                )

            # ---------------------------------------------------------
            # Label counts
            # ---------------------------------------------------------

            if "Label" in chunk.columns:

                counts = chunk["Label"].value_counts(dropna=False)

                for label, count in counts.items():

                    label_counts[label] = (
                        label_counts.get(label, 0) + int(count)
                    )

            # ---------------------------------------------------------
            # Constant feature check
            # ---------------------------------------------------------

            if constant_check is None:

                numeric_cols = numeric_chunk.columns.tolist()

                constant_check = {
                    col: set()
                    for col in numeric_cols
                }

            # Track unique values, capped at 2
            # We only need to know whether a column has
            # 0/1/>1 unique values.

            for col in constant_check:

                values = chunk[col].dropna().unique()

                if len(constant_check[col]) < 2:

                    for value in values[:2]:
                        constant_check[col].add(value)

        # -------------------------------------------------------------
        # Results
        # -------------------------------------------------------------

        print(f"\nTotal Rows: {total_rows:,}")

        print(f"Total NaN values found: {total_nan:,}")

        print(f"Total Infinite values found: {total_inf:,}")

        print("\nLabel counts:")

        for label, count in sorted(
            label_counts.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            print(f"  {str(label):30} {count:,}")

        # -------------------------------------------------------------
        # Constant columns
        # -------------------------------------------------------------

        constant_cols = [
            col
            for col, values in constant_check.items()
            if len(values) <= 1
        ]

        print(
            "\nConstant/Zero-Variance numeric columns:"
        )

        if constant_cols:
            for col in constant_cols:
                print(f"  - {col}")
        else:
            print("  None")

        # -------------------------------------------------------------
        # Exact duplicate check
        # -------------------------------------------------------------

        print("\nChecking exact duplicate rows...")

        duplicate_count = 0

        seen_rows = set()

        for chunk in pd.read_csv(
            path,
            chunksize=100_000,
            low_memory=False
        ):

            chunk.columns = chunk.columns.str.strip()

            for row in chunk.itertuples(index=False, name=None):

                row_key = tuple(row)

                if row_key in seen_rows:
                    duplicate_count += 1
                else:
                    seen_rows.add(row_key)

        print(f"Exact duplicate rows: {duplicate_count:,}")

        # -------------------------------------------------------------
        # Feature-only duplicate check
        # -------------------------------------------------------------

        print(
            "\nFeature-only duplicate analysis will be performed "
            "after we determine the final modeling features."
        )


if __name__ == "__main__":
    deep_audit()