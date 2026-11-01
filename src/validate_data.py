"""Register the raw dataset by checking its column contract and printing a summary."""

import json
import sys

import pandas as pd

from src.config import DATA_PATH, EXPECTED_COLUMNS, REPORT_DIR, TARGET


def validate_dataset(path=DATA_PATH):
    """Load the registered CSV, enforce the expected columns, and summarise it.

    Returns a JSON-serialisable summary. Raises ValueError when the column
    contract is broken so the workflow fails before later jobs run.
    """
    frame = pd.read_csv(path)
    missing = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
    unexpected = [column for column in frame.columns if column not in EXPECTED_COLUMNS]
    if missing or unexpected:
        raise ValueError(
            "Dataset column contract failed. "
            f"Missing: {missing or 'none'}. Unexpected: {unexpected or 'none'}."
        )

    summary = {
        "dataset_path": str(path),
        "row_count": int(frame.shape[0]),
        "column_count": int(frame.shape[1]),
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missing_values": {column: int(count) for column, count in frame.isna().sum().items()},
        "duplicate_rows": int(frame.duplicated().sum()),
        "target_counts": {
            str(label): int(count)
            for label, count in frame[TARGET].value_counts(dropna=False).items()
        },
        "purchase_rate": float(frame[TARGET].mean()),
    }
    return frame, summary


def print_summary(summary):
    """Print the registration summary in a form a workflow log can capture."""
    print("Dataset registration summary")
    print(f"Path: {summary['dataset_path']}")
    print(f"Rows: {summary['row_count']}")
    print(f"Columns: {summary['column_count']}")
    print("Expected columns: present")
    print("Missing values by column:")
    for column, count in summary["missing_values"].items():
        print(f"  {column}: {count}")
    print("Target counts (ProdTaken):")
    for label, count in summary["target_counts"].items():
        print(f"  {label}: {count}")
    print(f"Purchase rate: {summary['purchase_rate']:.4f}")
    print(f"Duplicate rows: {summary['duplicate_rows']}")


def main():
    _, summary = validate_dataset()
    print_summary(summary)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "data_registration_summary.json"
    report_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {report_path}")
    return summary


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Data registration failed: {error}", file=sys.stderr)
        sys.exit(1)
