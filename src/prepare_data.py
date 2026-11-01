"""Clean the registered dataset and write stratified train and test splits."""

import json
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CLIP_COLUMNS,
    DATA_PATH,
    DROP_COLUMNS,
    PROCESSED_DIR,
    TARGET,
)
from src.validate_data import validate_dataset


def clean_frame(frame):
    """Apply row-level cleaning that does not depend on the train/test split."""
    cleaned = frame.copy()
    gender_fixes = int((cleaned["Gender"] == "Fe Male").sum())
    cleaned["Gender"] = cleaned["Gender"].replace({"Fe Male": "Female"})
    cleaned = cleaned.drop(columns=DROP_COLUMNS)
    before = len(cleaned)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned, {
        "gender_label_fixes": gender_fixes,
        "dropped_columns": DROP_COLUMNS,
        "duplicate_rows_removed": before - len(cleaned),
        "rows_after_cleaning": int(len(cleaned)),
    }


def clip_outliers(train, test):
    """Clip heavy-tailed numeric columns using bounds learned from train only."""
    bounds = {}
    train_clipped = train.copy()
    test_clipped = test.copy()
    for column in CLIP_COLUMNS:
        lower = float(train_clipped[column].quantile(0.01))
        upper = float(train_clipped[column].quantile(0.99))
        bounds[column] = {"lower": lower, "upper": upper}
        train_clipped[column] = train_clipped[column].clip(lower, upper)
        test_clipped[column] = test_clipped[column].clip(lower, upper)
    return train_clipped, test_clipped, bounds


def prepare_dataset(source=DATA_PATH, output_dir=PROCESSED_DIR, test_size=0.2, random_state=42):
    """Validate, clean, split, and save the modelling tables.

    The split is stratified on ProdTaken so the 19% purchase rate is preserved
    in both sets. Outlier bounds are fit on the training rows and then applied
    to the test rows.
    """
    frame, registration = validate_dataset(source)
    cleaned, cleaning = clean_frame(frame)
    train, test = train_test_split(
        cleaned,
        test_size=test_size,
        random_state=random_state,
        stratify=cleaned[TARGET],
    )
    train, test, clip_bounds = clip_outliers(train, test)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    summary = {
        "source": str(source),
        "registration_rows": registration["row_count"],
        "cleaning": cleaning,
        "test_size": test_size,
        "random_state": random_state,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_purchase_rate": float(train[TARGET].mean()),
        "test_purchase_rate": float(test[TARGET].mean()),
        "feature_columns": [column for column in train.columns if column != TARGET],
        "clip_bounds": clip_bounds,
        "train_path": str(train_path),
        "test_path": str(test_path),
    }
    summary_path = output_dir / "prepare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    summary["summary_path"] = str(summary_path)
    return train, test, summary


def print_summary(summary):
    print("Data preparation summary")
    print(f"Rows registered: {summary['registration_rows']}")
    print(f"Gender labels corrected (Fe Male -> Female): {summary['cleaning']['gender_label_fixes']}")
    print(f"Columns dropped: {', '.join(summary['cleaning']['dropped_columns'])}")
    print(f"Duplicate rows removed: {summary['cleaning']['duplicate_rows_removed']}")
    print(f"Train rows: {summary['train_rows']} (purchase rate {summary['train_purchase_rate']:.4f})")
    print(f"Test rows: {summary['test_rows']} (purchase rate {summary['test_purchase_rate']:.4f})")
    print("Clip bounds learned from the training split:")
    for column, bounds in summary["clip_bounds"].items():
        print(f"  {column}: {bounds['lower']:.2f} to {bounds['upper']:.2f}")
    print(f"Saved {summary['train_path']}")
    print(f"Saved {summary['test_path']}")


def main():
    _, _, summary = prepare_dataset()
    print_summary(summary)
    return summary


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Data preparation failed: {error}", file=sys.stderr)
        sys.exit(1)
