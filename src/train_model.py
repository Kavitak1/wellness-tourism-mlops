"""Tune candidate models, log the experiments, and save the best pipeline."""

import json
import sys

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    CATEGORICAL_FEATURES,
    MODEL_DIR,
    NUMERIC_FEATURES,
    PROCESSED_DIR,
    REPORT_DIR,
    TARGET,
)

RANDOM_STATE = 42


def load_splits(processed_dir=PROCESSED_DIR):
    """Load the train and test tables published by the prepare job."""
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Train or test split is missing. Run data preparation first, "
            "or download the processed-data workflow artifact into data/processed."
        )
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def build_preprocessor():
    """Encode categorical inputs and pass numeric inputs through unchanged."""
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )


def candidate_models(scale_pos_weight):
    """Model families allowed by the project rubric, each with a small search grid."""
    return [
        (
            "decision_tree",
            DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {
                "model__max_depth": [4, 8, 12],
                "model__min_samples_leaf": [5, 20],
            },
        ),
        (
            "random_forest",
            RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            {
                "model__n_estimators": [150],
                "model__max_depth": [8, 14],
                "model__min_samples_leaf": [2, 5],
            },
        ),
        (
            "adaboost",
            AdaBoostClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [50, 100],
                "model__learning_rate": [0.5, 1.0],
            },
        ),
        (
            "gradient_boosting",
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [100],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
        ),
        (
            "xgboost",
            _xgboost_estimator(scale_pos_weight),
            {
                "model__n_estimators": [150],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.05, 0.1],
            },
        ),
    ]


def _xgboost_estimator(scale_pos_weight):
    from xgboost import XGBClassifier

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def evaluate(model, features, target):
    """Score a fitted pipeline on one split."""
    probabilities = model.predict_proba(features)[:, 1]
    predictions = model.predict(features)
    return {
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "f1": float(f1_score(target, predictions)),
        "recall": float(recall_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(target, predictions)),
    }


def train_models(processed_dir=PROCESSED_DIR, model_dir=MODEL_DIR, report_dir=REPORT_DIR):
    """Tune every candidate on the training artifact and persist the best one.

    Selection uses stratified cross-validated ROC-AUC on the training split.
    The held-out test split is scored once, after the model is chosen.
    """
    train, test = load_splits(processed_dir)
    feature_columns = [column for column in train.columns if column != TARGET]
    x_train = train[feature_columns]
    y_train = train[TARGET]
    x_test = test[feature_columns]
    y_test = test[TARGET]

    positive = float((y_train == 1).sum())
    negative = float((y_train == 0).sum())
    scale_pos_weight = negative / positive
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    experiments = []
    best_search = None
    best_name = None
    for name, estimator, grid in candidate_models(scale_pos_weight):
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(x_train, y_train)
        test_metrics = evaluate(search.best_estimator_, x_test, y_test)
        record = {
            "model": name,
            "best_params": search.best_params_,
            "cv_roc_auc": float(search.best_score_),
            "test_roc_auc": test_metrics["roc_auc"],
            "test_f1": test_metrics["f1"],
            "test_recall": test_metrics["recall"],
            "test_precision": test_metrics["precision"],
            "test_accuracy": test_metrics["accuracy"],
        }
        experiments.append(record)
        print(
            f"{name}: cv_roc_auc={record['cv_roc_auc']:.4f} "
            f"test_roc_auc={record['test_roc_auc']:.4f} "
            f"test_f1={record['test_f1']:.4f} "
            f"params={record['best_params']}"
        )
        if best_search is None or record["cv_roc_auc"] > best_search["cv_roc_auc"]:
            best_search = record
            best_name = name
            best_estimator = search.best_estimator_

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "best_model.joblib"
    joblib.dump(best_estimator, model_path)

    metadata = {
        "best_model": best_name,
        "selection_metric": "cv_roc_auc",
        "best_params": best_search["best_params"],
        "cv_roc_auc": best_search["cv_roc_auc"],
        "test_metrics": {
            "roc_auc": best_search["test_roc_auc"],
            "f1": best_search["test_f1"],
            "recall": best_search["test_recall"],
            "precision": best_search["test_precision"],
            "accuracy": best_search["test_accuracy"],
        },
        "feature_columns": feature_columns,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "target": TARGET,
        "positive_class": 1,
        "model_path": str(model_path),
    }
    metadata_path = model_dir / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    experiment_path = report_dir / "experiment_log.json"
    experiment_path.write_text(json.dumps(experiments, indent=2))

    print(f"Best model: {best_name}")
    print(f"Logged tuned parameters to {experiment_path}")
    print(f"Saved model to {model_path}")
    return experiments, metadata


def main():
    train_models()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Model training failed: {error}", file=sys.stderr)
        sys.exit(1)
