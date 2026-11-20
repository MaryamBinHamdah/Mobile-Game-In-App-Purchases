"""Train and compare simple spending-segment classifiers on the original source data.

The 250,000-row expanded file is intentionally not used for model evaluation: it is
synthetic augmentation of a 3,024-row source and contains generated copies. The
original source file is also described by its publisher as synthetic, so results
remain an educational demonstration rather than evidence about real players.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILE = ROOT / "data" / "01_original_kaggle_data.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs" / "model_evaluation"
TARGET = "SpendingSegment"
CLASS_ORDER = ["Minnow", "Dolphin", "Whale"]
# These columns are known at player/gameplay level and are not monetary outcomes.
NUMERIC_FEATURES = ["Age", "SessionCount", "AverageSessionLength"]
CATEGORICAL_FEATURES = ["Gender", "Country", "Device", "GameGenre"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def canonicalize_categories(frame: pd.DataFrame) -> pd.DataFrame:
    """Standardize known spelling/case variants without changing the target."""
    result = frame.copy()
    for column in CATEGORICAL_FEATURES:
        if column in result:
            result[column] = result[column].astype("string").str.strip()
            result[column] = result[column].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    if "Device" in result:
        result["Device"] = result["Device"].replace({"ios": "iOS", "IOS": "iOS", "android": "Android"})
    if "Gender" in result:
        result["Gender"] = result["Gender"].replace({"m": "Male", "M": "Male", "f": "Female", "F": "Female"})
    if "GameGenre" in result:
        result["GameGenre"] = result["GameGenre"].replace({"BattleRoyale": "Battle Royale", "battle royale": "Battle Royale"})
    for column in CATEGORICAL_FEATURES:
        if column in result:
            result[column] = result[column].astype(object).where(result[column].notna(), np.nan)
    return result


def load_source() -> pd.DataFrame:
    """Load the original source and apply only transparent, minimal validation."""
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Expected source dataset at {SOURCE_FILE}")
    data = pd.read_csv(SOURCE_FILE)
    required = set(FEATURES + [TARGET])
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Source data is missing required columns: {missing}")
    data = canonicalize_categories(data)
    # A duration written in seconds is converted into minutes; ordinary values are
    # already in minutes. This feature is parsed defensively because the source can
    # contain mixed numeric and text formatting.
    duration = data["AverageSessionLength"].astype("string").str.strip()
    seconds = duration.str.extract(r"^([-+]?\d*\.?\d+)\s*sec(?:onds?)?$", expand=False)
    seconds_mask = seconds.notna()
    data["AverageSessionLength"] = pd.to_numeric(duration, errors="coerce")
    data.loc[seconds_mask, "AverageSessionLength"] = pd.to_numeric(seconds[seconds_mask], errors="coerce") / 60
    for column in ["Age", "SessionCount"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["AverageSessionLength"] = pd.to_numeric(data["AverageSessionLength"], errors="coerce")
    data.loc[~data["Age"].between(13, 100), "Age"] = np.nan
    data.loc[data["SessionCount"] < 0, "SessionCount"] = np.nan
    data.loc[data["AverageSessionLength"] < 0, "AverageSessionLength"] = np.nan
    data[TARGET] = data[TARGET].astype("string").str.strip().str.title()
    data = data[data[TARGET].isin(CLASS_ORDER)].copy()
    if data.empty:
        raise ValueError("No valid target labels remain after basic validation.")
    return data.reset_index(drop=True)


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")),
               ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer(
        transformers=[("numeric", numeric, NUMERIC_FEATURES),
                      ("categorical", categorical, CATEGORICAL_FEATURES)],
        remainder="drop",
    )


def make_candidates() -> dict[str, tuple[Pipeline, dict[str, list]]]:
    """Return two accessible model families with small, reproducible search grids."""
    prep = make_preprocessor()
    logistic = Pipeline([
        ("preprocess", prep),
        ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)),
    ])
    forest = Pipeline([
        ("preprocess", prep),
        ("model", RandomForestClassifier(class_weight="balanced_subsample", random_state=42, n_jobs=-1)),
    ])
    return {
        "Logistic regression": (logistic, {
            "model__C": [0.1, 1.0, 10.0],
        }),
        "Random forest": (forest, {
            "model__n_estimators": [150],
            "model__max_depth": [None, 12],
            "model__min_samples_leaf": [2, 5],
        }),
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_source()
    X = data[FEATURES].copy()
    y = data[TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    min_class_train = int(y_train.value_counts().min())
    folds = min(5, min_class_train)
    if folds < 2:
        raise ValueError("At least two examples per class are needed for cross-validation.")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    summaries = []
    searches = {}
    for name, (pipeline, grid) in make_candidates().items():
        print(f"Tuning {name} with {folds}-fold stratified cross-validation...")
        search = GridSearchCV(
            pipeline, grid, scoring="f1_macro", cv=cv, n_jobs=-1,
            refit=True, return_train_score=False,
        )
        search.fit(X_train, y_train)
        searches[name] = search
        pred = search.predict(X_test)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, pred, labels=CLASS_ORDER, zero_division=0
        )
        summaries.append({
            "model": name,
            "cv_macro_f1": float(search.best_score_),
            "test_macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
            "test_balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "test_accuracy": float(accuracy_score(y_test, pred)),
            "whale_precision": float(precision[2]),
            "whale_recall": float(recall[2]),
            "best_parameters": search.best_params_,
        })
        pd.DataFrame(classification_report(
            y_test, pred, labels=CLASS_ORDER, output_dict=True, zero_division=0
        )).T.to_csv(OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_classification_report.csv")
        pd.DataFrame(
            confusion_matrix(y_test, pred, labels=CLASS_ORDER),
            index=[f"Actual {x}" for x in CLASS_ORDER],
            columns=[f"Predicted {x}" for x in CLASS_ORDER],
        ).to_csv(OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_confusion_matrix.csv")

    # A simple reference point helps show whether the models improve on always
    # choosing the most common class. It is not a candidate deployment model.
    majority = DummyClassifier(strategy="most_frequent")
    majority.fit(X_train, y_train)
    baseline_pred = majority.predict(X_test)
    baseline_precision, baseline_recall, _, _ = precision_recall_fscore_support(
        y_test, baseline_pred, labels=CLASS_ORDER, zero_division=0
    )
    summaries.append({
        "model": "Majority-class baseline",
        "cv_macro_f1": np.nan,
        "test_macro_f1": float(f1_score(y_test, baseline_pred, average="macro", zero_division=0)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, baseline_pred)),
        "test_accuracy": float(accuracy_score(y_test, baseline_pred)),
        "whale_precision": float(baseline_precision[2]),
        "whale_recall": float(baseline_recall[2]),
        "best_parameters": {"strategy": "most_frequent"},
    })
    pd.DataFrame(
        confusion_matrix(y_test, baseline_pred, labels=CLASS_ORDER),
        index=[f"Actual {x}" for x in CLASS_ORDER],
        columns=[f"Predicted {x}" for x in CLASS_ORDER],
    ).to_csv(OUTPUT_DIR / "majority_class_confusion_matrix.csv")

    # Choose based on pre-declared macro-F1 and use Whale recall as the tie-breaker.
    candidate_summaries = [row for row in summaries if row["model"] != "Majority-class baseline"]
    winner = max(candidate_summaries, key=lambda row: (row["test_macro_f1"], row["whale_recall"]))
    best_pipeline = searches[winner["model"]].best_estimator_
    joblib.dump(best_pipeline, MODEL_DIR / "spending_segment_model.joblib")
    comparison = pd.DataFrame(summaries).sort_values(
        ["test_macro_f1", "whale_recall"], ascending=False
    )
    comparison["best_parameters"] = comparison["best_parameters"].map(json.dumps)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    metadata = {
        "target": TARGET,
        "classes": CLASS_ORDER,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "selected_model": winner["model"],
        "selected_parameters": winner["best_parameters"],
        "evaluation_rows": int(len(y_test)),
        "source_rows_used": int(len(data)),
        "split": {"test_size": 0.25, "random_state": 42, "stratified": True},
        "validation": {"method": "GridSearchCV", "folds": folds, "scoring": "macro-F1", "random_state": 42},
        "comparison": summaries,
        "warning": "The original dataset publisher describes the data as synthetic. These scores are educational results, not evidence about real players or campaign impact.",
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nModel comparison:")
    print(comparison[["model", "cv_macro_f1", "test_macro_f1", "test_balanced_accuracy", "test_accuracy", "whale_recall"]].to_string(index=False))
    print(f"\nSelected model: {winner['model']}")
    print("Saved model and evaluation evidence under models/ and outputs/model_evaluation/.")


if __name__ == "__main__":
    main()
