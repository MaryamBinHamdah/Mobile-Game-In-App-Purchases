"""Train and export the project's selected Gini Decision Tree pipeline.

Usage:
    python train_model.py /path/to/mobile_game_players_eda_features.csv
"""
from pathlib import Path
import sys
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

FEATURE_COLS = ["Age", "Country", "Device", "GameGenre", "SessionCount", "AverageSessionLength"]
LABELS = ["Minnow", "Dolphin", "Whale"]
RANDOM_STATE = 42


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python train_model.py /path/to/mobile_game_players_eda_features.csv")
    csv_path = Path(sys.argv[1]).expanduser().resolve()
    if not csv_path.exists():
        raise SystemExit(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path).drop_duplicates().copy()
    required = FEATURE_COLS + ["SpendingSegment"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise SystemExit(f"Required columns missing: {missing}")
    df = df.dropna(subset=["SpendingSegment"])
    df = df[df["SpendingSegment"].isin(LABELS)].copy()
    X = df[FEATURE_COLS]
    y = df["SpendingSegment"]

    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_cols),
            ("categorical", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_cols),
        ],
        remainder="drop",
    )
    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", DecisionTreeClassifier(
            criterion="gini", class_weight="balanced", random_state=RANDOM_STATE
        )),
    ])
    model.fit(X, y)
    output = Path(__file__).resolve().parent / "model.joblib"
    joblib.dump(model, output, compress=3)
    print(f"Trained on {len(df):,} rows; labels: {y.value_counts().to_dict()}")
    print(f"Saved model: {output} ({output.stat().st_size:,} bytes)")
    print("Note: the deployed artifact is refitted on all eligible project records after the test-set comparison.")


if __name__ == "__main__":
    main()
