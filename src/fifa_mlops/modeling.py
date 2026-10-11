"""Model zoo, pipelines, splitting and evaluation (pure scikit-learn, no MLflow)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import CV_FOLDS, FEATURES_PATH, RANDOM_STATE, RAW_FEATURES, TARGET, TEST_SIZE, TIER_ORDER
from .feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES, FifaFeatureEngineer, load_stats

SEED = RANDOM_STATE

# ---- six different models (Version-1 / baseline hyper-parameters) --------------------
def get_models() -> dict:
    return {
        "LogisticRegression": LogisticRegression(C=1.0, max_iter=2000, random_state=SEED),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(C=1.0, kernel="rbf", probability=True, random_state=SEED),
        "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=SEED),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=SEED),
    }

# ---- changed hyper-parameters used to create Version 2 and Version 3 ------------------
TUNED_PARAMS = {
    "LogisticRegression": [dict(C=10.0, max_iter=3000), dict(C=0.1, max_iter=3000)],
    "KNN": [dict(n_neighbors=7, weights="distance"), dict(n_neighbors=3, weights="distance")],
    "SVM": [dict(C=10.0, gamma="scale"), dict(C=100.0, gamma=0.05)],
    "DecisionTree": [dict(max_depth=8, min_samples_leaf=2), dict(max_depth=3)],
    "RandomForest": [dict(n_estimators=300, max_depth=10, min_samples_leaf=2),
                     dict(n_estimators=500, max_depth=6, max_features="sqrt")],
    "GradientBoosting": [dict(n_estimators=200, learning_rate=0.05, max_depth=3),
                         dict(n_estimators=300, learning_rate=0.03, max_depth=2, subsample=0.8)],
}


def build_pipeline(estimator, stats="frozen") -> Pipeline:
    """raw columns -> feature engineering -> scaling/one-hot -> classifier.

    stats="frozen" : use data/features/feature_stats_v3.json (identical to dataset v3.0)
    stats=None     : learn the statistics from the training rows only (standalone scripts / CI)
    stats=<dict>   : use the given statistics
    """
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("fe", FifaFeatureEngineer(load_stats() if isinstance(stats, str) else stats)),
                     ("prep", prep), ("clf", estimator)])


def load_modeling_data() -> pd.DataFrame:
    """Reads the feature-engineered dataset v3.0."""
    return pd.read_csv(FEATURES_PATH)


def split_data(table: pd.DataFrame | None = None):
    """Stratified 80/20 split (random_state=42). Works on dataset v3.0 or any table with the same columns."""
    table = load_modeling_data() if table is None else table
    X, y = table[RAW_FEATURES], table[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)


def evaluate(pipeline, X_train, y_train, X_test, y_test, cv: bool = True) -> dict:
    pred = pipeline.predict(X_test)
    m = {
        "accuracy": accuracy_score(y_test, pred),
        "precision_macro": precision_score(y_test, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
        "train_accuracy": accuracy_score(y_train, pipeline.predict(X_train)),
    }
    if cv:
        skf = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
        scores = cross_val_score(_clone_unfitted(pipeline), X_train, y_train, cv=skf, scoring="accuracy")
        m["cv_accuracy_mean"], m["cv_accuracy_std"] = float(scores.mean()), float(scores.std())
    return {k: float(v) for k, v in m.items()}


def _clone_unfitted(pipeline):
    from sklearn.base import clone
    return clone(pipeline)


def confusion(pipeline, X_test, y_test) -> pd.DataFrame:
    pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, pred, labels=TIER_ORDER)
    return pd.DataFrame(cm, index=[f"true_{t}" for t in TIER_ORDER], columns=[f"pred_{t}" for t in TIER_ORDER])


def select_best(results: list[dict]) -> dict:
    """Best model = highest held-out test macro-F1 (tie-break: 5-fold CV accuracy)."""
    return max(results, key=lambda r: (round(r["metrics"]["f1_macro"], 6), r["metrics"]["cv_accuracy_mean"]))


def train_all(models: dict | None = None) -> list[dict]:
    """Train + evaluate every model (no MLflow). Returns list of result dicts."""
    X_train, X_test, y_train, y_test = split_data()
    out = []
    for name, est in (models or get_models()).items():
        pipe = build_pipeline(est).fit(X_train, y_train)
        out.append({"name": name, "pipeline": pipe,
                    "metrics": evaluate(pipe, X_train, y_train, X_test, y_test)})
    return out
