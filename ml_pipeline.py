"""EXPERIMENT 3 - End-to-end ML pipeline (Module 2, Practice 2) on the FIFA ranking dataset.

One command runs the complete workflow automatically:

    python ml_pipeline.py

load_data -> prepare_data -> split_data -> build_pipeline -> train_model -> evaluate_model
-> save_model -> reload + predict -> save_report

Outputs
    artifacts/fifa_pipeline.joblib   complete fitted pipeline (feature engineering + preprocessing + model)
    outputs/metrics.json             machine-readable metrics
    outputs/pipeline_report.txt      human-readable report
    outputs/sample_prediction.csv    one team + its prediction
"""
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(BASE_DIR / "src"), str(BASE_DIR)]

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from fifa_mlops.config import RANDOM_STATE, RAW_FEATURES, RAW_PATH, TARGET, TEST_SIZE
from fifa_mlops.data_engineering import add_label_and_analysis_columns, clean, load_raw, validate_schema
from fifa_mlops.modeling import build_pipeline

ARTIFACT_DIR = BASE_DIR / "artifacts"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_PATH = ARTIFACT_DIR / "fifa_pipeline.joblib"

# selected model = tuned Random Forest (best family in the Unit-II MLflow comparison)
MODEL_PARAMS = dict(n_estimators=500, max_depth=6, max_features="sqrt", random_state=RANDOM_STATE)


def load_data() -> pd.DataFrame:
    """Load the raw CSV, validate and clean it, and add the tier label."""
    raw = load_raw(RAW_PATH)
    validate_schema(raw)
    df, _ = clean(raw)
    return add_label_and_analysis_columns(df)


def prepare_data(df: pd.DataFrame):
    """Separate the target and keep ONLY previous-snapshot input features (no leakage)."""
    return df[RAW_FEATURES], df[TARGET]


def split_data(X, y):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def build_ml_pipeline():
    return build_pipeline(RandomForestClassifier(**MODEL_PARAMS), stats=None)   # stats learned from train rows


def train_model(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision_macro": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
    }


def save_model(pipeline):
    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)


def test_saved_model(X_test):
    """Reload the saved artifact and predict one team from raw feature values."""
    loaded = joblib.load(MODEL_PATH)
    sample = X_test.iloc[[0]].copy()
    return sample, str(loaded.predict(sample)[0])


def save_report(metrics, sample, prediction, shape, n_train, n_test):
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=4))
    out = sample.copy(); out["prediction"] = prediction
    out.to_csv(OUTPUT_DIR / "sample_prediction.csv", index=False)
    lines = ["FIFA END-TO-END ML PIPELINE REPORT", "=" * 36,
             f"Dataset shape        : {shape[0]} x {shape[1]}", f"Input features       : {len(RAW_FEATURES)} raw (+ engineered inside pipeline)",
             f"Training records     : {n_train}", f"Testing records      : {n_test}",
             f"Model                : RandomForestClassifier {MODEL_PARAMS}", ""]
    lines += [f"{k:<17}: {v}" for k, v in metrics.items()]
    lines += ["", "Sample team          : " + sample.iloc[0].to_dict().__repr__(), f"Sample prediction    : {prediction}", "",
              "Status               : Completed successfully"]
    (OUTPUT_DIR / "pipeline_report.txt").write_text("\n".join(lines))


def main():
    print("FIFA END-TO-END ML PIPELINE")
    print("STEP 1: Loading dataset...");            df = load_data();                      print("   done", df.shape)
    print("STEP 2: Preparing data...");             X, y = prepare_data(df);               print("   features:", list(X.columns))
    print("STEP 3: Splitting dataset...");          X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"   train={len(X_train)} test={len(X_test)}")
    print("STEP 4: Building ML pipeline...");       pipeline = build_ml_pipeline()
    print("STEP 5: Training model...");             pipeline = train_model(pipeline, X_train, y_train)
    print("STEP 6: Evaluating model...");           metrics = evaluate_model(pipeline, X_test, y_test); print("  ", metrics)
    print("STEP 7: Saving trained pipeline...");    save_model(pipeline);                  print("   saved", MODEL_PATH.relative_to(BASE_DIR))
    print("STEP 8: Loading saved pipeline + predict..."); sample, pred = test_saved_model(X_test); print("   prediction:", pred)
    print("STEP 9: Saving execution results...");   save_report(metrics, sample, pred, df.shape, len(X_train), len(X_test))
    print("PIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
