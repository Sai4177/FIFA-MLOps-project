"""CI Practical-2: automated ML training (FIFA tier classifier).

Generates on the CI runner:  fifa_tier_model.pkl , metrics.json , fifa_dataset.csv
Run:  python train_model.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "src"), str(HERE)]

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from fifa_mlops.config import RANDOM_STATE, RAW_FEATURES, TARGET, TEST_SIZE
from fifa_mlops.data_engineering import add_label_and_analysis_columns, clean, load_raw, validate_schema
from fifa_mlops.modeling import build_pipeline

MODEL_FILE = HERE / "fifa_tier_model.pkl"
METRICS_FILE = HERE / "metrics.json"
DATASET_FILE = HERE / "fifa_dataset.csv"


def main():
    raw = load_raw()
    validate_schema(raw)
    df, _ = clean(raw)
    data = add_label_and_analysis_columns(df)
    data.to_csv(DATASET_FILE, index=False)

    X, y = data[RAW_FEATURES], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    model = build_pipeline(RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=2,
                                                  random_state=RANDOM_STATE), stats=None)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "accuracy": float(round(accuracy_score(y_test, pred), 4)),
        "f1_macro": float(round(f1_score(y_test, pred, average="macro"), 4)),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "classes": [str(c) for c in model.classes_],
    }
    joblib.dump(model, MODEL_FILE)
    METRICS_FILE.write_text(json.dumps(metrics, indent=4))
    print("Model trained. Metrics:", json.dumps(metrics))


if __name__ == "__main__":
    main()
