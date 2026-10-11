"""Runs the same 6-model comparison + v1/v2/v3 hyper-parameter versions WITHOUT MLflow
and writes reports/preview_*.csv|png.  Handy to check numbers before opening the MLflow UI."""
import sys
import warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

from fifa_mlops import modeling as M
from fifa_mlops.config import REPORTS_DIR

REPORTS_DIR.mkdir(exist_ok=True)
split = M.split_data()
X_train, X_test, y_train, y_test = split
print("scikit-learn", sklearn.__version__, "| train", len(X_train), "test", len(X_test))

results = M.train_all()
cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "cv_accuracy_mean", "cv_accuracy_std", "train_accuracy"]
table = pd.DataFrame([{"model": r["name"], **{c: round(r["metrics"][c], 4) for c in cols}} for r in results])
table = table.sort_values(["f1_macro", "cv_accuracy_mean"], ascending=False).reset_index(drop=True)
table.insert(0, "rank", range(1, len(table) + 1))
table.to_csv(REPORTS_DIR / "preview_model_comparison.csv", index=False)
print(table.to_string(index=False))

best = M.select_best(results)["name"]
print("\nBest model:", best)
rows = [{"registry_version": 1, "label": "baseline", "hyper_parameters": "default (see table above)",
         **{c: round(M.select_best(results)["metrics"][c], 4) for c in ("accuracy", "f1_macro", "cv_accuracy_mean")}}]
for i, params in enumerate(M.TUNED_PARAMS[best], start=2):
    pipe = M.build_pipeline(M.get_models()[best].set_params(**params)).fit(X_train, y_train)
    m = M.evaluate(pipe, X_train, y_train, X_test, y_test)
    rows.append({"registry_version": i, "label": f"tuned-{i-1}", "hyper_parameters": str(params),
                 **{c: round(m[c], 4) for c in ("accuracy", "f1_macro", "cv_accuracy_mean")}})
ver = pd.DataFrame(rows)
ver.to_csv(REPORTS_DIR / "preview_registry_versions.csv", index=False)
print(ver.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4.2))
t = table.sort_values("f1_macro")
ax.barh(t["model"], t["accuracy"], color="#9ecae1", label="Test accuracy")
ax.barh(t["model"], t["f1_macro"], height=0.4, color="#08519c", label="Test macro-F1")
ax.set_xlim(0.6, 1.0); ax.set_xlabel("score"); ax.set_title("FIFA tier classification - 6-model comparison")
ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(REPORTS_DIR / "preview_model_comparison.png", dpi=140)
