"""Feature engineering: processed v2  ->  feature-engineered v3.

Only PREVIOUS-snapshot columns are used (previous_rank, previous_points, association).
`FifaFeatureEngineer` is a scikit-learn transformer so the exact same feature logic is
embedded in the trained/registered model and applied again at serving time.
"""
import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import FEATURES_PATH, PROCESSED_PATH, RAW_FEATURES, STATS_PATH, TARGET
from .versioning import record_version

NUMERIC_FEATURES = [
    "previous_rank", "previous_points", "log_prev_points", "log_prev_rank",
    "pct_of_leader", "confed_mean_prev_points", "points_vs_confed_mean",
    "points_z_in_confed", "confed_size",
]
CATEGORICAL_FEATURES = ["association"]
ENGINEERED_FEATURES = [c for c in NUMERIC_FEATURES if c not in ("previous_rank", "previous_points")]


def compute_stats(df: pd.DataFrame) -> dict:
    """Group/global statistics of the PREVIOUS points (no label information)."""
    g = df.groupby("association")["previous_points"]
    return {
        "leader_points": float(df["previous_points"].max()),
        "global_mean": float(df["previous_points"].mean()),
        "global_std": float(df["previous_points"].std(ddof=0)),
        "confed": {
            k: {"mean": float(g.mean()[k]), "std": float(g.std(ddof=0)[k]), "size": int(g.size()[k])}
            for k in g.groups
        },
    }


class FifaFeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds engineered columns. `stats=None` -> learn them in fit(); else frozen stats."""

    def __init__(self, stats=None):
        self.stats = stats

    def fit(self, X, y=None):
        self.stats_ = self.stats if self.stats is not None else compute_stats(X)
        return self

    def transform(self, X):
        s = self.stats_
        X = pd.DataFrame(X).copy()
        pp = X["previous_points"].astype(float)
        info = X["association"].map(lambda a: s["confed"].get(a))
        mean = info.map(lambda d: d["mean"] if d else s["global_mean"]).astype(float)
        std = info.map(lambda d: d["std"] if d else s["global_std"]).astype(float).clip(lower=1.0)
        size = info.map(lambda d: d["size"] if d else 0).astype(float)
        out = X[RAW_FEATURES].copy()
        out["log_prev_points"] = np.log1p(pp)
        out["log_prev_rank"] = np.log(X["previous_rank"].astype(float))
        out["pct_of_leader"] = pp / s["leader_points"]
        out["confed_mean_prev_points"] = mean
        out["points_vs_confed_mean"] = pp - mean
        out["points_z_in_confed"] = (pp - mean) / std
        out["confed_size"] = size
        return out


def build_features(save: bool = True) -> pd.DataFrame:
    """Processed v2.0  ->  Feature-engineered v3.0 (with frozen statistics file)."""
    processed = pd.read_csv(PROCESSED_PATH)
    stats = compute_stats(processed)
    feats = FifaFeatureEngineer(stats).fit(processed).transform(processed)
    table = pd.concat([processed[["team", "team_code"]], feats, processed[[TARGET]]], axis=1).round(6)
    if save:
        FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(FEATURES_PATH, index=False)
        STATS_PATH.write_text(json.dumps(stats, indent=2))
        record_version("v3.0", "feature-engineered", FEATURES_PATH, table, "v2.0",
                       "Engineered: log points/rank, % of leader, confederation mean/z-score/size.")
    return table


def load_stats() -> dict:
    return json.loads(STATS_PATH.read_text())


if __name__ == "__main__":
    t = build_features()
    print(t.head())
    print("Feature table shape:", t.shape)
