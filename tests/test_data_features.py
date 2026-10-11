"""Data-engineering / feature-engineering / versioning tests (no MLflow needed)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import pandas as pd

from fifa_mlops import config as C
from fifa_mlops.data_engineering import add_label_and_analysis_columns, clean, load_raw
from fifa_mlops.feature_engineering import FifaFeatureEngineer, build_features, compute_stats
from fifa_mlops.data_engineering import build_processed
from fifa_mlops.versioning import load_versions


class TestDataFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        build_processed(); build_features()
        cls.v2 = pd.read_csv(C.PROCESSED_PATH)
        cls.v3 = pd.read_csv(C.FEATURES_PATH)

    def test_raw_to_processed(self):
        raw = load_raw()
        clean_df, rep = clean(raw)
        self.assertEqual((len(raw), len(clean_df)), (211, 211))
        self.assertTrue(rep["rank_is_unique"])
        self.assertEqual(set(self.v2["tier"]), set(C.TIER_ORDER))

    def test_no_target_leakage(self):
        for leaky in ("rank", "points", "rank_change", "points_change", "tier"):
            self.assertNotIn(leaky, C.RAW_FEATURES)
            self.assertNotIn(leaky, self.v3.drop(columns=["tier"]).columns)

    def test_v3_matches_feature_transformer(self):
        again = FifaFeatureEngineer(compute_stats(self.v2)).fit(self.v2).transform(self.v2).round(6)
        pd.testing.assert_frame_equal(again.reset_index(drop=True),
                                      self.v3[again.columns].reset_index(drop=True), check_dtype=False)

    def test_version_registry_lineage(self):
        d = {v["version"]: v for v in load_versions()["datasets"]}
        self.assertEqual([d[k]["parent"] for k in ("v1.0", "v2.0", "v3.0")], [None, "v1.0", "v2.0"])
        self.assertEqual(d["v1.0"]["rows"], 211)


if __name__ == "__main__":
    unittest.main()
