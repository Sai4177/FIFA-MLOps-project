"""CI Practical-2 tests: did the ML pipeline work? (run AFTER python train_model.py)"""
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "src"), str(HERE)]

import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load(HERE / "fifa_tier_model.pkl")
        cls.metrics = json.loads((HERE / "metrics.json").read_text())

    def test_artifacts_created(self):
        for name in ("fifa_tier_model.pkl", "metrics.json", "fifa_dataset.csv"):
            self.assertTrue((HERE / name).exists(), name)

    def test_dataset_size(self):
        self.assertEqual(len(pd.read_csv(HERE / "fifa_dataset.csv")), 211)

    def test_metrics_content(self):
        for key in ("accuracy", "f1_macro", "training_records", "testing_records"):
            self.assertIn(key, self.metrics)
        self.assertEqual(self.metrics["training_records"] + self.metrics["testing_records"], 211)
        self.assertGreater(self.metrics["accuracy"], 0.5)

    def test_model_predicts_valid_tier(self):
        sample = pd.DataFrame([{"previous_rank": 3, "previous_points": 1770.65, "association": "CONMEBOL"}])
        self.assertIn(self.model.predict(sample)[0], {"Elite", "Strong", "Mid", "Developing"})

    def test_top_and_bottom_teams(self):
        top = pd.DataFrame([{"previous_rank": 1, "previous_points": 1837.56, "association": "CONMEBOL"}])
        bottom = pd.DataFrame([{"previous_rank": 205, "previous_points": 838.33, "association": "AFC"}])
        self.assertEqual(self.model.predict(top)[0], "Elite")
        self.assertEqual(self.model.predict(bottom)[0], "Developing")

    def test_unseen_association_does_not_crash(self):
        odd = pd.DataFrame([{"previous_rank": 50, "previous_points": 1400.0, "association": "XXX"}])
        self.assertIn(self.model.predict(odd)[0], {"Elite", "Strong", "Mid", "Developing"})


if __name__ == "__main__":
    unittest.main()
