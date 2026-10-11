"""CI Practical-5 tests for the Flask prediction application (run AFTER python train_model.py)."""
import unittest

from app import app


class TestPredictionApplication(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["status"], "ok")
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["status"], "healthy")

    def test_high_performance_prediction(self):
        r = self.client.post("/predict", json={"previous_rank": 3, "previous_points": 1770.65, "association": "CONMEBOL"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["prediction"], "Elite")

    def test_low_performance_prediction(self):
        r = self.client.post("/predict", json={"previous_rank": 190, "previous_points": 897.1, "association": "AFC"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["prediction"], "Developing")

    def test_missing_field_validation(self):
        r = self.client.post("/predict", json={"previous_rank": 3})
        self.assertEqual(r.status_code, 400)
        self.assertIn("missing_fields", r.get_json())

    def test_invalid_association_rejected(self):
        r = self.client.post("/predict", json={"previous_rank": 3, "previous_points": 1770.65, "association": "MARS"})
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
