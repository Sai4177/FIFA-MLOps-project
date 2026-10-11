"""CI Practical-1 unit tests for the pure-Python tier logic."""
import unittest

from tier_logic import rank_to_tier


class TestTierLogic(unittest.TestCase):
    def test_elite_case(self):
        self.assertEqual(rank_to_tier(1), "Elite")
        self.assertEqual(rank_to_tier(20), "Elite")

    def test_strong_case(self):
        self.assertEqual(rank_to_tier(21), "Strong")
        self.assertEqual(rank_to_tier(60), "Strong")

    def test_mid_case(self):
        self.assertEqual(rank_to_tier(61), "Mid")
        self.assertEqual(rank_to_tier(120), "Mid")

    def test_developing_case(self):
        self.assertEqual(rank_to_tier(121), "Developing")
        self.assertEqual(rank_to_tier(211), "Developing")

    def test_invalid_rank(self):
        with self.assertRaises(ValueError):
            rank_to_tier(0)
        with self.assertRaises(TypeError):
            rank_to_tier("5")


if __name__ == "__main__":
    unittest.main()
