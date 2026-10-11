"""Smoke-tests the MLflow logic (tracking.py) end to end.
Uses the real mlflow if installed, otherwise the in-memory stub in tests/stubs."""
import sys, unittest, warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
try:
    import mlflow  # noqa: F401
    USING_STUB = False
except ImportError:
    sys.path.insert(0, str(ROOT / "tests" / "stubs"))
    USING_STUB = True


@unittest.skipUnless(USING_STUB, "stub-based flow test (real MLflow is exercised by scripts/exp1_mlflow_registry.py)")
class TestTrackingFlow(unittest.TestCase):
    def test_six_models_then_three_versions_and_alias(self):
        warnings.filterwarnings("ignore")
        import mlflow as ml
        from fifa_mlops import tracking as T
        results = T.run_comparison()
        self.assertEqual(len(results), 6)
        reg = T.register_versions(results)
        self.assertEqual([v["version"] for v in reg["versions"]], [1, 2, 3])
        self.assertTrue(any(e[0] == "alias" for e in ml.LOG))


if __name__ == "__main__":
    unittest.main()
