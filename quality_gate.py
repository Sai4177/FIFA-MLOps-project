"""CI Practical-3: ML quality gate. Exit code 1 stops the pipeline when accuracy is too low.

Threshold: 0.85 by default. Override for the failure demo without editing code:
    MIN_ACCURACY=0.99 python quality_gate.py
"""
import json
import os
import sys

MINIMUM_ACCURACY = float(os.environ.get("MIN_ACCURACY", "0.85"))

print("Reading model evaluation metrics...")
with open("metrics.json", "r") as file:
    metrics = json.load(file)

accuracy = metrics["accuracy"]
print("Model Accuracy   :", round(accuracy, 4))
print("Required Accuracy:", MINIMUM_ACCURACY)

if accuracy < MINIMUM_ACCURACY:
    print("QUALITY GATE FAILED")
    print("Model performance is below the required threshold.")
    sys.exit(1)

print("QUALITY GATE PASSED")
print("Model performance satisfies the required threshold.")
sys.exit(0)
