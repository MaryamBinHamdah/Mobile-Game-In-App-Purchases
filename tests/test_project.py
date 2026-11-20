"""Quick project checks: run with `python -m unittest discover -s tests`."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import joblib

from src.train_model import CLASS_ORDER, FEATURES, TARGET, load_source

ROOT = Path(__file__).resolve().parents[1]


class ProjectChecks(unittest.TestCase):
    def test_original_source_and_target_are_present(self):
        data = load_source()
        self.assertEqual(len(data), 3024)
        self.assertTrue(set(CLASS_ORDER).issubset(set(data[TARGET].unique())))
        self.assertEqual(data[TARGET].isna().sum(), 0)

    def test_model_features_exclude_target_and_purchase_outcomes(self):
        forbidden = {
            TARGET, "UserID", "InAppPurchaseAmount", "PurchaseCount30d",
            "FirstPurchaseDaysAfterInstall", "PaymentMethod", "LastPurchaseDate",
        }
        self.assertTrue(set(FEATURES).isdisjoint(forbidden))
        self.assertEqual(len(FEATURES), 7)

    def test_cleaning_audit_has_no_missing_values_or_duplicates(self):
        audit_path = ROOT / "outputs" / "data_quality" / "cleaning_audit.json"
        self.assertTrue(audit_path.exists(), "Run `python src/clean_data.py` first.")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        self.assertEqual(audit["after"]["rows"], 247500)
        self.assertEqual(audit["after"]["missing_cells"], 0)
        self.assertEqual(audit["after"]["exact_duplicates"], 0)

    def test_saved_model_returns_valid_segment_and_probabilities(self):
        model_path = ROOT / "models" / "spending_segment_model.joblib"
        self.assertTrue(model_path.exists(), "Run `python src/train_model.py` first.")
        model = joblib.load(model_path)
        sample = load_source()[FEATURES].iloc[[0]]
        prediction = model.predict(sample)[0]
        probabilities = model.predict_proba(sample)[0]
        self.assertIn(prediction, CLASS_ORDER)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
