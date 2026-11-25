"""Interface smoke tests for FeaturesLR. Synthetic texts only, not real data."""
import unittest

import numpy as np

from models.features_lr import FEATURE_NAMES, FeaturesLR, extract_features_df


class FeaturesLRTests(unittest.TestCase):
    texts = ["Hello there! I don't know.\nShort line", "Additionally, it is crucial to ensure robust results.",
             "", "12345 ;;; (()) — —", "We went home. It rained, again, and again?"] * 4
    labels = np.array([0, 1, 0, 1, 0] * 4)

    def test_at_least_ten_features(self):
        self.assertGreaterEqual(len(FEATURE_NAMES), 10)
        self.assertEqual(list(extract_features_df(["x"]).columns), FEATURE_NAMES)

    def test_empty_text_gives_finite_values(self):
        self.assertTrue(np.isfinite(extract_features_df([""]).to_numpy()).all())

    def test_fit_predict_contract(self):
        inputs = list(self.texts)
        model = FeaturesLR()
        self.assertIs(model.fit(inputs, self.labels), model)
        pred = model.predict(["new unseen text", ""])
        self.assertEqual(pred.shape, (2,))
        self.assertTrue(np.issubdtype(pred.dtype, np.integer))
        self.assertTrue(np.isin(pred, [0, 1]).all())
        self.assertEqual(inputs, self.texts)  # inputs are not changed

    def test_feature_subset(self):
        model = FeaturesLR(features=["log_number_tokens", "commas_ratio"]).fit(self.texts, self.labels)
        self.assertEqual(set(model.coefficients().index), {"log_number_tokens", "commas_ratio"})
        with self.assertRaises(ValueError):
            FeaturesLR(features=["not_a_feature"])


if __name__ == "__main__":
    unittest.main()
