import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from config import CHALLENGES, MODEL_REGISTRY, make_logistic_regression
from data_loader import load_data
from evaluation import evaluate, validate_predictions
from main import run


class FakeModel:
    """Synthetic integration fixture, not an assignment model."""
    def __init__(self):
        self.clf = DummyClassifier(strategy="most_frequent")

    def fit(self, texts, labels):
        assert all(t.startswith("train") for t in texts)
        self.clf.fit(np.zeros((len(texts), 1)), labels)
        return self

    def predict(self, texts):
        return self.clf.predict(np.zeros((len(texts), 1)))


class CommonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for ch in CHALLENGES:
            folder = self.root / "data" / ch
            folder.mkdir(parents=True)
            pd.DataFrame({"text": ["train one", "train two", "train three"], "label": [0, 1, 1]}).to_csv(folder / "train.csv", index=False)
            pd.DataFrame({"id": ["003", "001", "002"], "text": [" NA ", "", "New\nline"], "label": [0, 1, 1]}).to_csv(folder / "test.csv", index=False)

    def test_loader_preserves_raw_text_and_order(self):
        data = load_data("A", self.root / "data")
        self.assertEqual(data.X_test, [" NA ", "", "New\nline"])
        self.assertEqual(data.test_ids, ["003", "001", "002"])
        np.testing.assert_array_equal(data.y_test, [0, 1, 1])

    def test_invalid_label(self):
        path = self.root / "data/A/test.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "label"] = 3
        frame.to_csv(path, index=False)
        with self.assertRaises(ValueError):
            load_data("A", self.root / "data")

    def test_metrics(self):
        metrics = evaluate([0, 1, 1], [1, 1, 1])
        self.assertAlmostEqual(metrics["macro_f1"], 0.4)
        self.assertAlmostEqual(metrics["accuracy"], 2/3)
        self.assertEqual(metrics["confusion_matrix"], [[0, 1], [0, 2]])
        for invalid in ([1], [0, 2], [0.0, 1.0], [[0], [1]], ["0", "1"]):
            with self.assertRaises(ValueError):
                validate_predictions(invalid, 2)

    def test_factory(self):
        self.assertEqual(make_logistic_regression().get_params()["random_state"], 42)
        self.assertIsNot(make_logistic_regression(), make_logistic_regression())

    @patch("main.new_model", side_effect=lambda _: FakeModel())
    def test_full_integration(self, factory):
        out = self.root / "results"
        result = run(tuple(MODEL_REGISTRY), CHALLENGES, self.root / "data", out)
        self.assertEqual(len(result), 15)
        self.assertEqual(factory.call_count, 15)
        for ch in CHALLENGES:
            p = pd.read_csv(out / f"predictions_{ch}.csv", dtype={"id": str})
            self.assertEqual(list(p.columns), ["id", "label"])
            self.assertEqual(p.id.tolist(), ["003", "001", "002"])
        self.assertEqual(pd.read_csv(out / "selected_models.csv").model.tolist(), ["baseline"]*3)
        e = pd.read_csv(out / "core/word2vec_lr/A/errors.csv", dtype={"id": str})
        self.assertEqual(e.id.tolist(), ["003"])
        self.assertTrue((out / "comparison.csv").is_file())

    @patch("main.new_model", side_effect=lambda _: FakeModel())
    def test_partial_integration(self, factory):
        out = self.root / "partial"
        result = run(("word2vec_lr",), ("A",), self.root / "data", out)
        self.assertEqual(len(result), 1)
        self.assertEqual(factory.call_count, 1)
        self.assertFalse((out / "predictions_A.csv").exists())
        self.assertTrue((out / "core/word2vec_lr/A/classification_report.txt").is_file())


if __name__ == "__main__":
    unittest.main()
