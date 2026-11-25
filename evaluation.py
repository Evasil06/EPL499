"""Shared validation, metrics, and output conventions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from data_loader import ChallengeData


def validate_predictions(predictions, n: int) -> np.ndarray:
    pred = np.asarray(predictions)
    if pred.shape != (n,):
        raise ValueError(f"predict() must return shape ({n},), got {pred.shape}")
    if not np.issubdtype(pred.dtype, np.integer) or not np.isin(pred, [0, 1]).all():
        raise ValueError("predict() must return integer labels 0/1, not probabilities or strings")
    return pred.astype(np.int64, copy=False)


def evaluate(y_true, predictions) -> dict:
    y_true = np.asarray(y_true)
    pred = validate_predictions(predictions, len(y_true))
    return {
        "macro_f1": float(f1_score(y_true, pred, labels=[0, 1], average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "classification_report": classification_report(y_true, pred, labels=[0, 1], output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }


def save_predictions(path: str | Path, ids: list[str], predictions):
    pred = validate_predictions(predictions, len(ids))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, "label": pred}).to_csv(path, index=False, encoding="utf-8")


def evaluate_and_save(data: ChallengeData, predictions, output_dir: str | Path) -> dict:
    """Use for core and extra runs. Does not fit or alter the model."""
    pred = validate_predictions(predictions, len(data.X_test))
    metrics = evaluate(data.y_test, pred)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_predictions(output / "predictions.csv", data.test_ids, pred)
    report = classification_report(data.y_test, pred, labels=[0, 1], digits=4, zero_division=0)
    (output / "classification_report.txt").write_text(report, encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    wrong = pd.DataFrame({"id": data.test_ids, "true_label": data.y_test,
                          "predicted_label": pred, "text": data.X_test})
    wrong.loc[wrong.true_label != wrong.predicted_label].to_csv(output / "errors.csv", index=False, encoding="utf-8")
    return metrics
