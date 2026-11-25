"""Copy this to your assigned module; implement and rename the class."""
import numpy as np
from config import make_logistic_regression, SEED


class YourModel:
    def __init__(self):
        # Baseline uses MultinomialNB instead, with its exact required settings.
        self.classifier = make_logistic_regression()

    def fit(self, texts: list[str], labels: np.ndarray):
        # Fit representation, any scaler, and classifier using ONLY these inputs.
        raise NotImplementedError

    def predict(self, texts: list[str]) -> np.ndarray:
        # Transform with fitted state; return one integer 0/1 per input text.
        # Do not refit, reorder or drop inputs; support empty/unknown-token texts.
        raise NotImplementedError
