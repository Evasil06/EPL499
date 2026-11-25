"""Structural interface; models do not have to inherit from this protocol."""
from typing import Protocol, Self
import numpy as np


class TextClassifier(Protocol):
    def fit(self, texts: list[str], labels: np.ndarray) -> Self: ...
    def predict(self, texts: list[str]) -> np.ndarray: ...
