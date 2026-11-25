"""Train Word2Vec and Logistic Regression on one challenge's training texts."""

import re

import numpy as np
from gensim.models import Word2Vec

from config import SEED, make_logistic_regression


def _tokenize(text: str) -> list[str]:
    """Lowercase and keep alphabetic word tokens, including stopwords."""
    tokens = re.findall(r"\w+", text.lower())
    return [token for token in tokens if token.isalpha()]


def _document_vector(tokens, word_vectors) -> np.ndarray:
    vectors = []
    for token in tokens:
        if token in word_vectors:
            vectors.append(word_vectors[token])

    if not vectors:
        return np.zeros(word_vectors.vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0)


class Word2VecLR:
    def __init__(self):
        self._word2vec = None
        self._classifier = None

    def fit(self, texts: list[str], labels: np.ndarray):
        token_lists = [_tokenize(text) for text in texts]
        word2vec = Word2Vec(
            sentences=token_lists,
            vector_size=100,
            window=5,
            min_count=2,
            sg=1,
            epochs=10,
            workers=1,
            seed=SEED,
        )
        document_vectors = [_document_vector(tokens, word2vec.wv) for tokens in token_lists]
        classifier = make_logistic_regression()
        classifier.fit(document_vectors, labels)

        self._word2vec = word2vec
        self._classifier = classifier
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        if self._word2vec is None:
            raise RuntimeError("Fit Word2VecLR before calling predict().")
        if len(texts) == 0:
            return np.empty(0, dtype=np.int64)

        word_vectors = self._word2vec.wv
        document_vectors = [_document_vector(_tokenize(text), word_vectors) for text in texts]
        return np.asarray(self._classifier.predict(document_vectors), dtype=np.int64)
