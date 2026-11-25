import unittest
from unittest.mock import patch

import numpy as np

from evaluation import validate_predictions
from main import new_model
from models.word2vec_lr import Word2VecLR, _document_vector, _tokenize


class Word2VecLRTests(unittest.TestCase):
    def setUp(self):
        self.texts = ["The river flows.", "river flows", "The ocean waves.", "ocean waves", ""]
        self.labels = np.array([0, 0, 1, 1, 0])
        self.model = new_model("word2vec_lr")

    def test_registry_fit_and_predict(self):
        self.assertIsInstance(self.model, Word2VecLR)
        self.assertIs(self.model.fit(self.texts, self.labels), self.model)

        texts = ["OCEAN waves!", "", "unseen vocabulary", "River flows."]
        predictions = self.model.predict(texts)
        validate_predictions(predictions, len(texts))
        np.testing.assert_array_equal(predictions, [self.model.predict([text])[0] for text in texts])
        validate_predictions(self.model.predict([]), 0)

    def test_known_and_unknown_document_vectors(self):
        self.model.fit(self.texts, self.labels)
        word_vectors = self.model._word2vec.wv
        self.assertEqual(_tokenize("The RIVER, and R2D2!"), ["the", "river", "and"])

        expected = np.mean([word_vectors["river"], word_vectors["river"], word_vectors["ocean"]], axis=0)
        mixed = _document_vector(["river", "river", "unseen", "ocean"], word_vectors)
        self.assertEqual(mixed.shape, (100,))
        np.testing.assert_array_equal(mixed, expected)
        for tokens in ([], ["unseen"]):
            np.testing.assert_array_equal(_document_vector(tokens, word_vectors), np.zeros(100))

    def test_prediction_uses_frozen_state(self):
        self.model.fit(self.texts, self.labels)
        word2vec = self.model._word2vec
        vocabulary = word2vec.wv.key_to_index.copy()
        vectors = word2vec.wv.vectors.copy()

        with (
            patch("models.word2vec_lr.Word2Vec") as constructor,
            patch.object(word2vec, "build_vocab") as build_vocab,
            patch.object(word2vec, "train") as train,
            patch.object(self.model._classifier, "fit") as fit,
        ):
            self.model.predict(["river unseen unseen", "ocean unseen", ""])
        constructor.assert_not_called()
        build_vocab.assert_not_called()
        train.assert_not_called()
        fit.assert_not_called()
        self.assertEqual(word2vec.wv.key_to_index, vocabulary)
        np.testing.assert_array_equal(word2vec.wv.vectors, vectors)
