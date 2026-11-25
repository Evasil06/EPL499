from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from config import make_logistic_regression


class BoWLR:
    def __init__(self):
        self.pipeline = Pipeline([
            ("vectorizer", CountVectorizer(max_features=10000, min_df=5)),
            ("classifier", make_logistic_regression()),
        ])

    def fit(self, texts, labels):
        self.pipeline.fit(texts, labels)
        return self

    def predict(self, texts):
        return self.pipeline.predict(texts).astype(int)
