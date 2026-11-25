from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


class BaselineNB:
    def __init__(self):
        self.pipeline = Pipeline([
            ("vectorizer", CountVectorizer(max_features=10000, min_df=5)),
            ("classifier", MultinomialNB(alpha=1.0)),
        ])

    def fit(self, texts, labels):
        self.pipeline.fit(texts, labels)
        return self

    def predict(self, texts):
        return self.pipeline.predict(texts).astype(int)
