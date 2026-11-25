from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


class TfidfLR:
    def __init__(self):
        self.pipeline = Pipeline([
            ("vectorizer", TfidfVectorizer(max_features=10000, min_df=5)),
            ("classifier", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, class_weight=None, random_state=42)),
        ])

    def fit(self, texts, labels):
        self.pipeline.fit(texts, labels)
        return self

    def predict(self, texts):
        return self.pipeline.predict(texts).astype(int)
