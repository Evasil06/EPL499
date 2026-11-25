"""Member 3: engineered features + Logistic Regression.

Follows the LAB6 approach: tokenize the raw text with NLTK, compute simple
feature functions that each return a dictionary, turn the dictionaries into a
pandas DataFrame (one column per feature) and train a classifier on it.

Pipeline: raw text -> 19 features -> StandardScaler (fitted on train only)
          -> Logistic Regression (shared factory from config.py).
"""
from collections import Counter
from math import log
import string

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import make_logistic_regression


def download_nltk_resources():
    """Same resources as LAB2/LAB6; downloaded once, only when first needed."""
    for resource, path in [("punkt_tab", "tokenizers/punkt_tab"), ("stopwords", "corpora/stopwords")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


# --- word lists (like the opinion lexicon in LAB6) --------------------------
FIRST_PERSON_WORDS = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
# word_tokenize splits "don't" into "do" + "n't", so contractions appear as these tokens
CONTRACTION_TOKENS = {"n't", "'s", "'re", "'ll", "'ve", "'m", "'d"}
# Our own list of words that LLMs tend to over-use when writing or polishing text.
AI_WORDS = {
    "additionally", "furthermore", "moreover", "notably", "overall", "crucial",
    "essential", "vital", "pivotal", "significant", "significantly", "various",
    "ensure", "ensuring", "enhance", "enhancing", "comprehensive", "robust",
    "seamless", "seamlessly", "utilize", "utilizing", "foster", "fostering",
    "delve", "landscape", "realm", "showcase", "showcasing", "highlighting",
    "underscore", "underscores", "valuable", "effectively", "particularly",
}
DASHES = {"—", "–"}  # em dash, en dash

# Feature name -> (group, how it is computed, why it may help)
FEATURES = {
    "log_number_tokens": ("length", "log(1 + number of tokens)",
                          "AI rewriting compresses or expands text, so length differs between classes."),
    "log_number_characters": ("length", "log(1 + number of characters)",
                              "Same idea in characters; kept to study its correlation with log_number_tokens."),
    "avg_word_length": ("lexical", "mean number of letters per word",
                        "LLMs prefer longer words (e.g. 'utilize' instead of 'use')."),
    "long_words_ratio": ("lexical", "share of words with 8 or more letters",
                         "Same 'fancier vocabulary' tendency, less affected by a few very long words."),
    "type_token_ratio": ("lexical", "unique words / total words",
                         "Vocabulary richness: AI text tends to repeat the same words more."),
    "stopwords_ratio": ("lexical", "share of words that are NLTK English stopwords",
                        "Function-word usage is a classic authorship signal that is hard to fake."),
    "avg_sentence_length": ("sentence", "mean number of tokens per sentence (sent_tokenize)",
                            "LLM sentences cluster around a comfortable medium length."),
    "sentence_length_std": ("sentence", "standard deviation of tokens per sentence",
                            "Humans mix very short and very long sentences; AI is more uniform."),
    "commas_ratio": ("punctuation", "commas / tokens",
                     "AI polishing adds clauses and lists, so more commas."),
    "punctuation_ratio": ("punctuation", "punctuation tokens (string.punctuation) / tokens",
                          "Overall punctuation density differs between human and machine text."),
    "existence_!": ("punctuation", "1 if the text contains '!', else 0 (as in LAB6)",
                    "Exclamations are a human, emotional trait that AI editing removes."),
    "existence_?": ("punctuation", "1 if the text contains '?', else 0",
                    "Questions to the reader are typical of conversational human text."),
    "existence_dash": ("punctuation", "1 if the text contains an em or en dash, else 0",
                       "Em dashes are a well-known LLM habit."),
    "newlines_ratio": ("layout", "line breaks / tokens",
                       "Original layout (lists, headings) is often merged into prose by AI editing."),
    "digits_ratio": ("surface", "tokens containing a digit / tokens",
                     "Dates, figures and references are dropped or rounded during rewriting."),
    "capitalized_ratio": ("surface", "words starting with a capital letter / words",
                          "Proper nouns and names reflect factual, human-sourced text."),
    "first_person_ratio": ("style", "first-person pronouns (I, we, my...) / words",
                           "Personal voice is a human trait that AI rewriting tends to neutralise."),
    "contractions_ratio": ("style", "contraction tokens (n't, 's, 're...) / tokens",
                           "Polishing expands contractions; humanisers may add them back."),
    "ai_words_ratio": ("style", "words from our AI_WORDS list per 100 words",
                       "Words like 'additionally', 'crucial', 'ensure' are over-used by LLMs."),
}
FEATURE_NAMES = list(FEATURES)


# --- feature functions: each takes the text/tokens and returns a dictionary --
def length_features(text, tokens):
    return {"log_number_tokens": log(1 + len(tokens)),
            "log_number_characters": log(1 + len(text))}


def lexical_features(words, stop_words):
    n = len(words)
    if n == 0:
        return {"avg_word_length": 0, "long_words_ratio": 0, "type_token_ratio": 0, "stopwords_ratio": 0}
    return {"avg_word_length": sum(len(w) for w in words) / n,
            "long_words_ratio": sum(1 for w in words if len(w) >= 8) / n,
            "type_token_ratio": len(set(words)) / n,
            "stopwords_ratio": sum(1 for w in words if w in stop_words) / n}


def sentence_features(sentence_tokens):
    lengths = [len(tokens) for tokens in sentence_tokens]
    if not lengths:
        return {"avg_sentence_length": 0, "sentence_length_std": 0}
    return {"avg_sentence_length": float(np.mean(lengths)),
            "sentence_length_std": float(np.std(lengths))}


def punctuation_features(text, tokens):
    n = max(len(tokens), 1)
    return {"commas_ratio": tokens.count(",") / n,
            "punctuation_ratio": sum(1 for t in tokens if t in string.punctuation) / n,
            "existence_!": 1 if "!" in text else 0,
            "existence_?": 1 if "?" in text else 0,
            "existence_dash": 1 if any(d in text for d in DASHES) else 0}


def layout_surface_features(text, tokens, raw_words):
    n = max(len(tokens), 1)
    return {"newlines_ratio": text.count("\n") / n,
            "digits_ratio": sum(1 for t in tokens if any(c.isdigit() for c in t)) / n,
            "capitalized_ratio": sum(1 for w in raw_words if w[0].isupper()) / max(len(raw_words), 1)}


def style_features(tokens, words):
    n_words = max(len(words), 1)
    counts = Counter(words)
    return {"first_person_ratio": sum(counts[w] for w in FIRST_PERSON_WORDS) / n_words,
            "contractions_ratio": sum(1 for t in tokens if t.lower() in CONTRACTION_TOKENS) / max(len(tokens), 1),
            "ai_words_ratio": 100 * sum(counts[w] for w in AI_WORDS) / n_words}


def extract_features(text, stop_words):
    """All features of one raw text as a dictionary (like count_lexical_features in LAB6)."""
    text = text or ""
    # Only tokenization, as in LAB6 for 'no' / '!'. Splitting into sentences first
    # and tokenizing each one is exactly what word_tokenize(text) does internally.
    sentence_tokens = [word_tokenize(s) for s in sent_tokenize(text)]
    tokens = [t for sentence in sentence_tokens for t in sentence]
    raw_words = [t for t in tokens if t.isalpha()]  # remove punctuation and digits (LAB2/LAB6)
    words = [w.lower() for w in raw_words]          # lowercase (LAB2/LAB6)

    features = {}
    features.update(length_features(text, tokens))
    features.update(lexical_features(words, stop_words))
    features.update(sentence_features(sentence_tokens))
    features.update(punctuation_features(text, tokens))
    features.update(layout_surface_features(text, tokens, raw_words))
    features.update(style_features(tokens, words))
    return features


def extract_features_df(texts):
    """One row per text, one column per feature (as in LAB6: list of dicts -> DataFrame)."""
    download_nltk_resources()
    stop_words = set(stopwords.words("english"))
    return pd.DataFrame([extract_features(t, stop_words) for t in texts], columns=FEATURE_NAMES)


class FeaturesLR:
    """Engineered features -> StandardScaler -> Logistic Regression.

    The features have very different scales (a 0/1 existence feature next to a
    log length), so they are standardised before LR. The scaler is fitted on
    the training set only. `features` (optional) selects a subset of columns;
    it is only used by the member-3 experiments.
    """

    def __init__(self, features=None):
        self.features = list(features) if features is not None else list(FEATURE_NAMES)
        unknown = set(self.features) - set(FEATURE_NAMES)
        if unknown or not self.features:
            raise ValueError(f"Invalid feature selection: {sorted(unknown) or 'empty'}")

    def fit(self, texts, labels):
        return self.fit_features(extract_features_df(texts), labels)

    def predict(self, texts):
        return self.predict_features(extract_features_df(texts))

    def fit_features(self, X_train, y_train):
        """Fit on an already-extracted feature DataFrame (the experiments reuse it)."""
        self.clf = make_pipeline(StandardScaler(), make_logistic_regression())
        self.clf.fit(X_train[self.features], np.asarray(y_train))
        return self

    def predict_features(self, X_test):
        return self.clf.predict(X_test[self.features]).astype(np.int64)

    def coefficients(self):
        """LR weight of each (standardised) feature; positive pushes towards label 1."""
        return pd.Series(self.clf[-1].coef_[0], index=self.features)
