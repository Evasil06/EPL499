"""Member 3: hand-engineered stylometric features + Logistic Regression.

Each text is turned into a fixed vector of 20 numbers that describe *how* it is
written (length, vocabulary, sentence rhythm, punctuation, layout, typical
"AI style" words) rather than *what* it is about. Features are standardised
with a scaler fitted on the training texts only, then fed to the shared LR.

All feature definitions and the reasoning behind each live in FEATURES below,
so the report table can be generated straight from this file.
"""
import math
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import make_logistic_regression

# --- tokenisation helpers (kept simple and deterministic) -------------------
WORD_RE = re.compile(r"[A-Za-zÀ-ɏ]+(?:['’][A-Za-z]+)?|\d+")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
SPECIAL_PUNCT = set(";:()[]\"'“”‘’")
DASHES = ("—", "–", " - ")  # em dash, en dash, spaced hyphen
FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
# Words that are over-used by LLM editing/generation. Our own list.
AI_MARKERS = {
    "additionally", "furthermore", "moreover", "notably", "overall", "crucial",
    "essential", "vital", "pivotal", "significant", "significantly", "various",
    "ensure", "ensuring", "enhance", "enhancing", "comprehensive", "robust",
    "seamless", "seamlessly", "utilize", "utilizing", "foster", "fostering",
    "delve", "landscape", "realm", "showcase", "showcasing", "highlighting",
    "underscore", "underscores", "valuable", "effectively", "particularly",
}
MATTR_WINDOW = 50

# name -> (group, definition, why it may help)
FEATURES = {
    "log_chars": ("length", "log(1 + number of characters)",
                  "AI rewriting tends to compress or expand text; length shifts between classes."),
    "log_words": ("length", "log(1 + number of word tokens)",
                  "Same idea at word level; deliberately kept to study its correlation with log_chars."),
    "avg_word_len": ("lexical", "mean characters per word",
                     "LLMs prefer longer, Latinate words (e.g. 'utilize' over 'use')."),
    "long_word_ratio": ("lexical", "share of words with 8 or more letters",
                        "Captures the same 'fancier vocabulary' tendency, less sensitive to outliers."),
    "mattr": ("lexical", f"moving-average type/token ratio over {MATTR_WINDOW}-word windows",
              "Vocabulary richness that does not fall with length; AI text is often more repetitive."),
    "hapax_ratio": ("lexical", "share of distinct words that occur exactly once",
                    "Human writing contains more one-off, idiosyncratic words."),
    "stopword_ratio": ("lexical", "share of words that are English stopwords",
                       "Function-word usage is a classic authorship signal that is hard to fake."),
    "avg_sent_len": ("rhythm", "mean words per sentence",
                     "LLM sentences cluster around a 'comfortable' medium length."),
    "sent_len_cv": ("rhythm", "std / mean of sentence lengths (burstiness)",
                    "Humans mix very short and very long sentences; AI is more uniform."),
    "comma_rate": ("punctuation", "commas per word",
                   "AI polishing adds subordinate clauses and lists, i.e. more commas."),
    "special_punct_rate": ("punctuation", "; : ( ) [ ] quotes per character",
                           "Human/encyclopedic text uses more brackets, quotes and semicolons."),
    "dash_rate": ("punctuation", "em/en dashes and spaced hyphens per 100 words",
                  "Em dashes are a well-known LLM habit; humanisers may remove or replace them."),
    "question_excl_rate": ("punctuation", "? and ! per sentence",
                           "Conversational human text asks and exclaims more."),
    "newline_rate": ("layout", "line breaks per 100 words",
                     "Original layout (lists, headings) is often merged into prose by AI editing."),
    "short_line_ratio": ("layout", "share of non-empty lines with fewer than 5 words",
                         "Headings / list items such as 'History' or 'References' signal untouched human text."),
    "digit_ratio": ("surface", "digits per character",
                    "Dates, figures and references are dropped or rounded during rewriting."),
    "upper_ratio": ("surface", "uppercase letters per letter",
                    "Proper nouns, acronyms and title case reflect factual, human-sourced text."),
    "first_person_rate": ("style", "first-person pronouns per word",
                          "Personal voice is a human trait that AI rewriting tends to neutralise."),
    "contraction_rate": ("style", "words with an apostrophe (don't, it's) per word",
                         "Informal contractions are expanded by polishing, re-added by humanisers."),
    "ai_marker_rate": ("style", "AI-style marker words per 100 words (own lexicon)",
                       "Words such as 'additionally', 'crucial', 'ensure' are over-represented in LLM output."),
}
FEATURE_NAMES = list(FEATURES)


def _safe_div(a, b):
    return a / b if b else 0.0


def _mattr(words, window=MATTR_WINDOW):
    if not words:
        return 0.0
    if len(words) <= window:
        return len(set(words)) / len(words)
    counts = Counter(words[:window])
    total = len(counts)
    for i in range(window, len(words)):
        out, inn = words[i - window], words[i]
        counts[out] -= 1
        if counts[out] == 0:
            del counts[out]
        counts[inn] += 1
        total += len(counts)
    return total / (len(words) - window + 1)


def text_features(text: str) -> dict:
    """Compute all features for one raw text. Works for empty strings."""
    text = text or ""
    words = WORD_RE.findall(text)
    lower = [w.lower() for w in words]
    n_words, n_chars = len(words), len(text)
    letters = [c for c in text if c.isalpha()]
    sentences = [s for s in SENT_SPLIT_RE.split(text) if WORD_RE.search(s)]
    sent_lens = np.array([len(WORD_RE.findall(s)) for s in sentences], dtype=float)
    lines = [ln for ln in text.split("\n") if ln.strip()]
    counts = Counter(lower)
    n_sent = max(len(sentences), 1)

    return {
        "log_chars": math.log1p(n_chars),
        "log_words": math.log1p(n_words),
        "avg_word_len": _safe_div(sum(len(w) for w in words), n_words),
        "long_word_ratio": _safe_div(sum(len(w) >= 8 for w in words), n_words),
        "mattr": _mattr(lower),
        "hapax_ratio": _safe_div(sum(v == 1 for v in counts.values()), len(counts)),
        "stopword_ratio": _safe_div(sum(w in ENGLISH_STOP_WORDS for w in lower), n_words),
        "avg_sent_len": float(sent_lens.mean()) if len(sent_lens) else 0.0,
        "sent_len_cv": float(sent_lens.std() / sent_lens.mean()) if len(sent_lens) and sent_lens.mean() else 0.0,
        "comma_rate": _safe_div(text.count(","), n_words),
        "special_punct_rate": _safe_div(sum(c in SPECIAL_PUNCT for c in text), n_chars),
        "dash_rate": 100 * _safe_div(sum(text.count(d) for d in DASHES), n_words),
        "question_excl_rate": _safe_div(text.count("?") + text.count("!"), n_sent),
        "newline_rate": 100 * _safe_div(text.count("\n"), n_words),
        "short_line_ratio": _safe_div(sum(len(WORD_RE.findall(ln)) < 5 for ln in lines), len(lines)),
        "digit_ratio": _safe_div(sum(c.isdigit() for c in text), n_chars),
        "upper_ratio": _safe_div(sum(c.isupper() for c in letters), len(letters)),
        "first_person_rate": _safe_div(sum(w in FIRST_PERSON for w in lower), n_words),
        "contraction_rate": _safe_div(sum(("'" in w or "’" in w) for w in words), n_words),
        "ai_marker_rate": 100 * _safe_div(sum(w in AI_MARKERS for w in lower), n_words),
    }


def extract_features(texts, names=None) -> np.ndarray:
    """Stateless: returns an (n_texts, n_features) matrix in FEATURE_NAMES order."""
    names = list(names or FEATURE_NAMES)
    rows = [text_features(t) for t in texts]
    return np.array([[r[n] for n in names] for r in rows], dtype=float).reshape(len(rows), len(names))


class FeaturesLR:
    """Engineered features -> StandardScaler (train-fitted) -> shared LR.

    `features` (optional) selects a subset of FEATURE_NAMES; used only by the
    member-3 contribution experiments. The core run uses all features.
    """

    def __init__(self, features=None):
        self.features = list(features) if features is not None else list(FEATURE_NAMES)
        unknown = set(self.features) - set(FEATURE_NAMES)
        if unknown or not self.features:
            raise ValueError(f"Invalid feature selection: {sorted(unknown) or 'empty'}")

    def _pipeline(self):
        return make_pipeline(StandardScaler(), make_logistic_regression())

    def fit(self, texts, labels):
        return self.fit_matrix(extract_features(texts, self.features), labels)

    def fit_matrix(self, X, labels):
        """Fit on an already-extracted training matrix (experiments reuse it)."""
        self.pipeline_ = self._pipeline().fit(X, np.asarray(labels))
        return self

    def predict(self, texts):
        return self.predict_matrix(extract_features(texts, self.features))

    def predict_matrix(self, X):
        return self.pipeline_.predict(X).astype(np.int64)

    def coefficients(self):
        """Standardised LR weights (positive pushes towards label 1)."""
        return dict(zip(self.features, self.pipeline_[-1].coef_[0]))
