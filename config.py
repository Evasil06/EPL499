"""Shared conventions. Change only by group agreement."""
import random
import numpy as np
from sklearn.linear_model import LogisticRegression

SEED = 42
CHALLENGES = ("A", "B", "C")
LR_PARAMS = dict(C=1.0, solver="lbfgs", max_iter=1000,
                 class_weight=None, random_state=SEED)

# Registry order is also the tie-break order for final predictions.
MODEL_REGISTRY = {
    "baseline": ("models.baseline", "BaselineNB", 1),
    "bow_lr": ("models.bow_lr", "BoWLR", 1),
    "tfidf_lr": ("models.tfidf_lr", "TfidfLR", 1),
    "word2vec_lr": ("models.word2vec_lr", "Word2VecLR", 2),
    "features_lr": ("models.features_lr", "FeaturesLR", 3),
}


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)


def make_logistic_regression(**overrides):
    """Use without overrides for the core comparison; document extras."""
    return LogisticRegression(**(LR_PARAMS | overrides))
