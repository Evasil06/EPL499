from sklearn.linear_model import LogisticRegression


SEED = 42


def make_logistic_regression():
    return LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        class_weight=None,
        random_state=SEED,
    )
