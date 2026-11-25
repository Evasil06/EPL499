"""Member 3: which engineered features contribute, and why.

Every experiment changes ONE thing and keeps everything else fixed: same
train/test split, same features, StandardScaler fitted on train only, and the
same Logistic Regression (config.make_logistic_regression, no overrides).
A new model is trained for every variant.

Experiments, for each challenge:
  1. all_features       reference model (same as the core run)
  2. leave_one_out      drop one feature at a time
  3. single_feature     train on one feature alone
  4. groups             drop a whole group / keep only that group
  5. random_forest      feature importances, then LR on the top 10 only (as in LAB6)
  6. length_pair        the two correlated length features alone / together / dropped
  7. scaling            with vs without StandardScaler
  8. coefficients       LR weights of the reference model
  9. correlations       correlation matrix of the training features

Results are saved as CSV tables in output_dir.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from config import SEED, make_logistic_regression, seed_everything
from data_loader import load_data
from evaluation import evaluate, evaluate_and_save
from models.features_lr import FEATURE_NAMES, FEATURES, FeaturesLR, extract_features_df

TOP_N = 10
LENGTH_PAIR = ["log_number_tokens", "log_number_characters"]


def train_and_evaluate(X_train, y_train, X_test, y_test, features):
    """Train a new FeaturesLR on the given columns and evaluate it on the test set."""
    seed_everything()
    model = FeaturesLR(features=features).fit_features(X_train, y_train)
    y_pred = model.predict_features(X_test)
    return evaluate(y_test, y_pred), y_pred, model


def result_row(challenge, experiment, variant, metrics, reference_f1):
    report = metrics["classification_report"]
    return {"challenge": challenge, "experiment": experiment, "variant": variant,
            "macro_f1": metrics["macro_f1"], "difference_vs_all": metrics["macro_f1"] - reference_f1,
            "accuracy": metrics["accuracy"],
            "f1_label_0": report["0"]["f1-score"], "f1_label_1": report["1"]["f1-score"]}


def run(data_dir: Path, output_dir: Path, challenges: tuple[str, ...]):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    definitions = pd.DataFrame([{"feature": name, "group": group, "definition": how, "rationale": why}
                                for name, (group, how, why) in FEATURES.items()])
    definitions.to_csv(output_dir / "feature_definitions.csv", index=False)
    groups = definitions.groupby("group")["feature"].apply(list).to_dict()

    results, importances, coefficients = [], [], []
    for challenge in challenges:
        data = load_data(challenge, data_dir)
        # Features of one text never depend on other texts, so we extract them once
        # and reuse the DataFrames; every model below is still fitted on train only.
        X_train = extract_features_df(data.X_train)
        X_test = extract_features_df(data.X_test)
        y_train, y_test = data.y_train, data.y_test

        def add(experiment, variant, features, save=False):
            metrics, y_pred, model = train_and_evaluate(X_train, y_train, X_test, y_test, features)
            if save:
                evaluate_and_save(data, y_pred, output_dir / variant / challenge)
            results.append(result_row(challenge, experiment, variant, metrics, reference_f1))
            return model

        # 1. reference model with all features
        reference_metrics, y_pred, reference_model = train_and_evaluate(X_train, y_train, X_test, y_test, FEATURE_NAMES)
        reference_f1 = reference_metrics["macro_f1"]
        evaluate_and_save(data, y_pred, output_dir / "all_features" / challenge)
        results.append(result_row(challenge, "reference", "all_features", reference_metrics, reference_f1))
        print(f"[member3] {challenge} all features: macro-F1={reference_f1:.4f}")

        # 2. leave one feature out
        for feature in FEATURE_NAMES:
            add("leave_one_out", f"without_{feature}", [f for f in FEATURE_NAMES if f != feature])

        # 3. one feature alone
        for feature in FEATURE_NAMES:
            add("single_feature", f"only_{feature}", [feature])

        # 4. feature groups
        for group, group_features in groups.items():
            add("groups", f"without_{group}", [f for f in FEATURE_NAMES if f not in group_features], save=True)
            add("groups", f"only_{group}", group_features, save=True)

        # 5. Random Forest importances and LR on the top features (LAB6)
        forest = RandomForestClassifier(n_estimators=100, random_state=SEED)
        forest.fit(X_train, y_train)
        importance = pd.DataFrame({"feature": X_train.columns, "importance": forest.feature_importances_})
        importance = importance.sort_values(by="importance", ascending=False)
        importance.insert(0, "challenge", challenge)
        importances.append(importance)
        add("random_forest", f"top_{TOP_N}_random_forest", importance["feature"].head(TOP_N).tolist(), save=True)

        # 6. the two highly correlated length features
        add("length_pair", "only_log_number_tokens", ["log_number_tokens"])
        add("length_pair", "only_log_number_characters", ["log_number_characters"])
        pair_model = add("length_pair", "only_both_length_features", LENGTH_PAIR)
        add("length_pair", "without_both_length_features", [f for f in FEATURE_NAMES if f not in LENGTH_PAIR])

        # 7. scaling: same LR without StandardScaler
        seed_everything()
        unscaled = make_logistic_regression().fit(X_train, y_train)
        results.append(result_row(challenge, "scaling", "no_standard_scaler",
                                  evaluate(y_test, unscaled.predict(X_test)), reference_f1))

        # 8. coefficients (standardised, so they can be compared with each other)
        weights = reference_model.coefficients()
        pair_weights = pair_model.coefficients()
        coefficients.append(pd.DataFrame({
            "challenge": challenge, "feature": weights.index, "coefficient": weights.values,
            "coefficient_length_pair_only": [pair_weights.get(f, np.nan) for f in weights.index]}))

        # 9. correlations on the training features only
        X_train.corr().to_csv(output_dir / f"correlations_{challenge}.csv")

    results = pd.DataFrame(results)
    results.to_csv(output_dir / "all_experiments.csv", index=False)
    pd.concat(importances).to_csv(output_dir / "random_forest_importances.csv", index=False)
    pd.concat(coefficients).to_csv(output_dir / "coefficients.csv", index=False)

    # Easy-to-read tables for the report: one row per variant, one column per challenge
    for experiment in ["leave_one_out", "single_feature", "groups", "length_pair"]:
        table = results[results.experiment == experiment]
        column = "difference_vs_all" if experiment == "leave_one_out" else "macro_f1"
        table.pivot(index="variant", columns="challenge", values=column).to_csv(output_dir / f"{experiment}.csv")
