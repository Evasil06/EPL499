"""Run selected core models; optionally run member-owned extra experiments."""
import argparse
import importlib
from pathlib import Path
import numpy as np
import pandas as pd
from config import CHALLENGES, MODEL_REGISTRY, seed_everything
from data_loader import load_data
from evaluation import evaluate_and_save, save_predictions, validate_predictions


def new_model(name):
    module_name, class_name, _ = MODEL_REGISTRY[name]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise RuntimeError(f"Missing {module_name.replace('.', '/')}.py; ask its assigned member to implement {class_name}.") from exc
        raise
    return getattr(module, class_name)()


def run(models, challenges, data_dir, output_dir, extras=False):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, best = [], {}
    full = tuple(models) == tuple(MODEL_REGISTRY) and tuple(challenges) == CHALLENGES
    for challenge in challenges:
        data = load_data(challenge, data_dir)
        for name in models:
            seed_everything()
            model = new_model(name)  # Fresh model for every challenge.
            model.fit(data.X_train.copy(), data.y_train.copy())
            predictions = validate_predictions(model.predict(data.X_test.copy()), len(data.X_test))
            metrics = evaluate_and_save(data, predictions, output / "core" / name / challenge)
            score = metrics["macro_f1"]
            rows.append({"model": name, "challenge": challenge, "macro_f1": score,
                         "accuracy": metrics["accuracy"],
                         "f1_label_0": metrics["classification_report"]["0"]["f1-score"],
                         "f1_label_1": metrics["classification_report"]["1"]["f1-score"]})
            print(f"{name:15} {challenge}: macro-F1={score:.4f}")
            # Strict > gives ties to the first model in registry order.
            if challenge not in best or score > best[challenge][0]:
                best[challenge] = (score, name, data.test_ids, predictions.copy())
    summary = pd.DataFrame(rows)
    model_tag = "all" if tuple(models) == tuple(MODEL_REGISTRY) else "-".join(models)
    challenge_tag = "all" if tuple(challenges) == CHALLENGES else "-".join(challenges)
    summary.to_csv(output / f"summary_{model_tag}_{challenge_tag}.csv", index=False)
    if full:
        pivot = summary.pivot(index="model", columns="challenge", values="macro_f1").reindex(list(MODEL_REGISTRY))
        pivot["average"] = pivot[list(CHALLENGES)].mean(axis=1)
        pivot.to_csv(output / "comparison.csv")
        selected = []
        for challenge, (score, name, ids, predictions) in best.items():
            save_predictions(output / f"predictions_{challenge}.csv", ids, predictions)
            selected.append({"challenge": challenge, "model": name, "validation_macro_f1": score})
        pd.DataFrame(selected).to_csv(output / "selected_models.csv", index=False)
    if extras:
        owners = sorted({MODEL_REGISTRY[name][2] for name in models})
        for owner in owners:
            module_name = f"experiments.member{owner}"
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError as exc:
                if exc.name != module_name:
                    raise
                if owner == 1:  # His/her core comparisons already run above.
                    continue
                raise RuntimeError(f"Missing {module_name.replace('.', '/')}.py for required comparative experiments.") from exc
            seed_everything()
            target = output / "extras" / f"member{owner}"
            target.mkdir(parents=True, exist_ok=True)
            module.run(data_dir=Path(data_dir), output_dir=target, challenges=tuple(challenges))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["all", *MODEL_REGISTRY], default="all")
    parser.add_argument("--challenge", choices=["all", *CHALLENGES], default="all")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="results")
    parser.add_argument("--extras", action="store_true", help="Also run member experiment hooks")
    args = parser.parse_args()
    models = tuple(MODEL_REGISTRY) if args.model == "all" else (args.model,)
    challenges = CHALLENGES if args.challenge == "all" else (args.challenge,)
    run(models, challenges, args.data_dir, args.output, args.extras)


if __name__ == "__main__":
    main()
