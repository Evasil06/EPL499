"""Run the original Word2VecLR and three controlled comparisons on EPL499."""

import argparse
import csv
import inspect
import json
import os
import platform
import random
import traceback
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
from gensim.models import Word2Vec

import config
from config import SEED, make_logistic_regression
from models.word2vec_lr import Word2VecLR, _document_vector, _tokenize


CHALLENGES = ("A", "B", "C")
EXPECTED_COUNTS = {
    "A": {"train": [5250, 5250], "validation": [1125, 1125]},
    "B": {"train": [5250, 10500], "validation": [1125, 2250]},
    "C": {"train": [5250, 10500], "validation": [1125, 2250]},
}
CORE_WORD2VEC = {
    "vector_size": 100,
    "window": 5,
    "min_count": 2,
    "sg": 1,
    "epochs": 10,
    "workers": 1,
    "seed": SEED,
}
EXPERIMENTS = {
    "original": {},
    "dimensions_300": {"vector_size": 300},
    "architecture_cbow": {"sg": 0},
    "context_window_10": {"window": 10},
}
RESEARCH_QUESTIONS = {
    "dimensions_300": (
        "Does increasing embedding width from 100 to 300 improve macro-F1? "
        "More dimensions let the model represent more distinctions between words "
        "but also increase the number of features supplied to Logistic Regression. "
        "Lab 8 cells 60 and 74 demonstrate the 100 versus 300 comparison."
    ),
    "architecture_cbow": (
        "Does predicting the target from surrounding words with CBOW change macro-F1? "
        "Skip-gram predicts surrounding words from the target. The objectives can "
        "learn different representations from the same token sequences. Lab 8 "
        "describes CBOW in cells 60 and 74. Training it is a project adaptation."
    ),
    "context_window_10": (
        "Does increasing the maximum context window from 5 to 10 improve macro-F1? "
        "A wider window can incorporate more topical context and make nearby "
        "word choices less prominent. The window comparison is a project adaptation."
    ),
}
TOKENIZATION = (
    "Lowercase, extract regex word runs, then keep alphabetic tokens only. "
    "Keep stopwords and repeats. Apostrophes and hyphens split word runs. "
    "No stemming or lemmatization. One token sequence per document."
)
POOLING = (
    "Mean of training-vocabulary token vectors, counting repeats. Ignore OOV "
    "tokens and use a fixed-width zero vector if no known token remains."
)


class ExperimentalWord2VecLR(Word2VecLR):
    """Reuse the original tokenizer, pooling and prediction in the variants."""

    def __init__(self, overrides):
        super().__init__()
        self._parameters = {**CORE_WORD2VEC, **overrides}

    def fit(self, texts, labels):
        token_lists = [_tokenize(text) for text in texts]
        word2vec = Word2Vec(sentences=token_lists, **self._parameters)
        document_vectors = [_document_vector(tokens, word2vec.wv) for tokens in token_lists]
        classifier = make_logistic_regression()
        classifier.fit(document_vectors, labels)
        self._word2vec = word2vec
        self._classifier = classifier
        return self


def shared_tools():
    # Current config.py lacks the constant imported by data_loader.py.
    # Supply it during the import only. The loader keeps its own imported value.
    added_challenges = not hasattr(config, "CHALLENGES")
    if added_challenges:
        config.CHALLENGES = CHALLENGES
    try:
        from data_loader import load_data
        from evaluation import evaluate_and_save
    finally:
        if added_challenges:
            del config.CHALLENGES
    return load_data, evaluate_and_save


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def record_dataset(data, data_dir, output_dir):
    counts = {}
    for split, filename, labels in (
        ("train", "train.csv", data.y_train),
        ("validation", "test.csv", data.y_test),
    ):
        counts[split] = {
            "file": (Path(data_dir) / data.challenge / filename).as_posix(),
            "rows": len(labels),
            "counts_by_label_0_1": np.bincount(labels, minlength=2).tolist(),
            "assignment_counts_by_label_0_1": EXPECTED_COUNTS[data.challenge][split],
        }
        counts[split]["matches_assignment"] = (
            counts[split]["counts_by_label_0_1"] == EXPECTED_COUNTS[data.challenge][split]
        )
    write_json(output_dir / f"dataset_{data.challenge}.json", counts)


def write_coverage(path, data, predictions, vocabulary):
    # Count token occurrences using the original tokenizer and frozen vocabulary.
    # This is an evaluation diagnostic. It never changes the fitted model.
    rows = []
    for doc_id, text, true, pred in zip(
        data.test_ids, data.X_test, data.y_test, predictions, strict=True
    ):
        tokens = _tokenize(text)
        known = sum(token in vocabulary for token in tokens)
        rows.append(
            {
                "id": doc_id, "true_label": true, "predicted_label": pred,
                "tokens": len(tokens), "known_tokens": known,
                "oov_tokens": len(tokens) - known,
            }
        )
    write_csv(path, list(rows[0]), rows)


def fit_and_evaluate(data, experiment, overrides, output, evaluate_and_save):
    random.seed(SEED)
    np.random.seed(SEED)
    model = ExperimentalWord2VecLR(overrides) if overrides else Word2VecLR()

    start = perf_counter()
    model.fit(data.X_train, data.y_train)
    training_seconds = perf_counter() - start

    start = perf_counter()
    predictions = model.predict(data.X_test)
    prediction_seconds = perf_counter() - start

    word2vec = model._word2vec
    effective = {
        name: getattr(word2vec, name, parameter.default)
        for name, parameter in inspect.signature(Word2Vec).parameters.items()
        if name not in {"sentences", "corpus_file", "callbacks", "hashfxn"}
    }
    effective["hashfxn"] = word2vec.hashfxn.__name__
    parameters = {name: effective[name] for name in CORE_WORD2VEC}
    if parameters != {**CORE_WORD2VEC, **overrides}:
        raise ValueError(f"Unexpected Word2Vec settings for {experiment}: {parameters}")

    metrics = evaluate_and_save(data, predictions, output)
    correct = int(np.count_nonzero(predictions == data.y_test))
    result = {
        "experiment": experiment,
        "challenge": data.challenge,
        "architecture": "Skip-gram" if parameters["sg"] == 1 else "CBOW",
        **parameters,
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "correct": correct,
        "incorrect": len(predictions) - correct,
        "training_seconds": training_seconds,
        "prediction_seconds": prediction_seconds,
    }
    if experiment == "original":
        write_coverage(output / "vocabulary_coverage.csv", data, predictions, word2vec.wv)
    write_json(
        output / "run.json",
        {
            **result,
            "word2vec": effective,
            "vocabulary_size": len(word2vec.wv),
            "logistic_regression": model._classifier.get_params(),
            "logistic_regression_iterations": model._classifier.n_iter_.tolist(),
            "comparison_overrides": overrides,
            "research_question_and_rationale": RESEARCH_QUESTIONS.get(experiment),
            "tokenization": TOKENIZATION,
            "pooling": POOLING,
        },
    )
    return result


def write_tables(output_dir, rows, challenges):
    if not rows:
        return
    write_csv(output_dir / "results.csv", list(rows[0]), rows)
    scores = {(row["experiment"], row["challenge"]): row["macro_f1"] for row in rows}
    comparison = []
    for name in EXPERIMENTS:
        values = [scores.get((name, challenge)) for challenge in challenges]
        if all(value is None for value in values):
            continue
        row = {"configuration": name}
        row.update({f"macro_f1_{challenge}": value for challenge, value in zip(challenges, values)})
        row["average_macro_f1"] = (
            sum(values) / len(values) if all(value is not None for value in values) else None
        )
        comparison.append(row)
    write_csv(output_dir / "comparison.csv", list(comparison[0]), comparison)


def run(data_dir: Path, output_dir: Path, challenges: tuple[str, ...]):
    load_data, evaluate_and_save = shared_tools()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: version(name)
                for name in ("numpy", "pandas", "scipy", "scikit-learn", "gensim")
            },
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "validation": "The supplied labeled test.csv is used for evaluation and comparison.",
            "timing": "Wall-clock fit and predict only. Excludes CSV I/O, evaluation and coverage.",
        },
    )
    datasets = {}
    failures = []
    for challenge in challenges:
        try:
            data = load_data(challenge, data_dir)
            record_dataset(data, data_dir, output_dir)
            datasets[challenge] = data
        except Exception:
            failures.append({"challenge": challenge, "traceback": traceback.format_exc()})
            print(f"Could not load challenge {challenge}.\n{failures[-1]['traceback']}", flush=True)

    rows = []
    for experiment, overrides in EXPERIMENTS.items():
        for challenge, data in datasets.items():
            print(f"Starting {experiment} / {challenge}", flush=True)
            try:
                result = fit_and_evaluate(
                    data, experiment, overrides, output_dir / experiment / challenge,
                    evaluate_and_save,
                )
            except Exception:
                failures.append(
                    {"experiment": experiment, "challenge": challenge, "traceback": traceback.format_exc()}
                )
                print(failures[-1]["traceback"], flush=True)
                continue
            rows.append(result)
            write_tables(output_dir, rows, challenges)
            print(
                f"{experiment} / {challenge}  macro-F1={result['macro_f1']:.6f}  "
                f"fit={result['training_seconds']:.2f}s  predict={result['prediction_seconds']:.2f}s",
                flush=True,
            )
    write_json(output_dir / "failures.json", failures)
    if failures:
        raise RuntimeError(f"{len(failures)} failed evaluations or loads. See {output_dir / 'failures.json'}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results/member2"))
    parser.add_argument("--challenge", choices=("all", *CHALLENGES), default="all")
    args = parser.parse_args()
    challenges = CHALLENGES if args.challenge == "all" else (args.challenge,)
    run(args.data_dir, args.output, challenges)


if __name__ == "__main__":
    main()
