from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, f1_score

from models.baseline import BaselineNB
from models.bow_lr import BoWLR
from models.tfidf_lr import TfidfLR


DATA_DIR = Path("data")
MODEL_CLASSES = {
    "Baseline NB": BaselineNB,
    "BoW LR": BoWLR,
    "TF-IDF LR": TfidfLR,
}
CHALLENGES = ["A", "B", "C"]


def main():
    reports = []
    predictions_rows = []
    errors_rows = []
    macro_f1_results = {model_name: {} for model_name in MODEL_CLASSES}

    for challenge in CHALLENGES:
        df_train = pd.read_csv(DATA_DIR / challenge / "train.csv")
        df_test = pd.read_csv(DATA_DIR / challenge / "test.csv")

        train_texts = df_train["text"].tolist()
        train_labels = df_train["label"].astype(int).tolist()
        test_texts = df_test["text"].tolist()
        y_test = df_test["label"].astype(int).to_numpy()

        for model_name, model_class in MODEL_CLASSES.items():
            model = model_class()
            model.fit(train_texts, train_labels)
            y_pred = model.predict(test_texts)

            macro_f1 = f1_score(y_test, y_pred, average="macro")
            macro_f1_results[model_name][challenge] = macro_f1
            report = classification_report(y_test, y_pred)
            reports.append(f"Challenge {challenge} - {model_name}\n{report}Macro-F1: {macro_f1:.6f}\n")

            for index, prediction in enumerate(y_pred):
                row = df_test.iloc[index]
                predictions_rows.append({
                    "challenge": challenge,
                    "model": model_name,
                    "id": row["id"],
                    "label": int(prediction),
                })
                if int(prediction) != int(y_test[index]):
                    errors_rows.append({
                        "challenge": challenge,
                        "model": model_name,
                        "id": row["id"],
                        "true_label": int(y_test[index]),
                        "predicted_label": int(prediction),
                        "text": row["text"],
                    })

    results_df = pd.DataFrame(macro_f1_results).T[["A", "B", "C"]]
    results_df.index.name = "model"
    results_df.to_csv("macro_f1_results1.csv")
    pd.DataFrame(predictions_rows).to_csv("predictions1.csv", index=False)
    pd.DataFrame(errors_rows).to_csv("errors1.csv", index=False)
    Path("classification_report1.txt").write_text("\n".join(reports), encoding="utf-8")

    print("Macro-F1 results")
    print(results_df)


if __name__ == "__main__":
    main()
