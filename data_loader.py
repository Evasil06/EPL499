"""Read supplied CSVs without text cleaning, shuffling, or resplitting."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from config import CHALLENGES


@dataclass(frozen=True)
class ChallengeData:
    challenge: str
    X_train: list[str]
    y_train: np.ndarray
    X_test: list[str]
    y_test: np.ndarray
    test_ids: list[str]


def _read(path: Path, *, test: bool) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"id": "string", "text": "string"},
                        keep_default_na=False, encoding="utf-8-sig")
    required = {"text", "label"} | ({"id"} if test else set())
    if missing := required - set(frame.columns):
        raise ValueError(f"{path}: missing columns {sorted(missing)}; found {list(frame.columns)}")
    if frame.empty:
        raise ValueError(f"{path}: empty dataset")
    labels = pd.to_numeric(frame["label"], errors="coerce")
    if not labels.isin([0, 1]).all():
        raise ValueError(f"{path}: labels must be 0 or 1")
    if test and frame["id"].str.strip().eq("").any():
        raise ValueError(f"{path}: blank test ID")
    frame["label"] = labels.astype("int64")
    return frame


def load_data(challenge: str, data_dir: str | Path = "data") -> ChallengeData:
    if challenge not in CHALLENGES:
        raise ValueError(f"Unknown challenge: {challenge}")
    folder = Path(data_dir) / challenge
    train = _read(folder / "train.csv", test=False)
    test = _read(folder / "test.csv", test=True)
    if set(train["label"]) != {0, 1}:
        raise ValueError(f"{folder}: training data must include both classes")
    return ChallengeData(challenge, train["text"].tolist(), train["label"].to_numpy(),
                         test["text"].tolist(), test["label"].to_numpy(), test["id"].tolist())
