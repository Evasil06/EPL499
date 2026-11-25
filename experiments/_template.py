"""Copy to experiments/member2.py or member3.py (member1.py is optional)."""
from pathlib import Path
from config import seed_everything
from data_loader import load_data
from evaluation import evaluate_and_save


def run(data_dir: Path, output_dir: Path, challenges: tuple[str, ...]):
    for challenge in challenges:
        data = load_data(challenge, data_dir)
        seed_everything()
        # For each controlled variant, create a fresh model, fit train only,
        # predict validation, then call:
        # evaluate_and_save(data, predictions, output_dir / "variant_name" / challenge)
        # Write a comparison table and document the variable being changed.
        raise NotImplementedError
