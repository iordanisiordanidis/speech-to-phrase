"""Tests."""

import csv
from pathlib import Path
from typing import List

from speech_to_phrase import Settings

TESTS_DIR = Path(__file__).parent
ROOT_DIR = TESTS_DIR.parent

LOCAL_DIR = ROOT_DIR / "local"
SETTINGS = Settings(
    models_dir=LOCAL_DIR / "models",
    train_dir=TESTS_DIR / "train",
    tools_dir=LOCAL_DIR,
    hass_token="",
    hass_websocket_uri="",
    retrain_on_connect=False,
    custom_sentences_dirs=[TESTS_DIR / "custom_sentences"],
)


def load_test_sentences(language: str) -> List[str]:
    """Load test sentences for a language from a CSV file."""
    sentences: List[str] = []

    csv_path = TESTS_DIR / "test_sentences" / f"{language}.csv"
    with open(csv_path, "r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            sentences.append(row["sentence"])

    return sentences
