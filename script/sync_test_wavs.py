"""Synchronize test WAV files with test sentences.

Uses script/generate_test_wav.sh which uses HA Cloud TTS and sox.
"""

import argparse
import csv
import subprocess
from pathlib import Path
from typing import Set

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
TESTS_DIR = ROOT_DIR / "tests"
TEST_SENTENCES_DIR = TESTS_DIR / "test_sentences"
TEST_WAV_DIR = TESTS_DIR / "wav"

TTS_LOCALES = {"en": "en-US"}


def main() -> None:
    """Synchronize test WAV files."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hass-token", required=True, help="Long-lived access token for Home Assistant"
    )
    parser.add_argument(
        "--hass-url",
        default="http://localhost:8123",
        help="URL of Home Assistant server",
    )
    parser.add_argument("--delete", action="store_true", help="Delete extra files")
    args = parser.parse_args()

    for csv_path in sorted(TEST_SENTENCES_DIR.glob("*.csv")):
        lang_code = csv_path.stem
        locale = TTS_LOCALES.get(lang_code, f"{lang_code}-{lang_code.upper()}")
        wav_dir = TEST_WAV_DIR / lang_code

        # Exclude OOV (out-of-vocabulary) files
        actual_sentences = {
            wav_path.stem
            for wav_path in wav_dir.glob("*.wav")
            if not wav_path.stem.startswith("oov_")
        }
        expected_sentences: Set[str] = set()

        with open(csv_path, "r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                sentence = row["sentence"]
                expected_sentences.add(sentence)  # exact casing/spacing

        if actual_sentences == expected_sentences:
            print(lang_code, "OK")
            continue

        extra_sentences = actual_sentences - expected_sentences
        for sentence in extra_sentences:
            if args.delete:
                (wav_dir / f"{sentence}.wav").unlink()
                print(lang_code, "deleted:", sentence)
            else:
                print(lang_code, "extra:", sentence)

        missing_sentences = expected_sentences - actual_sentences
        for sentence in missing_sentences:
            print(lang_code, "missing:", sentence)
            subprocess.check_call(
                [
                    SCRIPT_DIR / "generate_test_wav.sh",
                    str(wav_dir),
                    locale,
                    sentence,
                    args.hass_token,
                    args.hass_url,
                ]
            )


if __name__ == "__main__":
    main()
