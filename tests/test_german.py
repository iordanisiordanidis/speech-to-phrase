"""German tests."""

import shutil

import pytest
import pytest_asyncio
from hassil.intents import Intents
from hassil.recognize import recognize
from home_assistant_intents import get_intents
from pysilero_vad import SileroVoiceActivityDetector

from speech_to_phrase import MODELS, Language, Things, train, transcribe
from speech_to_phrase.audio import wav_audio_stream
from speech_to_phrase.hass_api import Area, Entity, Floor

from . import SETTINGS, TESTS_DIR, load_test_sentences

LANGUAGE = Language.GERMAN.value
TEST_SENTENCES = load_test_sentences(LANGUAGE)
MODEL = MODELS[LANGUAGE]

WAV_DIR = TESTS_DIR / "wav" / LANGUAGE

THINGS = Things(
    entities=[
        Entity(names=["New York"], domain="weather"),
        Entity(names=["EcoBee"], domain="climate"),
        Entity(names=["Lampe"], domain="light"),
        Entity(names=["Außenluftfeuchtigkeit"], domain="sensor"),
        Entity(names=["Garagentor"], domain="cover"),
        Entity(names=["Vordertür"], domain="lock"),
    ],
    areas=[Area(names=["Büro"]), Area(names=["Küchen"])],
    floors=[Floor(names=["erster Stock", "ersten Stock"])],
)

VAD = SileroVoiceActivityDetector()


@pytest.fixture(scope="session")
def german_intents() -> Intents:
    intents_dict = get_intents("de")
    lists_dict = intents_dict.get("lists", {})
    lists_dict.update(THINGS.to_lists_dict())
    intents_dict["lists"] = lists_dict

    return Intents.from_dict(intents_dict)


@pytest_asyncio.fixture(scope="session")
async def train_german() -> None:
    """Train German Kaldi model once per session."""
    if SETTINGS.train_dir.exists():
        shutil.rmtree(SETTINGS.train_dir)

    await train(MODEL, SETTINGS, THINGS)


@pytest.mark.parametrize("text", TEST_SENTENCES)
@pytest.mark.asyncio
async def test_transcribe(
    text: str,
    train_german,  # pylint: disable=redefined-outer-name
    german_intents: Intents,  # pylint: disable=redefined-outer-name
) -> None:
    """Test transcribing expected sentences."""
    assert recognize(
        text, german_intents, intent_context={"area": "Küchen"}
    ), f"Sentence not recognized: {text}"

    # Check transcript
    wav_path = WAV_DIR / f"{text}.wav"
    assert wav_path.exists(), f"Missing {wav_path}"

    transcript = await transcribe(MODEL, SETTINGS, wav_audio_stream(wav_path, VAD))
    assert text == transcript


@pytest.mark.parametrize("wav_num", [1, 2, 3, 4])
@pytest.mark.asyncio
async def test_oov(
    wav_num: int, train_german  # pylint: disable=redefined-outer-name
) -> None:
    """Test transcribing out-of-vocabulary (OOV) sentences."""
    wav_path = WAV_DIR / f"oov_{wav_num}.wav"
    assert wav_path.exists(), f"Missing {wav_path}"

    transcript = await transcribe(MODEL, SETTINGS, wav_audio_stream(wav_path, VAD))
    assert not transcript
