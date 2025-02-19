"""Greek tests."""

import shutil

import pytest
import pytest_asyncio
from hassil.intents import Intents
from hassil.recognize import recognize
from home_assistant_intents import get_intents
from pysilero_vad import SileroVoiceActivityDetector

from speech_to_phrase import MODELS, Language, Things, train, transcribe
from speech_to_phrase.audio import wav_audio_stream
from speech_to_phrase.hass_api import Area, Entity

from . import SETTINGS, TESTS_DIR

LANGUAGE = Language.GREEK.value
MODEL = MODELS[LANGUAGE]

WAV_DIR = TESTS_DIR / "wav" / LANGUAGE

THINGS = Things(
    entities=[
        Entity(names=["Νέα Υόρκη"], domain="weather"),
        Entity(names=["Φωτιστικό Σαλονιο'υ"], domain="light"),
        # Entity(names=["Bed Light"], domain="light"),
        # Entity(names=["Outdoor Humidity"], domain="sensor"),
        Entity(names=["γκαράζ"], domain="cover"),
        # Entity(names=["Front Door"], domain="lock"),
    ],
    areas=[Area(names=["Κουζίνα"]), Area(names=["Σαλόνι"])],
    floors=[],
)

VAD = SileroVoiceActivityDetector()


@pytest.fixture(scope="session")
def greek_intents() -> Intents:
    intents_dict = get_intents("el")
    lists_dict = intents_dict.get("lists", {})
    lists_dict.update(THINGS.to_lists_dict())
    intents_dict["lists"] = lists_dict

    return Intents.from_dict(intents_dict)


@pytest_asyncio.fixture(scope="session")
async def train_greek() -> None:
    """Train Greek Kaldi model once per session."""
    if SETTINGS.train_dir.exists():
        shutil.rmtree(SETTINGS.train_dir)

    await train(MODEL, SETTINGS, THINGS)


@pytest.mark.parametrize(
    "text",
    [
        # What time is it?
        "τι ώρα είναι",
        # What's the date?
        "τι μέρα είναι σήμερα",
        # What's the weather like?
        "τι καιρό κάνει",
        # How is the weather in New York?
        "πώς είναι ο καιρός στο Νέα Υόρκη",
        # What is the temperature?
        "ποια είναι η θερμοκρασία",
        # What is the temperature in the kitchen?
        "ποια είναι η θερμοκρασία στο Κουζίνα",
        # Turn on the lights.
        "άναψε τα φώτα",
        # Turn off the lights.
        "σβήσε τα φώτα",
        # Turn on the living room light.
        "άναψε το Φωτιστικό Σαλονιο'υ",
        # Turn on the lights in the living room.
        "άναψε τα φώτα στο Σαλόνι",
        # Turn off the living room light.
        "σβήσε το Φωτιστικό Σαλονιο'υ",
        # Turn off the lights in the living room.
        "σβήσε τα φώτα στο Σαλόνι",
        # Set the living room light to white.
        "ρύθμισε το Φωτιστικό Σαλονιο'υ σε λευκό",
        # Set the brightness in the living room to 50 percent.
        "ρύθμισε τη φωτεινότητα στο Σαλόνι στο 50 τοις εκατό",
        # Close the garage.
        "κλείσε το γκαράζ",
        # Is the garage open?
        "το γκαράζ είναι ανοιχτό",
        # Set a timer for 5 minutes.
        "ρύθμισε χρονόμετρο για 5 λεπτά",
        # Set a timer for 30 seconds.
        "ρύθμισε χρονόμετρο για 30 δευτερόλεπτα",
        # Set a timer for 3 hours and 10 minutes.
        "ρύθμισε χρονόμετρο για 3 ώρες και 10 λεπτά",
        # Cancel the timer.
        "ακύρωσε το χρονόμετρο",
        # Cancel all timers.
        "ακύρωσε όλα τα χρονόμετρα",
        # Timer status.
        "κατάσταση χρονόμετρου",
        # Resume my timer.
        "σύνεχισε το μου χρονόμετρο",
        # Pause.
        "παύσε",
        # Resume.
        "συνέχισε",
        # Next.
        "επόμενο",
        # Nevermind.
        "άσε το",
    ],
)
@pytest.mark.asyncio
async def test_transcribe(
    text: str,
    train_greek,  # pylint: disable=redefined-outer-name
    greek_intents: Intents,  # pylint: disable=redefined-outer-name
) -> None:
    """Test transcribing expected sentences."""
    assert recognize(
        text, greek_intents, intent_context={"area": "Κουζίνα"}
    ), f"Sentence not recognized: {text}"

    wav_path = WAV_DIR / f"{text}.wav"
    assert wav_path.exists(), f"Missing {wav_path}"

    transcript = await transcribe(MODEL, SETTINGS, wav_audio_stream(wav_path, VAD))
    assert text == transcript


@pytest.mark.parametrize("wav_num", [1, 2, 3, 4])
@pytest.mark.asyncio
async def test_oov(
    wav_num: int, train_greek  # pylint: disable=redefined-outer-name
) -> None:
    """Test transcribing out-of-vocabulary (OOV) sentences."""
    wav_path = WAV_DIR / f"oov_{wav_num}.wav"
    assert wav_path.exists(), f"Missing {wav_path}"

    transcript = await transcribe(MODEL, SETTINGS, wav_audio_stream(wav_path, VAD))
    assert not transcript
