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
from speech_to_phrase.hass_api import Area, Entity, Floor

from . import SETTINGS, TESTS_DIR

LANGUAGE = Language.GREEK.value
MODEL = MODELS[LANGUAGE]

WAV_DIR = TESTS_DIR / "wav" / LANGUAGE

THINGS = Things(
    entities=[
        Entity(names=["Νέα Υόρκη"], domain="weather"),
        Entity(names=["EcoBee"], domain="climate"),
        Entity(names=["Standing Light"], domain="light"),
        Entity(names=["Bed Light"], domain="light"),
        Entity(names=["Εξωτερική Υγρασία"], domain="sensor"),
        Entity(names=["Γκαραζόπορτα"], domain="cover"),
        Entity(names=["Μπροστινή Πόρτα"], domain="lock"),
        Entity(names=["Party Time"], domain="script"),
        Entity(names=["Mood Lighting"], domain="scene"),
    ],
    areas=[Area(names=["Γραφείο"]), Area(names=["Κουζίνα"])],
    floors=[Floor(names=["Πρώτος Όροφος"])],
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
    """Train Greek Pocketsphinx model once per session."""
    if SETTINGS.train_dir.exists():
        shutil.rmtree(SETTINGS.train_dir)

    await train(MODEL, SETTINGS, THINGS)

@pytest.mark.parametrize(
    "text",
    [
        "τι ώρα είναι",
        "ποια είναι η ημερομηνία",
        "πως είναι ο καιρός",
        "πως είναι ο καιρός στη Νέα Υόρκη",
        "ποια είναι η θερμοκρασία",
        "ποια είναι η θερμοκρασία του EcoBee",
        "άναψε το standing light",
        "σβήσε τα φώτα στο γραφείο",
        "άναψε τα φώτα στον πρώτο όροφο",
        "όρισε τα φώτα της κουζίνας σε πράσινο",
        "όρισε τη φωτεινότητα του bed light στο 50 τοις εκατό",
        "ποια είναι η εξωτερική υγρασία",
        "κλείσε την γκαραζόπορτα",
        "είναι η γκαραζόπορτα ανοικτή",
        "κλείδωσε την μπροστινή πόρτα",
        "είναι η μπροστινή πόρτα κλειδωμένη",
        "όρισε ένα χρονόμετρο για πέντε λεπτά",
        "όρισε ένα χρονόμετρο για τριάντα δευτερόλεπτα",
        "όρισε ένα χρονόμετρο για τρεις ώρες και δέκα λεπτά",
        "παύση του χρονόμετρου",
        "συνέχισε το χρονόμετρο",
        "ακύρωσε το χρονόμετρο",
        "ακύρωσε όλα τα χρονόμετρα",
        "πόσος χρόνος απομένει στο χρονόμετρο",
        "παύση",
        "συνέχισε",
        "επόμενο",
        "εκτέλεσε το party time",
        "ενεργοποίησε το mood lighting",
        "άκυρο",
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
