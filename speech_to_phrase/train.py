"""Model training."""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass

from hassil.intents import Intents
from hassil.util import merge_dict
from yaml import SafeDumper, safe_dump, safe_load

from .const import Settings, TrainingError, WordCasing
from .g2p import LexiconDatabase
from .hass_api import Things
from .hassil_fst import Fst, G2PInfo, intents_to_fst
from .models import Model, ModelType, download_model
from .train_coqui_stt import train_coqui_stt
from .train_kaldi import train_kaldi

_LOGGER = logging.getLogger(__name__)


@dataclass
class TrainingInfo:
    """Information used to determine if training is required."""

    model_version: str
    sentences_hash: str
    things_hash: str


async def train(
    model: Model, settings: Settings, things: Things, force_retrain: bool = False
) -> None:
    """Train a speech model.

    If the model does not exist, it will be downloaded.
    If the previous training information is identical, training will be skipped.
    """
    model_dir = settings.model_data_dir(model.id)
    if not model_dir.exists():
        await download_model(model, settings)

    training_info = TrainingInfo(
        model_version=model.version,
        sentences_hash=_get_sentences_hash(model, settings),
        things_hash=things.get_hash(),
    )

    training_info_path = settings.model_training_info_path(model.id)
    if (not force_retrain) and training_info_path.exists():
        with open(training_info_path, "r", encoding="utf-8") as training_info_file:
            last_training_info = TrainingInfo(**json.load(training_info_file))

        if last_training_info == training_info:
            _LOGGER.debug("Skipping training of %s", model.id)
            return

    _LOGGER.info("Started training: %s", model.id)
    train_dir = settings.model_train_dir(model.id).absolute()
    train_dir.mkdir(parents=True, exist_ok=True)

    # Written at the end of training
    training_info_path.unlink(missing_ok=True)

    # Create intents
    intents = _create_intents(model, settings, things)

    if model.type == ModelType.KALDI:
        lexicon = LexiconDatabase(settings.models_dir / model.id / "lexicon.db")
        fst = _create_intents_fst(model, lexicon, intents)
        await train_kaldi(model, settings, lexicon, fst)
    elif model.type == ModelType.COQUI_STT:
        lexicon = LexiconDatabase()
        fst = _create_intents_fst(model, lexicon, intents)
        await train_coqui_stt(model, settings, fst)
    else:
        raise TrainingError(f"Unexpected model type for {model.id}: {model.type}")

    # Write training info
    with open(training_info_path, "w", encoding="utf-8") as training_info_file:
        json.dump(
            asdict(training_info),
            training_info_file,
        )

    _LOGGER.info("Finished training: %s", model.id)


# -----------------------------------------------------------------------------


def _create_intents(model: Model, settings: Settings, things: Things) -> Intents:
    """Create intents from sentences and things from Home Assistant."""
    sentences_path = settings.sentences / f"{model.sentences_language}.yaml"
    with open(sentences_path, "r", encoding="utf-8") as sentences_file:
        sentences_dict = safe_load(sentences_file)

    lists_dict = sentences_dict.get("lists", {})
    lists_dict.update(things.to_lists_dict())

    sentences_dict["lists"] = lists_dict

    # Sentence triggers
    if things.trigger_sentences:
        intents_dict = sentences_dict.get("intents", {})
        intents_dict["TriggerSentences"] = {
            "data": [{"sentences": things.trigger_sentences}]
        }
        sentences_dict["intents"] = intents_dict

    # Custom sentences
    for custom_sentences_dir in settings.custom_sentences_dirs:
        dir_for_language = custom_sentences_dir / model.language
        if not dir_for_language.is_dir():
            # Try language family
            dir_for_language = custom_sentences_dir / model.language_family

            if not dir_for_language.is_dir():
                continue

        for custom_sentences_path in sorted(dir_for_language.glob("*.yaml")):
            _LOGGER.debug("Loading custom sentences from %s", custom_sentences_path)

            with open(
                custom_sentences_path, "r", encoding="utf-8"
            ) as custom_sentences_file:
                merge_dict(sentences_dict, safe_load(custom_sentences_file) or {})

    # Write YAML with training sentences (includes HA lists, triggers, etc.)
    SafeDumper.ignore_aliases = lambda *args: True  # type: ignore[assignment]
    training_sentences_path = settings.training_sentences_path(model.id)
    with open(
        training_sentences_path, "w", encoding="utf-8"
    ) as training_sentences_file:
        safe_dump(sentences_dict, training_sentences_file, sort_keys=False)

    _LOGGER.debug("Wrote debug YAML to %s", training_sentences_path)

    return Intents.from_dict(sentences_dict)


def _create_intents_fst(
    model: Model, lexicon: LexiconDatabase, intents: Intents
) -> Fst:
    """Create a finite state transducer (FST) directly from intents.

    This allows for efficiently generating an n-gram language model using
    opengrm instead of enumerating all possible sentences.
    """
    casing_func = WordCasing.get_function(model.casing)

    fst = intents_to_fst(
        intents,
        number_language=model.number_language,
        g2p_info=G2PInfo(lexicon, casing_func),
    ).remove_spaces()

    # Remove dead branches
    fst.prune()

    return fst


def _get_sentences_hash(
    model: Model, settings: Settings, chunk_size: int = 8192
) -> str:
    """Get a hash of sentences YAML files (builtin and custom)."""
    hasher = hashlib.sha256()

    # Builtin sentences
    sentences_path = settings.sentences / f"{model.sentences_language}.yaml"
    with open(sentences_path, "rb") as sentences_file:
        chunk = sentences_file.read(chunk_size)
        hasher.update(chunk)

    # Custom sentences
    for custom_sentences_dir in settings.custom_sentences_dirs:
        dir_for_language = custom_sentences_dir / model.language
        if not dir_for_language.is_dir():
            # Try language family
            dir_for_language = custom_sentences_dir / model.language_family

            if not dir_for_language.is_dir():
                continue

        for custom_sentences_path in sorted(dir_for_language.glob("*.yaml")):
            with open(custom_sentences_path, "rb") as custom_sentences_file:
                chunk = custom_sentences_file.read(chunk_size)
                hasher.update(chunk)

    return hasher.hexdigest()
