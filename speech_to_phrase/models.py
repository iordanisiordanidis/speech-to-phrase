"""Speech-to-phrase models."""

import logging
import shutil
import tarfile
import tempfile
from collections.abc import Collection
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Set

import aiohttp

from .const import Language, Settings, WordCasing
from .util import get_language_family

_LOGGER = logging.getLogger(__name__)

URL_FORMAT = "https://huggingface.co/datasets/rhasspy/rhasspy-speech/resolve/main/models/{model_id}.tar.gz?download=true"


class ModelType(str, Enum):
    """Type of model."""

    KALDI = "kaldi"
    COQUI_STT = "coqui-stt"


@dataclass
class Model:
    """Speech-to-phrase model."""

    id: str
    type: ModelType
    language: str
    language_family: str
    description: str
    version: str
    author: str
    url: str
    casing: WordCasing
    sentences_language: str
    number_language: str
    is_enabled: bool = True

    # Kaldi
    spn_phone: str = "SPN"


MODELS: Dict[str, Model] = {
    Language.ENGLISH.value: Model(
        id="en_US-rhasspy",
        type=ModelType.KALDI,
        language="en_US",
        language_family="en",
        description="U.S. English Kaldi model",
        version="1.0",
        author="Rhasspy",
        url="https://github.com/rhasspy/rhasspy",
        casing=WordCasing.LOWER,
        sentences_language="en",
        number_language="en",
    ),
    Language.FRENCH.value: Model(
        id="fr_FR-rhasspy",
        type=ModelType.KALDI,
        language="fr_FR",
        language_family="fr",
        description="French Kaldi model",
        version="1.0",
        author="Rhasspy",
        url="https://github.com/rhasspy/rhasspy",
        casing=WordCasing.LOWER,
        sentences_language="fr",
        number_language="fr",
    ),
    Language.GERMAN.value: Model(
        id="de_DE-zamia",
        type=ModelType.KALDI,
        language="de_DE",
        language_family="de",
        description="German Kaldi model",
        version="1.0",
        author="gooofy",
        url="https://github.com/gooofy/zamia-speech/",
        casing=WordCasing.LOWER,
        sentences_language="de",
        number_language="de",
    ),
    Language.DUTCH.value: Model(
        id="nl_NL-cgn",
        type=ModelType.KALDI,
        language="nl_NL",
        language_family="nl",
        description="Dutch Kaldi model",
        version="1.0",
        author="CGN",
        url="https://github.com/opensource-spraakherkenning-nl/Kaldi_NL",
        casing=WordCasing.LOWER,
        sentences_language="nl",
        number_language="nl",
        spn_phone="[SPN]",
    ),
    Language.SPANISH.value: Model(
        id="es_ES-rhasspy",
        type=ModelType.KALDI,
        language="es_ES",
        language_family="es",
        description="Spanish Kaldi model",
        version="1.1",
        author="Rhasspy",
        url="https://github.com/rhasspy/rhasspy",
        casing=WordCasing.LOWER,
        sentences_language="es",
        number_language="es",
    ),
    Language.ITALIAN.value: Model(
        id="it_IT-rhasspy",
        type=ModelType.KALDI,
        language="it_IT",
        language_family="it",
        description="Italian Kaldi model",
        version="1.1",
        author="Rhasspy",
        url="https://github.com/rhasspy/rhasspy",
        casing=WordCasing.LOWER,
        sentences_language="it",
        number_language="it",
    ),
    Language.GREEK.value: Model(
        is_enabled=False,  # WIP
        id="el_GR-coqui",
        type=ModelType.COQUI_STT,
        language="el_GR",
        language_family="el",
        description="Greek Coqui STT model",
        version="0.1",
        author="Francis Tyers",
        url="https://github.com/coqui-ai/STT-models",
        casing=WordCasing.LOWER,
        sentences_language="el",
        number_language="el",
    ),
    # Language.RUSSIAN.value: Model(
    #     id="ru_RU-rhasspy",
    #     type=ModelType.KALDI,
    #     language="ru_RU",
    #     language_family="ru",
    #     description="Russian Kaldi model",
    #     version="1.0",
    #     author="Rhasspy",
    #     url="https://github.com/rhasspy/rhasspy",
    #     casing=WordCasing.LOWER,
    #     sentences_language="ru",
    #     number_language="ru",
    # ),
    # Language.CZECH.value: Model(
    #     id="cs-CZ-rhasspy",
    #     type=ModelType.KALDI,
    #     language="cs_CZ",
    #     language_family="cs",
    #     description="Czech Kaldi model",
    #     version="1.0",
    #     author="Rhasspy",
    #     url="https://github.com/rhasspy/rhasspy",
    #     casing=WordCasing.LOWER,
    #     sentences_language="cs",
    #     number_language="cs",
    # ),
}

DEFAULT_MODEL = MODELS[Language.ENGLISH]


def get_models_for_languages(languages: Collection[str]) -> List[Model]:
    """Get models compatible with languages.

    Prefer exact matches first, like "en_US", then language families like "en".
    Only a single model from a language family will match.
    """
    language_families: Set[str] = {get_language_family(lang) for lang in languages}
    matching_models: List[Model] = []
    used_model_ids: Set[str] = set()
    used_language_families: Set[str] = set()

    # Exact language match
    for model in MODELS.values():
        if not model.is_enabled:
            continue

        if model.language in languages:
            matching_models.append(model)
            used_model_ids.add(model.id)
            used_language_families.add(model.language_family)

    # Language family match
    for model in MODELS.values():
        if not model.is_enabled:
            continue

        if (model.id in used_model_ids) or (
            model.language_family in used_language_families
        ):
            # Already used
            continue

        if model.language_family in language_families:
            matching_models.append(model)
            used_model_ids.add(model.id)
            used_language_families.add(model.language_family)

    return matching_models


async def download_model(model: Model, settings: Settings) -> None:
    """Download a model.

    If the model is already downloaded, it is deleted and downloaded again.
    """
    model_dir = settings.models_dir / model.id
    if model_dir.exists():
        _LOGGER.debug("Deleting existing model directory: %s", model_dir)
        shutil.rmtree(model_dir)

    settings.models_dir.mkdir(parents=True, exist_ok=True)

    url = URL_FORMAT.format(model_id=model.id)
    _LOGGER.debug(
        "Downloading model %s at %s to %s", model.id, url, settings.models_dir
    )

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()

                with tempfile.NamedTemporaryFile(
                    mode="wb+", suffix=".tar.gz"
                ) as temp_file:
                    async for chunk in response.content.iter_chunked(2048):
                        temp_file.write(chunk)

                    temp_file.seek(0)
                    with tarfile.open(temp_file.name, mode="r:gz") as tar:
                        tar.extractall(path=settings.models_dir)

        _LOGGER.debug("Downloaded model %s", model.id)
    except Exception:
        _LOGGER.exception("Unexpected error while downloading model %s", model.id)

        # Delete directory is it can be re-downloaded
        if model_dir.exists():
            shutil.rmtree(model_dir)

        raise
