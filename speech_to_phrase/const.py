"""Constants."""

from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from .speech_tools import SpeechTools

# Kaldi
EPS = "<eps>"
SIL = "SIL"
SPN = "SPN"
UNK = "<unk>"

# Audio
RATE = 16000
WIDTH = 2
CHANNELS = 1


class Language(str, Enum):
    """Available languages."""

    ENGLISH = "en"
    FRENCH = "fr"


class Settings:
    """Speech-to-phrase settings."""

    def __init__(
        self,
        models_dir: Union[str, Path],
        train_dir: Union[str, Path],
        tools_dir: Union[str, Path],
        hass_token: str,
        hass_websocket_uri: str,
        retrain_on_connect: bool,
        sentences_dir: Optional[Union[str, Path]] = None,
        default_language: str = Language.ENGLISH.value,
    ) -> None:
        """Initialize settings."""
        self.models_dir = Path(models_dir)
        self.train_dir = Path(train_dir)
        self.tools = SpeechTools.from_tools_dir(tools_dir)
        self.hass_token = hass_token
        self.hass_websocket_uri = hass_websocket_uri
        self.retrain_on_connect = retrain_on_connect

        if not sentences_dir:
            # Builtin sentences
            sentences_dir = Path(__file__).parent / "sentences"

        self.sentences = Path(sentences_dir)
        self.default_language = default_language

    def model_data_dir(self, model_id: str) -> Path:
        """Path to model data."""
        return self.models_dir / model_id

    def model_train_dir(self, model_id: str) -> Path:
        """Path to training artifacts for a model."""
        return self.train_dir / model_id

    def model_training_info_path(self, model_id: str) -> Path:
        """Path to training info file for a model."""
        return self.model_train_dir(model_id) / "training_info.json"


class WordCasing(str, Enum):
    """Casing applied to text when training model."""

    KEEP = "keep"
    LOWER = "lower"
    UPPER = "upper"

    @staticmethod
    def get_function(casing: "WordCasing") -> Callable[[str], str]:
        """Get a Python function to apply casing."""
        if casing == WordCasing.LOWER:
            return str.lower

        if casing == WordCasing.UPPER:
            return str.upper

        return lambda s: s
